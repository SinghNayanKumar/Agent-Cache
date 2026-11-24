"""Executable specification of ownership, NOT a GPU allocator or SGLang patch.

Each tuple is an exact token prefix. This deliberately inefficient representation
makes invariants inspectable in CPU tests. Production uses radix nodes/pages.
"""
from dataclasses import dataclass, field

@dataclass
class Node:
    owners: set = field(default_factory=set)
    locks: int = 0
    copies: int = 0

class Registry:
    def __init__(self):
        self.nodes = {}
        self.branches = {}
        self.closed = set()

    @staticmethod
    def path(tokens):
        t = tuple(tokens)
        return [t[:i] for i in range(1, len(t) + 1)]

    def open(self, branch, tokens, parent=None):
        if branch in self.branches or branch in self.closed:
            raise ValueError('Branch IDs cannot be reused; allocate a new ID')
        t = tuple(tokens)
        if parent is not None:
            p = self.branches[parent]['tokens']
            if p[:len(t)] != t:
                raise ValueError('Fork checkpoint is not a parent prefix')
        self.branches[branch] = {'tokens': t, 'state': 'active'}
        for key in self.path(t):
            self.nodes.setdefault(key, Node()).owners.add(branch)

    def update(self, branch, tokens):
        b = self.branches[branch]
        t, old = tuple(tokens), b['tokens']
        if t[:len(old)] != old:
            raise ValueError('Branch requests must append; use fork for a rewind')
        b['tokens'] = t
        for key in self.path(t):
            self.nodes.setdefault(key, Node()).owners.add(branch)

    def state(self, branch, value):
        if value not in ('active', 'suspended'):
            raise ValueError(value)
        self.branches[branch]['state'] = value

    def lock(self, tokens, delta=1):
        keys = self.path(tokens)
        if delta not in (-1, 1):
            raise ValueError('delta must be +/-1')
        if any(k not in self.nodes or self.nodes[k].locks + delta < 0 for k in keys):
            raise ValueError('Invalid lock operation')
        for k in keys:
            self.nodes[k].locks += delta

    def abandon(self, branch):
        if branch in self.closed:
            return 0  # duplicate is idempotent
        self.branches.pop(branch)  # unknown ID is an error
        self.closed.add(branch)
        for n in self.nodes.values():
            n.owners.discard(branch)
        return self.collect()

    def collect(self):
        freed = 0
        for k in sorted(list(self.nodes), key=len, reverse=True):
            n = self.nodes[k]
            has_child = any(len(x) == len(k)+1 and x[:-1] == k for x in self.nodes)
            if not n.owners and not n.locks and not n.copies and not has_child:
                del self.nodes[k]
                freed += 1
        return freed

    def assert_valid(self):
        for k, n in self.nodes.items():
            assert n.locks >= 0 and n.copies >= 0
            assert len(k) == 1 or k[:-1] in self.nodes
            assert n.owners <= self.branches.keys()
        for b, state in self.branches.items():
            assert all(b in self.nodes[k].owners for k in self.path(state['tokens']))
