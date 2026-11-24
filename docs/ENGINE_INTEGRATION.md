# A0 physical cache integration: remaining implementation work

The harness is operational client code. This document specifies the missing engine patch. None of these project endpoints exists in unmodified SGLang. Do not create a web handler that reports success without invoking real scheduler/cache/allocator operations.

## 1. Locate the pinned engine's relevant paths

In your SGLang checkout:

```bash
rg -n 'class .*Radix|class .*Component|cache_finished_req|match_prefix|evict_device' python/sglang/srt/mem_cache
rg -n 'close_session|session_id|flush_cache' python/sglang/srt
rg -n 'lock_ref|inc_lock_ref|dec_lock_ref|free\(' python/sglang/srt/mem_cache
```

Modern upstream has a unified component cache; older versions have different trees. Follow that revision's real eviction and node-split paths. Do not assume a method named in a tutorial still exists. Restrict the initial model to ordinary full attention and tensor parallel degree 1.

## 2. Metadata and event transport

Each live logical branch needs: session ID, branch ID, immutable checkpoint identity, last accepted event ID, lifecycle state, and reverse links to owned radix nodes/pages. Exact prefix identity includes model/adapter/position/cache format. Do not use a hash without collision-safe token verification.

Request handlers enqueue an event to the scheduler and await an acknowledgment. Mutation must occur at a scheduler-safe point, not concurrently in the web worker. Use the engine's existing coordination primitives. Event IDs are monotonic within a session; duplicates with identical content are idempotent, conflicting duplicates or old unseen IDs are rejected.

The proposed endpoint contract:

### GET /agentcache/capabilities

```json
{
  "schema": 1,
  "physical_reclamation": true,
  "exact_context": true,
  "patch_commit": "ACTUAL_IMPLEMENTATION_COMMIT"
}
```

Only expose the `true` values when physical reclamation and exact-context paths are actually enabled. Treat this handshake as a configuration check, not scientific proof of correctness.

### POST /agentcache/event

The body is one manifest event plus a unique `session_id`. For example:

```json
{
  "session_id": "run/task",
  "id": 3,
  "depends_on": [2],
  "op": "fork",
  "branch": "child1",
  "parent": "root",
  "input_ids": [100, 101, 102]
}
```

The `open` event registers a logical checkpoint even when its KV has not been computed. `fork` validates an exact prefix of the parent's checkpoint. `suspend` changes residency preference only. `resume` changes lifecycle state; a subsequent request supplies its full prompt. `abandon` releases only that handle, never implicitly all its children. Duplicate closes remain safe.

Return an acknowledged event ID and truthful counters:

```json
{
  "success": true,
  "ack_event_id": 3,
  "logical_closed": false,
  "reclaimed_tokens": 0,
  "reclaimed_bytes": 0,
  "deferred_locked_tokens": 0
}
```

These numbers illustrate the schema, not expected results. For asynchronous work, report a logical acknowledgment separately from physical reclamation completion.

### POST /generate

The harness adds:

```json
{"agentcache":{"session_id":"run/task","branch_id":"child1"}}
```

Extend the request schema, tokenizer-manager transport, scheduler message and request object. Reject unknown handles; never silently discard this field. An untagged warm-up request must still execute ordinary inference safely. The full `input_ids` remain mandatory. Cache hits follow exact-prefix matching only.

When a tagged request is admitted, acquire normal engine locks before any pruning can reuse its blocks. After completion, associate the committed cache prefix with its branch. Include generated-token KV in ownership/accounting when it was inserted, even though the replay manifest's next prompt is independent. A subsequent manifest request need only append to the previous **submitted** prefix, not to newly generated text. Separate submitted checkpoint identity from the engine's generated continuation cache to support that replay behavior.

### POST /agentcache/reset

Only accept when idle. Close the branch registry and release ownership without violating request locks. The harness calls normal `/flush_cache` afterwards. Reset must clear tombstones/event state as well as live branches so task runs are independent.

## 3. Reclamation algorithm

Use `Registry` only as a small logical model. Production references belong to physical radix entries, not Python tuples containing every prefix.

On abandon:

1. Close that handle and remove only its ownership references.
2. Traverse affected nodes in postorder using a branch-to-node reverse index.
3. For each eligible node, require no surviving branch owner, no request lock, no pending copy/use dependency, and no retained child requiring the prefix.
4. Invoke the engine's normal eviction/removal primitive so its tree, allocator, page tables and accounting remain consistent.
5. Defer locked nodes; retry after engine completion releases the locks.
6. Stop at shared ancestors. Partially owned pages cannot be freed; honor page alignment and copy-on-write rules.

If a cache entry is ordinarily evicted under capacity pressure, update reverse mappings so a later abandon cannot free the same allocation again. Use allocation generation IDs or equivalent ownership checks to avoid stale references to recycled indices. Node splits redistribute ownership and locks correctly.

The minimal A0 policy eagerly reclaims safely abandoned state, then uses ordinary LRU for capacity pressure. Live branch ownership is a soft cache preference; do not permanently pin every suspended branch and cause an artificial OOM. All attention-required state must be present or recomputed before execution.

## 4. Instrumentation

Emit a per-request debug record keyed by request ID and branch:

- native prefix hit tokens and physically processed prefill tokens, including boundary recomputation;
- actual allocator used/available token slots and byte counts;
- candidate, reclaimed and deferred page counts;
- event acknowledgment and physical-free timestamps;
- request lock count and allocation-generation identity in debug builds.

Do not infer reclaimed HBM bytes from the number of deleted Python objects. Preallocated CUDA pool size may remain constant after freeing usable blocks. Process-level `nvidia-smi` memory is a separate metric.

For numerical testing, expose full next-token logits on a small fixed test set in a debug build, save float arrays, and compare against NC with the same kernels/dtype. Establish tolerance from baseline repeatability first. The client `compare` command checks tokens/text only and must not be described as a logit test.

## 5. Required GPU acceptance cases

- A/B share root P; abandon A; B's outputs and prefix remain valid.
- Parent handle closes while child lives; child's required ancestors survive.
- Duplicate close, stale event and unknown session: no double-free.
- Abandon during a running request: allocation survives until request completes.
- Ordinary LRU evicts owned state, then abandon runs: no stale allocator reference.
- Radix edge splits after fork: ownership moves correctly.
- Generated continuation differs from the next recorded prompt: no invalid reuse.
- Same suffix after different prefixes: no suffix-only sharing.
- Close/reopen using new handle and same full prompt: restore/recompute correctly.
- Many allocate/free cycles: used/free counters reconcile, no leaks.
- Missing/delayed close: lower benefit, never different prompt semantics.

A0 should pass these before a speed comparison. Any failure blocks A0 claims. Host-transfer races, copy completion and offloading are deferred to A1 and are not validated by this midterm GPU-only harness.

## 6. Attribution control

Compare A0 with B1_branch, which uses upstream per-branch session IDs and closes them on abandonment. If the built-in policy performs similarly, report that outcome. A new endpoint name or identical LRU behavior is not a systems contribution.
