# Normalized trace schema (version 1)

One JSON document contains metadata and an ordered list of tasks. Every request contains its full frozen input-token sequence. Events execute in listed order; `depends_on` validates causal references but does not enable concurrent replay.

```json
{
  "schema": 1,
  "meta": {
    "kind": "real",
    "split": "heldout",
    "tokenizer": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "tokenizer_revision": "FULL_MODEL_SHA",
    "toy": false,
    "source": "source URL or local artifact identifier",
    "source_sha256": "HASH_OF_ORIGINAL_LOG",
    "agent_commit": "AGENT_COMMIT",
    "chat_template_sha256": "HASH_OF_TEMPLATE"
  },
  "traces": [{
    "task_id": "repository-issue-id",
    "events": [
      {"id": 0, "depends_on": [], "op": "open", "branch": "root", "input_ids": [1,2]},
      {"id": 1, "depends_on": [0], "op": "request", "branch": "root", "input_ids": [1,2], "max_new_tokens": 128},
      {"id": 2, "depends_on": [1], "op": "suspend", "branch": "root"},
      {"id": 3, "depends_on": [2], "op": "fork", "branch": "a", "parent": "root", "input_ids": [1,2]},
      {"id": 4, "depends_on": [3], "op": "request", "branch": "a", "input_ids": [1,2,3,4], "max_new_tokens": 128},
      {"id": 5, "depends_on": [4], "op": "abandon", "branch": "a"},
      {"id": 6, "depends_on": [5], "op": "resume", "branch": "root"},
      {"id": 7, "depends_on": [6], "op": "request", "branch": "root", "input_ids": [1,2,5], "max_new_tokens": 128},
      {"id": 8, "depends_on": [7], "op": "abandon", "branch": "root"}
    ]
  }]
}
```

The IDs [1,2,...] illustrate shape only; replace them with real tokenizer outputs. Requests on one branch must extend its last submitted prompt. A rewind or edit needs a new branch at the verified common prefix. Parent references are retained while descendants live; closing the parent does not close children. Close every live handle at task end. Never reuse a closed branch ID.

For standard append-only agents: `open(root)`, all `request(root)` events with complete history, `abandon(root)`. A failed tool result does not create a fork. Tokenizer chat-template endings can make consecutive serialized prompts non-prefix even when messages append; inspect actual IDs. Record genuine checkpoint/edit events when the controller knows them, or use a clearly documented preprocessing rule to identify serialization rewrites without labelling them agent search branches. Do not silently drop tokens to create a prefix hit.

For midterm fixed-length replay, use max_new_tokens=128 throughout. To use recorded output lengths, set per-request values once in the manifest and use the same file for every policy; the runner ignores EOS and enforces exactly that count. An output length exceeding the configured context is infeasible, not a failed cache policy.

Store original log hashes, task split manifest, tokenizer revision, chat-template hash, agent revision and event provenance. The validator checks structural consistency, not the truth of source labels. Manually audit at least two converted trajectories against original logs.

The bundled generator creates a fixed fan-out/revisit topology. It does not implement arbitrary search depth, randomized revisit probability, real SWE-bench acquisition or log-specific parsing; those are extensions after the midterm core works.
