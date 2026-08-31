# ByteRush Competition Showcase

The Dashboard is now a frozen, seven-page 16:9 competition showcase rather than
a live run selector. It joins the real server-side `experiments/` and
`artifacts/` through the final model's `source_node_id`, rejects interrupted
runs, and presents one auditable champion story. Use the sticky top navigation,
normal page scrolling, or the browser's Page Up/Page Down controls to move
between presentation pages.

## Data contract

The showcase builder accepts a ByteRush data root containing:

```text
experiments/                 canonical AgentManager snapshots and search nodes
artifacts/                   verified final model, multi-seed metrics, submission
```

It never reads raw train/test data and never computes a test metric. A candidate
is accepted only when:

- the final manifest contains validation GAUC, nDCG@5, and Primary;
- successful seed count meets the requested seed count;
- `test_metrics_used_for_selection` and `test_metrics_computed` are both false;
- model, checkpoint, submission, and submission metadata all exist;
- the final `source_node_id` belongs to a settled experiment snapshot; and
- the source node can be traced back to the scored FM root.

## Build from the real AutoDL outputs

The showcase code may live in a separate Git worktree. Point it at the original
ByteRush directory instead of copying large artifacts into Git:

```bash
cd /root/autodl-tmp/ByteRush-dashboard-showcase
python dashboard/build_showcase.py \
  --data-root /root/autodl-tmp/ByteRush
```

The command writes `dashboard/generated/showcase_manifest.json`. The generated
file is an output and should normally remain untracked.

## Launch

```bash
cd /root/autodl-tmp/ByteRush-dashboard-showcase
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.fileWatcherType none
```

Use `dashboard/showcase_config.yaml` for presentation copy and to pin the final
artifact directory. Do not enter metric values in that file; scores always come
from the verified artifact manifest.

The Search page offers three animated levels of detail: `Champion path`,
`Curated evidence`, and `Full search`. Hover a node to read its exact Primary,
GAUC, nDCG@5, status, and node ID.

## Optional live technical appendix

The original near-real-time run explorer remains available separately:

```bash
export BYTERUSH_DATA_ROOT=/root/autodl-tmp/ByteRush
python -m streamlit run dashboard/research_console.py \
  --server.address 0.0.0.0 \
  --server.port 8502 \
  --server.fileWatcherType none
```

The competition Showcase is frozen and curated; the Research Console is the
place to inspect every live or historic run.

## Checks

```bash
python -m unittest dashboard.test_showcase_loader
python -m py_compile dashboard/app.py dashboard/build_showcase.py dashboard/showcase_loader.py
```
