# ByteRush Research Console

This Streamlit dashboard is deliberately read-only. It watches the artifacts
written by the AgentManager and never changes an experiment, protected data
file, evaluator, or checkpoint.

## Run on AutoDL

Install the light presentation dependencies once:

```bash
cd /root/autodl-tmp/ByteRush
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
python -m pip install -r dashboard/requirements.txt
```

Launch the dashboard beside the experiment runner:

```bash
cd /root/autodl-tmp/ByteRush
export BYTERUSH_DATA_ROOT=/root/autodl-tmp/ByteRush
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
python -m streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

The AgentManager writes one atomic `dashboard_snapshot.json` in each experiment
directory after every saved step. That snapshot is the sole dashboard source:

- stage number comes from the AgentManager stage definition, never keywords in
  a hypothesis;
- each node's parent, status, hypothesis, code, terminal output, metrics and
  artifact paths share one node ID;
- the stage cards count only nodes first created by that stage (parents carried
  forward remain visible in the tree but are not counted twice);
- the headline and detail metrics come from the same winning node; and
- the evolution chart is built from the selected run summary in every snapshot,
  not the separately maintained ledger.

The dashboard rescans snapshots every 10 seconds. A new AgentManager result
therefore updates its stage card, search-tree node, detail panel, and trend
point together without export, conversion, or manual data entry.

## Backfill existing runs

For experiment directories created before the callback existed, generate the
same schema from their saved `journal.json` files once:

```bash
python dashboard/rebuild_snapshots.py --experiments experiments
```

Directories without a valid Journal remain intentionally absent from the
dashboard; a truncated JSON log is never guessed or silently merged.

## Local preview

Copy `experiments/` (including generated snapshots) from AutoDL into the local
repository, then run the same command. Alternatively set `BYTERUSH_DATA_ROOT`
to any mounted copy of the remote ByteRush directory.
