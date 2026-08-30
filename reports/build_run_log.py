#!/usr/bin/env python3
"""Build / update a consolidated Markdown run log for KuaiRand agent runs.

Each experiment under experiments/<ts>_kuairand_*/ is scanned for timing, tokens,
model usage, best validation metric, and the model architecture the agent wrote.
Results are merged into a JSON ledger and rendered to reports/RUN_LOG.md.

Usage:
  python reports/build_run_log.py                 # rescan everything, regenerate report
  python reports/build_run_log.py --run <dir>     # add/refresh ONE run, then regenerate
  python reports/build_run_log.py --min-seconds 0 # include even sub-minute smoke runs

The ledger (reports/runs_ledger.json) is the source of truth; re-running is
idempotent, so wiring this after each launch keeps the report always current.
"""
from __future__ import annotations
import argparse, json, glob, re, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repo root
EXP = ROOT / "experiments"
REPORTS = ROOT / "reports"
LEDGER = REPORTS / "runs_ledger.json"
REPORT_MD = REPORTS / "RUN_LOG.md"

P_PRIM = re.compile(r'"primary":\s*([0-9.]+)')
P_GAUC = re.compile(r'"GAUC":\s*([0-9.]+)')
P_NDCG = re.compile(r'"nDCG@5":\s*([0-9.]+)')
P_DIR = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})')


def references() -> dict:
    """Reference primary scores from baseline_scores.json (validation split).

    The real bar to beat is fm_official; `random` is only a sanity check and must
    NOT be used as the baseline. oracle_ceiling is the denominator for headroom.
    """
    bs = ROOT / "kuairand-starter-kit" / "baseline_scores.json"
    out = {"random": None, "item_popularity": None, "fm_official": None, "oracle_ceiling": None}
    if not bs.exists():
        return out
    try:
        sc = json.load(open(bs)).get("scores", {})
        for name in out:
            out[name] = sc.get(name, {}).get("valid", {}).get("primary")
    except Exception:
        pass
    return out


def best_metric(run: Path):
    """Best validation primary (+aligned GAUC/nDCG) across ALL history.json in the run.

    Searches the whole run tree — early runs store history.json under
    <stage>/process_*/working/, not under logs/.
    """
    best = None
    for hf in glob.glob(str(run / "**/history.json"), recursive=True):
        try:
            txt = open(hf, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for m in P_PRIM.finditer(txt):
            v = float(m.group(1))
            if best is None or v > best[0]:
                seg = txt[max(0, m.start() - 220): m.start() + 60]
                g, n = P_GAUC.search(seg), P_NDCG.search(seg)
                best = (v, float(g.group(1)) if g else None, float(n.group(1)) if n else None)
    return best


def detect_architecture(run: Path):
    """Classify what the agent's winning code actually does, from best_solution_*.py."""
    sols = sorted(glob.glob(str(run / "logs/**/best_solution_*.py"), recursive=True))
    if not sols:
        return None, None
    f = sols[-1]                                          # highest stage
    stage = None
    ms = re.findall(r"stage_[0-9][^/]*", f)
    if ms:
        stage = ms[-1]
    try:
        code = open(f, encoding="utf-8", errors="ignore").read().lower()
    except Exception:
        return None, stage
    has_hist = "history" in code and ("_build_history" in code or "history_features" in code)
    if "targetattention" in code or "din" in code and "attention" in code:
        base = "DIN target-attention"
    elif "bpr" in code or "pairwise" in code:
        base = "DeepFM + BPR pairwise loss"
    elif "class deepfm" in code and ("widedeep" in code or "wide_deep" in code or "wide & deep" in code):
        base = "DeepFM + Wide&Deep"
    elif "class deepfm" in code or "deepfm" in code:
        base = "DeepFM"
    elif "widedeep" in code or "wide & deep" in code:
        base = "Wide&Deep"
    elif "mlp interaction" in code or "mlp_interaction" in code:
        base = "Embedding + MLP interaction"
    elif "factorization" in code:
        base = "Factorization Machine"
    else:
        base = "candidate model"
    if has_hist and "history" not in base.lower():
        base += " + history"
    return base, stage


def parse_dirtime(name: str):
    m = P_DIR.match(name)
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S") if m else None


def extract_run(run: Path) -> dict:
    name = run.name
    dt = parse_dirtime(name)
    d = {
        "run": name,
        "start": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None,
        "sort_key": dt.isoformat() if dt else name,
        "duration_s": None, "training_s": None,
        "llm_total_tokens": None, "llm_calls": None, "models": {},
        "executed_iters": None, "seed_evals": None, "gpu_peak_mib": None,
        "best_primary": None, "best_gauc": None, "best_ndcg": None,
        "architecture": None, "stage_reached": None,
        "run_started_utc": None, "run_finished_utc": None,
    }
    rs = run / "resource_summary.json"
    if rs.exists():
        try:
            s = json.load(open(rs))
            d["duration_s"] = round(s.get("wall_clock_seconds", 0), 1)
            d["training_s"] = round(s.get("gpu_active_seconds", 0), 1)
            d["llm_total_tokens"] = s.get("llm_total_tokens")
            d["llm_calls"] = s.get("llm_calls")
            d["models"] = {k: {"tokens": v.get("total_tokens"), "calls": v.get("calls")}
                           for k, v in s.get("llm_by_model", {}).items()}
            d["executed_iters"] = s.get("executed_iterations")
            d["seed_evals"] = s.get("seed_evaluations")
            d["gpu_peak_mib"] = s.get("gpu_peak_used_memory_mib")
            d["run_started_utc"] = s.get("run_started_at_utc")
            d["run_finished_utc"] = s.get("run_finished_at_utc")
        except Exception as e:
            d["error"] = str(e)
    bm = best_metric(run)
    if bm:
        d["best_primary"], d["best_gauc"], d["best_ndcg"] = bm
    d["architecture"], d["stage_reached"] = detect_architecture(run)
    return d


def load_ledger() -> dict:
    if LEDGER.exists():
        try:
            return json.load(open(LEDGER))
        except Exception:
            pass
    return {"baseline_primary": None, "runs": {}}


def fmt_secs(s):
    if s is None:
        return "—"
    s = float(s)
    if s < 90:
        return f"{s:.0f}s"
    return f"{s/60:.0f}m"


def fmt_tokens(t):
    if t is None:
        return "—"
    if t >= 1_000_000:
        return f"{t/1_000_000:.2f}M"
    if t >= 1_000:
        return f"{t/1_000:.0f}k"
    return str(t)


def fmt_delta(x):
    if x is None:
        return "—"
    if abs(x) < 5e-5:
        return "±0"
    return f"{'+' if x >= 0 else '−'}{abs(x):.4f}"


def render_markdown(ledger: dict, min_seconds: float) -> str:
    baseline = ledger.get("baseline_primary")          # fm_official
    refs = ledger.get("references", {})
    oracle = refs.get("oracle_ceiling")
    runs = sorted(ledger["runs"].values(), key=lambda r: r["sort_key"])

    # deltas vs previous scored run
    prev = None
    for r in runs:
        bp = r.get("best_primary")
        r["_vs_base"] = round(bp - baseline, 4) if (bp is not None and baseline is not None) else None
        r["_vs_prev"] = round(bp - prev, 4) if (bp is not None and prev is not None) else None
        if bp is not None:
            prev = bp

    scored = [r for r in runs if r.get("best_primary") is not None]
    # show any run that produced a metric OR has resource tracking; count the rest
    shown = [r for r in runs if (r.get("best_primary") is not None) or (r.get("duration_s") is not None)]
    dropped = len(runs) - len(shown)

    tracked = [r for r in shown if r.get("duration_s")]
    tot_tokens = sum(r.get("llm_total_tokens") or 0 for r in tracked)
    tot_calls = sum(r.get("llm_calls") or 0 for r in tracked)
    tot_wall = sum(r.get("duration_s") or 0 for r in tracked)
    tot_train = sum(r.get("training_s") or 0 for r in tracked)
    best = max((r["best_primary"] for r in scored), default=None)
    best_run = max(scored, key=lambda r: r["best_primary"], default=None) if scored else None

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = []
    L.append("# KuaiRand Agent Run Log")
    L.append("")
    L.append(f"*ByteRush · TechJam — KuaiRand-Pure long_view ranking · primary = mean(GAUC, nDCG@5)*  ")
    L.append(f"*Auto-generated {now} · {len(shown)} runs shown / {len(runs)} launched ({dropped} sub-minute aborts hidden)*")
    L.append("")
    L.append("## Reference scores (validation)")
    L.append("")
    L.append("The bar to beat is **`fm_official`** — `random` is only a sanity check. "
             "`oracle_ceiling` is the theoretical max (nDCG capped by all-negative users) and the denominator for headroom.")
    L.append("")
    L.append("| Reference | Primary | Note |")
    L.append("|---|---|---|")
    L.append(f"| random | `{refs.get('random')}` | sanity check only — **not** the baseline |")
    L.append(f"| item_popularity | `{refs.get('item_popularity')}` | official non-trained baseline |")
    L.append(f"| **fm_official** | **`{refs.get('fm_official')}`** | **official FM baseline — the real bar** |")
    L.append(f"| oracle_ceiling | `{refs.get('oracle_ceiling')}` | theoretical upper bound |")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| FM baseline (fm_official) | `{baseline:.4f}` |" if baseline is not None else "| FM baseline | n/a |")
    if best is not None and baseline is not None:
        hr = ""
        if oracle and oracle > baseline:
            hr = f" · {100*(best-baseline)/(oracle-baseline):.1f}% of oracle headroom"
        when = f" ({best_run['start'][:16]})" if best_run and best_run.get("start") else ""
        L.append(f"| **Best validation primary** | **`{best:.4f}`** ({fmt_delta(best-baseline)} vs FM baseline{hr}){when} |")
    L.append(f"| Scored runs | {len(scored)} of {len(runs)} launched |")
    L.append(f"| Total LLM tokens | {fmt_tokens(tot_tokens)} across ~{tot_calls:,} calls |")
    L.append(f"| Agent wall-clock | {tot_wall/3600:.1f} h |")
    L.append(f"| GPU-active training | {tot_train/60:.0f} min |")
    L.append("")
    L.append(f"> **Reality check:** measured against the real FM baseline (`{baseline:.4f}`), the agent's best is "
             f"only **{fmt_delta(best-baseline)}** — essentially matching the provided FM. Early runs reproduced FM at "
             f"~0.6014; later architecture changes (DeepFM, BPR, DIN attention) added <0.004. The '+0.12' figure from an "
             f"earlier draft compared against `random` and was misleading.")
    L.append("")

    L.append("## Per-run ledger")
    L.append("")
    L.append("Chronological. `Primary` = best validation node in that run. `vs prev` compares to the previous scored run. "
             "Shipped submissions use a more conservative 3-seed mean.")
    L.append("")
    hdr = ["Start", "Architecture / change", "Dur", "Train", "Tokens", "Calls", "Iters", "Seeds", "Primary", "vs base", "vs prev"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in shown:
        start = (r.get("start") or r["run"])[:16]
        arch = r.get("architecture") or "—"
        if (r.get("best_primary") is None) and (r.get("executed_iters") or 0) <= 3 and (r.get("duration_s") or 0) < 60:
            arch += " · aborted"
        prim = f"{r['best_primary']:.4f}" if r.get("best_primary") is not None else "—"
        row = [
            start, arch, fmt_secs(r.get("duration_s")), fmt_secs(r.get("training_s")),
            fmt_tokens(r.get("llm_total_tokens")),
            str(r.get("llm_calls") or "—"),
            str(r.get("executed_iters") if r.get("executed_iters") is not None else "—"),
            str(r.get("seed_evals") if r.get("seed_evals") is not None else "—"),
            f"`{prim}`", fmt_delta(r.get("_vs_base")), fmt_delta(r.get("_vs_prev")),
        ]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # model usage
    agg = {}
    for r in tracked:
        for m, v in (r.get("models") or {}).items():
            a = agg.setdefault(m, {"tokens": 0, "calls": 0})
            a["tokens"] += v.get("tokens") or 0
            a["calls"] += v.get("calls") or 0
    if agg:
        L.append("## Model usage")
        L.append("")
        L.append("| Model | Role | Tokens | Calls |")
        L.append("|---|---|---|---|")
        roles = {"deepseek-v4-pro": "plan + code", "deepseek-v4-flash": "feedback + select + summary",
                 "deepseek-v4-flash-vision-exp": "plot / figure reading"}
        for m, v in sorted(agg.items(), key=lambda kv: -kv[1]["tokens"]):
            L.append(f"| `{m}` | {roles.get(m,'—')} | {fmt_tokens(v['tokens'])} | {v['calls']:,} |")
        L.append("")

    L.append("## Workflow (per run)")
    L.append("")
    L.append("AI-Scientist-v2 tree search, `bfts_config_kuairand.yaml` (max_stage 4, 1 worker):")
    L.append("")
    L.append("1. **Stage 1 — Initial implementation:** run trusted starting code, confirm a working candidate on GPU.")
    L.append("2. **Stage 2 — Baseline tuning:** learning-rate / regularization sweeps.")
    L.append("3. **Stage 3 — Creative research:** architecture changes (MLP, DeepFM, Wide&Deep, ranking losses, attention).")
    L.append("4. **Stage 4 — Ablation + freeze:** leave-one-component-out ablation, 3-seed re-validation, freeze checkpoint → `submission.csv`.")
    L.append("")
    L.append("---")
    L.append("*Sources: per-run `resource_summary.json` (timing/tokens/GPU), `history.json` (metrics), "
             "`best_solution_*.py` (architecture). Metrics are validation-only; test labels are held out. "
             "Regenerate: `python reports/build_run_log.py`.*")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="add/refresh a single experiment dir (name or path)")
    ap.add_argument("--min-seconds", type=float, default=60.0,
                    help="minimum wall-clock to include an unscored run (default 60)")
    args = ap.parse_args()

    REPORTS.mkdir(exist_ok=True)
    ledger = load_ledger()
    refs = references()
    ledger["references"] = refs
    ledger["baseline_primary"] = refs.get("fm_official")   # the real bar, not `random`

    if args.run:
        run = Path(args.run)
        if not run.is_absolute():
            run = EXP / run.name
        if not run.is_dir():
            raise SystemExit(f"run dir not found: {run}")
        ledger["runs"][run.name] = extract_run(run)
        print(f"[build_run_log] refreshed 1 run: {run.name}")
    else:
        n = 0
        for run in sorted(EXP.glob("*")):
            if run.is_dir() and "kuairand" in run.name:
                ledger["runs"][run.name] = extract_run(run)
                n += 1
        print(f"[build_run_log] scanned {n} runs")

    json.dump(ledger, open(LEDGER, "w"), indent=2)
    REPORT_MD.write_text(render_markdown(ledger, args.min_seconds), encoding="utf-8")
    print(f"[build_run_log] wrote {REPORT_MD.relative_to(ROOT)} and {LEDGER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
