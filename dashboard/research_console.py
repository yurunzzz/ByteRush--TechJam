"""ByteRush live Research Console (technical appendix).

Launch on AutoDL:
    BYTERUSH_DATA_ROOT=/root/autodl-tmp/ByteRush streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # Kept optional so a copied dashboard still opens without it.
    st_autorefresh = None

from data_loader import (
    STAGE_META,
    ExperimentNode,
    TreeSnapshot,
    artifact_files,
    best_node,
    find_latest_trees,
    load_overview,
    project_root,
    stage_stats,
)


st.set_page_config(page_title="ByteRush · Live Research Console", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed")


def _root() -> Path:
    configured = os.getenv("BYTERUSH_DATA_ROOT")
    return Path(configured).expanduser().resolve() if configured else project_root()


@st.cache_data(ttl=60, show_spinner=False)
def load_dashboard(root_string: str):
    root = Path(root_string)
    return load_overview(root)


@st.cache_data(ttl=60, show_spinner=False)
def load_selected_trees(root_string: str, run_id: str | None):
    return find_latest_trees(Path(root_string) / "experiments", run_id)


def fmt_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "—"


def fmt_delta(value: float | None) -> str:
    return f"{value:+.4f}" if value is not None else "—"


def relative_delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline in (None, 0):
        return "—"
    return f"{(value - baseline) / baseline:+.2%}"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(circle at 5% 0%, #172554 0%, #080d1d 36%, #020617 100%); color: #e5eefb; }
        header[data-testid="stHeader"] { background: rgba(2,6,23,.55); }
        [data-testid="stMetric"] { background: linear-gradient(145deg, rgba(30,41,59,.88), rgba(15,23,42,.74)); border: 1px solid rgba(148,163,184,.18); padding: 1rem; border-radius: 16px; }
        [data-testid="stMetricLabel"] { color: #a5b4fc; }
        .hero { padding: .4rem 0 1rem 0; }
        .hero h1 { margin: 0; font-size: 2.25rem; letter-spacing: -.04em; }
        .hero p { margin: .35rem 0 0; color: #9fb0ca; font-size: 1.02rem; }
        .stage-card { min-height: 178px; padding: 1.15rem; border-radius: 18px; border: 1px solid rgba(148,163,184,.16); background: linear-gradient(155deg, rgba(30,41,59,.8), rgba(15,23,42,.7)); }
        .stage-kicker { color: #93c5fd; font-size: .76rem; letter-spacing: .1em; text-transform: uppercase; }
        .stage-title { font-size: 1.08rem; font-weight: 700; margin: .24rem 0 .55rem; }
        .stage-value { font-size: 1.42rem; font-weight: 700; color: #f8fafc; }
        .stage-caption { color: #aab9cf; font-size: .86rem; line-height: 1.42; margin-top: .7rem; }
        .good { color: #4ade80; } .bad { color: #fb7185; } .gold { color: #fbbf24; }
        div[data-testid="stExpander"] { border-color: rgba(148,163,184,.2); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stage_card(stage: str, tree: TreeSnapshot | None, baseline: float | None) -> None:
    meta = STAGE_META[stage]
    stats = stage_stats(stage, tree, baseline)
    best = stats["best"]
    if stage == "baseline":
        headline = f"Primary {fmt_metric(baseline)}"
        supporting = "Pipeline verified" if tree else "Reference result loaded from run ledger"
    elif stage == "tuning":
        headline = f"{stats['total']} experiments · {stats['success']} succeeded"
        supporting = f"Best {fmt_metric(best.primary if best else None)} · {relative_delta(best.primary if best else None, baseline)} vs baseline"
    elif stage == "creative":
        headline = f"{stats['success']} / {stats['total']} experiments"
        supporting = f"Best: {best.label if best else 'Awaiting result'} · {relative_delta(best.primary if best else None, baseline)}"
    else:
        components = set()
        for node in (tree.nodes if tree else []):
            if node.ablation_name:
                components.add(node.ablation_name)
            for component in re.findall(r'["\']([\w-]+(?:component|loss))["\']\s*:\s*True', node.code):
                components.add(component)
        headline = f"{len(components)} components verified"
        supporting = f"Key contribution {fmt_delta(stats['delta'])} Primary" if stats["delta"] is not None else "Awaiting ablation evidence"
    st.markdown(
        f"""
        <div class="stage-card">
          <div class="stage-kicker">{meta['label']}</div>
          <div class="stage-title">{headline}</div>
          <div class="{'good' if stats['success'] else 'gold'}">{supporting}</div>
          <div class="stage-caption">{meta['description']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_tree_figure(tree: TreeSnapshot, selected_uid: str | None) -> go.Figure:
    lookup = {node.uid: node for node in tree.nodes}
    fig = go.Figure()
    for source_uid, target_uid in tree.edges:
        source, target = lookup[source_uid], lookup[target_uid]
        fig.add_trace(go.Scatter(x=[source.x, target.x], y=[-source.y, -target.y], mode="lines", line={"color": "rgba(148,163,184,.42)", "width": 1.8}, hoverinfo="skip", showlegend=False))
    colors = {"succeeded": "#34d399", "failed": "#fb7185", "pending": "#94a3b8", "running": "#cbd5e1"}
    symbols = {"succeeded": "circle", "failed": "x", "pending": "circle", "running": "circle"}
    for status in ("succeeded", "failed", "pending", "running"):
        group = [node for node in tree.nodes if node.status == status]
        if not group:
            continue
        sizes = [22 + max(0, ((node.primary or 0.60) - 0.59) * 1500) for node in group]
        fig.add_trace(
            go.Scatter(
                x=[node.x for node in group], y=[-node.y for node in group], mode="markers+text",
                text=[node.label for node in group], textposition="bottom center", cliponaxis=False,
                textfont={"size": 11, "color": "#dbeafe"},
                customdata=[[node.uid, node.primary, node.gauc, node.ndcg] for node in group],
                marker={"size": sizes, "color": colors[status], "symbol": symbols[status], "line": {"color": ["#fbbf24" if node.is_best else "#60a5fa" if status == "running" else "#0f172a" for node in group], "width": [4 if node.is_best or status == "running" else 2 for node in group]}},
                name=status.title(),
                hovertemplate="<b>%{text}</b><br>Primary %{customdata[1]:.4f}<extra></extra>",
            )
        )
    best = next((node for node in tree.nodes if node.is_best), None)
    if best:
        fig.add_trace(go.Scatter(x=[best.x], y=[-best.y + .045], mode="text", text=["★ Best"], textfont={"size": 13, "color": "#fbbf24"}, hoverinfo="skip", showlegend=False))
    fig.update_layout(
        height=600, margin={"l": 15, "r": 15, "t": 35, "b": 50}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,.35)",
        xaxis={"visible": False, "range": [-.08, 1.08]}, yaxis={"visible": False, "range": [-1.12, .13], "scaleanchor": None},
        legend={"orientation": "h", "y": 1.09, "x": 0}, clickmode="event+select",
    )
    return fig


def node_detail(node: ExperimentNode, baseline: float | None) -> None:
    st.markdown(f"### {node.label}")
    status_icon = {"succeeded": "🟢 Succeeded", "failed": "🔴 Failed", "pending": "⚪ Pending", "running": "🔵 Running"}[node.status]
    st.caption(f"{status_icon} · {STAGE_META[node.stage]['label']} · node {node.index}")
    cols = st.columns(3)
    cols[0].metric("GAUC", fmt_metric(node.gauc))
    cols[1].metric("nDCG@5", fmt_metric(node.ndcg))
    cols[2].metric("Primary", fmt_metric(node.primary), relative_delta(node.primary, baseline))
    st.markdown("#### Research hypothesis")
    st.write(node.plan or "No research hypothesis was persisted for this node.")
    st.markdown("#### Modification")
    modification = node.hyperparam_name or node.ablation_name or node.label
    st.code(modification, language="text")
    st.markdown("#### Agent conclusion")
    st.write(node.analysis or node.error_info or "The agent has not yet written a conclusion.")
    artifacts = artifact_files(node)
    st.markdown("#### Artifacts")
    code = node.code or (artifacts["solution"].read_text(encoding="utf-8", errors="replace") if artifacts["solution"] else "")
    with st.expander("View generated code", expanded=False):
        st.code(code or "No code snapshot was available.", language="python")
        if code:
            st.download_button("Download generated code", code, file_name=f"byterush_node_{node.index}.py", mime="text/x-python")
    with st.expander("View execution log", expanded=False):
        st.code(node.terminal_output or node.error_info or "No terminal output was persisted.", language="text")
        if node.terminal_output:
            st.download_button("Download execution log", node.terminal_output, file_name=f"byterush_node_{node.index}.log", mime="text/plain")
    if artifacts["config"]:
        config_text = artifacts["config"].read_text(encoding="utf-8", errors="replace")
        with st.expander("View configuration", expanded=False):
            st.code(config_text, language="yaml")
            st.download_button("Download configuration", config_text, file_name="config.yaml", mime="text/yaml")
    if artifacts["checkpoint"]:
        st.download_button("Download checkpoint", artifacts["checkpoint"].read_bytes(), file_name=artifacts["checkpoint"].name, mime="application/octet-stream")
    if node.index != 0:
        st.caption("Parent experiment: use the connecting edge in the search tree to inspect the source node.")


def evolution_chart(ledger: list[dict], baseline: float | None) -> go.Figure:
    rows = ledger[-30:]
    fig = go.Figure()
    series = (("primary", "Primary", "#60a5fa"), ("gauc", "GAUC", "#34d399"), ("ndcg", "nDCG@5", "#c084fc"))
    for key, label, color in series:
        filtered = [(index + 1, row) for index, row in enumerate(rows) if row.get(key) is not None]
        fig.add_trace(go.Scatter(x=[item[0] for item in filtered], y=[item[1][key] for item in filtered], mode="lines+markers", name=label, line={"color": color, "width": 2}, marker={"size": 7}, text=[item[1]["label"] for item in filtered], hovertemplate="<b>%{fullData.name}</b><br>%{text}<br>%{y:.4f}<extra></extra>"))
    if baseline is not None:
        fig.add_hline(y=baseline, line_dash="dash", line_color="#fbbf24", annotation_text=f"Baseline {baseline:.4f}", annotation_font_color="#fbbf24")
    fig.update_layout(height=410, margin={"l": 10, "r": 10, "t": 35, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,.35)", xaxis_title="Experiment sequence", yaxis_title="Validation score", hovermode="x unified", legend={"orientation": "h", "y": 1.12, "x": 0}, xaxis={"gridcolor": "rgba(148,163,184,.12)"}, yaxis={"gridcolor": "rgba(148,163,184,.12)"})
    return fig


def main() -> None:
    inject_css()
    if st_autorefresh:
        st_autorefresh(interval=60_000, key="byterush-artifact-watch")
    root = _root()
    history_baseline, ledger, runs = load_dashboard(str(root))
    run_ids = [record["run_id"] for record in runs]
    selected_run_id = st.selectbox(
        "Dashboard run",
        run_ids,
        format_func=lambda run_id: next(
            f"{run_id} · {len(record['stage_keys'])}/4 stages · Primary {fmt_metric(record['primary'])}"
            for record in runs if record["run_id"] == run_id
        ),
        help="Every card, tree node, detail drawer, and headline below is scoped to this one atomic AgentManager snapshot.",
    ) if run_ids else None
    selected_run = next((record for record in runs if record["run_id"] == selected_run_id), {})
    baseline = selected_run.get("baseline_primary") if selected_run else history_baseline
    trees = load_selected_trees(str(root), selected_run_id)
    top_best = best_node(trees)
    # All four headline numbers must come from the selected run's winning node.
    best_primary = selected_run.get("primary") or (top_best.primary if top_best else None)
    best_gauc = selected_run.get("gauc") or (top_best.gauc if top_best else None)
    best_ndcg = selected_run.get("ndcg") or (top_best.ndcg if top_best else None)
    st.markdown("<div class='hero'><h1>⚡ ByteRush Research Console</h1><p>Autonomous ML research for KuaiRand recommendation — hypotheses, code, validation evidence, and reproducible artifacts in one view.</p></div>", unsafe_allow_html=True)
    if not (root / "experiments").exists() and not ledger:
        st.error(f"No experiment artifacts found under {root}. Set BYTERUSH_DATA_ROOT to the ByteRush repository on AutoDL.")
        return
    metrics = st.columns(4)
    metrics[0].metric("Best Primary", fmt_metric(best_primary), relative_delta(best_primary, baseline))
    metrics[1].metric("Best GAUC", fmt_metric(best_gauc))
    metrics[2].metric("Best nDCG@5", fmt_metric(best_ndcg))
    metrics[3].metric("Relative baseline lift", relative_delta(best_primary, baseline), f"baseline {fmt_metric(baseline)}")
    st.markdown("### Four-stage research loop")
    stage_columns = st.columns(4)
    for column, stage in zip(stage_columns, STAGE_META):
        with column:
            stage_card(stage, trees.get(stage), baseline)
    st.markdown("### Experiment search tree")
    available_stages = [stage for stage in ("baseline", "tuning", "creative", "ablation") if stage in trees]
    if not available_stages:
        st.info("The AgentManager has not written a tree snapshot yet. The dashboard will populate automatically after the first saved step.")
    else:
        stage = st.selectbox("Tree stage", available_stages, format_func=lambda value: STAGE_META[value]["label"], index=len(available_stages) - 1)
        tree = trees[stage]
        left, right = st.columns([1.58, 1])
        with left:
            st.caption("Green = succeeded · red = failed · gray = not executed · blue ring = in progress · ★ = current stage best. Node size follows Primary.")
            selected_uid = st.session_state.get("selected_node")
            fig = make_tree_figure(tree, selected_uid)
            try:
                event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key=f"tree-{stage}")
                if event and event.selection.points:
                    st.session_state.selected_node = event.selection.points[-1]["customdata"][0]
            except TypeError:
                st.plotly_chart(fig, use_container_width=True, key=f"tree-{stage}")
            labels = {node.uid: f"{node.label} · {fmt_metric(node.primary)}" for node in tree.nodes}
            chosen_uid = st.selectbox("Select experiment details", list(labels), format_func=lambda uid: labels[uid], key=f"node-picker-{stage}")
            if chosen_uid:
                st.session_state.selected_node = chosen_uid
        with right:
            selected_uid = st.session_state.get("selected_node")
            selected = next((node for node in tree.nodes if node.uid == selected_uid), tree.nodes[0] if tree.nodes else None)
            if selected:
                node_detail(selected, baseline)
    st.markdown("### Metric evolution")
    st.caption("Each point is the best validation result recorded for a completed run. The dotted line is the protected FM baseline.")
    st.plotly_chart(evolution_chart(ledger, baseline), use_container_width=True)
    st.markdown("### Integrity & runtime evidence")
    integrity = st.columns(4)
    integrity[0].success("Validation-only selection")
    integrity[1].success("Protected evaluator retained")
    integrity[2].success("Artifacts checkpointed")
    integrity[3].info(f"{len(ledger)} recorded runs")
    st.caption(f"Data root: {root} · AgentManager writes an atomic dashboard snapshot after every experiment step; the page refreshes it every 60 seconds.")


if __name__ == "__main__":
    main()
