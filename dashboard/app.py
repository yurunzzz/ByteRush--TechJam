"""ByteRush competition showcase.

Build first, then launch:
    python dashboard/build_showcase.py --data-root /root/autodl-tmp/ByteRush
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import html
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from showcase_loader import ShowcaseBuildError, load_showcase_payload


st.set_page_config(
    page_title="ByteRush · Autonomous Recommendation Research",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STAGE_COLORS = {1: "#67e8f9", 2: "#a7f3d0", 3: "#c4b5fd", 4: "#f9a8d4"}


def _manifest_path() -> Path:
    configured = os.getenv("BYTERUSH_SHOWCASE_MANIFEST")
    return Path(configured).expanduser().resolve() if configured else SCRIPT_DIR / "generated" / "showcase_manifest.json"


@st.cache_data(show_spinner=False)
def _load(path: str) -> dict[str, Any]:
    return load_showcase_payload(Path(path))


def _metric(value: float | None, digits: int = 6) -> str:
    return f"{value:.{digits}f}" if value is not None else "—"


def _delta(value: float | None, digits: int = 6) -> str:
    return f"{value:+.{digits}f}" if value is not None else "—"


def _percent(value: float | None) -> str:
    return f"{value:+.3%}" if value is not None else "—"


def _safe(text: Any) -> str:
    return html.escape(str(text or ""))


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #eaf2ff; --muted: #9eacc4; --panel: rgba(13, 21, 43, .78);
          --cyan: #67e8f9; --mint: #a7f3d0; --lilac: #c4b5fd; --pink: #f9a8d4;
        }
        html { scroll-behavior: smooth; }
        .stApp {
          background:
            radial-gradient(circle at 8% 0%, rgba(103,232,249,.13), transparent 28%),
            radial-gradient(circle at 88% 8%, rgba(196,181,253,.14), transparent 27%),
            linear-gradient(155deg, #07101f 0%, #080d19 48%, #050912 100%);
          color: var(--ink);
        }
        header[data-testid="stHeader"] { background: rgba(5,9,18,.7); backdrop-filter: blur(16px); }
        .block-container { max-width: 1280px; padding-top: 1.2rem; padding-bottom: 5rem; }
        .show-nav { position: sticky; top: 2.5rem; z-index: 8; display:flex; align-items:center;
          justify-content:space-between; gap:1rem; padding:.72rem 1rem; margin:0 0 2rem;
          border:1px solid rgba(255,255,255,.1); border-radius:999px;
          background:rgba(7,16,31,.78); backdrop-filter:blur(18px); box-shadow:0 16px 50px rgba(0,0,0,.2); }
        .show-brand { font-weight:800; letter-spacing:-.02em; }
        .show-brand span { color:var(--cyan); }
        .show-links { display:flex; gap:1rem; flex-wrap:wrap; }
        .show-links a { color:#c7d2e6 !important; text-decoration:none; font-size:.82rem; }
        .hero { position:relative; overflow:hidden; border:1px solid rgba(255,255,255,.1); border-radius:30px;
          padding:4.3rem 4rem 3.8rem; background:linear-gradient(135deg,rgba(21,37,66,.92),rgba(17,25,49,.65));
          box-shadow:0 34px 90px rgba(0,0,0,.32); }
        .hero:after { content:""; position:absolute; width:440px; height:440px; right:-120px; top:-170px;
          background:radial-gradient(circle,rgba(167,243,208,.2),rgba(196,181,253,.08) 45%,transparent 70%); }
        .eyebrow { color:var(--cyan); font-size:.76rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
        .hero h1 { font-size:clamp(2.8rem,6vw,5.4rem); line-height:.96; letter-spacing:-.065em; margin:.75rem 0 1.2rem;
          max-width:900px; background:linear-gradient(100deg,#fff 10%,#c9f8ff 45%,#e1d9ff 85%);
          -webkit-background-clip:text; color:transparent; }
        .hero p { max-width:720px; color:#b7c5dc; font-size:1.1rem; line-height:1.75; }
        .pills { display:flex; flex-wrap:wrap; gap:.6rem; margin-top:1.5rem; }
        .pill { padding:.48rem .75rem; border-radius:999px; font-size:.78rem; color:#dce8fb;
          border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.05); }
        .metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1rem; margin:1.2rem 0 0; }
        .metric-card { padding:1.25rem 1.3rem; border:1px solid rgba(255,255,255,.1); border-radius:20px;
          background:linear-gradient(145deg,rgba(19,31,58,.86),rgba(12,20,40,.72)); min-height:138px; }
        .metric-card .label { color:#9eb0c9; font-size:.79rem; text-transform:uppercase; letter-spacing:.08em; }
        .metric-card .value { color:#fff; font-size:2rem; font-weight:800; margin:.45rem 0 .25rem; letter-spacing:-.04em; }
        .metric-card .lift { color:var(--mint); font-size:.86rem; }
        .section-anchor { scroll-margin-top:5rem; }
        .section-head { margin:5.2rem 0 1.5rem; }
        .section-head h2 { font-size:2.15rem; letter-spacing:-.045em; margin:.35rem 0 .5rem; }
        .section-head p { color:var(--muted); max-width:760px; line-height:1.65; }
        .stage-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1rem; }
        .stage-card { position:relative; min-height:210px; padding:1.35rem; border-radius:22px;
          border:1px solid rgba(255,255,255,.1); background:var(--panel); overflow:hidden; }
        .stage-card:before { content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--accent); }
        .stage-card .num { color:var(--accent); font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
        .stage-card h3 { font-size:1.15rem; margin:.55rem 0 1.25rem; }
        .stage-card .big { font-size:1.75rem; font-weight:800; }
        .stage-card .small { color:var(--muted); font-size:.82rem; line-height:1.55; margin-top:.6rem; }
        .model-flow { display:grid; grid-template-columns:1fr auto 1.25fr auto 1fr; gap:1rem; align-items:center; }
        .flow-box { padding:1.45rem; border-radius:22px; background:var(--panel); border:1px solid rgba(255,255,255,.1); min-height:150px; }
        .flow-box strong { display:block; font-size:1.15rem; margin-bottom:.5rem; }
        .flow-box p { color:var(--muted); line-height:1.55; font-size:.9rem; }
        .flow-split { display:grid; grid-template-columns:1fr 1fr; gap:.7rem; }
        .mini-path { padding:1rem; border-radius:16px; background:rgba(255,255,255,.04); }
        .arrow { color:var(--cyan); font-size:1.6rem; }
        .proof-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1rem; }
        .proof { padding:1.2rem; border-radius:18px; border:1px solid rgba(167,243,208,.18); background:rgba(9,31,37,.52); }
        .proof b { display:block; color:var(--mint); margin-bottom:.35rem; }
        .proof span { color:#aebdd0; font-size:.83rem; line-height:1.5; }
        .footer { margin-top:5rem; padding:2rem 0; color:#72829c; border-top:1px solid rgba(255,255,255,.08); }
        div[data-testid="stExpander"] { border:1px solid rgba(255,255,255,.1); border-radius:16px; background:rgba(12,20,40,.55); }
        [data-testid="stDataFrame"] { border:1px solid rgba(255,255,255,.08); border-radius:16px; overflow:hidden; }
        @media (max-width:900px) {
          .hero { padding:3rem 1.5rem; } .metric-grid,.stage-grid,.proof-grid { grid-template-columns:1fr 1fr; }
          .model-flow { grid-template-columns:1fr; } .arrow { transform:rotate(90deg); text-align:center; }
          .show-links { display:none; }
        }
        @media (max-width:560px) { .metric-grid,.stage-grid,.proof-grid { grid-template-columns:1fr; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_head(anchor: str, eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"<div id='{anchor}' class='section-anchor section-head'><div class='eyebrow'>{_safe(eyebrow)}</div>"
        f"<h2>{_safe(title)}</h2><p>{_safe(description)}</p></div>",
        unsafe_allow_html=True,
    )


def hero(payload: dict[str, Any]) -> None:
    project, winner, selection = payload["project"], payload["winner"], payload["selection"]
    final, delta, relative = winner["final"], winner["delta"], winner["relative_delta"]
    st.markdown(
        f"""
        <div class="show-nav">
          <div class="show-brand"><span>⚡</span> {_safe(project['name'])}</div>
          <div class="show-links"><a href="#results">Results</a><a href="#agent">Agent</a><a href="#search">Search tree</a><a href="#model">Winner</a><a href="#evidence">Evidence</a></div>
        </div>
        <div class="hero">
          <div class="eyebrow">{_safe(project['competition'])} · Final research showcase</div>
          <h1>{_safe(project['subtitle'])}</h1>
          <p>{_safe(project['tagline'])}</p>
          <div class="pills">
            <span class="pill">✓ Validation-only selection</span><span class="pill">✓ {selection['successful_seed_count']}-seed verified</span>
            <span class="pill">✓ Submission ready</span><span class="pill">Winner · {_safe(winner['label'])}</span>
          </div>
        </div>
        <div id="results" class="section-anchor metric-grid">
          <div class="metric-card"><div class="label">Primary</div><div class="value">{_metric(final['primary'])}</div><div class="lift">{_delta(delta['primary'])} · {_percent(relative['primary'])}</div></div>
          <div class="metric-card"><div class="label">GAUC</div><div class="value">{_metric(final['GAUC'])}</div><div class="lift">{_delta(delta['GAUC'])} vs FM</div></div>
          <div class="metric-card"><div class="label">nDCG@5</div><div class="value">{_metric(final['nDCG@5'])}</div><div class="lift">{_delta(delta['nDCG@5'])} vs FM</div></div>
          <div class="metric-card"><div class="label">Verified seeds</div><div class="value">{selection['successful_seed_count']}</div><div class="lift">σ Primary {_metric(winner['metrics']['primary']['std'])}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_comparison(payload: dict[str, Any]) -> go.Figure:
    winner = payload["winner"]
    names, keys = ["GAUC", "nDCG@5", "Primary"], ["GAUC", "nDCG@5", "primary"]
    baseline = [winner["baseline"][key] for key in keys]
    final = [winner["final"][key] for key in keys]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="FM baseline", x=names, y=baseline, marker_color="#64748b", text=[_metric(value) for value in baseline], textposition="outside"))
    fig.add_trace(go.Bar(name=winner["label"], x=names, y=final, marker_color=["#67e8f9", "#c4b5fd", "#a7f3d0"], text=[_metric(value) for value in final], textposition="outside"))
    fig.update_layout(
        barmode="group", height=430, margin={"l": 15, "r": 15, "t": 20, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,21,43,.42)", font={"color": "#dce8fb"},
        legend={"orientation": "h", "y": 1.1},
        yaxis={"range": [min(baseline + final) - .003, max(baseline + final) + .003], "gridcolor": "rgba(255,255,255,.08)", "tickformat": ".3f"},
        xaxis={"gridcolor": "rgba(255,255,255,0)"},
    )
    return fig


def seed_figure(payload: dict[str, Any]) -> go.Figure:
    metrics = payload["winner"]["metrics"]
    fig = go.Figure()
    palette = {"GAUC": "#67e8f9", "nDCG@5": "#c4b5fd", "primary": "#a7f3d0"}
    for name in ("GAUC", "nDCG@5", "primary"):
        values = metrics[name]["values"]
        fig.add_trace(go.Scatter(
            x=list(range(1, len(values) + 1)), y=values, mode="lines+markers", name="Primary" if name == "primary" else name,
            line={"color": palette[name], "width": 2}, marker={"size": 9},
            hovertemplate=f"{name}<br>Seed %{{x}} · %{{y:.6f}}<extra></extra>",
        ))
    fig.update_layout(
        height=420, margin={"l": 15, "r": 15, "t": 25, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,21,43,.42)", font={"color": "#dce8fb"},
        xaxis={"title": "Independent verification seed", "dtick": 1, "gridcolor": "rgba(255,255,255,.08)"},
        yaxis={"title": "Validation score", "gridcolor": "rgba(255,255,255,.08)"},
        legend={"orientation": "h", "y": 1.12}, hovermode="x unified",
    )
    return fig


def _tree_subset(payload: dict[str, Any], mode: str) -> tuple[list[dict[str, Any]], list[list[str]]]:
    search = payload["search"]
    all_nodes = {node["id"]: node for node in search["nodes"]}
    if mode == "Champion path":
        keep = set(search["champion_path"])
    elif mode == "Curated evidence":
        ranked = sorted(
            (node for node in all_nodes.values() if node["status"] == "succeeded" and node["primary"] is not None),
            key=lambda node: node["primary"], reverse=True,
        )[:18]
        keep = set(search["champion_path"]) | {node["id"] for node in ranked}
        changed = True
        while changed:
            changed = False
            for node_id in list(keep):
                parent = all_nodes.get(node_id, {}).get("parent_id")
                if parent and parent in all_nodes and parent not in keep:
                    keep.add(parent)
                    changed = True
    else:
        keep = set(all_nodes)
    nodes = [node for node in all_nodes.values() if node["id"] in keep]
    edges = [edge for edge in search["edges"] if edge[0] in keep and edge[1] in keep]
    return nodes, edges


def search_tree(payload: dict[str, Any], mode: str) -> go.Figure:
    nodes, edges = _tree_subset(payload, mode)
    by_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_stage[int(node["stage_number"])].append(node)
    position: dict[str, tuple[float, float]] = {}
    for stage, members in by_stage.items():
        members.sort(key=lambda node: (not node["is_champion_path"], -(node["primary"] or 0), node["id"]))
        for index, node in enumerate(members):
            position[node["id"]] = (stage, (index + 1) / (len(members) + 1))
    lookup = {node["id"]: node for node in nodes}
    fig = go.Figure()
    for parent, child in edges:
        if parent not in position or child not in position:
            continue
        x0, y0 = position[parent]
        x1, y1 = position[child]
        highlighted = lookup[parent]["is_champion_path"] and lookup[child]["is_champion_path"]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
            line={"color": "rgba(167,243,208,.75)" if highlighted else "rgba(148,163,184,.22)", "width": 3 if highlighted else 1.2},
        ))
    for stage in sorted(by_stage):
        members = by_stage[stage]
        fig.add_trace(go.Scatter(
            x=[position[node["id"]][0] for node in members], y=[position[node["id"]][1] for node in members], mode="markers",
            name=f"Stage {stage}", customdata=[[node["label"], node["primary"], node["status"], node["id"][:8]] for node in members],
            marker={
                "size": [25 if node["is_final"] else 19 if node["is_champion_path"] else 12 for node in members],
                "color": ["#fbbf24" if node["is_final"] else STAGE_COLORS.get(stage, "#94a3b8") for node in members],
                "symbol": ["diamond" if node["is_final"] else "x" if node["status"] == "failed" else "circle" for node in members],
                "line": {"color": "#f8fafc", "width": [3 if node["is_final"] else 1 for node in members]},
                "opacity": [.32 if node["status"] == "failed" else 1 for node in members],
            },
            hovertemplate="<b>%{customdata[0]}</b><br>Primary %{customdata[1]:.6f}<br>%{customdata[2]} · %{customdata[3]}<extra></extra>",
        ))
    fig.update_layout(
        height=600 if mode == "Full search" else 510, margin={"l": 10, "r": 10, "t": 20, "b": 25},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,21,43,.42)", font={"color": "#dce8fb"},
        xaxis={"tickmode": "array", "tickvals": [1, 2, 3, 4], "ticktext": ["Diverse roots", "Tuning", "Research", "Verification"], "range": [.7, 4.3], "gridcolor": "rgba(255,255,255,.08)"},
        yaxis={"visible": False, "range": [0, 1]}, legend={"orientation": "h", "y": 1.1, "font": {"color": "#dce8fb"}},
    )
    return fig


def stage_cards(payload: dict[str, Any]) -> None:
    descriptions = {
        "baseline": "Establish the FM reference and generate diverse model-family roots.",
        "tuning": "Tune promising roots under one controlled change at a time.",
        "creative": "Explore new objectives, features, and cross-parent transfers.",
        "ablation": "Verify stability across seeds and freeze submission artifacts.",
    }
    cards = []
    for index, stage in enumerate(payload["search"]["stages"], 1):
        accent = STAGE_COLORS.get(index, "#94a3b8")
        cards.append(
            f"<div class='stage-card' style='--accent:{accent}'><div class='num'>{_safe(stage['eyebrow'])}</div>"
            f"<h3>{_safe(stage['title'])}</h3><div class='big'>{stage['succeeded']}<span style='font-size:.9rem;color:#8fa1ba'> / {stage['total']} succeeded</span></div>"
            f"<div class='small'>{_safe(descriptions.get(stage['key'], ''))}</div></div>"
        )
    st.markdown("<div class='stage-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def model_story(payload: dict[str, Any]) -> None:
    winner = payload["winner"]
    st.markdown(
        f"""
        <div class="model-flow">
          <div class="flow-box"><div class="eyebrow">Inputs</div><strong>User · video · context</strong><p>Encoded interaction features enter the same trusted validation pipeline used by the baseline.</p></div>
          <div class="arrow">→</div>
          <div class="flow-box"><div class="eyebrow">{_safe(winner['label'])}</div><div class="flow-split">
            <div class="mini-path"><strong>Wide path</strong><p>Memorizes reliable low-order interaction patterns.</p></div>
            <div class="mini-path"><strong>Deep path</strong><p>Learns nonlinear higher-order feature combinations.</p></div>
          </div></div>
          <div class="arrow">→</div>
          <div class="flow-box"><div class="eyebrow">Ranking</div><strong>Long-view score</strong><p>One score per exposed video, used to rank each user's candidates.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def evidence_table(payload: dict[str, Any]) -> None:
    rows = [{
        "Stage": f"Stage {node['stage_number']}", "Candidate": node["label"], "Primary": node["primary"],
        "GAUC": node["gauc"], "nDCG@5": node["ndcg"], "Champion path": "Yes" if node["is_champion_path"] else "No",
    } for node in payload["search"]["top_candidates"]]
    st.dataframe(rows, width="stretch", hide_index=True, column_config={
        "Primary": st.column_config.NumberColumn(format="%.6f"), "GAUC": st.column_config.NumberColumn(format="%.6f"),
        "nDCG@5": st.column_config.NumberColumn(format="%.6f"),
    })


def main() -> None:
    inject_css()
    manifest_path = _manifest_path()
    try:
        payload = _load(str(manifest_path))
    except ShowcaseBuildError as exc:
        st.error(str(exc))
        st.code(
            "python dashboard/build_showcase.py --data-root /root/autodl-tmp/ByteRush\n"
            "python -m streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501",
            language="bash",
        )
        return

    hero(payload)
    story = payload.get("story") or {}
    section_head("impact", "Measured impact", "A small number, earned the hard way.", "Every result below comes from the same frozen model and the same protected validation protocol; no test metric was used for selection.")
    left, right = st.columns([1.42, 1])
    with left:
        st.plotly_chart(metric_comparison(payload), width="stretch", key="metric-comparison")
    with right:
        st.markdown("### Why this result matters")
        st.write(story.get("problem") or "Recommendation research normally requires repeated manual experiments.")
        st.write(story.get("solution") or "ByteRush automates the evidence-guided research loop.")
        baseline = payload["winner"]["baseline"]
        st.info(f"Protected FM reference · Primary {_metric(baseline['primary'])}")

    section_head("agent", "Autonomous research loop", "Four stages. One evidence trail.", "The Agent first broadens the search space, then spends its budget on promising directions, learns from failures, and verifies the final promotion.")
    stage_cards(payload)
    section_head("search", "Search provenance", "Follow the path to the champion.", "The default view removes noise and shows only the exact ancestry of the frozen model. Switch views to inspect the broader evidence behind the decision.")
    mode = st.segmented_control("Tree detail", ["Champion path", "Curated evidence", "Full search"], default="Champion path", label_visibility="collapsed")
    st.plotly_chart(search_tree(payload, mode or "Champion path"), width="stretch", key=f"search-tree-{mode}")
    stats = payload["search"]
    cols = st.columns(4)
    cols[0].metric("Generated nodes", stats["total_nodes"])
    cols[1].metric("Successful", stats["successful_nodes"])
    cols[2].metric("Failed safely", stats["failed_nodes"])
    cols[3].metric("Research rounds", stats["research_rounds"])

    section_head("model", "Frozen winner", f"Why {payload['winner']['label']} won.", story.get("winner_summary") or payload["winner"]["principal_change"])
    model_story(payload)
    takeaways = story.get("takeaways") or []
    if takeaways:
        st.markdown("### What the Agent learned")
        for index, takeaway in enumerate(takeaways, 1):
            st.markdown(f"**{index:02d}.** {takeaway}")

    section_head("stability", "Final verification", "Stable across independent seeds.", "The promoted model was not frozen after a lucky single run. Stage 4 repeated training and compared the full validation metric set.")
    st.plotly_chart(seed_figure(payload), width="stretch", key="seed-stability")
    stability = st.columns(3)
    for column, name in zip(stability, ("GAUC", "nDCG@5", "primary")):
        metric = payload["winner"]["metrics"][name]
        column.metric("Primary" if name == "primary" else name, _metric(metric["mean"]), f"σ {_metric(metric['std'])}")

    section_head("evidence", "Audit trail", "A result the judges can inspect.", "The story stays concise, but the underlying candidates, hashes, model, checkpoint, and submission remain traceable.")
    integrity = payload["integrity"]
    st.markdown(
        f"""
        <div class="proof-grid">
          <div class="proof"><b>Validation-only</b><span>Test metrics used for selection: no</span></div>
          <div class="proof"><b>{integrity['successful_seed_count']} verified seeds</b><span>Final model stability checked before freezing</span></div>
          <div class="proof"><b>{integrity['submission_rows']:,} predictions</b><span>Submission file generated; no hidden labels accessed</span></div>
          <div class="proof"><b>Immutable evidence</b><span>Model, checkpoint, and CSV recorded with SHA-256 hashes</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Leading candidates")
    evidence_table(payload)
    with st.expander("Reproducibility paths and hashes"):
        st.json({"run": payload["selection"], "files": payload["files"], "integrity": payload["integrity"]}, expanded=True)
    with st.expander("Training history"):
        st.dataframe(payload["training_history"], width="stretch", hide_index=True)
    st.markdown(
        f"<div class='footer'>Frozen showcase · run {_safe(payload['selection']['run_id'])} · source node {_safe(payload['selection']['source_node_id'][:12])} · metrics are validation-only</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
