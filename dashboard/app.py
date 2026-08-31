"""ByteRush 16:9 competition showcase.

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
    page_title="ByteRush! · Autonomous Recommendation Research",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STAGE_COLORS = {1: "#57d7ef", 2: "#86e7c1", 3: "#ae94f4", 4: "#ffbd4a"}
SLIDE_KEYS = (
    "slide-01-hero", "slide-02-impact", "slide-03-agent", "slide-04-search",
    "slide-05-selection", "slide-06-stability", "slide-07-evidence",
)


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
    slide_selectors = ",".join(f".st-key-{key}" for key in SLIDE_KEYS)
    st.markdown(
        f"""
        <style>
        :root {{
          --ink:#edf5ff; --muted:#a9b8cf; --panel:rgba(13,23,46,.84);
          --cyan:#57d7ef; --mint:#86e7c1; --lilac:#ae94f4; --gold:#ffbd4a;
        }}
        html {{ scroll-behavior:smooth; scroll-snap-type:y proximity; }}
        .stApp {{
          background:
            radial-gradient(circle at 8% 0%,rgba(87,215,239,.12),transparent 27%),
            radial-gradient(circle at 90% 8%,rgba(174,148,244,.13),transparent 28%),
            linear-gradient(155deg,#07111f 0%,#090e1b 52%,#050811 100%);
          color:var(--ink);
        }}
        header[data-testid="stHeader"] {{ background:rgba(5,9,18,.58); backdrop-filter:blur(16px); }}
        .block-container {{ max-width:1540px; padding-top:4.15rem; padding-bottom:5rem; }}
        {slide_selectors} {{
          position:relative; width:100%; aspect-ratio:16/9; min-height:760px; margin:1.8rem 0 3rem;
          padding:clamp(1.8rem,3vw,3.5rem); border:1px solid rgba(255,255,255,.1); border-radius:30px;
          background:linear-gradient(145deg,rgba(12,23,44,.9),rgba(7,13,27,.84));
          box-shadow:0 34px 90px rgba(0,0,0,.3); scroll-snap-align:start; overflow:hidden;
          animation:slideReveal .7s cubic-bezier(.2,.7,.2,1) both;
        }}
        .st-key-slide-01-hero {{ background:
          radial-gradient(circle at 83% 20%,rgba(134,231,193,.17),transparent 28%),
          radial-gradient(circle at 12% 5%,rgba(87,215,239,.12),transparent 30%),
          linear-gradient(135deg,rgba(19,38,70,.96),rgba(11,18,37,.9)); }}
        @keyframes slideReveal {{ from {{ opacity:.2; transform:translateY(22px); }} to {{ opacity:1; transform:translateY(0); }} }}
        @keyframes graphReveal {{ from {{ opacity:.15; transform:translateX(20px) scale(.985); }} to {{ opacity:1; transform:translateX(0) scale(1); }} }}
        .st-key-tree-frame-champion-path,.st-key-tree-frame-curated-evidence,.st-key-tree-frame-full-search {{ animation:graphReveal .65s ease both; }}
        .show-nav {{ position:sticky; top:4.15rem; z-index:20; display:flex; align-items:center; justify-content:space-between;
          gap:1rem; padding:.7rem 1rem; margin:0 auto 1rem; max-width:1480px; border:1px solid rgba(255,255,255,.1);
          border-radius:999px; background:rgba(7,16,31,.82); backdrop-filter:blur(18px); box-shadow:0 16px 50px rgba(0,0,0,.24); }}
        .show-brand {{ font-weight:900; font-size:1rem; letter-spacing:-.02em; }}
        .show-brand span {{ color:var(--cyan); }}
        .show-links {{ display:flex; gap:1.2rem; flex-wrap:wrap; }}
        .show-links a {{ color:#d3def0 !important; text-decoration:none; font-size:.82rem; font-weight:650; }}
        .show-links a:hover {{ color:var(--cyan) !important; }}
        .slide-anchor {{ scroll-margin-top:5rem; }}
        .slide-index {{ position:absolute; right:2.2rem; bottom:1.55rem; color:#65758e; font-size:.75rem; letter-spacing:.14em; text-transform:uppercase; }}
        .eyebrow {{ color:var(--cyan); font-size:.86rem; font-weight:900; letter-spacing:.17em; text-transform:uppercase; }}
        .section-head {{ margin:.1rem 0 1.2rem; }}
        .section-head h2 {{ font-size:clamp(2.1rem,3.1vw,3.4rem); line-height:1.04; letter-spacing:-.05em; margin:.45rem 0 .65rem; }}
        .section-head p {{ color:var(--muted); max-width:980px; font-size:1.08rem; line-height:1.58; margin:0; }}
        .team-word {{
          display:inline-block; margin:.45rem 0 0; font-family:Impact,"Arial Black","Avenir Next Condensed",sans-serif;
          font-size:clamp(6.8rem,13vw,12.8rem); line-height:.86; font-style:italic; letter-spacing:-.065em;
          background:linear-gradient(105deg,#ffffff 4%,#7cecff 36%,#a5f3d2 62%,#c9b8ff 90%);
          -webkit-background-clip:text; color:transparent; filter:drop-shadow(0 0 28px rgba(87,215,239,.2)); transform:skewX(-4deg);
        }}
        .team-word span {{ color:var(--gold); -webkit-text-fill-color:var(--gold); text-shadow:0 0 30px rgba(255,189,74,.35); }}
        .hero-subtitle {{ margin:.9rem 0 0; max-width:940px; font-size:clamp(1.25rem,2vw,2rem); color:#dce8f9; font-weight:650; letter-spacing:-.02em; }}
        .hero-tagline {{ margin:.55rem 0 1.25rem; color:#9fb0c7; font-size:1rem; }}
        .pills {{ display:flex; flex-wrap:wrap; gap:.58rem; margin-top:1rem; }}
        .pill {{ padding:.47rem .75rem; border-radius:999px; font-size:.78rem; color:#dce8fb; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.05); }}
        .hero-metrics {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; margin-top:1.7rem; }}
        .hero-metric {{ padding:1.18rem 1.35rem; border:1px solid rgba(255,255,255,.11); border-radius:22px;
          background:linear-gradient(145deg,rgba(19,35,65,.9),rgba(10,19,38,.75)); box-shadow:inset 0 1px rgba(255,255,255,.04); }}
        .hero-metric .label {{ color:#a6b6cc; font-size:.8rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }}
        .hero-metric .value {{ color:#fff; font-size:clamp(2.2rem,3.5vw,3.8rem); line-height:1; font-weight:900; margin:.42rem 0 .55rem; letter-spacing:-.055em; }}
        .hero-metric .jump {{ color:var(--mint); font-size:1.08rem; font-weight:850; }}
        .hero-metric .jump strong {{ display:inline-block; margin-left:.35rem; padding:.13rem .42rem; border-radius:8px; background:rgba(134,231,193,.1); color:#b7f7dc; }}
        .hero-metric .baseline {{ color:#8192aa; font-size:.76rem; margin-top:.35rem; }}
        .stage-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1rem; margin-top:2rem; }}
        .stage-card {{ position:relative; min-height:300px; padding:1.55rem; border-radius:24px; border:1px solid rgba(255,255,255,.1); background:var(--panel); overflow:hidden; }}
        .stage-card:before {{ content:""; position:absolute; inset:0 auto 0 0; width:4px; background:var(--accent); }}
        .stage-card .num {{ color:var(--accent); font-size:.82rem; font-weight:900; letter-spacing:.14em; text-transform:uppercase; }}
        .stage-card h3 {{ font-size:1.42rem; margin:.7rem 0 1.8rem; }}
        .stage-card .big {{ font-size:2.45rem; font-weight:900; color:#fff; }}
        .stage-card .small {{ color:var(--muted); font-size:.94rem; line-height:1.62; margin-top:.8rem; }}
        .impact-copy {{ padding:1.4rem 1.5rem; border-radius:22px; background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.08); }}
        .impact-copy h3 {{ font-size:1.45rem; margin:.1rem 0 .8rem; }}
        .impact-copy p {{ color:#b6c4d8; font-size:1.02rem; line-height:1.68; }}
        .baseline-chip {{ display:inline-flex; margin-top:.7rem; padding:.65rem .8rem; border-radius:12px; background:rgba(87,215,239,.08); color:#bceffc; font-weight:800; }}
        .axis-note {{ color:#72839c; font-size:.76rem; margin-top:-.7rem; }}
        .search-stat-row {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.8rem; margin-top:-.1rem; }}
        .search-stat {{ padding:.72rem .9rem; border-radius:14px; background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.07); }}
        .search-stat b {{ color:var(--gold); font-size:1.45rem; margin-right:.35rem; }}
        .search-stat span {{ color:#9fb0c7; font-size:.82rem; }}
        .selection-track {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.75rem; margin-top:2rem; align-items:stretch; }}
        .selection-step {{ position:relative; padding:1.3rem 1.15rem; min-height:300px; border-radius:22px; background:var(--panel); border:1px solid rgba(255,255,255,.1); }}
        .selection-step:not(:last-child):after {{ content:"→"; position:absolute; z-index:3; right:-1rem; top:44%; color:var(--cyan); font-size:1.45rem; font-weight:900; }}
        .selection-step .icon {{ font-size:2rem; }}
        .selection-step .stage {{ color:var(--accent); font-size:.75rem; font-weight:900; letter-spacing:.13em; text-transform:uppercase; margin-top:.8rem; }}
        .selection-step h3 {{ font-size:1.18rem; margin:.45rem 0 .7rem; }}
        .selection-step .score {{ color:#fff; font-size:1.72rem; font-weight:900; letter-spacing:-.04em; }}
        .selection-step .why {{ color:#9fb0c7; font-size:.85rem; line-height:1.55; margin-top:.7rem; }}
        .selection-verdict {{ margin-top:1.15rem; padding:1rem 1.2rem; border-radius:17px; background:linear-gradient(90deg,rgba(134,231,193,.1),rgba(174,148,244,.08)); border:1px solid rgba(134,231,193,.16); color:#d8f8e9; font-size:1rem; }}
        .proof-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.9rem; }}
        .proof {{ padding:1.05rem; border-radius:17px; border:1px solid rgba(134,231,193,.18); background:rgba(9,31,37,.52); }}
        .proof b {{ display:block; color:var(--mint); font-size:1rem; margin-bottom:.35rem; }}
        .proof span {{ color:#aebdd0; font-size:.82rem; line-height:1.45; }}
        .footer {{ color:#70819a; font-size:.78rem; margin-top:.6rem; }}
        [data-testid="stMetric"] {{ background:rgba(15,27,50,.72); border:1px solid rgba(255,255,255,.08); padding:.85rem 1rem; border-radius:16px; }}
        [data-testid="stMetricLabel"] {{ color:#b9c8dc; font-size:1rem; }}
        [data-testid="stMetricValue"] {{ color:#fff; font-size:2rem; }}
        [data-testid="stMetricDelta"] {{ color:var(--mint); font-size:.92rem; }}
        div[data-testid="stExpander"] {{ border:1px solid rgba(255,255,255,.09); border-radius:14px; background:rgba(12,20,40,.55); }}
        [data-testid="stDataFrame"] {{ border:1px solid rgba(255,255,255,.08); border-radius:15px; overflow:hidden; font-size:.9rem; }}
        div[role="radiogroup"] label {{ font-size:1rem !important; font-weight:750; }}
        @media (max-width:1000px) {{
          {slide_selectors} {{ aspect-ratio:auto; min-height:auto; overflow:visible; }}
          .hero-metrics,.stage-grid,.proof-grid {{ grid-template-columns:1fr 1fr; }}
          .selection-track {{ grid-template-columns:1fr 1fr; }}
          .selection-step:not(:last-child):after {{ display:none; }}
          .show-links {{ display:none; }}
        }}
        @media (max-width:620px) {{ .hero-metrics,.stage-grid,.proof-grid,.selection-track {{ grid-template-columns:1fr; }} .team-word {{ font-size:5.2rem; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def navigation() -> None:
    st.markdown(
        """
        <div class="show-nav">
          <div class="show-brand"><span>⚡</span> ByteRush!</div>
          <div class="show-links">
            <a href="#home">01 Home</a><a href="#impact">02 Results</a><a href="#agent">03 Agent</a>
            <a href="#search">04 Search</a><a href="#selection">05 Selection</a><a href="#stability">06 Verification</a><a href="#evidence">07 Evidence</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def slide_index(number: str, label: str) -> None:
    st.markdown(f"<div class='slide-index'>{_safe(number)} · {_safe(label)}</div>", unsafe_allow_html=True)


def section_head(anchor: str, eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"<div id='{anchor}' class='slide-anchor section-head'><div class='eyebrow'>{_safe(eyebrow)}</div>"
        f"<h2>{_safe(title)}</h2><p>{_safe(description)}</p></div>",
        unsafe_allow_html=True,
    )


def hero(payload: dict[str, Any]) -> None:
    project, winner, selection = payload["project"], payload["winner"], payload["selection"]
    final, delta, relative, baseline = winner["final"], winner["delta"], winner["relative_delta"], winner["baseline"]
    st.markdown(
        f"""
        <div id="home" class="slide-anchor">
          <div class="eyebrow">{_safe(project['competition'])} · Final research showcase</div>
          <div class="team-word">ByteRush<span>!</span></div>
          <div class="hero-subtitle">{_safe(project['subtitle'])}</div>
          <div class="hero-tagline">{_safe(project['tagline'])}</div>
          <div class="pills">
            <span class="pill">✓ Validation-only selection</span><span class="pill">✓ {selection['successful_seed_count']}-seed verified</span>
            <span class="pill">✓ 170,588 predictions ready</span><span class="pill">🏆 {_safe(winner['label'])}</span>
          </div>
          <div class="hero-metrics">
            <div class="hero-metric"><div class="label">Primary · Frozen winner</div><div class="value">{_metric(final['primary'])}</div><div class="jump">↑ {_delta(delta['primary'])}<strong>{_percent(relative['primary'])}</strong></div><div class="baseline">FM baseline {_metric(baseline['primary'])}</div></div>
            <div class="hero-metric"><div class="label">GAUC · Global ranking</div><div class="value">{_metric(final['GAUC'])}</div><div class="jump">↑ {_delta(delta['GAUC'])}</div><div class="baseline">FM baseline {_metric(baseline['GAUC'])}</div></div>
            <div class="hero-metric"><div class="label">nDCG@5 · Top-five quality</div><div class="value">{_metric(final['nDCG@5'])}</div><div class="jump">↑ {_delta(delta['nDCG@5'])}</div><div class="baseline">FM baseline {_metric(baseline['nDCG@5'])}</div></div>
          </div>
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
    fig.add_trace(go.Bar(
        name="FM baseline", x=names, y=baseline, marker_color="#5f718c",
        text=[_metric(value) for value in baseline], textposition="outside", textfont={"size": 18, "color": "#c7d2e3"},
    ))
    fig.add_trace(go.Bar(
        name=winner["label"], x=names, y=final, marker_color=["#57d7ef", "#ae94f4", "#86e7c1"],
        text=[_metric(value) for value in final], textposition="outside", textfont={"size": 20, "color": "#ffffff"},
    ))
    for index, key in enumerate(keys):
        fig.add_annotation(
            x=names[index], y=final[index], yshift=34, text=f"<b>↑ {_delta(winner['delta'][key])}</b>",
            showarrow=False, font={"size": 17, "color": "#a7f3d0"},
        )
    fig.update_layout(
        barmode="group", height=500, margin={"l": 20, "r": 25, "t": 55, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,16,34,.45)",
        font={"color": "#e7effb", "size": 18}, legend={"orientation": "h", "y": 1.12, "font": {"size": 18}},
        yaxis={"range": [min(baseline + final) - .003, max(baseline + final) + .008], "gridcolor": "rgba(255,255,255,.09)", "tickformat": ".3f", "tickfont": {"size": 17}},
        xaxis={"gridcolor": "rgba(255,255,255,0)", "tickfont": {"size": 20, "color": "#f2f6fc"}},
        hoverlabel={"font": {"size": 16}, "bgcolor": "#111d35"},
    )
    return fig


def seed_figure(payload: dict[str, Any]) -> go.Figure:
    metrics = payload["winner"]["metrics"]
    fig = go.Figure()
    palette = {"GAUC": "#57d7ef", "nDCG@5": "#ae94f4", "primary": "#86e7c1"}
    for name in ("GAUC", "nDCG@5", "primary"):
        values = metrics[name]["values"]
        fig.add_trace(go.Scatter(
            x=list(range(1, len(values) + 1)), y=values, mode="lines+markers", name="Primary" if name == "primary" else name,
            line={"color": palette[name], "width": 3}, marker={"size": 12, "line": {"color": "#f8fafc", "width": 1}},
            hovertemplate=f"<b>{name}</b><br>Seed %{{x}} · %{{y:.6f}}<extra></extra>",
        ))
    fig.update_layout(
        height=380, margin={"l": 20, "r": 20, "t": 35, "b": 35}, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,16,34,.45)", font={"color": "#e7effb", "size": 18},
        xaxis={"title": {"text": "Independent verification seed", "font": {"size": 19}}, "dtick": 1, "tickfont": {"size": 17}, "gridcolor": "rgba(255,255,255,.09)"},
        yaxis={"title": {"text": "Validation score", "font": {"size": 19}}, "tickfont": {"size": 17}, "gridcolor": "rgba(255,255,255,.09)"},
        legend={"orientation": "h", "y": 1.12, "font": {"size": 18}}, hovermode="x unified", hoverlabel={"font": {"size": 16}},
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


def _tree_icon(node: dict[str, Any], mode: str) -> tuple[str, str] | None:
    if node["is_final"]:
        return "🏆", "Frozen winner"
    if not node.get("parent_id"):
        return "🧭", "FM baseline"
    if mode == "Champion path":
        if node["stage_number"] == 1:
            return "🌱", "Wide & Deep root"
        if node["stage_number"] == 2:
            return "🎛️", "Tuned incumbent"
        if node["stage_number"] == 3:
            return "🧪", "Research candidate"
    return None


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
            line={"color": "rgba(134,231,193,.86)" if highlighted else "rgba(148,163,184,.25)", "width": 4 if highlighted else 1.6},
        ))
    for stage in sorted(by_stage):
        members = by_stage[stage]
        fig.add_trace(go.Scatter(
            x=[position[node["id"]][0] for node in members], y=[position[node["id"]][1] for node in members], mode="markers",
            name=f"Stage {stage}", customdata=[[node["label"], node["primary"], node["gauc"], node["ndcg"], node["status"], node["id"][:8]] for node in members],
            marker={
                "size": [38 if node["is_final"] else 31 if node["is_champion_path"] else 17 for node in members],
                "color": ["#ffbd4a" if node["is_final"] else STAGE_COLORS.get(stage, "#94a3b8") for node in members],
                "symbol": ["diamond" if node["is_final"] else "x" if node["status"] == "failed" else "circle" for node in members],
                "line": {"color": "#f8fafc", "width": [4 if node["is_final"] else 2 for node in members]},
                "opacity": [.34 if node["status"] == "failed" else 1 for node in members],
            },
            hovertemplate="<b>%{customdata[0]}</b><br>Primary <b>%{customdata[1]:.6f}</b><br>GAUC %{customdata[2]:.6f}<br>nDCG@5 %{customdata[3]:.6f}<br>%{customdata[4]} · %{customdata[5]}<extra></extra>",
        ))
    icons = [(node, _tree_icon(node, mode)) for node in nodes]
    icons = [(node, icon) for node, icon in icons if icon is not None and (mode == "Champion path" or node["is_final"] or not node.get("parent_id"))]
    if icons:
        fig.add_trace(go.Scatter(
            x=[position[node["id"]][0] for node, _ in icons], y=[position[node["id"]][1] for node, _ in icons],
            mode="text", text=[icon[0] for _, icon in icons], textfont={"size": 17}, hoverinfo="skip", showlegend=False,
        ))
        for node, icon in icons:
            x, y = position[node["id"]]
            fig.add_annotation(
                x=x, y=min(y + .105, .97), text=f"<b>{icon[1]}</b><br><span style='color:#a9b8cf'>{_metric(node['primary'])}</span>",
                showarrow=False, bgcolor="rgba(7,16,31,.86)", bordercolor=STAGE_COLORS.get(node["stage_number"], "#ffbd4a"),
                borderwidth=1, borderpad=5, font={"size": 14, "color": "#f7fbff"},
            )
    fig.update_layout(
        height=360, margin={"l": 15, "r": 15, "t": 42, "b": 35}, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,16,34,.48)", font={"color": "#e7effb", "size": 18},
        xaxis={"tickmode": "array", "tickvals": [1, 2, 3, 4], "ticktext": ["Diverse roots", "Tuning", "Research", "Verification"], "range": [.72, 4.28], "gridcolor": "rgba(255,255,255,.08)", "tickfont": {"size": 19, "color": "#dce8f9"}},
        yaxis={"visible": False, "range": [0, 1.06]}, legend={"orientation": "h", "y": 1.12, "font": {"color": "#e7effb", "size": 17}},
        hoverlabel={"font": {"size": 16}, "bgcolor": "#111d35"}, transition={"duration": 650, "easing": "cubic-in-out"}, uirevision="byterush-search-tree",
    )
    return fig


def stage_cards(payload: dict[str, Any]) -> None:
    descriptions = {
        "baseline": "Verify the FM reference, then open diverse MLP, Wide & Deep, and DCN research roots.",
        "tuning": "Allocate controlled trials to promising roots and promote the strongest validated candidate.",
        "creative": "Challenge the incumbent with new objectives, features, and cross-parent transfers.",
        "ablation": "Repeat independent seeds, test stability, then freeze the model and submission artifacts.",
    }
    cards = []
    for index, stage in enumerate(payload["search"]["stages"], 1):
        accent = STAGE_COLORS.get(index, "#94a3b8")
        cards.append(
            f"<div class='stage-card' style='--accent:{accent}'><div class='num'>{_safe(stage['eyebrow'])}</div>"
            f"<h3>{_safe(stage['title'])}</h3><div class='big'>{stage['succeeded']}<span style='font-size:1rem;color:#91a3bb'> / {stage['total']} succeeded</span></div>"
            f"<div class='small'>{_safe(descriptions.get(stage['key'], ''))}</div></div>"
        )
    st.markdown("<div class='stage-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def selection_story(payload: dict[str, Any]) -> None:
    search, winner = payload["search"], payload["winner"]
    lookup = {node["id"]: node for node in search["nodes"]}
    path = [lookup[node_id] for node_id in search["champion_path"] if node_id in lookup]
    baseline = path[0]
    diverse = next((node for node in path[1:] if node["stage_number"] == 1), path[1])
    tuned = next((node for node in path if node["stage_number"] == 2), diverse)
    final = path[-1]
    creative = next(stage for stage in search["stages"] if stage["key"] == "creative")
    steps = [
        ("🧭", "Stage 1A", "Lock the reference", baseline["primary"], "Run the organizer-aligned FM pipeline to create the protected comparison point.", 1),
        ("🌱", "Stage 1B", "Open diverse roots", diverse["primary"], "Compare FM, MLP, Wide & Deep, and DCN. Wide & Deep becomes the strongest root.", 1),
        ("🎛️", "Stage 2", "Promote after tuning", tuned["primary"], "Controlled hyperparameter trials improve the Wide & Deep root and make it the incumbent.", 2),
        ("🧪", "Stage 3", "Challenge the incumbent", None, f"{creative['succeeded']} of {creative['total']} creative trials succeed, but none beats the promotion gate.", 3),
        ("🏆", "Stage 4", "Verify and freeze", final["primary"], f"{payload['selection']['successful_seed_count']} independent seeds confirm stability; model and submission are frozen.", 4),
    ]
    cards = []
    for icon, stage, title, score, why, number in steps:
        score_text = _metric(score) if score is not None else "Incumbent held"
        cards.append(
            f"<div class='selection-step' style='--accent:{STAGE_COLORS[number]}'><div class='icon'>{icon}</div><div class='stage'>{stage}</div>"
            f"<h3>{_safe(title)}</h3><div class='score'>{_safe(score_text)}</div><div class='why'>{_safe(why)}</div></div>"
        )
    st.markdown(
        "<div class='selection-track'>" + "".join(cards) + "</div>"
        f"<div class='selection-verdict'><b>Selection verdict:</b> {_safe(winner['label'])} wins because it delivers the highest promoted validation result, survives Stage 3 challenges, and remains stable across multi-seed verification—not because of a single lucky score.</div>",
        unsafe_allow_html=True,
    )


def evidence_table(payload: dict[str, Any]) -> None:
    rows = [{
        "Stage": f"Stage {node['stage_number']}", "Candidate": node["label"], "Primary": node["primary"],
        "GAUC": node["gauc"], "nDCG@5": node["ndcg"], "Champion": "Yes" if node["is_champion_path"] else "No",
    } for node in payload["search"]["top_candidates"][:6]]
    st.dataframe(rows, width="stretch", hide_index=True, height=255, column_config={
        "Primary": st.column_config.NumberColumn(format="%.6f"), "GAUC": st.column_config.NumberColumn(format="%.6f"),
        "nDCG@5": st.column_config.NumberColumn(format="%.6f"),
    })


def main() -> None:
    inject_css()
    try:
        payload = _load(str(_manifest_path()))
    except ShowcaseBuildError as exc:
        st.error(str(exc))
        st.code(
            "python dashboard/build_showcase.py --data-root /root/autodl-tmp/ByteRush\n"
            "python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 18501 --server.fileWatcherType none",
            language="bash",
        )
        return

    navigation()
    story = payload.get("story") or {}

    with st.container(key="slide-01-hero"):
        hero(payload)
        slide_index("01", "Opening")

    with st.container(key="slide-02-impact"):
        section_head("impact", "Measured impact", "The gain that matters.", "All three values come from the same five-seed frozen model under the protected validation protocol. Larger text and explicit deltas make the improvement immediately readable.")
        left, right = st.columns([1.55, .85], vertical_alignment="center")
        with left:
            st.plotly_chart(metric_comparison(payload), width="stretch", key="metric-comparison", config={"displaylogo": False})
            st.markdown("<div class='axis-note'>Focused y-axis reveals small but meaningful ranking improvements; exact values are printed above every bar.</div>", unsafe_allow_html=True)
        with right:
            st.markdown(
                f"<div class='impact-copy'><h3>Why this is credible</h3><p>{_safe(story.get('problem'))}</p><p>{_safe(story.get('solution'))}</p>"
                f"<div class='baseline-chip'>FM Primary {_metric(payload['winner']['baseline']['primary'])} → {_metric(payload['winner']['final']['primary'])}</div></div>",
                unsafe_allow_html=True,
            )
        slide_index("02", "Results")

    with st.container(key="slide-03-agent"):
        section_head("agent", "Autonomous research loop", "Four stages. One evidence trail.", "The Agent broadens the search space, concentrates budget on promising roots, challenges the incumbent, and only freezes a model after stability verification.")
        stage_cards(payload)
        slide_index("03", "Agent loop")

    with st.container(key="slide-04-search"):
        section_head("search", "Search provenance", "Watch the evidence expand.", "Switch between the exact champion ancestry, the strongest supporting candidates, and the full search. Colored icons mark the meaningful start, promotion, and frozen end states.")
        mode = st.segmented_control("Tree detail", ["Champion path", "Curated evidence", "Full search"], default="Champion path", label_visibility="collapsed") or "Champion path"
        tree_key = mode.lower().replace(" ", "-")
        with st.container(key=f"tree-frame-{tree_key}"):
            st.plotly_chart(search_tree(payload, mode), width="stretch", key="search-tree-canvas", config={"displaylogo": False, "scrollZoom": False})
        stats = payload["search"]
        st.markdown(
            f"<div class='search-stat-row'><div class='search-stat'><b>{stats['total_nodes']}</b><span>generated nodes</span></div>"
            f"<div class='search-stat'><b>{stats['successful_nodes']}</b><span>successful experiments</span></div>"
            f"<div class='search-stat'><b>{stats['failed_nodes']}</b><span>failed safely</span></div>"
            f"<div class='search-stat'><b>{stats['research_rounds']}</b><span>research rounds</span></div></div>",
            unsafe_allow_html=True,
        )
        slide_index("04", "Search tree")

    with st.container(key="slide-05-selection"):
        section_head("selection", "Champion decision", f"How the Agent selected {payload['winner']['label']}.", "This is the actual promotion logic: establish FM, compare model families, tune the best root, test creative challengers, then require multi-seed confirmation before freezing.")
        selection_story(payload)
        slide_index("05", "Selection logic")

    with st.container(key="slide-06-stability"):
        section_head("stability", "Final verification", "Stable across independent seeds.", "The champion was not frozen after one lucky run. Stage 4 repeated training and compared GAUC, nDCG@5, and Primary together.")
        st.plotly_chart(seed_figure(payload), width="stretch", key="seed-stability", config={"displaylogo": False})
        stability = st.columns(3)
        for column, name in zip(stability, ("GAUC", "nDCG@5", "primary")):
            metric = payload["winner"]["metrics"][name]
            column.metric("Primary" if name == "primary" else name, _metric(metric["mean"]), f"σ {_metric(metric['std'])}")
        slide_index("06", "Verification")

    with st.container(key="slide-07-evidence"):
        section_head("evidence", "Audit trail", "A result the judges can inspect.", "The narrative stays concise, while every score remains traceable to a run, source node, frozen model, checkpoint, and submission hash.")
        integrity = payload["integrity"]
        st.markdown(
            f"<div class='proof-grid'><div class='proof'><b>Validation-only</b><span>No test metric entered model selection.</span></div>"
            f"<div class='proof'><b>{integrity['successful_seed_count']} verified seeds</b><span>Stability checked before freezing.</span></div>"
            f"<div class='proof'><b>{integrity['submission_rows']:,} predictions</b><span>Submission ready; hidden labels untouched.</span></div>"
            f"<div class='proof'><b>SHA-256 evidence</b><span>Model, checkpoint, and CSV are fingerprinted.</span></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("### Leading validated candidates")
        evidence_table(payload)
        with st.expander("Reproducibility paths and hashes"):
            st.json({"run": payload["selection"], "files": payload["files"], "integrity": payload["integrity"]}, expanded=True)
        st.markdown(
            f"<div class='footer'>Frozen showcase · run {_safe(payload['selection']['run_id'])} · source node {_safe(payload['selection']['source_node_id'][:12])} · validation-only metrics</div>",
            unsafe_allow_html=True,
        )
        slide_index("07", "Evidence")


if __name__ == "__main__":
    main()
