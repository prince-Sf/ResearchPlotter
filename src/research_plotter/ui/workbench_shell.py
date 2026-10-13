"""Shared visual shell for the Streamlit Research Workbench.

This module intentionally contains presentation and local navigation only. It
does not implement scientific analysis, data loading, or experiment logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


THEME_KEY = "rp_theme_dark"

_PAGE_ICONS = {
    "Dashboard": "⌂",
    "Dataset Analyzer": "▤",
    "Paper Analyzer": "▧",
    "Figure Reproduction": "▨",
    "Equation / Formula Plotter": "ƒx",
    "Graph Image Analyzer": "⌕",
    "Model Simulation": "◇",
    "Parameter Sweep": "☷",
    "Sensitivity Analysis": "▥",
    "Noise / Uncertainty": "∿",
    "Convergence / Validation": "◉",
    "PINN Experiments": "♧",
    "Figure Gallery": "▧",
    "Experiment History": "◷",
    "Reports / Export": "▤",
}


def render_theme_styles(st: Any) -> None:
    """Apply one shared light/dark palette to Streamlit and native widgets."""
    dark = bool(st.session_state.get(THEME_KEY, False))
    if dark:
        palette = {
            "bg": "#08111B", "surface": "#101D29", "elevated": "#142333",
            "sidebar": "#0B151D", "text": "#F2F6FA", "muted": "#A8B8C8",
            "accent": "#39E6BB", "accent_soft": "#12352F", "border": "#203443",
            "shadow": "0 12px 30px rgba(0,0,0,.20)", "input": "#0D1924",
            "sidebar_muted": "#99ABB8", "active": "rgba(57,230,187,.13)",
        }
        theme_name = "dark"
    else:
        palette = {
            "bg": "#F7F6F1", "surface": "#FFFFFF", "elevated": "#FCFCFA",
            "sidebar": "#121B18", "text": "#17202C", "muted": "#667085",
            "accent": "#176C45", "accent_soft": "#DDF3E4", "border": "#E6E8E3",
            "shadow": "0 10px 26px rgba(29,48,38,.055)", "input": "#FFFFFF",
            "sidebar_muted": "#B2C0B7", "active": "rgba(96,147,83,.28)",
        }
        theme_name = "light"
    variables = ";".join(f"--rp-{name}:{value}" for name, value in palette.items())
    st.markdown(
        f"""<style>
        :root {{{variables};color-scheme:{theme_name};}}
        html, body, .stApp, [data-testid="stAppViewContainer"] {{
            background:var(--rp-bg); color:var(--rp-text);
            font-family:Inter,"Segoe UI",system-ui,-apple-system,sans-serif;
        }}
        [data-testid="stAppViewContainer"] > .main {{background:var(--rp-bg);}}
        [data-testid="stHeader"] {{background:var(--rp-bg); border-bottom:1px solid var(--rp-border);}}
        [data-testid="stToolbar"] {{right:1rem;}}
        div.block-container {{max-width:1540px; padding-top:1.25rem; padding-bottom:3rem;}}
        section[data-testid="stSidebar"] {{
            background:var(--rp-sidebar); border-right:1px solid var(--rp-border);
            min-width:18.5rem !important; max-width:18.5rem !important;
        }}
        section[data-testid="stSidebar"] > div {{background:var(--rp-sidebar);}}
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{color:#EAF1EC;}}
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{color:var(--rp-sidebar-muted);}}
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {{
            border:1px solid transparent; border-radius:11px; padding:.35rem .5rem;
            color:#E8EFEA; transition:background .12s ease,border-color .12s ease;
        }}
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{background:rgba(255,255,255,.075);}}
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
            background:var(--rp-active); border-color:color-mix(in srgb,var(--rp-accent) 35%,transparent);
            box-shadow:inset 3px 0 0 var(--rp-accent); color:#F4FFF8;
        }}
        [data-testid="stMarkdownContainer"] h1 {{color:var(--rp-text); letter-spacing:-.035em; font-weight:760;}}
        [data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3 {{color:var(--rp-text); letter-spacing:-.02em;}}
        [data-testid="stCaptionContainer"], small {{color:var(--rp-muted);}}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background:var(--rp-surface); border-color:var(--rp-border) !important;
            border-radius:16px; box-shadow:var(--rp-shadow);
        }}
        [data-testid="stMetric"] {{background:transparent;}}
        [data-testid="stMetricLabel"] {{color:var(--rp-muted);}}
        [data-testid="stMetricValue"] {{color:var(--rp-text); font-weight:720;}}
        [data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea,[data-testid="stSelectbox"] [role="combobox"] {{
            color:var(--rp-text); background:var(--rp-input); border-color:var(--rp-border); border-radius:9px;
        }}
        [data-testid="stFileUploader"] section {{background:var(--rp-surface); border-color:var(--rp-border); border-radius:12px;}}
        [data-testid="stExpander"] {{background:var(--rp-surface); border-color:var(--rp-border); border-radius:12px;}}
        [data-testid="stTabs"] [role="tab"] {{color:var(--rp-muted);}}
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{color:var(--rp-accent);}}
        [data-testid="stDataFrame"],[data-testid="stTable"] {{border:1px solid var(--rp-border); border-radius:10px; overflow:hidden;}}
        [data-testid="stAlert"] {{border-radius:11px;}}
        div.stButton > button {{border-radius:10px; border-color:var(--rp-border); min-height:2.55rem;}}
        div.stButton > button[kind="primary"] {{background:var(--rp-accent); color:#fff; border-color:var(--rp-accent);}}
        div.stButton > button:hover {{border-color:var(--rp-accent); color:var(--rp-accent);}}
        div.stButton > button[kind="primary"]:hover {{background:color-mix(in srgb,var(--rp-accent) 84%,black); color:#fff;}}
        .rp-brand {{display:flex;gap:.7rem;align-items:center;padding:.45rem .2rem 1.15rem;}}
        .rp-brand strong {{font-size:1.22rem;letter-spacing:-.035em;color:#F4F7F4;}}
        .rp-brand strong span {{color:var(--rp-accent);}}
        .rp-brand small {{display:block;color:var(--rp-sidebar-muted);font-size:.66rem;margin-top:.15rem;white-space:nowrap;}}
        .rp-sidebar-section {{font-size:.69rem;letter-spacing:.09em;color:var(--rp-sidebar-muted);padding:.25rem .45rem .5rem;font-weight:650;}}
        .rp-sidebar-note {{margin-top:1rem;padding:.8rem;border:1px solid rgba(180,210,190,.17);border-radius:13px;background:rgba(255,255,255,.045);color:#F1F6F2;font-size:.84rem;}}
        .rp-sidebar-note small {{display:block;color:var(--rp-sidebar-muted);margin-top:.2rem;}}
        .rp-topline {{border-bottom:1px solid var(--rp-border);padding-bottom:.4rem;margin-bottom:1.1rem;}}
        .rp-local-workspace {{color:var(--rp-muted);font-size:.82rem;text-align:right;padding-top:.72rem;white-space:nowrap;}}
        .rp-hero {{position:relative;overflow:hidden;min-height:130px;padding:1rem .4rem 1.1rem;}}
        .rp-hero .rp-hello {{color:var(--rp-muted);font-size:1rem;margin-bottom:.1rem;}}
        .rp-hero h1 {{font-size:clamp(2.25rem,4vw,3.35rem);margin:.1rem 0 .2rem;line-height:1.05;}}
        .rp-hero h1 span {{color:var(--rp-accent);}}
        .rp-hero p {{color:var(--rp-muted);font-size:1rem;max-width:780px;margin:0;}}
        .rp-hero svg {{position:absolute;right:1%;top:0;width:min(38%,420px);height:100%;opacity:.78;pointer-events:none;}}
        .rp-kicker {{font-size:.71rem;letter-spacing:.1em;color:var(--rp-muted);font-weight:650;text-transform:uppercase;}}
        .rp-metric-icon {{font-size:1.35rem;line-height:1.1;color:var(--rp-accent);}}
        .rp-metric-title {{font-size:.82rem;color:var(--rp-muted);margin:.25rem 0;}}
        .rp-metric-number {{font-size:1.8rem;line-height:1.1;color:var(--rp-text);font-weight:730;letter-spacing:-.035em;}}
        .rp-metric-note {{font-size:.73rem;color:var(--rp-muted);margin-top:.35rem;}}
        .rp-card-accent-green {{border-top:3px solid #15945A;}}
        .rp-card-accent-blue {{border-top:3px solid #3576D3;}}
        .rp-card-accent-amber {{border-top:3px solid #D88A23;}}
        .rp-card-accent-violet {{border-top:3px solid #7657C7;}}
        .rp-status-ok {{color:#16A05C;font-weight:650;}}
        .rp-status-warn {{color:#D88921;font-weight:650;}}
        .rp-status-bad {{color:#D74747;font-weight:650;}}
        .rp-quick-title {{font-weight:680;color:var(--rp-text);}}
        @media (max-width:900px) {{
            section[data-testid="stSidebar"] {{min-width:15rem !important;max-width:15rem !important;}}
            .rp-hero svg {{opacity:.3;width:44%;}}
        }}
        @media (max-width:650px) {{
            div.block-container {{padding-left:1rem;padding-right:1rem;}}
            .rp-hero svg {{display:none;}}
            .rp-hero {{min-height:0;}}
        }}
        </style>""",
        unsafe_allow_html=True,
    )


def render_sidebar(st: Any, navigation: list[str]) -> str:
    """Render persistent branded navigation while keeping stable page values."""
    st.markdown(
        """<div class="rp-brand" aria-label="ResearchPlotter scientific workbench">
          <svg width="48" height="42" viewBox="0 0 48 42" aria-hidden="true">
            <path d="M4 34 L14 19 L22 26 L34 9 L43 15" fill="none" stroke="#39E6BB" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M4 37 H44" stroke="#39E6BB" stroke-width="1.5" opacity=".62"/>
            <circle cx="4" cy="34" r="2.7" fill="#8BF3D9"/><circle cx="14" cy="19" r="2.7" fill="#8BF3D9"/>
            <circle cx="22" cy="26" r="2.7" fill="#8BF3D9"/><circle cx="34" cy="9" r="2.7" fill="#8BF3D9"/>
            <circle cx="43" cy="15" r="2.7" fill="#8BF3D9"/>
          </svg><div><strong>Research<span>Plotter</span></strong><small>Scientific modeling · analysis · visualization</small></div>
        </div><div class="rp-sidebar-section">WORKBENCH</div>""",
        unsafe_allow_html=True,
    )
    selected = st.radio(
        "Workbench navigation",
        navigation,
        key="main_navigation",
        label_visibility="collapsed",
        format_func=lambda page: f"{_PAGE_ICONS.get(page, '•')}　{page}",
    )
    st.markdown(
        '<div class="rp-sidebar-note">Evidence-led research<small>Reproducible scientific experiments.</small></div>',
        unsafe_allow_html=True,
    )
    return selected


@st.cache_data(ttl=10, show_spinner=False)
def _search_index(root: Path, outputs: Path) -> list[dict[str, str]]:
    """Collect searchable local entities without opening arbitrary files."""
    results: list[dict[str, str]] = []

    def add_file(path: Path, kind: str, page: str) -> None:
        try:
            relative = str(path.resolve().relative_to(root.resolve()))
        except (OSError, ValueError):
            relative = path.name
        results.append({"kind": kind, "label": path.name, "path": str(path), "page": page, "search": f"{kind} {path.name} {relative}"})

    for folder in (root / "data" / "examples", outputs / "uploaded" / "datasets"):
        if folder.exists():
            for path in folder.iterdir():
                if path.is_file() and path.suffix.lower() in {".csv", ".txt", ".xlsx", ".xls", ".json", ".parquet"}:
                    add_file(path, "Dataset", "Dataset Analyzer")
    for folder in (root / "Papers", root / "papers", outputs / "uploaded" / "papers"):
        if folder.exists():
            for path in folder.glob("*.pdf"):
                if path.is_file():
                    add_file(path, "Research paper", "Paper Analyzer")
    if outputs.exists():
        for path in outputs.rglob("experiment.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            results.append({
                "kind": "Experiment",
                "label": str(payload.get("experiment_id") or path.parent.name),
                "path": str(path),
                "page": "Experiment History",
                "id": str(payload.get("experiment_id") or path.parent.name),
                "search": " ".join(str(payload.get(key, "")) for key in ("experiment_id", "workflow", "status")),
            })
        figure_paths = sorted((path for path in outputs.rglob("*.png") if path.is_file()), key=lambda path: path.stat().st_mtime, reverse=True)[:300]
        for path in figure_paths:
            add_file(path, "Figure", "Figure Gallery")
        for folder in (outputs / "reports", outputs / "paper_reproduction"):
            if folder.exists():
                report_paths = sorted((path for path in folder.rglob("*.json") if path.is_file() and path.name != "experiment.json"), key=lambda path: path.stat().st_mtime, reverse=True)
                for path in report_paths[:500]:
                    add_file(path, "Report", "Reports / Export")
    return results


def _open_selected_search_result() -> None:
    """Apply the selected result as session state before the next app render."""
    import streamlit as st

    label = st.session_state.get("rp_search_selected")
    result = st.session_state.get("rp_search_result_map", {}).get(label)
    if not result:
        return
    st.session_state["main_navigation"] = result["page"]
    kind = result["kind"]
    path = Path(result["path"])
    if kind == "Experiment":
        st.session_state["history_selected"] = result["id"]
    elif kind == "Dataset":
        st.session_state["dataset_open_path"] = str(path)
        st.session_state["dataset_source"] = "Open selected local dataset"
    elif kind == "Research paper":
        st.session_state["paper_open_path"] = str(path)
        st.session_state["paper_source_kind"] = "Open selected local paper"
    elif kind == "Figure":
        st.session_state["gallery_figure"] = path
    elif kind == "Report":
        st.session_state["report_selected"] = path


@st.cache_data(ttl=10, show_spinner=False)
def _attention_count(workbench: Path) -> int:
    count = 0
    if workbench.exists():
        for path in workbench.rglob("experiment.json"):
            try:
                count += json.loads(path.read_text(encoding="utf-8")).get("status") in {"FAILED", "BLOCKED", "NEEDS_REVIEW", "PARTIAL"}
            except (OSError, json.JSONDecodeError):
                continue
    return count


def render_top_navigation(st: Any, *, root: Path, outputs: Path) -> None:
    """Render a functional local search, persistent theme toggle, and status."""
    columns = st.columns([7.6, 1.25, 1.4, 1.7], vertical_alignment="center")
    with columns[0]:
        query = st.text_input(
            "Search local workspace",
            placeholder="Search experiments, datasets, papers, figures, reports…",
            key="rp_global_search",
            label_visibility="collapsed",
        ).strip()
    with columns[1]:
        st.toggle("Dark mode", key=THEME_KEY, help="Switch the complete workbench between light and dark themes.")
    with columns[2]:
        attention_count = _attention_count(outputs / "reports" / "workbench")
        if attention_count:
            st.caption(f"● {attention_count} need review")
        else:
            st.caption("● Workspace ready")
    with columns[3]:
        st.markdown('<div class="rp-local-workspace">Local workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="rp-topline"></div>', unsafe_allow_html=True)

    if not query:
        return
    lowered = query.casefold()
    matches = [item for item in _search_index(root, outputs) if lowered in item["search"].casefold()]
    matches = matches[:30]
    if not matches:
        st.caption("No matching local experiments, datasets, papers, figures, or reports were found.")
        return
    labels = [f"{item['kind']} · {item['label']}" for item in matches]
    result_map = dict(zip(labels, matches))
    st.session_state["rp_search_result_map"] = result_map
    if st.session_state.get("rp_search_selected") not in result_map:
        st.session_state["rp_search_selected"] = labels[0]
    result_cols = st.columns([5, 1.4])
    result_cols[0].selectbox("Matching local items", labels, key="rp_search_selected", label_visibility="collapsed")
    result_cols[1].button("Open result", on_click=_open_selected_search_result, type="primary", key="rp_search_open")

