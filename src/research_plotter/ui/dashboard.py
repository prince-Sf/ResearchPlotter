"""Evidence-led dashboard built from local ResearchPlotter artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from typing import Any

import streamlit as st


_METRICS = (
    ("Workbench experiments", "⚗", "experiment", "Experiment History", "green", "Saved local experiment records"),
    ("Generated figures", "▧", "figure", "Figure Gallery", "blue", "PNG figures available in outputs/"),
    ("Paper analyses", "▤", "analysis", "Paper Analyzer", "amber", "Machine-readable paper analyses"),
    ("Reproduction reports", "▤", "reproduction", "Reports / Export", "violet", "Paper reproduction reports"),
)

_DEPENDENCIES = (
    ("NumPy", "numpy", True),
    ("SciPy", "scipy", True),
    ("Pandas", "pandas", True),
    ("Matplotlib", "matplotlib", True),
    ("SymPy", "sympy", True),
    ("PyTorch", "torch", False),
    ("Streamlit", "streamlit", True),
    ("PyMuPDF", "fitz", True),
    ("PDF reader", "pypdf", True),
    ("Excel readers", ("openpyxl", "xlrd"), False),
    ("Parquet", "pyarrow", False),
)


@st.cache_data(ttl=10, show_spinner=False)
def _read_records(workbench: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not workbench.exists():
        return records
    for path in workbench.rglob("experiment.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        item["_record_path"] = str(path)
        item["_created_sort"] = str(item.get("created_at") or "")
        records.append(item)
    return sorted(records, key=lambda item: item["_created_sort"], reverse=True)


def _dependency_available(module: str | tuple[str, ...]) -> bool:
    names = (module,) if isinstance(module, str) else module
    try:
        return all(importlib.util.find_spec(name) is not None for name in names)
    except (ImportError, ValueError, AttributeError):
        return False


@st.cache_data(ttl=10, show_spinner=False)
def _counts(outputs: Path, records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "experiment": len(records),
        "figure": sum(1 for path in outputs.rglob("*.png") if path.is_file()) if outputs.exists() else 0,
        "analysis": sum(1 for path in outputs.rglob("paper_analysis.json") if path.is_file()) if outputs.exists() else 0,
        "reproduction": sum(1 for path in outputs.rglob("reproduction_report.json") if path.is_file()) if outputs.exists() else 0,
    }


def _navigate(destination: str) -> None:
    import streamlit as st

    st.session_state["main_navigation"] = destination


def _open_history_record(experiment_id: str) -> None:
    import streamlit as st

    st.session_state["history_selected"] = experiment_id
    _navigate("Experiment History")


def render_dashboard(st: Any, *, root: Path, outputs: Path, workbench: Path) -> None:
    records = _read_records(workbench)
    counts = _counts(outputs, records)
    dark = bool(st.session_state.get("rp_theme_dark", False))
    line = "#39E6BB" if dark else "#176C45"
    faint = "rgba(57,230,187,.10)" if dark else "rgba(23,108,69,.08)"
    st.markdown(
        f"""<section class="rp-hero">
          <svg viewBox="0 0 450 145" role="img" aria-label="Decorative scientific trajectory illustration">
            <path d="M0 115 C62 92 64 38 124 56 S200 138 255 81 S335 17 390 53 S430 90 450 36" fill="none" stroke="{line}" stroke-width="2" opacity=".42"/>
            <path d="M0 132 C75 106 83 75 145 86 S211 117 270 64 S350 34 395 77 S430 105 450 69" fill="none" stroke="{line}" stroke-width="1.5" stroke-dasharray="5 5" opacity=".57"/>
            <path d="M0 120 C90 89 107 110 169 94 S236 46 294 72 S372 119 450 84 L450 145 L0 145 Z" fill="{faint}"/>
            <g fill="{line}"><circle cx="64" cy="72" r="3"/><circle cx="145" cy="86" r="3"/><circle cx="255" cy="81" r="3"/><circle cx="342" cy="37" r="3"/><circle cx="414" cy="91" r="3"/></g>
          </svg>
          <div class="rp-hello">Welcome back,</div>
          <h1>Research<span>Plotter</span></h1>
          <p>A scientific workbench for data, models, simulations, paper analysis, and reproducible figures.</p>
        </section>""",
        unsafe_allow_html=True,
    )

    columns = st.columns(4, gap="medium")
    for column, (title, icon, key, destination, color, note) in zip(columns, _METRICS):
        with column.container(border=True):
            st.markdown(f'<div class="rp-metric-icon">{icon}</div><div class="rp-metric-title">{title}</div><div class="rp-metric-number">{counts[key]:,}</div><div class="rp-metric-note">{note}</div>', unsafe_allow_html=True)
            st.button("Open", on_click=_navigate, args=(destination,), key=f"dashboard_metric_{key}")

    st.subheader("Dependency status")
    st.caption("Availability reflects this Python environment; it is not a claim that every optional experiment has been validated.")
    with st.container(border=True):
        available_count = sum(_dependency_available(module) for _, module, _ in _DEPENDENCIES)
        if available_count == len(_DEPENDENCIES):
            st.success("All listed integrations are installed in this environment.")
        else:
            st.warning(f"{available_count} of {len(_DEPENDENCIES)} listed integrations are available.")
        dep_columns = st.columns(3)
        for index, (label, module, required) in enumerate(_DEPENDENCIES):
            available = _dependency_available(module)
            state = "available" if available else ("missing · required" if required else "optional · not installed")
            style = "rp-status-ok" if available else ("rp-status-bad" if required else "rp-status-warn")
            dep_columns[index % 3].markdown(f"●　**{label}**　<span class=\"{style}\">{state}</span>", unsafe_allow_html=True)

    lower = st.columns([1.55, 1.0], gap="medium")
    with lower[0].container(border=True):
        header = st.columns([4, 1], vertical_alignment="center")
        header[0].markdown("### ◷ Recent experiments")
        header[1].button("View all →", on_click=_navigate, args=("Experiment History",), key="dashboard_view_history")
        st.caption("Latest local workbench runs. Completion status is separate from scientific reproducibility status.")
        if not records:
            st.info("No workbench runs yet. Try Dataset Analyzer with the bundled measurements example.")
        else:
            rows = []
            for item in records[:5]:
                created = item.get("created_at")
                try:
                    created = datetime.fromisoformat(str(created).replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                except (TypeError, ValueError):
                    created = "—"
                details = item.get("details", {})
                rows.append({
                    "Experiment": item.get("experiment_id", "unknown"),
                    "Workflow": item.get("workflow", "unknown"),
                    "Status": item.get("status", "UNKNOWN"),
                    "Created": created,
                    "Scientific status": details.get("status", details.get("reproducibility_status", "not assessed")),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True, height=242)
            recent_records = records[:5]
            recent_ids = [str(item.get("experiment_id", "unknown")) for item in recent_records]
            selected_id = st.selectbox("Experiment details", recent_ids, key="dashboard_recent_selected")
            st.button("Open selected experiment", on_click=_open_history_record, args=(selected_id,), key="dashboard_open_experiment")

    with lower[1].container(border=True):
        st.markdown("### ⚡ Quick Start")
        st.caption("Open an existing workflow to get started.")
        shortcuts = (
            ("ƒx　Plot an Equation", "Equation / Formula Plotter", "Visualize a safe mathematical expression"),
            ("▤　Analyze a Dataset", "Dataset Analyzer", "Explore and plot scientific data"),
            ("▧　Reproduce a Figure", "Figure Reproduction", "Review evidence and run supported cases"),
            ("◇　Simulate a Model", "Model Simulation", "Configure and run a model"),
        )
        for index in range(0, len(shortcuts), 2):
            pair = st.columns(2, gap="small")
            for column, (label, page, detail) in zip(pair, shortcuts[index:index + 2]):
                with column:
                    st.button(f"{label}\n{detail}", on_click=_navigate, args=(page,), key=f"dashboard_quick_{index}_{page}", use_container_width=True)
        st.caption("Empirical graph fits are candidates, not evidence that a paper used the same equation.")

