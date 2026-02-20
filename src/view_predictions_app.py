from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.main, .main > div,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] > div,
.stApp {
    background-color: #f7f8fa !important;
    color: #1a1f2e !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background-color: #eef0f4 !important;
    border-right: 1px solid #d8dce6 !important;
}
[data-testid="stSidebar"] * { color: #1a1f2e !important; }
h1 {
    font-size: clamp(1.2rem, 2vw, 1.7rem) !important;
    font-weight: 600 !important;
    color: #1a1f2e !important;
    letter-spacing: .5px;
    border-bottom: 2px solid #e2e5ec;
    padding-bottom: .4rem;
}
h2, h3 {
    color: #1a1f2e !important;
    font-weight: 600 !important;
    font-size: clamp(.8rem, 1.1vw, 1rem) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
p, span, label, div { color: #1a1f2e !important; }
.stCaption, small { color: #9ca3af !important; font-size: .75rem !important; }
.stButton > button {
    background: #ffffff !important;
    color: #1a1f2e !important;
    border: 1px solid #c8cdd8 !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: clamp(.7rem, .9vw, .85rem) !important;
    padding: 5px 16px !important;
    transition: border-color .15s, box-shadow .15s;
}
.stButton > button:hover {
    border-color: #4a6cf7 !important;
    box-shadow: 0 0 0 2px rgba(74,108,247,.15) !important;
}
.stSelectbox label {
    font-size: clamp(.68rem, .85vw, .78rem) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6b7280 !important;
}
.stSelectbox > div > div,
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #c8cdd8 !important;
    border-radius: 4px !important;
    color: #1a1f2e !important;
}
[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background-color: #ffffff !important;
    border: 1px solid #d8dce6 !important;
}
[data-baseweb="popover"] li,
[data-baseweb="menu"] li {
    background-color: #ffffff !important;
    color: #1a1f2e !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover { background-color: #eef0f4 !important; }
[data-testid="stTable"] { background-color: #ffffff !important; }
[data-testid="stTable"] table {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: clamp(.68rem, .85vw, .78rem) !important;
    border-collapse: collapse !important;
    width: 100%;
    background-color: #ffffff !important;
}
[data-testid="stTable"] th {
    background-color: #eef0f4 !important;
    color: #6b7280 !important;
    border-bottom: 2px solid #d8dce6 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 6px 10px !important;
    white-space: nowrap;
}
[data-testid="stTable"] td {
    border-bottom: 1px solid #e8ebf0 !important;
    padding: 5px 10px !important;
    color: #1a1f2e !important;
    background-color: #ffffff !important;
}
[data-testid="stDataFrame"],
.stDataFrame iframe,
.dataframe-container {
    background-color: #ffffff !important;
    border: 1px solid #e2e5ec !important;
    border-radius: 4px;
}
[data-testid="stRadio"] label { color: #1a1f2e !important; }
[data-testid="stRadio"] { background: transparent !important; }
hr { border-color: #e2e5ec !important; }
[data-testid="stAlert"] {
    background-color: #eff6ff !important;
    color: #1a1f2e !important;
    border: 1px solid #bfdbfe !important;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f7f8fa; }
::-webkit-scrollbar-thumb { background: #c8cdd8; border-radius: 3px; }
@media (max-width: 1400px) {
    [data-testid="stHorizontalBlock"] { gap: .5rem !important; }
    .stButton > button { padding: 4px 10px !important; font-size: .7rem !important; }
}
</style>
"""

RISK_CFG = {
    "low":     {"bg": "#f0fdf4", "border": "#86efac", "text": "#16a34a", "bar": "#22c55e", "icon": "▼", "hu": "ALACSONY"},
    "medium":  {"bg": "#fffbeb", "border": "#fcd34d", "text": "#d97706", "bar": "#f59e0b", "icon": "◆", "hu": "KÖZEPES"},
    "high":    {"bg": "#fff1f2", "border": "#fca5a5", "text": "#dc2626", "bar": "#ef4444", "icon": "▲", "hu": "MAGAS"},
    "unknown": {"bg": "#f7f8fa", "border": "#c8cdd8", "text": "#6b7280", "bar": "#c8cdd8", "icon": "●", "hu": "ISMERETLEN"},
}


def risk_banner_html(risk_label: str, p_aggregate: float | None) -> str:
    label = str(risk_label).strip().lower()
    c = RISK_CFG.get(label, RISK_CFG["unknown"])
    pct_bar = ""
    if p_aggregate is not None:
        pct = round(p_aggregate * 100, 1)
        pct_bar = (
            '<div style="margin-top:12px;">'
            '<div style="display:flex;justify-content:space-between;'
            'font-size:.7rem;color:#6b7280;font-family:\'IBM Plex Mono\',monospace;margin-bottom:4px;">'
            '<span>P_AGGREGATE</span>'
            f'<span>{pct}%</span>'
            '</div>'
            '<div style="background:#e8ebf0;border-radius:4px;height:6px;overflow:hidden;">'
            f'<div style="width:{pct}%;height:100%;background:{c["bar"]};border-radius:4px;"></div>'
            '</div>'
            '</div>'
        )
    pulse = ""
    if label == "high":
        pulse = (
            "<style>@keyframes soft-pulse{"
            "0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.2)}"
            "50%{box-shadow:0 0 0 5px rgba(239,68,68,.0)}}"
            "#risk-card{animation:soft-pulse 2s ease-in-out infinite;}</style>"
        )
    return (
        f"{pulse}"
        '<div id="risk-card" style="'
        f'background:{c["bg"]};border:1.5px solid {c["border"]};border-radius:8px;'
        'padding:14px 18px;margin-bottom:10px;">'
        '<div style="font-size:.65rem;font-weight:600;text-transform:uppercase;'
        'letter-spacing:2px;color:#9ca3af;font-family:\'IBM Plex Sans\',sans-serif;margin-bottom:3px;">'
        'Összesített kockázat'
        '</div>'
        '<div style="display:flex;align-items:center;gap:10px;">'
        f'<span style="font-size:1.7rem;color:{c["text"]};">{c["icon"]}</span>'
        f'<span style="font-size:1.9rem;font-weight:700;color:{c["text"]};'
        'font-family:\'IBM Plex Sans\',sans-serif;letter-spacing:2px;">'
        f'{c["hu"]}'
        '</span>'
        '</div>'
        f'{pct_bar}'
        '</div>'
    )


def status_pill(label: str, color: str, icon: str) -> str:
    return (
        f'<div style="display:inline-flex;align-items:center;gap:5px;'
        f'border:1px solid {color};border-radius:20px;padding:3px 10px;'
        f'margin:2px 2px 2px 0;font-size:.72rem;color:{color};'
        f'background-color:#ffffff;'
        f'font-family:\'IBM Plex Sans\',sans-serif;font-weight:600;">'
        f'{icon} {label}</div>'
    )


@st.cache_data(show_spinner=True)
def load_merged_frame(mode: str) -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    if mode == "maintenance":
        preds_path = base_dir / "predictions_maint.csv"
        if not preds_path.exists():
            raise FileNotFoundError(
                f"{preds_path} not found. Futtasd: 'py -m src.predict_demo --maintenance --ultra-coverage'"
            )
    else:
        preds_path = base_dir / "predictions.csv"
        if not preds_path.exists():
            raise FileNotFoundError(
                f"{preds_path} not found. Futtasd: 'py -m src.predict_demo --ultra-coverage'"
            )
    preds = pd.read_csv(preds_path, index_col="Date", parse_dates=True)
    return preds.sort_index()


def find_nearest_index(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    if len(idx) == 0:
        return None
    pos = idx.get_indexer([ts], method="nearest")[0]
    return None if pos == -1 else idx[pos]


def _sync_dropdown_keys(mode_key: str, current_ts: pd.Timestamp,
                        date_labels: list, label_date_fn,
                        df: pd.DataFrame) -> bool:
    """
    Write the correct values into the selectbox session-state keys
    ONLY when no widget with those keys has been created yet in this run.
    Returns True if keys were changed and a rerun is needed.
    """
    date_key = f"date_sel_{mode_key}"
    time_key = f"time_sel_{mode_key}"

    want_date = label_date_fn(current_ts.date())
    tfd = sorted({ts.time() for ts in df.index if ts.date() == current_ts.date()})
    want_time = current_ts.time() if current_ts.time() in tfd else (tfd[0] if tfd else None)

    changed = False
    if st.session_state.get(date_key) != want_date:
        st.session_state[date_key] = want_date
        changed = True
    if st.session_state.get(time_key) != want_time:
        st.session_state[time_key] = want_time
        changed = True
    return changed


def main() -> None:
    st.set_page_config(page_title="Powerszerviz előrejelzés", layout="wide", page_icon="⚡")
    st.markdown(CSS, unsafe_allow_html=True)

    mode = st.sidebar.radio("Mód", ["Hiba", "Karbantartás"], index=0)
    is_maint = mode == "Karbantartás"
    mode_key = "maintenance" if is_maint else "fault"

    st.title("⚡ Powerszerviz – " + ("Karbantartás" if is_maint else "Hiba") + " Előrejelzés")
    st.caption(
        "Az adatok csak a motor működése (vagy pufferperiódus) alatt keletkeznek — "
        "a többi időszakban nincsenek sorok, így az idővonalban hézagok lehetnek."
    )

    df = load_merged_frame(mode_key)

    # ── Session state init ────────────────────────────────────────────────────
    if "current_ts" not in st.session_state:
        st.session_state["current_ts"] = df.index[0]
    if "last_mode" not in st.session_state:
        st.session_state["last_mode"] = mode_key
    if st.session_state["last_mode"] != mode_key:
        st.session_state["current_ts"] = df.index[0]
        st.session_state["last_mode"] = mode_key
        # Clear stale widget keys from old mode
        for k in [f"date_sel_fault", f"date_sel_maintenance",
                  f"time_sel_fault", f"time_sel_maintenance"]:
            st.session_state.pop(k, None)

    # ── Event days ────────────────────────────────────────────────────────────
    event_col_name = "maintenance_event" if is_maint else "fault_event"
    if event_col_name in df.columns:
        event_days = set(df.index[df[event_col_name].astype(bool)].date)
    else:
        event_days = set()

    available_dates = sorted({ts.date() for ts in df.index})

    def label_date(d):
        return f"🔴 {d}" if d in event_days else str(d)

    date_labels   = [label_date(d) for d in available_dates]
    label_to_date = dict(zip(date_labels, available_dates))

    current_ts: pd.Timestamp = st.session_state["current_ts"]
    date_key = f"date_sel_{mode_key}"
    time_key = f"time_sel_{mode_key}"

    # ── KEY INSIGHT: sync widget keys to current_ts BEFORE any widget renders.
    # This is only safe here because no widget with these keys exists yet in
    # this script run. If keys changed, rerun immediately so the widgets
    # see the updated values on a clean pass.
    # This is triggered after button navigation (which sets a "nav_pending" flag).
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.pop("nav_pending", False):
        tfd = sorted({ts.time() for ts in df.index if ts.date() == current_ts.date()})
        want_time = current_ts.time() if current_ts.time() in tfd else (tfd[0] if tfd else None)
        st.session_state[date_key] = label_date(current_ts.date())
        st.session_state[time_key] = want_time
        # No rerun needed here — keys are set before widgets; script continues cleanly.

    # ── Layout ────────────────────────────────────────────────────────────────
    col_date, col_time, col_risk = st.columns([2, 2, 3])

    with col_date:
        selected_date_label = st.selectbox(
            "Dátum  (🔴 = esemény nap)",
            options=date_labels,
            key=date_key,
        )
        date_selected = label_to_date[selected_date_label]
        d_prev_col, d_next_col = st.columns(2)
        with d_prev_col:
            date_prev_clicked = st.button("◀ Előző nap", key="btn_date_prev", use_container_width=True)
        with d_next_col:
            date_next_clicked = st.button("Köv. nap ▶", key="btn_date_next", use_container_width=True)

    with col_time:
        times_for_date = sorted({ts.time() for ts in df.index if ts.date() == date_selected})
        # Clamp stored time to valid options (date may have changed via selectbox)
        if st.session_state.get(time_key) not in times_for_date:
            st.session_state[time_key] = times_for_date[0] if times_for_date else None
        time_selected = st.selectbox(
            "Perc (csak ahol van adat)",
            options=times_for_date if times_for_date else ["–"],
            key=time_key,
        )
        t_prev_col, t_next_col = st.columns(2)
        with t_prev_col:
            time_prev_clicked = st.button("◀ Előző perc", key="btn_time_prev", use_container_width=True)
        with t_next_col:
            time_next_clicked = st.button("Köv. perc ▶", key="btn_time_next", use_container_width=True)

    # ── Navigation logic ──────────────────────────────────────────────────────
    # Buttons: update current_ts, set nav_pending flag, then rerun.
    # The next run will sync the widget keys BEFORE the widgets are created.
    # Selectbox: values already in session state via key=; just update current_ts.

    if time_prev_clicked:
        pos = df.index.get_loc(current_ts)
        if isinstance(pos, slice):
            pos = pos.start
        if pos - 1 >= 0:
            st.session_state["current_ts"] = df.index[pos - 1]
        st.session_state["nav_pending"] = True
        st.rerun()

    elif time_next_clicked:
        pos = df.index.get_loc(current_ts)
        if isinstance(pos, slice):
            pos = pos.start
        if pos + 1 < len(df.index):
            st.session_state["current_ts"] = df.index[pos + 1]
        st.session_state["nav_pending"] = True
        st.rerun()

    elif date_prev_clicked:
        cur_idx = available_dates.index(current_ts.date()) if current_ts.date() in available_dates else 0
        if cur_idx > 0:
            new_date = available_dates[cur_idx - 1]
            tfd = sorted({ts.time() for ts in df.index if ts.date() == new_date})
            if tfd:
                st.session_state["current_ts"] = pd.Timestamp(datetime.combine(new_date, tfd[0]))
        st.session_state["nav_pending"] = True
        st.rerun()

    elif date_next_clicked:
        cur_idx = available_dates.index(current_ts.date()) if current_ts.date() in available_dates else 0
        if cur_idx < len(available_dates) - 1:
            new_date = available_dates[cur_idx + 1]
            tfd = sorted({ts.time() for ts in df.index if ts.date() == new_date})
            if tfd:
                st.session_state["current_ts"] = pd.Timestamp(datetime.combine(new_date, tfd[0]))
        st.session_state["nav_pending"] = True
        st.rerun()

    else:
        # Manual selectbox pick — widget keys already updated by Streamlit.
        # Just sync current_ts from the widget values.
        if times_for_date and time_selected != "–":
            combined_ts = pd.Timestamp(datetime.combine(date_selected, time_selected))
            if combined_ts in df.index:
                current_ts = combined_ts
                st.session_state["current_ts"] = current_ts

    # ── Risk banner + status ──────────────────────────────────────────────────
    row = df.loc[current_ts]
    risk_agg_col = "risk_maint_aggregate" if is_maint else "risk_aggregate"
    p_agg_col    = "p_maint_aggregate"    if is_maint else "p_aggregate"
    risk_val     = row.get(risk_agg_col, "unknown")
    p_agg        = row.get(p_agg_col, None)
    try:
        p_agg_f = float(p_agg)
    except (TypeError, ValueError):
        p_agg_f = None

    with col_risk:
        st.markdown(risk_banner_html(risk_val, p_agg_f), unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.7rem;'
            'color:#9ca3af !important;margin-bottom:1px;">IDŐBÉLYEG</div>'
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.8rem;'
            f'color:#1a1f2e !important;margin-bottom:5px;">{current_ts}</div>',
            unsafe_allow_html=True,
        )
        pills_html = ""
        if is_maint:
            if bool(row.get("maintenance_event", False)):
                pills_html += status_pill("KARBANTARTÁSI ESEMÉNY", "#d97706", "🔧")
        else:
            if bool(row.get("fault_event", False)):
                pills_html += status_pill("HIBAPERC", "#dc2626", "⚠️")
            if bool(row.get("fault_episode", False)):
                pills_html += status_pill("HIBASZAKASZ KEZDETE", "#ea580c", "🟥")
        if bool(row.get("Gen_cb_cld", False)):
            pills_html += status_pill("MOTOR MŰKÖDIK", "#16a34a", "✅")
        else:
            pills_html += status_pill("MOTOR LEÁLLT", "#6b7280", "⏸️")
        st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Full-width day overview table ─────────────────────────────────────────
    st.subheader("Napi áttekintő")

    day_start = pd.Timestamp(datetime.combine(current_ts.date(), time(0, 0)))
    day_end   = day_start + timedelta(days=1)
    day_df    = df.loc[day_start:day_end].copy()

    if day_df.empty:
        st.info("Erre a napra nincs adat.")
    else:
        if is_maint:
            display_cols = [c for c in [
                "maintenance_event", "Gen_cb_cld",
                "p_maint_1h", "p_maint_6h", "p_maint_24h", "p_maint_3d", "p_maint_aggregate",
                "risk_maint_1h", "risk_maint_6h", "risk_maint_24h", "risk_maint_3d", "risk_maint_aggregate",
            ] if c in day_df.columns]
            ev_col = "maintenance_event"
        else:
            display_cols = [c for c in [
                "fault_event", "fault_episode", "Gen_cb_cld",
                "p_fault_1h", "p_fault_6h", "p_fault_24h", "p_fault_3d",
                "p_aggregate",
                "risk_1h", "risk_6h", "risk_24h", "risk_3d", "risk_aggregate",
            ] if c in day_df.columns]
            ev_col = "fault_event"

        table_df = day_df[display_cols].copy()
        risk_col_names = [c for c in table_df.columns if "risk" in c]

        RISK_TEXT = {
            "low":    "color: #16a34a",
            "medium": "color: #d97706",
            "high":   "color: #dc2626",
        }

        def style_row(row: pd.Series) -> list[str]:
            is_event = ev_col in row.index and bool(row.get(ev_col, False))
            styles = []
            for col in row.index:
                if is_event:
                    bg      = "background-color: #fca5a5;"
                    base_fg = "color: #7f1d1d; font-weight: 700;"
                else:
                    bg      = "background-color: #ffffff;"
                    base_fg = "color: #1a1f2e;"
                if col in risk_col_names:
                    fg = RISK_TEXT.get(str(row[col]).strip().lower(), "color: #1a1f2e")
                    styles.append(f"{bg} {fg}; font-weight: 600;")
                else:
                    styles.append(f"{bg} {base_fg}")
            return styles

        styled = (
            table_df.style
            .apply(style_row, axis=1)
            .set_table_styles([
                {"selector": "thead th", "props": [
                    ("background-color", "#eef0f4"),
                    ("color", "#6b7280"),
                    ("border-bottom", "2px solid #d8dce6"),
                    ("text-transform", "uppercase"),
                    ("letter-spacing", "1px"),
                    ("font-size", ".72rem"),
                    ("padding", "6px 10px"),
                    ("white-space", "nowrap"),
                ]},
                {"selector": "table", "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("font-family", "'IBM Plex Mono', monospace"),
                    ("font-size", ".78rem"),
                    ("background-color", "#ffffff"),
                ]},
                {"selector": "td", "props": [
                    ("border-bottom", "1px solid #e8ebf0"),
                    ("padding", "4px 10px"),
                ]},
                {"selector": "tr:hover td", "props": [
                    ("filter", "brightness(0.97)"),
                ]},
            ])
        )
        st.write(styled)

    st.markdown("---")

    # ── Model outputs ─────────────────────────────────────────────────────────
    st.subheader("Modell kimenetek")
    if is_maint:
        model_cols = [
            "p_maint_1h", "p_maint_6h", "p_maint_24h", "p_maint_3d", "p_maint_aggregate",
            "risk_maint_1h", "risk_maint_6h", "risk_maint_24h", "risk_maint_3d", "risk_maint_aggregate",
        ]
    else:
        model_cols = [
            "p_fault_1h", "p_fault_6h", "p_fault_24h", "p_fault_3d",
            "p_aggregate", "minutes_to_fault",
            "risk_1h", "risk_6h", "risk_24h", "risk_3d", "risk_ttf", "risk_aggregate",
        ]
    present_cols = [c for c in model_cols if c in df.columns]
    small_df = pd.DataFrame({c: [str(row.get(c))] for c in present_cols}).T
    small_df.columns = ["érték"]
    st.table(small_df)


if __name__ == "__main__":
    main()