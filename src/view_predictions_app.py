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

# ─────────────────────────────────────────────────────────────────────────────
# CSS – forces light theme on every element regardless of OS dark-mode setting.
# Targets both Streamlit's internal data-testid selectors AND the raw HTML
# elements that vary between themes.
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap');

/* ── Hard-reset: force light background on every Streamlit shell element ── */
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

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background-color: #eef0f4 !important;
    border-right: 1px solid #d8dce6 !important;
}
[data-testid="stSidebar"] * { color: #1a1f2e !important; }

/* ── Headings ── */
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

/* ── Paragraph / caption text ── */
p, span, label, div {
    color: #1a1f2e !important;
}
.stCaption, small { color: #9ca3af !important; font-size: .75rem !important; }

/* ── Buttons ── */
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

/* ── Selectbox ── */
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
[data-baseweb="menu"] li:hover {
    background-color: #eef0f4 !important;
}

/* ── st.table (small model output table) ── */
[data-testid="stTable"] {
    background-color: #ffffff !important;
}
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

/* ── Styled dataframe (day overview) ── */
[data-testid="stDataFrame"],
.stDataFrame iframe,
.dataframe-container {
    background-color: #ffffff !important;
    border: 1px solid #e2e5ec !important;
    border-radius: 4px;
}

/* ── Radio buttons ── */
[data-testid="stRadio"] label { color: #1a1f2e !important; }
[data-testid="stRadio"] { background: transparent !important; }

/* ── Dividers ── */
hr { border-color: #e2e5ec !important; }

/* ── Info boxes ── */
[data-testid="stAlert"] {
    background-color: #eff6ff !important;
    color: #1a1f2e !important;
    border: 1px solid #bfdbfe !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f7f8fa; }
::-webkit-scrollbar-thumb { background: #c8cdd8; border-radius: 3px; }

/* ── Responsive column gap tightening on smaller widths ── */
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

    html = (
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
    return html


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


def main() -> None:
    st.set_page_config(
        page_title="Powerszerviz előrejelzés",
        layout="wide",
        page_icon="⚡",
    )
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

    # ── Session state ─────────────────────────────────────────────────────────
    if "current_ts" not in st.session_state:
        st.session_state["current_ts"] = df.index[0]
    if "last_mode" not in st.session_state:
        st.session_state["last_mode"] = mode_key
    if st.session_state["last_mode"] != mode_key:
        st.session_state["current_ts"] = df.index[0]
        st.session_state["last_mode"] = mode_key

    # ── Event days ────────────────────────────────────────────────────────────
    event_col_name = "maintenance_event" if is_maint else "fault_event"
    if event_col_name in df.columns:
        event_days = set(df.index[df[event_col_name].astype(bool)].date)
    else:
        event_days = set()

    available_dates = sorted({ts.date() for ts in df.index})

    def label_date(d):
        return f"🔴 {d}" if d in event_days else str(d)

    date_labels  = [label_date(d) for d in available_dates]
    label_to_date = dict(zip(date_labels, available_dates))

    current_ts: pd.Timestamp = st.session_state["current_ts"]

    # ── Controls row: date col | time col | risk banner ───────────────────────
    col_date, col_time, col_risk = st.columns([2, 2, 3])

    # ── Collect all button clicks first (before any widget renders) ───────────
    with col_date:
        date_prev_clicked = st.button("◀ Előző nap", key="btn_date_prev")
        date_next_clicked = st.button("Következő nap ▶", key="btn_date_next")

    with col_time:
        time_prev_clicked = st.button("◀ Előző perc", key="btn_time_prev")
        time_next_clicked = st.button("Következő perc ▶", key="btn_time_next")

    # ── Apply navigation logic, updating current_ts and session state ─────────
    navigated = False

    if time_prev_clicked:
        pos = df.index.get_loc(current_ts)
        if isinstance(pos, slice):
            pos = pos.start
        if pos - 1 >= 0:
            current_ts = df.index[pos - 1]
        navigated = True

    elif time_next_clicked:
        pos = df.index.get_loc(current_ts)
        if isinstance(pos, slice):
            pos = pos.start
        if pos + 1 < len(df.index):
            current_ts = df.index[pos + 1]
        navigated = True

    elif date_prev_clicked:
        cur_date = current_ts.date()
        cur_date_idx = available_dates.index(cur_date) if cur_date in available_dates else 0
        if cur_date_idx > 0:
            new_date = available_dates[cur_date_idx - 1]
            times_on_day = sorted({ts.time() for ts in df.index if ts.date() == new_date})
            if times_on_day:
                current_ts = pd.Timestamp(datetime.combine(new_date, times_on_day[0]))
        navigated = True

    elif date_next_clicked:
        cur_date = current_ts.date()
        cur_date_idx = available_dates.index(cur_date) if cur_date in available_dates else 0
        if cur_date_idx < len(available_dates) - 1:
            new_date = available_dates[cur_date_idx + 1]
            times_on_day = sorted({ts.time() for ts in df.index if ts.date() == new_date})
            if times_on_day:
                current_ts = pd.Timestamp(datetime.combine(new_date, times_on_day[0]))
        navigated = True

    # Always persist the resolved timestamp
    st.session_state["current_ts"] = current_ts

    # ── Pre-write selectbox session state keys so dropdowns always reflect
    #    current_ts, regardless of what Streamlit cached previously ────────────
    date_key = f"date_sel_{mode_key}"
    time_key = f"time_sel_{mode_key}"

    correct_date_label = label_date(current_ts.date())
    st.session_state[date_key] = correct_date_label

    times_for_current_date = sorted({ts.time() for ts in df.index if ts.date() == current_ts.date()})
    if current_ts.time() in times_for_current_date:
        st.session_state[time_key] = current_ts.time()
    elif times_for_current_date:
        st.session_state[time_key] = times_for_current_date[0]

    # ── Now render the selectboxes (they will show the pre-written values) ─────
    with col_date:
        selected_date_label = st.selectbox(
            "Dátum  (🔴 = esemény nap)",
            options=date_labels,
            key=date_key,
        )
        date_selected = label_to_date[selected_date_label]

    with col_time:
        times_for_date = sorted({ts.time() for ts in df.index if ts.date() == date_selected})
        if times_for_date:
            time_selected = st.selectbox(
                "Perc (csak ahol van adat)",
                options=times_for_date,
                key=time_key,
            )
            # Only apply selectbox change when user picked manually (no button pressed)
            if not navigated:
                combined_ts = pd.Timestamp(datetime.combine(date_selected, time_selected))
                if combined_ts in df.index:
                    current_ts = combined_ts
                    st.session_state["current_ts"] = current_ts

    # ── Risk banner + status (ctrl5, same row as controls) ───────────────────
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