# app.py
# --- Football Analytics Toolkit (dynamic paths + ignored image output) ---

import re
import difflib
from io import BytesIO
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Plotly
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# Matplotlib & friends
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm

# mplsoccer
from mplsoccer import PyPizza

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit
import streamlit as st

# Images / requests
from PIL import Image
import requests


# =========================
# ====== PATHS & IO =======
# =========================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
IMG_DIR = ROOT_DIR / "analysis program images"     # keep ignored in git
IMG_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    """Make a filesystem-safe, short-ish filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")[:180]


# =========================
# ====== CONFIG & DATA ====
# =========================
# Fonts (prefer repo copies; fall back to defaults)
FONT_REGULAR_PATH = ROOT_DIR / "Gabarito-Regular.ttf"
FONT_BOLD_PATH = ROOT_DIR / "Gabarito-Bold.ttf"
try:
    font_normal = (
        fm.FontProperties(fname=str(FONT_REGULAR_PATH))
        if FONT_REGULAR_PATH.exists()
        else None
    )
    font_bold = (
        fm.FontProperties(fname=str(FONT_BOLD_PATH))
        if FONT_BOLD_PATH.exists()
        else None
    )
except Exception:
    font_normal = None
    font_bold = None

# Position groups
position_groups = {
    "GK": ["GK"],
    "DF": ["CB", "FB", "WB", "LB", "RB", "LWB", "RWB", "SW", "DF"],
    "MF": ["DM", "CM", "AM", "LM", "RM", "LW", "RW", "MF"],
    "FW": ["CF", "ST", "WF", "FW"],
}

# Pizza plot categories
pizza_plot_categories = {
    "Passing": [
        "compPass/90",
        "attPass/90",
        "pass%",
        "progPasses/90",
        "thirdPasses/90",
        "PPA/90",
        "xA/90",
        "kp/90",
        "xAG/90",
        "tb/90",
    ],
    "Defending": [
        "tackles/90",
        "Tkl+Int/90",
        "interceptions/90",
        "pAdjtacklesWon/90",
        "pAdjinterceptions/90",
        "clearances/90",
        "dribbledPast/90",
        "Blocked/90",
        "errors/90",
        "shotsBlocked/90",
        "passesBlocked/90",
        "tackleSuccessRate",
        "ballRecoveries/90",
        "midThirdTackles/90",
    ],
    "Carrying": [
        "progCarries/90",
        "thirdCarries/90",
        "Carries/90",
        "takeOnsAtt/90",
        "Succ/90",
        "att3rdTouches/90",
        "fouled/90",
    ],
    "Shooting": [
        "goals/90",
        "Sh/90",
        "SoT/90",
        "npg/90",
        "xG/90",
        "SoT%",
        "G/SoT",
        "goals",
        "xGOP/90",
        "G/Sh",
    ],
    "Aerial": ["headersWon/90", "headersWon%"],
    "Ball Retention": ["touches/90", "Dispossessed/90", "Mis/90", "sPass%", "ballRecoveries/90"],
}

# Category palette + mapping
category_palette = [
    "#2E4374",
    "#1A78CF",
    "#D70232",
    "#FF9300",
    "#44C3A1",
    "#CA228D",
    "#E1C340",
    "#7575A9",
    "#9DDFD3",
]
category_by_param = {}
for i, (_, stats_list) in enumerate(pizza_plot_categories.items()):
    for stat in stats_list:
        category_by_param[stat] = category_palette[i % len(category_palette)]

# Greens for similarity bars
greens_hex = ["#1fa44b"] * 6 + ["#158940", "#0e6d31", "#085222", "#053617"]

# Display names (pretty labels)
stat_display_names = {
    "tackles/90": "Tackles/90",
    "pAdjtacklesWon/90": "Possession Adjusted Tackles Won/90",
    "pAdjinterceptions/90": "Possession Adjusted Interceptions/90",
    "Tkl+Int/90": "Tkl+Int/90",
    "interceptions/90": "Interceptions/90",
    "clearances/90": "Clearances/90",
    "dribbledPast/90": "Dribbled Past/90",
    "Blocked/90": "Blocks/90",
    "errors/90": "Errors/90",
    "Mis/90": "Miscontrols/90",
    "passesBlocked/90": "Passes Blocked/90",
    "progCarries/90": "Progressive Carries/90",
    "thirdCarries/90": "Final Third Carries/90",
    "Carries/90": "Carries/90",
    "takeOnsAtt/90": "Take-Ons Attempted/90",
    "Succ/90": "Successful Dribbles/90",
    "compPass/90": "Passes Completed/90",
    "attPass/90": "Passes Attempted/90",
    "pass%": "Pass %",
    "progPasses/90": "Progressive Passes/90",
    "thirdPasses/90": "Passes to Final Third/90",
    "PPA/90": "Passes to Pen Area/90",
    "xA/90": "Expected Assists/90",
    "kp/90": "Key Passes/90",
    "goals/90": "Goals/90",
    "assists/90": "Assists/90",
    "Sh/90": "Shots/90",
    "SoT/90": "Shots on Target/90",
    "xG/90": "xG/90",
    "SCA90": "SCA/90",
    "GCA90": "GCA/90",
    "ballRecoveries/90": "Ball Recoveries/90",
    "headersWon%": "Headers Won %",
    "headersWon/90": "Headers Won/90",
    "tackleSuccessRate": "Tackles Won %",
    "xGOP/90": "xG Overperformance/90",
    "xAG/90": "Expected Assisted Goals/90",
    "ga/90": "Goal Contributions/90",
    "Dribblesucc%": "Dribble Success %",
    "att3rdTouches/90": "Attacking Third Touches/90",
    "shotsBlocked/90": "Shots Blocked/90",
    "npg/90": "Non Penalty Goals/90",
    "G/SoT": "Goals/SoT",
    "SoT%": "Shots On Target%",
    "goals": "Goals",
    "assists": "Assists",
    "tb/90": "Through Balls/90",
    "midThirdTackles/90": "Mid Third Tackles/90",
    "sPass%": "Short Pass %",
    "G/Sh": "Goals per Shot",
    # extras if present:
    "touches/90": "Touches/90",
    "fouled/90": "Fouled/90",
    "progCarryDist/90": "Progressive Carry Distance/90",
    "lPass%": "Long Pass %",
    "crosses/90": "Crosses/90",
    "compCross/90": "Completed Crosses/90",
    "defThirdTackles/90": "Defensive Third Tackles/90",
    "defPenTouches/90": "Defensive Penalty Touches/90",
    "def3rdTouches/90": "Defensive Third Touches/90",
    "mid3rdTouches/90": "Middle Third Touches/90",
}

# Role weights
position_weights = {
    "CB - Ball-Playing": {
        "progPasses/90": 1.7,
        "compPass/90": 1.6,
        "pass%": 1.9,
        "headersWon%": 1.4,
        "headersWon/90": 1.3,
        "interceptions/90": 1.3,
        "clearances/90": 1.2,
        "tackles/90": 1.2,
        "tackleSuccessRate": 1.8,
        "progCarryDist/90": 1.5,
        "lPass%": 1.5,
        "pAdjtacklesWon/90": 1.1,
    },
    "CB - Stopper": {
        "tackles/90": 1.8,
        "Tkl+Int/90": 1.7,
        "clearances/90": 1.6,
        "headersWon/90": 1.6,
        "headersWon%": 1.5,
        "Blocked/90": 1.4,
        "interceptions/90": 1.3,
        "passesBlocked/90": 1.0,
        "tackleSuccessRate": 1.95,
    },
    "CB - Sweeper": {
        "interceptions/90": 1.7,
        "clearances/90": 1.6,
        "tackles/90": 1.4,
        "progPasses/90": 1.4,
        "compPass/90": 1.3,
        "headersWon%": 1.2,
        "ballRecoveries/90": 1.5,
        "tackleSuccessRate": 1.7,
    },
    "FB/WB - Overlapping Attacker": {
        "crosses/90": 1.9,
        "compCross/90": 1.8,
        "tackleSuccessRate": 1.7,
        "assists/90": 1.6,
        "xAG/90": 1.5,
        "progCarries/90": 1.4,
        "PPA/90": 1.3,
        "thirdCarries/90": 1.2,
        "tackles/90": 1.4,
    },
    "FB/WB - Inverted": {
        "progPasses/90": 1.7,
        "compPass/90": 1.6,
        "xAG/90": 1.5,
        "kp/90": 1.4,
        "progCarries/90": 1.4,
        "thirdCarries/90": 1.3,
        "tackleSuccessRate": 1.2,
        "interceptions/90": 1.1,
        "goals/90": 1.0,
    },
    "FB - Defensive": {
        "tackles/90": 1.6,
        "interceptions/90": 1.5,
        "clearances/90": 1.6,
        "passesBlocked/90": 1.5,
        "Blocked/90": 1.6,
        "defThirdTackles/90": 1.8,
        "ballRecoveries/90": 1.2,
        "sPass%": 1.1,
        "Dispossessed/90": 1.0,
        "Mis/90": 1.0,
        "pass%": 1.0,
        "thirdPasses/90": 1.0,
        "crosses/90": 1.3,
        "dribbledPast/90": 0.6,
    },
    "DM - Ball Winner": {
        "tackles/90": 1.9,
        "Tkl+Int/90": 1.8,
        "interceptions/90": 1.7,
        "ballRecoveries/90": 1.6,
        "passesBlocked/90": 1.5,
        "clearances/90": 1.6,
        "progCarries/90": 1.2,
        "progPasses/90": 1.1,
        "shotsBlocked/90": 1.3,
        "tackleSuccessRate": 1.4,
        "defPenTouches/90": 1.3,
        "defThirdTackles/90": 1.5,
    },
    "DM - Deep-Lying Playmaker": {
        "compPass/90": 1.9,
        "progPasses/90": 1.8,
        "attPass/90": 1.9,
        "xA/90": 1.6,
        "thirdPasses/90": 1.5,
        "pass%": 1.7,
        "kp/90": 1.4,
        "tb/90": 1.6,
        "interceptions/90": 1.2,
        "tackleSuccessRate": 1.2,
        "def3rdTouches/90": 1.5,
        "progCarries/90": 1.3,
    },
    "CM - Box-to-Box Engine": {
        "Tkl+Int/90": 1.7,
        "progCarries/90": 1.7,
        "ballRecoveries/90": 1.6,
        "tackles/90": 1.6,
        "progPasses/90": 1.5,
        "thirdCarries/90": 1.4,
        "xA/90": 1.3,
        "xG/90": 1.2,
    },
    "CM - Shuttler/Link Player": {
        "compPass/90": 1.7,
        "progPasses/90": 1.6,
        "Carries/90": 1.6,
        "thirdCarries/90": 1.2,
        "pass%": 1.5,
        "tackles/90": 1.5,
        "interceptions/90": 1.3,
        "ballRecoveries/90": 1.3,
    },
    "CM - Mezzala": {
        "progCarries/90": 1.8,
        "thirdCarries/90": 1.7,
        "xA/90": 1.6,
        "Sh/90": 1.3,
        "attPass/90": 1.4,
        "PPA/90": 1.3,
        "kp/90": 1.3,
        "tackles/90": 1.2,
        "goals/90": 1.1,
        "mid3rdTouches/90": 1.7,
        "midThirdTackles/90": 1.5,
    },
    "AM - Classic 10": {
        "xAG/90": 1.9,
        "kp/90": 1.8,
        "SCA90": 1.7,
        "GCA90": 1.6,
        "PPA/90": 1.8,
        "assists/90": 1.5,
        "thirdPasses/90": 1.4,
        "progPasses/90": 1.2,
        "tb/90": 1.3,
        "goals/90": 1.2,
    },
    "ST - Target Man": {
        "headersWon/90": 1.9,
        "headersWon%": 1.9,
        "assists/90": 0.4,
        "xA/90": 1.2,
        "goals/90": 1.7,
        "npg/90": 1.8,
        "progPasses/90": 0.2,
    },
    "ST - Poacher": {
        "xG/90": 1.35,
        "goals/90": 0.3,
        "npg/90": 1.6,
        "G/Sh": 1.7,
        "SoT/90": 1.0,
        "headersWon%": 1.4,
        "goals": 1.9,
        "progCarries/90": -0.4,
        "Distance": 0.6,
    },
    "ST - Complete Forward": {
        "goals/90": 1.7,
        "npg/90": 1.7,
        "assists/90": 1.6,
        "xA/90": 1.6,
        "headersWon/90": 1.5,
        "progCarries/90": 1.5,
        "Sh/90": 1.4,
        "kp/90": 1.4,
        "goals": 1.2,
    },
    "Winger - Classic": {
        "crosses/90": 1.9,
        "compCross/90": 1.8,
        "assists/90": 1.7,
        "xA/90": 1.7,
        "PPA/90": 1.6,
        "takeOnsAtt/90": 1.5,
        "progCarries/90": 1.4,
        "tackleSuccessRate": 1.2,
    },
    "Winger - Inverted": {
        "goals/90": 1.8,
        "xG/90": 1.8,
        "Sh/90": 1.7,
        "progCarries/90": 1.6,
        "thirdCarries/90": 1.6,
        "xA/90": 1.5,
        "takeOnsAtt/90": 1.6,
        "SoT/90": 1.4,
        "kp/90": 1.3,
    },
    "AM - Shadow Striker": {
        "xG/90": 1.9,
        "goals/90": 1.9,
        "npg/90": 1.8,
        "Sh/90": 1.7,
        "progCarries/90": 1.6,
        "SCA90": 1.5,
        "thirdCarries/90": 1.5,
        "SoT/90": 1.5,
        "xAG/90": 1.2,
    },
}

# Role-specific tweaks
position_adjustments = {
    "CB - Ball-Playing": lambda x: min(x * 1.03, 100),
    "CB - Stopper": lambda x: min(x * 1.05, 100),
    "FB/WB - Overlapping Attacker": lambda x: min(x * 1.04, 100),
    "DM - Ball Winner": lambda x: min(x * 1.06, 100),
    "DM - Deep-Lying Playmaker": lambda x: min(x * 1.07, 100),
    "ST - Target Man": lambda x: min(x * 1.05, 100),
    "Winger - Inverted": lambda x: min(x * 1.03, 100),
}

# Archetypes
archetype_params = {role: list(weights.keys()) for role, weights in position_weights.items()}
category_archetypes = {f"{cat} Pizza": stats for cat, stats in pizza_plot_categories.items()}
archetype_params_full = {**archetype_params, **category_archetypes}

# ----------------------
# App config/theme
# ----------------------
st.set_page_config(page_title="Football Analytics Toolkit", layout="wide")
FONT_FAMILY = "Gabarito, Montserrat, Arial, sans-serif"
POSTER_BG = "#f1ffcd"  # pastel green vibe


# ==============
# IO + CACHING
# ==============
@st.cache_data
def load_csvs(base_folder: Path, which: str):
    base_folder = Path(base_folder)
    files = []
    if which == "BigDB_ALL.csv (Minor Leagues)":
        files = [base_folder / "BigDB_ALL.csv"]
    elif which == "BigDB.csv (Big 5 European Leagues)":
        files = [base_folder / "BigDB.csv"]
    else:
        files = [base_folder / "BigDB_ALL.csv", base_folder / "BigDB.csv"]
    dfs = []
    for p in files:
        if p.is_file():
            dfs.append(pd.read_csv(p))
        else:
            st.warning(f"File not found: {p}")
    if not dfs:
        st.stop()
    return pd.concat(dfs, ignore_index=True)


# ============
# UTILITIES
# ============
def filter_by_minutes(df, min_minutes):
    df = df.copy()
    df["Mins"] = pd.to_numeric(df["Mins"].astype(str).str.replace(",", ""), errors="coerce")
    return df[df["Mins"] >= min_minutes]


def position_relative_percentile(df, player_row, stat_col):
    pos_str = player_row.get("Pos", "")
    positions = [p.strip() for p in str(pos_str).split(",") if p.strip()]
    if not positions or stat_col not in df.columns:
        return np.nan
    mask = df["Pos"].apply(
        lambda s: any(pos in str(s).split(",") for pos in positions) if isinstance(s, str) else False
    )
    position_df = df[mask]
    if stat_col not in position_df.columns:
        return np.nan
    stat_vals = position_df[stat_col].replace([np.inf, -np.inf], np.nan).dropna()
    player_val = player_row.get(stat_col, np.nan)
    if pd.isnull(player_val) or stat_vals.empty:
        return np.nan
    return round(stats.percentileofscore(stat_vals, player_val, kind="mean"), 2)


def calculate_role_score(player_row, role_stats, df, role_name=None):
    if not role_stats:
        return 50.0
    pos_str = player_row.get("Pos", "")
    player_positions = [p.strip() for p in str(pos_str).split(",")] if isinstance(pos_str, str) else []
    pos_filtered_db = (
        df[
            df["Pos"].apply(
                lambda s: any(pos in str(s).split(",") for pos in player_positions) if isinstance(s, str) else False
            )
        ]
        if player_positions
        else df
    )

    percentiles, weights = [], []
    for stat in role_stats:
        if stat not in pos_filtered_db.columns:
            continue
        stat_vals = pos_filtered_db[stat].replace([np.inf, -np.inf], np.nan).dropna()
        player_val = player_row.get(stat, np.nan)
        if pd.notnull(player_val) and not stat_vals.empty:
            pctl = stats.percentileofscore(stat_vals, player_val, kind="mean")
            weight = position_weights.get(role_name, {}).get(stat, 1.0)
            percentiles.append(pctl)
            weights.append(weight)
    if not percentiles:
        return 50.0
    role_score = np.average(percentiles, weights=weights) if len(percentiles) >= 3 else np.mean(percentiles)
    role_score = position_adjustments.get(role_name, lambda x: x)(role_score)
    return float(np.clip(round(role_score, 1), 0, 100))


def break_label(label, max_len=15):
    if label.endswith("/90"):
        before = label[:-3].rstrip()
        if len(label) <= max_len:
            return label
        words, lines, current = before.split(" "), [], ""
        for w in words:
            check_len = len(current + (" " if current else "") + w + "/90")
            if len(lines) == 0 and check_len > max_len:
                if current:
                    lines.append(current)
                current = w
            elif len(current + (" " if current else "") + w) > max_len:
                lines.append(current)
                current = w
            else:
                current += (" " if current else "") + w
        lines.append(current + "/90")
        return "\n".join(lines)
    else:
        if len(label) <= max_len:
            return label
        words, lines, current = label.split(" "), [], ""
        for w in words:
            if len(current + (" " if current else "") + w) > max_len:
                lines.append(current)
                current = w
            else:
                current += (" " if current else "") + w
        if current:
            lines.append(current)
        return "\n".join(lines)


def league_strip_prefix(comp):
    for prefix in ("eng ", "it ", "es ", "de ", "fr "):
        if str(comp).lower().startswith(prefix):
            return comp[len(prefix) :]
    return comp


# ---- saving helpers ----
def save_plotly_to_ignored(fig: go.Figure, filename: str):
    """Try to save a Plotly figure as PNG to the ignored image folder."""
    try:
        png = fig.to_image(format="png", scale=2)  # requires kaleido
        out = IMG_DIR / safe_filename(filename)
        with open(out, "wb") as f:
            f.write(png)
    except Exception:
        # If kaleido isn't installed, silently skip.
        pass


def save_matplotlib_to_ignored(fig, filename: str, face=True):
    try:
        out = IMG_DIR / safe_filename(filename)
        fig.savefig(
            out,
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor=(fig.get_facecolor() if face else None),
        )
    except Exception:
        pass


# ---- download helpers (for UI buttons) ----
def fig_to_png_bytes_plotly(fig):
    # requires kaleido: pip install -U kaleido
    return fig.to_image(format="png", scale=2)


def fig_to_png_bytes_matplotlib(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ---------- DOT NAV + HELPERS ----------
def set_mode(new_mode: str):
    st.session_state["mode"] = new_mode


def dot_nav(mode_labels_to_keys, default_key):
    if "mode" not in st.session_state:
        st.session_state["mode"] = default_key
    st.markdown(
        """
        <style>
        .dot { font-size:22px; text-align:center; }
        .active { color:#1A78CF; }
        .inactive { color:#A8A8A8; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(len(mode_labels_to_keys))
    for i, (label, key) in enumerate(mode_labels_to_keys):
        active = st.session_state["mode"] == key
        with cols[i]:
            if st.button(label, key=f"lbl_{key}", use_container_width=True):
                set_mode(key)
            st.markdown(
                f"<div class='dot {'active' if active else 'inactive'}'>{'●' if active else '○'}</div>",
                unsafe_allow_html=True,
            )


def find_player_row(df, name_query):
    if not name_query:
        return None
    exact = df[df["Player"].str.lower() == name_query.lower()]
    if not exact.empty:
        return exact.iloc[0]
    all_names = df["Player"].dropna().unique().tolist()
    close = difflib.get_close_matches(name_query, all_names, n=1, cutoff=0.6)
    if close:
        return df[df["Player"] == close[0]].iloc[0]
    return None


# =================
# PLOT FUNCTIONS
# =================
def show_percentile_bar_chart(player_row, stat_cols, df, role_name):
    vals = [player_row.get(s, np.nan) for s in stat_cols]
    percentiles = [
        position_relative_percentile(df, player_row, s) if np.isfinite(v) else 0
        for s, v in zip(stat_cols, vals)
    ]
    labels = [stat_display_names.get(s, s) for s in stat_cols]
    bar = go.Bar(
        x=percentiles,
        y=labels,
        orientation="h",
        text=[f"{p:.1f}%" for p in percentiles],
        textposition="auto",
        marker=dict(color=percentiles, colorscale="RdYlGn"),
    )
    title = f"{player_row['Player']} — {role_name}<br><sup>Percentiles vs same-position players</sup>"
    fig = go.Figure([bar])
    fig.update_layout(
        title=dict(text=title, font=dict(color="#000")),
        plot_bgcolor=POSTER_BG,
        paper_bgcolor=POSTER_BG,
        xaxis=dict(
            title="Percentile",
            range=[0, 100],
            gridcolor="#eee",
            color="#000",
            tickfont=dict(color="#000"),
            linecolor="#000",
        ),
        yaxis=dict(automargin=True, color="#000", tickfont=dict(color="#000"), linecolor="#000"),
        template="simple_white",
        height=60 + 32 * len(labels),
        margin=dict(l=200, r=40, t=80, b=40),
        font=dict(family=FONT_FAMILY, color="#000"),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # auto-save
    save_plotly_to_ignored(fig, f"percentiles_{player_row['Player']}_{role_name}.png")

    df_out = pd.DataFrame({"Stat": labels, "Percentile": percentiles, "Value": vals})
    return fig, df_out


def show_pizza(player_row, stat_cols, df_filtered, role_name, lightmode=False, toppings=True):
    raw_vals = [player_row.get(s, float("nan")) for s in stat_cols]
    pcts = [
        position_relative_percentile(df_filtered, player_row, s)
        if np.isfinite(player_row.get(s, np.nan))
        else 0
        for s in stat_cols
    ]
    slice_colors = [category_by_param.get(s, "#2E4374") for s in stat_cols]
    value_color = "#222222" if lightmode else "#fffff0"
    text_colors = [value_color for _ in slice_colors]
    display_params = [break_label(stat_display_names.get(p, p), 15) for p in stat_cols]

    bg = "#f1ffcd" if lightmode else "#222222"
    param_color = "#222222" if lightmode else "#fffff0"

    baker = PyPizza(
        params=display_params,
        background_color=bg,
        straight_line_color="#000000",
        straight_line_lw=0.3,
        last_circle_color="#000000",
        last_circle_lw=1,
        other_circle_lw=0,
        inner_circle_size=0.30,
    )
    fig, ax = baker.make_pizza(
        pcts,
        alt_text_values=raw_vals,
        figsize=(10, 11),
        color_blank_space="same",
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=["cornflowerblue"] * len(slice_colors),
        blank_alpha=0.4,
        kwargs_slices=dict(edgecolor="#000000", zorder=2, linewidth=1),
        kwargs_params=dict(color=param_color, fontsize=18, fontproperties=font_normal, va="center"),
        kwargs_values=dict(
            color=value_color,
            fontsize=15,
            fontproperties=font_normal,
            zorder=3,
            bbox=dict(edgecolor="#000000", facecolor="cornflowerblue", boxstyle="round,pad=0.2", lw=1),
        ),
    )

    if toppings:
        club = player_row.get("Squad", "?")
        nationality = player_row.get("Nation", "?")
        age = player_row.get("Age", "?")
        mins = player_row.get("Mins", 0)
        season = "2024/25"
        header_color = "#222222" if lightmode else "#fffff0"
        fig.text(
            0.5,
            0.985,
            player_row["Player"],
            ha="center",
            va="top",
            fontsize=22,
            fontweight="bold",
            color=header_color,
            fontproperties=font_normal,
        )
        fig.text(
            0.5,
            0.952,
            f"{club} | {nationality} | {age} | {season}",
            ha="center",
            va="top",
            fontsize=15,
            color=header_color,
            fontproperties=font_normal,
        )
        fig.text(
            0.5,
            0.928,
            f"Role: {role_name} | Minutes played: {mins}",
            ha="center",
            va="top",
            fontsize=12,
            color=header_color,
            fontproperties=font_normal,
        )
        fig.text(
            0.5,
            0.01,
            "willumanalytics",
            ha="center",
            va="bottom",
            fontsize=9,
            color=("#666" if lightmode else "#CCC"),
            fontproperties=font_normal,
            alpha=0.85,
        )

    st.pyplot(fig, clear_figure=True)

    # auto-save to ignored folder
    fname = f"pizza_{safe_filename(player_row['Player'])}_{safe_filename(role_name)}_{'light' if lightmode else 'dark'}_{'toppings' if toppings else 'notoppings'}.png"
    save_matplotlib_to_ignored(fig, fname)

    return fig


def plot_role_leaderboard(df_filtered, role_name, role_stats):
    dfc = df_filtered.copy()
    dfc["RoleScore"] = [calculate_role_score(row, role_stats, dfc, role_name) for _, row in dfc.iterrows()]
    top = dfc.nlargest(10, "RoleScore").reset_index(drop=True)
    blue_gradient = [
        "#112052",
        "#1A3474",
        "#1F2C7B",
        "#274097",
        "#3457A4",
        "#4370BA",
        "#578BC8",
        "#6FA7D6",
        "#8EC3E0",
        "#B2DFFC",
    ]
    bar_colors = blue_gradient[-len(top) :][::-1]

    def lum(c):
        r, g, b = [x * 255 for x in mcolors.hex2color(c)]
        return 0.299 * r + 0.587 * g + 0.114 * b

    bar_text_colors = ["white" if lum(c) < 150 else "#222" for c in bar_colors]
    labels = [
        f"{row['Player']} • {row['RoleScore']:.1f} • {row.get('Age','?')} • {row.get('Squad','?')} • {int(row.get('Mins',0)):,} mins"
        for _, row in top.iterrows()
    ]
    fig = go.Figure(
        [
            go.Bar(
                x=top["RoleScore"],
                y=[f"#{i+1}" for i in range(len(top))],
                orientation="h",
                text=labels,
                textposition="inside",
                insidetextanchor="middle",
                marker=dict(color=bar_colors, line=dict(color="#333", width=1)),
                textfont=dict(color=bar_text_colors, size=13, family=FONT_FAMILY),
                customdata=top.index.values,
                hovertext=[f"{row['Player']} ({row['Squad']})" for _, row in top.iterrows()],
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"Top {role_name}s", font=dict(color="#000")),
        plot_bgcolor=POSTER_BG,
        paper_bgcolor=POSTER_BG,
        xaxis=dict(
            title="Role Suitability Score (0–100)",
            range=[0, 100],
            gridcolor=POSTER_BG,
            color="#000",
            tickfont=dict(color="#000"),
            linecolor="#000",
        ),
        yaxis=dict(autorange="reversed", showgrid=False, color="#000", tickfont=dict(color="#000"), linecolor="#000"),
        margin=dict(l=120, r=40, t=60, b=40),
        height=600,
        width=None,
        font=dict(family=FONT_FAMILY, color="#000"),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # auto-save
    save_plotly_to_ignored(fig, f"role_leaders_{role_name}.png")

    return top, fig


def show_top_players_by_stat(df, tidy_label, stat_col):
    top = df.nlargest(10, stat_col).reset_index(drop=True)
    blue_gradient = ["#112052", "#1A3474", "#1F2C7B", "#274097", "#3457A4", "#4370BA", "#6FA7D6", "#8EC3E0", "#B2DFFC"][
        ::-1
    ]
    bar_colors = blue_gradient[-len(top) :][::-1]

    def lum(c):
        r, g, b = [x * 255 for x in mcolors.hex2color(c)]
        return 0.299 * r + 0.587 * g + 0.114 * b

    bar_text_colors = ["white" if lum(c) < 150 else "#222" for c in bar_colors]
    labels = [
        f"{row['Player']} • {row.get(stat_col,0):.2f} • {row.get('Age','?')} • {row.get('Squad','?')} • {int(row.get('Mins',0)):,} mins"
        for _, row in top.iterrows()
    ]
    fig = go.Figure(
        [
            go.Bar(
                x=top[stat_col],
                y=[f"#{i+1}" for i in range(len(top))],
                orientation="h",
                text=labels,
                textposition="inside",
                insidetextanchor="middle",
                marker=dict(color=bar_colors, line=dict(color="#333", width=1)),
                textfont=dict(color=bar_text_colors, size=13, family=FONT_FAMILY),
                hovertext=[f"{row['Player']} ({row['Squad']})" for _, row in top.iterrows()],
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"Top 10: {tidy_label}", font=dict(color="#000")),
        plot_bgcolor=POSTER_BG,
        paper_bgcolor=POSTER_BG,
        xaxis=dict(title=tidy_label, gridcolor=POSTER_BG, color="#000", tickfont=dict(color="#000"), linecolor="#000"),
        yaxis=dict(autorange="reversed", showgrid=False, color="#000", tickfont=dict(color="#000"), linecolor="#000"),
        margin=dict(l=120, r=40, t=60, b=40),
        height=600,
        width=None,
        font=dict(family=FONT_FAMILY, color="#000"),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # auto-save
    save_plotly_to_ignored(fig, f"stat_leaders_{safe_filename(tidy_label)}.png")

    return top, fig


def plot_similarity_and_select(df_filtered, player_row, stat_cols, role_name):
    X = df_filtered[stat_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    player_vec = scaler.transform([player_row[stat_cols].fillna(0)])
    sim = cosine_similarity(player_vec, X_scaled)[0]
    df_filtered = df_filtered.reset_index(drop=True)
    try:
        self_idx = df_filtered.index[df_filtered["Player"] == player_row["Player"]][0]
        sim[self_idx] = -1
    except Exception:
        pass
    order = np.argsort(-sim)
    top_idx = order[:10]
    top_players = df_filtered.iloc[top_idx].copy()
    top_players["Similarity"] = sim[top_idx] * 100.0
    fig = go.Figure(
        [
            go.Bar(
                y=top_players["Player"][::-1],
                x=top_players["Similarity"][::-1],
                orientation="h",
                text=[f"{s:.2f}%" for s in top_players["Similarity"][::-1]],
                textposition="inside",
                marker=dict(color=["#1fa44b"] * len(top_players), line=dict(width=1, color="#333")),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"Most similar to {player_row['Player']} — {role_name}", font=dict(color="#000")),
        plot_bgcolor=POSTER_BG,
        paper_bgcolor=POSTER_BG,
        xaxis=dict(
            title="Similarity (%)",
            range=[0, 100],
            gridcolor="#222222",
            color="#222222",
            tickfont=dict(color="#000"),
            linecolor="#000",
        ),
        yaxis=dict(autorange="reversed", color="#000", tickfont=dict(color="#000"), linecolor="#000"),
        template="simple_white",
        height=500,
        font=dict(family=FONT_FAMILY, color="#000"),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # auto-save
    save_plotly_to_ignored(fig, f"similarity_{player_row['Player']}_{role_name}.png")

    pick = st.selectbox("Pick a player to compare on radar", top_players["Player"].tolist())
    return top_players[top_players["Player"] == pick].iloc[0] if pick else None


def plot_radar_percentiles(base_row, other_row, stat_cols, df_filtered, role_name):
    base_vals = [position_relative_percentile(df_filtered, base_row, s) for s in stat_cols]
    other_vals = [position_relative_percentile(df_filtered, other_row, s) for s in stat_cols]
    labels = [stat_display_names.get(s, s) for s in stat_cols]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=base_vals, theta=labels, fill="toself", name=f"{base_row['Player']}"))
    fig.add_trace(go.Scatterpolar(r=other_vals, theta=labels, fill="toself", name=f"{other_row['Player']}"))
    fig.update_layout(
        title=dict(text=f"{base_row['Player']} vs {other_row['Player']} — {role_name}", font=dict(color="#000")),
        polar=dict(
            bgcolor=POSTER_BG,
            radialaxis=dict(visible=True, range=[0, 100], color="#000", gridcolor="#ddd", tickfont=dict(color="#000")),
            angularaxis=dict(color="#000", tickfont=dict(color="#000")),
        ),
        showlegend=True,
        plot_bgcolor=POSTER_BG,
        paper_bgcolor=POSTER_BG,
        height=650,
        font=dict(family=FONT_FAMILY, color="#000"),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # auto-save
    save_plotly_to_ignored(fig, f"radar_{base_row['Player']}_vs_{other_row['Player']}_{role_name}.png")

    return fig


# ======== POSTER HELPERS (kept, not exposed in UI) ========
def pizza_fig_to_array(fig, dpi=220):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    return np.array(img)


def build_pizza_figure(
    player_row, stat_cols, df, role_name, lightmode=True, toppings=True, show_role_score=False
):
    raw_vals = [player_row.get(s, float("nan")) for s in stat_cols]
    pcts = [
        position_relative_percentile(df, player_row, s) if np.isfinite(player_row.get(s, np.nan)) else 0
        for s in stat_cols
    ]
    slice_colors = [category_by_param.get(s, "#2E4374") for s in stat_cols]
    value_color = "#222222" if lightmode else "#fffff0"
    text_colors = [value_color for _ in slice_colors]
    display_params = [break_label(stat_display_names.get(p, p), 15) for p in stat_cols]
    bg = POSTER_BG if lightmode else "#222222"
    param_color = value_color
    baker = PyPizza(
        params=display_params,
        background_color=bg,
        straight_line_color="#000000",
        straight_line_lw=0.3,
        last_circle_color="#000000",
        last_circle_lw=1,
        other_circle_lw=0,
        inner_circle_size=0.30,
    )
    fig, ax = baker.make_pizza(
        pcts,
        alt_text_values=raw_vals,
        figsize=(8.5, 8.8),
        color_blank_space="same",
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=["cornflowerblue"] * len(slice_colors),
        blank_alpha=0.40,
        kwargs_slices=dict(edgecolor="#000000", zorder=2, linewidth=1),
        kwargs_params=dict(color=param_color, fontsize=15, fontproperties=font_normal, va="center"),
        kwargs_values=dict(
            color=value_color,
            fontsize=12,
            fontproperties=font_normal,
            zorder=3,
            bbox=dict(edgecolor="#000000", facecolor="cornflowerblue", boxstyle="round,pad=0.18", lw=1),
        ),
    )
    if show_role_score:
        role_stats = archetype_params_full.get(role_name, [])
        score = calculate_role_score(player_row, role_stats, df, role_name)
        box_color = "#1A78CF" if score >= 70 else "#D70232" if score < 40 else "#FF9300"
        fig.text(
            0.86,
            0.90,
            f"Role Score: {score:.1f}",
            ha="center",
            va="top",
            fontsize=13,
            fontproperties=font_normal,
            fontweight="bold",
            color=("#f1ffcd" if box_color != "#FF9300" else "#222"),
            bbox=dict(boxstyle="round,pad=0.2", facecolor=box_color, edgecolor="#000000", linewidth=1),
            zorder=5,
        )
    return fig


def build_role_matrix_axes(ax, df, player_row, role_x, role_y):
    ax.set_facecolor(POSTER_BG)
    dfx = df.copy()
    dfx["RoleScore_X"] = [calculate_role_score(r, archetype_params_full[role_x], dfx, role_x) for _, r in dfx.iterrows()]
    dfx["RoleScore_Y"] = [calculate_role_score(r, archetype_params_full[role_y], dfx, role_y) for _, r in dfx.iterrows()]
    ax.scatter(dfx["RoleScore_X"], dfx["RoleScore_Y"], s=10, edgecolor="#333", linewidth=0.6, alpha=0.85, color="#1f77b4")
    me = dfx[dfx["Player"] == player_row["Player"]]
    if not me.empty:
        ax.scatter(
            me["RoleScore_X"],
            me["RoleScore_Y"],
            s=60,
            edgecolor="#000",
            linewidth=1.2,
            color="#B80019",
            zorder=5,
        )
        ax.text(
            me["RoleScore_X"].values[0] + 1.5,
            me["RoleScore_Y"].values[0],
            player_row["Player"],
            fontsize=10,
            color="#111",
            fontproperties=font_normal,
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel(role_x)
    ax.set_ylabel(role_y)
    ax.grid(color="#d9e6a6", linewidth=1, alpha=1.0)
    ax.set_title(f"Role Matrix: {role_x} vs {role_y}", fontsize=14, pad=8, fontproperties=font_normal, color="#000")


def make_dashboard_poster(player_row, df, main_role, role_x, role_y, headshot_url=None):
    central_fig = build_pizza_figure(
        player_row, archetype_params_full[main_role], df, main_role, lightmode=True, toppings=True, show_role_score=True
    )
    central_img = pizza_fig_to_array(central_fig, dpi=220)
    cats = ["Shooting", "Carrying", "Passing"]
    mini_imgs = []
    for c in cats:
        mini_fig = build_pizza_figure(
            player_row, pizza_plot_categories.get(c, []), df, c, lightmode=True, toppings=False, show_role_score=False
        )
        mini_imgs.append(pizza_fig_to_array(mini_fig, dpi=220))

    fig = plt.figure(figsize=(12.5, 17), facecolor=POSTER_BG)
    gs = fig.add_gridspec(14, 12)

    # Header
    ax_hdr = fig.add_subplot(gs[0:2, 0:12])
    ax_hdr.axis("off")
    name = player_row.get("Player", "?")
    try:
        age_val = player_row.get("Age", "?")
        age = int(float(age_val)) if pd.notnull(age_val) else "?"
    except Exception:
        age = player_row.get("Age", "?")
    club = player_row.get("Squad", "?")
    nation = player_row.get("Nation", "?")
    pos = player_row.get("Pos", "?")
    mins = int(pd.to_numeric(player_row.get("Mins", 0), errors="coerce") or 0)

    if headshot_url:
        try:
            r = requests.get(headshot_url, timeout=5)
            im = Image.open(io.BytesIO(r.content)).convert("RGBA")
            ax_img = fig.add_subplot(gs[0:2, 0:2])
            ax_img.imshow(im)
            ax_img.axis("off")
        except Exception:
            pass

    ax_hdr.text(0.02, 0.70, name, fontsize=26, fontweight="bold", fontproperties=font_normal, color="#111")
    ax_hdr.text(
        0.02,
        0.30,
        f"{club}  ·  {nation}  ·  {pos}  ·  {mins:,} mins  ·  Age {age}",
        fontsize=13.5,
        fontproperties=font_normal,
        color="#222",
    )

    # Center pizza
    ax_center = fig.add_subplot(gs[2:9, 0:8])
    ax_center.axis("off")
    ax_center.imshow(central_img)
    ax_center.set_title(f"Selected Role: {main_role}", fontsize=14, pad=8, fontproperties=font_normal)

    # Mini pizzas
    for i, (img, title) in enumerate(zip(mini_imgs, ["Shooting", "Carrying", "Passing"])):
        ax_m = fig.add_subplot(gs[2 + i * 3 : 2 + i * 3 + 3, 8:12])
        ax_m.axis("off")
        ax_m.imshow(img)
        ax_m.set_title(f"{title}:", fontsize=13, pad=6, fontproperties=font_normal, loc="left")

    # Role matrix
    ax_mat = fig.add_subplot(gs[9:14, 0:12])
    build_role_matrix_axes(ax_mat, df, player_row, role_x, role_y)

    fig.text(0.01, 0.01, "@nstntly", fontsize=10, color="#777", fontproperties=font_normal)

    # auto-save
    save_matplotlib_to_ignored(
        fig, f"poster_{safe_filename(player_row['Player'])}_{safe_filename(main_role)}.png", face=True
    )

    return fig


# ====================
# STREAMLIT RENDERER (hidden from dot menu)
# ====================
def render_player_dashboard(player_row, df):
    if player_row is None:
        st.info("Type or pick a player in the sidebar first.")
        return
    st.markdown("### Player Dashboard (Poster)")
    arch_keys_local = list(archetype_params_full.keys())
    default_main = arch_keys_local.index("ST - Target Man") if "ST - Target Man" in arch_keys_local else 0
    default_x = arch_keys_local.index("Winger - Inverted") if "Winger - Inverted" in arch_keys_local else 0
    default_y = arch_keys_local.index("ST - Target Man") if "ST - Target Man" in arch_keys_local else 1
    main_role = st.selectbox("Selected Role (big pizza)", arch_keys_local, index=default_main)
    role_x = st.selectbox("Matrix X role", arch_keys_local, index=default_x)
    role_y = st.selectbox("Matrix Y role", arch_keys_local, index=default_y)
    headshot_url = st.text_input("Optional headshot image URL (for header)", "")
    fig = make_dashboard_poster(player_row, df, main_role, role_x, role_y, headshot_url or None)
    st.pyplot(fig, clear_figure=False)
    upscale_dpi = st.slider("Download DPI (higher = larger PNG)", 200, 600, 360, step=20)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=upscale_dpi, bbox_inches="tight", facecolor=POSTER_BG)
    buf.seek(0)
    st.download_button(
        "Download Poster PNG",
        data=buf.getvalue(),
        file_name=f"{player_row['Player'].replace(' ','_')}_dashboard_poster.png",
        mime="image/png",
    )


# =====================
# SIDEBAR + CONTROLS
# =====================
st.sidebar.header("Data & Filters")
BASE = st.sidebar.text_input("Base folder", str(DATA_DIR))
db_choice = st.sidebar.selectbox(
    "Database", ["BigDB_ALL.csv (Minor Leagues)", "BigDB.csv (Big 5 European Leagues)", "Both"], index=2
)
df = load_csvs(Path(BASE), db_choice)

# Position filter
st.sidebar.write("Available positions: GK, DF, MF, FW")
pos_sel = st.sidebar.multiselect("Filter by position(s) (optional)", ["GK", "DF", "MF", "FW"])
if pos_sel:
    mask = df["Pos"].apply(lambda x: any(p in str(x).split(",") for p in pos_sel))
    df = df[mask]

min_minutes = st.sidebar.number_input("Minimum minutes", min_value=0, max_value=10000, value=900, step=30)
df = filter_by_minutes(df, min_minutes)

# Player selection + fuzzy
player_list = df["Player"].dropna().unique().tolist()
player_name = st.sidebar.selectbox("Player (dropdown)", player_list) if len(player_list) else None
typed_query = st.sidebar.text_input("Or type a player name")
typed_row = find_player_row(df, typed_query) if typed_query else None
player_row = typed_row if typed_row is not None else (df[df["Player"] == player_name].iloc[0] if player_name else None)

# Archetype selection
arch_keys = list(archetype_params_full.keys())
arch_choice = st.sidebar.selectbox("Archetype", arch_keys) if arch_keys else None
stat_cols_for_arch = archetype_params_full.get(arch_choice, []) if arch_choice else []

# League filter helper
def league_filter_ui(dfin):
    dfin = dfin.copy()
    dfin["LeagueName"] = dfin["Comp"].apply(league_strip_prefix)
    leagues = sorted(dfin["LeagueName"].dropna().unique().tolist())
    chosen = st.multiselect("Filter by league(s) (optional)", leagues)
    if chosen:
        dfin = dfin[dfin["LeagueName"].isin(chosen)]
    return dfin.drop(columns=["LeagueName"], errors="ignore")


# ======================
# MODE SELECTOR (Top dots)
# ======================
MODE_ITEMS = [
    ("Similar", "1"),
    ("Percentiles", "2"),
    ("Pizza", "3"),
    ("Role Leaders", "4"),
    ("Best Roles", "5"),
    ("Stat Leaders", "6"),
    ("Pizza Dark (no toppings)", "8"),
    ("Pizza Light", "9"),
    ("Pizza Light (no toppings)", "10"),
    ("Stat Scatter", "12"),
    ("Role Matrix", "13"),
]
dot_nav(MODE_ITEMS, default_key="1")
mode = st.session_state["mode"]

st.title("Football Analytics Toolkit — Streamlit")


# ===============
# MODE HANDLERS
# ===============
if mode == "1":
    if player_row is None or arch_choice is None:
        st.info("Pick a player and archetype in the sidebar.")
    else:
        role_name = arch_choice
        role_stats = stat_cols_for_arch
        df_for_calc = df.copy()
        st.subheader("Similarity")
        other_row = plot_similarity_and_select(df_for_calc, player_row, role_stats, role_name)
        if other_row is not None:
            st.subheader("Radar comparison (percentiles)")
            radar_fig = plot_radar_percentiles(player_row, other_row, role_stats, df_for_calc, role_name)
            try:
                png_bytes = fig_to_png_bytes_plotly(radar_fig)
                st.download_button(
                    "Download radar (PNG)",
                    data=png_bytes,
                    file_name=f"radar_{player_row['Player'].replace(' ','_')}_vs_{other_row['Player'].replace(' ','_')}.png",
                    mime="image/png",
                )
            except Exception:
                st.caption("To enable PNG downloads for Plotly charts, install **kaleido**: pip install -U kaleido")
            lines = [
                f"{other_row['Player']} (Age: {other_row.get('Age','?')}, Club: {other_row.get('Squad','?')}, Minutes: {other_row.get('Mins','?')})",
                f"Compared vs baseline: {player_row['Player']} — Role: {role_name}",
            ]
            for s in role_stats:
                v = other_row.get(s, np.nan)
                p = position_relative_percentile(df_for_calc, other_row, s)
                if pd.notnull(v):
                    lines.append(f"{stat_display_names.get(s,s)}: {v:.2f} (Percentile: {p:.1f}%)")
            txt = "\n".join(lines)
            st.text_area("Copy summary", txt, height=220)

elif mode == "2":
    if player_row is None or arch_choice is None:
        st.info("Pick a player and archetype in the sidebar.")
    else:
        role_name = arch_choice
        fig, df_csv = show_percentile_bar_chart(player_row, stat_cols_for_arch, df, role_name)
        try:
            png = fig_to_png_bytes_plotly(fig)
            st.download_button(
                "Download percentile chart (PNG)",
                data=png,
                file_name=f"percentiles_{player_row['Player'].replace(' ','_')}_{role_name.replace(' ','_')}.png",
                mime="image/png",
            )
        except Exception:
            st.caption("To enable PNG downloads for Plotly charts, install **kaleido**: pip install -U kaleido")
        st.download_button(
            "Download percentiles (CSV)",
            data=df_csv.to_csv(index=False).encode("utf-8"),
            file_name=f"percentiles_{player_row['Player'].replace(' ','_')}_{role_name.replace(' ','_')}.csv",
            mime="text/csv",
        )

elif mode == "3":
    if player_row is None or arch_choice is None:
        st.info("Pick a player and archetype in the sidebar.")
    else:
        role_name = arch_choice
        fig = show_pizza(player_row, stat_cols_for_arch, df, role_name, lightmode=False, toppings=True)
        st.download_button(
            "Download pizza (dark, toppings) PNG",
            data=fig_to_png_bytes_matplotlib(fig),
            file_name=f"pizza_{player_row['Player'].replace(' ','_')}_{role_name.replace(' ','_')}_dark_toppings.png",
            mime="image/png",
        )

elif mode == "4":
    role_name = st.selectbox("Role", list(position_weights.keys()))
    role_stats = list(position_weights[role_name].keys())
    df_role = league_filter_ui(df)
    top, fig = plot_role_leaderboard(df_role, role_name, role_stats)
    csv = top[["Player", "Age", "Squad", "Nation", "Mins", "RoleScore"] + role_stats].copy()
    csv["Role"] = role_name
    st.download_button(
        "Download CSV",
        csv.to_csv(index=False).encode("utf-8"),
        file_name=f"top_{role_name.replace(' ','_')}.csv",
    )

elif mode == "5":
    if player_row is None:
        st.info("Pick a player in the sidebar.")
    else:
        pos_str = player_row.get("Pos", "")
        position_to_roles = {
            "GK": ["GK"],
            "DF": ["CB", "FB", "WB", "LB", "RB", "LWB", "RWB", "SW"],
            "MF": ["DM", "CM", "AM", "LM", "RM", "LW", "RW"],
            "FW": ["ST", "CF", "WF", "Winger"],
        }
        sel_groups = st.multiselect("Relevant position groups (optional)", ["GK", "DF", "MF", "FW"], default=[])
        if not sel_groups:
            if any(k in pos_str for k in ["GK"]):
                sel_groups.append("GK")
            if any(k in pos_str for k in ["CB", "FB", "WB", "LB", "RB", "LWB", "RWB", "SW", "DF"]):
                sel_groups.append("DF")
            if any(k in pos_str for k in ["DM", "CM", "AM", "LM", "RM", "LW", "RW", "MF"]):
                sel_groups.append("MF")
            if any(k in pos_str for k in ["ST", "CF", "WF", "Winger", "FW"]):
                sel_groups.append("FW")
            sel_groups = list(dict.fromkeys(sel_groups))
        relevant_roles = []
        for group in sel_groups:
            kws = position_to_roles.get(group, [])
            relevant_roles.extend([r for r in archetype_params_full if any(kw in r for kw in kws)])
        relevant_roles = list(dict.fromkeys(relevant_roles)) or list(archetype_params_full.keys())
        df_for_calc = df.copy()
        role_scores, role_details = [], {}
        for role in relevant_roles:
            stats_list = archetype_params_full.get(role, [])
            score = calculate_role_score(player_row, stats_list, df_for_calc, role)
            detail_rows = []
            for s in stats_list:
                val = player_row.get(s, np.nan)
                if pd.notnull(val):
                    pctl = position_relative_percentile(df_for_calc, player_row, s)
                    detail_rows.append((stat_display_names.get(s, s), val, pctl))
            role_scores.append((role, score))
            role_details[role] = detail_rows
        role_scores.sort(key=lambda x: x[1], reverse=True)
        st.subheader(f"Role suitability — {player_row['Player']}")
        fig = go.Figure(
            [
                go.Bar(
                    x=[s for _, s in role_scores],
                    y=[r for r, _ in role_scores],
                    orientation="h",
                    marker=dict(
                        color=[sample_colorscale("Blues", [0.2 + 0.8 * i / len(role_scores)])[0] for i in range(len(role_scores))],
                        line=dict(color="#333", width=1),
                    ),
                    text=[f"{s:.1f}" for _, s in role_scores],
                    textposition="inside",
                    insidetextanchor="middle",
                )
            ]
        )
        fig.update_layout(
            title=dict(text=f"Role suitability — {player_row['Player']}", font=dict(color="#000")),
            plot_bgcolor=POSTER_BG,
            paper_bgcolor=POSTER_BG,
            xaxis=dict(title="Suitability (0–100)", range=[0, 100], color="#000", tickfont=dict(color="#000"), linecolor="#000"),
            yaxis=dict(autorange="reversed", color="#000", tickfont=dict(color="#000"), linecolor="#000"),
            height=min(80 + 28 * len(role_scores), 1200),
            template="simple_white",
            font=dict(family=FONT_FAMILY, color="#000"),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # auto-save
        save_plotly_to_ignored(fig, f"role_suitability_{player_row['Player']}.png")

        role_pick = st.selectbox("View stat details for role", [r for r, _ in role_scores])
        det = role_details.get(role_pick, [])
        if det:
            st.write("**Stat details (player value, positional percentile):**")
            st.dataframe(pd.DataFrame(det, columns=["Stat", "Value", "Percentile (%)"]))

elif mode == "6":
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ["Age", "Mins"]]
    display_names = {c: stat_display_names.get(c, c) for c in numeric_cols}
    if not numeric_cols:
        st.warning("No numeric stat columns found.")
    else:
        x = st.selectbox("Choose a stat", [display_names[c] for c in numeric_cols])
        stat_col = [c for c in numeric_cols if display_names[c] == x][0]
        df_league = league_filter_ui(df)
        top, fig = show_top_players_by_stat(df_league, x, stat_col)

elif mode == "8":
    if player_row is None or arch_choice is None:
        st.info("Pick a player and archetype in the sidebar.")
    else:
        role_name = arch_choice
        fig = show_pizza(player_row, stat_cols_for_arch, df, role_name, lightmode=False, toppings=False)
        st.download_button(
            "Download pizza (dark, no toppings) PNG",
            data=fig_to_png_bytes_matplotlib(fig),
            file_name=f"pizza_{player_row['Player'].replace(' ','_')}_{role_name.replace(' ','_')}_dark_notoppings.png",
            mime="image/png",
        )

elif mode == "9":
    if player_row is None or arch_choice is None:
        st.info("Pick a player and archetype in the sidebar.")
    else:
        role_name = arch_choice
        fig = show_pizza(player_row, stat_cols_for_arch, df, role_name, lightmode=True, toppings=True)
        st.download_button(
            "Download pizza (light, toppings) PNG",
            data=fig_to_png_bytes_matplotlib(fig),
            file_name=f"pizza_{player_row['Player'].replace(' ','_')}_{role_name.replace(' ','_')}_light_toppings.png",
            mime="image/png",
        )

elif mode == "10":
    if player_row is None or arch_choice is None:
        st.info("Pick a player and archetype in the sidebar.")
    else:
        role_name = arch_choice
        fig = show_pizza(player_row, stat_cols_for_arch, df, role_name, lightmode=True, toppings=False)
        st.download_button(
            "Download pizza (light, no toppings) PNG",
            data=fig_to_png_bytes_matplotlib(fig),
            file_name=f"pizza_{player_row['Player'].replace(' ','_')}_{role_name.replace(' ','_')}_light_notoppings.png",
            mime="image/png",
        )

elif mode == "12":
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ["Age", "Mins"]]
    display_names = {c: stat_display_names.get(c, c) for c in numeric_cols}
    if len(numeric_cols) >= 2:
        x_pick = st.selectbox("X-axis stat", [display_names[c] for c in numeric_cols], index=0)
        y_pick = st.selectbox("Y-axis stat", [display_names[c] for c in numeric_cols], index=1)
        x_col = [c for c in numeric_cols if display_names[c] == x_pick][0]
        y_col = [c for c in numeric_cols if display_names[c] == y_pick][0]
        dfx = df.dropna(subset=[x_col, y_col]).copy()
        dfx["Age"] = dfx["Age"].fillna("?")
        hover = [
            f"<b>{r['Player']}</b><br>Team: {r.get('Squad','?')}<br>Age: {r.get('Age','?')}<br>{x_pick}: {r[x_col]:.2f}<br>{y_pick}: {r[y_col]:.2f}<br>Minutes: {int(r.get('Mins',0)):,}"
            for _, r in dfx.iterrows()
        ]
        fig = go.Figure(
            [
                go.Scatter(
                    x=dfx[x_col],
                    y=dfx[y_col],
                    mode="markers",
                    marker=dict(size=9, color=dfx["Age"], colorscale="Blues", line=dict(width=1, color="#333")),
                    text=dfx["Player"],
                    hovertext=hover,
                    hoverinfo="text",
                )
            ]
        )
        fig.update_layout(
            title=dict(text=f"{x_pick} vs {y_pick} — scatter", font=dict(color="#000")),
            xaxis=dict(title=x_pick, gridcolor=POSTER_BG, zeroline=False, linecolor="#000", color="#000", tickfont=dict(color="#000")),
            yaxis=dict(title=y_pick, gridcolor=POSTER_BG, zeroline=False, linecolor="#000", color="#000", tickfont=dict(color="#000")),
            plot_bgcolor=POSTER_BG,
            paper_bgcolor=POSTER_BG,
            height=800,
            font=dict(family=FONT_FAMILY, color="#000"),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # auto-save
        save_plotly_to_ignored(fig, f"scatter_{safe_filename(x_pick)}_vs_{safe_filename(y_pick)}.png")
    else:
        st.warning("Need at least two numeric stat columns.")

elif mode == "13":
    role_x = st.selectbox("X-axis role", arch_keys, index=0, key="role_x")
    role_y = st.selectbox("Y-axis role", arch_keys, index=1, key="role_y")
    dfc = df.copy()
    dfc["RoleScore_X"] = [calculate_role_score(row, archetype_params_full[role_x], dfc, role_x) for _, row in dfc.iterrows()]
    dfc["RoleScore_Y"] = [calculate_role_score(row, archetype_params_full[role_y], dfc, role_y) for _, row in dfc.iterrows()]
    hover = [
        f"<b>{r['Player']}</b><br>Team: {r.get('Squad','?')}<br>Age: {r.get('Age','?')}<br>{role_x}: {r['RoleScore_X']:.1f}<br>{role_y}: {r['RoleScore_Y']:.1f}<br>Minutes: {int(r.get('Mins',0)):,}"
        for _, r in dfc.iterrows()
    ]
    fig = go.Figure(
        [
            go.Scatter(
                x=dfc["RoleScore_X"],
                y=dfc["RoleScore_Y"],
                mode="markers",
                marker=dict(size=9, color=dfc["Age"], colorscale="Blues", line=dict(width=1, color="#333")),
                text=dfc["Player"],
                hovertext=hover,
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"{role_x} vs {role_y} — role suitability", font=dict(color="#000")),
        xaxis=dict(title=role_x, gridcolor=POSTER_BG, zeroline=False, linecolor="#000", color="#000", tickfont=dict(color="#000")),
        yaxis=dict(title=role_y, gridcolor=POSTER_BG, zeroline=False, linecolor="#000", color="#000", tickfont=dict(color="#000")),
        plot_bgcolor=POSTER_BG,
        paper_bgcolor=POSTER_BG,
        height=800,
        font=dict(family=FONT_FAMILY, color="#000"),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # auto-save
    save_plotly_to_ignored(fig, f"role_matrix_{safe_filename(role_x)}_vs_{safe_filename(role_y)}.png")
