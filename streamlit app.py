import os
import re
import base64
import difflib
from io import BytesIO
import io

import numpy as np
import pandas as pd
from scipy import stats
import kaleido  # for Plotly image export
# Plotly
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# Matplotlib & friends
import matplotlib as mpl
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

import json


# =========================
# ====== CONFIG & DATA ====
# =========================

# ===== Fonts (repo-first, crash-safe fallbacks) =====
APP_DIR = os.path.dirname(os.path.abspath(__file__))

GABARITO_REG_PATH = os.path.join(APP_DIR, "fonts", "Gabarito-Regular.ttf")
GABARITO_BOLD_PATH = os.path.join(APP_DIR, "fonts", "Gabarito-Bold.ttf")

if "custom_arches" not in st.session_state:
    st.session_state["custom_arches"] = {}

def _fontprops_or_fallback(ttf_path: str, fallback_family: str = "DejaVu Sans"):
    try:
        if os.path.isfile(ttf_path):
            try:
                fm.fontManager.addfont(ttf_path)
            except Exception:
                pass
            return fm.FontProperties(fname=ttf_path)
    except Exception:
        pass
    return fm.FontProperties(family=fallback_family)

font_normal = _fontprops_or_fallback(GABARITO_REG_PATH)
font_bold   = _fontprops_or_fallback(GABARITO_BOLD_PATH)

FONT_FAMILY = "Gabarito, DejaVu Sans, Arial, sans-serif"
mpl.rcParams["font.family"] = ["Gabarito", "DejaVu Sans", "Arial", "sans-serif"]


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Gabarito:wght@400;700&family=Material+Symbols+Outlined');

:root, body, .stApp { --app-font: 'Gabarito', 'DejaVu Sans', Arial, sans-serif; }

.stApp *:not(.material-symbols-outlined):not([class*="material-symbols"]):not([data-testid="stIcon"]):not(.stIconMaterial):not([data-testid="stIconMaterial"]) {
  font-family: var(--app-font) !important;
}

.material-symbols-outlined,
[class*="material-symbols"],
[data-testid="stIcon"],
[data-testid="stIcon"] *,
.stIconMaterial,
.stIconMaterial *,
[data-testid="stIconMaterial"],
[data-testid="stIconMaterial"] * {
  font-family: 'Material Symbols Outlined' !important;
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
  letter-spacing: normal !important;
  line-height: 1 !important;
}

[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] * {
  font-family: 'Material Symbols Outlined' !important;
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
  letter-spacing: normal !important;
  line-height: 1 !important;
}

[data-testid="stExpander"] summary {
  font-family: var(--app-font) !important;
}
[data-testid="stExpander"] summary::-webkit-details-marker { display: none; }
[data-testid="stExpander"] summary::marker { content: ""; }

.stButton>button {
  font-family: var(--app-font) !important;
  line-height: normal !important;
  pointer-events: auto !important;
}
</style>
""", unsafe_allow_html=True)

POSTER_BG = "#f1ffcd"
HOVER_BG = "#f5f5dc"

position_groups = {
    'GK': ['GK'],
    'DF': ['CB', 'FB', 'WB', 'LB', 'RB', 'LWB', 'RWB', 'SW', 'DF'],
    'MF': ['DM', 'CM', 'AM', 'LM', 'RM', 'LW', 'RW', 'MF'],
    'FW': ['CF', 'ST', 'WF', 'FW']
}

pizza_plot_categories = {
    "Passing": ["compPass/90", "attPass/90", "pass%", "progPasses/90", "thirdPasses/90", "PPA/90", "xA/90", "kp/90", "xAG/90", "tb/90","pAdjprogPasses/90","pAdjxAG/90"],
    "Defending": ["tackles/90", "Tkl+Int/90", "interceptions/90", "pAdjtacklesWon/90", "pAdjinterceptions/90", "clearances/90", "dribbledPast/90", "Blocked/90", "errors/90", "shotsBlocked/90", "passesBlocked/90", "tackleSuccessRate", "ballRecoveries/90", "midThirdTackles/90","pAdjclearances/90","pAdjshotsBlocked/90","pAdjtackles/90"],
    "Carrying": ["progCarries/90", "thirdCarries/90", "Carries/90", "takeOnsAtt/90", "Succ/90", "att3rdTouches/90", "fouled/90","pAdjprogCarries/90","pAdjtouches/90"],
    "Shooting": ["goals/90", "Sh/90", "SoT/90", "npg/90", "xG/90", "SoT%", "G/SoT", "goals", "xGOP/90", "G/Sh","pAdjxG/90","Distance"],
    "Aerial": ["headersWon/90", "headersWon%"],
    "Ball Retention": ["touches/90", "Dispossessed/90", "Mis/90", "sPass%", "ballRecoveries/90"],
}

category_palette = ["#2E4374", "#1A78CF", "#D70232", "#FF9300", "#44C3A1", "#CA228D", "#E1C340", "#7575A9", "#9DDFD3"]
category_by_param = {}
for i, (cat, stats_list) in enumerate(pizza_plot_categories.items()):
    for stat in stats_list:
        category_by_param[stat] = category_palette[i % len(category_palette)]

greens_hex = ["#1fa44b"]*6 + ["#158940", "#0e6d31", "#085222", "#053617"]

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
    "pAdjprogPasses/90" : "Posession Adjusted Progressive Passes/90",
    "pAdjxG/90" : "Possession Adjusted Expected Goals/90",
    "pAdjtb/90" : "Possession Adjusted Through Balls/90",
    "pAdjtouches/90" : "Possession Adjusted Touches/90",
    "pAdjxAG/90" : "Possession Adjusted Expected Assisted Goals/90",
    "pAdjclearances/90" : "Possession Adjusted Clearances/90",
    "pAdjshotsBlocked/90" : "Possession Adjusted Shots Blocked/90",
    "pAdjprogCarries/90" : "Possession Adjusted Progressive Carries/90",
    "pAdjtackles/90" : "Possession Adjusted Tackles/90",
    "Distance" : "Average Shot Distance (M)"
}

# ===== Weighted role definitions =====
position_weights = {
    "Water Carrier": {
        "progCarries/90": 1.2, "thirdCarries/90": 1.1, "passesBlocked/90": 1.0,
        "pAdjtackles/90": 0.9, "pAdjinterceptions/90": 0.8, "pAdjclearances/90": 0.7, "pass%": 1.1
    },

    "Progresser": {
        "pAdjprogCarries/90": 1,
        "pAdjprogPasses/90": 1.8, "lPass%": 1.2, "Succ/90": 1.1, "PPA/90": 1.3,
    },

    "CB - Ball-Playing": {
        "progPasses/90": 1.7, "pass%": 1.9,
        "headersWon%": 1.4, "headersWon/90": 1.3,
        "interceptions/90": 1.3, "pAdjclearances/90": -0.1, "tackles/90": 1.1,
        "tackleSuccessRate": 1.8, "progCarryDist/90": 1.5, "lPass%": 1.5, "pAdjtacklesWon/90": 1.7
    },
    "CB - Stopper": {
        "tackles/90": 1.8, "Tkl+Int/90": 1.7, "clearances/90": 1.6,
        "headersWon/90": 1.6, "headersWon%": 1.5, "Blocked/90": 1.4,
        "interceptions/90": 1.3, "passesBlocked/90": 1.0, "tackleSuccessRate": 1.95
    },
    "CB - Sweeper": {
        "interceptions/90": 1.7, "clearances/90": 1.6, "tackles/90": 1.4,
        "progPasses/90": 1.4, "compPass/90": 1.3, "headersWon%": 1.2,
        "ballRecoveries/90": 1.5, "tackleSuccessRate": 1.7
    },
    "FB/WB - Overlapping Attacker": {
        "crosses/90": 1.9, "compCross/90": 1.8, "tackleSuccessRate": 1.7,
        "assists/90": 1.6, "xAG/90": 1.5, "progCarries/90": 1.4,
        "PPA/90": 1.3, "thirdCarries/90": 1.2, "tackles/90": 1.4
    },
    "FB/WB - Inverted": {
        "progPasses/90": 1.7, "compPass/90": 1.6, "xAG/90": 1.5, "kp/90": 1.4,
        "progCarries/90": 1.4, "thirdCarries/90": 1.3, "tackleSuccessRate": 1.2,
        "interceptions/90": 1.1, "goals/90": 1.0
    },
    "FB - Defensive": {
        "tackles/90": 1.6, "interceptions/90": 1.5, "clearances/90": 1.6,
        "passesBlocked/90": 1.5, "Blocked/90": 1.6, "defThirdTackles/90": 1.8,
        "ballRecoveries/90": 1.2, "sPass%": 1.1, "Dispossessed/90": 1.0,
        "Mis/90": 1.0, "pass%": 1.0, "thirdPasses/90": 1.0, "crosses/90": 1.3,
        "dribbledPast/90": 0.6
    },
    "DM - Ball Winner": {
        "tackles/90": 1.9, "Tkl+Int/90": 1.8, "interceptions/90": 1.7,
        "ballRecoveries/90": 1.6, "passesBlocked/90": 1.5, "clearances/90": 1.6,
        "progCarries/90": 1.2, "progPasses/90": 1.1, "shotsBlocked/90": 1.3,
        "tackleSuccessRate": 1.4, "defPenTouches/90": 1.3, "defThirdTackles/90": 1.5
    },
    "DM - Deep-Lying Playmaker": {
        "compPass/90": 1.9, "progPasses/90": 1.8, "attPass/90": 1.9,
        "xA/90": 1.6, "thirdPasses/90": 1.5, "pass%": 1.7, "kp/90": 1.4, "tb/90": 1.6,
        "interceptions/90": 1.2, "tackleSuccessRate": 1.2, "def3rdTouches/90": 1.5, "progCarries/90": 1.3
    },
    "CM - Box-to-Box Engine": {
        "Tkl+Int/90": 1.7, "progCarries/90": 1.7, "ballRecoveries/90": 1.6,
        "tackles/90": 1.6, "progPasses/90": 1.5, "thirdCarries/90": 1.4, "xA/90": 1.3, "xG/90": 1.2
    },
    "CM - Shuttler/Link Player": {
        "compPass/90": 1.7, "progPasses/90": 1.6, "Carries/90": 1.6,
        "thirdCarries/90": 1.2, "pass%": 1.5, "tackles/90": 1.5,
        "interceptions/90": 1.3, "ballRecoveries/90": 1.3
    },
    "CM - Mezzala": {
        "progCarries/90": 1.8, "thirdCarries/90": 1.7, "xA/90": 1.6,
        "Sh/90": 1.3, "attPass/90": 1.4, "PPA/90": 1.3, "kp/90": 1.3,
        "tackles/90": 1.2, "goals/90": 1.1, "mid3rdTouches/90": 1.7, "midThirdTackles/90": 1.5
    },
    "AM - Classic 10": {
        "xAG/90": 1.9, "kp/90": 1.8, "SCA90": 1.7, "GCA90": 1.6,
        "PPA/90": 1.8, "assists/90": 1.5, "thirdPasses/90": 1.4,
        "progPasses/90": 1.2, "tb/90": 1.3, "goals/90": 1.2
    },
    "ST - Target Man": {
        "headersWon/90": 1.9, "headersWon%": 1.9,
        "assists/90": 0.4, "xA/90": 1.2, "goals/90": 1.7,
        "npg/90": 1.8, "progPasses/90": 0.2
    },
    "ST - Poacher": {
        "xG/90": 1.35, "goals/90": 0.3, "npg/90": 1.6, "G/Sh": 1.7,
        "SoT/90": 1.0, "headersWon%": 1.4, "goals": 1.9, "progCarries/90": -0.4, "Distance": 0.6
    },
    "ST - Complete Forward": {
        "goals/90": 1.7, "npg/90": 1.7, "assists/90": 1.6, "xA/90": 1.6,
        "headersWon/90": 1.5, "progCarries/90": 1.5, "Sh/90": 1.4, "kp/90": 1.4, "goals": 1.2
    },
    "Winger - Classic": {
        "crosses/90": 1.9, "compCross/90": 1.8, "assists/90": 1.7, "xA/90": 1.7,
        "PPA/90": 1.6, "takeOnsAtt/90": 1.5, "progCarries/90": 1.4, "tackleSuccessRate": 1.2
    },
    "Winger - Inverted": {
        "goals/90": 1.8, "xG/90": 1.8, "Sh/90": 1.7, "progCarries/90": 1.6,
        "thirdCarries/90": 1.6, "xA/90": 1.5, "takeOnsAtt/90": 1.6, "SoT/90": 1.4, "kp/90": 1.3
    },
    "AM - Shadow Striker": {
        "xG/90": 1.9, "goals/90": 1.9, "npg/90": 1.8, "Sh/90": 1.7,
        "progCarries/90": 1.6, "SCA90": 1.5, "thirdCarries/90": 1.5, "SoT/90": 1.5, "xAG/90": 1.2
    }
}

position_adjustments = {
    "CB - Ball-Playing": lambda x: min(x * 1.03, 100),
    "CB - Stopper":      lambda x: min(x * 1.05, 100),
    "FB/WB - Overlapping Attacker": lambda x: min(x * 1.04, 100),
    "DM - Ball Winner":  lambda x: min(x * 1.06, 100),
    "DM - Deep-Lying Playmaker": lambda x: min(x * 1.07, 100),
    "ST - Target Man":   lambda x: min(x * 1.05, 100),
    "Winger - Inverted": lambda x: min(x * 1.03, 100)
}

# --- Catalogs: weighted roles + category pizzas (base)
archetype_params = {role: list(weights.keys()) for role, weights in position_weights.items()}
category_archetypes = {f"{cat} Pizza": stats for cat, stats in pizza_plot_categories.items()}
BASE_ARCHETYPES = {**archetype_params, **category_archetypes}


# ---------- Custom Archetype Persistence ----------
ARCHETYPE_STORE = os.path.join(APP_DIR, "custom_archetypes.json")

@st.cache_data(show_spinner=False)
def load_archetype_store() -> dict:
    """
    Returns { archetype_name: { "stats": [col,...], "weights": {col: float, ...} }, ... }
    """
    if os.path.isfile(ARCHETYPE_STORE):
        try:
            with open(ARCHETYPE_STORE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}

def write_archetype_store(store: dict) -> None:
    try:
        with open(ARCHETYPE_STORE, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_archetype_params_full() -> dict[str, list[str]]:
    """
    Single source of truth:
    - Start with BASE_ARCHETYPES (weighted roles + category pizzas)
    - Merge persisted customs from file (suffix ' (Saved)' on name collision)
    - Merge session customs from st.session_state['custom_arches'] (override or add)
    Also merges custom weights into position_weights so scoring works everywhere.
    """
    arch = dict(BASE_ARCHETYPES)

    # 1) From disk
    store = load_archetype_store()
    for name, payload in (store or {}).items():
        stats = [s for s in (payload.get("stats") or []) if isinstance(s, str)]
        if not stats:
            continue
        final_name = name if name not in arch else f"{name} (Saved)"
        arch[final_name] = stats
        wmap = {k: float(v) for k, v in (payload.get("weights") or {}).items() if isinstance(k, str)}
        if wmap:
            position_weights[final_name] = wmap  # add/override weights for this custom role

    # 2) From this session
    sess = st.session_state.get("custom_arches", {})
    if isinstance(sess, dict):
        for name, payload in sess.items():
            stats = [s for s in (payload.get("stats") or []) if isinstance(s, str)]
            if not stats:
                continue
            arch[name] = stats  # session overrides any same-name
            wmap = {k: float(v) for k, v in (payload.get("weights") or {}).items()}
            if wmap:
                position_weights[name] = wmap

    return arch


# =========================
# ======= UI HELPERS ======
# =========================

st.set_page_config(page_title="willumanalytics", layout="wide")

@st.cache_data
def load_csvs(base_folder: str, which: str):
    files = []
    if which == "BigDB_ALL.csv (Minor Leagues)":
        files = [os.path.join(base_folder, "BigDB_ALL.csv")]
    elif which == "BigDB.csv (Big 5 European Leagues)":
        files = [os.path.join(base_folder, "BigDB.csv")]
    else:
        files = [os.path.join(base_folder, "BigDB_ALL.csv"),
                 os.path.join(base_folder, "BigDB.csv")]
    dfs = []
    for p in files:
        if os.path.isfile(p) or base_folder == "":
            try:
                dfs.append(pd.read_csv(p if base_folder != "" else os.path.basename(p)))
            except FileNotFoundError:
                st.warning(f"File not found: {p if base_folder != '' else os.path.basename(p)}")
        else:
            st.warning(f"File not found: {p}")
    if not dfs:
        st.stop()
    df = pd.concat(dfs, ignore_index=True)

    df['Age'] = pd.to_numeric(df.get('Age', np.nan), errors='coerce')
    df['Age_num'] = df['Age'].round().astype('Int64')
    return df

def apply_hover_style(fig):
    fig.update_layout(
        hoverlabel=dict(
            bgcolor=HOVER_BG,
            font=dict(family=FONT_FAMILY, size=13, color="#000")
        )
    )
    fig.update_layout(font=dict(family=FONT_FAMILY))

def _positions_suffix(include_age: bool = True, leagues: list[str] | None = None) -> str:
    parts = []
    pos = st.session_state.get("pos_filter", [])
    if pos:
        parts.append(f"Positions: {', '.join(pos)}")
    if include_age:
        max_age = st.session_state.get("max_age_filter", None)
        if max_age is not None:
            parts.append(f"Max Age ≤ {int(max_age)}")
    if leagues:
        parts.append(f"Leagues: {', '.join(leagues)}")
    return f" | {' • '.join(parts)}" if parts else ""

def _with_pos_filter(title: str, include_age: bool = True, leagues: list[str] | None = None) -> str:
    return f"{title}{_positions_suffix(include_age=include_age, leagues=leagues)}"

def scatter_labels_and_styles(dfx: pd.DataFrame, x_col: str, y_col: str, search_name: str | None):
    label_set = set()
    if not dfx.empty:
        for col, fn in [(x_col, pd.DataFrame.nlargest), (x_col, pd.DataFrame.nsmallest),
                        (y_col, pd.DataFrame.nlargest), (y_col, pd.DataFrame.nsmallest)]:
            try:
                if fn is pd.DataFrame.nlargest:
                    picks = dfx.nlargest(3, col)['Player'].tolist()
                else:
                    picks = dfx.nsmallest(3, col)['Player'].tolist()
                label_set.update(picks)
            except Exception:
                pass

    highlight = None
    if search_name:
        row = find_player_row(dfx, search_name)
        if row is not None:
            highlight = row['Player']
            label_set.add(highlight)

    if highlight:
        marker_colors = np.where(dfx['Player'] == highlight, "#B80019", "#B8B8B8")
        marker_sizes  = np.where(dfx['Player'] == highlight, 14, 8)
        marker_lines  = np.where(dfx['Player'] == highlight, "#000000", "#666666")
    else:
        marker_colors = ["#6FA7D6"] * len(dfx)
        marker_sizes  = [9] * len(dfx)
        marker_lines  = ["#333333"] * len(dfx)

    text_labels = dfx['Player'].where(dfx['Player'].isin(label_set), "")
    return text_labels, marker_colors, marker_sizes, marker_lines, highlight

def filter_by_minutes(df, min_minutes):
    df = df.copy()
    df['Mins'] = pd.to_numeric(df['Mins'].astype(str).str.replace(',', ''), errors='coerce')
    return df[df['Mins'] >= min_minutes]

def position_relative_percentile(df, player_row, stat_col):
    pos_str = player_row.get('Pos', ' ')
    positions = [p.strip() for p in str(pos_str).split(',') if p.strip()]
    if not positions or stat_col not in df.columns:
        return np.nan
    mask = df['Pos'].apply(
        lambda s: any(pos in str(s).split(',') for pos in positions) if isinstance(s, str) else False
    )
    position_df = df[mask]
    if stat_col not in position_df.columns:
        return np.nan
    stat_vals = position_df[stat_col].replace([np.inf, -np.inf], np.nan).dropna()
    player_val = player_row.get(stat_col, np.nan)
    if pd.isnull(player_val) or stat_vals.empty:
        return np.nan
    return round(stats.percentileofscore(stat_vals, player_val, kind='mean'), 2)

def calculate_role_score(player_row, role_stats, df, role_name=None):
    if not role_stats:
        return 50.0
    pos_str = player_row.get('Pos', ' ')
    player_positions = [p.strip() for p in str(pos_str).split(',')] if isinstance(pos_str, str) else []
    pos_filtered_db = df[df['Pos'].apply(
        lambda s: any(pos in str(s).split(',') for pos in player_positions) if isinstance(s, str) else False
    )] if player_positions else df

    percentiles, weights = [], []
    for stat in role_stats:
        if stat not in pos_filtered_db.columns:
            continue
        stat_vals = pos_filtered_db[stat].replace([np.inf, -np.inf], np.nan).dropna()
        player_val = player_row.get(stat, np.nan)
        if pd.notnull(player_val) and not stat_vals.empty:
            pctl = stats.percentileofscore(stat_vals, player_val, kind='mean')
            weight = position_weights.get(role_name, {}).get(stat, 1.0)
            percentiles.append(pctl)
            weights.append(weight)
    if not percentiles:
        return 50.0
    role_score = np.average(percentiles, weights=weights) if len(percentiles) >= 3 else np.mean(percentiles)
    role_score = position_adjustments.get(role_name, lambda x: x)(role_score)
    return float(np.clip(round(role_score, 1), 0, 100))

def calculate_custom_archetype_score(player_row, stat_cols, df, weight_map: dict[str, float] | None = None):
    if not stat_cols:
        return np.nan
    weight_map = weight_map or {}
    percentiles, weights = [], []
    for s in stat_cols:
        if s not in df.columns:
            continue
        pctl = position_relative_percentile(df, player_row, s)
        if pd.notnull(pctl):
            percentiles.append(pctl)
            weights.append(float(weight_map.get(s, 1.0)))
    if not percentiles:
        return np.nan
    try:
        if len(percentiles) >= 3 and any(w != 0 for w in weights):
            score = np.average(percentiles, weights=weights)
        else:
            score = float(np.mean(percentiles))
    except Exception:
        score = float(np.mean(percentiles))
    return float(np.clip(round(score, 2), 0, 100))

def get_contrast_text_color(hex_color):
    r, g, b = mcolors.hex2color(hex_color)
    brightness = (r*299 + g*587 + b*114) * 255 / 1000
    return "#000000" if brightness > 140 else "#F2F2F2"

def break_label(label, max_len=24):
    if label.endswith("/90"):
        before = label[:-3].rstrip()
        if len(label) <= max_len:
            return label
        words, lines, current = before.split(" "), [], ""
        for w in words:
            check_len = len(current + (" " if current else "") + w + "/90")
            if len(lines) == 0 and check_len > max_len:
                if current: lines.append(current)
                current = w
            elif len(current + (" " if current else "") + w) > max_len:
                lines.append(current); current = w
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
                lines.append(current); current = w
            else:
                current += (" " if current else "") + w
        if current: lines.append(current)
        return "\n".join(lines)

# ---- Season folder helpers ----
def _list_season_dirs(root: str = "") -> list[str]:
    """
    Return season-like subfolders under working dir, e.g. ['25_26','24_25'] (sorted newest first).
    """
    root = root or os.getcwd()
    try:
        seasons = [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and re.fullmatch(r"\d{2}_\d{2}", d)
        ]
        seasons.sort(reverse=True)  # e.g., '25_26' before '24_25'
        return seasons
    except Exception:
        return []

def _season_label(folder_name: str) -> str:
    """
    Convert '25_26' -> '25/26 DBs' for nicer display.
    """
    if re.fullmatch(r"\d{2}_\d{2}", folder_name):
        return f"{folder_name[:2]}/{folder_name[3:]} DBs"
    return folder_name

def _season_choice_map(root: str = "") -> tuple[list[str], dict[str, str]]:
    """
    Build display choices + map display -> absolute folder path (or "" for current folder).
    """
    root = root or os.getcwd()
    season_dirs = _list_season_dirs(root)
    if not season_dirs:
        # fallback to just the working directory
        return ["Current Folder"], {"Current Folder": ""}

    choices = [_season_label(s) for s in season_dirs]
    mapping = { _season_label(s): os.path.join(root, s) for s in season_dirs }
    return choices, mapping

def league_strip_prefix(comp):
    for prefix in ('eng ', 'it ', 'es ', 'de ', 'fr '):
        if str(comp).lower().startswith(prefix):
            return comp[len(prefix):]
    return comp

def fig_to_png_bytes_plotly(fig):
    return fig.to_image(format="png", scale=2)

def fig_to_png_bytes_matplotlib(fig):
    buf = BytesIO()
    try:
        fig.canvas.draw()
    except Exception:
        pass
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def set_mode(new_mode: str):
    st.session_state["mode"] = new_mode

def _custom_arch_json_payload() -> dict:
    """Return a JSON-serializable snapshot of this user's custom archetypes (from session)."""
    arches = st.session_state.get("custom_arches", {})
    return {
        "archetypes": [
            {"name": name,
             "stats": data.get("stats", []),
             "weights": {k: float(v) for k, v in (data.get("weights", {}) or {}).items()}
            }
            for name, data in arches.items()
        ]
    }

def export_custom_archetypes_button(label="Download my custom archetypes (JSON)"):
    payload = _custom_arch_json_payload()
    st.download_button(
        label,
        data=json.dumps(payload, indent=2).encode("utf-8"),
        file_name="my_archetypes.json",
        mime="application/json",
        key="dl_custom_arches"
    )

def import_custom_archetypes_uploader(df_cols: set[str]):
    """
    File uploader UI: merge valid archetypes into this user's session only.
    We drop unknown stats (not in df) and coerce weights to floats.
    """
    up = st.file_uploader("Load custom archetypes (JSON)", type=["json"], key="upload_custom_arches")
    if up is None:
        return
    try:
        data = json.load(up)
        added = 0
        for item in data.get("archetypes", []):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            stats = [s for s in (item.get("stats") or []) if s in df_cols]
            w_raw = item.get("weights") or {}
            weights = {k: float(v) for k, v in w_raw.items() if k in stats}
            if stats:
                st.session_state["custom_arches"][name] = {"stats": stats, "weights": weights}
                added += 1
        if added:
            st.success(f"Loaded {added} archetype(s) into **your session**.")
        else:
            st.warning("No valid archetypes found in the JSON (unknown stat names or empty entries).")
    except Exception as e:
        st.error(f"Couldn't read JSON: {e}")

def stats_for_archetype(arch_name: str) -> list[str]:
    """Get stat list for either a built-in archetype or a user custom one."""
    arch_map = get_archetype_params_full()
    return arch_map.get(arch_name, [])


# =================
# PERCENTILES (GLOBAL vs POSITIONAL) + CACHED ROLE SCORES
# =================

@st.cache_data
def _compute_percentiles_table(dfin: pd.DataFrame, selected_stats: list[str], baseline: str = "positional") -> pd.DataFrame:
    """
    Compute percentiles for selected_stats.
    baseline="positional": each player's percentile is vs players who share ANY of their listed positions.
    baseline="global": percentile is vs the entire provided dfin.
    Returns columns pct_<stat>.
    """
    if not selected_stats or dfin.empty:
        return pd.DataFrame(index=dfin.index)

    dfin = dfin.copy()
    # Ensure numeric & clean
    for s in selected_stats:
        if s in dfin.columns:
            dfin[s] = pd.to_numeric(dfin[s], errors='coerce').replace([np.inf, -np.inf], np.nan)

    pct_df = pd.DataFrame(index=dfin.index)

    if baseline == "global":
        # Vectorized: rank-average / count * 100
        for s in selected_stats:
            if s not in dfin.columns:
                continue
            series = dfin[s]
            n = series.notna().sum()
            if n == 0:
                pct_df[f"pct_{s}"] = np.nan
                continue
            ranks = series.rank(method="average", na_option="keep")
            pct_df[f"pct_{s}"] = (ranks / n) * 100.0
        return pct_df

    # baseline == "positional" (row-wise, fallback if needed by callers)
    for idx, row in dfin.iterrows():
        rec = {}
        for s in selected_stats:
            rec[f"pct_{s}"] = position_relative_percentile(dfin, row, s)
        pct_df.loc[idx, list(rec.keys())] = list(rec.values())
    return pct_df

def _pos_signature(s: str) -> str:
    toks = [t.strip() for t in str(s).split(',') if t.strip()]
    return ",".join(sorted(set(toks))) if toks else ""

@st.cache_data(show_spinner=False)
def _precompute_positional_percentiles(df: pd.DataFrame, stats_needed: tuple[str, ...]) -> pd.DataFrame:
    """
    Vectorized positional percentiles for all players and given stats.
    Returns a DataFrame with columns exactly = stats_needed, values in 0..100.
    """
    if not stats_needed or df.empty:
        return pd.DataFrame(index=df.index)

    dfx = df.copy()
    dfx["PosSig"] = dfx["Pos"].apply(_pos_signature)

    for s in stats_needed:
        if s in dfx.columns:
            dfx[s] = pd.to_numeric(dfx[s], errors="coerce").replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame(index=dfx.index, columns=list(stats_needed), dtype=float)

    groups = dfx.groupby("PosSig").groups  # dict: possig -> index (Int64Index)
    for possig, idx in groups.items():
        if len(idx) == 0:
            continue
        idx = pd.Index(idx)

        tokens = set(possig.split(",")) if possig else set()
        # players who share ANY of these position tokens
        mask = dfx["Pos"].apply(lambda s: any(t in str(s).split(",") for t in tokens)) if tokens else pd.Series(False, index=dfx.index)

        for s in stats_needed:
            if s not in dfx.columns:
                continue
            series = dfx.loc[mask, s]
            n = series.notna().sum()
            if n == 0:
                continue
            ranks = series.rank(method="average", na_option="keep")
            pcts = (ranks / n) * 100.0

            # assign only to rows in this exact signature group
            idx_here = series.index.intersection(idx)
            out.loc[idx_here, s] = pcts.loc[idx_here]

    return out

@st.cache_data(show_spinner=False)
def compute_role_scores_cached(df: pd.DataFrame, role_name: str) -> pd.Series:
    """
    Fast, cached RoleScore for ALL players for a given role/archetype (positional baseline).
    Works for built-ins and custom archetypes.
    """
    arch_map = get_archetype_params_full()
    stats_list = tuple(arch_map.get(role_name, []))
    if not stats_list:
        return pd.Series(50.0, index=df.index)

    pcts = _precompute_positional_percentiles(df, stats_list)

    # If it's a weighted built-in role (or custom with weights), use weights; otherwise equal weights
    weights_map = position_weights.get(role_name, {})
    w = pd.Series({s: float(weights_map.get(s, 1.0)) for s in stats_list}, dtype=float)
    P = pcts.reindex(columns=stats_list)

    mask = ~P.isna()
    num = (P * w).sum(axis=1, skipna=True)
    den = (mask * w).sum(axis=1)
    scores = (num / den).where(den > 0)

    # Optional built-in adjustment
    adj = position_adjustments.get(role_name, lambda x: x)
    scores = scores.apply(lambda x: adj(x) if pd.notnull(x) else np.nan).clip(0, 100)

    # Keeper / no-shooting-data rule
    shooting_cols = [c for c in pizza_plot_categories.get("Shooting", []) if c in df.columns]
    if shooting_cols:
        no_shoot = df[shooting_cols].apply(pd.to_numeric, errors="coerce").isna().all(axis=1)
        scores = scores.mask(no_shoot, 0.0)

    return scores

@st.cache_data(show_spinner=False)
def precompute_role_scores(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Cached, vectorized scores for all archetypes (built-ins + custom).
    - Weighted roles use position_weights
    - Everything else (category pizzas + custom without weights) = equal-weight average
    """
    if df_in.empty:
        return pd.DataFrame(index=df_in.index)

    arch_map = get_archetype_params_full()
    needed_stats = sorted({s for stats in arch_map.values() for s in stats})
    pcts_pos = _precompute_positional_percentiles(df_in, tuple(needed_stats))

    scores = {}

    # Weighted roles/customs
    for role, weights in position_weights.items():
        stats_list = [s for s in weights.keys() if s in pcts_pos.columns]
        if not stats_list:
            continue
        W = pd.Series({s: float(weights[s]) for s in stats_list}, dtype=float)
        P = pcts_pos[stats_list]

        mask = ~P.isna()
        num = (P * W).sum(axis=1, skipna=True)
        den = (mask * W).sum(axis=1)
        role_score = (num / den).where(den > 0)

        adj_fn = position_adjustments.get(role, lambda x: x)
        role_score = role_score.apply(lambda x: adj_fn(x) if pd.notnull(x) else np.nan).clip(0, 100)

        shooting_cols = [c for c in pizza_plot_categories.get("Shooting", []) if c in df_in.columns]
        if shooting_cols:
            no_shoot = df_in[shooting_cols].apply(pd.to_numeric, errors="coerce").isna().all(axis=1)
            role_score = role_score.mask(no_shoot, 0.0)

        scores[role] = role_score

    # Unweighted roles (anything in arch_map not present in position_weights)
    unweighted_keys = [k for k in arch_map.keys() if k not in position_weights]
    for key in unweighted_keys:
        stats_list = [s for s in arch_map.get(key, []) if s in pcts_pos.columns]
        if not stats_list:
            scores[key] = np.nan
            continue

        P = pcts_pos[stats_list]
        cat_score = P.mean(axis=1, skipna=True)

        raw_cols = [s for s in arch_map.get(key, []) if s in df_in.columns]
        if raw_cols:
            no_cat = df_in[raw_cols].apply(pd.to_numeric, errors="coerce").isna().all(axis=1)
            cat_score = cat_score.mask(no_cat, 0.0)

        scores[key] = cat_score.clip(0, 100)

    return pd.DataFrame(scores, index=df_in.index)


# =================
# PLOT FUNCTIONS
# =================

def show_percentile_bar_chart(player_row, stat_cols, df, role_name):
    # raw values + percentiles (positional)
    vals = [player_row.get(s, np.nan) for s in stat_cols]
    percentiles = [
        position_relative_percentile(df, player_row, s) if np.isfinite(v) else 0
        for s, v in zip(stat_cols, vals)
    ]
    labels = [stat_display_names.get(s, s) for s in stat_cols]

    # ---- NEW: richer hovertext with raw values ----
    hover = []
    for lbl, v, p in zip(labels, vals, percentiles):
        try:
            vtxt = f"{float(v):.2f}" if pd.notnull(v) else "—"
        except Exception:
            vtxt = "—"
        hover.append(f"<b>{lbl}</b><br>Value: {vtxt}<br>Percentile: {p:.1f}%")

    bar = go.Bar(
        x=percentiles,
        y=labels,
        orientation='h',
        text=[f"{p:.1f}%" for p in percentiles],
        textposition="auto",
        marker=dict(color=percentiles, colorscale="RdYlGn"),
        hovertext=hover,
        hoverinfo="text"
    )

    title = _with_pos_filter(
        f"{player_row['Player']} — {role_name}<br><sup>Percentiles vs same-position players</sup>"
    )
    fig = go.Figure([bar])
    fig.update_layout(
        title=dict(text=title, font=dict(color="#000", family=FONT_FAMILY)),
        plot_bgcolor=POSTER_BG, paper_bgcolor=POSTER_BG,
        xaxis=dict(title="Percentile", range=[0, 100], gridcolor="#eee", color="#000", tickfont=dict(color="#000"), linecolor="#000"),
        yaxis=dict(automargin=True, color="#000", tickfont=dict(color="#000"), linecolor="#000"),
        template="simple_white", height=60 + 32*len(labels),
        margin=dict(l=200, r=40, t=80, b=40),
    )
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    df_out = pd.DataFrame({"Stat": labels, "Percentile": percentiles, "Value": vals})
    return fig, df_out

def _player_pcts_global_then_positional(df_src: pd.DataFrame, player_row: pd.Series, stat_cols: list[str]) -> list[float]:
    """Global first; if positions are filtered in UI, recalc positionally."""
    idx = player_row.name
    # Global (cached)
    gdf = _compute_percentiles_table(df_src, stat_cols, baseline="global")
    pcts = []
    for s in stat_cols:
        v = np.nan
        if idx in gdf.index and f"pct_{s}" in gdf.columns:
            v = gdf.at[idx, f"pct_{s}"]
        pcts.append(0 if pd.isna(player_row.get(s, np.nan)) else (float(v) if pd.notnull(v) else 0.0))

    # If user applied a position filter in the sidebar, switch to positional
    if st.session_state.get("pos_filter"):
        posp = _precompute_positional_percentiles(df_src, tuple(stat_cols))
        p2 = []
        for s in stat_cols:
            vv = posp.loc[idx, s] if (idx in posp.index and s in posp.columns) else np.nan
            p2.append(0 if pd.isna(player_row.get(s, np.nan)) else (float(vv) if pd.notnull(vv) else 0.0))
        pcts = p2
    return [float(np.clip(x, 0, 100)) for x in pcts]

def show_pizza(player_row, stat_cols, df_filtered, role_name, lightmode=False, toppings=True, show_role_score=False):
    raw_vals = [player_row.get(s, float('nan')) for s in stat_cols]
    # Global -> (optional) positional
    pcts = _player_pcts_global_then_positional(df_filtered, player_row, stat_cols)

    slice_colors = [category_by_param.get(s, "#2E4374") for s in stat_cols]
    text_colors = [get_contrast_text_color(c) for c in slice_colors]
    display_params = [break_label(stat_display_names.get(p, p), 20) for p in stat_cols]

    bg = "#f1ffcd" if lightmode else "#222222"
    param_color = "#222222" if lightmode else "#fffff0"

    baker = PyPizza(
        params=display_params,
        background_color=bg,
        straight_line_color="#000000",
        straight_line_lw=.3,
        last_circle_color="#000000",
        last_circle_lw=1,
        other_circle_lw=0,
        inner_circle_size=0.30
    )
    fig, ax = baker.make_pizza(
        pcts,
        alt_text_values=raw_vals,
        figsize=(10, 11),
        color_blank_space="same",
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=slice_colors,
        blank_alpha=0.4,
        kwargs_slices=dict(edgecolor="#000000", zorder=2, linewidth=1),
        kwargs_params=dict(color=param_color, fontsize=13, fontproperties=font_normal, va="center"),
        kwargs_values=dict(
            color="#222222" if lightmode else "#fffff0",
            fontsize=12, fontproperties=font_normal, zorder=3,
            bbox=dict(edgecolor="#000000", facecolor="cornflowerblue", boxstyle="round,pad=0.2", lw=1)
        )
    )

    if toppings:
        club = player_row.get("Squad", "?")
        nationality = player_row.get("Nation", "?")
        age_val = player_row.get("Age", np.nan)
        age_txt = f"{int(round(age_val))}" if pd.notnull(age_val) else "?"
        mins = player_row.get("Mins", 0)
        season = "2024/25"
        header_color = "#222222" if lightmode else "#fffff0"
        fig.text(0.5, 0.985, player_row['Player'], ha='center', va='top', fontsize=22,
                 fontweight='bold', color=header_color, fontproperties=font_normal)
        fig.text(0.5, 0.952, f"{club} | {nationality} | {age_txt} | {season}", ha='center', va='top', fontsize=15,
                 color=header_color, fontproperties=font_normal)
        suffix = _positions_suffix(include_age=False)
        fig.text(0.5, 0.928, f"Role: {role_name} | Minutes played: {mins}{suffix}",
            ha='center', va='top', fontsize=12, color=header_color, fontproperties=font_normal)
        fig.text(0.5, 0.01, "willumanalytics",
                 ha='center', va='bottom', fontsize=9, color=("#666" if lightmode else "#CCC"),
                 fontproperties=font_normal, alpha=0.85)

        if show_role_score:
            # Use cached precomputed scores first (works for roles *and* category pizzas)
            score = None
            try:
                rs_df = precompute_role_scores(df_filtered)
                if role_name in rs_df.columns and player_row.name in rs_df.index:
                    sc = rs_df.loc[player_row.name, role_name]
                    if pd.notnull(sc):
                        score = float(sc)
            except Exception:
                score = None

            if score is None:
                # Fallback on-the-fly
                role_stats = get_archetype_params_full().get(role_name, [])
                score = calculate_role_score(player_row, role_stats, df_filtered, role_name)

            box_color = "#1A78CF" if score >= 70 else "#D70232" if score < 40 else "#FF9300"
            fig.text(
                0.82, 0.90, f"Role Score: {score:.1f}",
                ha="center", va="top",
                fontsize=14, fontproperties=font_normal, fontweight="bold",
                color=("#f1ffcd" if box_color != "#FF9300" else "#222"),
                bbox=dict(boxstyle="round,pad=0.25", facecolor=box_color, edgecolor="#000000", linewidth=1),
                zorder=5
            )

    st.pyplot(fig, clear_figure=False)
    return fig

def plot_role_leaderboard(df_filtered, role_name, role_stats, title_suffix: str = ""):
    # Use fast cached series
    try:
        dfc = df_filtered.copy()
        dfc['RoleScore'] = compute_role_scores_cached(dfc, role_name)
    except Exception:
        dfc = df_filtered.copy()
        dfc['RoleScore'] = [calculate_role_score(row, role_stats, dfc, role_name) for _, row in dfc.iterrows()]

    top = dfc.nlargest(10, 'RoleScore').reset_index(drop=True)

    pastel_blues = ["#E8F1FE","#DCEBFE","#CFE5FE","#C2DFFE","#B5D9FE",
                    "#A8D2FD","#9BCBFD","#8EC4FD","#81BCFD","#74B4FC"][::-1]
    bar_colors = pastel_blues[-len(top):][::-1]

    def lum(c):
        r,g,b = [x*255 for x in mcolors.hex2color(c)]
        return 0.299*r + 0.587*g + 0.114*b
    bar_text_colors = ['white' if lum(c) < 150 else '#222' for c in bar_colors]

    labels = [f"{row['Player']} • {row['RoleScore']:.1f} • {int(row.get('Age',np.nan)) if pd.notnull(row.get('Age',np.nan)) else '?'} • {row.get('Squad','?')} • {int(row.get('Mins',0)):,} mins" for _, row in top.iterrows()]
    fig = go.Figure([
        go.Bar(
            x=top['RoleScore'],
            y=[f"#{i+1}" for i in range(len(top))],
            orientation='h',
            text=labels,
            textposition='inside',
            insidetextanchor='middle',
            marker=dict(color=bar_colors, line=dict(color='#333', width=1)),
            textfont=dict(color=bar_text_colors, size=13, family=FONT_FAMILY),
            customdata=top.index.values,
            hovertext=[f"{row['Player']} ({row['Squad']})" for _, row in top.iterrows()],
            hoverinfo="text"
        )
    ])
    title_txt = _with_pos_filter(f"Top {role_name}s{title_suffix}")
    fig.update_layout(
        title=dict(text=title_txt, font=dict(color="#000", family=FONT_FAMILY)),
        plot_bgcolor=POSTER_BG, paper_bgcolor=POSTER_BG,
        xaxis=dict(title='Role Suitability Score (0–100)', range=[0,100], gridcolor=POSTER_BG, color="#000", tickfont=dict(color="#000"), linecolor="#000"),
        yaxis=dict(autorange='reversed', showgrid=False, color="#000", tickfont=dict(color="#000"), linecolor="#000"),
        margin=dict(l=120, r=40, t=60, b=40),
        height=600, width=None,
    )
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)
    return top, fig

def show_top_players_by_stat(df, tidy_label, stat_col, title_suffix: str = ""):
    top = df.nlargest(10, stat_col).reset_index(drop=True)
    pastel_blues = ["#E8F1FE","#DCEBFE","#CFE5FE","#C2DFFE","#B5D9FE",
                    "#A8D2FD","#9BCBFD","#8EC4FD","#81BCFD","#74B4FC"][::-1]
    bar_colors = pastel_blues[-len(top):][::-1]

    def lum(c):
        r,g,b = [x*255 for x in mcolors.hex2color(c)]
        return 0.299*r + 0.587*g + 0.114*b
    bar_text_colors = ['white' if lum(c) < 150 else '#222' for c in bar_colors]
    labels = [f"{row['Player']} • {row.get(stat_col,0):.2f} • {int(row.get('Age',np.nan)) if pd.notnull(row.get('Age',np.nan)) else '?'} • {row.get('Squad','?')} • {int(row.get('Mins',0)):,} mins" for _, row in top.iterrows()]

    fig = go.Figure([
        go.Bar(
            x=top[stat_col], y=[f"#{i+1}" for i in range(len(top))], orientation='h',
            text=labels, textposition='inside', insidetextanchor='middle',
            marker=dict(color=bar_colors, line=dict(color='#333', width=1)),
            textfont=dict(color=bar_text_colors, size=13, family=FONT_FAMILY),
            hovertext=[f"{row['Player']} ({row['Squad']})" for _, row in top.iterrows()],
            hoverinfo="text"
        )
    ])
    title_txt = _with_pos_filter(f"Top 10: {tidy_label}{title_suffix}")
    fig.update_layout(
        title=dict(text=title_txt, font=dict(color="#000", family=FONT_FAMILY)),
        plot_bgcolor=POSTER_BG, paper_bgcolor=POSTER_BG,
        xaxis=dict(title=tidy_label, gridcolor=POSTER_BG, color="#000", tickfont=dict(color="#000"), linecolor="#000"),
        yaxis=dict(autorange='reversed', showgrid=False, color="#000", tickfont=dict(color="#000"), linecolor="#000"),
        margin=dict(l=120, r=40, t=60, b=40), height=600, width=None,
    )
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)
    return top, fig

def plot_similarity_and_select(df_filtered, player_row, stat_cols, role_name):
    X = df_filtered[stat_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    player_vec = scaler.transform([player_row[stat_cols].fillna(0)])
    sim = cosine_similarity(player_vec, X_scaled)[0]
    df_filtered = df_filtered.reset_index(drop=True)
    try:
        self_idx = df_filtered.index[df_filtered['Player'] == player_row['Player']][0]
        sim[self_idx] = -1
    except Exception:
        pass
    order = np.argsort(-sim)
    top_idx = order[:10]
    top_players = df_filtered.iloc[top_idx].copy()
    top_players['Similarity'] = sim[top_idx] * 100.0

    bar_fill = "#1fa44b"
    r, g, b = [x*255 for x in mcolors.hex2color(bar_fill)]
    luminance = 0.299*r + 0.587*g + 0.114*b
    inside_text_color = "white" if luminance < 150 else "#222"

    fig = go.Figure([
        go.Bar(
            y=top_players['Player'][::-1],
            x=top_players['Similarity'][::-1],
            orientation='h',
            text=[f"{s:.2f}%" for s in top_players['Similarity'][::-1]],
            textposition='inside',
            marker=dict(color=bar_fill, line=dict(width=1, color='#333')),
            textfont=dict(color=inside_text_color, family=FONT_FAMILY)
        )
    ])
    fig.update_layout(
        title=dict(text=_with_pos_filter(f"Most similar to {player_row['Player']} — {role_name}"),
                   font=dict(color="#000", family=FONT_FAMILY)),
        plot_bgcolor=POSTER_BG, paper_bgcolor=POSTER_BG,
        xaxis=dict(title="Similarity (%)", range=[0,100], gridcolor="#222222", color="#222222", tickfont=dict(color="#000"), linecolor="#000"),
        yaxis=dict(autorange='reversed', color="#000", tickfont=dict(color="#000"), linecolor="#000"),
        template="simple_white", height=500,
    )
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)
    pick = st.selectbox("Pick a player to compare on radar", top_players['Player'].tolist())
    return top_players[top_players['Player'] == pick].iloc[0] if pick else None

def plot_radar_percentiles(base_row, other_row, stat_cols, df_filtered, role_name):
    base_vals = [position_relative_percentile(df_filtered, base_row, s) for s in stat_cols]
    other_vals = [position_relative_percentile(df_filtered, other_row, s) for s in stat_cols]
    labels = [stat_display_names.get(s, s) for s in stat_cols]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=base_vals,  theta=labels, fill='toself', name=f"{base_row['Player']}"))
    fig.add_trace(go.Scatterpolar(r=other_vals, theta=labels, fill='toself', name=f"{other_row['Player']}"))
    fig.update_layout(
        title=dict(text=_with_pos_filter(f"{base_row['Player']} vs {other_row['Player']} — {role_name}"), font=dict(color="#000", family=FONT_FAMILY)),
        polar=dict(
            bgcolor=POSTER_BG,
            radialaxis=dict(visible=True, range=[0, 100], color="#000", gridcolor="#ddd", tickfont=dict(color="#000")),
            angularaxis=dict(color="#000", tickfont=dict(color="#000"))
        ),
        showlegend=True,
        legend=dict(
            font=dict(family=FONT_FAMILY, size=12, color="#000"),
        ),
        plot_bgcolor=POSTER_BG, paper_bgcolor=POSTER_BG,
        height=650,
    )
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)
    return fig

def style_scatter_axes(ax, title_text):
    ax.set_title(title_text, fontsize=14, pad=8, fontproperties=font_normal, color="#000")
    ax.tick_params(colors="#000", labelsize=10)
    ax.xaxis.label.set_color("#000")
    ax.yaxis.label.set_color("#000")
    for spine in ax.spines.values():
        spine.set_color("#000")
        spine.set_linewidth(1)

def pizza_fig_to_array(fig, dpi=220):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    return np.array(img)

def build_pizza_figure(
    player_row, stat_cols, df, role_name,
    lightmode=True, toppings=True, show_role_score=False
):
    raw_vals = [player_row.get(s, float('nan')) for s in stat_cols]
    # Global -> (optional) positional
    pcts = _player_pcts_global_then_positional(df, player_row, stat_cols)

    slice_colors = [category_by_param.get(s, "#2E4374") for s in stat_cols]
    text_colors  = [get_contrast_text_color(c) for c in slice_colors]
    display_params = [break_label(stat_display_names.get(p, p), 20) for p in stat_cols]
    bg = POSTER_BG if lightmode else "#222222"
    param_color = "#222222" if lightmode else "#fffff0"

    baker = PyPizza(
        params=display_params,
        background_color=bg,
        straight_line_color="#000000",
        straight_line_lw=.3,
        last_circle_color="#000000",
        last_circle_lw=1,
        other_circle_lw=0,
        inner_circle_size=0.30
    )
    fig, ax = baker.make_pizza(
        pcts,
        alt_text_values=raw_vals,
        figsize=(8.5, 8.8),
        color_blank_space="same",
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=slice_colors,
        blank_alpha=0.40,
        kwargs_slices=dict(edgecolor="#000000", zorder=2, linewidth=1),
        kwargs_params=dict(color=param_color, fontsize=15, fontproperties=font_normal, va="center"),
        kwargs_values=dict(
            color="#222222" if lightmode else "#fffff0",
            fontsize=12, fontproperties=font_normal, zorder=3,
            bbox=dict(edgecolor="#000000", facecolor="cornflowerblue", boxstyle="round,pad=0.18", lw=1)
        )
    )
    if show_role_score and toppings:
        score = None
        try:
            rs_df = precompute_role_scores(df)
            if role_name in rs_df.columns and player_row.name in rs_df.index:
                sc = rs_df.loc[player_row.name, role_name]
                if pd.notnull(sc):
                    score = float(sc)
        except Exception:
            score = None

        if score is None:
            role_stats = get_archetype_params_full().get(role_name, [])
            score = calculate_role_score(player_row, role_stats, df, role_name)

        box_color = "#1A78CF" if score >= 70 else "#D70232" if score < 40 else "#FF9300"
        fig.text(
            0.86, 0.90, f"Role Score: {score:.1f}",
            ha="center", va="top",
            fontsize=13, fontproperties=font_normal, fontweight="bold",
            color=("#f1ffcd" if box_color != "#FF9300" else "#222"),
            bbox=dict(boxstyle="round,pad=0.2", facecolor=box_color, edgecolor="#000000", linewidth=1),
            zorder=5
        )
    return fig

def build_role_matrix_axes(ax, df_display, df_calc, player_row, role_x, role_y):
    ax.set_facecolor(POSTER_BG)

    dfx = df_display.copy()
    ser_x = compute_role_scores_cached(df_calc, role_x)
    ser_y = compute_role_scores_cached(df_calc, role_y)
    dfx['RoleScore_X'] = ser_x.reindex(dfx.index)
    dfx['RoleScore_Y'] = ser_y.reindex(dfx.index)

    ax.scatter(dfx['RoleScore_X'], dfx['RoleScore_Y'], s=10, edgecolor="#333", linewidth=0.6, alpha=0.85, color="#1f77b4")
    me = dfx[dfx['Player'] == player_row['Player']]
    if not me.empty:
        ax.scatter(me['RoleScore_X'], me['RoleScore_Y'], s=120, edgecolor="#000", linewidth=1.2, color="#B80019", zorder=5)
        ax.text(me['RoleScore_X'].values[0]+1.5, me['RoleScore_Y'].values[0],
                player_row['Player'], fontsize=10, color="#111", fontproperties=font_normal)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel(role_x); ax.set_ylabel(role_y)
    ax.grid(color="#d9e6a6", linewidth=1, alpha=1.0)
    style_scatter_axes(ax, _with_pos_filter(f"Role Matrix: {role_x} vs {role_y}"))

def make_dashboard_poster(player_row, df_display, df_calc, main_role, role_x, role_y, headshot_url=None):
    arch_map = get_archetype_params_full()
    central_stats = arch_map.get(main_role, [])
    central_fig   = build_pizza_figure(player_row, central_stats, df_calc, main_role, lightmode=True, toppings=True, show_role_score=True)
    central_img   = pizza_fig_to_array(central_fig, dpi=220)
    cats = ["Shooting", "Carrying", "Passing"]
    mini_imgs = []
    for c in cats:
        mini_fig = build_pizza_figure(player_row, pizza_plot_categories.get(c, []), df_calc, c, lightmode=True, toppings=False, show_role_score=False)
        mini_imgs.append(pizza_fig_to_array(mini_fig, dpi=220))
    fig = plt.figure(figsize=(12.5, 17), facecolor=POSTER_BG)
    gs = fig.add_gridspec(14, 12)
    ax_hdr = fig.add_subplot(gs[0:2, 0:12]); ax_hdr.axis('off')
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
            ax_img = fig.add_subplot(gs[0:2, 0:2]); ax_img.imshow(im); ax_img.axis('off')
        except Exception:
            pass
    ax_hdr.text(0.02, 0.70, name, fontsize=26, fontweight="bold", fontproperties=font_normal, color="#111")
    ax_hdr.text(0.02, 0.30, f"{club}  ·  {nation}  ·  {pos}  ·  {mins:,} mins  ·  Age {age}",
                fontsize=13.5, fontproperties=font_normal, color="#222")
    ax_center = fig.add_subplot(gs[2:9, 0:8]); ax_center.axis('off'); ax_center.imshow(central_img)
    ax_center.set_title(f"Selected Role: {main_role}", fontsize=14, pad=8, fontproperties=font_normal)
    for i, (img, title) in enumerate(zip(mini_imgs, ["Shooting", "Carrying", "Passing"])):
        ax_m = fig.add_subplot(gs[2 + i*3 : 2 + i*3 + 3, 8:12]); ax_m.axis('off'); ax_m.imshow(img)
        ax_m.set_title(f"{title}:", fontsize=13, pad=6, fontproperties=font_normal, loc="left")
    ax_mat = fig.add_subplot(gs[9:14, 0:12])
    build_role_matrix_axes(ax_mat, df_display, df_calc, player_row, role_x, role_y)
    fig.text(0.01, 0.01, "willumanalytics", fontsize=10, color="#777", fontproperties=font_normal)
    return fig

def render_player_dashboard(player_row, df_display, df_calc):
    if player_row is None:
        st.info("Type or pick a player in the sidebar first.")
        return
    st.markdown("### Player Dashboard (Poster)")
    arch_keys_local = list(get_archetype_params_full().keys())
    default_main = arch_keys_local.index("ST - Target Man") if "ST - Target Man" in arch_keys_local else 0
    default_x = arch_keys_local.index("Winger - Inverted") if "Winger - Inverted" in arch_keys_local else 0
    default_y = arch_keys_local.index("ST - Target Man") if "ST - Target Man" in arch_keys_local else 1
    main_role = st.selectbox("Selected Role (big pizza)", arch_keys_local, index=default_main)
    role_x = st.selectbox("Matrix X role", arch_keys_local, index=default_x)
    role_y = st.selectbox("Matrix Y role", arch_keys_local, index=default_y)
    headshot_url = st.text_input("Optional headshot image URL (for header)", "")
    fig = make_dashboard_poster(player_row, df_display, df_calc, main_role, role_x, role_y, headshot_url or None)
    st.pyplot(fig, clear_figure=False)
    upscale_dpi = st.slider("Download DPI (higher = larger PNG)", 200, 600, 360, step=20)
    buf = BytesIO()
    try:
        fig.canvas.draw()
    except Exception:
        pass
    fig.savefig(buf, format="png", dpi=upscale_dpi, bbox_inches="tight", facecolor=POSTER_BG)
    buf.seek(0)
    st.download_button(
        "Download Poster PNG",
        data=buf,
        file_name=f"{player_row['Player'].replace(' ','_')}_dashboard_poster.png",
        mime="image/png"
    )

# ---- League filter UI ----
def league_filter_ui(dfin, widget_key: str | None = None, return_selection: bool = False):
    """
    Show a single 'Filter by league(s)' multiselect and return the filtered DF.
    - widget_key: unique key per mode (e.g., "mode4", "mode6", "mode14")
    - return_selection: if True, returns (filtered_df, chosen_leagues); else just filtered_df
    """
    dfin = dfin.copy()
    dfin['LeagueName'] = dfin['Comp'].apply(league_strip_prefix)

    leagues = sorted(dfin['LeagueName'].dropna().unique().tolist())

    # Stable, unique key per mode so we can use this function multiple times
    if widget_key is None:
        widget_key = f"league_filter_{st.session_state.get('mode', 'global')}"

    chosen = st.multiselect(
        "Filter by league(s) (optional)",
        leagues,
        key=widget_key
    )

    if chosen:
        dfin = dfin[dfin['LeagueName'].isin(chosen)]

    # persist selection in session if you need it elsewhere
    st.session_state[f"{widget_key}_selected"] = chosen

    dfin = dfin.drop(columns=['LeagueName'], errors='ignore')
    return (dfin, chosen) if return_selection else dfin


# =====================
# ====== SIDEBAR ======
# =====================
MODE_ITEMS = [
    ("Similar", "1"),
    ("Percentiles", "2"),
    ("Pizza", "3"),
    ("Role Leaders", "4"),
    ("Best Roles", "5"),
    ("Stat Leaders", "6"),
    ("Custom Archetype", "7"),
    ("Stat Scatter", "12"),
    ("Role Matrix", "13"),
    ("Player Finder", "14"),
    ("Glossary / Help", "15"),
    ("Head-to-Head Radar", "16"),
]

def dot_nav(mode_labels_to_keys, default_key):
    if "mode" not in st.session_state:
        st.session_state["mode"] = default_key
    st.markdown("""
        <style>
          .mode-label { font-size: 12px; line-height: 1.1; padding-left: .25rem; }
          .mode-label.active { color:#1A78CF; font-weight:600; }
          .mode-label.inactive { color:#666; }
          div.dotbtn > div > button, .dotbtn button {
            border-radius: 999px !important;
            padding: 0.2rem 0.55rem !important;
            min-height: auto !important;
            line-height: 1 !important;
          }
        </style>
    """, unsafe_allow_html=True)

    cols = st.columns(len(mode_labels_to_keys))
    for i, (label, key) in enumerate(mode_labels_to_keys):
        active = (st.session_state["mode"] == key)
        dot_char = "●" if active else "○"
        with cols[i]:
            left, right = st.columns([1, 5])
            with left:
                if st.button(dot_char, key=f"dot_{key}", help=label):
                    set_mode(key)
            with right:
                st.markdown(
                    f"<div class='mode-label {'active' if active else 'inactive'}'>{label}</div>",
                    unsafe_allow_html=True
                )

dot_nav(MODE_ITEMS, default_key="1")
mode = st.session_state["mode"]

st.title("willum's analytics")

st.sidebar.header("Data & Filters")

# No front-end filepath control — use working directory like before.
# Root is working dir; season folders live here as ./24_25, ./25_26, ...
DATA_ROOT = ""  # keep empty to use current working directory

# 1) Pick season folder (shown as "25/26 DBs" etc.)
_season_choices, _season_map = _season_choice_map(DATA_ROOT)

# Prefer the newest (e.g., 25/26) if present
_default_season_idx = 0
try:
    newest_label = _season_label("25_26")
    if newest_label in _season_choices:
        _default_season_idx = _season_choices.index(newest_label)
except Exception:
    pass

season_choice_label = st.sidebar.selectbox(
    "Season",
    _season_choices,
    index=_default_season_idx,
    key="season_select",
    help="Select which season's folder to use (working dir / NN_NN)."
)
BASE = _season_map.get(season_choice_label, "")

# 2) Pick database(s) inside that season folder
db_choice = st.sidebar.selectbox(
    "Database",
    ["BigDB_ALL.csv (Minor Leagues)", "BigDB.csv (Big 5 European Leagues)", "Both"],
    index=2,
    key="db_select"
)

# Load the raw dataset for this season+database choice
df = load_csvs(BASE, db_choice)

st.sidebar.write("Available positions: GK, DF, MF, FW")

# default selection on first load (kept in session)
if "pos_filter" not in st.session_state:
    st.session_state["pos_filter"] = ["DF", "MF", "FW"]

def _passes_group_selection(pos_str: str, selected_groups: list[str]) -> bool:
    tokens = [t.strip() for t in str(pos_str).split(',') if t.strip()]
    for group in selected_groups:
        allowed = position_groups.get(group, [group])
        if any(tok in allowed or tok == group for tok in tokens):
            return True
    return False

# ---- Position filter
pos_sel = st.sidebar.multiselect(
    "Filter by position(s) (optional)",
    ["GK", "DF", "MF", "FW"],
    default=st.session_state["pos_filter"]
)
if pos_sel:
    mask = df["Pos"].apply(lambda s: _passes_group_selection(s, pos_sel))
    df = df[mask]
st.session_state["pos_filter"] = pos_sel[:]

# ---- Minutes filter
min_minutes = st.sidebar.number_input("Minimum minutes", min_value=0, max_value=10000, value=900, step=30)
df = filter_by_minutes(df, min_minutes)

# ================== IMPORTANT SPLIT ==================
# Keep this **pre–age filter** copy for ALL calculations
st.session_state["df_for_calc"] = df.copy()

# Age filter ONLY affects what the user sees/selections
max_age = st.sidebar.number_input(
    "Maximum age",
    min_value=14, max_value=50, value=40, step=1,
    key="max_age_filter"
)
df_display = df[pd.to_numeric(df['Age'], errors='coerce') <= max_age]

# From here on, use df = df_display for UI
df = df_display

# NEW/CHANGED — keep dropdown selection stable across filter changes
player_list = df['Player'].dropna().unique().tolist()

if len(player_list):
    prev_choice = st.session_state.get("player_dropdown")
    default_index = player_list.index(prev_choice) if prev_choice in player_list else 0
    player_name = st.sidebar.selectbox(
        "Player (dropdown)",
        player_list,
        index=default_index,
        key="player_dropdown"  # <- stateful
    )
else:
    player_name = None

# NEW/CHANGED — make typed name stateful and use it to auto-fill scatter searches
typed_query = st.sidebar.text_input("Or type a player name", key="typed_query")
if typed_query:
    # propagate the typed name to both scatter search boxes when it changes
    if st.session_state.get("_last_typed_query") != typed_query:
        st.session_state["stat_scatter_search"] = typed_query
        st.session_state["role_scatter_search"] = typed_query
        st.session_state["_last_typed_query"] = typed_query

def find_player_row(df, name_query):
    if not name_query:
        return None
    exact = df[df["Player"].astype(str).str.lower() == name_query.lower()]
    if not exact.empty:
        return exact.iloc[0]
    all_names = df["Player"].dropna().astype(str).unique().tolist()
    close = difflib.get_close_matches(name_query, all_names, n=1, cutoff=0.6)
    if close:
        return df[df["Player"] == close[0]].iloc[0]
    return None

typed_row = find_player_row(df, typed_query) if typed_query else None
player_row = (
    typed_row if typed_row is not None
    else (df[df['Player'] == player_name].iloc[0] if player_name else None)
)

# --- Sidebar: Archetype (now includes custom + saved) ---
arch_dict_sidebar = get_archetype_params_full()
arch_options_sidebar = list(arch_dict_sidebar.keys())
arch_choice = st.sidebar.selectbox("Archetype", arch_options_sidebar, key="sidebar_arch_choice") if arch_options_sidebar else None
stat_cols_for_arch = arch_dict_sidebar.get(arch_choice, []) if arch_choice else []


# =========================
# ========= MODES =========
# =========================

if mode == "1":
    if player_row is None or arch_choice is None:
        st.info("Pick a player and archetype in the sidebar.")
    else:
        role_name = arch_choice
        role_stats = stat_cols_for_arch

        # Similarity among the visible pool
        st.subheader("Similarity")
        other_row = plot_similarity_and_select(df, player_row, role_stats, role_name)

        if other_row is not None:
            st.subheader("Radar comparison (percentiles)")
            # Percentiles computed vs pre-age pool
            radar_fig = plot_radar_percentiles(player_row, other_row, role_stats, st.session_state["df_for_calc"], role_name)
            try:
                png_bytes = fig_to_png_bytes_plotly(radar_fig)
                st.download_button(
                    "Download radar (PNG)",
                    data=png_bytes,
                    file_name=f"radar_{player_row['Player'].replace(' ','_')}_vs_{other_row['Player'].replace(' ','_')}.png",
                    mime="image/png",
                )
            except Exception:
                st.caption("Hover in the top right of the image to download")
            lines = [f"{other_row['Player']} (Age: {int(other_row.get('Age',np.nan)) if pd.notnull(other_row.get('Age',np.nan)) else '?'} , Club: {other_row.get('Squad','?')}, Minutes: {other_row.get('Mins','?')})",
                    f"Compared vs baseline: {player_row['Player']} — Role: {role_name}"]

            for s in role_stats:
                v = other_row.get(s, np.nan)
                p = position_relative_percentile(st.session_state["df_for_calc"], other_row, s)
                if pd.notnull(v):
                    lines.append(f"{stat_display_names.get(s,s)}: {v:.2f} (Percentile: {p:.1f}%)")

            txt = "\n".join(lines)

            st.markdown("**Copy summary**")
            st.code(txt)

            safe_base = re.sub(r"\W+", "_", f"{other_row['Player']}_vs_{player_row['Player']}").strip("_")
            st.download_button(
                "Download summary (.txt)",
                data=txt.encode("utf-8"),
                file_name=f"summary_{safe_base}.txt",
                mime="text/plain"
            )

elif mode == "2":
    if player_row is None or arch_choice is None:
        st.info("Pick a player and archetype in the sidebar.")
    else:
        role_name = arch_choice
        # Compute percentiles vs pre-age pool
        fig, df_csv = show_percentile_bar_chart(player_row, stat_cols_for_arch, st.session_state["df_for_calc"], role_name)
        try:
            png = fig_to_png_bytes_plotly(fig)
            st.download_button(
                "Download percentile chart (PNG)",
                data=png,
                file_name=f"percentiles_{player_row['Player'].replace(' ','_')}_{role_name.replace(' ','_')}.png",
                mime="image/png"
            )
        except Exception:
            st.caption("To enable PNG downloads for Plotly charts, install **kaleido**: pip install -U kaleido")
        st.download_button(
            "Download percentiles (CSV)",
            data=df_csv.to_csv(index=False).encode("utf-8"),
            file_name=f"percentiles_{player_row['Player'].replace(' ','_')}_{role_name.replace(' ','_')}.csv",
            mime="text/csv"
        )
elif mode == "3":
    if player_row is None or arch_choice is None:
        st.info("Pick a player and archetype in the sidebar.")
    else:
        role_name = arch_choice
        style_options = [
            "Light (toppings)",
            "Dark (toppings)",
            "Light (no toppings)",
            "Dark (no toppings)",
        ]
        chosen_style = st.selectbox("Pizza style", style_options, index=0)
        style_cfg = {
            "Light (toppings)":       (True,  True),
            "Dark (toppings)":        (False, True),
            "Light (no toppings)":    (True,  False),
            "Dark (no toppings)":     (False, False),
        }
        lightmode, toppings = style_cfg[chosen_style]
        # Use pre–age pool for percentiles + role score badge
        fig = show_pizza(
            player_row,
            stat_cols_for_arch,
            st.session_state["df_for_calc"],
            role_name,
            lightmode=lightmode,
            toppings=toppings,
            show_role_score=toppings
        )
        style_slug = ("light" if lightmode else "dark") + ("_toppings" if toppings else "_notoppings")
        btn_label = f"Download pizza ({chosen_style}) PNG"
        st.download_button(
            btn_label,
            data=fig_to_png_bytes_matplotlib(fig),
            file_name=f"pizza_{player_row['Player'].replace(' ','_')}_{role_name.replace(' ','_')}_{style_slug}.png",
            mime="image/png"
        )

elif mode == "4":
    arch_map = get_archetype_params_full()  # includes customs (session + disk) and built-ins
    role_options = list(arch_map.keys())

    role_name = st.selectbox("Role / Archetype", role_options, key="mode4_role_pick")
    role_stats = arch_map.get(role_name, [])

    # League filter for this mode
    df_role, leagues_chosen = league_filter_ui(df, widget_key="mode4", return_selection=True)
    suffix = f" • Leagues: {', '.join(leagues_chosen)}" if leagues_chosen else ""
    top, fig = plot_role_leaderboard(df_role, role_name, role_stats, title_suffix=suffix)

    # Download CSV (include raw columns if present)
    csv_cols = ['Player','Age','Squad','Nation','Mins','RoleScore'] + [c for c in role_stats if c in top.columns]
    csv = top[csv_cols].copy()
    csv['Role'] = role_name
    st.download_button(
        "Download CSV",
        csv.to_csv(index=False).encode("utf-8"),
        file_name=f"top_{re.sub(r'\\W+','_',role_name)}.csv"
    )

elif mode == "5":
    if player_row is None:
        st.info("Pick a player in the sidebar.")
    else:
        archetype_catalog = get_archetype_params_full()  # includes customs

        pos_str = player_row.get('Pos','')
        position_to_roles = {
            "GK": ["GK"], "DF": ["CB", "FB", "WB", "LB", "RB", "LWB", "RWB", "SW"],
            "MF": ["DM", "CM", "AM", "LM", "RM", "LW", "RW"],
            "FW": ["ST", "CF", "WF", "Winger"]
        }
        sel_groups = st.multiselect("Relevant position groups (optional)", ["GK","DF","MF","FW"], default=[])
        if not sel_groups:
            if any(k in pos_str for k in ["GK"]): sel_groups.append("GK")
            if any(k in pos_str for k in ["CB","FB","WB","LB","RB","LWB","RWB","SW","DF"]): sel_groups.append("DF")
            if any(k in pos_str for k in ["DM","CM","AM","LM","RM","LW","RW","MF"]): sel_groups.append("MF")
            if any(k in pos_str for k in ["ST","CF","WF","Winger","FW"]): sel_groups.append("FW")
            sel_groups = list(dict.fromkeys(sel_groups))

        relevant_roles = []
        for group in sel_groups:
            kws = position_to_roles.get(group, [])
            relevant_roles.extend([r for r in archetype_catalog if any(kw in r for kw in kws)])
        relevant_roles = list(dict.fromkeys(relevant_roles)) or list(archetype_catalog.keys())

        df_for_calc = df.copy()
        rs_df = precompute_role_scores(df_for_calc)

        role_scores, role_details = [], {}
        for role in relevant_roles:
            # try cached first
            score = None
            if role in rs_df.columns and player_row.name in rs_df.index:
                sc = rs_df.loc[player_row.name, role]
                if pd.notnull(sc):
                    score = float(sc)
            if score is None:
                stats_list = archetype_catalog.get(role, [])
                score = calculate_role_score(player_row, stats_list, df_for_calc, role)

            stats_list = archetype_catalog.get(role, [])
            detail_rows = []
            for s in stats_list:
                val = player_row.get(s, np.nan)
                if pd.notnull(val):
                    pctl = position_relative_percentile(df_for_calc, player_row, s)
                    detail_rows.append((stat_display_names.get(s,s), val, pctl))
            role_scores.append((role, score))
            role_details[role] = detail_rows

        role_scores.sort(key=lambda x: x[1], reverse=True)

        st.subheader(_with_pos_filter(f"Role suitability — {player_row['Player']}", include_age=False))
        fig = go.Figure([go.Bar(
            x=[s for _, s in role_scores],
            y=[r for r, _ in role_scores],
            orientation='h',
            marker=dict(
                color=[sample_colorscale('Blues', [0.2+0.8*i/len(role_scores)])[0] for i in range(len(role_scores))],
                line=dict(color='#333', width=1)
            ),
            text=[f"{s:.1f}" for _, s in role_scores], textposition='inside', insidetextanchor="middle"
        )])
        fig.update_layout(
            title=dict(text=_with_pos_filter(f"Role suitability — {player_row['Player']}", include_age=False), font=dict(color="#000", family=FONT_FAMILY)),
            plot_bgcolor=POSTER_BG, paper_bgcolor=POSTER_BG,
            xaxis=dict(title="Suitability (0–100)", range=[0,100], color="#000", tickfont=dict(color="#000"), linecolor="#000"),
            yaxis=dict(autorange='reversed', color="#000", tickfont=dict(color="#000"), linecolor="#000"),
            height=min(80 + 28*len(role_scores), 1200),
            template="simple_white",
        )
        apply_hover_style(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        role_pick = st.selectbox("View stat details for role", [r for r,_ in role_scores])
        det = role_details.get(role_pick, [])
        if det:
            st.write("**Stat details (player value, positional percentile):**")
            st.dataframe(pd.DataFrame(det, columns=["Stat","Value","Percentile (%)"]))

elif mode == "6":
    # Stat Leaders
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ['Age','Mins']]
    display_names = {c: stat_display_names.get(c,c) for c in numeric_cols}
    if not numeric_cols:
        st.warning("No numeric stat columns found.")
    else:
        x_label = st.selectbox("Choose a stat", [display_names[c] for c in numeric_cols])
        stat_col = [c for c in numeric_cols if display_names[c] == x_label][0]

        df_league, leagues_chosen = league_filter_ui(df, widget_key="mode6", return_selection=True)
        league_suffix = f" • Leagues: {', '.join(leagues_chosen)}" if leagues_chosen else ""
        top, fig = show_top_players_by_stat(df_league, x_label, stat_col, title_suffix=league_suffix)

elif mode == "7":
    st.subheader("Custom Archetype")

    # One league filter for this mode
    df_league = league_filter_ui(df)

    # Available numeric stats
    numeric_cols = [c for c in df_league.columns if pd.api.types.is_numeric_dtype(df_league[c]) and c not in ['Age','Mins']]
    if not numeric_cols:
        st.warning("No numeric stat columns found.")
    else:
        display_names = {c: stat_display_names.get(c, c) for c in numeric_cols}

        # Name + stat selection (limit to 10)
        name_default = st.session_state.get("custom_arch_name", "Custom Archetype")
        custom_name = st.text_input("Archetype name", value=name_default, key="custom_arch_name")

        picked_labels = st.multiselect("Pick up to 10 stats", [display_names[c] for c in numeric_cols])
        stat_cols = [c for c in numeric_cols if display_names[c] in picked_labels][:10]
        if len(picked_labels) > 10:
            st.warning("Only the first 10 selected stats are used.")

        if not stat_cols:
            st.info("Choose at least one stat to build your archetype.")
        else:
            # Weights UI
            st.markdown("**Weights** (higher = more important; negatives allowed):")
            cols = st.columns(min(5, len(stat_cols)))
            weights = {}
            for i, s in enumerate(stat_cols):
                with cols[i % len(cols)]:
                    weights[s] = st.number_input(
                        f"{display_names[s]}",
                        value=1.0, step=0.1, format="%.2f", key=f"wt_{s}"
                    )

            # Save to THIS SESSION sidebar
            if st.button("➕ Add to my sidebar (this session only)", type="primary"):
                st.session_state["custom_arches"][custom_name] = {
                    "stats": stat_cols,
                    "weights": weights
                }
                st.success(f"Added **{custom_name}** to your sidebar for this session.")

            # ---- Export / Import ----
            st.divider()
            st.markdown("### Save / Load your custom archetypes")
            export_custom_archetypes_button("Download my custom archetypes (JSON)")
            import_custom_archetypes_uploader(df_cols=set(df_league.columns))

            st.divider()

            # Build leaderboard with this custom (uses positional percentiles)
            dfc = df_league.copy()
            dfc['CustomScore'] = [
                calculate_custom_archetype_score(r, stat_cols, df_league, weights)
                for _, r in df_league.iterrows()
            ]
            top = dfc.dropna(subset=['CustomScore']).nlargest(10, 'CustomScore').reset_index(drop=True)

            pastel_blues = ["#E8F1FE","#DCEBFE","#CFE5FE","#C2DFFE","#B5D9FE",
                            "#A8D2FD","#9BCBFD","#8EC4FD","#81BCFD","#74B4FC"][::-1]
            bar_colors = pastel_blues[-len(top):][::-1]
            def lum(c):
                r,g,b = [x*255 for x in mcolors.hex2color(c)]
                return 0.299*r + 0.587*g + 0.114*b
            bar_text_colors = ['white' if lum(c) < 150 else '#222' for c in bar_colors]
            labels = [
                f"{row['Player']} • {row['CustomScore']:.2f} • "
                f"{int(row.get('Age',np.nan)) if pd.notnull(row.get('Age',np.nan)) else '?'} • "
                f"{row.get('Squad','?')} • {int(row.get('Mins',0)):,} mins"
                for _, row in top.iterrows()
            ]

            # Title mentions leagues if filtered (best-effort cosmetic)
            fig = go.Figure([
                go.Bar(
                    x=top['CustomScore'],
                    y=[f"#{i+1}" for i in range(len(top))],
                    orientation='h',
                    text=labels,
                    textposition='inside',
                    insidetextanchor='middle',
                    marker=dict(color=bar_colors, line=dict(color='#333', width=1)),
                    textfont=dict(color=bar_text_colors, size=13, family=FONT_FAMILY),
                    hovertext=[f"{row['Player']} ({row['Squad']})" for _, row in top.iterrows()],
                    hoverinfo="text"
                )
            ])
            fig.update_layout(
                title=dict(
                    text=_with_pos_filter(f"Top 10 — {custom_name}"),
                    font=dict(color="#000", family=FONT_FAMILY)
                ),
                plot_bgcolor=POSTER_BG, paper_bgcolor=POSTER_BG,
                xaxis=dict(title='Score (0–100)', range=[0,100], gridcolor=POSTER_BG, color="#000", tickfont=dict(color="#000"), linecolor="#000"),
                yaxis=dict(autorange='reversed', showgrid=False, color="#000", tickfont=dict(color="#000"), linecolor="#000"),
                margin=dict(l=120, r=40, t=60, b=40),
                height=600,
            )
            apply_hover_style(fig)
            st.plotly_chart(fig, use_container_width=True, theme=None)

            csv_cols = ['Player','Age','Squad','Nation','Mins','CustomScore'] + stat_cols
            st.download_button(
                "Download leaderboard CSV",
                data=top[csv_cols].to_csv(index=False).encode("utf-8"),
                file_name=f"top_custom_{re.sub(r'\\W+','_',custom_name.strip())}.csv",
                mime="text/csv",
                key="dl_custom_leaderboard"
            )

            st.divider()
            st.markdown("### Score any player")
            default_lookup = player_row['Player'] if player_row is not None else ""
            lookup_name = st.text_input("Type a player name", value=default_lookup, key="custom_lookup")
            target_row = find_player_row(df_league, lookup_name) if lookup_name else None
            if target_row is None and lookup_name:
                st.warning("No close match found.")
            if target_row is not None:
                score = calculate_custom_archetype_score(target_row, stat_cols, df_league, weights)
                st.metric(f"{target_row['Player']} — {custom_name}", f"{score:.2f}")
                _ = show_percentile_bar_chart(target_row, stat_cols, df_league, custom_name)

elif mode == "12":
    # Stat Scatter
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ['Age','Mins']]
    display_names = {c: stat_display_names.get(c,c) for c in numeric_cols}
    if len(numeric_cols) >= 2:
        x_pick = st.selectbox("X-axis stat", [display_names[c] for c in numeric_cols], index=0)
        y_pick = st.selectbox("Y-axis stat", [display_names[c] for c in numeric_cols], index=1)
        search_name = st.text_input("Search & highlight a player (optional)", key="stat_scatter_search")

        x_col = [c for c in numeric_cols if display_names[c] == x_pick][0]
        y_col = [c for c in numeric_cols if display_names[c] == y_pick][0]

        dfx = df.dropna(subset=[x_col, y_col]).copy()
        try:
            dfx['Age'] = pd.to_numeric(dfx['Age'], errors='coerce').astype('Int64')
        except Exception:
            pass

        hover = [
            f"<b>{r['Player']}</b><br>"
            f"Team: {r.get('Squad','?')}<br>"
            f"Age: {r.get('Age','?')}<br>"
            f"{x_pick}: {r[x_col]:.2f}<br>{y_pick}: {r[y_col]:.2f}<br>"
            f"Minutes: {int(r.get('Mins',0)):,}"
            for _, r in dfx.iterrows()
        ]

        text_labels, marker_colors, marker_sizes, marker_lines, highlight = scatter_labels_and_styles(
            dfx, x_col, y_col, search_name.strip() if search_name else None
        )

        fig = go.Figure([
            go.Scatter(
                x=dfx[x_col], y=dfx[y_col],
                mode='markers+text',
                text=text_labels,
                textposition='top center',
                textfont=dict(family=FONT_FAMILY, size=12, color="#000"),
                marker=dict(
                    size=marker_sizes,
                    color=marker_colors,
                    line=dict(width=1, color=marker_lines)
                ),
                hovertext=hover, hoverinfo='text'
            )
        ])

        fig.update_layout(
            title=dict(text=_with_pos_filter(f"{x_pick} vs {y_pick} — scatter"), font=dict(color="#000", family=FONT_FAMILY)),
            xaxis=dict(title=x_pick, gridcolor=POSTER_BG, zeroline=False, linecolor='#000', color="#000", tickfont=dict(color="#000")),
            yaxis=dict(title=y_pick, gridcolor=POSTER_BG, zeroline=False, linecolor='#000', color="#000", tickfont=dict(color="#000")),
            plot_bgcolor=POSTER_BG, paper_bgcolor=POSTER_BG, height=800,
        )
        apply_hover_style(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        if search_name and not highlight:
            st.caption("No close match found for that player.")
    else:
        st.warning("Need at least two numeric stat columns.")

elif mode == "13":
    # Role Matrix scatter
    arch_dict_local = get_archetype_params_full()
    role_options = list(arch_dict_local.keys())

    role_x = st.selectbox("X-axis role", role_options, index=0, key="mode13_role_x")
    role_y = st.selectbox("Y-axis role", role_options, index=1, key="mode13_role_y")
    search_name = st.text_input("Search & highlight a player (optional)", key="role_scatter_search")

    dfc = df.copy()
    dfc['RoleScore_X'] = compute_role_scores_cached(st.session_state["df_for_calc"], role_x).reindex(dfc.index)
    dfc['RoleScore_Y'] = compute_role_scores_cached(st.session_state["df_for_calc"], role_y).reindex(dfc.index)

    try:
        dfc['Age'] = pd.to_numeric(dfc['Age'], errors='coerce').astype('Int64')
    except Exception:
        pass

    hover = [
        f"<b>{r['Player']}</b><br>"
        f"Team: {r.get('Squad','?')}<br>"
        f"Age: {r.get('Age','?')}<br>"
        f"{role_x}: {r['RoleScore_X']:.1f}<br>"
        f"{role_y}: {r['RoleScore_Y']:.1f}<br>"
        f"Minutes: {int(r.get('Mins',0)):,}"
        for _, r in dfc.iterrows()
    ]

    text_labels, marker_colors, marker_sizes, marker_lines, highlight = scatter_labels_and_styles(
        dfc, 'RoleScore_X', 'RoleScore_Y', search_name.strip() if search_name else None
    )

    fig = go.Figure([
        go.Scatter(
            x=dfc['RoleScore_X'], y=dfc['RoleScore_Y'],
            mode='markers+text',
            text=text_labels, textposition='top center',
            textfont=dict(family=FONT_FAMILY, size=12, color="#000"),
            marker=dict(size=marker_sizes, color=marker_colors, line=dict(width=1, color=marker_lines)),
            hovertext=hover, hoverinfo='text'
        )
    ])

    fig.update_layout(
        title=dict(text=_with_pos_filter(f"{role_x} vs {role_y} — role suitability"),
                   font=dict(color="#000", family=FONT_FAMILY)),
        xaxis=dict(title=role_x, gridcolor=POSTER_BG, zeroline=False, linecolor='#000', color="#000",
                   tickfont=dict(color="#000")),
        yaxis=dict(title=role_y, gridcolor=POSTER_BG, zeroline=False, linecolor='#000', color="#000",
                   tickfont=dict(color="#000")),
        plot_bgcolor=POSTER_BG, paper_bgcolor=POSTER_BG, height=800,
    )
    apply_hover_style(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    if search_name and not highlight:
        st.caption("No close match found for that player.")

elif mode == "14":
    st.subheader("Player Finder")

    # Optional league filter
    df_league, leagues_chosen = league_filter_ui(df, widget_key="mode14", return_selection=True)
    if leagues_chosen:
        st.caption(f"Leagues filtered: {', '.join(leagues_chosen)}")

    # Available numeric stats (same rule as elsewhere)
    numeric_cols = [c for c in df_league.columns if pd.api.types.is_numeric_dtype(df_league[c]) and c not in ['Age', 'Mins']]
    if not numeric_cols:
        st.warning("No numeric stat columns found.")
    else:
        display_names = {c: stat_display_names.get(c, c) for c in numeric_cols}

        picked_labels = st.multiselect("Pick up to 10 stats to filter by", [display_names[c] for c in numeric_cols])
        selected_stats = [c for c in numeric_cols if display_names[c] in picked_labels][:10]
        if len(picked_labels) > 10:
            st.warning("Only the first 10 selected stats are used.")

        if not selected_stats:
            st.info("Choose at least one stat to start filtering.")
        else:
            st.markdown("**Minimum percentiles** (per stat):")

            cols = st.columns(min(5, len(selected_stats)))
            thresholds = {}
            for i, s in enumerate(selected_stats):
                with cols[i % len(cols)]:
                    thresholds[s] = st.slider(
                        f"{display_names[s]}",
                        min_value=0, max_value=100, value=80, step=1,
                        help="Players must meet or exceed this percentile (GLOBAL, across the filtered dataset)."
                    )

            max_results = st.slider("Max players to show", 10, 200, 60, step=10)

            # GLOBAL percentiles for Player Finder (cached) — uses visible dataset
            with st.spinner("Computing percentiles..."):
                pct_df = _compute_percentiles_table(df_league, selected_stats, baseline="global")

            dfq = df_league.join(pct_df)

            mask = pd.Series(True, index=dfq.index)
            for s in selected_stats:
                mask &= dfq[f"pct_{s}"].fillna(-1) >= thresholds[s]

            matches = dfq[mask].copy()

            if matches.empty:
                st.info("No players matched those filters. Try lowering one or more thresholds.")
            else:
                pct_cols = [f"pct_{s}" for s in selected_stats]
                matches["AvgSelectedPct"] = matches[pct_cols].mean(axis=1, skipna=True)
                matches = matches.sort_values("AvgSelectedPct", ascending=False)

                st.success(f"{len(matches)} players matched. Showing top {min(max_results, len(matches))}.")

                st.markdown("""
                <style>
                .pf-card {
                    border: 1px solid #e1e1e1;
                    background: #ffffff;
                    border-radius: 16px;
                    padding: 12px 14px;
                    margin-bottom: 12px;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                }
                .pf-card, .pf-card * { color: #000 !important; }
                .pf-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 4px; }
                .pf-sub   { font-size: 0.9rem;  margin-bottom: 8px; }
                .pf-stats { font-size: 0.9rem;  line-height: 1.35; }
                .pf-statline {
                    display: flex; justify-content: space-between;
                    border-bottom: 1px dashed #eee; padding: 4px 0;
                }
                .pf-statname { opacity: 1 !important; }
                .pf-statpct  { font-weight: 700; }
                </style>
                """, unsafe_allow_html=True)

                cols = st.columns(3)
                shown = 0
                for _, r in matches.head(max_results).iterrows():
                    col = cols[shown % 3]
                    with col:
                        try:
                            age_val = r.get("Age", np.nan)
                            age_txt = f"{int(round(age_val))}" if pd.notnull(age_val) else "?"
                        except Exception:
                            age_txt = str(r.get("Age", "?"))

                        squad = r.get("Squad", "?")
                        pos   = r.get("Pos", "?")
                        name  = r.get("Player", "?")

                        lines_html = []
                        for s in selected_stats:
                            val = r.get(s, np.nan)
                            if pd.isna(val):
                                val_txt = "—"
                            else:
                                try:
                                    val_txt = f"{float(val):.2f}"
                                except Exception:
                                    val_txt = str(val)
                            pct = r.get(f"pct_{s}", np.nan)
                            pct_txt = f"{pct:.1f}%" if pd.notnull(pct) else "—"
                            lines_html.append(
                                f"<div class='pf-statline'><span class='pf-statname'>{stat_display_names.get(s, s)}</span>"
                                f"<span class='pf-statpct'>{val_txt} ({pct_txt})</span></div>"
                            )

                        st.markdown(
                            f"""
                            <div class="pf-card">
                              <div class="pf-title">{name}</div>
                              <div class="pf-sub">{squad} • Age {age_txt} • {pos}</div>
                              <div class="pf-stats">{''.join(lines_html)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    shown += 1

elif mode == "15":
    st.subheader("Glossary & Help")
    st.markdown("""
**Quick guide**

- **Sidebar**
  - *Season / Database*: pick which CSVs to load.
  - *Positions / Minutes / Max age*: filter the visible dataset.  
    Percentiles and Role Scores always use the **pre–age** pool to prevent shifting scores when you move the age slider.
  - *Archetype*: includes built-ins, category pizzas, **and your custom archetypes** (session + saved JSON).

- **Modes**
  1. *Similar*: find similar players for the picked archetype; shows a radar (positional percentiles).
  2. *Percentiles*: bar chart for the player vs same-position players.
  3. *Pizza*: radar-like pizza with optional role score badge.
  4. *Role Leaders*: top players by chosen role/archetype.
  5. *Best Roles*: which roles fit your selected player best.
  6. *Stat Leaders*: top 10 by any single stat.
  7. *Custom Archetype*: build/weight stats, rank players, export/import JSON.
  12. *Stat Scatter*: scatter any two stats; search highlights a player.
  13. *Role Matrix*: scatter with two role scores as axes.
  14. *Player Finder*: multi-stat global percentile filters.
  16. *Head-to-Head*: radar comparison for two players on the chosen archetype.
    """)

elif mode == "16":
    if arch_choice is None:
        st.info("Pick an archetype in the sidebar.")
    else:
        role_name = arch_choice
        role_stats = stat_cols_for_arch

        df_for_calc_local = st.session_state["df_for_calc"]
        players = df['Player'].dropna().unique().tolist()  # choose from visible

        default_a = player_row['Player'] if player_row is not None else (players[0] if players else None)
        pA_name = st.selectbox("Player A", players, index=(players.index(default_a) if default_a in players else 0) if players else 0)
        pB_name = st.selectbox("Player B", players, index=(players.index(default_a) if default_a in players else 0) if players else 0, key="h2h_b")

        pA = df_for_calc_local[df_for_calc_local['Player'] == pA_name].iloc[0] if pA_name else None
        pB = df_for_calc_local[df_for_calc_local['Player'] == pB_name].iloc[0] if pB_name else None

        if pA is None or pB is None or pA_name == pB_name:
            st.info("Pick two different players.")
        else:
            radar_fig = plot_radar_percentiles(pA, pB, role_stats, df_for_calc_local, role_name)
            try:
                png_bytes = fig_to_png_bytes_plotly(radar_fig)
                st.download_button(
                    "Download radar (PNG)",
                    data=png_bytes,
                    file_name=f"radar_{pA_name.replace(' ','_')}_vs_{pB_name.replace(' ','_')}_{role_name.replace(' ','_')}.png",
                    mime="image/png",
                )
            except Exception:
                pass

            rows = []
            for s in role_stats:
                vA = pA.get(s, np.nan)
                vB = pB.get(s, np.nan)
                pA_pct = position_relative_percentile(df_for_calc_local, pA, s)
                pB_pct = position_relative_percentile(df_for_calc_local, pB, s)
                rows.append({
                    "Stat": stat_display_names.get(s, s),
                    f"{pA_name} (val)": vA,
                    f"{pA_name} (pct)": pA_pct,
                    f"{pB_name} (val)": vB,
                    f"{pB_name} (pct)": pB_pct
                })
            cmp_df = pd.DataFrame(rows)
            st.dataframe(cmp_df)

            st.download_button(
                "Download comparison (CSV)",
                data=cmp_df.to_csv(index=False).encode("utf-8"),
                file_name=f"h2h_{pA_name.replace(' ','_')}_vs_{pB_name.replace(' ','_')}_{role_name.replace(' ','_')}.csv",
                mime="text/csv"
            )
