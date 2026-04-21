"""
app.py  ─  VisionLab · Premium Dashboard
─────────────────────────────────────────
UI-only file.  All CV logic stays in processor.py.
Processing is triggered once (Apply button) and cached
in st.session_state so the UI and the logic are fully decoupled.

User flow:  Sidebar (Upload → Select → Configure → Apply)
            Tab 1 "View"    → uploaded image + metadata
            Tab 2 "Process" → result + technique summary
            Tab 3 "Compare" → side-by-side diff
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from processor import ImageProcessor

# ══════════════════════════════════════════════════════════════════════════════
# 0 · PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VisionLab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# 1 · GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Fonts ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Root palette ───────────────────────────────────────────────────────── */
:root {
    --bg-base:    #060910;
    --bg-surface: #0d1117;
    --bg-card:    #161b22;
    --bg-hover:   #1c2333;
    --border:     #21262d;
    --border-hi:  #30363d;
    --accent-1:   #58a6ff;
    --accent-2:   #bc8cff;
    --accent-g:   linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
    --green:      #3fb950;
    --yellow:     #d29922;
    --red:        #f85149;
    --text-pri:   #e6edf3;
    --text-sec:   #8b949e;
    --text-muted: #484f58;
}

/* ── Shell ──────────────────────────────────────────────────────────────── */
.stApp { background: var(--bg-base) !important; }
.block-container { padding: 2.5rem 3rem 4rem !important; max-width: 100% !important; }

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 4px; }

/* ══════════════════════ SIDEBAR ════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }
[data-testid="stSidebar"] * { color: var(--text-pri) !important; }

/* Sidebar section labels */
.sb-section-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: var(--text-muted) !important;
    margin: 1.4rem 0 .5rem;
}

/* ══════════════════════ GLOBAL TEXT ════════════════════════════════════════ */
h1,h2,h3,h4,h5,h6,p,span,label,div { color: var(--text-pri); }
[data-testid="stMarkdownContainer"] p { color: var(--text-sec); line-height: 1.7; }

/* ══════════════════════ HERO HEADER ════════════════════════════════════════ */
.hero {
    position: relative; overflow: hidden;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.2rem 2.4rem 2rem;
    margin-bottom: 2rem;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(88,166,255,.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute; bottom: -80px; left: 40%;
    width: 300px; height: 300px; border-radius: 50%;
    background: radial-gradient(circle, rgba(188,140,255,.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: .72rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
    color: var(--accent-1); margin-bottom: .5rem;
}
.hero-title {
    font-size: 2rem; font-weight: 800; line-height: 1.2;
    background: var(--accent-g);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .5rem;
}
.hero-sub { font-size: .88rem; color: var(--text-sec); max-width: 560px; }
.hero-badges { display: flex; gap: 8px; margin-top: 1rem; flex-wrap: wrap; }
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 999px; font-size: .7rem; font-weight: 600;
}
.pill-blue   { background: rgba(88,166,255,.12); border: 1px solid rgba(88,166,255,.3); color: #58a6ff; }
.pill-purple { background: rgba(188,140,255,.12); border: 1px solid rgba(188,140,255,.3); color: #bc8cff; }
.pill-green  { background: rgba(63,185,80,.12);  border: 1px solid rgba(63,185,80,.3);  color: #3fb950; }
.pill-yellow { background: rgba(210,153,34,.12); border: 1px solid rgba(210,153,34,.3); color: #d29922; }

/* ══════════════════════ STAT CARDS ═════════════════════════════════════════ */
.stats-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 1.6rem; }
@media (max-width: 900px) { .stats-row { grid-template-columns: repeat(2,1fr); } }
.stat-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 20px;
    transition: border-color .2s, transform .15s;
}
.stat-card:hover { border-color: var(--border-hi); transform: translateY(-2px); }
.stat-card .sc-label { font-size: .65rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; color: var(--text-muted); }
.stat-card .sc-value { font-size: 1.5rem; font-weight: 700; color: var(--text-pri); margin: 4px 0 2px; line-height: 1; }
.stat-card .sc-sub   { font-size: .68rem; color: var(--text-muted); }
.stat-card .sc-icon  { font-size: 1.1rem; margin-bottom: 6px; }

/* ══════════════════════ IMAGE CARD ═════════════════════════════════════════ */
.img-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 0; overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,.4);
    transition: box-shadow .25s ease, border-color .25s ease;
}
.img-card:hover {
    box-shadow: 0 12px 48px rgba(88,166,255,.12);
    border-color: rgba(88,166,255,.25);
}
.img-card-header {
    display: flex; align-items: center; gap: 8px;
    padding: 12px 16px; border-bottom: 1px solid var(--border);
    background: rgba(255,255,255,.02);
}
.img-card-dot { width: 10px; height: 10px; border-radius: 50%; }
.img-card-title { font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--text-sec); }
.img-card-body { padding: 12px; }

/* ══════════════════════ COMPARE SLIDER WRAPPER ═════════════════════════════ */
.compare-label {
    font-size: .7rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
    color: var(--text-muted); text-align: center; margin-bottom: 8px;
}

/* ══════════════════════ TECHNIQUE BADGE ════════════════════════════════════ */
.technique-card {
    background: linear-gradient(135deg, rgba(88,166,255,.08) 0%, rgba(188,140,255,.08) 100%);
    border: 1px solid rgba(88,166,255,.2);
    border-radius: 12px; padding: 18px 20px; margin-bottom: 1.2rem;
}
.technique-card .tc-name { font-size: 1rem; font-weight: 700; color: var(--accent-1); }
.technique-card .tc-desc { font-size: .8rem; color: var(--text-sec); margin-top: 4px; line-height: 1.6; }

/* ══════════════════════ TABS ════════════════════════════════════════════════ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border);
    gap: 0; padding: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important; border: none !important; outline: none !important;
    color: var(--text-sec) !important; font-size: .84rem; font-weight: 500;
    padding: 12px 22px; border-bottom: 2px solid transparent !important;
    transition: color .15s, border-color .15s;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent-1) !important;
    border-bottom: 2px solid var(--accent-1) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top: 1.6rem !important; }

/* ══════════════════════ BUTTONS ═════════════════════════════════════════════ */
div.stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    color: #fff !important; font-weight: 600; font-size: .84rem;
    border: none !important; border-radius: 8px !important;
    padding: 11px 28px !important; letter-spacing: .02em;
    width: 100%; box-shadow: 0 2px 16px rgba(31,111,235,.35);
    transition: opacity .2s, transform .12s;
}
div.stButton > button:hover  { opacity: .87 !important; transform: translateY(-1px); }
div.stButton > button:active { transform: translateY(0); }

/* ══════════════════════ EXPANDER ════════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: var(--text-sec) !important; font-size: .8rem !important;
    font-weight: 500 !important; padding: 12px 16px !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text-pri) !important; }

/* ══════════════════════ FORM ELEMENTS ═══════════════════════════════════════ */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 8px !important; color: var(--text-pri) !important;
}
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-hi) !important; border-radius: 10px !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] { padding: 0 !important; }

/* ══════════════════════ ALERTS ══════════════════════════════════════════════ */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ══════════════════════ METRICS ═════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label          { color: var(--text-muted) !important; font-size: .68rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--text-pri) !important; font-size: 1.2rem !important; font-weight: 700 !important; }

/* ══════════════════════ DIVIDERS ════════════════════════════════════════════ */
hr { border-color: var(--border) !important; margin: 1.4rem 0 !important; }

/* ══════════════════════ STEP FLOW ═══════════════════════════════════════════ */
.step-flow { display: flex; align-items: center; gap: 0; margin-bottom: 2rem; list-style: none; padding: 0; }
.step { display: flex; align-items: center; gap: 8px; }
.step-num {
    width: 28px; height: 28px; border-radius: 50%; display: flex;
    align-items: center; justify-content: center;
    font-size: .72rem; font-weight: 700;
}
.step-active .step-num  { background: var(--accent-1); color: #000; }
.step-done   .step-num  { background: var(--green);    color: #000; }
.step-idle   .step-num  { background: var(--border-hi); color: var(--text-muted); }
.step-label { font-size: .75rem; font-weight: 600; color: var(--text-sec); }
.step-active .step-label { color: var(--accent-1); }
.step-done   .step-label { color: var(--green); }
.step-sep { flex: 1; height: 1px; background: var(--border); margin: 0 12px; min-width: 20px; }

/* Sidebar logo area */
.sb-logo { padding-bottom: 1rem; border-bottom: 1px solid var(--border); margin-bottom: .2rem; }
.sb-logo-title { font-size: 1.1rem; font-weight: 800; letter-spacing: -.01em;
    background: var(--accent-g); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.sb-logo-sub { font-size: .7rem; color: var(--text-muted) !important; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2 · TECHNIQUE REGISTRY  (pure data — no logic, no UI)
# ══════════════════════════════════════════════════════════════════════════════
TECHNIQUES: dict[str, dict] = {
    "Gaussian Blur": {
        "icon": "🌫️", "tab": "Filters",
        "desc": "Smooths the image by convolving with a Gaussian kernel. "
                "Larger kernels produce stronger blur.",
        "params": [
            {"key": "blur_k",    "label": "Kernel Size", "type": "slider",
             "min": 1, "max": 31, "default": 5, "step": 2},
        ],
    },
    "Sharpen": {
        "icon": "🔪", "tab": "Filters",
        "desc": "Enhances fine details using a Laplacian unsharp-mask kernel.",
        "params": [
            {"key": "sharp_str", "label": "Strength", "type": "slider_float",
             "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1},
        ],
    },
    "Sobel Edges": {
        "icon": "📐", "tab": "Edge Detection",
        "desc": "Computes gradient magnitude in X and Y directions. "
                "Highlights contours and structural boundaries.",
        "params": [
            {"key": "sobel_k",     "label": "Kernel Size", "type": "select",
             "options": [1, 3, 5, 7], "default": 3},
            {"key": "sobel_scale", "label": "Scale",       "type": "slider_float",
             "min": 0.5, "max": 5.0, "default": 1.0, "step": 0.1},
        ],
    },
    "Canny Edges": {
        "icon": "⚡", "tab": "Edge Detection",
        "desc": "Multi-stage optimal edge detector: noise reduction → gradient → "
                "non-max suppression → hysteresis thresholding.",
        "params": [
            {"key": "canny_t1", "label": "Low Threshold",  "type": "slider",
             "min": 0, "max": 500, "default": 50, "step": 1},
            {"key": "canny_t2", "label": "High Threshold", "type": "slider",
             "min": 0, "max": 500, "default": 150, "step": 1},
        ],
    },
    "Padding": {
        "icon": "🔲", "tab": "Transforms",
        "desc": "Adds a configurable border around the image using various "
                "padding strategies.",
        "params": [
            {"key": "pad_mode",   "label": "Mode",   "type": "select",
             "options": ["zero", "edge", "reflect", "wrap"], "default": "zero"},
            {"key": "pad_top",    "label": "Top",    "type": "slider",
             "min": 0, "max": 150, "default": 20, "step": 1},
            {"key": "pad_bottom", "label": "Bottom", "type": "slider",
             "min": 0, "max": 150, "default": 20, "step": 1},
            {"key": "pad_left",   "label": "Left",   "type": "slider",
             "min": 0, "max": 150, "default": 20, "step": 1},
            {"key": "pad_right",  "label": "Right",  "type": "slider",
             "min": 0, "max": 150, "default": 20, "step": 1},
        ],
    },
    "Otsu Threshold": {
        "icon": "⚖️", "tab": "Segmentation",
        "desc": "Finds the optimal global threshold automatically by minimising "
                "intra-class variance. No parameters required.",
        "params": [],
    },
    "K-Means": {
        "icon": "🎨", "tab": "Segmentation",
        "desc": "Clusters pixels into K colour groups. Each pixel is replaced "
                "by its cluster's average colour.",
        "params": [
            {"key": "kmeans_k", "label": "Clusters (K)", "type": "slider",
             "min": 2, "max": 16, "default": 3, "step": 1},
        ],
    },
    "RLE Compression": {
        "icon": "📦", "tab": "Compression",
        "desc": "Lossless Run-Length Encoding. Consecutive identical grayscale "
                "pixel values are stored as (value, count) pairs.",
        "params": [],
    },

    # ── New Filters ──────────────────────────────────────────────────────────
    "Mean Filter": {
        "icon": "🔲", "tab": "Filters",
        "desc": "Simple box/mean filter that replaces each pixel with the average of its neighbourhood.",
        "params": [{"key": "mean_k", "label": "Kernel Size", "type": "slider", "min": 1, "max": 31, "default": 5, "step": 2}],
    },
    "Median Filter": {
        "icon": "📊", "tab": "Filters",
        "desc": "Replaces each pixel with the median of its neighbourhood. Effective against salt-and-pepper noise.",
        "params": [{"key": "median_k", "label": "Kernel Size", "type": "slider", "min": 1, "max": 31, "default": 5, "step": 2}],
    },
    "Laplacian Edges": {
        "icon": "🔍", "tab": "Edge Detection",
        "desc": "Second-order derivative edge detector. Highlights regions of rapid intensity change.",
        "params": [],
    },
    "Midpoint Filter": {
        "icon": "⚖️", "tab": "Filters",
        "desc": "Replaces each pixel with (local_max + local_min) / 2. Effective for uniformly distributed noise.",
        "params": [{"key": "midpoint_k", "label": "Kernel Size", "type": "slider", "min": 1, "max": 15, "default": 3, "step": 2}],
    },
    "Alpha-Trimmed Mean": {
        "icon": "✂️", "tab": "Filters",
        "desc": "Trims d/2 lowest and d/2 highest pixels in each window, then averages the rest.",
        "params": [
            {"key": "alpha_k", "label": "Kernel Size", "type": "slider", "min": 3, "max": 15, "default": 5, "step": 2},
            {"key": "alpha_d", "label": "Trim d",      "type": "slider", "min": 0, "max": 8,  "default": 2, "step": 2},
        ],
    },
    "Harmonic Mean Filter": {
        "icon": "〰️", "tab": "Filters",
        "desc": "Harmonic mean per window: n / Σ(1/x). Good for removing salt noise.",
        "params": [{"key": "harmonic_k", "label": "Kernel Size", "type": "slider", "min": 3, "max": 15, "default": 5, "step": 2}],
    },
    "Contraharmonic Filter": {
        "icon": "🔄", "tab": "Filters",
        "desc": "Contraharmonic mean Σ(x^(Q+1))/Σ(x^Q). Q>0 removes pepper; Q<0 removes salt noise.",
        "params": [
            {"key": "contra_k", "label": "Kernel Size", "type": "slider",       "min": 3,    "max": 15,  "default": 5,   "step": 2},
            {"key": "contra_q", "label": "Q Order",    "type": "slider_float",  "min": -3.0, "max": 3.0, "default": 1.5, "step": 0.5},
        ],
    },
    "Box Low-Pass Filter": {
        "icon": "📉", "tab": "Filters",
        "desc": "Normalised flat (box) kernel applied via convolution — a classic low-pass smoothing filter.",
        "params": [{"key": "box_k", "label": "Kernel Size", "type": "slider", "min": 1, "max": 31, "default": 5, "step": 2}],
    },
    "High-Pass Filter": {
        "icon": "📈", "tab": "Filters",
        "desc": "Original minus low-pass result. Isolates high-frequency detail and edge information.",
        "params": [{"key": "hp_k", "label": "Kernel Size", "type": "slider", "min": 1, "max": 31, "default": 5, "step": 2}],
    },
    "Prewitt Edges": {
        "icon": "🗺️", "tab": "Edge Detection",
        "desc": "Prewitt operator edge detection in X and Y directions. Returns gradient magnitude.",
        "params": [],
    },

    # ── New Segmentation ─────────────────────────────────────────────────────
    "Global Threshold": {
        "icon": "🎚️", "tab": "Segmentation",
        "desc": "Manual binary threshold: pixels above the value become white, below become black.",
        "params": [{"key": "thresh_val", "label": "Threshold", "type": "slider", "min": 0, "max": 255, "default": 127, "step": 1}],
    },
    "Adaptive Threshold": {
        "icon": "🧠", "tab": "Segmentation",
        "desc": "Computes a local threshold per block using Gaussian weighting — handles uneven illumination.",
        "params": [
            {"key": "block_size", "label": "Block Size", "type": "slider", "min": 3,  "max": 51, "default": 11, "step": 2},
            {"key": "thresh_c",  "label": "C Constant", "type": "slider", "min": 0,  "max": 10, "default": 2,  "step": 1},
        ],
    },
    "Multi-Otsu": {
        "icon": "🌈", "tab": "Segmentation",
        "desc": "Multi-level Otsu thresholding via skimage. Divides the image into N intensity classes.",
        "params": [{"key": "otsu_classes", "label": "Classes", "type": "slider", "min": 2, "max": 5, "default": 3, "step": 1}],
    },
    "Region Growing": {
        "icon": "🌱", "tab": "Segmentation",
        "desc": "Flood-fill from a seed point — grows the region to neighbouring pixels within a threshold.",
        "params": [
            {"key": "seed_y",    "label": "Seed Y (%)",  "type": "slider", "min": 0, "max": 100, "default": 50, "step": 1},
            {"key": "seed_x",   "label": "Seed X (%)",  "type": "slider", "min": 0, "max": 100, "default": 50, "step": 1},
            {"key": "rg_thresh","label": "Threshold",    "type": "slider", "min": 0, "max": 50,  "default": 10, "step": 1},
        ],
    },
    "Watershed": {
        "icon": "💧", "tab": "Segmentation",
        "desc": "Marker-based watershed segmentation. Segment boundaries are drawn in red.",
        "params": [],
    },
    "SLIC Superpixels": {
        "icon": "🧩", "tab": "Segmentation",
        "desc": "SLIC superpixel segmentation via skimage. Returns original image with yellow superpixel boundaries.",
        "params": [
            {"key": "n_segments",  "label": "Num Segments", "type": "slider",       "min": 50,  "max": 500,  "default": 100,  "step": 50},
            {"key": "compactness", "label": "Compactness",  "type": "slider_float", "min": 1.0, "max": 20.0, "default": 10.0, "step": 1.0},
        ],
    },

    # ── New Compression ──────────────────────────────────────────────────────
    "Huffman Coding": {
        "icon": "🌳", "tab": "Compression",
        "desc": "Huffman entropy coding. Variable-length codes assigned by pixel frequency — lossless.",
        "params": [],
    },
    "LZW Compression": {
        "icon": "📚", "tab": "Compression",
        "desc": "LZW dictionary-based lossless compression on flattened grayscale pixels.",
        "params": [],
    },
    "DPCM Encoding": {
        "icon": "📡", "tab": "Compression",
        "desc": "Differential PCM — encodes pixel differences rather than raw values. Lossless.",
        "params": [],
    },

    # ── New Padding ──────────────────────────────────────────────────────────
    "Symmetric Padding": {
        "icon": "↔️", "tab": "Transforms",
        "desc": "Mirror-reflect padding (cv2.BORDER_REFLECT) — includes the edge pixel itself.",
        "params": [
            {"key": "pad_top",    "label": "Top",    "type": "slider", "min": 0, "max": 150, "default": 20, "step": 1},
            {"key": "pad_bottom", "label": "Bottom", "type": "slider", "min": 0, "max": 150, "default": 20, "step": 1},
            {"key": "pad_left",   "label": "Left",   "type": "slider", "min": 0, "max": 150, "default": 20, "step": 1},
            {"key": "pad_right",  "label": "Right",  "type": "slider", "min": 0, "max": 150, "default": 20, "step": 1},
        ],
    },
    "Asymmetric Padding": {
        "icon": "⬛", "tab": "Transforms",
        "desc": "Zero-constant padding via numpy.pad with independently configurable per-side sizes.",
        "params": [
            {"key": "pad_top",    "label": "Top",    "type": "slider", "min": 0, "max": 150, "default": 10, "step": 1},
            {"key": "pad_bottom", "label": "Bottom", "type": "slider", "min": 0, "max": 150, "default": 30, "step": 1},
            {"key": "pad_left",   "label": "Left",   "type": "slider", "min": 0, "max": 150, "default": 10, "step": 1},
            {"key": "pad_right",  "label": "Right",  "type": "slider", "min": 0, "max": 150, "default": 30, "step": 1},
        ],
    },
    "Color Padding": {
        "icon": "🎨", "tab": "Transforms",
        "desc": "Constant-colour border — choose any BGR colour and border thickness per side.",
        "params": [
            {"key": "pad_top",    "label": "Top",    "type": "slider", "min": 0,   "max": 150, "default": 20,  "step": 1},
            {"key": "pad_bottom", "label": "Bottom", "type": "slider", "min": 0,   "max": 150, "default": 20,  "step": 1},
            {"key": "pad_left",   "label": "Left",   "type": "slider", "min": 0,   "max": 150, "default": 20,  "step": 1},
            {"key": "pad_right",  "label": "Right",  "type": "slider", "min": 0,   "max": 150, "default": 20,  "step": 1},
            {"key": "pad_b",      "label": "Blue",   "type": "slider", "min": 0,   "max": 255, "default": 128, "step": 1},
            {"key": "pad_g",      "label": "Green",  "type": "slider", "min": 0,   "max": 255, "default": 128, "step": 1},
            {"key": "pad_r",      "label": "Red",    "type": "slider", "min": 0,   "max": 255, "default": 128, "step": 1},
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 3 · HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_processor() -> ImageProcessor:
    return ImageProcessor()


def load_image(uploaded_file) -> np.ndarray:
    pil = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def image_info(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    ch   = img.shape[2] if img.ndim == 3 else 1
    return {"w": w, "h": h, "ch": ch, "kb": img.nbytes / 1024}


def render_image_card(img: np.ndarray, title: str, dot_color: str = "#58a6ff") -> None:
    """Render an image inside a macOS-style card."""
    st.markdown(f"""
    <div class="img-card">
        <div class="img-card-header">
            <div class="img-card-dot" style="background:{dot_color}"></div>
            <div class="img-card-dot" style="background:#d29922"></div>
            <div class="img-card-dot" style="background:#3fb950"></div>
            <span class="img-card-title" style="margin-left:6px;">{title}</span>
        </div>
        <div class="img-card-body">
    """, unsafe_allow_html=True)
    st.image(to_rgb(img), use_container_width=True)
    st.markdown("</div></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4 · PROCESSING DISPATCHER  (logic — no UI here)
# ══════════════════════════════════════════════════════════════════════════════

def run_processing(
    technique: str,
    image: np.ndarray,
    p: dict,
) -> tuple[np.ndarray, dict]:
    """
    Executes the selected technique and returns (result_image, extra_metrics).
    All processor.py calls are centralised here — no calls anywhere in the UI.
    """
    proc   = get_processor()
    extras = {}

    if technique == "Gaussian Blur":
        result = proc.apply_blur(image, p.get("blur_k", 5))

    elif technique == "Sharpen":
        result = proc.apply_sharpen(image, p.get("sharp_str", 1.0))

    elif technique == "Sobel Edges":
        result = proc.apply_sobel(image, p.get("sobel_k", 3), p.get("sobel_scale", 1.0))

    elif technique == "Canny Edges":
        result = proc.apply_canny(image, p.get("canny_t1", 50), p.get("canny_t2", 150))

    elif technique == "Padding":
        result = proc.apply_padding(
            image,
            p.get("pad_top", 20), p.get("pad_bottom", 20),
            p.get("pad_left", 20), p.get("pad_right", 20),
            p.get("pad_mode", "zero"),
        )

    elif technique == "Otsu Threshold":
        result = proc.apply_otsu_threshold(image)

    elif technique == "K-Means":
        result = proc.apply_kmeans(image, p.get("kmeans_k", 3))

    elif technique == "RLE Compression":
        gray_shape = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape
        encoded    = proc.rle_encode(image)
        ratio      = proc.rle_compression_ratio(image)
        result     = proc.rle_decode(encoded, gray_shape)
        extras = {
            "label":       "RLE",
            "orig_kb":     gray_shape[0] * gray_shape[1] / 1024,
            "encoded_kb":  len(encoded) * 8 / 1024,
            "ratio":       ratio,
        }

    # ── New Filters ──────────────────────────────────────────────────────────
    elif technique == "Mean Filter":
        result = proc.apply_mean_filter(image, p.get("mean_k", 5))

    elif technique == "Median Filter":
        result = proc.apply_median_filter(image, p.get("median_k", 5))

    elif technique == "Laplacian Edges":
        result = proc.apply_laplacian(image)

    elif technique == "Midpoint Filter":
        result = proc.apply_midpoint_filter(image, p.get("midpoint_k", 3))

    elif technique == "Alpha-Trimmed Mean":
        result = proc.apply_alpha_trimmed_mean(image, p.get("alpha_k", 5), p.get("alpha_d", 2))

    elif technique == "Harmonic Mean Filter":
        result = proc.apply_harmonic_mean_filter(image, p.get("harmonic_k", 5))

    elif technique == "Contraharmonic Filter":
        result = proc.apply_contraharmonic_mean_filter(image, p.get("contra_k", 5), p.get("contra_q", 1.5))

    elif technique == "Box Low-Pass Filter":
        result = proc.apply_box_lowpass_filter(image, p.get("box_k", 5))

    elif technique == "High-Pass Filter":
        result = proc.apply_highpass_filter(image, p.get("hp_k", 5))

    elif technique == "Prewitt Edges":
        result = proc.apply_prewitt_filter(image)

    # ── New Segmentation ─────────────────────────────────────────────────────
    elif technique == "Global Threshold":
        result = proc.apply_global_threshold(image, p.get("thresh_val", 127))

    elif technique == "Adaptive Threshold":
        result = proc.apply_adaptive_threshold(image, p.get("block_size", 11), p.get("thresh_c", 2))

    elif technique == "Multi-Otsu":
        result = proc.apply_multi_otsu(image, p.get("otsu_classes", 3))

    elif technique == "Region Growing":
        h, w   = image.shape[:2]
        seed_y = int(p.get("seed_y", 50) / 100 * h)
        seed_x = int(p.get("seed_x", 50) / 100 * w)
        result = proc.apply_region_growing(image, (seed_y, seed_x), p.get("rg_thresh", 10))

    elif technique == "Watershed":
        result = proc.apply_watershed(image)

    elif technique == "SLIC Superpixels":
        result = proc.apply_slic_superpixels(
            image, p.get("n_segments", 100), p.get("compactness", 10.0)
        )

    # ── New Compression ──────────────────────────────────────────────────────
    elif technique == "Huffman Coding":
        gray_shape                  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape
        encoded_bits, codebook, ratio = proc.huffman_encode(image)
        result                      = proc.huffman_decode(encoded_bits, codebook, gray_shape)
        extras = {
            "label":      "Huffman",
            "orig_kb":    gray_shape[0] * gray_shape[1] / 1024,
            "encoded_kb": len(encoded_bits) / 8 / 1024,
            "ratio":      ratio,
        }

    elif technique == "LZW Compression":
        gray_shape       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape
        codes, ratio     = proc.lzw_encode(image)
        result           = proc.lzw_decode(codes, gray_shape)
        extras = {
            "label":      "LZW",
            "orig_kb":    gray_shape[0] * gray_shape[1] / 1024,
            "encoded_kb": len(codes) * 12 / 8 / 1024,
            "ratio":      ratio,
        }

    elif technique == "DPCM Encoding":
        gray_shape        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape
        residuals, ratio  = proc.dpcm_encode(image)
        result            = proc.dpcm_decode(residuals, gray_shape)
        extras = {
            "label":      "DPCM",
            "orig_kb":    gray_shape[0] * gray_shape[1] / 1024,
            "encoded_kb": residuals.nbytes / 1024,
            "ratio":      ratio,
        }

    # ── New Padding ──────────────────────────────────────────────────────────
    elif technique == "Symmetric Padding":
        result = proc.apply_symmetric_padding(
            image,
            p.get("pad_top", 20), p.get("pad_bottom", 20),
            p.get("pad_left", 20), p.get("pad_right", 20),
        )

    elif technique == "Asymmetric Padding":
        result = proc.apply_asymmetric_padding(
            image,
            p.get("pad_top", 10), p.get("pad_bottom", 30),
            p.get("pad_left", 10), p.get("pad_right", 30),
        )

    elif technique == "Color Padding":
        result = proc.apply_color_padding(
            image,
            p.get("pad_top", 20), p.get("pad_bottom", 20),
            p.get("pad_left", 20), p.get("pad_right", 20),
            (p.get("pad_b", 128), p.get("pad_g", 128), p.get("pad_r", 128)),
        )

    else:
        result = image.copy()

    return result, extras


# ══════════════════════════════════════════════════════════════════════════════
# 5 · SIDEBAR  (ALL controls live here)
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    """Render sidebar and populate st.session_state with all user choices."""

    sb = st.sidebar

    # Logo
    sb.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-title">🔬 VisionLab</div>
        <div class="sb-logo-sub">Image Processing Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1 · Upload ────────────────────────────────────────────────────
    sb.markdown('<div class="sb-section-label">① Upload</div>', unsafe_allow_html=True)
    uploaded = sb.file_uploader(
        "Drop image here",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"],
        label_visibility="collapsed",
    )

    # ── Step 2 · Select technique ──────────────────────────────────────────
    sb.markdown('<div class="sb-section-label">② Technique</div>', unsafe_allow_html=True)
    technique = sb.selectbox(
        "Technique", list(TECHNIQUES.keys()),
        label_visibility="collapsed",
        key="widget_technique",
    )
    meta = TECHNIQUES[technique]
    sb.markdown(
        f"<div style='font-size:.72rem;color:#484f58;line-height:1.5;margin-top:4px;'>"
        f"{meta['icon']} <em>{meta['desc'][:90]}{'…' if len(meta['desc'])>90 else ''}</em></div>",
        unsafe_allow_html=True,
    )

    # ── Step 3 · Configure parameters ─────────────────────────────────────
    if meta["params"]:
        sb.markdown('<div class="sb-section-label">③ Parameters</div>', unsafe_allow_html=True)
        collected: dict = {}
        for p in meta["params"]:
            if p["type"] == "slider":
                collected[p["key"]] = sb.slider(
                    p["label"], p["min"], p["max"], p["default"], step=p["step"], key=p["key"]
                )
            elif p["type"] == "slider_float":
                collected[p["key"]] = sb.slider(
                    p["label"], float(p["min"]), float(p["max"]),
                    float(p["default"]), step=float(p["step"]), key=p["key"]
                )
            elif p["type"] == "select":
                opts = p["options"]
                idx  = opts.index(p["default"]) if p["default"] in opts else 0
                collected[p["key"]] = sb.selectbox(p["label"], opts, index=idx, key=p["key"])
        st.session_state["sel_params"] = collected
    else:
        st.session_state["sel_params"] = {}
        sb.markdown(
            "<div style='font-size:.72rem;color:#484f58;margin-top:.5rem;'>"
            "No parameters needed for this technique.</div>",
            unsafe_allow_html=True,
        )

    # ── Step 4 · Apply ─────────────────────────────────────────────────────
    sb.markdown('<div class="sb-section-label">④ Apply</div>', unsafe_allow_html=True)

    apply_clicked = sb.button("⚡  Apply Technique", key="apply_btn")

    # Store upload & technique in session state (use non-widget keys)
    st.session_state["sel_uploaded"]  = uploaded
    st.session_state["sel_technique"] = technique

    # Run processing ONLY when Apply is clicked (logic decoupled from display)
    if apply_clicked:
        if uploaded is None:
            st.session_state["error"] = "Please upload an image first."
        else:
            image = load_image(uploaded)
            with st.spinner("Processing…"):
                result, extras = run_processing(
                    technique, image, st.session_state.get("sel_params", {})
                )
            st.session_state["original"] = image
            st.session_state["result"]   = result
            st.session_state["extras"]   = extras
            st.session_state["error"]    = None

    # Footer
    sb.markdown("---")
    sb.markdown(
        "<div style='font-size:.65rem;color:#30363d;text-align:center;'>"
        "VisionLab v2.0 · OpenCV + Streamlit</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6 · HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════

def render_hero() -> None:
    uploaded  = st.session_state.get("sel_uploaded")
    has_img   = uploaded is not None
    has_res   = st.session_state.get("result") is not None
    technique = st.session_state.get("sel_technique", "—")
    meta      = TECHNIQUES.get(technique, {})

    img_pill = (
        '<span class="pill pill-green">● Image Loaded</span>'
        if has_img else
        '<span class="pill pill-yellow">○ No Image</span>'
    )
    res_pill = (
        f'<span class="pill pill-blue">✓ Result Ready</span>'
        if has_res else
        '<span class="pill pill-purple">◌ Awaiting Processing</span>'
    )

    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">Computer Vision · OpenCV · Streamlit</div>
        <div class="hero-title">Image Processing Dashboard</div>
        <div class="hero-sub">
            Upload an image, choose a technique, configure its parameters in the sidebar, then hit
            <strong style="color:#58a6ff">Apply</strong> — results appear instantly across all tabs.
        </div>
        <div class="hero-badges">
            {img_pill}
            {res_pill}
            <span class="pill pill-purple">{meta.get('icon','')} {technique}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Step flow indicator
    s1 = "step-done"   if has_img else "step-active"
    s2 = "step-done"   if has_img else "step-idle"
    s3 = "step-active" if (has_img and not has_res) else ("step-done" if has_res else "step-idle")
    s4 = "step-done"   if has_res else "step-idle"
    st.markdown(f"""
    <ul class="step-flow">
        <li class="step {s1}"><div class="step-num">1</div><div class="step-label">Upload</div></li>
        <div class="step-sep"></div>
        <li class="step {s2}"><div class="step-num">2</div><div class="step-label">Select</div></li>
        <div class="step-sep"></div>
        <li class="step {s3}"><div class="step-num">3</div><div class="step-label">Process</div></li>
        <div class="step-sep"></div>
        <li class="step {s4}"><div class="step-num">4</div><div class="step-label">Compare</div></li>
    </ul>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7a · TAB "View"  — original image + metadata
# ══════════════════════════════════════════════════════════════════════════════

def tab_view() -> None:
    uploaded = st.session_state.get("sel_uploaded")
    if uploaded is None:
        st.info("👈 Upload an image from the sidebar to get started.", icon="ℹ️")
        return

    image = load_image(uploaded)
    info  = image_info(image)

    st.subheader("Image Metadata")
    st.markdown('<div class="stats-row">', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4, gap="small")
    with m1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="sc-icon">📐</div>
            <div class="sc-label">Width</div>
            <div class="sc-value">{info['w']}</div>
            <div class="sc-sub">pixels</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="sc-icon">📏</div>
            <div class="sc-label">Height</div>
            <div class="sc-value">{info['h']}</div>
            <div class="sc-sub">pixels</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="sc-icon">🎨</div>
            <div class="sc-label">Channels</div>
            <div class="sc-value">{info['ch']}</div>
            <div class="sc-sub">colour depth</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="sc-icon">💾</div>
            <div class="sc-label">Raw Size</div>
            <div class="sc-value">{info['kb']:.0f}</div>
            <div class="sc-sub">kilobytes</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Full image preview
    st.subheader("Original Image")
    render_image_card(image, "Source · " + uploaded.name, "#3fb950")

    st.divider()

    # Technique reference
    with st.expander("📖  Technique Reference", expanded=False):
        st.markdown("""
| Technique | Category | Description |
|:---|:---|:---|
| **Gaussian Blur** | Filter | Smooth / noise reduction |
| **Sharpen** | Filter | Detail enhancement |
| **Sobel Edges** | Edge Detection | Gradient magnitude |
| **Canny Edges** | Edge Detection | Optimal multi-stage detector |
| **Padding** | Transform | Border extension |
| **Otsu Threshold** | Segmentation | Auto binary threshold |
| **K-Means** | Segmentation | Colour cluster segmentation |
| **RLE Compression** | Compression | Lossless run-length encoding |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# 7b · TAB "Process"  — result image + technique summary
# ══════════════════════════════════════════════════════════════════════════════

def tab_process() -> None:
    result    = st.session_state.get("result")
    technique = st.session_state.get("sel_technique", "")
    extras    = st.session_state.get("extras", {})
    meta      = TECHNIQUES.get(technique, {})

    if result is None:
        if st.session_state.get("error"):
            st.error(st.session_state["error"], icon="🚫")
        else:
            st.info("Hit **Apply** in the sidebar to run processing.", icon="⚡")
        return

    # Technique card
    params_used = st.session_state.get("sel_params", {})
    params_str  = "  ·  ".join(f"{k} = {v}" for k, v in params_used.items()) or "No parameters"
    st.markdown(f"""
    <div class="technique-card">
        <div class="tc-name">{meta.get('icon','')}  {technique}</div>
        <div class="tc-desc">{meta.get('desc','')}</div>
        <div style="margin-top:10px;font-size:.7rem;color:#484f58;">
            Parameters: {params_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Result + output stats side-by-side
    col_img, col_stats = st.columns([3, 1], gap="large")

    with col_img:
        st.subheader("Result")
        render_image_card(result, f"{technique} · Output", "#bc8cff")

    with col_stats:
        st.subheader("Output Info")
        info = image_info(result)
        st.metric("Width",    f"{info['w']} px")
        st.metric("Height",   f"{info['h']} px")
        st.metric("Size",     f"{info['kb']:.0f} KB")

        # Compression-specific metrics (RLE / Huffman / LZW / DPCM)
        if extras:
            st.divider()
            st.subheader(f"{extras.get('label','Compression')} Metrics")
            st.metric("Orig Size",    f"{extras['orig_kb']:.1f} KB")
            st.metric("Encoded Size", f"{extras['encoded_kb']:.1f} KB")
            ratio = extras["ratio"]
            st.metric("Ratio", f"{ratio:.2f}×",
                      delta="effective" if ratio >= 1 else "expanded",
                      delta_color="normal" if ratio >= 1 else "inverse")

    st.divider()

    with st.expander("🔬  Technical Details", expanded=False):
        st.markdown(f"""
**Technique:** {technique}  
**Category:** {meta.get('tab','—')}  
**Parameters used:**
```
{params_str}
```
**Output shape:** {result.shape}  
**Output dtype:** {result.dtype}
        """)


# ══════════════════════════════════════════════════════════════════════════════
# 7c · TAB "Compare"  — side-by-side diff
# ══════════════════════════════════════════════════════════════════════════════

def tab_compare() -> None:
    original  = st.session_state.get("original")
    result    = st.session_state.get("result")
    technique = st.session_state.get("sel_technique", "")

    if original is None or result is None:
        st.info("Both an uploaded image and a processed result are needed for comparison.", icon="ℹ️")
        return

    st.subheader("Side-by-Side Comparison")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="compare-label">ORIGINAL</div>', unsafe_allow_html=True)
        render_image_card(original, "Original", "#3fb950")

    with col2:
        st.markdown(f'<div class="compare-label">{technique.upper()}</div>', unsafe_allow_html=True)
        render_image_card(result, technique, "#bc8cff")

    st.divider()

    # Delta metrics
    st.subheader("Delta Metrics")
    orig_info = image_info(original)
    res_info  = image_info(result)

    dm1, dm2, dm3, dm4 = st.columns(4, gap="small")
    dm1.metric("Orig Width",    f"{orig_info['w']} px")
    dm2.metric("Result Width",  f"{res_info['w']} px",
               delta=f"{res_info['w'] - orig_info['w']:+d} px")
    dm3.metric("Orig Height",   f"{orig_info['h']} px")
    dm4.metric("Result Height", f"{res_info['h']} px",
               delta=f"{res_info['h'] - orig_info['h']:+d} px")

    st.divider()

    # Pixel diff (only if shapes match)
    if original.shape == result.shape:
        with st.expander("🔍  Pixel Difference Map", expanded=False):
            diff    = cv2.absdiff(original, result)
            diff_en = cv2.convertScaleAbs(diff, alpha=5)   # amplify for visibility
            render_image_card(diff_en, "Difference (5× amplified)", "#f85149")
            mean_diff = float(np.mean(diff))
            st.caption(f"Mean absolute pixel difference: **{mean_diff:.2f}**")
    else:
        with st.expander("ℹ️  Pixel Difference", expanded=False):
            st.info(
                "Pixel diff is unavailable — the output dimensions differ from the input "
                "(e.g. Padding changes image size).",
                icon="ℹ️",
            )


# ══════════════════════════════════════════════════════════════════════════════
# 8 · MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Initialize session state keys to avoid KeyError on first run
    for key in ("sel_uploaded", "original", "result", "extras", "error", "sel_params"):
        st.session_state.setdefault(key, None)

    # Sidebar — all controls
    render_sidebar()

    # Hero + step flow
    render_hero()

    # Tabs
    t_view, t_process, t_compare = st.tabs([
        "🖼️  View",
        "⚡  Process",
        "🔍  Compare",
    ])

    with t_view:
        tab_view()

    with t_process:
        tab_process()

    with t_compare:
        tab_compare()


if __name__ == "__main__":
    main()
