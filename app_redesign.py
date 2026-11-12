"""
Lingala Intent Classifier - Complete Redesign
Team 07 Datalynk | Strathmore NLP Hub Datathon 2026
Track A — Intent Classification
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from nlp.availability import probe_status
from nlp.config import (
    DATATHON_NAME,
    SUPPORTED_TRACKS,
    TARGET_LANGUAGE,
    TEAM_NAME,
    TEAM_NUMBER,
    artifact_dir,
)
from nlp.data_loader import load_dev_metrics, load_test_predictions
from nlp.predictor import predict
from nlp.summary import load_contribution, summary
from nlp.ui import build_disclaimer, confidence_band, format_probability

TRACK = "intent"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Lingala Intent Classifier | Team 07 Datalynk",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# AUTHENTICATION MODULE
# ============================================================================

def init_auth() -> None:
    """Initialize authentication database."""
    from nlp.auth import init_db
    init_db()


def check_auth() -> bool:
    """Check if user is authenticated."""
    return st.session_state.get("authenticated", False)


def get_current_user() -> dict[str, Any] | None:
    """Get current user data."""
    return st.session_state.get("user")


def authenticate(username: str, password: str) -> tuple[bool, str]:
    """Authenticate user and return success status with message."""
    from nlp.auth import authenticate_user
    success, user_data, message = authenticate_user(username, password)
    if success:
        st.session_state["authenticated"] = True
        st.session_state["user"] = user_data
        st.session_state["username"] = username
    return success, message


def register_user(username: str, email: str, password: str, full_name: str) -> tuple[bool, str]:
    """Register a new user."""
    from nlp.auth import create_user
    success, message = create_user(username, email, password, full_name)
    return success, message


def logout() -> None:
    """Log out current user."""
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    st.session_state["username"] = None


def guest_login() -> None:
    """Login as guest user."""
    st.session_state["authenticated"] = True
    st.session_state["user"] = {"username": "guest", "full_name": "Guest User"}
    st.session_state["username"] = "guest"


# ============================================================================
# DATA & HELPERS
# ============================================================================

@st.cache_data
def get_taxonomy() -> pd.DataFrame:
    path = Path("taxonomy.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


@st.cache_data
def get_contribution() -> pd.DataFrame:
    return load_contribution()


@st.cache_data
def get_summary() -> dict[str, Any]:
    return summary()


@st.cache_data
def get_test_metrics(track: str) -> dict[str, Any] | None:
    path = artifact_dir(track) / "test_metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data
def get_dev_metrics(track: str) -> dict[str, Any] | None:
    try:
        return load_dev_metrics(track)
    except Exception:
        return None


@st.cache_data
def get_test_predictions(track: str) -> pd.DataFrame | None:
    try:
        return load_test_predictions(track)
    except Exception:
        return None


@st.cache_data
def get_intent_info() -> dict[str, dict[str, str]]:
    """Intent meanings/boundaries sourced from taxonomy and annotation guide."""
    info: dict[str, dict[str, str]] = {
        "weather_query": {
            "name": "Weather enquiry",
            "meaning": "Asks about weather conditions or forecasts.",
            "boundary": "Asking for news about a storm is news_query.",
            "example": "Mbula ekobeta lobi?",
        },
        "datetime_query": {
            "name": "Date or time enquiry",
            "meaning": "Asks for the current date or time.",
            "boundary": "Setting an alarm is alarm_set.",
            "example": "Ngonga ezali boni?",
        },
        "alarm_set": {
            "name": "Set alarm",
            "meaning": "Requests setting an alarm to ring at a specific time.",
            "boundary": "Scheduling a meeting is calendar_set.",
            "example": "Tya alarme na ngonga ya motoba.",
        },
        "calendar_set": {
            "name": "Create calendar event",
            "meaning": "Requests creating a calendar event or scheduled reminder.",
            "boundary": "Reading existing events is calendar_query.",
            "example": "Bakisa likita na ngai na kalandriye.",
        },
        "calendar_query": {
            "name": "Read calendar",
            "meaning": "Asks about existing calendar events or schedule.",
            "boundary": "Creating a new event is calendar_set.",
            "example": "Nazali na makita nini lelo?",
        },
        "lists_createoradd": {
            "name": "Create or add to list",
            "meaning": "Requests creating a list or adding an item to a list.",
            "boundary": "Reading an existing list is lists_query.",
            "example": "Bakisa mapa na liste ya biloko ya kosomba.",
        },
        "lists_query": {
            "name": "Read list",
            "meaning": "Asks what is already in an existing list.",
            "boundary": "Adding an item is lists_createoradd.",
            "example": "Nini ezali na liste na ngai ya kosomba?",
        },
        "play_music": {
            "name": "Play music",
            "meaning": "Requests music playback.",
            "boundary": "Asking who wrote a song is not a playback request.",
            "example": "Yemba miziki.",
        },
        "news_query": {
            "name": "News enquiry",
            "meaning": "Asks for news updates or headlines.",
            "boundary": "Forecast requests are weather_query.",
            "example": "Yebisa ngai bansango ya sika.",
        },
        "qa_definition": {
            "name": "Word or concept meaning",
            "meaning": "Asks for the meaning of a word or concept.",
            "boundary": "General factual questions outside definitions are out of scope.",
            "example": "Liloba biodiversité elimboli nini?",
        },
    }
    tax = get_taxonomy()
    if not tax.empty:
        for _, row in tax.iterrows():
            label = row.get("intent", "")
            if label in info:
                info[label]["meaning"] = row.get("working_definition", info[label]["meaning"])
                info[label]["boundary"] = row.get("boundary", info[label]["boundary"])
    return info


@st.cache_resource
def get_backend(track: str) -> dict[str, Any]:
    """Cache model pack/status; only reload when process restarts."""
    return probe_status(track)


def intent_name(label: str) -> str:
    return get_intent_info().get(label, {}).get("name", label)


def _run_prediction(text: str, pack: dict[str, Any]) -> dict[str, Any] | None:
    """Run prediction and return result with probabilities."""
    try:
        result = predict(text, pack)
        proba = None
        if hasattr(pack["model"], "predict_proba"):
            try:
                raw = pack["model"].predict_proba([text])[0]
                classes = list(pack["model"].classes_)
                proba = sorted(
                    [{"label": c, "probability": float(p)} for c, p in zip(classes, raw)],
                    key=lambda x: x["probability"],
                    reverse=True,
                )
            except Exception:
                proba = None
        return {"result": result, "proba": proba}
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return None


# ============================================================================
# DESIGN SYSTEM & STYLES
# ============================================================================

def inject_css() -> None:
    """Inject custom CSS for professional SaaS-style design."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        :root {
            --bg-primary: #FAFBFC;
            --bg-white: #FFFFFF;
            --text-primary: #192A38;
            --text-secondary: #667785;
            --border-color: #E6EBEF;
            --accent-teal: #16A394;
            --accent-teal-dark: #148A7D;
            --navy-primary: #142B3B;
            --navy-secondary: #203E50;
            --success: #17845D;
            --warning: #C18B32;
            --error: #C64B4B;
        }
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background: var(--bg-primary);
            color: var(--text-primary);
        }
        
        /* Navigation */
        [data-testid="stSidebar"] {
            background: var(--bg-white);
            border-right: 1px solid var(--border-color);
        }
        
        [data-testid="stSidebar"] * {
            color: var(--text-primary);
        }
        
        [data-testid="stSidebar"] hr {
            border-color: var(--border-color);
        }
        
        /* Branding */
        .brand-block {
            padding: 1.5rem 1rem 2rem 1rem;
        }
        
        .brand-logo {
            font-size: 1.75rem;
            font-weight: 800;
            color: var(--navy-primary);
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }
        
        .brand-sub {
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        .brand-badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: #E0F2F1;
            color: var(--accent-teal);
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1rem;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.6rem 1.25rem;
            transition: all 0.2s ease;
        }
        
        .stButton > button[kind="primary"] {
            background: var(--accent-teal);
            color: var(--bg-white);
            border: none;
            box-shadow: 0 2px 8px rgba(22, 163, 148, 0.25);
        }
        
        .stButton > button[kind="primary"]:hover {
            background: var(--accent-teal-dark);
            box-shadow: 0 4px 12px rgba(22, 163, 148, 0.35);
            transform: translateY(-1px);
        }
        
        .stButton > button[kind="secondary"] {
            background: var(--bg-white);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: #F1F5F9;
            border-color: var(--text-secondary);
        }
        
        .stButton > button[kind="ghost"] {
            background: transparent;
            color: var(--text-secondary);
            border: none;
        }
        
        .stButton > button[kind="ghost"]:hover {
            background: #F8FAFC;
            color: var(--text-primary);
        }
        
        /* Cards */
        .card {
            background: var(--bg-white);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        
        .card h3 {
            color: var(--navy-primary);
            margin-top: 0;
            margin-bottom: 1rem;
            font-size: 1.125rem;
            font-weight: 700;
        }
        
        /* Hero section */
        .hero {
            background: linear-gradient(135deg, var(--navy-primary) 0%, var(--navy-secondary) 100%);
            color: var(--bg-white);
            padding: 4rem 2rem;
            border-radius: 24px;
            margin-bottom: 3rem;
            box-shadow: 0 10px 40px rgba(20, 43, 59, 0.15);
        }
        
        .hero h1 {
            margin: 0 0 1rem 0;
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }
        
        .hero p {
            color: #CBD5E1;
            font-size: 1.125rem;
            line-height: 1.6;
            margin: 0;
        }
        
        .hero-cta {
            display: flex;
            gap: 1rem;
            margin-top: 2rem;
            flex-wrap: wrap;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--navy-primary);
            font-weight: 700;
            line-height: 1.2;
        }
        
        h1 { font-size: 2.5rem; margin-bottom: 1rem; }
        h2 { font-size: 2rem; margin-top: 2.5rem; margin-bottom: 1rem; }
        h3 { font-size: 1.5rem; margin-top: 2rem; margin-bottom: 0.75rem; }
        h4 { font-size: 1.25rem; margin-top: 1.5rem; margin-bottom: 0.5rem; }
        
        p {
            color: var(--text-primary);
            line-height: 1.6;
            margin-bottom: 1rem;
        }
        
        /* Inputs */
        .stTextArea textarea, .stTextInput input {
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 0.75rem 1rem;
            font-size: 1rem;
        }
        
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: var(--accent-teal);
            box-shadow: 0 0 0 3px rgba(22, 163, 148, 0.15);
        }
        
        /* Tables */
        .dataframe {
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        .dataframe th {
            background: #F8FAFC;
            color: var(--text-primary);
            font-weight: 600;
            border-bottom: 2px solid var(--border-color);
        }
        
        .dataframe td {
            border-bottom: 1px solid var(--border-color);
        }
        
        /* Metrics */
        .metric-card {
            background: var(--bg-white);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--navy-primary);
            line-height: 1;
        }
        
        .metric-label {
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }
        
        /* Badges */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .badge-teal { background: #E0F2F1; color: var(--accent-teal); }
        .badge-blue { background: #E3F2FD; color: #1976D2; }
        .badge-warning { background: #FFF3E0; color: var(--warning); }
        
        /* Workflow steps */
        .workflow {
            display: flex;
            gap: 2rem;
            margin: 2rem 0;
            flex-wrap: wrap;
        }
        
        .workflow-step {
            flex: 1;
            min-width: 200px;
            text-align: center;
            padding: 1.5rem;
            background: var(--bg-white);
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }
        
        .workflow-icon {
            font-size: 2rem;
            margin-bottom: 0.75rem;
            color: var(--accent-teal);
        }
        
        .workflow-label {
            font-weight: 600;
            color: var(--navy-primary);
        }
        
        .workflow-arrow {
            font-size: 1.5rem;
            color: var(--text-secondary);
            align-self: center;
        }
        
        /* Footer */
        .footer {
            margin-top: 4rem;
            padding: 2rem 0;
            border-top: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 0.875rem;
        }
        
        /* Responsive */
        @media (max-width: 1024px) {
            .hero h1 { font-size: 2.5rem; }
        }
        
        @media (max-width: 768px) {
            .hero { padding: 2rem 1rem; }
            .hero h1 { font-size: 2rem; }
            .workflow { flex-direction: column; }
            .workflow-arrow { display: none; }
            h1 { font-size: 2rem; }
            h2 { font-size: 1.5rem; }
            .hero-cta { flex-direction: column; }
            .hero-cta button { width: 100%; }
        }
        
        @media (max-width: 480px) {
            .hero h1 { font-size: 1.75rem; }
            .workflow-step { padding: 1rem; }
        }
        
        /* Navigation sidebar */
        .nav-section {
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
        }
        
        .nav-section-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            font-weight: 700;
            margin-bottom: 0.75rem;
            padding-left: 0.5rem;
        }
        
        .nav-item {
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            margin-bottom: 0.25rem;
        }
        
        .nav-item:hover {
            background: #F8FAFC;
        }
        
        .nav-item.active {
            background: #E0F2F1;
            color: var(--accent-teal);
            font-weight: 600;
        }
        
        /* Result card */
        .result-card {
            background: var(--bg-white);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--accent-teal);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1.5rem;
        }
        
        .result-card h4 {
            color: var(--accent-teal);
            margin-top: 0;
        }
        
        /* Practice warning */
        .practice-warning {
            background: #FFF3E0;
            border: 1px solid var(--warning);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .practice-warning p {
            margin: 0;
            color: var(--warning);
            font-weight: 500;
        }
        
        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 3rem 1rem;
            color: var(--text-secondary);
        }
        
        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }
        
        /* Form validation */
        .stError {
            background: #FEF2F2;
            border: 1px solid var(--error);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .stSuccess {
            background: #F0FDF4;
            border: 1px solid var(--success);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()
