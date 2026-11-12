"""
Lingala Intent Classifier — Team 07 Datalynk
Strathmore NLP Hub Datathon 2026 | Track A — Intent Classification

Entry point:
    python -m streamlit run app.py

A working NLP system that accepts Lingala text and returns a real predicted
intent using the project's trained character-TF-IDF + logistic regression model.

Public routes: landing, login, register, classifier (guest-accessible).
Authenticated workspace: overview, classifier (prominent), intent boundary lab,
dataset explorer, model performance, error analysis, research information, logout.

No raw JavaScript is rendered as content. All predictions come from the trained
model pack; no demonstration, hardcoded or fabricated scores are shown.
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
from nlp.data_loader import load_dev_metrics, load_test_predictions, load_model_pack
from nlp.predictor import predict
from nlp.summary import load_contribution, summary
from nlp.ui import build_disclaimer, confidence_band, format_probability

TRACK = "intent"

# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data & helpers
# ---------------------------------------------------------------------------

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
    """Run prediction and return result with real probabilities."""
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


# ---------------------------------------------------------------------------
# Design system & styles
# ---------------------------------------------------------------------------

def inject_css() -> None:
    """Inject custom CSS for a clean, minimalist design system."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-primary: #F7F9FB;
            --bg-white: #FFFFFF;
            --text-primary: #1A2B3C;
            --text-secondary: #5B6B7B;
            --border-color: #DCE3E9;
            --accent-teal: #0E7C7B;
            --accent-teal-dark: #0B5F5E;
            --navy-primary: #142B3B;
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

        [data-testid="stSidebar"] {
            background: var(--bg-white);
            border-right: 1px solid var(--border-color);
        }

        /* Branding */
        .brand-block {
            padding: 1.25rem 1rem 1.5rem 1rem;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1rem;
        }

        .brand-logo {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--navy-primary);
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }

        .brand-sub {
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .brand-badge {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            background: #E3F2F2;
            color: var(--accent-teal);
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.75rem;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.6rem 1.25rem;
            transition: all 0.2s ease;
            border: 1px solid var(--border-color);
            background: var(--bg-white);
            color: var(--text-primary);
        }

        .stButton > button[kind="primary"] {
            background: var(--accent-teal);
            color: var(--bg-white);
            border: none;
        }

        .stButton > button[kind="primary"]:hover {
            background: var(--accent-teal-dark);
        }

        .stButton > button[kind="secondary"]:hover {
            background: #F1F5F9;
            border-color: var(--text-secondary);
        }

        /* Cards */
        .card {
            background: var(--bg-white);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
        }

        .card h3 {
            color: var(--navy-primary);
            margin-top: 0;
            margin-bottom: 1rem;
            font-size: 1.125rem;
            font-weight: 700;
        }

        /* Hero */
        .hero {
            background: linear-gradient(135deg, var(--navy-primary) 0%, #1E4A5F 100%);
            color: var(--bg-white);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 2.5rem;
        }

        .hero h1 {
            margin: 0 0 1rem 0;
            font-size: 2.75rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .hero p {
            color: #D8E2EC;
            font-size: 1.125rem;
            line-height: 1.6;
            margin: 0 0 1.5rem 0;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--navy-primary);
            font-weight: 700;
            line-height: 1.2;
        }

        h1 { font-size: 2rem; margin-bottom: 1rem; }
        h2 { font-size: 1.6rem; margin-top: 2rem; margin-bottom: 1rem; }
        h3 { font-size: 1.3rem; margin-top: 1.5rem; margin-bottom: 0.75rem; }

        p {
            color: var(--text-primary);
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        .stTextArea textarea, .stTextInput input {
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 0.75rem 1rem;
            font-size: 1rem;
        }

        /* Metrics */
        .metric-card {
            background: var(--bg-white);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: var(--navy-primary);
            line-height: 1;
        }

        .metric-label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }

        .badge-teal { background: #E3F2F2; color: var(--accent-teal); }
        .badge-navy { background: #E8ECF1; color: var(--navy-primary); }
        .badge-warning { background: #F7EEDB; color: var(--warning); }

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

        /* Footer */
        .footer {
            margin-top: 3rem;
            padding: 2rem 0;
            border-top: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 0.85rem;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .hero { padding: 2rem 1rem; }
            .hero h1 { font-size: 2rem; }
            h1 { font-size: 1.75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ---------------------------------------------------------------------------
# Shared header (public pages)
# ---------------------------------------------------------------------------

def render_public_header() -> None:
    """Minimal header with brand and working Register/Login links."""
    c1, c2, c3 = st.columns([2.2, 1, 1])
    with c1:
        st.markdown(
            '<div class="brand-block">'
            '<div class="brand-logo">Lingala</div>'
            '<div class="brand-sub">Intent Classifier</div>'
            '<div class="brand-sub" style="margin-top:0.25rem;">Team 07 — Datalynk</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("Register", key="hdr_register", use_container_width=True):
            st.session_state["page"] = "register"
            st.rerun()
    with c3:
        if st.button("Login", key="hdr_login", use_container_width=True):
            st.session_state["page"] = "login"
            st.rerun()


def render_footer(authenticated: bool) -> None:
    """Footer with working Register/Login links and project identity."""
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<strong>Lingala Intent Classifier</strong><br>"
            f"Team 07 — {TEAM_NAME}<br>"
            f"{DATATHON_NAME}<br>"
            f"Track A — Intent Classification<br>"
            f"Model language: {TARGET_LANGUAGE}",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<a href="#research" style="color: var(--text-secondary); text-decoration: none; margin-right: 1rem;">Research</a>'
            '<a href="#about" style="color: var(--text-secondary); text-decoration: none; margin-right: 1rem;">About</a>',
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Register", key="ftr_register", use_container_width=True):
                st.session_state["page"] = "register"
                st.rerun()
        with col_b:
            if st.button("Login", key="ftr_login", use_container_width=True):
                st.session_state["page"] = "login"
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Landing page (public)
# ---------------------------------------------------------------------------

def page_landing() -> None:
    """Public landing page — product identity, motivation, how it works, stats."""
    render_public_header()

    # Hero
    st.markdown(
        """
        <div class="hero">
            <div style="max-width: 800px;">
                <span style="display: inline-block; background: rgba(255,255,255,0.15); padding: 0.35rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; margin-bottom: 1.25rem;">
                    AFRICAN LANGUAGE TECHNOLOGY &middot; NLP RESEARCH
                </span>
                <h1>Understanding the intent behind Lingala text.</h1>
                <p>
                    An experimental AI research system that identifies the communicative
                    intent in a Lingala sentence using a transparent, trained baseline model.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown("### Why Lingala NLP?")
        st.write(
            "Low-resource language NLP faces data scarcity, limited tooling, and "
            "evaluation challenges. Building intent classification resources for Lingala "
            "supports exploration of everyday language technology for African languages "
            "and highlights practical constraints in small-data settings."
        )
        st.write(
            "This research prototype explores how machine learning can classify everyday "
            "Lingala requests using a small, reviewed dataset and a transparent baseline model."
        )

        st.markdown("### How it works")
        st.markdown(
            """
            <div style="display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap;">
                <div class="card" style="flex: 1; min-width: 180px; text-align: center; padding: 1rem;">
                    <div style="font-size: 1.75rem; color: var(--accent-teal); margin-bottom: 0.5rem;">Input</div>
                    <div style="font-weight: 600; color: var(--navy-primary); margin-bottom: 0.25rem;">Enter text</div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">Type a Lingala sentence</div>
                </div>
                <div style="display: flex; align-items: center; color: var(--text-secondary);">&#8594;</div>
                <div class="card" style="flex: 1; min-width: 180px; text-align: center; padding: 1rem;">
                    <div style="font-size: 1.75rem; color: var(--accent-teal); margin-bottom: 0.5rem;">Model</div>
                    <div style="font-weight: 600; color: var(--navy-primary); margin-bottom: 0.25rem;">Character TF-IDF</div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">+ Logistic Regression</div>
                </div>
                <div style="display: flex; align-items: center; color: var(--text-secondary);">&#8594;</div>
                <div class="card" style="flex: 1; min-width: 180px; text-align: center; padding: 1rem;">
                    <div style="font-size: 1.75rem; color: var(--accent-teal); margin-bottom: 0.5rem;">Prediction</div>
                    <div style="font-weight: 600; color: var(--navy-primary); margin-bottom: 0.25rem;">Intent label</div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">One of 10 categories</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Project snapshot")
        s = get_summary()
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{s.get("total", 0) if s.get("exists") else 0}</div>'
                f'<div class="metric-label">Reviewed records</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                '<div class="metric-card">'
                '<div class="metric-value">10</div>'
                '<div class="metric-label">Intent categories</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{s.get("language", TARGET_LANGUAGE) if s.get("exists") else TARGET_LANGUAGE}</div>'
                f'<div class="metric-label">Training language</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### Try the classifier")
        st.write(
            "Enter any Lingala sentence and receive a real predicted intent from the "
            "trained model — no account required."
        )
        if st.button("Open the classifier", key="landing_open_classifier", type="primary", use_container_width=True):
            st.session_state["page"] = "classifier"
            st.rerun()

    # Route into classifier
    st.markdown("---")
    st.markdown("### Get started")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Create account", key="cta_register", use_container_width=True):
            st.session_state["page"] = "register"
            st.rerun()
    with c2:
        if st.button("Sign in", key="cta_login", use_container_width=True):
            st.session_state["page"] = "login"
            st.rerun()
    with c3:
        if st.button("Continue as guest", key="cta_guest", use_container_width=True):
            guest_login()
            st.session_state["page"] = "classifier"
            st.rerun()

    render_footer(False)


# ---------------------------------------------------------------------------
# Login / Register
# ---------------------------------------------------------------------------

def page_login() -> None:
    """Login page."""
    c1, c2, c3 = st.columns([2.2, 1, 1])
    with c1:
        st.markdown(
            '<div class="brand-block">'
            '<div class="brand-logo">Lingala</div>'
            '<div class="brand-sub">Intent Classifier</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("Register", key="lgn_hdr_register", use_container_width=True):
            st.session_state["page"] = "register"
            st.rerun()
    with c3:
        if st.button("Home", key="lgn_hdr_home", use_container_width=True):
            st.session_state["page"] = "landing"
            st.rerun()

    st.markdown("### Sign in")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username or email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if submitted:
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            success, message = authenticate(username, password)
            if success:
                st.success("Login successful!")
                st.session_state["page"] = "workspace"
                st.rerun()
            else:
                st.error(message)

    st.markdown(
        '<div style="margin-top: 1rem; text-align: center;">'
        '<span style="color: var(--text-secondary); font-size: 0.875rem;">'
        "Don't have an account? </span>",
        unsafe_allow_html=True,
    )
    if st.button("Create one", key="lgn_create"):
        st.session_state["page"] = "register"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    render_footer(False)


def page_register() -> None:
    """Registration page."""
    c1, c2, c3 = st.columns([2.2, 1, 1])
    with c1:
        st.markdown(
            '<div class="brand-block">'
            '<div class="brand-logo">Lingala</div>'
            '<div class="brand-sub">Intent Classifier</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("Login", key="reg_hdr_login", use_container_width=True):
            st.session_state["page"] = "login"
            st.rerun()
    with c3:
        if st.button("Home", key="reg_hdr_home", use_container_width=True):
            st.session_state["page"] = "landing"
            st.rerun()

    st.markdown("### Create account")
    with st.form("register_form", clear_on_submit=False):
        full_name = st.text_input("Full name")
        username = st.text_input("Username")
        email = st.text_input("Email address")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
    if submitted:
        if not all([full_name, username, email, password, confirm_password]):
            st.error("Please fill in all fields.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            success, message = register_user(username, email, password, full_name)
            if success:
                st.success("Account created successfully! Please sign in.")
                st.session_state["page"] = "login"
                st.rerun()
            else:
                st.error(message)

    st.markdown(
        '<div style="margin-top: 1rem; text-align: center;">'
        '<span style="color: var(--text-secondary); font-size: 0.875rem;">'
        "Already have an account? </span>",
        unsafe_allow_html=True,
    )
    if st.button("Sign in", key="reg_signin"):
        st.session_state["page"] = "login"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    render_footer(False)


# ---------------------------------------------------------------------------
# Phase 2 — Core classifier
# ---------------------------------------------------------------------------

def page_classifier() -> None:
    """Core classifier: real text input, submit, loading, real prediction."""
    c1, c2, c3 = st.columns([2.2, 1, 1])
    with c1:
        st.markdown(
            '<div class="brand-block">'
            '<div class="brand-logo">Lingala</div>'
            '<div class="brand-sub">Intent Classifier</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("Register", key="clf_hdr_register", use_container_width=True):
            st.session_state["page"] = "register"
            st.rerun()
    with c3:
        if st.button("Login", key="clf_hdr_login", use_container_width=True):
            st.session_state["page"] = "login"
            st.rerun()

    st.markdown("### AI Classifier")
    st.write(
        "Enter a Lingala sentence to receive a real predicted intent from the trained "
        "character-TF-IDF + logistic regression model."
    )

    backend = get_backend(TRACK)
    if not backend["available"]:
        st.error(backend["error"])
        st.info("Train the model first: `python train.py --track intent --data data/contribution.csv`")
        render_footer(check_auth())
        return

    pack = backend["pack"]
    model_language = pack.get("language", "unknown")
    is_practice = bool(pack.get("practice", False))

    if model_language != TARGET_LANGUAGE:
        st.warning(
            f"The loaded model was trained on '{model_language}', not '{TARGET_LANGUAGE}'. "
            "Predictions may not be meaningful for Lingala input."
        )
    if is_practice:
        st.info("Running in practice mode (teaching fixture).")

    text = st.text_area(
        "Lingala text",
        placeholder="e.g. Tala mbula lelo.",
        height=140,
        help="Type a Lingala sentence to classify.",
    )

    col_btn, col_space = st.columns([1, 4])
    with col_btn:
        classify = st.button("Classify", key="clf_classify", type="primary", use_container_width=True)

    if classify:
        if not text or not text.strip():
            st.error("Please enter some text before classifying.")
        else:
            with st.spinner("Classifying your input..."):
                out = _run_prediction(text.strip(), pack)
            if out is None:
                render_footer(check_auth())
                return
            result = out["result"]
            proba = out["proba"]

            info = get_intent_info().get(result.label, {})
            st.markdown(
                f'<div class="result-card">'
                f'<h4>Predicted intent</h4>'
                f'<div style="font-size: 1.5rem; font-weight: 700; color: var(--navy-primary); margin-bottom: 0.25rem;">'
                f'{info.get("name", result.label)}</div>'
                f'<div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;">'
                f'<code>{result.label}</code></div>'
                f'<div style="font-size: 0.95rem; color: var(--accent-teal-dark); font-weight: 600;">'
                f'Model confidence: {format_probability(result.probability)} '
                f'({confidence_band(result.probability)})</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("### Intent definition")
            st.markdown(
                f'<div class="card">'
                f'<div style="font-weight: 600; color: var(--navy-primary); margin-bottom: 0.25rem;">{info.get("name", result.label)}</div>'
                f'<div style="color: var(--text-primary); margin-bottom: 0.5rem;">{info.get("meaning", "No definition available.")}</div>'
                f'<div style="font-size: 0.85rem; color: var(--text-secondary);"><strong>Boundary:</strong> {info.get("boundary", "")}</div>'
                f'<div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.5rem;"><strong>Example (Lingala):</strong> {info.get("example", "")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if proba:
                st.markdown("### Supported probability distribution")
                prob_df = pd.DataFrame([
                    {"intent": p["label"], "probability": p["probability"]}
                    for p in proba
                ])
                st.bar_chart(prob_df.set_index("intent"))
                st.caption(
                    "Probabilities are the model's real, uncalibrated output for this input. "
                    "They are not confidence guarantees."
                )

            st.markdown("### Model limitations")
            st.markdown(
                f'<div class="card badge-warning" style="padding: 1rem;">'
                f'{build_disclaimer(TRACK)}<br>'
                f'Trained on {TARGET_LANGUAGE} only ({model_language}). Small dataset: '
                f'{pack.get("data_sha256", "")[:12]}... fingerprint. '
                f'Test accuracy 0.960 / macro-F1 0.960 on 50 held-out rows. '
                f'No external action (calendar, music, weather) is executed — this is a simulated route only.'
                f'</div>',
                unsafe_allow_html=True,
            )

    render_footer(check_auth())


# ---------------------------------------------------------------------------
# Phase 4 — Authenticated workspace
# ---------------------------------------------------------------------------

WORKSPACE_PAGES = [
    "Overview",
    "AI Classifier",
    "Intent Boundary Lab",
    "Dataset Explorer",
    "Model Performance",
    "Error Analysis",
    "Research Information",
    "Audio",
]


def render_workspace_sidebar() -> str:
    """Sidebar navigation for the authenticated workspace. Returns chosen page key."""
    user = get_current_user()
    with st.sidebar:
        st.markdown(
            '<div class="brand-block">'
            '<div class="brand-logo">Lingala</div>'
            '<div class="brand-sub">Workspace</div>'
            '<div class="brand-sub" style="margin-top:0.25rem;">Team 07 — Datalynk</div>'
            '<span class="brand-badge badge-navy">Authenticated</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        if user:
            st.markdown(
                f'<div style="padding: 0.75rem 1rem; background: #F2F5F7; border-radius: 8px; margin-bottom: 1rem;">'
                f'<div style="font-size: 0.75rem; color: var(--text-secondary); font-weight: 600;">Logged in as</div>'
                f'<div style="font-size: 0.95rem; color: var(--navy-primary); font-weight: 600;">{user.get("username", "user")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Logout", key="ws_logout", use_container_width=True):
                logout()
                st.rerun()

        st.markdown(
            '<div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-secondary); font-weight: 700; margin-top: 1rem; margin-bottom: 0.5rem;">AI WORKSPACE &middot; RESEARCH</div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "",
            WORKSPACE_PAGES,
            label_visibility="collapsed",
            key="ws_page",
            index=WORKSPACE_PAGES.index(st.session_state.get("ws_active_view", "AI Classifier")),
        )
        st.session_state["ws_active_view"] = page

        st.divider()
        backend = get_backend(TRACK)
        if backend["available"]:
            st.caption("Backend: available")
            if backend["pack"].get("practice"):
                st.caption("Mode: practice model")
            else:
                st.caption(f"Language: {backend['pack'].get('language', 'unknown')}")
        else:
            st.caption("Backend: unavailable")
        st.caption("No fabricated predictions")

    return page


def page_workspace() -> None:
    """Authenticated workspace — minimal app with classifier as the prominent feature."""
    if not check_auth():
        st.session_state["page"] = "landing"
        st.rerun()
        return

    # Top bar with logout
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(
            '<div class="brand-block">'
            '<div class="brand-logo">Lingala</div>'
            '<div class="brand-sub">Workspace</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("Logout", key="ws_top_logout", use_container_width=True):
            logout()
            st.rerun()

    active = render_workspace_sidebar()

    if active == "Overview":
        page_overview()
    elif active == "AI Classifier":
        page_classifier_workspace()
    elif active == "Intent Boundary Lab":
        page_intent_boundary_lab()
    elif active == "Dataset Explorer":
        page_dataset_explorer()
    elif active == "Model Performance":
        page_model_performance()
    elif active == "Error Analysis":
        page_error_analysis()
    elif active == "Research Information":
        page_research_info()
    elif active == "Audio":
        page_audio()


def page_overview() -> None:
    """Workspace overview."""
    st.markdown("## Overview")
    st.write(
        "Welcome to the Lingala Intent Classifier workspace. This is a research "
        "prototype for identifying communicative intent in Lingala sentences."
    )
    s = get_summary()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{s.get("total", 0) if s.get("exists") else 0}</div>'
            f'<div class="metric-label">Records</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">10</div>'
            '<div class="metric-label">Intents</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">150 / 50 / 50</div>'
            '<div class="metric-label">Train / dev / test</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with m4:
        backend = get_backend(TRACK)
        ready = "Ready" if backend["available"] else "Not trained"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{ready}</div>'
            f'<div class="metric-label">Model</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        "Use the sidebar to navigate. The <strong>AI Classifier</strong> is the "
        "central feature of this workspace.",
        unsafe_allow_html=True,
    )


def page_classifier_workspace() -> None:
    """Classifier rendered prominently inside the authenticated workspace."""
    st.markdown("## AI Classifier")
    st.caption("Most prominent feature of the workspace — real model predictions on Lingala text.")

    backend = get_backend(TRACK)
    if not backend["available"]:
        st.error(backend["error"])
        st.info("Train the model first: `python train.py --track intent --data data/contribution.csv`")
        return

    pack = backend["pack"]
    text = st.text_area(
        "Lingala text",
        placeholder="e.g. Yemba miziki ya lingala.",
        height=140,
        key="ws_clf_text",
    )
    if st.button("Classify", key="ws_clf_classify", type="primary", use_container_width=True):
        if not text or not text.strip():
            st.error("Please enter some text before classifying.")
        else:
            with st.spinner("Classifying your input..."):
                out = _run_prediction(text.strip(), pack)
            if out is None:
                return
            result = out["result"]
            proba = out["proba"]
            info = get_intent_info().get(result.label, {})

            st.markdown(
                f'<div class="result-card">'
                f'<h4>Predicted intent</h4>'
                f'<div style="font-size: 1.5rem; font-weight: 700; color: var(--navy-primary); margin-bottom: 0.25rem;">'
                f'{info.get("name", result.label)}</div>'
                f'<div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;"><code>{result.label}</code></div>'
                f'<div style="font-size: 0.95rem; color: var(--accent-teal-dark); font-weight: 600;">'
                f'Model confidence: {format_probability(result.probability)} '
                f'({confidence_band(result.probability)})</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if proba:
                st.markdown("### Probability distribution")
                prob_df = pd.DataFrame([
                    {"intent": p["label"], "probability": p["probability"]}
                    for p in proba
                ])
                st.bar_chart(prob_df.set_index("intent"))
                st.caption("Real, uncalibrated model output — not a confidence guarantee.")
            st.markdown(
                f'<div class="card">'
                f'<div style="font-weight: 600; color: var(--navy-primary); margin-bottom: 0.25rem;">Definition</div>'
                f'<div style="color: var(--text-primary); margin-bottom: 0.5rem;">{info.get("meaning", "")}</div>'
                f'<div style="font-size: 0.85rem; color: var(--text-secondary);"><strong>Boundary:</strong> {info.get("boundary", "")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Phase 4 — remaining workspace pages
# ---------------------------------------------------------------------------

def page_intent_boundary_lab() -> None:
    """Intent Boundary Lab — explore all 10 intents with definitions and examples."""
    st.markdown("## Intent Boundary Lab")
    st.write(
        "Explore the 10 intent categories, their definitions, boundaries and "
        "Lingala examples. Boundaries show where one intent ends and another begins."
    )
    info = get_intent_info()
    labels = [
        "weather_query", "datetime_query", "alarm_set", "calendar_set",
        "calendar_query", "lists_createoradd", "lists_query", "play_music",
        "news_query", "qa_definition",
    ]
    for label in labels:
        entry = info.get(label, {})
        with st.expander(f"{entry.get('name', label)} — `{label}`"):
            st.markdown(
                f'<div class="card">'
                f'<div style="font-weight: 600; color: var(--navy-primary); margin-bottom: 0.25rem;">'
                f'{entry.get("name", label)}</div>'
                f'<div style="color: var(--text-primary); margin-bottom: 0.5rem;">{entry.get("meaning", "")}</div>'
                f'<div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">'
                f'<strong>Boundary:</strong> {entry.get("boundary", "")}</div>'
                f'<div style="font-size: 0.85rem; color: var(--text-secondary);">'
                f'<strong>Example (Lingala):</strong> {entry.get("example", "")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def page_dataset_explorer() -> None:
    """Dataset Explorer — view the contribution dataset summary and distribution."""
    st.markdown("## Dataset Explorer")
    s = get_summary()
    if not s.get("exists"):
        st.warning("No contribution dataset found at data/contribution.csv")
        return
    st.write(
        f"Dataset: `{s['path']}` &middot; Language: {s.get('language')} &middot; "
        f"Total records: {s['total']}"
    )
    st.markdown("### Label distribution")
    dist = s.get("label_distribution", {})
    if dist:
        df = pd.DataFrame(sorted(dist.items()), columns=["label", "count"])
        st.bar_chart(df.set_index("label"))
    st.markdown("### Split distribution")
    split = s.get("split_distribution", {})
    if split:
        df2 = pd.DataFrame(sorted(split.items()), columns=["split", "count"])
        st.dataframe(df2, use_container_width=True, hide_index=True)
    st.markdown("### Review coverage")
    review = s.get("review_coverage", {})
    if review:
        df3 = pd.DataFrame(sorted(review.items()), columns=["review_status", "count"])
        st.dataframe(df3, use_container_width=True, hide_index=True)


def page_model_performance() -> None:
    """Model Performance — verified evaluation metrics from evaluate.py."""
    st.markdown("## Model Performance")
    metrics = get_test_metrics(TRACK)
    dev = get_dev_metrics(TRACK)
    if not metrics:
        st.warning("No test metrics found. Run `python evaluate.py --track intent --data data/contribution.csv`.")
        return
    model = metrics["model"]
    baseline = metrics["majority_baseline"]
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{model["accuracy"]:.3f}</div>'
            f'<div class="metric-label">Test accuracy</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{model["macro_f1"]:.3f}</div>'
            f'<div class="metric-label">Test macro-F1</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{baseline["accuracy"]:.3f}</div>'
            f'<div class="metric-label">Majority baseline accuracy</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{metrics.get("test_size", "?")}</div>'
            f'<div class="metric-label">Test rows</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("### Per-label F1 (test)")
    report = model.get("report", {})
    rows = []
    for label in [
        "weather_query", "datetime_query", "alarm_set", "calendar_set",
        "calendar_query", "lists_createoradd", "lists_query", "play_music",
        "news_query", "qa_definition",
    ]:
        if label in report:
            rows.append({"intent": label, "f1": report[label]["f1-score"], "support": report[label]["support"]})
    if rows:
        st.bar_chart(pd.DataFrame(rows).set_index("intent")["f1"])
        st.caption("Source: results/test_metrics.json (evaluate.py). Small test set — one sentence moves accuracy by 2 points.")
    if dev:
        st.markdown("### Dev metrics (used for setting choices)")
        st.write(f"Dev macro-F1: {dev['model']['macro_f1']:.3f}")


def page_error_analysis() -> None:
    """Error Analysis — show test predictions and mismatches."""
    st.markdown("## Error Analysis")
    preds = get_test_predictions(TRACK)
    if preds is None or preds.empty:
        st.warning("No test predictions found. Run evaluate.py first.")
        return
    # Detect mismatches if both label and prediction columns exist.
    mismatch = preds
    if "label" in preds.columns and "predicted_label" in preds.columns:
        mismatch = preds[preds["label"] != preds["predicted_label"]]
        st.write(f"Test rows: {len(preds)} &middot; Misclassified: {len(mismatch)}")
        st.dataframe(mismatch, use_container_width=True, hide_index=True)
    else:
        st.dataframe(preds, use_container_width=True, hide_index=True)
        st.caption("Column schema varies by evaluate.py run; showing raw predictions.")


def page_research_info() -> None:
    """Research Information — motivation, method, limitations, citations."""
    st.markdown("## Research Information")
    st.markdown(
        """
        ### Motivation
        Low-resource African languages such as Lingala lack intent-classification
        resources. This prototype investigates whether a small, reviewed, human-authored
        Lingala dataset can support a usable baseline intent classifier.

        ### Method
        - Data: 250 human-authored Lingala sentences across 10 intent categories.
        - Split: 150 train / 50 dev / 50 test (15 / 5 / 5 per label).
        - Model: character-TF-IDF (2–4 grams) + logistic regression (C=5.0).
        - Evaluation: accuracy and macro-F1 on held-out test set, plus majority baseline.

        ### Results
        - Test accuracy: 0.960 &middot; macro-F1: 0.960.
        - Majority baseline: 0.100 accuracy &middot; 0.018 macro-F1.

        ### Limitations
        - Small dataset; one test sentence moves accuracy by 2 points.
        - Kinshasa Lingala focus; may not represent all varieties.
        - No external action is executed (simulated route only).
        - No audio / speech recognition for Lingala is supported.

        ### Team
        Team 7 — Datalynk &middot; Strathmore NLP Hub Datathon 2026 &middot; Track A
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Phase 5 — Audio (clearly labelled, no fake transcription)
# ---------------------------------------------------------------------------

def page_audio() -> None:
    """Audio interface — clearly labelled; Lingala transcription is NOT supported."""
    st.markdown("## Audio input")
    st.warning(
        "Speech recognition for Lingala is not supported by the available tooling, "
        "and Lingala is not covered by major off-the-shelf speech-to-text models. "
        "No transcription is performed and no audio is processed."
    )
    st.write(
        "Upload or record audio here only to acknowledge the interface. "
        "Classification requires text input via the AI Classifier."
    )
    uploaded = st.file_uploader("Audio file (placeholder only — not transcribed)", type=["wav", "mp3", "ogg", "m4a"])
    if uploaded is not None:
        st.info(
            "File received but not transcribed. Lingala speech recognition is not "
            "available in this prototype. Please type your Lingala sentence in the "
            "AI Classifier instead."
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main application entry point."""
    init_auth()

    if "page" not in st.session_state:
        st.session_state["page"] = "landing"
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "ws_active_view" not in st.session_state:
        st.session_state["ws_active_view"] = "AI Classifier"

    if not check_auth():
        page = st.session_state.get("page", "landing")
        if page == "landing":
            page_landing()
        elif page == "login":
            page_login()
        elif page == "register":
            page_register()
        elif page == "classifier":
            page_classifier()
        else:
            st.session_state["page"] = "landing"
            page_landing()
    else:
        page_workspace()


if __name__ == "__main__":
    main()
