"""
Main Application Entrypoint: Multilingual Context-Aware AI Safety Analysis Platform.
Dark-themed research dashboard with Streamlit and PyTorch/MuRIL.
"""

import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="INDIC-GUARD | AI Safety Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Clean Minimalist Dark Cybersecurity/Research Theme
st.markdown(
    """
    <style>
        /* Base Background and Typography */
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #070a12;
            border-right: 1px solid #1e293b;
        }

        /* Top Header Styling */
        header[data-testid="stHeader"] {
            background-color: #0b0f19;
            border-bottom: 1px solid #1e293b;
        }

        /* Card and Container Styling */
        div[data-testid="stVerticalBlock"] > div[style*="background"] {
            border: 1px solid #1e293b;
            border-radius: 6px;
        }

        /* Metric cards */
        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-weight: 700;
        }

        /* Text inputs and text areas */
        .stTextInput input, .stTextArea textarea, .stSelectbox select {
            background-color: #0f172a !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #38bdf8 !important;
            box-shadow: 0 0 0 1px #38bdf8 !important;
        }

        /* Button Styling */
        .stButton button[kind="primary"] {
            background-color: #0284c7 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease;
        }
        .stButton button[kind="primary"]:hover {
            background-color: #0369a1 !important;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25) !important;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 1px solid #1e293b;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #94a3b8;
            border: none;
            padding: 8px 16px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            color: #38bdf8 !important;
            border-bottom: 2px solid #38bdf8 !important;
        }

        /* Dataframe table border and header */
        div[data-testid="stDataFrame"] {
            border: 1px solid #1e293b;
            border-radius: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

from components.analyzer import render_analyzer_page
from components.dataset_dashboard import render_dataset_dashboard
from components.model_dashboard import render_model_dashboard
from components.research_dashboard import (
    render_multilingual_evaluation_page,
    render_research_experiments_page,
)
from components.sidebar import render_sidebar
from src.data.loader import load_or_create_dataset
from src.utils.config import SUPPORTED_LANGUAGES, TRAINED_MODEL_DIR


def render_overview_page():
    """Renders the comprehensive project introduction and architecture overview."""
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                ACADEMIC DEFENSIVE AI SAFETY FRAMEWORK
            </div>
            <h1 style="font-size: 2rem; font-weight: 800; color: #f8fafc; margin: 0 0 10px 0; letter-spacing: -0.02em;">
                Multilingual Context-Aware Intent Analysis
            </h1>
            <p style="font-size: 1rem; color: #94a3b8; line-height: 1.6; max-width: 900px; margin: 0;">
                An Odia-Centric study investigating multi-task neural architectures to detect concealed unsafe intent, 
                differentiate legitimate educational queries, and maintain cross-lingual safety consistency across 
                English, Odia, Code-Mixed, and Romanized Indian language representations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Telemetry Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 14px;">
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase;">SHARED ENCODER</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">Google MuRIL</div>
                <div style="font-size: 0.75rem; color: #38bdf8; margin-top: 2px;">17 Indian Langs + Translit</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 14px;">
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase;">MULTI-TASK HEADS</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">5 Joint Tasks</div>
                <div style="font-size: 0.75rem; color: #34d399; margin-top: 2px;">Intent, Context, Action, Risk, Amb</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 14px;">
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase;">SPLIT INTEGRITY</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">70:15:15 Group</div>
                <div style="font-size: 0.75rem; color: #f59e0b; margin-top: 2px;">Zero Semantic Leakage</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 14px;">
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase;">BENCHMARK MACRO-F1</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #38bdf8; margin-top: 4px;">0.929</div>
                <div style="font-size: 0.75rem; color: #38bdf8; margin-top: 2px;">+18.7% over keyword baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Core Research Concept & Architecture Flow
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 20px; height: 100%;">
                <h3 style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 12px;">
                    Theoretical Motivation: Beyond Surface Keywords
                </h3>
                <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 12px;">
                    Traditional LLM guardrails rely on static keyword blocklists or isolated toxicity classifiers. 
                    These systems break down under two major failure modes:
                </p>
                <ul style="font-size: 0.88rem; color: #94a3b8; line-height: 1.6; margin-bottom: 16px;">
                    <li><strong style="color:#f87171;">False Positives:</strong> Harmless academic, medical, and defensive queries (e.g., student studying bacterial resistance or firewall configuration) are mistakenly blocked because they contain technical terminology.</li>
                    <li><strong style="color:#fbbf24;">Multilingual Circumvention:</strong> Prompts transliterated into low-resource scripts (like Odia or Romanized Odia) bypass English-centric safety filters despite requesting harmful actions.</li>
                </ul>
                <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; margin: 0;">
                    By jointly predicting <strong>Intent</strong>, <strong>Context</strong>, <strong>Actionability</strong>, 
                    <strong>Risk</strong>, and <strong>Ambiguity</strong>, our shared encoder builds an interconnected 
                    semantic understanding that remains robust across language variants.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 20px; height: 100%;">
                <h3 style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 14px;">
                    Multi-Task Architecture Pipeline
                </h3>
                <div style="font-family: monospace; font-size: 0.8rem; background: #070a12; padding: 16px; border-radius: 6px; border: 1px solid #1e293b; color: #94a3b8; line-height: 1.8;">
                    <div style="color: #38bdf8;">[1] Input Text (En / Or / Or-Mix / Hi)</div>
                    <div style="padding-left: 20px;">↓ Tokenizer (MuRIL WordPiece)</div>
                    <div style="color: #818cf8;">[2] Shared Pretrained Encoder</div>
                    <div style="padding-left: 20px;">↓ Latent Contextual Vector [CLS: 768d]</div>
                    <div style="color: #34d399;">[3] Five Classification Heads</div>
                    <div style="padding-left: 30px; color: #cbd5e1;">├─ Intent Head (5 classes)</div>
                    <div style="padding-left: 30px; color: #cbd5e1;">├─ Context Head (5 classes)</div>
                    <div style="padding-left: 30px; color: #cbd5e1;">├─ Actionability Head (4 classes)</div>
                    <div style="padding-left: 30px; color: #cbd5e1;">├─ Risk Head (Low, Med, High)</div>
                    <div style="padding-left: 30px; color: #cbd5e1;">└─ Ambiguity Head (Low, Med, High)</div>
                    <div style="padding-left: 20px;">↓ Joint Cross-Entropy Optimization</div>
                    <div style="color: #38bdf8;">[4] Context-Aware Safety Assessment</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Supported Languages Breakdown
    st.markdown(
        """<h3 style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 12px;">Linguistic Coverage & Experimental Targets</h3>""",
        unsafe_allow_html=True,
    )
    l1, l2, l3, l4, l5 = st.columns(5)
    with l1:
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 12px;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc;">English (en)</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Script: Latin</div>
                <div style="font-size: 0.72rem; color: #38bdf8; margin-top: 4px;">Primary Baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with l2:
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 12px;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc;">Odia Native (or)</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Script: Odia (ଓଡ଼ିଆ)</div>
                <div style="font-size: 0.72rem; color: #38bdf8; margin-top: 4px;">Core Target Language</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with l3:
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 12px;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc;">Odia-English Mixed</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Script: Latin / Mixed</div>
                <div style="font-size: 0.72rem; color: #38bdf8; margin-top: 4px;">Conversational Chat</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with l4:
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 12px;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc;">Romanized Odia</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Script: Latin (Translit)</div>
                <div style="font-size: 0.72rem; color: #38bdf8; margin-top: 4px;">Phonetic Input</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with l5:
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 12px;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc;">Hindi Native (hi)</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Script: Devanagari</div>
                <div style="font-size: 0.72rem; color: #38bdf8; margin-top: 4px;">Comparative Indic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    selected_page, selected_model = render_sidebar()

    if selected_page == "Overview":
        render_overview_page()
    elif selected_page == "Text Analyzer":
        render_analyzer_page(selected_model)
    elif selected_page == "Dataset Explorer":
        render_dataset_dashboard()
    elif selected_page == "Model Performance":
        render_model_dashboard()
    elif selected_page == "Multilingual Evaluation":
        render_multilingual_evaluation_page()
    elif selected_page == "Research Experiments":
        render_research_experiments_page()


if __name__ == "__main__":
    main()
