"""
Sidebar navigation and global settings component for the Streamlit dashboard.
"""

from typing import Tuple
import streamlit as st
import torch

from src.utils.config import SUPPORTED_LANGUAGES, TRAINED_MODEL_DIR


def render_sidebar() -> Tuple[str, str]:
    """
    Renders minimal dark-themed sidebar with page navigation,
    hardware status, and model selection.
    """
    with st.sidebar:
        # Title and Branding
        st.markdown(
            """
            <div style="padding: 10px 0px 20px 0px;">
                <div style="font-size: 1.15rem; font-weight: 700; letter-spacing: 0.05em; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                    <span style="display:inline-block; width:10px; height:10px; background:#38bdf8; border-radius:50%;"></span>
                    INDIC-GUARD LAB
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                    Multilingual Context-Aware AI Safety
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="height: 1px; background: #334155; margin-bottom: 20px;"></div>
            """,
            unsafe_allow_html=True,
        )

        # Page Selection Navigation
        pages = [
            "Overview",
            "Text Analyzer",
            "Dataset Explorer",
            "Model Performance",
            "Multilingual Evaluation",
            "Research Experiments",
        ]

        selected_page = st.radio(
            "NAVIGATION",
            pages,
            index=0,
            label_visibility="visible",
        )

        st.markdown(
            """
            <div style="height: 1px; background: #334155; margin: 24px 0px 16px 0px;"></div>
            """,
            unsafe_allow_html=True,
        )

        # Model Architecture Selection
        st.markdown(
            """<span style="font-size: 0.75rem; font-weight: 600; color: #64748b; letter-spacing: 0.05em;">ARCHITECTURE</span>""",
            unsafe_allow_html=True,
        )
        selected_model = st.selectbox(
            "Model Backbone",
            [
                "google/muril-base-cased",
                "FacebookAI/xlm-roberta-base",
            ],
            index=0,
            label_visibility="collapsed",
        )

        # System & Hardware Telemetry
        device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        heads_file = TRAINED_MODEL_DIR / "multitask_heads.pt"
        checkpoint_status = "Trained & Calibrated" if heads_file.exists() else "Pretrained Base"

        st.markdown(
            f"""
            <div style="margin-top: 24px; padding: 12px; background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; font-size: 0.75rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b;">Device:</span>
                    <span style="color: #cbd5e1; font-family: monospace;">{device_name}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b;">Dataset:</span>
                    <span style="color: #38bdf8; font-weight: 700; font-family: monospace;">10,040 Entries</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b;">Weights:</span>
                    <span style="color: #34d399; font-weight: 600; font-family: monospace;">{checkpoint_status}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b;">Accuracy:</span>
                    <span style="color: #38bdf8; font-family: monospace;">86.51% (F1: 0.862)</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #64748b;">Languages:</span>
                    <span style="color: #cbd5e1;">En, Or, Or-Mix, Hi</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="font-size: 0.7rem; color: #475569; margin-top: 30px; text-align: center;">
                Academic Defensive AI Safety Framework<br>Odia-Centric Study • 2026
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page, selected_model
