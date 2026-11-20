"""
Confusion matrix generation and visualization using Plotly for dark-mode dashboards.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from src.utils.config import HEAD_CONFIG


def generate_confusion_matrix_figure(
    y_true: List[int],
    y_pred: List[int],
    head_name: str,
    normalize: bool = False,
) -> go.Figure:
    """
    Constructs an interactive dark-themed Plotly heatmap of the confusion matrix
    for a given classification head.
    """
    classes = HEAD_CONFIG.get(head_name, [])
    labels_indices = list(range(len(classes)))

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    valid_mask = (y_true_arr != -100)
    filtered_true = y_true_arr[valid_mask]
    filtered_pred = y_pred_arr[valid_mask]

    if len(filtered_true) == 0:
        cm = np.zeros((len(classes), len(classes)), dtype=int)
    else:
        cm = confusion_matrix(filtered_true, filtered_pred, labels=labels_indices)

    if normalize and cm.sum() > 0:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm.astype(float), row_sums, where=row_sums != 0)
        cm_display = np.round(cm_display, 2)
    else:
        cm_display = cm

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_display,
            x=classes,
            y=classes,
            colorscale=[[0, "#111827"], [0.5, "#1e3a8a"], [1.0, "#38bdf8"]],
            text=cm_display,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "#f8fafc"},
            showscale=True,
            colorbar=dict(
                tickfont=dict(color="#94a3b8"),
                title=dict(text="Proportion" if normalize else "Count", font=dict(color="#94a3b8")),
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Confusion Matrix: {head_name.replace('_', ' ').title()}",
            font=dict(color="#f1f5f9", size=16),
        ),
        xaxis=dict(
            title=dict(text="Predicted Label", font=dict(color="#94a3b8")),
            tickfont=dict(color="#cbd5e1"),
            gridcolor="#334155",
        ),
        yaxis=dict(
            title=dict(text="True Label", font=dict(color="#94a3b8")),
            tickfont=dict(color="#cbd5e1"),
            gridcolor="#334155",
            autorange="reversed",
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )

    return fig


def figure_from_matrix_dataframe(
    cm_df: pd.DataFrame,
    head_name: str,
    normalize: bool = False,
) -> go.Figure:
    """
    Constructs an interactive dark-themed Plotly heatmap from a precomputed confusion matrix DataFrame.
    """
    classes = list(cm_df.columns)
    cm = cm_df.values

    if normalize and cm.sum() > 0:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm.astype(float), row_sums, where=row_sums != 0)
        cm_display = np.round(cm_display, 2)
    else:
        cm_display = cm

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_display,
            x=classes,
            y=classes,
            colorscale=[[0, "#0b0f19"], [0.25, "#1e293b"], [0.6, "#0284c7"], [1.0, "#38bdf8"]],
            text=cm_display,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "#f8fafc"},
            showscale=True,
            colorbar=dict(
                tickfont=dict(color="#94a3b8"),
                title=dict(text="Proportion" if normalize else "Test Samples", font=dict(color="#94a3b8")),
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Test Set Evaluation: {head_name.replace('_', ' ').title()} (1,914 samples)",
            font=dict(color="#f1f5f9", size=15),
        ),
        xaxis=dict(
            title=dict(text="Predicted Class", font=dict(color="#94a3b8")),
            tickfont=dict(color="#cbd5e1"),
            gridcolor="#1e293b",
        ),
        yaxis=dict(
            title=dict(text="Ground Truth Label", font=dict(color="#94a3b8")),
            tickfont=dict(color="#cbd5e1"),
            gridcolor="#1e293b",
            autorange="reversed",
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )
    return fig
