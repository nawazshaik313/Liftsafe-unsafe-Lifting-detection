"""Shared Plotly figures and renderer for LiftSafe's dark glass dashboard."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import get_theme_mode

_DARK_MODEL_COLORS = {
    "Logistic Regression": "#b99aff",
    "Random Forest": "#71a8ff",
    "Temporal CNN": "#67e2ef",
}
_LIGHT_MODEL_COLORS = {
    "Logistic Regression": "#7650bf",
    "Random Forest": "#3f64c7",
    "Temporal CNN": "#087e96",
}
_DARK_SERIES_COLORS = ["#71a8ff", "#b99aff", "#67e2ef", "#70dbc4", "#e4a1ff"]
_LIGHT_SERIES_COLORS = ["#3f64c7", "#7650bf", "#087e96", "#14836d", "#a44ec0"]


def _palette(mode: str | None = None) -> dict[str, object]:
    """Return semantic chart colors for the selected app appearance."""
    selected_mode = mode or get_theme_mode() or "Dark"
    light = selected_mode.lower() == "light"
    return {
        "text": "#202842" if light else "#e9edfa",
        "muted": "#4f5b77" if light else "#b8c2dc",
        "legend": "#4f5b77" if light else "#cbd3e8",
        "grid": "rgba(63,79,125,.15)" if light else "rgba(190,205,236,0.12)",
        "axis": "rgba(63,79,125,.22)" if light else "rgba(223,238,255,0.16)",
        "plot": "rgba(255,255,255,.08)" if light else "rgba(255,255,255,0.012)",
        "hover_bg": "#ffffff" if light else "#171d34",
        "hover_text": "#202842" if light else "#f5f6ff",
        "hover_border": "rgba(118,80,191,.50)" if light else "rgba(196,164,255,0.55)",
        "bar_text": "#202842" if light else "#eff3ff",
        "series": _LIGHT_SERIES_COLORS if light else _DARK_SERIES_COLORS,
        "models": _LIGHT_MODEL_COLORS if light else _DARK_MODEL_COLORS,
        "heatmap": [[0, "#edf0fa"], [.45, "#9b8bd5"], [1, "#278da4"]]
        if light else [[0, "#151a31"], [.45, "#4e5399"], [1, "#4ac7d9"]],
    }


def glass_figure(height: int = 330, *, mode: str | None = None) -> go.Figure:
    """Create a transparent, responsive chart using the shared visual tokens."""
    palette = _palette(mode)
    figure = go.Figure()
    figure.update_layout(
        height=height,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=palette["plot"],
        font={
            "family": "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            "color": palette["text"],
            "size": 12,
        },
        margin={"l": 52, "r": 20, "t": 16, "b": 48},
        hoverlabel={
            "bgcolor": palette["hover_bg"],
            "bordercolor": palette["hover_border"],
            "font": {"color": palette["hover_text"], "size": 12},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": palette["legend"], "size": 11},
        },
    )
    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=palette["axis"],
        tickfont={"color": palette["muted"]},
        title_font={"color": palette["legend"], "size": 12},
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=palette["grid"],
        zeroline=False,
        linecolor=palette["axis"],
        tickfont={"color": palette["muted"]},
        title_font={"color": palette["legend"], "size": 12},
    )
    return figure


def render_glass_chart(figure: go.Figure, key: str) -> None:
    """Place a Plotly chart in the reusable bordered glass panel."""
    with st.container(border=True):
        st.plotly_chart(
            figure,
            width="stretch",
            theme=None,
            key=key,
            config={
                "displayModeBar": False,
                "responsive": True,
                "scrollZoom": False,
            },
        )


def line_figure(
    frame: pd.DataFrame,
    x: str | None,
    y_columns: Sequence[str],
    *,
    labels: dict[str, str] | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 330,
    threshold: float | None = None,
    threshold_label: str = "Review threshold",
    step: bool = False,
) -> go.Figure:
    """Build a consistently themed multi-series line chart from existing data."""
    figure = glass_figure(height)
    labels = labels or {}
    x_values = frame[x] if x and x in frame.columns else frame.index
    for index, column in enumerate(y_columns):
        if column not in frame:
            continue
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=frame[column],
                name=labels.get(column, column),
                mode="lines",
                line={
                    "color": _palette()["series"][index % len(_palette()["series"])],
                    "width": 2.5,
                    **({"shape": "hv"} if step else {}),
                },
                connectgaps=False,
                hovertemplate=(
                    "%{x:.2f} s<br>%{y:.2f}<extra>%{fullData.name}</extra>"
                    if x and x.endswith("seconds")
                    else "%{x}<br>%{y:.2f}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    if threshold is not None:
        figure.add_hline(
            y=threshold,
            line_color="#b575d6" if get_theme_mode() == "Light" else "#f1b7ff",
            line_dash="dot",
            line_width=1.5,
            annotation_text=threshold_label,
            annotation_position="top left",
            annotation_font_color="#7547a9" if get_theme_mode() == "Light" else "#f1d2ff",
        )
    figure.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
    )
    return figure


def bar_figure(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    name: str,
    color: str = "#71a8ff",
    y_title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Build a compact model-score or metric bar chart."""
    figure = glass_figure(height)
    upper_bound = max(max(values, default=0) * 1.22, 1.12)
    figure.add_trace(
        go.Bar(
            x=list(labels),
            y=list(values),
            name=name,
            marker={
                "color": [_palette()["series"][i % len(_palette()["series"])] for i in range(len(values))]
                if color == "#71a8ff" else color,
                "line": {"color": "rgba(63,79,125,.34)" if get_theme_mode() == "Light" else "rgba(230,240,255,0.48)", "width": 1},
            },
            text=[f"{value:.1%}" if 0 <= value <= 1 else f"{value:.1f}" for value in values],
            textposition="outside",
            textfont={"color": _palette()["bar_text"], "size": 11},
            hovertemplate="%{x}<br>%{y:.3f}<extra>%{fullData.name}</extra>",
        )
    )
    figure.update_layout(
        showlegend=False,
        yaxis_title=y_title,
        yaxis_range=[0, upper_bound],
        bargap=0.48,
    )
    return figure


def model_metric_figure(
    labels: Sequence[str], values: Sequence[float], metric: str, *, percent: bool = False
) -> go.Figure:
    """Build a model comparison with stable, model-specific accent colors."""
    figure = glass_figure(300)
    palette = _palette()
    colors = [palette["models"].get(label, palette["series"][i % len(palette["series"])]) for i, label in enumerate(labels)]
    figure.add_trace(
        go.Bar(
            x=list(labels),
            y=list(values),
            marker={"color": colors, "line": {"color": "rgba(63,79,125,.34)" if get_theme_mode() == "Light" else "rgba(230,240,255,.35)", "width": 1}},
            text=[f"{value:.2f}%" if percent else f"{value:.3f}" for value in values],
            textposition="outside",
            textfont={"color": palette["bar_text"], "size": 11},
            hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
        )
    )
    top = max(values, default=0) * 1.12
    figure.update_layout(
        showlegend=False,
        yaxis_title=metric,
        yaxis_range=[0, max(100 if percent else 1.08, top)],
        bargap=0.42,
    )
    return figure


def confusion_matrix_figure(matrix: Sequence[Sequence[int]], labels: Sequence[str]) -> go.Figure:
    """Build an annotated confusion matrix with mode-aware contrast."""
    palette = _palette()
    figure = glass_figure(340)
    figure.add_trace(
        go.Heatmap(
            z=matrix,
            x=list(labels),
            y=list(labels),
            text=matrix,
            texttemplate="%{text}",
            textfont={"color": palette["text"], "size": 17},
            colorscale=palette["heatmap"],
            colorbar={
                "title": {"text": "Trials", "font": {"color": palette["legend"]}},
                "tickfont": {"color": palette["muted"]},
                "outlinecolor": palette["axis"],
            },
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>Trials: %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title="Predicted condition",
        yaxis_title="Actual condition",
        yaxis={"autorange": "reversed", "showgrid": False},
        xaxis={"showgrid": False},
        margin={"l": 78, "r": 30, "t": 16, "b": 60},
    )
    return figure
