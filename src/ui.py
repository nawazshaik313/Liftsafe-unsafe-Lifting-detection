"""Shared Liquid Glass styling and page chrome for the Streamlit app."""

from html import escape
from pathlib import Path

import streamlit as st


_STYLESHEET = Path(__file__).resolve().parents[1] / "assets" / "liquid_glass.css"
_SIDEBAR_LOGO = Path(__file__).resolve().parents[1] / "assets" / "liftsafe_logo.svg"
_SIDEBAR_ICON = Path(__file__).resolve().parents[1] / "assets" / "liftsafe_icon.svg"


def inject_app_styles() -> None:
    """Load the shared stylesheet and the selected appearance on every page."""
    css = _STYLESHEET.read_text(encoding="utf-8")
    marker_start = "/* LIGHT_THEME_OVERRIDES_START */"
    marker_end = "/* LIGHT_THEME_OVERRIDES_END */"
    if marker_start in css and marker_end in css:
        before, themed = css.split(marker_start, maxsplit=1)
        light_overrides, after = themed.split(marker_end, maxsplit=1)
        css = before + (light_overrides if get_theme_mode() == "Light" else "") + after
    # st.html routes style-only markup to Streamlit's event container, so the
    # stylesheet applies to the app document without adding a visible block.
    st.html(f"<style>{css}</style>")


def get_theme_mode() -> str:
    """Return the user's persistent in-app appearance selection."""
    selected = st.session_state.get("liftsafe_theme")
    return selected if selected in {"Dark", "Light"} else "Light"


def render_theme_toggle() -> None:
    """Render a compact, persistent appearance selector in the sidebar."""
    st.sidebar.markdown('<div class="ls-appearance-label">Appearance</div>', unsafe_allow_html=True)
    st.sidebar.segmented_control(
        "Appearance",
        options=["Dark", "Light"],
        default="Light",
        key="liftsafe_theme",
        label_visibility="collapsed",
        help="Switch the Liquid Glass interface between dark and light modes.",
    )


def render_page_header(title: str, description: str, eyebrow: str = "LIFTSAFE") -> None:
    """Render a compact, consistent header without interpolating raw HTML."""
    st.markdown(
        '<div class="ls-page-header">'
        f'<div class="ls-eyebrow">{escape(eyebrow)}</div>'
        f'<div class="ls-title">{escape(title)}</div>'
        f'<div class="ls-description">{escape(description)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, detail: str | None = None) -> None:
    """Render a reusable section heading and optional supporting text."""
    supporting = (
        f'<div class="ls-section-detail">{escape(detail)}</div>' if detail else ""
    )
    st.markdown(
        '<div class="ls-section-heading">'
        f'<div class="ls-section-title">{escape(title)}</div>{supporting}'
        '</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Add lightweight project identity above Streamlit's native page links."""
    st.logo(str(_SIDEBAR_LOGO), size="medium", icon_image=str(_SIDEBAR_ICON))


def render_empty_state(icon: str, title: str, description: str) -> None:
    """Show an accessible, reusable empty-state panel inside the current flow."""
    st.markdown(
        '<div class="ls-empty-state" role="status">'
        f'<div class="ls-empty-icon" aria-hidden="true">{escape(icon)}</div>'
        f'<div class="ls-empty-title">{escape(title)}</div>'
        f'<div class="ls-empty-copy">{escape(description)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
