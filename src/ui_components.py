"""
Reusable UI Components and Styling Module for SentinelAI Streamlit Application.
Provides consistent dark industrial control-room cards, metric displays,
status badges, and interactive control bars.
"""
from typing import Any, Dict, List, Optional
import streamlit as st

from config import UI_COLORS


def apply_custom_css():
    """Injects bespoke dark industrial control-room CSS styling."""
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {UI_COLORS['background']};
            color: {UI_COLORS['text_primary']};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        
        [data-testid="stSidebar"] {{
            background-color: {UI_COLORS['sidebar_bg']};
            border-right: 1px solid {UI_COLORS['border']};
        }}
        
        .sentinel-header-container {{
            padding: 1.25rem 1.75rem;
            background: linear-gradient(135deg, #131C31 0%, #0D1527 100%);
            border: 1px solid #1E293B;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
        }}
        
        .kpi-metric-card {{
            background-color: #131C31;
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 0.75rem;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }}
        .kpi-metric-card:hover {{
            border-color: #06B6D4;
        }}
        
        .kpi-metric-label {{
            color: #94A3B8;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }}
        
        .kpi-metric-value {{
            color: #F8FAFC;
            font-size: 1.75rem;
            font-weight: 700;
            line-height: 1.2;
        }}
        
        .kpi-metric-sub {{
            color: #94A3B8;
            font-size: 0.8rem;
            margin-top: 0.35rem;
        }}
        
        .badge-healthy {{
            background-color: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid #10B981;
            padding: 3px 8px;
            border-radius: 5px;
            font-weight: 600;
            font-size: 0.78rem;
            display: inline-block;
        }}
        
        .badge-warning {{
            background-color: rgba(245, 158, 11, 0.15);
            color: #F59E0B;
            border: 1px solid #F59E0B;
            padding: 3px 8px;
            border-radius: 5px;
            font-weight: 600;
            font-size: 0.78rem;
            display: inline-block;
        }}
        
        .badge-critical {{
            background-color: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid #EF4444;
            padding: 3px 8px;
            border-radius: 5px;
            font-weight: 600;
            font-size: 0.78rem;
            display: inline-block;
        }}
        
        .investigation-panel {{
            background-color: #131C31;
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }}
        
        .panel-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #F8FAFC;
            margin-bottom: 0.5rem;
        }}
        
        .disclaimer-banner {{
            background-color: rgba(15, 23, 42, 0.85);
            border-left: 3px solid #06B6D4;
            padding: 0.85rem 1.25rem;
            border-radius: 0 6px 6px 0;
            font-size: 0.83rem;
            color: #94A3B8;
            margin-top: 0.75rem;
            margin-bottom: 1.25rem;
            line-height: 1.5;
        }}
    </style>
    """, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, badge_text: Optional[str] = None, badge_type: str = "healthy"):
    """Renders standardized industrial header banner with optional status badge."""
    apply_custom_css()
    badge_html = ""
    if badge_text:
        cls = "badge-healthy" if badge_type == "healthy" else ("badge-warning" if badge_type == "warning" else "badge-critical")
        badge_html = f'<div style="margin-top: 0.5rem;"><span class="{cls}">{badge_text}</span></div>'

    st.markdown(f"""
    <div class="sentinel-header-container">
        <h1 style="margin: 0; font-size: 1.95rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em;">
            {title}
        </h1>
        <p style="margin: 0.35rem 0 0 0; color: #94A3B8; font-size: 0.95rem;">
            {subtitle}
        </p>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(title: str, value: Any, subtitle: Optional[str] = None, value_color: Optional[str] = None):
    """Renders a responsive metric card."""
    color_style = f' style="color: {value_color};"' if value_color else ""
    sub_html = f'<div class="kpi-metric-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="kpi-metric-card">
        <div class="kpi-metric-label">{title}</div>
        <div class="kpi-metric-value"{color_style}>{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def get_status_badge_html(status: str) -> str:
    """Returns styled HTML badge for HEALTHY, DEGRADED, and CRITICAL statuses."""
    s = str(status).upper()
    if s == "HEALTHY":
        return '<span class="badge-healthy">● HEALTHY</span>'
    elif s in ("DEGRADED", "WARNING"):
        return '<span class="badge-warning">▲ DEGRADED</span>'
    elif s == "CRITICAL":
        return '<span class="badge-critical">■ CRITICAL</span>'
    else:
        return f'<span class="badge-healthy">{status}</span>'
