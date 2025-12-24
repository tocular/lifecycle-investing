"""
Main entry point for the Lifecycle Investing Calculator Streamlit application.
Wires together sidebar inputs and dashboard display components.
"""

import sys
import base64
from pathlib import Path

# Add project root to path for direct execution (e.g., streamlit run lifecycle_app/main.py)
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
from lifecycle_app.ui.sidebar import render_sidebar
from lifecycle_app.ui.dashboard import render_dashboard


def get_base64_image(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main() -> None:
    """
    Configure and run the Lifecycle Investing Calculator application.

    Sets up page configuration, renders the sidebar for user inputs,
    and displays the dashboard with portfolio analysis results.
    """
    # Configure page settings for optimal display
    st.set_page_config(
        page_title="Lifecycle Investing Calculator",
        page_icon="💴",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Cover image banner (Notion-style)
    cover_image_path = _project_root / "image" / "allison-saeng-ZWiaGCFcfJ0-unsplash.jpg"
    cover_base64 = get_base64_image(cover_image_path)
    st.markdown(f"""
        <style>
        .cover-container {{
            position: relative;
            width: 100%;
            height: 200px;
            background: linear-gradient(rgba(0,0,0,0.1), rgba(0,0,0,0.4)),
                        url('data:image/jpeg;base64,{cover_base64}') center/cover;
            border-radius: 0 0 12px 12px;
            margin: -1rem -1rem 1.5rem -1rem;
            padding: 0;
        }}
        .photo-credit {{
            position: absolute;
            bottom: 8px;
            right: 12px;
            font-size: 0.7rem;
            color: rgba(255, 255, 255, 0.7);
        }}
        .photo-credit a {{
            color: rgba(255, 255, 255, 0.7);
            text-decoration: none;
        }}
        .photo-credit a:hover {{
            text-decoration: underline;
        }}
        </style>
        <div class="cover-container">
            <div class="photo-credit">Photo by <a href="https://unsplash.com/@allisonsaeng" target="_blank">Allison Saeng</a> on <a href="https://unsplash.com" target="_blank">Unsplash</a></div>
        </div>
    """, unsafe_allow_html=True)

    # Main title and introduction
    st.title("Lifecycle Investing Calculator")
    st.markdown(
        "**Calculate your optimal portfolio allocation based on lifecycle investing principles**"
    )
    st.markdown(
        "This tool uses the total wealth framework, which considers your human capital "
        "(present value of future labor income), financial assets, and future expenses "
        "to determine an optimal investment strategy that evolves over your lifetime."
    )

    # Collect user inputs from sidebar
    inputs = render_sidebar()

    # Validate age constraints before rendering dashboard
    valid_inputs = (
        inputs["retirement_age"] > inputs["current_age"]
        and inputs["life_expectancy"] > inputs["retirement_age"]
    )

    if not valid_inputs:
        st.warning(
            "Please adjust your age settings in the sidebar. "
            "Retirement age must exceed current age, and life expectancy must exceed retirement age."
        )
        return

    # Map sidebar keys to dashboard expected keys (sidebar uses different naming convention)
    dashboard_inputs = {
        "current_age": inputs["current_age"],
        "retirement_age": inputs["retirement_age"],
        "life_expectancy": inputs["life_expectancy"],
        "annual_income": inputs["annual_income"],
        "working_expenses": inputs["expenses_working"],
        "retirement_expenses": inputs["expenses_retirement"],
        "financial_assets": inputs["financial_assets"],
        "risk_aversion": inputs["risk_aversion"],
        "income_beta": inputs["income_beta"],
    }

    # Render the main dashboard with error handling for calculation failures
    try:
        render_dashboard(dashboard_inputs)
    except Exception as e:
        st.error(
            "An error occurred while calculating your portfolio analysis. "
            "Please check your inputs and try again."
        )
        st.exception(e)
        return

    # Footer with disclaimer
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0; margin-top: 2rem; border-top: 1px solid #e0e0e0;">
            <small style="color: #6b7280;">
                <strong>Disclaimer:</strong> This tool is for educational purposes only and does not constitute financial advice.
            </small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
