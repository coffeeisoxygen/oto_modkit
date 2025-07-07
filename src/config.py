"""Configuration module untuk OCR Voucher project."""

import streamlit as st


def get_tesseract_path() -> str:
    """Get tesseract path dari secrets.toml atau default path.

    Returns:
        str: Path to tesseract executable
    """
    try:
        # Try to get from secrets.toml
        return st.secrets["tesseract_path"]
    except (KeyError, FileNotFoundError):
        # Fallback to default Windows path
        return r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Set tesseract path at module level
TESSERACT_PATH = get_tesseract_path()
