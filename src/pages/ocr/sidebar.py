"""component untuk sidebar pada halaman ocr."""

import streamlit as st


def render_sidebar():
    """Render sidebar dengan parameter tuning untuk deteksi voucher.

    Returns:
        dict: Dictionary berisi semua parameter untuk deteksi voucher
    """
    st.sidebar.header("🔧 Parameter Tuning")
    st.sidebar.markdown("Adjust parameter untuk memperbaiki deteksi area voucher")

    # Pre-processing parameters
    st.sidebar.subheader("Pre-processing")
    use_histogram_eq = st.sidebar.checkbox(
        "Histogram Equalization", value=True, help="Meningkatkan kontras gambar"
    )
    blur_kernel = st.sidebar.slider(
        "Gaussian Blur Kernel",
        1,
        15,
        3,
        step=2,
        help="Mengurangi noise (harus angka ganjil)",
    )
    morph_kernel = st.sidebar.slider(
        "Morphological Kernel", 1, 15, 3, help="Membersihkan noise"
    )

    # Edge detection parameters
    st.sidebar.subheader("Edge Detection")
    canny_low = st.sidebar.slider(
        "Canny Low Threshold", 10, 100, 30, help="Threshold rendah untuk edge detection"
    )
    canny_high = st.sidebar.slider(
        "Canny High Threshold",
        50,
        300,
        150,
        help="Threshold tinggi untuk edge detection",
    )

    # Area filtering parameters
    st.sidebar.subheader("Area Filtering")
    min_area_percent = st.sidebar.slider(
        "Min Area (%)",
        0.1,
        10.0,
        1.0,
        step=0.1,
        help="Minimal area sebagai % dari total gambar",
    )
    max_area_percent = st.sidebar.slider(
        "Max Area (%)",
        50.0,
        95.0,
        80.0,
        step=5.0,
        help="Maksimal area sebagai % dari total gambar",
    )

    # Aspect ratio parameters
    st.sidebar.subheader("Aspect Ratio")
    min_aspect_ratio = st.sidebar.slider(
        "Min Aspect Ratio", 0.5, 3.0, 1.5, step=0.1, help="Minimal rasio width/height"
    )
    max_aspect_ratio = st.sidebar.slider(
        "Max Aspect Ratio", 3.0, 15.0, 8.0, step=0.5, help="Maksimal rasio width/height"
    )

    # Relative size parameters
    st.sidebar.subheader("Relative Size")
    min_width_percent = st.sidebar.slider(
        "Min Width (%)",
        5.0,
        50.0,
        15.0,
        step=1.0,
        help="Minimal width sebagai % dari total width",
    )
    min_height_percent = st.sidebar.slider(
        "Min Height (%)",
        1.0,
        20.0,
        5.0,
        step=0.5,
        help="Minimal height sebagai % dari total height",
    )

    # Reset button
    if st.sidebar.button("🔄 Reset ke Default"):
        st.rerun()

    # Tips section
    st.sidebar.markdown("---")
    st.sidebar.markdown("💡 **Tips:**")
    st.sidebar.markdown(
        "- Jika voucher tidak terdeteksi, coba turunkan threshold Canny"
    )
    st.sidebar.markdown("- Jika terlalu banyak area terdeteksi, naikkan Min Area %")
    st.sidebar.markdown("- Untuk voucher horizontal, gunakan Min Aspect Ratio > 1.5")

    # Return all parameters as dictionary
    return {
        "use_histogram_eq": use_histogram_eq,
        "blur_kernel": blur_kernel,
        "morph_kernel": morph_kernel,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "min_area_percent": min_area_percent,
        "max_area_percent": max_area_percent,
        "min_aspect_ratio": min_aspect_ratio,
        "max_aspect_ratio": max_aspect_ratio,
        "min_width_percent": min_width_percent,
        "min_height_percent": min_height_percent,
    }
