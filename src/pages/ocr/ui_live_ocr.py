"""ui untuk melakukan upload dan di panggil dari multipage apps."""

import streamlit as st
from PIL import Image

from mylog import logger
from pages.ocr.area_interest import detect_and_display_areas
from pages.ocr.sidebar import render_sidebar


def main():
    """Main OCR page untuk voucher detection."""
    st.title("📱 Indosat Voucher OCR - Deteksi Area Voucher")

    # Render sidebar untuk parameter tuning
    params = render_sidebar()

    # Upload file section
    st.subheader("📁 Upload Gambar Voucher")
    uploaded_file = st.file_uploader(
        "Pilih gambar voucher",
        type=["png", "jpg", "jpeg"],
        help="Upload gambar voucher Indosat untuk dianalisis",
    )

    if uploaded_file is not None:
        logger.info(f"File uploaded: {uploaded_file.name}")

        # Load and store image in session state
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        st.session_state.upload_params = params

        # Preview gambar dalam expander
        with st.expander("📷 Preview Gambar Original", expanded=True):
            st.image(
                image, caption="Gambar Voucher yang di-upload", use_container_width=True
            )
            st.caption(f"Ukuran gambar: {image.width} x {image.height} pixels")

        # Detect and display areas of interest
        detect_and_display_areas(image, params)

    else:
        st.info("Silakan upload gambar voucher untuk memulai deteksi area voucher.")
        # Clear session state if no file uploaded
        if "uploaded_image" in st.session_state:
            del st.session_state.uploaded_image
        if "upload_params" in st.session_state:
            del st.session_state.upload_params
        logger.info("Page loaded. Waiting for file upload.")


if __name__ == "__main__":
    main()
else:
    # When called from main.py, just run main
    main()
