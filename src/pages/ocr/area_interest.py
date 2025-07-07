"""Module untuk deteksi dan display area of interest dari voucher."""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from mylog import logger, timer


@timer("Image Preprocessing")
def preprocess_image_for_detection(
    image: Image.Image,
    blur_kernel: int = 3,
    morph_kernel: int = 3,
    use_histogram_eq: bool = True,
) -> np.ndarray:
    """Pre-process image untuk meningkatkan deteksi voucher dengan parameter dinamis.

    :param image: PIL Image object.
    :param blur_kernel: Size of Gaussian blur kernel (must be odd).
    :param morph_kernel: Size of morphological operations kernel.
    :param use_histogram_eq: Whether to apply histogram equalization.
    :return: Processed image dalam format OpenCV.
    """
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 1. Optional histogram equalization untuk meningkatkan kontras
    processed = cv2.equalizeHist(gray) if use_histogram_eq else gray

    # 2. Gaussian blur untuk mengurangi noise
    if blur_kernel > 1:
        # Ensure kernel size is odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)

    # 3. Morphological operations untuk membersihkan noise
    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    return processed


@timer("Voucher Region Detection")
def detect_voucher_regions(
    image: Image.Image,
    canny_low: int = 30,
    canny_high: int = 150,
    min_area_percent: float = 1.0,
    max_area_percent: float = 80.0,
    min_aspect_ratio: float = 1.5,
    max_aspect_ratio: float = 8.0,
    min_width_percent: float = 15.0,
    min_height_percent: float = 5.0,
    blur_kernel: int = 3,
    morph_kernel: int = 3,
    use_histogram_eq: bool = True,
) -> list:
    """Detect voucher regions dengan parameter yang bisa disesuaikan.

    :param image: PIL Image object.
    :param canny_low: Lower threshold untuk Canny edge detection.
    :param canny_high: Higher threshold untuk Canny edge detection.
    :param min_area_percent: Minimal area sebagai persentase dari total gambar.
    :param max_area_percent: Maksimal area sebagai persentase dari total gambar.
    :param min_aspect_ratio: Minimal aspect ratio (width/height).
    :param max_aspect_ratio: Maksimal aspect ratio (width/height).
    :param min_width_percent: Minimal width sebagai persentase dari total width.
    :param min_height_percent: Minimal height sebagai persentase dari total height.
    :param blur_kernel: Kernel size untuk Gaussian blur.
    :param morph_kernel: Kernel size untuk morphological operations.
    :param use_histogram_eq: Whether to use histogram equalization.
    :return: List of tuples containing (x, y, w, h) coordinates of detected vouchers.
    """
    # Pre-process image dengan parameter dinamis
    processed_img = preprocess_image_for_detection(
        image, blur_kernel, morph_kernel, use_histogram_eq
    )

    # Edge detection dengan parameter dinamis
    edges = cv2.Canny(processed_img, canny_low, canny_high)

    # Morphological closing untuk menghubungkan garis putus-putus
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    logger.debug(f"Found {len(contours)} initial contours.")

    # Filter contours dengan parameter dinamis
    voucher_regions = []

    # Calculate thresholds berdasarkan parameter
    image_area = image.width * image.height
    min_area = image_area * (min_area_percent / 100)
    max_area = image_area * (max_area_percent / 100)
    min_relative_width = min_width_percent / 100
    min_relative_height = min_height_percent / 100

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Check aspect ratio
            aspect_ratio = w / h
            if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                # Check relative size
                relative_width = w / image.width
                relative_height = h / image.height

                if (
                    relative_width > min_relative_width
                    and relative_height > min_relative_height
                ):
                    voucher_regions.append((x, y, w, h))

    # Sort berdasarkan posisi Y (dari atas ke bawah)
    voucher_regions.sort(key=lambda region: region[1])
    logger.info(f"Filtered down to {len(voucher_regions)} potential voucher regions.")

    return voucher_regions


def draw_bounding_boxes(image: Image.Image, regions: list) -> Image.Image:
    """Draw bounding boxes on the image to show detected voucher regions.

    :param image: PIL Image object.
    :param regions: List of (x, y, w, h) coordinates.
    :return: PIL Image with bounding boxes drawn.
    """
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Draw bounding boxes
    for i, (x, y, w, h) in enumerate(regions):
        # Draw rectangle
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Add label
        label = f"Voucher #{i + 1}"
        cv2.putText(
            img_cv, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    # Convert back to PIL
    img_with_boxes = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_with_boxes


def detect_and_display_areas(image: Image.Image, params: dict) -> None:
    """Detect and display areas of interest untuk voucher.

    :param image: PIL Image object.
    :param params: Dictionary containing detection parameters.
    """
    st.write("🔍 Mendeteksi area voucher...")

    # Detect voucher regions dengan parameter dari sidebar
    voucher_regions = detect_voucher_regions(
        image,
        canny_low=params["canny_low"],
        canny_high=params["canny_high"],
        min_area_percent=params["min_area_percent"],
        max_area_percent=params["max_area_percent"],
        min_aspect_ratio=params["min_aspect_ratio"],
        max_aspect_ratio=params["max_aspect_ratio"],
        min_width_percent=params["min_width_percent"],
        min_height_percent=params["min_height_percent"],
        blur_kernel=params["blur_kernel"],
        morph_kernel=params["morph_kernel"],
        use_histogram_eq=params["use_histogram_eq"],
    )

    # Display results dalam expander
    with st.expander(
        f"📦 Area of Interest - Terdeteksi {len(voucher_regions)} area voucher",
        expanded=True,
    ):
        # Show current parameters
        with st.expander("⚙️ Parameter yang Sedang Digunakan", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Pre-processing:**")
                st.write(f"• Histogram EQ: {params['use_histogram_eq']}")
                st.write(f"• Blur Kernel: {params['blur_kernel']}")
                st.write(f"• Morph Kernel: {params['morph_kernel']}")
            with col2:
                st.write("**Edge Detection:**")
                st.write(f"• Canny Low: {params['canny_low']}")
                st.write(f"• Canny High: {params['canny_high']}")
                st.write("**Area Filtering:**")
                st.write(f"• Min Area: {params['min_area_percent']}%")
                st.write(f"• Max Area: {params['max_area_percent']}%")
            with col3:
                st.write("**Shape Filtering:**")
                st.write(f"• Min Aspect: {params['min_aspect_ratio']}")
                st.write(f"• Max Aspect: {params['max_aspect_ratio']}")
                st.write(f"• Min Width: {params['min_width_percent']}%")
                st.write(f"• Min Height: {params['min_height_percent']}%")

        if len(voucher_regions) == 0:
            st.warning("Tidak ada voucher terdeteksi dengan parameter saat ini.")
            st.info("💡 Coba adjust parameter di sidebar untuk memperbaiki deteksi:")
            st.markdown("- **Turunkan Canny Low/High** jika edge tidak terdeteksi")
            st.markdown("- **Turunkan Min Area %** jika voucher terlalu kecil")
            st.markdown("- **Adjust Aspect Ratio** sesuai bentuk voucher")
            logger.warning("No voucher regions were detected in the uploaded image.")
        else:
            # Draw bounding boxes on image
            image_with_boxes = draw_bounding_boxes(image, voucher_regions)

            st.image(
                image_with_boxes,
                caption=f"Gambar dengan {len(voucher_regions)} area voucher yang terdeteksi",
                use_container_width=True,
            )

            # Show details of each detected region
            st.subheader("Detail Area yang Terdeteksi:")
            for i, (x, y, w, h) in enumerate(voucher_regions):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric(f"Voucher #{i + 1}", "")
                with col2:
                    st.metric("X", x)
                with col3:
                    st.metric("Y", y)
                with col4:
                    st.metric("Width", w)
                with col5:
                    st.metric("Height", h)

                # Calculate and show area
                area = w * h
                st.caption(f"Area: {area:,} pixels | Aspect Ratio: {w / h:.2f}")
                st.divider()

    # Store detected regions in session state for next steps
    st.session_state.voucher_regions = voucher_regions
