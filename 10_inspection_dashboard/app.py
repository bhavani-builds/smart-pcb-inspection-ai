import streamlit as st
import cv2
import numpy as np
from pathlib import Path


# ------------------------------------------
# Page settings
# ------------------------------------------

st.set_page_config(
    page_title="Smart PCB Inspector",
    page_icon="🔧",
    layout="wide"
)


# ------------------------------------------
# Title
# ------------------------------------------

st.title(
    "🔧 Smart PCB Inspection System"
)

st.write(
    "Computer vision based prototype for PCB "
    "fault detection and localization."
)

st.divider()


# ------------------------------------------
# Upload PCB image
# ------------------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload a PCB image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ------------------------------------------
# Analyze image
# ------------------------------------------

if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


    if image is None:

        st.error(
            "Could not read the image."
        )

        st.stop()


    # Convert for Streamlit
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------
    # Image information
    # --------------------------------------

    height, width = image.shape[:2]


    # --------------------------------------
    # Basic image processing
    # --------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )


    # --------------------------------------
    # Find contours
    # --------------------------------------

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # --------------------------------------
    # Detect candidate regions
    # --------------------------------------

    result = image.copy()

    fault_count = 0

    fault_data = []


    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # Ignore very small regions
        if area < 150:

            continue

        # Ignore huge background regions
        if area > (
            0.5
            * width
            * height
        ):

            continue


        x, y, w, h = cv2.boundingRect(
            contour
        )


        # Store information
        fault_data.append({

            "x": x,

            "y": y,

            "width": w,

            "height": h,

            "area": area

        })


        # Draw bounding box
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )


        # Add label
        cv2.putText(
            result,
            f"Region {fault_count + 1}",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )


        fault_count += 1


    result_rgb = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------
    # Dashboard metrics
    # --------------------------------------

    st.subheader(
        "📊 Inspection Summary"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "PCB Width",
            f"{width}px"
        )


    with col2:

        st.metric(
            "PCB Height",
            f"{height}px"
        )


    with col3:

        st.metric(
            "Detected Regions",
            fault_count
        )


    with col4:

        if fault_count > 0:

            st.metric(
                "Inspection Status",
                "⚠️ CHECK"
            )

        else:

            st.metric(
                "Inspection Status",
                "✅ CLEAR"
            )


    st.divider()


    # --------------------------------------
    # Image comparison
    # --------------------------------------

    st.subheader(
        "🔍 Visual Inspection"
    )


    left, right = st.columns(2)


    with left:

        st.write(
            "**Original PCB**"
        )

        st.image(
            rgb_image,
            use_container_width=True
        )


    with right:

        st.write(
            "**Detected Regions**"
        )

        st.image(
            result_rgb,
            use_container_width=True
        )


    # --------------------------------------
    # Edge view
    # --------------------------------------

    with st.expander(
        "🔎 View Edge Analysis"
    ):

        st.image(
            edges,
            caption="Detected PCB edges",
            use_container_width=True
        )


    # --------------------------------------
    # Fault table
    # --------------------------------------

    st.subheader(
        "📍 Detected Regions"
    )


    if fault_data:

        for number, fault in enumerate(
            fault_data,
            start=1
        ):

            x = fault["x"]
            y = fault["y"]
            w = fault["width"]
            h = fault["height"]
            area = fault["area"]


            with st.expander(
                f"Region {number}"
            ):

                c1, c2, c3 = st.columns(3)


                with c1:

                    st.write(
                        f"**Location:** "
                        f"({x}, {y})"
                    )


                with c2:

                    st.write(
                        f"**Size:** "
                        f"{w} × {h}px"
                    )


                with c3:

                    st.write(
                        f"**Area:** "
                        f"{area:.0f}px²"
                    )


    else:

        st.success(
            "No candidate regions detected."
        )


# ------------------------------------------
# Instructions
# ------------------------------------------

else:

    st.info(
        "Upload a PCB image above to begin inspection."
    )

    st.markdown(
        """
### How it works

1. Upload a PCB image.
2. The image is converted to grayscale.
3. Noise is reduced.
4. Edges are detected.
5. Candidate regions are identified.
6. Detected regions are highlighted.

> **Prototype note:** detected regions are candidate visual anomalies,
> not confirmed manufacturing defects. A production system would
> require validated reference images and labeled inspection data.
        """
    )
