import streamlit as st
import cv2
import numpy as np


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Smart PCB Inspector",
    page_icon="🔧",
    layout="wide"
)


# ==========================================
# HEADER
# ==========================================

st.title("🔧 Smart PCB Inspector")

st.write(
    "Computer vision based PCB inspection and "
    "fault localization prototype."
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Inspection Settings")

threshold_value = st.sidebar.slider(
    "Difference Threshold",
    min_value=5,
    max_value=100,
    value=30
)

minimum_area = st.sidebar.slider(
    "Minimum Fault Area",
    min_value=20,
    max_value=1000,
    value=100
)


st.sidebar.info(
    "Adjust these values if the inspection "
    "detects too much or too little."
)


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📷 Upload PCB Image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# INSPECTION FUNCTION
# ==========================================

def inspect_pcb(image, threshold, minimum_area):

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

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = image.copy()

    regions = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < minimum_area:
            continue

        if area > (
            image.shape[0]
            * image.shape[1]
            * 0.5
        ):
            continue

        x, y, width, height = (
            cv2.boundingRect(contour)
        )

        center_x = (
            x + width // 2
        )

        center_y = (
            y + height // 2
        )

        # Prototype severity
        if area >= 3000:
            severity = "HIGH"

        elif area >= 1000:
            severity = "MEDIUM"

        else:
            severity = "LOW"

        regions.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "center_x": center_x,
            "center_y": center_y,
            "area": area,
            "severity": severity
        })

        # Draw fault box
        cv2.rectangle(
            result,
            (x, y),
            (
                x + width,
                y + height
            ),
            (0, 0, 255),
            3
        )

        # Draw center
        cv2.circle(
            result,
            (center_x, center_y),
            6,
            (255, 0, 0),
            -1
        )

        # Label
        cv2.putText(
            result,
            f"FAULT {len(regions)}",
            (
                x,
                max(y - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

    return result, edges, regions


# ==========================================
# MAIN APPLICATION
# ==========================================

if uploaded_file:

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
            "Unable to read the uploaded image."
        )

        st.stop()


    # --------------------------------------
    # Inspect button
    # --------------------------------------

    inspect_button = st.button(
        "🔍 INSPECT PCB",
        type="primary",
        use_container_width=True
    )


    if inspect_button:

        result, edges, regions = inspect_pcb(
            image,
            threshold_value,
            minimum_area
        )


        # ==================================
        # BASIC INFORMATION
        # ==================================

        height, width = image.shape[:2]


        if regions:

            status = "⚠️ REVIEW REQUIRED"

        else:

            status = "✅ NO CANDIDATE FAULTS"


        # Quality score
        if len(regions) == 0:

            quality = 100

        else:

            quality = max(
                20,
                100 - len(regions) * 10
            )


        # ==================================
        # SUMMARY
        # ==================================

        st.divider()

        st.header(
            "📊 Inspection Summary"
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


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
                "Candidate Faults",
                len(regions)
            )


        with col4:

            st.metric(
                "Quality Score",
                f"{quality}/100"
            )


        st.subheader(
            f"Status: {status}"
        )


        # ==================================
        # IMAGE RESULTS
        # ==================================

        st.divider()

        st.header(
            "🖼️ Visual Inspection"
        )


        original_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        result_rgb = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


        left, right = st.columns(2)


        with left:

            st.subheader(
                "Original PCB"
            )

            st.image(
                original_rgb,
                use_container_width=True
            )


        with right:

            st.subheader(
                "Inspection Result"
            )

            st.image(
                result_rgb,
                use_container_width=True
            )


        # ==================================
        # EDGE VIEW
        # ==================================

        with st.expander(
            "🔎 View Computer Vision Analysis"
        ):

            st.image(
                edges,
                caption="Detected PCB edges",
                use_container_width=True
            )


        # ==================================
        # FAULT DETAILS
        # ==================================

        st.divider()

        st.header(
            "📍 Fault Details"
        )


        if regions:

            for number, fault in enumerate(
                regions,
                start=1
            ):

                with st.expander(
                    f"🔴 Fault {number} — "
                    f"{fault['severity']}"
                ):

                    c1, c2 = st.columns(2)


                    with c1:

                        st.write(
                            f"**Location:** "
                            f"({fault['center_x']}, "
                            f"{fault['center_y']})"
                        )

                        st.write(
                            f"**Area:** "
                            f"{fault['area']:.0f} pixels"
                        )


                    with c2:

                        st.write(
                            f"**Dimensions:** "
                            f"{fault['width']} × "
                            f"{fault['height']} pixels"
                        )

                        st.write(
                            f"**Severity:** "
                            f"{fault['severity']}"
                        )


        else:

            st.success(
                "No candidate fault regions "
                "were detected."
            )


        # ==================================
        # INSPECTION REPORT
        # ==================================

        st.divider()

        st.header(
            "📄 Inspection Report"
        )


        report = []

        report.append(
            "SMART PCB INSPECTION REPORT"
        )

        report.append(
            "=" * 50
        )

        report.append(
            f"Image Size: {width} × {height}"
        )

        report.append(
            f"Status: {status}"
        )

        report.append(
            f"Candidate Faults: {len(regions)}"
        )

        report.append(
            f"Quality Score: {quality}/100"
        )

        report.append("")


        for number, fault in enumerate(
            regions,
            start=1
        ):

            report.append(
                f"FAULT {number}"
            )

            report.append(
                f"Location: "
                f"X={fault['center_x']}, "
                f"Y={fault['center_y']}"
            )

            report.append(
                f"Size: "
                f"{fault['width']} × "
                f"{fault['height']} pixels"
            )

            report.append(
                f"Area: "
                f"{fault['area']:.0f} pixels"
            )

            report.append(
                f"Severity: "
                f"{fault['severity']}"
            )

            report.append("")


        report.append(
            "NOTE: Candidate regions detected by "
            "a computer-vision prototype. "
            "Results require engineering validation."
        )


        report_text = "\n".join(
            report
        )


        st.download_button(
            "📥 Download Inspection Report",
            data=report_text,
            file_name="pcb_inspection_report.txt",
            mime="text/plain"
        )


else:

    # ======================================
    # START SCREEN
    # ======================================

    st.info(
        "Upload a PCB image above to begin."
    )


    st.markdown(
        """
### 🔧 How it works

**1. Upload** a PCB image

**2. Inspect** the board using OpenCV

**3. Detect** candidate visual anomalies

**4. Localize** their position

**5. Estimate** their size and severity

**6. Generate** an inspection report

---

### ⚠️ Prototype limitation

This application demonstrates a computer-vision
inspection workflow. A real manufacturing system
would require controlled imaging, board alignment,
validated reference images, labeled defect datasets,
and engineering verification.
"""
    )
