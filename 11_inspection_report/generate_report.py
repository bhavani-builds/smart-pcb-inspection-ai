import cv2
from pathlib import Path
from datetime import datetime


# ------------------------------------------
# Project paths
# ------------------------------------------

project_folder = (
    Path(__file__).resolve().parent.parent
)

image_path = (
    project_folder
    / "05_defect_simulation"
    / "defective_pcb.jpg"
)


# ------------------------------------------
# Load PCB image
# ------------------------------------------

image = cv2.imread(
    str(image_path)
)

if image is None:

    print("Could not load PCB image.")

    raise SystemExit


# ------------------------------------------
# Basic image processing
# ------------------------------------------

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


# ------------------------------------------
# Find candidate regions
# ------------------------------------------

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


faults = []


for contour in contours:

    area = cv2.contourArea(
        contour
    )

    if area < 150:
        continue

    x, y, width, height = (
        cv2.boundingRect(contour)
    )

    if area > (
        0.5
        * image.shape[0]
        * image.shape[1]
    ):
        continue

    center_x = (
        x + width // 2
    )

    center_y = (
        y + height // 2
    )


    # Simple prototype severity
    if area >= 3000:

        severity = "HIGH"

    elif area >= 1000:

        severity = "MEDIUM"

    else:

        severity = "LOW"


    faults.append({

        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "center_x": center_x,
        "center_y": center_y,
        "area": area,
        "severity": severity

    })


# ------------------------------------------
# Board information
# ------------------------------------------

height, width = image.shape[:2]


# ------------------------------------------
# Overall status
# ------------------------------------------

if len(faults) == 0:

    status = "CLEAR"

else:

    status = "REQUIRES INSPECTION"


# ------------------------------------------
# Calculate quality score
# ------------------------------------------

if len(faults) == 0:

    quality_score = 100

else:

    penalty = min(
        len(faults) * 10,
        80
    )

    quality_score = 100 - penalty


# ------------------------------------------
# Create report
# ------------------------------------------

output_folder = (
    project_folder
    / "11_inspection_report"
)

output_folder.mkdir(
    exist_ok=True
)


report_path = (
    output_folder
    / "pcb_inspection_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "=" * 70 + "\n"
    )

    report.write(
        "              SMART PCB INSPECTION REPORT\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )


    # Date and time
    report.write(
        "Inspection Date : "
        + datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        + "\n"
    )


    report.write(
        "Image Name      : "
        + image_path.name
        + "\n"
    )


    report.write(
        "Board Size      : "
        + f"{width} × {height} pixels"
        + "\n\n"
    )


    # Overall result
    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        "INSPECTION SUMMARY\n"
    )

    report.write(
        "-" * 70 + "\n\n"
    )


    report.write(
        f"Status          : {status}\n"
    )

    report.write(
        f"Detected Regions: {len(faults)}\n"
    )

    report.write(
        f"Quality Score   : "
        f"{quality_score}/100\n\n"
    )


    # Fault details
    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        "FAULT DETAILS\n"
    )

    report.write(
        "-" * 70 + "\n\n"
    )


    if faults:

        for number, fault in enumerate(
            faults,
            start=1
        ):

            report.write(
                f"Fault Region {number}\n"
            )

            report.write(
                f"  Location   : "
                f"X={fault['center_x']}, "
                f"Y={fault['center_y']}\n"
            )

            report.write(
                f"  Bounding Box: "
                f"{fault['width']} × "
                f"{fault['height']} pixels\n"
            )

            report.write(
                f"  Area       : "
                f"{fault['area']:.0f} pixels\n"
            )

            report.write(
                f"  Severity   : "
                f"{fault['severity']}\n\n"
            )

    else:

        report.write(
            "No candidate fault regions detected.\n"
        )


    # Recommendations
    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        "RECOMMENDATION\n"
    )

    report.write(
        "-" * 70 + "\n\n"
    )


    if faults:

        report.write(
            "The PCB contains candidate visual "
            "anomalies.\n"
        )

        report.write(
            "Review the highlighted regions "
            "manually before accepting the board.\n"
        )

    else:

        report.write(
            "No candidate visual anomalies were "
            "detected by this prototype.\n"
        )


    # Prototype disclaimer
    report.write(
        "\n\n"
    )

    report.write(
        "NOTE:\n"
    )

    report.write(
        "This report is generated by a computer "
        "vision prototype.\n"
    )

    report.write(
        "Detected regions are candidate anomalies "
        "and are not a certified manufacturing "
        "inspection result.\n"
    )


# ------------------------------------------
# Display result
# ------------------------------------------

print("=" * 70)
print("             PCB INSPECTION REPORT")
print("=" * 70)

print(
    f"\nStatus          : {status}"
)

print(
    f"Detected Regions: {len(faults)}"
)

print(
    f"Quality Score   : {quality_score}/100"
)

print(
    "\nReport generated successfully."
)

print(
    "\nSaved to:"
)

print(
    report_path
)
