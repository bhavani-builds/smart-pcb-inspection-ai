import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------
# Project paths
# ------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

normal_path = (
    project_folder
    / "03_pcb_region_detection"
    / "cropped_pcb.jpg"
)

defective_path = (
    project_folder
    / "05_defect_simulation"
    / "defective_pcb.jpg"
)


# ------------------------------------------
# Load images
# ------------------------------------------

normal = cv2.imread(str(normal_path))
defective = cv2.imread(str(defective_path))

if normal is None or defective is None:
    print("❌ Required PCB images were not found.")
    raise SystemExit


# Make dimensions equal
if normal.shape != defective.shape:
    defective = cv2.resize(
        defective,
        (normal.shape[1], normal.shape[0])
    )


# ------------------------------------------
# Convert to grayscale
# ------------------------------------------

normal_gray = cv2.cvtColor(
    normal,
    cv2.COLOR_BGR2GRAY
)

defective_gray = cv2.cvtColor(
    defective,
    cv2.COLOR_BGR2GRAY
)


# ------------------------------------------
# Difference detection
# ------------------------------------------

difference = cv2.absdiff(
    normal_gray,
    defective_gray
)


_, mask = cv2.threshold(
    difference,
    30,
    255,
    cv2.THRESH_BINARY
)


# Clean the mask
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (5, 5)
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)


# ------------------------------------------
# Find fault regions
# ------------------------------------------

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# ------------------------------------------
# Analyze faults
# ------------------------------------------

result = defective.copy()

faults = []


for contour in contours:

    area = cv2.contourArea(contour)

    if area < 100:
        continue


    x, y, width, height = cv2.boundingRect(
        contour
    )


    # --------------------------------------
    # Calculate center
    # --------------------------------------

    center_x = x + width // 2
    center_y = y + height // 2


    # --------------------------------------
    # Determine severity
    # --------------------------------------

    if area < 1000:

        severity = "LOW"

    elif area < 3000:

        severity = "MEDIUM"

    else:

        severity = "HIGH"


    # --------------------------------------
    # Store information
    # --------------------------------------

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
# Draw fault information
# ------------------------------------------

for number, fault in enumerate(
    faults,
    start=1
):

    x = fault["x"]
    y = fault["y"]

    width = fault["width"]
    height = fault["height"]

    center_x = fault["center_x"]
    center_y = fault["center_y"]

    severity = fault["severity"]


    # Draw bounding box
    cv2.rectangle(
        result,
        (x, y),
        (x + width, y + height),
        (0, 0, 255),
        3
    )


    # Draw center point
    cv2.circle(
        result,
        (center_x, center_y),
        7,
        (255, 0, 0),
        -1
    )


    # Label
    label = (
        f"FAULT {number} - {severity}"
    )

    cv2.putText(
        result,
        label,
        (x, max(y - 12, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )


# ------------------------------------------
# Display
# ------------------------------------------

result_rgb = cv2.cvtColor(
    result,
    cv2.COLOR_BGR2RGB
)


fig, ax = plt.subplots(
    figsize=(13, 8)
)

ax.imshow(
    result_rgb
)

ax.set_title(
    "PCB Fault Localization & Severity",
    fontsize=21,
    fontweight="bold"
)

ax.axis("off")

plt.tight_layout()


# ------------------------------------------
# Save visualization
# ------------------------------------------

output_folder = (
    project_folder
    / "07_fault_localization"
)

output_folder.mkdir(
    exist_ok=True
)


figure_path = (
    output_folder
    / "Figure_7.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------
# Print inspection report
# ------------------------------------------

print("=" * 70)
print("           PCB FAULT LOCALIZATION REPORT")
print("=" * 70)


print(
    f"\nTotal faults detected: {len(faults)}"
)


for number, fault in enumerate(
    faults,
    start=1
):

    print(
        f"\n🔴 FAULT {number}"
    )

    print(
        f"   Location : "
        f"X={fault['center_x']}, "
        f"Y={fault['center_y']}"
    )

    print(
        f"   Size     : "
        f"{fault['width']} × "
        f"{fault['height']} pixels"
    )

    print(
        f"   Area     : "
        f"{fault['area']:.0f} pixels"
    )

    print(
        f"   Severity : "
        f"{fault['severity']}"
    )


# ------------------------------------------
# Save text report
# ------------------------------------------

report_path = (
    output_folder
    / "fault_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PCB FAULT INSPECTION REPORT\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Total Faults: {len(faults)}\n\n"
    )


    for number, fault in enumerate(
        faults,
        start=1
    ):

        file.write(
            f"FAULT {number}\n"
        )

        file.write(
            f"Location: "
            f"X={fault['center_x']}, "
            f"Y={fault['center_y']}\n"
        )

        file.write(
            f"Size: "
            f"{fault['width']} × "
            f"{fault['height']} pixels\n"
        )

        file.write(
            f"Area: "
            f"{fault['area']:.0f} pixels\n"
        )

        file.write(
            f"Severity: "
            f"{fault['severity']}\n\n"
        )


print(
    "\n📄 Report saved to:"
)

print(
    report_path
)

print(
    "\n🖼️ Visualization saved to:"
)

print(
    figure_path
)
