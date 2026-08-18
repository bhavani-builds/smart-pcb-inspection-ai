import cv2
import pandas as pd
from pathlib import Path


# ------------------------------------------
# Project paths
# ------------------------------------------

project_folder = (
    Path(__file__).resolve().parent.parent
)

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

normal = cv2.imread(
    str(normal_path)
)

defective = cv2.imread(
    str(defective_path)
)


if normal is None or defective is None:

    print("❌ PCB images could not be loaded.")

    raise SystemExit


# Make image sizes equal
if normal.shape != defective.shape:

    defective = cv2.resize(
        defective,
        (
            normal.shape[1],
            normal.shape[0]
        )
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
# Find differences
# ------------------------------------------

difference = cv2.absdiff(
    normal_gray,
    defective_gray
)


# ------------------------------------------
# Create binary fault mask
# ------------------------------------------

_, mask = cv2.threshold(
    difference,
    30,
    255,
    cv2.THRESH_BINARY
)


# Clean mask
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
# Extract features
# ------------------------------------------

feature_rows = []


for number, contour in enumerate(
    contours,
    start=1
):

    area = cv2.contourArea(
        contour
    )


    # Ignore tiny noise
    if area < 100:

        continue


    x, y, width, height = cv2.boundingRect(
        contour
    )


    # Aspect ratio
    aspect_ratio = (
        width / height
        if height != 0
        else 0
    )


    # Region of interest
    roi = defective_gray[
        y:y + height,
        x:x + width
    ]


    # Average brightness
    average_brightness = (
        roi.mean()
        if roi.size > 0
        else 0
    )


    # Edge detection inside fault region
    roi_edges = cv2.Canny(
        roi,
        50,
        150
    )


    # Edge density
    edge_pixels = cv2.countNonZero(
        roi_edges
    )

    total_pixels = roi.size

    edge_density = (
        edge_pixels / total_pixels
        if total_pixels > 0
        else 0
    )


    # Store features
    feature_rows.append({

        "fault_id":
            number,

        "x":
            x,

        "y":
            y,

        "width":
            width,

        "height":
            height,

        "area":
            round(area, 2),

        "aspect_ratio":
            round(aspect_ratio, 3),

        "average_brightness":
            round(
                average_brightness,
                2
            ),

        "edge_density":
            round(
                edge_density,
                4
            )
    })


# ------------------------------------------
# Create DataFrame
# ------------------------------------------

features = pd.DataFrame(
    feature_rows
)


# ------------------------------------------
# Display results
# ------------------------------------------

print("=" * 70)
print("              PCB FAULT FEATURE EXTRACTION")
print("=" * 70)


if features.empty:

    print(
        "\nNo fault regions were detected."
    )

else:

    print(
        f"\nDetected fault regions: "
        f"{len(features)}"
    )

    print("\nExtracted features:\n")

    print(
        features.to_string(
            index=False
        )
    )


# ------------------------------------------
# Save features
# ------------------------------------------

output_folder = (
    project_folder
    / "08_feature_extraction"
)

output_folder.mkdir(
    exist_ok=True
)


output_file = (
    output_folder
    / "fault_features.csv"
)


features.to_csv(
    output_file,
    index=False
)


print(
    "\nFeature data saved to:"
)

print(
    output_file
)
