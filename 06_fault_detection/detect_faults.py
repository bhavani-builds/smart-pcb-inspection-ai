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

normal = cv2.imread(
    str(normal_path)
)

defective = cv2.imread(
    str(defective_path)
)


if normal is None:

    print("Could not load normal PCB.")

    raise SystemExit


if defective is None:

    print("Could not load defective PCB.")

    raise SystemExit


# Make sure both images have the same size
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
# Calculate difference
# ------------------------------------------

difference = cv2.absdiff(
    normal_gray,
    defective_gray
)


# ------------------------------------------
# Threshold the difference
# ------------------------------------------

_, threshold = cv2.threshold(
    difference,
    30,
    255,
    cv2.THRESH_BINARY
)


# ------------------------------------------
# Remove small noise
# ------------------------------------------

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (5, 5)
)

clean_mask = cv2.morphologyEx(
    threshold,
    cv2.MORPH_OPEN,
    kernel
)

clean_mask = cv2.morphologyEx(
    clean_mask,
    cv2.MORPH_CLOSE,
    kernel
)


# ------------------------------------------
# Find fault contours
# ------------------------------------------

contours, _ = cv2.findContours(
    clean_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# ------------------------------------------
# Draw detected faults
# ------------------------------------------

result = defective.copy()

fault_count = 0

fault_regions = []


for contour in contours:

    area = cv2.contourArea(
        contour
    )

    # Ignore tiny noise
    if area < 100:
        continue


    x, y, width, height = cv2.boundingRect(
        contour
    )


    # Save fault information
    fault_regions.append(
        (
            x,
            y,
            width,
            height,
            area
        )
    )


    # Draw red bounding box
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


    # Add fault label
    cv2.putText(
        result,
        f"FAULT {fault_count + 1}",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    fault_count += 1


# ------------------------------------------
# Convert for plotting
# ------------------------------------------

normal_rgb = cv2.cvtColor(
    normal,
    cv2.COLOR_BGR2RGB
)

defective_rgb = cv2.cvtColor(
    defective,
    cv2.COLOR_BGR2RGB
)

result_rgb = cv2.cvtColor(
    result,
    cv2.COLOR_BGR2RGB
)


# ------------------------------------------
# Display results
# ------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 11)
)


axes[0, 0].imshow(
    normal_rgb
)

axes[0, 0].set_title(
    "Normal PCB",
    fontsize=17,
    fontweight="bold"
)

axes[0, 0].axis("off")


axes[0, 1].imshow(
    defective_rgb
)

axes[0, 1].set_title(
    "Defective PCB",
    fontsize=17,
    fontweight="bold"
)

axes[0, 1].axis("off")


axes[1, 0].imshow(
    clean_mask,
    cmap="gray"
)

axes[1, 0].set_title(
    "Detected Fault Mask",
    fontsize=17,
    fontweight="bold"
)

axes[1, 0].axis("off")


axes[1, 1].imshow(
    result_rgb
)

axes[1, 1].set_title(
    "Fault Localization",
    fontsize=17,
    fontweight="bold"
)

axes[1, 1].axis("off")


fig.suptitle(
    "Automatic PCB Fault Detection — Stage 06",
    fontsize=22,
    fontweight="bold"
)

plt.tight_layout()


# ------------------------------------------
# Save result
# ------------------------------------------

output_folder = (
    project_folder
    / "06_fault_detection"
)

output_folder.mkdir(
    exist_ok=True
)


output_path = (
    output_folder
    / "Figure_6.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------
# Print report
# ------------------------------------------

print("=" * 65)
print("             AUTOMATIC FAULT DETECTION")
print("=" * 65)

print(
    f"\nDetected fault regions: {fault_count}"
)


for number, region in enumerate(
    fault_regions,
    start=1
):

    x, y, width, height, area = region

    print(
        f"\nFault {number}"
    )

    print(
        f"  Location : X={x}, Y={y}"
    )

    print(
        f"  Size     : {width} × {height}"
    )

    print(
        f"  Area     : {area:.0f} pixels"
    )


print(
    "\nResult saved to:"
)

print(
    output_path
)
