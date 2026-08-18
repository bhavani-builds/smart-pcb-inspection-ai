import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------
# Project paths
# ------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

image_path = (
    project_folder
    / "03_pcb_region_detection"
    / "cropped_pcb.jpg"
)


# ------------------------------------------
# Load PCB
# ------------------------------------------

image = cv2.imread(
    str(image_path)
)

if image is None:

    print("Could not load cropped PCB.")

    print(
        "Run Stage 03 first."
    )

    raise SystemExit


# Make a copy
defective = image.copy()


height, width = defective.shape[:2]


# ------------------------------------------
# Choose safe regions automatically
# ------------------------------------------

# We use positions relative to image size
# so the code works with different resolutions.

x1 = int(width * 0.25)
y1 = int(height * 0.35)

x2 = int(width * 0.55)
y2 = int(height * 0.60)


# ------------------------------------------
# Defect 1 — Broken trace simulation
# ------------------------------------------

cv2.line(
    defective,
    (x1, y1),
    (x1 + int(width * 0.20), y1),
    (0, 0, 0),
    12
)


# ------------------------------------------
# Defect 2 — Local damaged region
# ------------------------------------------

cv2.rectangle(
    defective,
    (x2, y2),
    (
        x2 + int(width * 0.08),
        y2 + int(height * 0.08)
    ),
    (0, 0, 0),
    -1
)


# ------------------------------------------
# Difference image
# ------------------------------------------

original_gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

defective_gray = cv2.cvtColor(
    defective,
    cv2.COLOR_BGR2GRAY
)


difference = cv2.absdiff(
    original_gray,
    defective_gray
)


# Highlight changed regions
_, mask = cv2.threshold(
    difference,
    30,
    255,
    cv2.THRESH_BINARY
)


# ------------------------------------------
# Convert for display
# ------------------------------------------

original_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

defective_rgb = cv2.cvtColor(
    defective,
    cv2.COLOR_BGR2RGB
)


# ------------------------------------------
# Create visualization
# ------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)


axes[0, 0].imshow(
    original_rgb
)

axes[0, 0].set_title(
    "Original PCB",
    fontsize=17,
    fontweight="bold"
)

axes[0, 0].axis("off")


axes[0, 1].imshow(
    defective_rgb
)

axes[0, 1].set_title(
    "Simulated Defects",
    fontsize=17,
    fontweight="bold"
)

axes[0, 1].axis("off")


axes[1, 0].imshow(
    difference,
    cmap="inferno"
)

axes[1, 0].set_title(
    "Difference Map",
    fontsize=17,
    fontweight="bold"
)

axes[1, 0].axis("off")


axes[1, 1].imshow(
    mask,
    cmap="gray"
)

axes[1, 1].set_title(
    "Detected Change Mask",
    fontsize=17,
    fontweight="bold"
)

axes[1, 1].axis("off")


fig.suptitle(
    "PCB Defect Simulation — Stage 05",
    fontsize=22,
    fontweight="bold"
)

plt.tight_layout()


# ------------------------------------------
# Save outputs
# ------------------------------------------

output_folder = (
    project_folder
    / "05_defect_simulation"
)

output_folder.mkdir(
    exist_ok=True
)


figure_path = (
    output_folder
    / "Figure_5.png"
)

defective_path = (
    output_folder
    / "defective_pcb.jpg"
)

mask_path = (
    output_folder
    / "defect_mask.png"
)


plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

cv2.imwrite(
    str(defective_path),
    defective
)

cv2.imwrite(
    str(mask_path),
    mask
)

plt.show()


# ------------------------------------------
# Information
# ------------------------------------------

print("=" * 60)
print("             PCB DEFECT SIMULATION")
print("=" * 60)

print(
    "\nSimulated defects:"
)

print(
    "1. Broken trace"
)

print(
    "2. Local damaged region"
)

print(
    "\nGenerated files:"
)

print(
    figure_path
)

print(
    defective_path
)

print(
    mask_path
)
