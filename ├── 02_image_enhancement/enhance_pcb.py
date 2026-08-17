import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------
# Project paths
# ------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

image_path = (
    project_folder
    / "01_pcb_input"
    / "pcb.jpg"
)


# ------------------------------------------
# Load image
# ------------------------------------------

image = cv2.imread(
    str(image_path)
)


if image is None:

    print("❌ Could not load PCB image.")

    raise SystemExit


# Convert BGR to RGB
rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ------------------------------------------
# Grayscale
# ------------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ------------------------------------------
# Noise reduction
# ------------------------------------------

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# ------------------------------------------
# Improve contrast
# ------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(
    blurred
)


# ------------------------------------------
# Edge detection
# ------------------------------------------

edges = cv2.Canny(
    enhanced,
    50,
    150
)


# ------------------------------------------
# Display results
# ------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)


axes[0, 0].imshow(
    rgb_image
)

axes[0, 0].set_title(
    "Original PCB",
    fontsize=16,
    fontweight="bold"
)

axes[0, 0].axis("off")


axes[0, 1].imshow(
    gray,
    cmap="gray"
)

axes[0, 1].set_title(
    "Grayscale",
    fontsize=16,
    fontweight="bold"
)

axes[0, 1].axis("off")


axes[1, 0].imshow(
    enhanced,
    cmap="gray"
)

axes[1, 0].set_title(
    "Enhanced PCB",
    fontsize=16,
    fontweight="bold"
)

axes[1, 0].axis("off")


axes[1, 1].imshow(
    edges,
    cmap="gray"
)

axes[1, 1].set_title(
    "Edge Detection",
    fontsize=16,
    fontweight="bold"
)

axes[1, 1].axis("off")


fig.suptitle(
    "PCB Image Enhancement — Stage 02",
    fontsize=22,
    fontweight="bold"
)

plt.tight_layout()

# ------------------------------------------
# Save result
# ------------------------------------------

output_path = (
    project_folder
    / "02_image_enhancement"
    / "Figure_2.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("=" * 60)
print("       PCB IMAGE ENHANCEMENT COMPLETED")
print("=" * 60)

print(
    "\nEnhanced image saved to:"
)

print(
    output_path
)
