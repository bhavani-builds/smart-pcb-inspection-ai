import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

image_path = (
    project_folder
    / "01_pcb_input"
    / "pcb.jpg"
)


# Load the PCB image
image = cv2.imread(str(image_path))

if image is None:
    print("Could not load the PCB image.")
    print("Check that pcb.jpg is inside 01_pcb_input.")
    raise SystemExit


original = image.copy()


# Convert to grayscale
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# Reduce small amounts of noise
blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# Find edges
edges = cv2.Canny(
    blurred,
    50,
    150
)


# Find outer contours
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# Find the largest contour
largest_contour = None
largest_area = 0

for contour in contours:

    area = cv2.contourArea(contour)

    if area > largest_area:
        largest_area = area
        largest_contour = contour


if largest_contour is None:

    print("No PCB region was detected.")
    raise SystemExit


# Get the board boundary
x, y, width, height = cv2.boundingRect(
    largest_contour
)


# Draw the detected boundary
boundary_image = original.copy()

cv2.rectangle(
    boundary_image,
    (x, y),
    (x + width, y + height),
    (0, 255, 0),
    4
)


# Crop the PCB
cropped_pcb = original[
    y:y + height,
    x:x + width
]


# Convert images for plotting
original_rgb = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

boundary_rgb = cv2.cvtColor(
    boundary_image,
    cv2.COLOR_BGR2RGB
)

cropped_rgb = cv2.cvtColor(
    cropped_pcb,
    cv2.COLOR_BGR2RGB
)


# Display results
fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


axes[0].imshow(original_rgb)
axes[0].set_title(
    "Original PCB",
    fontsize=17,
    fontweight="bold"
)
axes[0].axis("off")


axes[1].imshow(boundary_rgb)
axes[1].set_title(
    "Detected PCB Region",
    fontsize=17,
    fontweight="bold"
)
axes[1].axis("off")


axes[2].imshow(cropped_rgb)
axes[2].set_title(
    "Cropped PCB",
    fontsize=17,
    fontweight="bold"
)
axes[2].axis("off")


fig.suptitle(
    "PCB Region Detection — Stage 03",
    fontsize=22,
    fontweight="bold"
)

plt.tight_layout()


# Create output folder
output_folder = (
    project_folder
    / "03_pcb_region_detection"
)

output_folder.mkdir(
    exist_ok=True
)


# Save visualization
figure_path = (
    output_folder
    / "Figure_3.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)


# Save cropped PCB
crop_path = (
    output_folder
    / "cropped_pcb.jpg"
)

cv2.imwrite(
    str(crop_path),
    cropped_pcb
)


plt.show()


# Print information
print("=" * 60)
print("          PCB REGION DETECTION")
print("=" * 60)

print(
    f"\nDetected PCB area : {largest_area:.0f} pixels"
)

print(
    f"PCB position      : X={x}, Y={y}"
)

print(
    f"PCB dimensions    : {width} × {height} pixels"
)

print("\nPCB region detected successfully.")

print(
    "\nSaved visualization:",
    figure_path
)

print(
    "Saved cropped PCB:",
    crop_path
)
