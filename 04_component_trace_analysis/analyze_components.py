import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

image_path = (
    project_folder
    / "03_pcb_region_detection"
    / "cropped_pcb.jpg"
)


# Load the cropped PCB
image = cv2.imread(str(image_path))

if image is None:
    print("Could not load cropped PCB.")
    print("Run Stage 03 first.")
    raise SystemExit


original = image.copy()


# Convert to grayscale
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# Reduce noise
blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# Detect edges
edges = cv2.Canny(
    blurred,
    60,
    150
)


# Find contours
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# Draw useful contours
analysis_image = original.copy()

detected_objects = 0

for contour in contours:

    area = cv2.contourArea(contour)

    # Ignore very small noise
    if area < 100:
        continue

    x, y, width, height = cv2.boundingRect(
        contour
    )

    # Ignore very large regions
    if area > 0.8 * image.shape[0] * image.shape[1]:
        continue

    cv2.rectangle(
        analysis_image,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

    detected_objects += 1


# Convert for display
original_rgb = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

edges_rgb = cv2.cvtColor(
    edges,
    cv2.COLOR_GRAY2RGB
)

analysis_rgb = cv2.cvtColor(
    analysis_image,
    cv2.COLOR_BGR2RGB
)


# Create visualization
fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


axes[0].imshow(
    original_rgb
)

axes[0].set_title(
    "Cropped PCB",
    fontsize=17,
    fontweight="bold"
)

axes[0].axis("off")


axes[1].imshow(
    edges_rgb
)

axes[1].set_title(
    "PCB Edges",
    fontsize=17,
    fontweight="bold"
)

axes[1].axis("off")


axes[2].imshow(
    analysis_rgb
)

axes[2].set_title(
    "Detected Structures",
    fontsize=17,
    fontweight="bold"
)

axes[2].axis("off")


fig.suptitle(
    "PCB Component & Trace Analysis — Stage 04",
    fontsize=22,
    fontweight="bold"
)

plt.tight_layout()


# Output folder
output_folder = (
    project_folder
    / "04_component_trace_analysis"
)

output_folder.mkdir(
    exist_ok=True
)


# Save result
output_path = (
    output_folder
    / "Figure_4.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Print information
print("=" * 60)
print("       PCB COMPONENT & TRACE ANALYSIS")
print("=" * 60)

print(
    f"\nDetected structures: {detected_objects}"
)

print(
    "\nResult saved to:"
)

print(
    output_path
)
