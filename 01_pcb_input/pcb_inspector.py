import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------
# Project paths
# ------------------------------------------

project_folder = Path(__file__).resolve().parent

image_path = project_folder / "pcb.jpg"


# ------------------------------------------
# Load PCB image
# ------------------------------------------

image = cv2.imread(
    str(image_path)
)


# Check whether image was loaded
if image is None:

    print("❌ PCB image could not be loaded.")

    print(
        "Make sure pcb.jpg is inside the "
        "01_pcb_input folder."
    )

    raise SystemExit


# ------------------------------------------
# Get image information
# ------------------------------------------

height, width, channels = image.shape

file_size = image_path.stat().st_size / 1024


print("=" * 60)
print("        SMART PCB INSPECTION SYSTEM")
print("=" * 60)

print("\nPCB Image Information")
print("-" * 40)

print(
    f"Width       : {width} pixels"
)

print(
    f"Height      : {height} pixels"
)

print(
    f"Channels    : {channels}"
)

print(
    f"File Size   : {file_size:.2f} KB"
)


# ------------------------------------------
# Convert BGR → RGB
# ------------------------------------------

rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ------------------------------------------
# Convert to grayscale
# ------------------------------------------

gray_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ------------------------------------------
# Display PCB
# ------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 7)
)


# Original image

axes[0].imshow(
    rgb_image
)

axes[0].set_title(
    "Original PCB",
    fontsize=18,
    fontweight="bold"
)

axes[0].axis("off")


# Grayscale image

axes[1].imshow(
    gray_image,
    cmap="gray"
)

axes[1].set_title(
    "PCB Grayscale Analysis",
    fontsize=18,
    fontweight="bold"
)

axes[1].axis("off")


fig.suptitle(
    "PCB Image Inspection — Stage 01",
    fontsize=22,
    fontweight="bold"
)

plt.tight_layout()

# Save result
output_path = (
    project_folder
    / "Figure_1.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n✅ PCB inspection completed.")

print(
    "Result saved as:",
    output_path
)
