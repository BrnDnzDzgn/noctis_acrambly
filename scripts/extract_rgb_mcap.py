import os
import argparse
from PIL import Image
from io import BytesIO
from mcap_ros2.reader import read_ros2_messages


def center_crop(image, crop_w, crop_h):
    """Crop a centered crop_w x crop_h region, cutting equal margins from
    every side. Keeps the pixels at full resolution (no shrinking), so
    centered objects stay their original pixel size."""
    W, H = image.size
    # clamp crop size so we never ask for more than the image has
    crop_w = min(crop_w, W)
    crop_h = min(crop_h, H)
    left = (W - crop_w) // 2
    top = (H - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    return image.crop((left, top, right, bottom))


def extract_rgb(bag_path, output_dir, crop_w=None, crop_h=None):
    os.makedirs(output_dir, exist_ok=True)
    topic = "/camera/color/image_raw/compressed"
    count = 0
    for msg in read_ros2_messages(bag_path, topics=[topic]):
        ros_msg = msg.ros_msg
        # CompressedImage.data contains JPEG/PNG bytes.
        image = Image.open(BytesIO(bytes(ros_msg.data))).convert("RGB")

        # Optional centered crop (equal margins all sides), full resolution.
        if crop_w is not None and crop_h is not None:
            image = center_crop(image, crop_w, crop_h)

        output_path = os.path.join(
            output_dir,
            f"{count:06d}.png"
        )
        image.save(output_path)
        if count % 100 == 0:
            print(
                f"Saved {count}: "
                f"{image.width}x{image.height} -> {output_path}"
            )
        count += 1
    print(f"\nDone. Extracted {count} RGB frames.")
    if crop_w is not None and crop_h is not None:
        print(f"Center-cropped each frame to {crop_w}x{crop_h}.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to MCAP file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for extracted RGB images"
    )
    parser.add_argument(
        "--crop-w",
        type=int,
        default=None,
        help="Width of centered crop (keep objects full-res). "
             "Omit for no crop (full frame)."
    )
    parser.add_argument(
        "--crop-h",
        type=int,
        default=None,
        help="Height of centered crop. Omit for no crop (full frame)."
    )
    args = parser.parse_args()
    extract_rgb(args.bag, args.output, crop_w=args.crop_w, crop_h=args.crop_h)