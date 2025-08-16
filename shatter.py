import cv2
import numpy as np
import random
import os
from scipy.spatial import Voronoi

def shatter_image(image_path, output_folder, num_pieces=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    h, w, _ = img.shape

    # Oversample seed points to ensure enough valid polygons
    extra_points = int(num_pieces * 2)  # 2x oversampling
    points = np.column_stack((
        np.random.randint(0, w, size=extra_points),
        np.random.randint(0, h, size=extra_points)
    ))

    # Compute Voronoi diagram
    vor = Voronoi(points)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    metadata = []
    piece_count = 0

    for i, region_idx in enumerate(vor.point_region):
        if piece_count >= num_pieces:  # stop after desired pieces
            break

        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue  # skip infinite/empty regions

        polygon = np.array([vor.vertices[j] for j in region], dtype=np.int32)

        # Clip polygon points inside image bounds
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        # Create mask for piece
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        # Apply mask
        piece = cv2.bitwise_and(img, img, mask=mask)

        # Crop bounding box
        x, y, w_box, h_box = cv2.boundingRect(polygon)
        if w_box == 0 or h_box == 0:
            continue
        cropped_piece = piece[y:y+h_box, x:x+w_box]

        if cropped_piece is None or cropped_piece.size == 0:
            continue

        # Save piece
        filename = f"piece_{piece_count}.png"
        cv2.imwrite(os.path.join(output_folder, filename), cropped_piece)

        # Save metadata
        metadata.append(f"{filename},{x},{y},{w_box},{h_box}\n")
        piece_count += 1

    # Save metadata
    with open(os.path.join(output_folder, "metadata.csv"), "w") as f:
        f.writelines(metadata)

    print(f"Shattered image into {piece_count} pieces in {output_folder}")


if __name__ == "__main__":
    # CHANGE THESE PATHS:
    image_path = r"C:/Users/arnav/Personal - Arnav Chhajed/BuildThis_code/image_buildthis.jpg"  # input image
    output_folder = r"C:/Users/arnav/Personal - Arnav Chhajed/BuildThis_code/pieces"    # where pieces go

    shatter_image(image_path, output_folder, num_pieces=50)
