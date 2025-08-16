import cv2
import numpy as np
import os
from scipy.spatial import Voronoi

def shatter_image(image_path, output_folder, num_pieces=50, seed=42):
    np.random.seed(seed)

    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    h, w = img.shape[:2]

    # Generate random seed points
    points = np.column_stack((
        np.random.randint(0, w, size=num_pieces),
        np.random.randint(0, h, size=num_pieces)
    ))

    # Voronoi diagram
    vor = Voronoi(points)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    piece_count = 0
    for region_idx in vor.point_region:
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue  # skip infinite regions

        polygon = np.array([vor.vertices[i] for i in region], dtype=np.int32)

        # Clip polygon points inside image bounds
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        # Full-size transparent mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        # Add alpha channel if missing
        if img.shape[2] == 3:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        else:
            img_rgba = img.copy()

        # Apply mask to alpha channel
        piece = img_rgba.copy()
        piece[:, :, 3] = mask  # alpha = mask

        filename = f"piece_{piece_count}.png"
        cv2.imwrite(os.path.join(output_folder, filename), piece)
        piece_count += 1

        if piece_count >= num_pieces:
            break

    print(f"Shattered into {piece_count} full-size transparent pieces at {output_folder}")


if __name__ == "__main__":
    # CHANGE THESE PATHS:
    image_path = r"C:/Users/arnav/Personal - Arnav Chhajed/BuildThis_code/image_buildthis.jpg"  # input image
    output_folder = r"C:/Users/arnav/Personal - Arnav Chhajed/BuildThis_code/pieces"    # where pieces go

    shatter_image(image_path, output_folder, num_pieces=50)


