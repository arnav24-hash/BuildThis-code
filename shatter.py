import cv2
import numpy as np
import random
import os
from scipy.spatial import Voronoi, voronoi_plot_2d

def shatter_image(image_path, output_folder, num_pieces=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Load image
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # Generate random seed points inside image
    points = np.column_stack((
        np.random.randint(0, w, size=num_pieces),
        np.random.randint(0, h, size=num_pieces)
    ))

    # Compute Voronoi diagram
    vor = Voronoi(points)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue  # Skip open regions

        polygon = [vor.vertices[j] for j in region]
        polygon = np.array(polygon, dtype=np.int32)

        # Create mask for piece
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        # Apply mask to image
        piece = cv2.bitwise_and(img, img, mask=mask)

        # Crop bounding box for piece
        x, y, w_box, h_box = cv2.boundingRect(polygon)
        cropped_piece = piece[y:y+h_box, x:x+w_box]

        # Save piece with index (used for reconstruction)
        cv2.imwrite(os.path.join(output_folder, f"piece_{i}.png"), cropped_piece)

    print(f"Shattered image into {num_pieces} pieces in {output_folder}")
