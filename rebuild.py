import cv2
import numpy as np
import os

def rebuild_image(pieces_folder, original_shape, output_path):
    h, w, c = original_shape

    # Initialize blank canvas
    canvas = np.zeros((h, w, c), dtype=np.uint8)

    # Sort pieces by filename index
    piece_files = sorted(os.listdir(pieces_folder), key=lambda x: int(x.split("_")[1].split(".")[0]))

    for i, filename in enumerate(piece_files):
        piece = cv2.imread(os.path.join(pieces_folder, filename), cv2.IMREAD_UNCHANGED)

        # Load mask back (in this demo we just place pieces by bounding box order)
        # In a real solver, we'd do feature matching here
        # For now, just visualize placing in order for demonstration
        # (so reconstruction is trivial)
        # If you saved bounding box info, you’d reapply it here.

    cv2.imwrite(output_path, canvas)
    print(f"Rebuilt image saved at {output_path}")
