import cv2
import numpy as np
import os

def rebuild_image(pieces_folder, output_path):
    metadata_file = os.path.join(pieces_folder, "metadata.csv")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"metadata.csv not found in {pieces_folder}")

    # Read metadata (just filenames, since pieces are full-size already)
    with open(metadata_file, "r") as f:
        metadata = [line.strip() for line in f.readlines()]

    # Load first piece to get shape
    first_piece = cv2.imread(os.path.join(pieces_folder, metadata[0]))
    h, w, _ = first_piece.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Layer all pieces back
    for filename in metadata:
        piece_path = os.path.join(pieces_folder, filename)
        piece = cv2.imread(piece_path, cv2.IMREAD_COLOR)

        if piece is None:
            print(f"⚠Could not load {filename}, skipping.")
            continue

        # Mask for non-black pixels
        mask = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        roi = cv2.bitwise_and(piece, piece, mask=mask)
        bg = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(mask))
        canvas = cv2.add(bg, roi)

    cv2.imwrite(output_path, canvas)
    print(f"Rebuilt image saved at {output_path}")


if __name__ == "__main__":
    # CHANGE THESE PATHS:
    pieces_folder = r"C:/Users/arnav/Personal - Arnav Chhajed/BuildThis_code/pieces"
    output_path   = r"C:/Users/arnav/Personal - Arnav Chhajed/BuildThis_code/rebuilt.jpg"

    rebuild_image(pieces_folder, output_path)
