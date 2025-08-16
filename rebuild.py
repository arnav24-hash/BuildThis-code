import cv2
import numpy as np
import zipfile
import io
import json


def rebuild_image_from_zip(zip_bytes: bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        if "metadata.json" not in z.namelist():
            raise FileNotFoundError("metadata.json not found in uploaded ZIP")

        metadata = json.loads(z.read("metadata.json").decode("utf-8"))
        canvas_w = int(metadata.get("_canvas", {}).get("width", 0))
        canvas_h = int(metadata.get("_canvas", {}).get("height", 0))

        max_x = max_y = 0
        pieces_info = []
        for filename, info in metadata.items():
            if filename.startswith("_") or not isinstance(info, dict) or filename not in z.namelist():
                continue
            arr = np.frombuffer(z.read(filename), np.uint8)
            piece = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if piece is None:
                continue
            x, y = int(info.get("x", 0)), int(info.get("y", 0))
            h_piece, w_piece = piece.shape[:2]
            max_x, max_y = max(max_x, x + w_piece), max(max_y, y + h_piece)
            pieces_info.append((x, y, piece, np.asarray(info.get("poly", []), dtype=np.int32)))

        if not canvas_w or not canvas_h:
            canvas_w, canvas_h = max_x, max_y

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for x, y, piece, poly in pieces_info:
            if poly.size == 0:
                continue
            h_piece, w_piece = piece.shape[:2]
            mask = np.zeros((h_piece, w_piece), dtype=np.uint8)
            poly[:, 0] = np.clip(poly[:, 0], 0, w_piece - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, h_piece - 1)
            cv2.fillPoly(mask, [poly], 255)

            x1, y1 = min(canvas_w, x + w_piece), min(canvas_h, y + h_piece)
            roi_canvas = canvas[y:y1, x:x1]
            roi_piece = piece[: y1 - y, : x1 - x]
            roi_mask = mask[: y1 - y, : x1 - x]
            np.copyto(roi_canvas, roi_piece, where=roi_mask[:, :, None].astype(bool))

        return canvas