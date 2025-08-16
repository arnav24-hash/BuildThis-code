import cv2
import numpy as np
from scipy.spatial import Voronoi
import json


def _voronoi_finite_polygons_2d(vor: Voronoi, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if -1 not in vertices:
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v != -1]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = (vor.points[p1] + vor.points[p2]) / 2
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [new_region[i] for i in np.argsort(angles)]

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def shatter_image(img: np.ndarray, num_pieces: int = 50):
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None")

    h, w = img.shape[:2]
    if num_pieces < 1:
        raise ValueError("num_pieces must be >= 1")

    rng = np.random.default_rng(42)
    points = np.column_stack((rng.integers(0, w, size=num_pieces), rng.integers(0, h, size=num_pieces)))

    vor = Voronoi(points)
    regions, vertices = _voronoi_finite_polygons_2d(vor, radius=max(w, h) * 4)

    pieces, metadata = {}, {"_canvas": {"width": int(w), "height": int(h)}}
    clip_min, clip_max = np.array([0, 0]), np.array([w - 1, h - 1])

    for idx, region in enumerate(regions):
        poly = vertices[np.asarray(region)]
        poly = np.clip(poly, clip_min, clip_max).astype(np.int32)
        if poly.shape[0] < 3 or cv2.contourArea(poly.astype(np.float32)) < 1.0:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        x, y, bw, bh = cv2.boundingRect(poly)
        if bw == 0 or bh == 0:
            continue

        roi_img = img[y:y + bh, x:x + bw]
        roi_msk = mask[y:y + bh, x:x + bw]
        piece = cv2.bitwise_and(roi_img, roi_img, mask=roi_msk)

        ok, buf_img = cv2.imencode(".png", piece)
        if not ok:
            continue

        filename = f"piece_{idx}.png"
        pieces[filename] = buf_img.tobytes()
        metadata[filename] = {"x": int(x), "y": int(y), "poly": (poly - np.array([x, y])).astype(int).tolist()}

    if not pieces:
        raise RuntimeError("Failed to generate any pieces from Voronoi shatter.")

    return pieces, metadata