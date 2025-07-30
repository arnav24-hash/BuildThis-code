# To run this app, install only what's necessary:
# pip install streamlit opencv-python numpy scipy torch torchvision

import streamlit as st
import cv2
import numpy as np
import os
import io
import zipfile
import json
import tempfile
from scipy.spatial import Voronoi
import torch
import torchvision.transforms as T
from torchvision.models import resnet18

# Preload AI model and transforms once for efficiency
@st.cache_resource
def load_model_and_transform():
    model = resnet18(weights="IMAGENET1K_V1")
    model.eval()
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform

model, transform = load_model_and_transform()

# Apply custom styling for cyberpunk vibe
st.markdown(
    '''
    <style>
    .main {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #fff;}
    .stButton > button {background-color: #6a00f4; color: #fff; border:none; border-radius:8px; padding:0.5em 1em;}
    .stButton > button:hover {background-color: #480ca8;}
    .stRadio, .stFileUploader, .stSelectbox, .stTextInput, .stTextArea, .stNumberInput {background-color:#1f1f2e; color:#fff; border-radius:8px; padding:0.5em;}
    </style>
    ''', unsafe_allow_html=True)

st.title("Image Shatter & Rebuild Tool")
st.write("Use geometry and AI to split an image into pieces and reconstruct it precisely.")

mode = st.radio("Select mode:", ["Shatter Image", "Rebuild Image"])

if mode == "Shatter Image":
    file = st.file_uploader("Upload image to shatter", type=["png","jpg","jpeg"])
    if file:
        # Read input
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp.write(file.read())
            path = tmp.name
        image = cv2.imread(path)
        if image is None:
            st.error("Cannot read image. Please upload a valid file.")
            st.stop()
        H, W = image.shape[:2]

        # Generate Voronoi regions
        pts = np.random.rand(50,2) * [W, H]
        vor = Voronoi(pts)

        # Compute finite Voronoi polygons
        def finite_polygons_2d(vor):
            center = vor.points.mean(axis=0)
            radius = np.ptp(vor.points, axis=0).max() * 2
            all_ridges = {}
            for (p1,p2),(v1,v2) in zip(vor.ridge_points, vor.ridge_vertices):
                all_ridges.setdefault(p1, []).append((p2,v1,v2))
                all_ridges.setdefault(p2, []).append((p1,v1,v2))
            regions, vertices = [], vor.vertices.tolist()
            for p1, reg_idx in enumerate(vor.point_region):
                reg = vor.regions[reg_idx]
                if all(v>=0 for v in reg):
                    regions.append(reg)
                    continue
                # Reconstruct infinite region
                new_reg = [v for v in reg if v>=0]
                for p2, v1, v2 in all_ridges[p1]:
                    if v2 < 0:
                        v1, v2 = v2, v1
                    if v1 < 0:
                        tangent = vor.points[p2] - vor.points[p1]
                        tangent /= np.linalg.norm(tangent)
                        normal = np.array([-tangent[1], tangent[0]])
                        midpoint = vor.points[[p1,p2]].mean(axis=0)
                        direction = normal * np.sign(np.dot(midpoint-center, normal))
                        far_point = vor.vertices[v2] + direction * radius
                        vertices.append(far_point.tolist())
                        new_reg.append(len(vertices)-1)
                regions.append(new_reg)
            return regions, np.array(vertices)

        regions, vertices = finite_polygons_2d(vor)

        # Assemble ZIP with pieces and metadata
        buffer = io.BytesIO()
        metadata = {}
        with zipfile.ZipFile(buffer, 'w') as z:
            for i, reg in enumerate(regions):
                poly = np.array([vertices[v] for v in reg], dtype=np.int32)
                mask = np.zeros((H,W), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                x,y,w,h = cv2.boundingRect(poly)
                if w==0 or h==0: continue
                piece = cv2.bitwise_and(image, image, mask=mask)[y:y+h, x:x+w]
                if piece.size == 0: continue
                # Clip polygon to bbox
                rel_poly = np.clip(poly - [x,y], [0,0], [w-1,h-1]).tolist()
                metadata[f"piece_{i:03d}.png"] = {'x':int(x), 'y':int(y), 'poly':rel_poly}
                ok, buf_img = cv2.imencode('.png', piece)
                if ok:
                    z.writestr(f"piece_{i:03d}.png", buf_img.tobytes())
            z.writestr('metadata.json', json.dumps(metadata))

        st.success("Image shattered successfully!")
        st.download_button("Download pieces.zip", buffer.getvalue(), "pieces.zip", "application/zip")

else:
    zip_file = st.file_uploader("Upload pieces ZIP to rebuild", type=['zip'])
    if zip_file:
        # Extract
        with zipfile.ZipFile(zip_file) as z:
            folder = tempfile.mkdtemp()
            z.extractall(folder)
        meta_path = os.path.join(folder, 'metadata.json')
        if not os.path.exists(meta_path):
            st.error("metadata.json missing in ZIP.")
            st.stop()
        data = json.load(open(meta_path))

        # Compute canvas size
        pieces = []
        max_x = max_y = 0
        for name, info in data.items():
            img_path = os.path.join(folder, name)
            if not os.path.isfile(img_path): continue
            img = cv2.imread(img_path)
            if img is None: continue
            x, y = info['x'], info['y']
            h_img, w_img = img.shape[:2]
            max_x = max(max_x, x + w_img)
            max_y = max(max_y, y + h_img)
            pieces.append((x, y, img, info['poly']))
        if not pieces:
            st.error("No valid pieces found.")
            st.stop()

        # Initialize canvas
        canvas = np.zeros((max_y, max_x, 3), dtype=np.uint8)

        # Place each piece precisely
        errors = []
        for x, y, img, poly in pieces:
            h_img, w_img = img.shape[:2]
            # Create mask of piece
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            p = np.array(poly, dtype=np.int32)
            p[:,0] = np.clip(p[:,0], 0, w_img-1)
            p[:,1] = np.clip(p[:,1], 0, h_img-1)
            cv2.fillPoly(mask, [p], 255)
            # Determine overlapping region
            x0, y0 = max(x,0), max(y,0)
            x1, y1 = min(x+w_img, max_x), min(y+h_img, max_y)
            if x1<=x0 or y1<=y0:
                errors.append(f"Piece at ({x},{y}) out of bounds.")
                continue
            # Corresponding piece coordinates
            px0, py0 = x0 - x, y0 - y
            px1, py1 = px0 + (x1-x0), py0 + (y1-y0)
            roi_canvas = canvas[y0:y1, x0:x1]
            roi_piece = img[py0:py1, px0:px1]
            roi_mask = mask[py0:py1, px0:px1]
            if roi_canvas.shape[:2] != roi_piece.shape[:2]:
                errors.append(f"Shape mismatch for piece at ({x},{y}).")
                continue
            # Copy pixels
            np.copyto(roi_canvas, roi_piece, where=roi_mask[:,:,None].astype(bool))

        if errors:
            for e in errors:
                st.warning(e)
        st.success("Reconstruction complete!")
        st.image(canvas[:,:,::-1], use_column_width=True)
