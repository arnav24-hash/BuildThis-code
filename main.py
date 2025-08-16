import streamlit as st
import io
import zipfile
import cv2
import numpy as np
import json

from shatter import shatter_image
from rebuild import rebuild_image_from_zip

st.set_page_config(page_title="PuzzlePix", layout="centered")
st.title("PuzzlePix")

mode = st.radio("Choose a mode:", ["Shatter Image", "Rebuild Image"], horizontal=True)

if mode == "Shatter Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"]) 
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read uploaded image.")
        else:
            if st.button("Shatter Image", type="primary"):
                try:
                    pieces, metadata = shatter_image(img_bgr, num_pieces=50)
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
                        for filename, data in pieces.items():
                            z.writestr(filename, data)
                        z.writestr("metadata.json", json.dumps(metadata))
                    buffer.seek(0)

                    st.success(f"Image shattered successfully into {len(pieces)} pieces!")
                    st.download_button(
                        label="Download shattered pieces (ZIP)",
                        data=buffer.getvalue(),
                        file_name="pieces.zip",
                        mime="application/zip",
                    )
                except Exception as e:
                    st.error(f"Shatter failed: {e}")

elif mode == "Rebuild Image":
    uploaded_zip = st.file_uploader("Upload shattered pieces ZIP", type=["zip"]) 
    if uploaded_zip is not None:
        if st.button("Rebuild Image", type="primary"):
            try:
                rebuilt_bgr = rebuild_image_from_zip(uploaded_zip.read())
                ok, buf_img = cv2.imencode(".png", rebuilt_bgr)
                if not ok:
                    raise RuntimeError("Failed to encode rebuilt image")

                st.image(rebuilt_bgr[:, :, ::-1], caption="Rebuilt Image", use_column_width=True)
                st.download_button(
                    label="Download rebuilt image (PNG)",
                    data=buf_img.tobytes(),
                    file_name="rebuilt.png",
                    mime="image/png",
                )
            except Exception as e:
                st.error(f"Rebuild failed: {e}")
