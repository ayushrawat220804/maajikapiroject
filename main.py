import os
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Try to import face_recognition. On some Windows/Python versions (e.g., Python 3.13),
# installing dlib/face_recognition can be tricky. We handle missing imports gracefully.
try:
    import face_recognition  # type: ignore
    FACE_LIB_AVAILABLE = True
except Exception:  # ImportError or others
    FACE_LIB_AVAILABLE = False


# -----------------------------
# Simple AI Selfie Recognition
# -----------------------------
# This app lets you upload or capture a selfie.
# It detects faces and tries to recognize them
# using example face images stored in the `images/` folder.


IMAGES_DIR = Path("images")
TEST_DIR = Path("test")


def ensure_directories_exist() -> None:
    """Create required folders if they don't exist."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def load_known_faces(images_dir: Path):
    """Load known faces and labels from the images folder.

    Rule: The file name (without extension) is used as the person's name.
    For example: `elon_musk.jpg` -> name "elon_musk".
    """
    known_encodings = []
    known_names = []

    # Supported image files
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_files.extend(images_dir.glob(ext))

    if not FACE_LIB_AVAILABLE:
        return known_encodings, known_names

    for img_path in image_files:
        try:
            image = face_recognition.load_image_file(str(img_path))
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                # If no face was found, skip this file
                continue
            known_encodings.append(encodings[0])
            known_names.append(img_path.stem)
        except Exception:
            # If any file fails to load/encode, just skip it (keep beginner-friendly)
            continue

    return known_encodings, known_names


def draw_boxes_and_labels(image_rgb: np.ndarray, face_locations, face_names):
    """Draw rectangles and labels around detected faces on a copy of the image."""
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # Try to load a simple default font; if it fails, PIL uses a fallback
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw rectangle
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)

        # Label background box
        text = name
        if font is not None:
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
        else:
            # Basic estimate if font not available
            text_width, text_height = (7 * len(text), 12)

        draw.rectangle(
            ((left, bottom), (left + text_width + 8, bottom + text_height + 6)),
            fill=(0, 255, 0),
        )
        draw.text((left + 4, bottom + 3), text, fill=(0, 0, 0), font=font)

    return pil_image


def recognize_faces_in_image(image_rgb: np.ndarray, known_encodings, known_names, tolerance: float = 0.6):
    """Detect faces in an image and recognize them using known face encodings."""
    if not FACE_LIB_AVAILABLE:
        # face_recognition not available: return no detections
        return [], []

    # Find all faces in the image
    face_locations = face_recognition.face_locations(image_rgb)
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

    names = []
    for face_encoding in face_encodings:
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            # Choose the best match (smallest distance)
            best_index = int(np.argmin(face_distances)) if len(face_distances) else -1
            if best_index >= 0 and matches[best_index]:
                names.append(known_names[best_index])
            else:
                names.append("Unknown")
        else:
            # No known faces available
            names.append("Unknown")

    return face_locations, names


def bytes_to_rgb_image(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded bytes to an RGB numpy array suitable for face_recognition."""
    pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")
    return np.array(pil_img)


def main():
    st.set_page_config(page_title="AI Selfie Recognition", page_icon="📷")
    ensure_directories_exist()

    st.title("📷 AI Selfie Image Recognition")
    st.write(
        "Upload an image or take a selfie. The app will detect faces and show the name if recognized."
    )

    if not FACE_LIB_AVAILABLE:
        st.error(
            "face_recognition is not installed. On Windows, use Python 3.10/3.11 for easier setup, "
            "or install CMake and Visual C++ Build Tools to compile dlib."
        )

    with st.sidebar:
        st.header("Settings")
        tolerance = st.slider(
            "Recognition tolerance (lower = stricter)", min_value=0.4, max_value=0.8, value=0.6, step=0.05
        )
        st.caption(
            "Known faces come from the `images/` folder. File name is used as the label."
        )

    # Load known faces once (fast enough for a small project)
    known_encodings, known_names = load_known_faces(IMAGES_DIR)
    if known_names:
        st.success(f"Loaded {len(known_names)} known face(s): {', '.join(known_names)}")
    else:
        st.info("No known faces found. Add images into the `images/` folder (e.g., `alice.jpg`).")

    # Input options: upload or camera
    st.subheader("1) Choose an image")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"]) 
    st.markdown("**Or**")
    camera_photo = st.camera_input("Take a selfie with your webcam")

    image_rgb = None
    src_label = None

    if uploaded is not None:
        image_rgb = bytes_to_rgb_image(uploaded.read())
        src_label = uploaded.name
    elif camera_photo is not None:
        image_rgb = bytes_to_rgb_image(camera_photo.getvalue())
        src_label = "camera_photo.jpg"

    if image_rgb is not None:
        st.subheader("2) Detection & Recognition")
        with st.spinner("Processing image..."):
            locations, names = recognize_faces_in_image(
                image_rgb=image_rgb,
                known_encodings=known_encodings,
                known_names=known_names,
                tolerance=tolerance,
            )

        if not locations:
            st.warning("No faces detected. Try another image or adjust lighting.")
            st.image(image_rgb, caption="Input", use_column_width=True)
            return

        # Draw boxes and labels
        result_img = draw_boxes_and_labels(image_rgb, locations, names)
        st.image(result_img, caption="Result", use_column_width=True)

        # Show recognized names
        unique_names = sorted(set(names))
        st.write("Detected:", ", ".join(names))
        if any(n != "Unknown" for n in names):
            st.success("Recognized: " + ", ".join([n for n in unique_names if n != "Unknown"]))
        else:
            st.info("All detected faces are Unknown. Add known faces to `images/`.")

        # Optional: save the tested image into /test for records
        if st.toggle("Save this image to /test folder", value=False):
            try:
                out_path = TEST_DIR / src_label
                Image.fromarray(image_rgb).save(out_path)
                st.caption(f"Saved to {out_path.as_posix()}")
            except Exception as e:
                st.caption(f"Could not save image: {e}")


if __name__ == "__main__":
    main()


