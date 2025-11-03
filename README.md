AI Selfie Image Recognition (Beginner-Friendly)
==============================================

A simple local project for recognizing faces in selfies using Python, Streamlit, OpenCV, and face_recognition.

Project Structure
-----------------

```
images/        # put known faces here (file name becomes the label)
test/          # optional: saved test selfies
main.py        # Streamlit app
requirements.txt
README.md
```

Features
--------
- Upload or capture a selfie from your webcam.
- Detect faces and recognize them if a matching known face is found.
- Draw a green box and label around detected faces.
- Beginner-friendly code with simple comments.
- Runs locally with easy setup.

Setup (Windows-friendly)
------------------------

1) Create and activate a virtual environment (recommended)

```powershell
cd C:\Users\USER\aiproject
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- The `face_recognition` library relies on `dlib`. On Windows, the included `dlib-bin` helps install prebuilt wheels.
- Recommended Python version: **3.10 or 3.11**. Python 3.13 may try to compile `dlib` from source, which requires CMake and Visual C++ Build Tools. If you are on 3.13 and see build errors, install Python 3.11 and recreate the venv.

Add Known Faces
---------------

Place clear, front-facing photos in the `images/` folder. The file name (without extension) becomes the shown name.

Examples:
- `images/alice.jpg`  â†’ label: `alice`
- `images/bob.png`    â†’ label: `bob`

Tip: Use one face per image for best results.

Run the App
-----------

```powershell
streamlit run main.py
```

Then open the displayed local URL in your browser (usually `http://localhost:8501`).

Usage
-----
1. In the sidebar, adjust the recognition tolerance if needed (lower = stricter).
2. Upload an image or use the camera to take a selfie.
3. The app detects faces and shows names if recognized, otherwise "Unknown".
4. Optionally save the current image to the `test/` folder using the toggle.

Troubleshooting
---------------
- If no faces are detected, try a brighter environment and look at the camera.
- If everyone is "Unknown", add known faces to the `images/` folder and reload.
- If `dlib` installation fails, ensure you are on a recent Python version, upgrade `pip`, and try again. The `dlib-bin` dependency should provide a prebuilt wheel on Windows.

License
-------
For educational use in a BCA mini project.


