# FractureAI

A Flask-based bone fracture detection web app that combines a Keras CNN classifier with YOLOv8-based region detection and a thermal-style overlay for detected fracture areas.

## Features

- Upload an X-ray image via a web interface
- Classify the image as `NORMAL` or `ABNORMAL` using `mura_phase1.keras`
- For abnormal results, highlight suspected fracture regions with YOLOv8 and a thermal overlay
- Display confidence and severity level on the result page

## Project Structure

- `app.py` - Main Flask application
- `mura_phase1.keras` - Trained Keras model for fracture classification
- `yolov8n.pt` - YOLOv8 model weights used for region detection
- `templates/` - HTML templates for the upload and result pages
- `static/uploads/` - Saved uploaded images
- `static/output/` - Generated output images with thermal overlays

## Requirements

- Python 3.8+ recommended
- `Flask`
- `tensorflow`
- `opencv-python`
- `numpy`
- `ultralytics`

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install the required packages:

```bash
pip install flask tensorflow opencv-python numpy ultralytics
```

## Running the App

From the project root directory:

```bash
python app.py
```

Then open your browser at `http://127.0.0.1:5000/`.

## Usage

1. Open the home page.
2. Upload a bone X-ray image.
3. Submit the form.
4. The app will classify the image and show the result.
5. If the image is abnormal, the app will also display the YOLO-detected fracture region with a thermal overlay.

## Notes

- The current classifier threshold is set at `0.3` in `app.py`.
- The YOLO overlay image is saved as `static/output/thermal_output.jpg`.
- Replace `mura_phase1.keras` and `yolov8n.pt` with your own trained models if needed.

## Troubleshooting

- If the app fails to load the models, verify the model files are present in the project root.
- If the upload does not work, ensure `static/uploads/` is writable and created automatically by `app.py`.
- For YOLO issues, verify the `ultralytics` package and `yolov8n.pt` weight file are installed correctly.
