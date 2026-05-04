import os
import numpy as np
import cv2
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO

# -------------------- APP SETUP --------------------
app = Flask(__name__)

# Paths
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- LOAD MODELS --------------------
cnn_model = load_model("mura_phase1.keras")

# ⚠️ Replace with your trained model later
yolo_model = YOLO("yolov8n.pt")

IMG_SIZE = (224, 224)


# -------------------- SEVERITY FUNCTION --------------------
def get_severity(score):
    if score < 0.4:
        return "Mild"
    elif score < 0.7:
        return "Moderate"
    else:
        return "Severe"


# -------------------- YOLO + THERMAL DETECTION --------------------
def detect_fracture_area(img_path):
    img = cv2.imread(img_path)

    # Safety check
    if img is None:
        return None

    overlay = img.copy()

    results = yolo_model(img)

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Boundary safety
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Extract ROI
            roi = img[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Apply thermal heatmap
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

            # Resize safely
            heatmap = cv2.resize(heatmap, (x2 - x1, y2 - y1))

            # Blend heatmap with original
            overlay[y1:y2, x1:x2] = cv2.addWeighted(
                overlay[y1:y2, x1:x2], 0.5,
                heatmap, 0.5,
                0
            )

            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

    output_path = os.path.join(OUTPUT_FOLDER, "thermal_output.jpg")
    cv2.imwrite(output_path, overlay)

    return output_path


# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # -------------------- CNN PREDICTION --------------------
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = cnn_model.predict(img_array)[0][0]

    # -------------------- RESULT LOGIC --------------------
    if pred >= 0.3:
        label = "ABNORMAL"
        confidence = float(pred)
        severity = get_severity(pred)

        # YOLO + Thermal
        output_img = detect_fracture_area(filepath)

    else:
        label = "NORMAL"
        confidence = float(1 - pred)
        severity = None
        output_img = None

    # -------------------- RETURN RESULT --------------------
    return render_template(
        "result.html",
        label=label,
        confidence=round(confidence * 100, 2),
        image_path=filepath,
        heatmap_path=output_img,
        severity=severity
    )


# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app.run(debug=True)