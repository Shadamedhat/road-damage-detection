from flask import Flask, render_template, request
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import uuid
import os

app = Flask(__name__)

# =========================
# Project paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
RESULTS_DIR = BASE_DIR / "static" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# =========================
# Load fine-tuned YOLO model
# =========================
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))


# =========================
# Home page and prediction
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    detections = []
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None or not uploaded_file.filename:
            error = "Please upload an image."

        elif Path(uploaded_file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            error = "Upload a JPG, JPEG, PNG, or WEBP image."

        else:
            try:
                # Read uploaded image
                image = Image.open(uploaded_file.stream).convert("RGB")

                # Run YOLO object detection
                results = model.predict(
                    source=image,
                    conf=0.25,
                    imgsz=640,
                    verbose=False
                )

                result = results[0]

                # Save image with bounding boxes
                output_filename = f"{uuid.uuid4().hex}.jpg"
                output_path = RESULTS_DIR / output_filename

                result.save(filename=str(output_path))

                result_image = f"results/{output_filename}"

                # Extract detected classes and confidence scores
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    x1, y1, x2, y2 = [
                        round(float(value), 2)
                        for value in box.xyxy[0].tolist()
                    ]

                    detections.append({
                        "class_name": model.names[class_id],
                        "confidence": round(confidence * 100, 2),
                        "bbox": f"({x1}, {y1}, {x2}, {y2})"
                    })

            except (UnidentifiedImageError, OSError, ValueError):
                error = "Invalid image. Please upload a valid image file."

            except Exception:
                app.logger.exception("Prediction error")
                error = "Prediction failed. Please try another image."

    return render_template(
        "index.html",
        result_image=result_image,
        detections=detections,
        error=error
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
