from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from pathlib import Path
import uuid
import cv2

app = Flask(__name__, template_folder="app/templates")

# Project paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "results"

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)

# Load trained CocoaSense model
print("Loading CocoaSense model...")

model = YOLO(str(MODEL_PATH))

print("CocoaSense model loaded successfully!")

# Cocoa classes
CLASS_NAMES = {
    0: "C0",
    1: "C1",
    2: "C2",
    3: "C3"
}

model.model.names = CLASS_NAMES


# Home page
@app.route("/")
def index():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded."

    image = request.files["image"]

    if image.filename == "":
        return "No image selected."

    # Create unique filename
    filename = (
        str(uuid.uuid4())
        + Path(image.filename).suffix
    )

    input_path = UPLOAD_FOLDER / filename

    image.save(input_path)

    print("Image received:", filename)

    # Run YOLO
    results = model.predict(
        source=str(input_path),
        conf=0.25,
        save=False
    )

    result = results[0]

    # Save annotated image
    result_filename = (
        Path(filename).stem
        + "_result.jpg"
    )

    result_path = RESULT_FOLDER / result_filename

    annotated_image = result.plot()

    cv2.imwrite(
        str(result_path),
        annotated_image
    )

    # Count classes
    counts = {
        "C0": 0,
        "C1": 0,
        "C2": 0,
        "C3": 0
    }

    if result.boxes is not None:

        for cls in result.boxes.cls:

            class_id = int(cls)

            class_name = CLASS_NAMES.get(class_id)

            if class_name in counts:
                counts[class_name] += 1

    total = sum(counts.values())

    print("Detection counts:", counts)
    print("Total detections:", total)

    return render_template(
        "index.html",
        result_image=result_filename,
        counts=counts,
        total=total
    )


# Serve result images
@app.route("/results/<filename>")
def results(filename):

    return send_from_directory(
        RESULT_FOLDER,
        filename
    )


# Start Flask server
if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)