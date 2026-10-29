# Road Damage Detection Using YOLO and Flask

## 📌 Project Overview

This project uses YOLO object detection to identify road cracks from uploaded road images. A Flask web application allows users to upload an image and visualize detected road damage with bounding boxes, class labels, and confidence scores.

## 🎯 Objectives

- Detect road cracks using deep learning and computer vision.
- Train and compare multiple YOLO model variants.
- Experiment with hyperparameter tuning to improve detection performance.
- Evaluate models using mAP, precision, recall, inference speed, and model size.
- Deploy the selected model through a Flask web application.

## 🛠️ Technologies Used

- Python
- YOLO (Ultralytics)
- Flask
- HTML and CSS
- Pillow
- OpenCV / Computer Vision

## 🗂️ Dataset

The dataset is organized into training, validation, and test sets, with YOLO-format bounding-box annotations.

The model detects two road crack classes:

- `Retak_Kulit_Kerang` — use the exact class name from your dataset.
- `Retak_Memanjang` — use the exact class name from your dataset.

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   cd road-damage-detection
   ```

2. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Place the trained model weights file named `best.pt` in the project root directory.

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

Upload a road image to view the detection results.

## 📊 Model Evaluation

Multiple YOLO variants and training configurations were compared using:

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- Inference time
- Model size

The final model should be selected based on validation performance and deployment requirements, then evaluated on the held-out test set.

## 📁 Project Structure

```text
road-damage-detection/
├── app.py
├── best.pt
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── results/
```

## 🚀 Future Improvements

- Improve detection accuracy through further experimentation.
- Test the model on diverse road conditions and lighting.
- Deploy the application to a public hosting platform.

## 👩‍💻 Author

Shada Medhat
