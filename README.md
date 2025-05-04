 🛠️ Metal Surface Defect Detection using YOLOv8

This project fine-tunes a YOLOv8 model to detect six types of surface defects in metals using an augmented version of the NEU Surface Defect Dataset.

 📂 Dataset

- **Original Dataset**: [NEU Surface Defect Dataset](https://github.com/rccohn/NEU-Cluster) – contains ~1500 labeled images.
- **Augmentation**: The dataset was augmented using [Roboflow](https://roboflow.com) for better variety and class balance.
- **Classes**:
  - Scratches
  - Inclusion
  - Patches
  - Pitted Surface
  - Rolled-in Scale
  - Crazing

## 📌 Project Overview

- Visualizes class distribution and labels using bounding boxes.
- Trains a YOLOv8x model (from [Ultralytics](https://github.com/ultralytics/ultralytics)) for 20 epochs.
- Evaluates performance and runs inference on sample images from the validation set.

##  How to Use

1. Clone the repository and open the notebook.

2. Install the required packages:
```bash
pip install ultralytics roboflow opencv-python matplotlib
```

3. Use Roboflow to download the dataset.
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("ai-space-olvju").project("all-metal-defects")
dataset = project.version(1).download("yolov8")
```
4. Train the model using YOLOv8.
 ```python
 from ultralytics import YOLO
model = YOLO("yolov8x.pt")
model.train(data="path/to/data.yaml", epochs=20, imgsz=640)
```

5.Run inference on validation images and visualize predictions.





