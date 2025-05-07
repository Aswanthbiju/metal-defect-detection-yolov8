 🛠️ Metal Surface Defect Detection using YOLOv8

This project fine-tunes a YOLOv8 model to detect six types of surface defects in metals using an augmented version of the NEU Surface Defect Dataset.
Ideal for applications in the automation of initial stages of cell manufacturing.

 📂 Dataset

- **Original Dataset**: [NEU Surface Defect Dataset](https://github.com/rccohn/NEU-Cluster) – contains ~1500 labeled images of hot rolled steel sheets.
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

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")
val_dir = "/content/ALL-METAL-DEFECTS-1/valid/images"

results = model(f"{val_dir}/example.jpg")  # Replace with a real image filename
results[0].show()
```

---

## 🧪 Running the Notebook

All steps — from dataset loading to training and inference — can be run directly in **Google Colab**.  
Open `Metal_Defect_Detection_YOLO.ipynb` and follow the cells in order.

---

## 📚 References

- Ryan Cohn and Elizabeth Holm. *Unsupervised Machine Learning via Transfer Learning and k-Means Clustering to Classify Materials Image Data*, Integrating Materials and Manufacturing Innovation, 10(2), 2021, pp. 231–244. [DOI](https://doi.org/10.1007/s40192-021-00205-8), [arXiv](http://arxiv.org/abs/2007.08361)

- Kechen Song and Yunhui Yan. *A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects*, Applied Surface Science, 285 (2013), pp. 858–864.

---

Thank you for checking out this project!






