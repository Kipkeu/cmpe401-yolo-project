# Baseline Results (YOLO11n)

This folder contains all artifacts generated from the YOLO11n baseline training run.

## Included Files
- `results.png` – training/validation loss curves  
- `results.csv` – per-epoch metrics (mAP, precision, recall)  
- `confusion_matrix.png` – validation confusion matrix  
- `PR_curve.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`  
- `val_batch*_pred.jpg` – sample validation predictions  
- `metrics.json` – final summarized metrics  
- `args.yaml`, `hyp.yaml` – configuration used for training  
- `MODEL_WEIGHTS_LINKS.txt` – paths to best.pt and last.pt

## Reproduce Baseline Training
Training was run on Google Colab using:

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(
    data="/content/drive/MyDrive/visdrone_dataset/visdrone.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
