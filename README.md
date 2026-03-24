# CMPE 401 - YOLO11 Object Detection on VisDrone

## Part II – Loss Curve and Fitting Analysis

This section analyzes the training and validation behavior of the YOLO11n baseline model trained on the VisDrone dataset for 50 epochs. The analysis is based on the loss plots and performance curves generated during training (`results/baseline/results.png`, `results.csv`, and associated metric plots).

---

### 1. Training vs Validation Loss

Across all 50 epochs, the training losses (`box_loss`, `cls_loss`, `dfl_loss`) decrease smoothly and consistently. This indicates:

- Stable training dynamics  
- No exploding or vanishing gradients  
- Properly functioning data loader and augmentations  
- Correct model configuration and learning rate  

The validation losses (`val/box_loss`, `val/cls_loss`, `val/dfl_loss`) also show a gradual downward trend. This confirms that the model is generalizing to unseen validation data and learning useful features from VisDrone’s complex scenes.

---

### 2. Convergence Behavior

Both training and validation losses demonstrate clear convergence:

- Rapid reduction in losses during the first ~10–15 epochs  
- Gradual improvements from epochs 15–40  
- Loss curves flatten around epochs 40–50, indicating convergence  
- mAP50 and mAP50-95 steadily improve and plateau toward the end  

This indicates that the model has reached its optimal point under current hyperparameters.

---

### 3. Overfitting Assessment

There is *no significant overfitting* visible in the curves:

- Validation losses **do not increase** at any point  
- Training and validation losses trend downward together  
- Validation mAP continues improving across most epochs  

If overfitting were present, we would expect:

- Training loss decreases  
- Validation loss increases 
- Decreasing validation mAP  

None of these symptoms appear.

---

### 4. Underfitting Assessment

The model shows **mild underfitting**, which is expected for YOLO11n on VisDrone:

- Final recall is low (~0.41)  
- Final mAP50-95 is moderate (~0.316)  
- Validation loss continues to decrease even at epoch 50  
- Some predictions in dense scenes are missed  

This underfitting is not due to incorrect training setup — it is due to **model capacity limitations**.

---

### 5. Causes: Dataset Size & Model Capacity

#### **Dataset Factors (VisDrone 2019 DET)**
- 6,471 training images  
- 548 validation images  
- Large number of tiny, distant, or occluded objects  
- Highly dense scenes (20–80 objects per frame)  
- Multiple visually similar classes  

These characteristics make detection difficult, especially for small models.

#### **Model Capacity Factors (YOLO11n)**
- YOLO11n is the smallest and fastest model in the YOLO11 family  
- Limited representational capacity  
- Struggles with fine-grained and small-object detection  
- Cannot capture complex spatial relationships present in VisDrone  

As a result:

- **Precision is high (~0.89)** the model is accurate when it detects something  
- **Recall is low (~0.41)** it misses many small or occluded objects  

This leads to mild underfitting but *no overfitting*.

---

### 6. Summary

> The YOLO11n baseline converges cleanly, with decreasing training and validation losses and steadily improving mAP. There is no indication of overfitting. Mild underfitting is observed due to the limited capacity of YOLO11n relative to the complexity of the VisDrone dataset, which contains dense scenes and many small objects. This baseline establishes a strong foundation for further experiments using larger models (YOLO11s, YOLO11m), higher-resolution inputs, or enhanced augmentation strategies.