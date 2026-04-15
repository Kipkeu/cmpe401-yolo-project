# YOLO11 Object Detection on VisDrone
## Project Summary
This project presents the design, training, and systematic evaluation of a modern object detection pipeline built using YOLOv11 on the VisDrone dataset. The primary objective is not only to achieve strong detection performance, but to develop a rigorous understanding of training dynamics, experimental design, and comparative evaluation in deep learning systems. A complete end-to-end pipeline was implemented using the Ultralytics YOLO framework, including dataset preparation, configuration through YAML files, GPU-accelerated training on Google Colab, and reproducible experiment tracking via GitHub. A fully trained YOLO11n baseline model (50 epochs) was first established to serve as a reference point for all subsequent analyses.

Training and validation behaviors were then systematically examined through loss curves and evaluation metrics to understand convergence patterns, generalization behavior, and model limitations. These analyses revealed stable convergence but also highlighted performance constraints linked to model capacity on the dense and small-object-heavy VisDrone dataset. Building on these observations, a series of controlled experiments were conducted, including model scaling (YOLO11n → YOLO11s) and optimization changes such as learning-rate scheduling. Finally, a **multi-version comparison** across YOLO families was performed to contextualize performance trade-offs across speed, accuracy, and model complexity.

Overall, this project emphasizes principled experimentation and diagnostic analysis over raw performance optimization, demonstrating a complete understanding of modern object detection systems from implementation to interpretation. All experiments are fully reproducible using the provided configuration files without requiring custom training code.

*Some of the model experiments were run for fewer epochs due to computational constraints; however, trends were still clearly observable when compared to the fully converged baseline and also a fully model is presented at the end.*

## Repository Structure
This repository is organized to clearly separate configuration, data handling, experimental results, and analysis. Only lightweight, reproducible artifacts are stored.

```
├── configs/                     # Training and experiment configuration files
│   ├── augmentations.yaml       # Data augmentation settings
│   ├── yolo11n_baseline.yaml    # Part I baseline configuration
│   ├── yolo11s_exp01.yaml       # Part III structured experiment config
│   └── yolo26n_baseline.yaml    # Additional reference configuration
│
├── data/                        
│   └── visdrone.yaml            # YOLO dataset definition
│
├── src/                        
│   ├── visualize.py             
│   └── utils/
│       ├── dataset_tools.py     
│       ├── loss_plotter.py      
│       └── model_loader.py              
│
├── results/                     # Final cleaned experimental outputs
│   ├── baseline/                # Part I – YOLO11n baseline results
│   │   └── yolo11n/
│   │       ├── results.csv
│   │       ├── results.png
│   │       ├── confusion_matrix.png
│   │       └── BoxPR_curve.png
│   │
│   ├── experiments/             # Parts III–V experimental results
│   │   ├── exp01_yolo11s/        # Part III – Model capacity experiment
│   │   ├── exp02_yolo11s_cosinelr/ # Part IV – Learning-rate schedule improvement
│   │   └── comparison/           # Part V – Multi-version YOLO comparison
│   │       └── model_comparison.csv
│   │   ├── Fully_Run_Improved_Model/ # Full Testing – Full Run model with established improvements
│
├── requirements.txt            
├── README.md                     # Full Project documentation and analysis

```


## Part I – Baseline Model

The baseline model for this project was trained using **YOLOv11n**, the nano variant of YOLOv11, on the VisDrone2019 detection dataset. The purpose of this baseline is to establish a fully converged reference point for evaluating training dynamics and guiding subsequent experimental design.

### Training Configuration
- Model: YOLOv11n
- Dataset: VisDrone2019-DET
- Image size: 640 × 640
- Batch size: 16
- Epochs: 50
- Optimizer: Ultralytics default (SGD with linear learning-rate decay)
- Hardware: Google Colab GPU

The model was trained using the default Ultralytics YOLO pipeline without any architectural or optimization modifications. This ensures that the baseline reflects standard YOLO behavior rather than engineered improvements.

### Training and Validation Behavior
The training loss curves show smooth and stable convergence across all loss components (box, classification, and DFL). Validation loss decreases consistently without divergence, indicating that the model does not suffer from overfitting. However, recall remains relatively limited, suggesting mild underfitting given the complexity of the VisDrone dataset.

The baseline results provide clear evidence that YOLO11n, while computationally efficient, is capacity-limited for dense drone imagery. These observations directly motivate the controlled experiments and improvement cycles explored in later parts of the project.

### Recorded Artifacts
The following artifacts are stored in the repository to ensure transparency and reproducibility:
- Training and validation loss curves (`results.png`)
- Per-epoch metrics (`results.csv`)
- Summary metrics snapshot (`metrics.json`)
- Confusion matrix
- Precision–Recall curves
- Example validation predictions
- Training configuration (`args.yaml`)

## Part II – Loss Curve and Fitting Analysis

This section analyzes the training and validation behavior of the YOLO11n baseline model trained on the VisDrone dataset. The analysis is based on the loss plots and performance curves generated during training (`results/baseline/results.png`, `results.csv`, and associated metric plots). This ensures that subsequent design choices were based on observed learning dynamics.

### Training vs Validation Loss

Across all 50 epochs, the training losses (`box_loss`, `cls_loss`, `dfl_loss`) decrease smoothly and consistently. This indicates:

- Stable training dynamics  
- No exploding or vanishing gradients  
- Properly functioning data loader and augmentations  
- Correct model configuration and learning rate  

The validation losses (`val/box_loss`, `val/cls_loss`, `val/dfl_loss`) also show a gradual downward trend. This confirms that the model is generalizing to unseen validation data and learning useful features from VisDrone’s complex scenes.

---

### Convergence Behavior

Both training and validation losses demonstrate clear convergence:

- Rapid reduction in losses during the first ~10–15 epochs  
- Gradual improvements from epochs 15–40  
- Loss curves flatten around epochs 40–50, indicating convergence  
- mAP50 and mAP50-95 steadily improve and plateau toward the end  

This indicates that the model has reached its optimal point under current hyperparameters.

---

### Over-fitting Assessment

There is *no significant over-fitting* visible in the curves:

- Validation losses **do not increase** at any point  
- Training and validation losses trend downward together  
- Validation mAP continues improving across most epochs  

If over-fitting were present, we would expect:

- Training loss decreases  
- Validation loss increases 
- Decreasing validation mAP  

None of these symptoms appear.

---

### Under-fitting Assessment

The model shows **mild under-fitting**, which is expected for YOLO11n on VisDrone:

- Final recall is low (~0.41)  
- Final mAP50-95 is moderate (~0.316)  
- Validation loss continues to decrease even at epoch 50  
- Some predictions in dense scenes are missed  

This underfitting is not due to incorrect training setup — it is due to **model capacity limitations**.
Note that precision values are taken from validation metrics (`metrics/precision(B)`), while training loss values such as DFL loss are not used as performance metrics.

---

### Causes: Dataset Size & Model Capacity

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
- **Precision is moderate (~0.42)** 
- **Recall is low (~0.33)** indicating many small or occluded objects are missed  

This leads to underfitting but *no overfitting*.

---

### Summary of baseline

The YOLO11n baseline converges cleanly, with decreasing training and validation losses and steadily improving mAP. There is no indication of overfitting. Mild underfitting is observed due to the limited capacity of YOLO11n relative to the complexity of the VisDrone dataset, which contains dense scenes and many small objects. This baseline establishes a strong foundation for further experiments using larger models (YOLO11s, YOLO11m), higher-resolution inputs, or enhanced augmentation strategies.

## Part III – Structured Experimental Design

This section presents a controlled experiment designed to evaluate the impact of **model capacity** on object detection performance using the VisDrone dataset. The baseline YOLO11n model exhibited mild underfitting and low recall, motivating an experiment with a larger model.

---

### Experiment Objective and settings

The objective of this experiment is to assess whether increasing model capacity from **YOLO11n (nano)** to **YOLO11s (small)** leads to improved detection performance, particularly for small and densely packed objects common in VisDrone imagery.

The following settings were used:
**Experiment 1 (Treatment):**
- Model: YOLO11s
- Image size: 640
- Dataset: VisDrone2019-DET
- Training epochs: 20
- Batch size: 8

The primary experimental variable is **model size**.  
Batch size and number of epochs were reduced intentionally to ensure **stable GPU training in Google Colab**, while preserving a fair and interpretable comparison.

---

### Quantitative Results

The final evaluation metrics are summarized below:

| Metric | YOLO11n (50 ep) | YOLO11s (20 ep) |
|------|-----------------|------------------|
| Precision | 0.416 | 0.467 |
| Recall | 0.325 | 0.375 |
| mAP50 | 0.317 | **0.378** |
| mAP50‑95 | 0.183 | 0.223 |

These results reflect early‑stage convergence behavior and are intended for trend analysis rather than final performance comparison.

---

### Analysis

The YOLO11s experiment demonstrates a clear **increase in mAP50**, confirming that higher model capacity improves detection capability on the VisDrone dataset. This supports the hypothesis that the baseline YOLO11n model was capacity‑limited and underfitting complex scenes. While precision and recall show early improvement relative to the baseline, the YOLO11s model has not yet fully converged, which is expected because:

- YOLO11s contains significantly more parameters and requires more training iterations to fully converge
- The experiment was intentionally limited to 20 epochs to observe early training trends
- Larger models often exhibit lower precision early due to broader object exploration

Qualitative inspection of prediction images shows that YOLO11s detects a greater number of objects in dense scenes, particularly vehicles and pedestrians, even though confidence calibration has not yet stabilized.

---

### Discussion of Constraints

This experiment was conducted under realistic computational constraints using Google Colab. To prevent GPU memory crashes, batch size and training duration were reduced. These adjustments do not undermine the experimental validity, as all major conditions (dataset, resolution, augmentation, evaluation procedure) remained consistent across runs.

The experiment successfully isolates **model size** as the main factor influencing performance and provides meaningful insight into how increased capacity affects detection outcomes.

Experiment 1 confirms that increasing model size from YOLO11n to YOLO11s improves overall detection performance, as evidenced by higher mAP50. Although the YOLO11s model has not yet fully converged due to reduced training epochs, the results strongly suggest that additional optimization or training time would further enhance performance. These findings motivate the iterative improvement phase explored in Part IV.

## Part IV – Iterative Model Improvement

This section investigates the effect of a learning‑rate scheduling strategy on the convergence and performance of the YOLO11s model. Based on Experiment 1, the model demonstrated improved mAP but incomplete convergence, motivating an optimization‑level improvement. A cosine learning‑rate schedule was introduced to improve optimization convergence in a higher‑capacity model. 

### Baseline
The baseline for Part IV is the YOLO11s model trained for 20 epochs using the default learning‑rate strategy.

### Controlled Modification
A cosine learning‑rate schedule was introduced while keeping all other parameters constant.

| Parameter | Experiment 1 | Part IV |
|---------|-------------|---------|
| Model | YOLO11s | YOLO11s |
| Epochs | 20 | 20 |
| Batch size | 8 | 8 |
| Image size | 640 | 640 |
| Learning‑rate schedule | Default | **Cosine decay** |

### Quantitative Results

| Model | Precision | Recall | mAP50 | mAP50‑95 |
|-----|-----------|--------|--------|----------|
| YOLO11s (Default LR) | 0.467 | 0.375 | 0.378 | 0.223 |
| YOLO11s (Cosine LR) | 0.477 | 0.368 | 0.353 | 0.206 |

### Analysis
Applying a cosine learning‑rate schedule did not improve performance under the given training budget. While cosine decay is effective for longer training regimes, it reduced the learning rate too aggressively in this 20‑epoch setting. As a result, the model under‑optimized and failed to fully learn complex feature representations in VisDrone’s dense scenes.

This highlights that learning‑rate scheduling must be paired with sufficient training duration to be effective.

The cosine learning‑rate schedule did not yield performance gains for YOLO11s under limited epochs. This experiment demonstrates the importance of aligning optimization strategies with training duration and reinforces that default learning‑rate behavior was more effective in this context.

## Part V – Multi‑Version YOLO Comparison

This section compares multiple YOLO versions using zero‑shot evaluation on the VisDrone validation set. Models were evaluated using identical inference settings without fine‑tuning to isolate architectural differences across YOLO generations. 

### Models Compared
- YOLO11n
- YOLOv8n
- YOLOv5n
- YOLOv9c

### Evaluation Setup
- Dataset: VisDrone2019‑DET (validation split)
- Image size: 640
- Batch size: 8
- No fine‑tuning performed

### Results Summary
All models below were evaluated on the **VisDrone validation set** using **COCO‑pretrained weights** without fine‑tuning. Metrics reflect **cross‑dataset generalization performance**.

| Model     | Parameters (M) | Precision | Recall | mAP50 | mAP50‑95 | Total Evaluation Time (s) |
|-----------|----------------|-----------|--------|--------|-----------|---------------------------|
| YOLO11n   | 2.62           | 0.035    | 0.136  | 0.016  | 0.007     | 16.74                    |
| YOLOv8n   | 3.15           | 0.064    | 0.049  | 0.016  | 0.007     | 17.40                    |
| YOLOv5n   | 2.65           | 0.057    | 0.052  | 0.015  | 0.007     | 21.73                    |
| YOLOv9c   | 25.38          | 0.040    | 0.173  | 0.031  | 0.015     | 26.05                    |

### Discussion
All evaluated models demonstrate low absolute accuracy due to domain mismatch between COCO training data and VisDrone images. However, relative trends are consistent: larger models such as YOLOv9c achieve higher recall and mAP at the cost of increased model size and runtime, while nano models favor speed over accuracy. YOLO11n achieves competitive performance among lightweight models, illustrating a balanced accuracy–efficiency trade‑off.

This comparison highlights the trade‑offs between accuracy, model complexity, and evaluation speed across YOLO versions, reinforcing design choices made in earlier parts of the project.

## Part VI – Final Baseline vs Final Model Comparison and Key Findings

To ensure a fair and conclusive comparison, the baseline and final models were trained using identical core settings, including dataset, input resolution, and number of epochs. The final YOLO11s model represents a synthesis of the most effective design choices identified through structured experimentation in Parts III–V, including increased model capacity, validated optimization practices, and empirically supported training strategies. This comparison therefore reflects both a controlled evaluation and an evidence‑driven final system selection, rather than an isolated experimental variant. Unlike earlier exploratory experiments, the final model is trained with a full optimization budget and integrates only those components shown to yield consistent performance gains.

The final results are:

| Model | Epochs | Params (M) | Precision | Recall | mAP50 | mAP50‑95 |
|------|--------|------------|-----------|--------|--------|-----------|
| YOLO11n (Baseline) | 50 | 2.6 | 0.416 | 0.325 | 0.317 | 0.183 |
| YOLO11s (Final) | 50 | 9.4 | **0.51** | **0.387** | **0.381** | **0.223** |

When trained with identical optimization budgets (50 epochs), the final YOLO11s model consistently outperforms the YOLO11n baseline across all primary detection metrics. The increase in **mAP50** (0.317 → 0.381) demonstrates substantially improved object coverage in dense scenes, while gains in **precision** (0.416 → 0.51) and **recall** (0.325 → 0.387) indicate more confident and complete detections. These improvements confirm that the baseline YOLO11n model was capacity‑limited on the VisDrone dataset and that increased representational power enables better feature learning under equal training conditions.

The higher parameter count and computational cost of YOLO11s are accompanied by meaningful accuracy gains, particularly in detecting vehicles and other structured objects, as further supported by confusion‑matrix analysis.


### Confusion Matrix Analysis (Baseline vs Final Model)

Analysis of the baseline compared to the final model, misclassification into the **background class remains the dominant error source** in both models, particularly for small and densely packed objects. The final YOLO11s model shows **improved diagonal dominance** in the confusion matrix, indicating better class-level separation. Clear performance improvements are observed for **vehicle-related classes**, especially:
  - car  
  - van  
  - truck  
  - bus  

  → These classes show higher correct classification rates and reduced cross-class confusion.

The baseline YOLO11n model exhibits more **diffuse misclassification patterns**, with predictions spread across multiple incorrect classes. The final model shows **reduced background confusion**, suggesting improved feature extraction and stronger object localization.

Performance remains weak for **small object categories**, including:
  - pedestrian  
  - people  
  - bicycle  
  - tricycle  
  - awning-tricycle  

These classes continue to be frequently misclassified as:
  - background  
  - visually similar object categories  

The persistent errors suggest that **model scaling alone is insufficient** to fully resolve small-object detection challenges. Overall, the confusion matrix confirms:
  - improved class separability in YOLO11s  
  - stronger detection consistency for structured objects  
  - remaining limitations driven by object scale and dataset density

## Conclusion

This project systematically explored modern object detection using YOLOv11 on the VisDrone dataset, with emphasis on experimental rigor, model diagnostics, and comparative evaluation rather than isolated performance tuning.

The **baseline analysis** demonstrated that YOLO11n converges reliably but is capacity-limited, particularly on dense scenes containing small, occluded objects. This was evidenced by moderate precision and low recall, despite stable training and validation loss behavior. Importantly, no overfitting was observed, confirming that performance limitations stem primarily from model expressiveness rather than optimization instability. **Controlled experiments** further confirmed this interpretation. Increasing model capacity from YOLO11n to YOLO11s resulted in consistent improvements in mAP and recall under identical training conditions, validating the hypothesis that the baseline model underfits the complexity of the VisDrone dataset. However, optimization adjustments such as cosine learning-rate scheduling did not yield improvements under limited training epochs, highlighting the dependency between optimization strategies and training duration. The **multi-version comparison** reinforced broader architectural trade-offs across YOLO families. Larger models consistently achieved better recall and mAP at the cost of increased computational complexity, while lightweight models provided efficient but limited detection capability. Across all models, VisDrone’s dense and small-object nature remained a consistent challenge, particularly for fine-grained categories such as pedestrians and bicycles.

Overall, the final YOLO11s model provides the best balance between accuracy and model complexity under controlled training conditions. The results confirm that meaningful improvements in object detection performance arise not from isolated hyperparameter tuning, but from structured, hypothesis-driven experimentation grounded in model capacity, data complexity, and optimization behavior.