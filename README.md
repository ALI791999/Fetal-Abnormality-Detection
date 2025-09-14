# Fetal Abnormality Detection (HC18 Dataset)

This repository implements an **automated pipeline for fetal head circumference (HC) measurement and abnormality detection** using the [HC18 Head Circumference Challenge dataset](https://hc18.grand-challenge.org/).  
The system is designed to process 2D ultrasound images (18–38 gestational weeks) and perform:

- **Segmentation** of the fetal head plane from ultrasound images  
- **Head circumference (HC) estimation** via ellipse fitting  
- **Gestational age (GA) prediction** using INTERGROWTH-21st formulas  
- **Abnormality classification** (microcephaly, normal, macrocephaly) based on percentile growth charts  

This project combines modern deep learning with post-processing heuristics to yield anatomically coherent masks and clinically interpretable outputs.

---

## Model Overview

The architecture consists of three main components:

<p align="center">
  <img width="1027" height="491" alt="Model Architecture" src="https://github.com/user-attachments/assets/819165a9-f381-4a30-87e6-2d3519b48be5" />
</p>

1. **Learnable Resizer Block (512 → 256)**  
   - Replaces fixed interpolation with a trainable resizer.  
   - Learns to preserve anatomical boundaries while reducing noise (e.g., speckle).  
   - Produces a standardized 256×256 representation for the segmentation stage.  

2. **DeepLabV3+ Block**  
   - Uses an EfficientNet-B0 encoder and ASPP (Atrous Spatial Pyramid Pooling).  
   - Decoder fuses encoder features with ASPP output to generate detailed masks.  
   - Outputs a binary segmentation mask of the fetal head.  

3. **Post-Processing Block**  
   - Refines raw masks into anatomically plausible head contours.  
   - Steps include thresholding, largest connected component, morphological cleaning, and ellipse fitting.  
   - Final ellipse is used to compute HC and GA for abnormality classification.  

---

## Initial Requirement

### 1. Environment Setup
It is recommended to use **conda** or a virtual environment.  

```bash
# Create and activate a conda environment (Python 3.9+)
conda create -n fetal python=3.9 -y
conda activate fetal

### 2. Install Dependencies

Install all required libraries via the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Directory Structure

Fetal-Abnormality-Detection/ <br>
|
├── src/ # Source code (training, inference, post-processing, etc.)<br>
├── predictions/ # Predicted masks (created during inference)<br>
├── sample_inferences/ # Example outputs for quick reference<br>
├── data/ # (Download) HC18 dataset (excluded from repo)<br>
├── models/ # (Download) Trained checkpoints (excluded from repo)<br>
├── requirements.txt<br>
├── README.md<br>

Link for the "data" and "models" directories: https://drive.google.com/drive/folders/1kVK17OFyQlDXQlTyPmFiGER9PfR-DU0u?usp=sharing


## Training

To train the model on the HC18 dataset, simply run the training script:

```bash
python src/train.py
```
This will:  
- Train the Learnable Resizer + DeepLabV3+ pipeline end-to-end  
- Save model checkpoints in the `models/` directory  

## Inference

To generate predictions using a trained model, run:

```bash
python src/inference.py
```
This will:
- Load the saved model checkpoint
- Apply the segmentation model on validation/test data
- Save binary prediction masks in the predictions/ folder

<p align="center">
  <img width="1199" height="452" alt="Image" src="https://github.com/user-attachments/assets/57122645-5c91-4a1d-9901-fdd3fda8129f" />
</p>

---

## Performance (DICE Score)

Segmentation performance on the **HC18 test set** was evaluated using the **DICE coefficient**, a standard metric for overlap between predicted and ground-truth masks.  

You can reproduce the evaluation by running:

```bash
python src/DICE.py
```

### === Summary ===

- **N** = 199  
- **Mean Dice**: 0.9653  
- **Median Dice**: 0.9708  
- **Std Dice**: 0.0234  
- **Min / Max**: 0.8017 / 0.9935  
- **Global Dice**: 0.9668

## Abnormality Detection

After inference, you can perform abnormality detection by running:

```bash
python src/abnormality.py
```
This will:  
- Fit ellipses on the predicted masks  
- Compute head circumference (HC) using **[Ramanujan’s approximation](https://mathworld.wolfram.com/Ellipse.html)** for ellipse perimeter  
- Estimate gestational age (GA) using the **[INTERGROWTH-21st standards](https://intergrowth21.tghn.org/standards-tools/)**  
- Classify each case as **microcephaly / normal / macrocephaly** using **standard percentile growth tables** from INTERGROWTH-21st  
- Save results in an output file (e.g., `.xlsx` or `.csv`) for further analysis  


