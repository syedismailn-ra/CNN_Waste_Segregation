# Waste Material Segregation using CNNs

A deep learning project that classifies waste images into 7 categories using Convolutional Neural Networks (CNNs) built with TensorFlow/Keras.

> **IIIT Bangalore Assignment** - Deep Learning / Computer Vision

---

## Problem Statement

Automated waste segregation is a critical step in improving recycling efficiency and reducing landfill waste. This project builds a CNN-based image classifier to sort waste images into distinct material categories, enabling scalable and consistent sorting without manual intervention.

---

## Dataset

- **Total images:** 7,625
- **Classes (7):** Cardboard, Food Waste, Glass, Metal, Other, Paper, Plastic
- **Source format:** Folder-per-class structure
- **Original resolution:** 256×256 pixels (resized to 64×64 for training)

**Class distribution (imbalanced):**

| Class       | Count |
|-------------|-------|
| Plastic     | 2,295 |
| Paper       | ~1,000 |
| Metal       | ~1,000 |
| Food Waste  | ~1,000 |
| Other       | ~1,000 |
| Glass       | ~790  |
| Cardboard   | 540   |

---

## Project Structure

```
├── CNN_Waste_Segregation_SyedIsmailN.ipynb   # Main notebook
├── data.zip                                   # Dataset (not tracked)
└── README.md
```

---

## Methodology

### 1. Data Preparation
- Images loaded with PIL, converted to RGB, and normalised to [0, 1]
- Resized to 64×64 (balance between training speed and spatial detail)
- Labels encoded with `LabelEncoder` + one-hot encoded via `to_categorical`
- **Split:** 70% train / 15% validation / 15% test (stratified)

### 2. Model Architecture

A custom CNN with 3 convolutional blocks:

```
Conv2D(32) → BatchNorm → MaxPool
Conv2D(64) → BatchNorm → MaxPool
Conv2D(128) → BatchNorm → MaxPool
Flatten → Dense(256, ReLU) → Dropout(0.4) → Dense(7, Softmax)
```

- **Optimiser:** Adam
- **Loss:** Categorical Crossentropy
- **Callbacks:** EarlyStopping (patience=5), ReduceLROnPlateau (patience=3)

### 3. Data Augmentation (optional section)
Augmented the training set using horizontal flips, rotation (±20°), zoom (15%), and width/height shifts - effectively doubling the training data.

---

## Results

| Model              | Test Accuracy | Test Loss |
|--------------------|---------------|-----------|
| Baseline CNN       | 54.37%        | 1.3211    |
| CNN + Augmentation | **60.75%**    | 1.1300    |

**Per-class highlights (augmented model):**
- **Cardboard** - highest precision (0.62); visually distinctive
- **Plastic** - highest recall (0.79); benefits from large sample size
- **Glass** - worst recall (0.32); frequently confused with Plastic
- **Macro F1-score:** 0.51

---

## Key Findings

- The baseline model overfitted heavily (train accuracy ~91% vs test ~54%), suggesting the model memorised training patterns rather than generalising.
- Data augmentation improved generalisation by +6.4 percentage points.
- Class imbalance (Plastic: 2,295 vs Cardboard: 540) biased predictions towards majority classes.
- Visually similar classes (Glass ↔ Plastic, Metal ↔ Other) were the main sources of misclassification.

---

## Future Work

- **Stronger regularisation** - additional Dropout layers or L2 weight decay to combat overfitting
- **Larger input resolution** - 128×128 to recover spatial detail lost at 64×64
- **Transfer learning** - MobileNetV2 or EfficientNetB0 pretrained on ImageNet, expected to push accuracy well above 80%
- **Class-weighted loss** - penalise misclassification of minority classes (Cardboard, Glass) more heavily
- **Targeted oversampling** - SMOTE or class-specific augmentation for underrepresented categories

---

## Dependencies

```
numpy==1.26.4
pandas==2.2.2
matplotlib==3.10.0
seaborn==0.13.2
Pillow==11.1.0
tensorflow==2.18.0
keras==3.8.0
scikit-learn==1.6.1
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn Pillow tensorflow scikit-learn
```

---

## Usage

1. Clone the repository and place `data.zip` in the root directory.
2. Open `CNN_Waste_Segregation_SyedIsmailN.ipynb` in Jupyter or Google Colab.
3. Run all cells sequentially — the notebook will unzip the data, train the model, and display evaluation metrics.

---

## Author

**Syed Ismail N**   
IIIT Bangalore Assignment
