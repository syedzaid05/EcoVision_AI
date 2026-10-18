# EcoVision AI

Waste image classification using **MobileNetV2 transfer learning** (TensorFlow/Keras),
with a **Streamlit** UI. Given a photo of a waste item, the model predicts which of six
recycling categories it belongs to.

Classes: **Cardboard / Glass / Metal / Paper / Plastic / Trash**.

---

## Table of contents

- [Deployment](#deployment)
- [Overview](#overview)
- [Pipeline](#pipeline)
- [Results](#results)
- [Dataset](#dataset-trashnet)
- [Model architecture](#model-architecture)
- [Preprocessing & augmentation](#preprocessing--augmentation)
- [Training](#training)
- [Hyperparameters](#hyperparameters)
- [Hardware notes](#hardware-notes-8gb-ram--i5-cpu-only)
- [Environment](#environment)
- [Setup](#setup)
- [Get the dataset](#get-the-dataset)
- [Usage](#usage)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [Limitations & future work](#limitations--future-work)
- [Acknowledgements](#acknowledgements)

---

## Overview

Waste sorting is a real-world computer-vision problem. This project fine-tunes a
pretrained **MobileNetV2** CNN (trained on ImageNet) to classify waste photos into six
classes, using **transfer learning** rather than training from scratch — essential on
limited hardware and data.

Key design choices, driven by an **8GB RAM / i5 CPU-only laptop**:

- `tensorflow-cpu` (native GPU is unsupported on Windows for modern TF).
- A **two-phase transfer-learning** strategy: train only the classifier head first, then
  unfreeze the top layers for light fine-tuning.
- **MobileNetV2**, chosen for its small size (~2.26M params) and speed on CPU.
- **Class weighting** to counteract the heavily imbalanced `trash` class.
- `tf.data` streaming with caching OFF to stay within RAM.

---

## Pipeline

```
Input Image (RGB)
  |
  v
Image Preprocessing          resize 224x224, normalize pixels to [-1, 1]
  |
  v
MobileNetV2 (pretrained)     ImageNet feature extractor
  |
  v
Global Average Pooling       7x7x1280 -> 1280-d vector
  |
  v
Dropout (0.3)                reduce overfitting
  |
  v
Dense + Softmax (6)          six class probabilities
  |
  v
Predicted Waste Category     Cardboard / Glass / Metal / Paper / Plastic / Trash
```

---

## Results

| Metric | Value |
|---|---|
| Validation accuracy | **0.8554** |
| Validation loss | 0.4079 |
| Macro precision | 0.8423 |
| Macro recall | 0.8350 |
| Macro F1 | 0.8374 |
| Weighted precision | 0.8586 |
| Weighted recall | 0.8554 |
| Weighted F1 | 0.8555 |
| Total parameters | 2,265,670 |
| Trainable params (head) | 7,686 |
| Frozen base params | 2,257,984 |
| Fine-tuned base layers | 54 / 154 |
| Epochs run | 8 head + 5 fine-tune |
| Training time (CPU) | ~23 min |
| Model file size | ~23.5 MB |

### Classification report (validation, 505 images)

```
              precision    recall  f1-score   support

   cardboard     0.9250    0.8916    0.9080        83
       glass     0.7934    0.9320    0.8571       103
       metal     0.8919    0.8462    0.8684        78
       paper     0.9060    0.8548    0.8797       124
     plastic     0.8235    0.7955    0.8092        88
       trash     0.7143    0.6897    0.7018        29

    accuracy                         0.8554       505
   macro avg     0.8423    0.8350    0.8374       505
weighted avg     0.8586    0.8554    0.8555       505
```

Confusion matrix: [`outputs/confusion_matrix.png`](outputs/confusion_matrix.png)

### Per-epoch history

**Head phase** (backbone frozen, `lr = 1e-3`):

| Epoch | Train acc | Train loss | Val acc | Val loss |
|---|---|---|---|---|
| 1 | 0.4782 | 1.3778 | 0.7366 | 0.7524 |
| 2 | 0.6944 | 0.8230 | 0.7822 | 0.6035 |
| 3 | 0.7433 | 0.6938 | 0.8059 | 0.5565 |
| 4 | 0.7725 | 0.6281 | 0.8238 | 0.4985 |
| 5 | 0.8056 | 0.5581 | 0.8495 | 0.4515 |
| 6 | 0.8046 | 0.5456 | 0.8337 | 0.4749 |
| 7 | 0.8071 | 0.5367 | 0.8535 | 0.4444 |
| 8 | 0.8215 | 0.4902 | 0.8515 | 0.4086 |

**Fine-tune phase** (54/154 base layers unfrozen, `lr = 1e-5`):

| Epoch | Train acc | Train loss | Val acc | Val loss |
|---|---|---|---|---|
| 1 | 0.7122 | 0.8310 | **0.8554** | **0.4079** |
| 2 | 0.7661 | 0.6890 | 0.8475 | 0.4098 |
| 3 | 0.7844 | 0.6286 | 0.8515 | 0.4104 |
| 4 | 0.7992 | 0.5809 | 0.8535 | 0.4118 |
| 5 | 0.8081 | 0.5276 | 0.8495 | 0.4112 |

Best checkpoint (`val_accuracy = 0.8554`) came from fine-tune epoch 1. Later epochs
began to overfit (train accuracy rose while validation flattened/dipped), so
`restore_best_weights` reverted to this peak. Raw logs:
`outputs/history_head.csv`, `outputs/history_finetune.csv`.

---

## Dataset (TrashNet)

Source: [garythung/trashnet](https://github.com/garythung/trashnet) — the `dataset-resized/`
folder (512x384 originals, resized by the dataset authors).

| Class | Images | Share |
|---|---|---|
| cardboard | 403 | 15.9% |
| glass | 501 | 19.8% |
| metal | 410 | 16.2% |
| paper | 594 | 23.5% |
| plastic | 482 | 19.1% |
| trash | 137 | 5.4% |
| **Total** | **2,527** | 100% |

Split via `image_dataset_from_directory(validation_split=0.2)`:
**2,022 train / 505 validation**. A single `.DS_Store` file is ignored by the loader.

### Class imbalance handling

The `trash` class has ~4x fewer images than `paper`. Inverse-frequency **class weights**
are applied as per-sample weights during training. This raised `trash` recall from
**0.138 (unweighted 1-epoch baseline) to 0.690** in the final model.

| Class | Weight |
|---|---|
| cardboard | 1.0531 |
| glass | 0.8467 |
| metal | 1.0151 |
| paper | 0.7170 |
| plastic | 0.8553 |
| trash | 3.1204 |

---

## Model architecture

Built in [`src/model.py`](src/model.py). The MobileNetV2 preprocessing is baked into the
model as a `Rescaling` layer, so the saved model accepts **raw 0-255 RGB images** at
inference (no external preprocessing function needed).

| # | Layer | Output shape | Params | Notes |
|---|---|---|---|---|
| 0 | InputLayer (`image`) | (224, 224, 3) | 0 | raw pixel values 0-255 |
| 1 | Rescaling (`preprocess`) | (224, 224, 3) | 0 | `scale=1/127.5, offset=-1` |
| 2 | MobileNetV2 (`base`) | (7, 7, 1280) | 2,257,984 | ImageNet weights, `include_top=False` |
| 3 | GlobalAveragePooling2D | (1280,) | 0 | pooled feature vector |
| 4 | Dropout (0.3) | (1280,) | 0 | regularization |
| 5 | Dense + Softmax | (6,) | 7,686 | `6*1280 + 6` |

- MobileNetV2 uses **inverted residual blocks** with **depthwise separable convolutions**
  (width multiplier 1.0), which is why it is fast on CPU.
- Phase 1 freezes the entire base. Phase 2 unfreezes layers **from index 100 onward**
  (54 of 154 layers) and trains with a small learning rate.

---

## Preprocessing & augmentation

Implemented in [`src/dataset.py`](src/dataset.py).

**Inference / validation preprocessing**
1. Resize to **224x224** (`image_dataset_from_directory`).
2. Cast to `float32` (values kept in 0-255).
3. `Rescaling(1/127.5, offset=-1)` inside the model -> range **[-1, 1]**, matching
   `mobilenet_v2.preprocess_input`.

**Training-only augmentation** (applied on-the-fly via `tf.data.map`):

| Transform | Setting |
|---|---|
| `RandomFlip` | horizontal |
| `RandomRotation` | factor 0.1 |
| `RandomZoom` | factor 0.1 |
| `RandomContrast` | factor 0.1 |

Pipeline flow: `decode -> resize -> augment (train only) -> shuffle(1000) -> prefetch(AUTOTUNE)`.
**Caching is disabled** (`CACHE = False`) because decoded 224x224 images would consume
~1.5 GB of the available 8 GB.

---

## Training

Two-phase transfer learning in [`train.py`](train.py):

1. **Head phase** — base frozen; trains only the new classifier. Optimizer Adam
   (`lr = 1e-3`), up to 8 epochs, with class weights.
2. **Fine-tune phase** — unfreezes base layers from index 100; Adam (`lr = 1e-5`), 5 epochs.

**Loss:** `categorical_crossentropy` (one-hot labels from the dataset).
**Class weights:** computed from the training subset, attached as sample weights.

**Callbacks:**

| Callback | Configuration |
|---|---|
| `ModelCheckpoint` | monitor `val_accuracy`, save best only |
| `EarlyStopping` | monitor `val_loss`, patience 5, restore best weights |
| `ReduceLROnPlateau` | monitor `val_loss`, factor 0.5, patience 2, min_lr 1e-7 |
| `CSVLogger` | writes per-epoch history CSV |

---

## Hyperparameters

Defined in [`config.py`](config.py); most are overridable via CLI flags.

| Name | Value | Description |
|---|---|---|
| `IMG_SIZE` | 224 | input resolution |
| `BATCH_SIZE` | 32 | batch size |
| `NUM_CLASSES` | 6 | output classes |
| `VAL_SPLIT` | 0.2 | validation fraction |
| `SEED` | 123 | split/shuffle seed |
| `EPOCHS_HEAD` | 15 (run: 8) | head-phase epochs |
| `EPOCHS_FINETUNE` | 10 (run: 5) | fine-tune epochs |
| `LR_HEAD` | 1e-3 | head-phase learning rate |
| `LR_FINETUNE` | 1e-5 | fine-tune learning rate |
| `FINE_TUNE_AT` | 100 | unfreeze base layers from this index |
| `DROPOUT` | 0.3 | dropout rate |
| `CACHE` | False | cache decoded images (off for 8GB) |
| `USE_CLASS_WEIGHTS` | True | apply inverse-frequency class weights |

---

## Hardware notes (8GB RAM / i5, CPU-only)

- `tensorflow-cpu` is installed — **no GPU** (native GPU TF is unavailable on Windows).
- Backbone starts **frozen**, then the top **54/154** layers are fine-tuned at a low LR.
- Batch size 32, `tf.data` streaming, **caching disabled** to keep RAM low.
- Throughput: ~1 min/epoch fully frozen, ~2 min/epoch while fine-tuning.

---

## Environment

`.venv` created with **Python 3.12** (TensorFlow does not reliably support Python 3.13 on
Windows; Python 3.13.5 was present but unused).

| Package | Version |
|---|---|
| Python | 3.12.3 |
| tensorflow-cpu | 2.19.1 |
| keras | 3.15.1 |
| numpy | 2.1.3 |
| Pillow | 12.3.0 |
| matplotlib | 3.11.2 |
| scikit-learn | 1.9.1 |
| streamlit | 1.66.0 |
| pandas | 3.0.6 |

> Note: each shell invocation is independent, so call the venv's interpreter directly
> (`.venv\Scripts\python.exe`) or activate it in your own terminal.

---

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Get the dataset

```powershell
.\.venv\Scripts\python.exe download_data.py
```

`download_data.py` downloads the TrashNet repo zip, extracts it, detects the **nested**
`dataset-resized.zip`, and places the class folders at `data/dataset-resized/`.

Manual alternative — download the TrashNet repo and place `dataset-resized/` at
`data/dataset-resized/`:

```
data/dataset-resized/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

---

## Usage

### Train

```powershell
.\.venv\Scripts\python.exe train.py
```

The run that produced these results:

```powershell
.\.venv\Scripts\python.exe train.py --epochs-head 8 --epochs-finetune 5
```

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `data/dataset-resized` | dataset root |
| `--epochs-head` | 15 | frozen-backbone epochs |
| `--epochs-finetune` | 10 | fine-tune epochs |
| `--batch-size` | 32 | batch size |
| `--img-size` | 224 | input resolution |
| `--fine-tune-at` | 100 | unfreeze base layers after this index |
| `--lr-head` | 1e-3 | head-phase learning rate |
| `--lr-finetune` | 1e-5 | fine-tune learning rate |
| `--no-finetune` | off | skip fine-tuning |
| `--no-class-weights` | off | disable class weighting |
| `--cache` | off | cache decoded images in RAM |
| `--model-path` | `models/waste_mobilenetv2.keras` | output path |

Best checkpoint -> `models/waste_mobilenetv2.keras`.

### Evaluate

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

Prints loss/accuracy and a full classification report, and writes
`outputs/confusion_matrix.png`. Use `--test-dir <dir>` to evaluate on a held-out set.

### Predict a single image

```powershell
.\.venv\Scripts\python.exe predict.py path\to\image.jpg
```

```
Image: ...\glass\glass1.jpg
Predicted: glass (73.25%)

  glass       73.25%
  metal       15.15%
  plastic      5.93%
  cardboard    3.14%
  paper        1.89%
  trash        0.65%
```

### Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Opens at **http://localhost:8501**. Upload a JPG/PNG/WEBP image to see the predicted class,
confidence, a horizontal probability bar chart, and a numeric probability table. If no
trained model is found, the app shows a prompt to run `train.py` first.

---

## Project layout

```
config.py            hyperparameters, class names, paths
download_data.py     fetch + extract TrashNet (handles the nested zip)
train.py             two-phase training + class weights
evaluate.py          loss/accuracy, classification report, confusion matrix
predict.py           CLI single-image inference
app.py               Streamlit UI
src/dataset.py       tf.data pipelines, augmentation, class weights
src/model.py         MobileNetV2 model builder
models/              saved .keras model  (waste_mobilenetv2.keras)
outputs/             history CSVs, confusion_matrix.png
data/                TrashNet dataset (gitignored)
.venv/               Python 3.12 virtual environment (gitignored)
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: tensorflow` | Use Python 3.12 venv and `.venv\Scripts\python.exe`. |
| TensorFlow install fails on Python 3.13 | TF does not reliably support 3.13 on Windows — use 3.12. |
| Very slow `pip install` | PyPI bandwidth; the `tensorflow-cpu` wheel alone is ~376 MB. |
| `Prefetch autotuner ... ram budget exceeded` | Harmless warning on 8GB; sequencing is still correct. |
| `Activate.ps1` has no effect on my commands | Each tool shell is fresh; call the venv python directly. |
| Dataset not found error | Run `download_data.py` or place `dataset-resized/` under `data/`. |
| GPU not detected | Expected — `tensorflow-cpu` on Windows is CPU-only. |

---

## Limitations & future work

- **Small dataset** (2,527 images) with a rare `trash` class (137) — accuracy is capped.
- **No separate test set**; the validation split doubles as the reporting set.
- Fine-tuning with `lr = 1e-5` did not beat the frozen-head peak — the backbone was already
  strong and the dataset small, so extra tuning mostly overfit.
- Possible improvements: stronger augmentation, k-fold cross-validation, backbones like
  EfficientNetB0 / MobileNetV3, focal loss, Grad-CAM explainability, and wrapping the model
  as a REST API for deployment.

---

## Acknowledgements

- [TrashNet](https://github.com/garythung/trashnet) dataset (Gary Thung & Mindy Yang).
- [MobileNetV2](https://arxiv.org/abs/1801.04381) (Sandler et al., 2018) via
  `keras.applications`.
- Built with TensorFlow / Keras and Streamlit.

---

## Deployment

The app deploys free on [Streamlit Community Cloud](https://share.streamlit.io) and
auto-redeploys on every push to `main`:

1. Sign in at share.streamlit.io with GitHub → **Create app**
2. Select this repo, branch `main`, main file **`app.py`** (Python 3.12)
3. **Deploy** — the trained model (`models/waste_mobilenetv2.keras`) is committed to the
   repo, so the cloud app loads it directly with **no secrets or extra setup required**
4. First visit after ~12h idle takes 1–2 min (TensorFlow cold start)

App theme/config lives in `.streamlit/config.toml`; secrets (if ever needed) go in the
gitignored `.streamlit/secrets.toml`.
