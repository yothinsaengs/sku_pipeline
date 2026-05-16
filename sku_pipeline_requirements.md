# SKU Detection + Classification Pipeline — Project Requirements

## 1. Problem Statement

Training a one-stage object detection model requires labeling **all known classes** in every image.
This is expensive and rigid — even if you only care about maintaining one or a few classes, you
must re-label everything when the class list changes.

**Solution:** Decouple detection from classification using a two-stage pipeline:

```
Input Image
    │
    ▼
[Stage 1] General SKU Detector
    │  Detects all product-like ROIs (class-agnostic)
    │  Output: bounding boxes
    ▼
[Stage 2] ROI Classifier
    │  Classifies each cropped ROI
    │  Trained with sigmoid + label smoothing
    │  Config-driven positive/negative mapping
    ▼
Final Output: box + class label + confidence per ROI
```

**Key benefits:**
- Stage 1 never needs to know about specific SKU classes
- Stage 2 can be retrained/extended without touching the detector
- Negative data is explicitly controlled via config — not implicit from missing labels
- Removing surrounding context is a deliberate trade-off (pros: focus; cons: loss of spatial
  context) — measured explicitly in POC metrics

---

## 2. Scope — POC Goals

Train and validate the **Stage 2 Classifier** on Google Colab (GPU), with the codebase also
runnable locally on Mac (CPU/MPS) for sanity checks — no code changes required between environments.

Compare classification metrics against an equivalent one-stage model baseline (e.g. YOLOv8) to
validate the hypothesis that the two-stage approach is competitive or better for targeted classes.

Stage 1 (General SKU Detector) is **assumed pre-trained or provided** — the POC focuses on
Stage 2 and the data pipeline feeding it. Detector recall should be logged separately so failures
can be attributed to Stage 1 vs Stage 2.

---

## 3. Data Format

### 3.1 Input: Label Studio YOLO Export

```
dataset/
├── images/
│   ├── img001.jpg
│   └── img002.jpg
└── labels/
    ├── img001.txt
    └── img002.txt
```

Each `.txt` label file follows YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to `[0, 1]` relative to image dimensions.

### 3.2 Train / Val / Test Split

Split is **per-image** — all boxes from a given image go entirely into one split.
This prevents spatial context leakage across splits.

---

## 4. Config File (`config.yaml`)

Single source of truth for all experiment parameters.

```yaml
# ── Dataset ───────────────────────────────────────────────────────────────────
dataset:
  images_dir: "dataset/images"
  labels_dir: "dataset/labels"
  split:
    train: 0.7
    val:   0.15
    test:  0.15
  split_seed: 42

# ── Class Mapping ─────────────────────────────────────────────────────────────
classes:
  - id: 0
    name: "cola_can"
    role: positive          # sigmoid target → 1.0 (with label smoothing)
  - id: 1
    name: "water_bottle"
    role: positive
  - id: 2
    name: "shelf_label"
    role: negative          # all sigmoid outputs trained → ~0.0
  - id: 3
    name: "hand"
    role: negative
# Any class_id NOT listed is ignored (skipped during dataset build)

# ── Label Smoothing ───────────────────────────────────────────────────────────
label_smoothing: 0.1        # ε applied to positive targets only

# ── Input / Preprocessing ─────────────────────────────────────────────────────
input:
  min_size: 4               # pixels — discard boxes smaller than this
  max_size: 224             # pixels — longest side after padding
  context_margin: 0.15      # expand bbox by this fraction before crop (e.g. 0.15 = +15%)
  pad_color: 114            # gray value for letterbox padding (keep aspect ratio)

# ── Sampling / Class Balance ──────────────────────────────────────────────────
sampling:
  pos_neg_ratio: "1:2"      # enforced per batch (batch-level ratio sampler)
  # Experiment: vary negative budget size to find minimum viable negative count

# ── Model ─────────────────────────────────────────────────────────────────────
model:
  backbone: "efficientnetv2_s"    # timm pretrained ImageNet (s / m / l configurable)
  pretrained: true
  head:
    global_pool: "avg"            # AdaptiveAvgPool → fixed vector regardless of input size
    dropout: 0.3
  num_classes: 2                  # number of positive classes (one sigmoid output per class)

# ── Training ──────────────────────────────────────────────────────────────────
training:
  epochs: 50
  batch_size: 32
  optimizer: "adamw"
  lr: 0.0003
  weight_decay: 0.01
  scheduler: "cosine"
  device: "auto"                  # auto-detect: cuda → mps → cpu
  finetune: "all"                 # fine-tune all backbone layers from start

  early_stopping:
    enabled: true
    monitor: "val_f1_macro"
    patience: 10                  # stop if no improvement for N epochs
    min_delta: 0.001

  resume:
    enabled: false
    checkpoint_path: ""           # path to .pth checkpoint to resume from

  sanity_check:
    enabled: true
    num_batches: 2                # run forward pass on N batches before full training

  log_every_n_steps: 50          # print metrics to log file every N steps

  checkpoints:
    keep_last_n: 3                # keep last N epoch checkpoints + best model
    save_dir: "checkpoints"       # relative to run folder

  auto_eval_after_train: true     # auto-run test set evaluation after training completes

# ── WandB ─────────────────────────────────────────────────────────────────────
wandb:
  enabled: false                  # off by default — enable via --wandb flag at runtime
  project: "sku_classifier"
  log:
    - loss
    - acc
    - f1_macro
    - precision_macro
    - recall_macro                # logged per epoch for train + val

# ── Loss ──────────────────────────────────────────────────────────────────────
loss:
  type: "focal"                   # options: "focal" | "bce"
  focal:
    alpha: 0.25
    gamma: 2.0

# ── Decision Threshold ────────────────────────────────────────────────────────
threshold: 0.5                    # fixed for POC; per-class tuning deferred

# ── Augmentation ──────────────────────────────────────────────────────────────
augmentation:
  # Safe — always applied
  safe:
    horizontal_flip: true
    vertical_flip: false
    color_jitter:
      brightness: 0.2
      contrast:   0.2
      saturation: 0.2
      hue:        0.05
    random_rotation_deg: 15

  # Risky — applied randomly with given probability (random mix)
  random_mix:
    - name: "aggressive_color_jitter"
      prob: 0.3
      params: { brightness: 0.5, contrast: 0.5, saturation: 0.5 }
    - name: "slight_perspective"
      prob: 0.2
      params: { distortion_scale: 0.2 }
    - name: "gaussian_blur"
      prob: 0.2
      params: { kernel_size: 3 }

  # Random Occlusion — simulate partial obstruction of SKU crops
  random_occlusion:
    enabled: true
    prob: 0.3                     # probability of applying per crop
    max_count: 4                  # random number of occluders: 1 to max_count
    shapes: ["rect", "ellipse", "polygon"]  # arbitrary shape types
    scale_range: [0.05, 0.25]     # occluder size as fraction of crop size
    fill: "random"                # fill color: "random" | "black" | "mean"

  # Class Mixing — neg+pos blending for smoother decision boundary
  class_mixing:
    enabled: true
    strategy: "both"              # "mixup" | "cutmix" | "both"
    pairs: "neg_pos"              # only mix negative + positive samples
    prob: 0.3                     # probability of applying mixing per batch
    mixup:
      alpha: 0.4                  # Beta distribution α — controls blend strength λ
    cutmix:
      alpha: 1.0                  # Beta distribution α for region area ratio

# ── Extra Negative Sources ────────────────────────────────────────────────────
extra_negatives:
  enabled: false

  # Synthetic negatives — random crop-paste from own dataset images (CutPaste-style)
  synthetic_cutpaste:
    enabled: true
    count_per_epoch: 500          # how many synthetic negatives to generate per epoch
    shapes: ["rect", "ellipse", "polygon"]  # patch shapes to sample
    scale_range: [0.05, 0.3]      # patch size as fraction of source image
    source: "non_roi_regions"     # crop from background regions (outside all bboxes)
    paste_background: "random_crop_same_image"  # paste onto random bg crop from same image

  # Real open-source negatives — diverse out-of-domain coverage
  open_source:
    - name: "coco_background"
      path: "data/extra_neg/coco"
      type: "random_crop"         # crop from non-object regions
      enabled: false
    - name: "openimages"
      path: "data/extra_neg/openimages"
      type: "random_crop"
      enabled: false
    - name: "sku110k_non_target"
      path: "data/extra_neg/sku110k"
      type: "random_crop"         # hard negatives — same retail domain
      enabled: false
    - name: "dtd_textures"
      path: "data/extra_neg/dtd"
      type: "full_image"          # pure texture, no objects
      enabled: false
    - name: "places365"
      path: "data/extra_neg/places365"
      type: "random_crop"
      enabled: false

  # Mix ratio per batch: labeled_neg : synthetic : open_source
  mix_ratio:
    labeled:    0.5
    synthetic:  0.3
    open_source: 0.2

# ── Class Split Experiment ────────────────────────────────────────────────────
class_split_experiment:
  enabled: false                  # set true to activate generalization experiment

  # Option A: explicit assignment
  mode: "explicit"                # "explicit" | "random"
  explicit:
    positive: ["cola_can", "water_bottle"]
    negative: ["shelf_label", "hand"]   # or "ALL_REMAIN"
    # eval (unseen) = all classes not listed above

  # Option B: random sampling — picks n/4 pos and n/4 neg from all known classes
  random:
    seed: 42                      # fixed seed; override per run if needed
    pos_fraction: 0.25            # fraction of all classes assigned as train positive
    neg_fraction: 0.25            # fraction of all classes assigned as train negative
    # eval (unseen) = remaining ~0.5 of classes

  # Key metric: FP rate on unseen eval classes — does negative signal generalize?

# ── Logging & Checkpointing ───────────────────────────────────────────────────
logging:
  run_dir: "runs"                 # runs/<timestamp>_<experiment_name>/
  log_file: "train.log"          # saved inside run folder (console output redirected here)
  metrics_csv: true
  plots: true                     # seaborn/matplotlib plots saved per run
  aug_preview:
    enabled: true                 # save grid of augmented crops before training starts
    num_samples: 16               # number of crops to show in grid
    save_file: "aug_preview.png"  # saved to run folder
  cross_run_csv: "runs/all_runs.csv"  # append per-run summary metrics for easy comparison
```

---

## 5. Config Validation

Before any training starts, the CLI validates `config.yaml` and raises clear, descriptive
errors if values are invalid. Validation runs automatically on `train` and `eval` commands.

**Checks include:**

| Check | Example error |
|---|---|
| Required fields present | `missing field: classes` |
| Class roles valid | `role must be "positive" or "negative", got "pos"` |
| Fractions sum correctly | `class_split pos_fraction + neg_fraction must be ≤ 1.0` |
| Paths exist | `images_dir not found: dataset/images` |
| Numeric ranges valid | `context_margin must be in [0, 1], got 1.5` |
| Loss type recognized | `loss.type must be "focal" or "bce", got "mse"` |
| Mix ratio sums to 1.0 | `extra_negatives.mix_ratio must sum to 1.0` |
| At least one positive class | `no classes with role: positive defined` |

Errors include the offending key path and suggested fix:
```
ConfigError: training.early_stopping.patience must be > 0, got -1
  → set a positive integer, e.g. patience: 10
```

---

## 6. Model Architecture

### 5.1 Backbone
- **EfficientNetV2-S** (or M/L — configurable via `config.yaml`) loaded from `timm`, pretrained on ImageNet
- Feature extractor frozen for first N epochs, then unfrozen for full fine-tuning

### 5.2 Multi-Scale Input Strategy
- Input crop size varies from **4 × 4 to 224 × 224 pixels** due to wide variation in SKU box sizes
- Preprocessing pipeline per ROI:
  1. Expand bbox by `context_margin` (adds neighborhood pixel context)
  2. Letterbox pad with gray (`pad_color: 114`) to preserve aspect ratio
  3. Resize longest side to `max_size` (224)
- Boxes smaller than `min_size` (4px) are discarded
- **Head uses `AdaptiveAvgPool2d`** → collapses any spatial size to a fixed vector, making the architecture natively multi-scale

### 5.3 Classification Head
```
AdaptiveAvgPool2d(1, 1)
    │
Flatten
    │
Dropout(p=0.3)
    │
Linear(in_features, num_classes)
    │
Sigmoid   ← independent binary probability per positive class
```

### 5.4 Output Semantics
- Shape: `[batch, num_classes]`, values in `[0, 1]`
- Each output is an **independent binary probability** (not softmax — classes are not mutually exclusive)
- **Negative samples** train all outputs toward `~0.0`
- **Positive samples** train the corresponding output toward `1.0 - ε` (label smoothing)

---

## 7. Loss Functions (Experiment Ablation)

Both implemented and switchable via `config.yaml → loss.type`.

### 6.1 Focal Loss (default)
```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```
- Down-weights easy examples, focuses training on hard ones
- Naturally handles class imbalance without explicit reweighting
- Params `alpha` and `gamma` configurable in `config.yaml`

### 6.2 Binary Cross-Entropy + Label Smoothing (baseline)
```
target_smooth = (1 - ε)   for positive
target_smooth = 0.0        for negative
```
- Standard baseline; compare against Focal Loss across same runs

---

## 8. Sampling Strategy

### 7.1 Batch-Level Ratio Sampler
- Enforce **1 positive : 2 negative** ratio per batch
- Implemented as a custom `torch.utils.data.Sampler`
- Oversamples positives or subsamples negatives as needed to maintain ratio

### 7.2 Negative Budget Experiment
**Goal:** find the minimum viable number of negative samples that keeps FP rate acceptably low,
given that negative labeling is expensive.

Procedure:
- Fix positive sample count
- Train across multiple negative budget sizes (e.g. 50 / 100 / 200 / 500 negatives)
- Keep `pos_neg_ratio`, loss type, and all other config identical across runs
- Compare FP rate, Precision, and F1 across runs
- Plot `neg_budget_experiment.png` summarizing results

This is **controlled ratio sampling**, not hard negative mining (which requires active selection
of the model's worst-error negatives — deferred to future work).

### 7.3 Random Occlusion Augmentation

Simulates partial obstruction of SKU crops — common in retail scenes where products are
partially blocked by hands, shelf edges, price tags, or other items.

**Pipeline per crop:**
1. Sample a random occluder count: `1` to `max_count`
2. For each occluder, sample a shape: `rect`, `ellipse`, or `polygon`
3. Sample occluder size from `scale_range` (fraction of crop size)
4. Place at a random position within the crop
5. Fill with: `random` color, `black`, or `mean` pixel value of the crop

Configured under `config.yaml → augmentation.random_occlusion`.
Applied as part of the safe/random_mix augmentation pipeline with its own `prob`.

---

### 7.4 Class Mixing Augmentation (MixUp / CutMix)

**Goal:** smooth the decision boundary between positive and negative classes by training on
blended samples. Forces the model to output proportionally lower confidence on "contaminated"
positives rather than hard-firing on any positive-looking texture.

**Pairs:** only **negative + positive** samples are mixed — neg+neg mixing is skipped as it
adds no useful gradient signal.

**Label blending:**

- **MixUp** — blend pixel values: `image = λ·pos + (1-λ)·neg`
  ```
  target = λ · pos_target + (1-λ) · neg_target
  ```
  λ sampled from `Beta(α, α)` — higher α = stronger blending toward 0.5

- **CutMix** — paste a rectangular region from neg onto pos:
  ```
  target = (cut_area / total_area) · pos_target + (1 - cut_area/total_area) · neg_target
  ```
  Region area ratio sampled from `Beta(α, α)`

Both strategies configurable via `config.yaml → augmentation.class_mixing.strategy`:
`"mixup"` | `"cutmix"` | `"both"` (randomly alternate per sample)

Applied with probability `prob` per batch — does not replace safe/random_mix augmentations.

### 7.5 Class Split Generalization Experiment

**Goal:** determine whether the negative signal generalizes to unseen classes — i.e. if the model
is trained on only `n/4` positive and `n/4` negative classes, does it suppress FP on the
remaining `n/2` unseen classes at eval time?

**Setup:**

| Split | Classes | Role |
|---|---|---|
| Train positive | `n/4` classes | Model learns to fire on these |
| Train negative | `n/4` classes (or `ALL_REMAIN`) | Model learns to suppress these |
| Eval (unseen) | remaining `~n/2` classes | Never seen during training |

**Class assignment modes (configured in `config.yaml → class_split_experiment`):**

- `explicit` — manually name which classes are train pos / train neg / eval.
  `negative` accepts a list of class names or the special value `"ALL_REMAIN"` (every class
  not listed as positive becomes negative).
- `random` — randomly sample `pos_fraction` and `neg_fraction` of all known classes each run;
  remainder becomes the unseen eval set. Controlled by a fixed `seed` (overridable per run).

**Key metric:** FP rate on unseen eval classes.

**Output plot:** `class_split_experiment.png` — FP rate on unseen classes vs number of training
negative classes used, to identify the minimum negative class coverage needed for generalization.

---

## 9. Extra Negative Sources

To improve generalization of the negative signal — especially for unseen classes at inference —
two tiers of additional negatives can be mixed into training alongside labeled negatives.
All controlled via `config.yaml → extra_negatives`.

### 8.1 Synthetic Negatives — CutPaste-Style Crop-Paste

Random shape patches cropped from **non-ROI regions of your own dataset images** and pasted
onto background crops from the same image. Labeled as negative (all outputs → ~0.0).

**Why it works:**
- Zero extra data needed — mines from images you already have
- Domain-matched texture, lighting, and color distribution
- Infinitely scalable — generative, not a fixed dataset
- Teaches the model "random plausible-looking patch ≠ known SKU"

**Pipeline:**
1. For each source image, identify background regions (pixels outside all bboxes)
2. Sample a random patch using one of: `rect`, `ellipse`, `polygon`
3. Scale patch to `scale_range` fraction of the source image size
4. Paste onto a random background crop from the same image
5. Apply standard augmentation pipeline
6. Assign all-zero target (negative)

### 8.2 Real Open-Source Negatives

Diverse out-of-domain crops for broader negative coverage:

| Dataset | Type | Value |
|---|---|---|
| COCO | Random crop from non-object regions | General scene diversity |
| OpenImages V7 | Random crop | Massive scale, diverse classes |
| SKU-110K | Random crop (non-target classes) | Hard negatives — same retail domain |
| DTD (Describable Textures) | Full image | Pure texture/surface, no objects |
| Places365 | Random crop | Scene-level background diversity |

Each source is individually `enabled: true/false` in config.

### 8.3 Batch Mix Ratio

Per batch, negatives are drawn from three pools at a configurable ratio:

```
labeled_neg : synthetic_cutpaste : open_source
    0.5     :        0.3         :     0.2
```

Adjustable in `config.yaml → extra_negatives.mix_ratio`.

---

## 10. Metrics & Evaluation

All metrics computed **per class** and **macro/weighted aggregated**. Saved to the run folder.

| Metric | Notes |
|---|---|
| Accuracy | Overall and per-class |
| Precision | Per-class, macro, weighted |
| Recall | Per-class, macro, weighted |
| F1 Score | Per-class, macro, weighted |
| False Positive Rate | Key metric for negative budget experiment |
| Confusion Matrix | Full N×N including negative class |
| ROC / AUC | Per-class |
| PR Curve | Per-class (critical for imbalanced evaluation) |
| Loss Curve | Train + val per epoch |

Decision threshold: **fixed at 0.5** for POC. Per-class threshold tuning deferred.

---

## 11. Plots (seaborn / matplotlib)

All plots saved as `.png` inside the timestamped run folder.

| File | Content |
|---|---|
| `confusion_matrix.png` | Normalized annotated heatmap (seaborn) |
| `loss_curve.png` | Train vs val loss per epoch |
| `f1_per_class.png` | Bar chart, per-class F1 |
| `pr_curve.png` | Per-class precision-recall curves |
| `roc_curve.png` | Per-class ROC + AUC |
| `neg_budget_experiment.png` | FP rate & F1 vs negative sample count (across runs) |
| `class_split_experiment.png` | FP rate on unseen classes vs number of train negative classes |

---

## 12. Experiment Tracking & Logging

Every training run creates a timestamped folder — no external dependency (no wandb).

```
runs/
└── 20240601_143022_focal_1to2/
    ├── config.yaml               # exact config snapshot used for this run
    ├── train.log                 # full training log (metrics every N steps + epoch summary)
    ├── metrics.csv               # per-epoch: loss, acc, f1, precision, recall (train + val)
    ├── checkpoints/
    │   ├── best_model.pth        # saved at best val F1 macro
    │   ├── epoch_048.pth         # last N checkpoints (rolling)
    │   └── epoch_049.pth
    ├── confusion_matrix.png
    ├── loss_curve.png
    ├── f1_per_class.png
    ├── pr_curve.png
    ├── roc_curve.png
    └── run_summary.json          # final test metrics + config snapshot
```

Run folder naming: `YYYYMMDD_HHMMSS_<loss_type>_<pos_neg_ratio>`

**Log file format (`train.log`):**
```
[2024-06-01 14:30:22] Sanity check passed (2 batches)
[2024-06-01 14:30:25] Epoch 1/50 | Step 50  | loss: 0.412 | acc: 0.731
[2024-06-01 14:30:28] Epoch 1/50 | Step 100 | loss: 0.387 | acc: 0.754
[2024-06-01 14:30:31] Epoch 1 complete | val_f1: 0.761 | val_loss: 0.341 ← best
[2024-06-01 14:45:10] Early stopping triggered — no improvement for 10 epochs
```

---

## 13. Device Compatibility

| Environment | Device | Purpose |
|---|---|---|
| Mac (local) | MPS or CPU | Sanity check, small data subset |
| Google Colab | CUDA (T4 / A100) | Full training run |

Auto-detection (no code changes between environments):
```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

`batch_size` and `num_workers` should be configurable in `config.yaml` to accommodate
memory differences between Colab GPU and Mac CPU.

---

## 14. Suggested Project Structure

```
sku_classifier/
├── setup.py                    # installable package — enables sku-clf CLI entry point
├── config.yaml                 # default config (user copies and edits)
├── sku_clf/
│   ├── __main__.py             # python -m sku_clf entry point
│   ├── cli.py                  # CLI argument parsing (argparse); flags override yaml
│   ├── train.py                # training loop
│   ├── evaluate.py             # evaluation on test set
│   ├── infer.py                # single image inference (Mode A + B)
│   ├── infer_batch.py          # batch inference (Mode A + B)
│   ├── data/
│   │   ├── dataset.py          # YOLOLabelDataset — reads labels, crops ROI, augments
│   │   ├── sampler.py          # BatchRatioSampler (1:2 pos:neg enforcement)
│   │   ├── augment.py          # all augmentations incl. occlusion, cutpaste, mixup/cutmix
│   │   └── extra_neg.py        # synthetic CutPaste + open-source negative loaders
│   ├── models/
│   │   ├── backbone.py         # EfficientNetV2 + AdaptiveAvgPool head
│   │   └── loss.py             # FocalLoss, BCE + label smoothing
│   └── utils/
│       ├── metrics.py          # all classification metrics (per-class + aggregated)
│       ├── plots.py            # seaborn/matplotlib plot functions
│       ├── logger.py           # timestamped run folder, CSV + log file
│       └── io.py               # zip extraction, directory/file resolution
└── runs/                       # auto-created per training run
```

**Installation:**
```bash
pip install -e .
```

**Usage — both styles work identically:**
```bash
# CLI entry point
sku-clf train  --config config.yaml --data dataset.zip --epochs 30
sku-clf eval   --config config.yaml --checkpoint runs/.../best_model.pth
sku-clf infer  --image img.jpg --boxes img.txt --checkpoint runs/.../best_model.pth
sku-clf infer-batch --images_dir imgs/ --boxes_dir boxes/ --checkpoint runs/.../best_model.pth

# Module style
python -m sku_clf train  --config config.yaml --data dataset.zip
python -m sku_clf infer  --crops crop.jpg --checkpoint runs/.../best_model.pth
```

**Flag override examples:**
```bash
sku-clf train --config config.yaml --lr 0.0001 --epochs 100 --loss focal
# --flag values take priority over config.yaml
```

**Dataset input — zip or directory, both supported:**
```bash
sku-clf train --data dataset.zip      # auto-extracted before training
sku-clf train --data dataset/         # directory used directly
```
Zip auto-extraction looks for `images/` and `labels/` inside the archive.

---

## 15. Inference Pipeline

Two input modes supported by both `infer` and `infer-batch` commands:

### Mode A — Image + YOLO Boxes File
Provide a raw image and a YOLO-format `.txt` boxes file.
Class index in the boxes file is **ignored** — all boxes are treated as ROIs.

```
sku-clf infer --image img.jpg --boxes img.txt --checkpoint runs/.../best_model.pth
```

Box file format (class index ignored):
```
<ignored_class_id> <x_center> <y_center> <width> <height>
```

### Mode B — Pre-Cropped Images
Provide already-cropped ROI images directly — skips the bbox expansion step entirely.

```
sku-clf infer --crops path/to/crop.jpg --checkpoint runs/.../best_model.pth
sku-clf infer-batch --crops_dir path/to/crops/ --checkpoint runs/.../best_model.pth
```

### 14.1 Single Inference (`infer`)

```
sku-clf infer --image img.jpg --boxes img.txt --checkpoint runs/.../best_model.pth --config config.yaml
```

**Output:** prints `{ bbox, class_name, confidence }` per ROI to stdout.
Optionally saves annotated image with `--save_vis output.jpg`.

### 14.2 Batch Inference (`infer-batch`)

```
sku-clf infer-batch --images_dir path/to/images/ --boxes_dir path/to/boxes/ \
    --checkpoint runs/.../best_model.pth --output_dir results/
```

**Output per image:** `.json` file with all ROI predictions
**Summary:** `batch_summary.csv` — one row per image, count per detected class

---

## 16. Open Questions / Future Work

- Stage 1 (General SKU Detector) training spec — out of scope for POC
- Per-class confidence threshold tuning post-POC
- Hard negative mining as follow-up if negative budget experiment shows insufficient FP suppression
- TorchScript / ONNX export for edge deployment
- Multi-label support per ROI (currently single positive class per box assumed)
