# Experiments

This folder is for project-specific experiments that should not live inside the reusable
`sku_clf` package.

Run these scripts after installing the package:

```bash
./scripts/setup.sh
source venv/bin/activate
```

## Class Split Experiment

Train on a selected positive/negative class split while treating unselected classes as
evaluation-only negatives:

```bash
python experiments/class_split_experiment.py \
  --config sku_classifier/config.yaml \
  --data /path/to/yolo_dataset \
  --positive BEAR_BRAND_ENRICH_PINK_MILK_170ML \
  --negative-all-remaining
```

The script imports `sku_clf`, builds experiment-specific dataloaders, and calls the package
trainer. The package itself stays focused on normal config-driven training.

## General SKU Negative Experiment

Train every known non-`GENERAL_SKU_SINGLE` class as a positive classifier output, while
sampling only a small fraction of `GENERAL_SKU_SINGLE` boxes as all-zero negatives:

```bash
./scripts/general_sku_negative_experiment.sh /path/to/yolo_dataset \
  --train-gss-fraction 0.1 \
  --pos-neg-ratio 1:1 \
  --val-gss-fraction 1.0 \
  --test-gss-fraction 1.0 \
  --dry-run
```

Defaults:

- image-level split first: `0.7` train, `0.2` val, `0.1` test
- box sampling happens only after each image has already been assigned to a split
- GSS train negative fraction: `0.2` sampled globally within the train split
- batch sampler ratio: `1:1` positive:negative
- train batches sample with replacement when one side is scarce, so the same origin ROI can repeat
  and receive different training augmentations
- GSS val/test negative fraction: `1.0`
- validation/test samples are not upsampled or augmented
- overlap filtering enabled: specific SKU boxes beat overlapping GSS boxes at IoU `0.5`
