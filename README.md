# vit_tune

A configuration-driven training pipeline for fine-tuning Vision Transformer (ViT) classifiers on custom image datasets. The project wraps together data indexing, stratified splitting, augmentation, and modern training utilities (AMP, cosine schedules, gradient clipping, Neptune logging) into a single script-driven workflow.

## Key features

- **Drop-in dataset support** – point the trainer at a directory of images named `<class>_<id>.<ext>` and the indexer discovers classes automatically.
- **Stratified data splitting** – configurable train/validation/test ratios with a minimum examples-per-class filter to drop underrepresented categories.
- **Transformer backbones from TIMM** – create any ViT compatible with [`timm.create_model`](https://rwightman.github.io/pytorch-image-models/) and optionally load pretrained weights.
- **Production-friendly training loop** – mixed precision (AMP), gradient clipping, cosine learning-rate schedules with warmup, and checkpoint management.
- **Detailed metrics** – accuracy, weighted F1, and normalized confusion matrices logged to disk (and to Neptune when enabled).
- **Reproducible experiments** – all behaviour is described in a YAML configuration file, with seeds propagated to PyTorch and Python.

## Getting started

### 1. Create and activate an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # create this from your own dependency lock, see below
```

> The project relies on PyTorch, `timm`, `scikit-learn`, and (optionally) `neptune`. If you do not have a lock file, install the core packages manually:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # select the wheel for your CUDA version
> pip install timm scikit-learn pyyaml tqdm neptune matplotlib
> ```

### 2. Arrange your dataset

The indexer expects images inside `data.dataset_dir` to follow the pattern `<class-name>_<sample-id>.<extension>`. All common image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) are supported. Example layout:

```
/your-dataset
├── cat_001.jpg
├── cat_002.jpg
├── dog_001.jpg
└── ...
```

Classes with fewer samples than `data.min_samples_per_class` are ignored. The resulting class histogram is written to `class_stats.json` in the run directory and, if Neptune logging is active, uploaded as run metadata.

### 3. Configure an experiment

Copy `config.example.yaml` and tailor it to your use case:

```bash
cp config.example.yaml config.yaml
```

Each top-level section controls part of the pipeline:

| Section | Highlights |
| ------- | ---------- |
| `data` | Dataset directory, minimum samples per class, image size, worker count, and split ratios. |
| `train` | Training epochs, batch size, learning rate, weight decay, warmup schedule, AMP toggle, gradient clipping, and explicit device override. |
| `model` | Name of the ViT architecture from TIMM, whether to load pretrained weights, stochastic depth (`drop_path_rate`), and an optional cache directory for model weights. |
| `loss` | Choose cross-entropy or focal loss, enable inverse-frequency class weighting, and configure the focal gamma. |
| `checkpoint` | Where to store checkpoints, how often to save, and which metric (`monitor`) to optimize (`max`/`min`). |
| `neptune` | Optional experiment tracking (set `enabled: true`, provide `project`, and either set `api_token` or rely on the `NEPTUNE_API_TOKEN` environment variable). |
| `out` | Output directory for artifacts (`history.json`, `test_metrics.json`, checkpoints) and a global random seed. |

Any CLI device override (`--device`) takes precedence over the value in the configuration file.

### 4. Launch training

```bash
python train.py --config config.yaml
```

Additional flags:

- `--device cuda:0` – force training onto a specific accelerator (falls back to auto-detection when omitted).
- Configure AMP, gradient clipping, warmup epochs, etc. via `config.yaml`.
- When Neptune is enabled, logs include configuration snapshots, metrics, and rendered confusion matrices.

During training the script writes the following to `out.output_dir`:

- `class_stats.json` – class counts after applying the minimum sample filter.
- `labels.json` – index-to-label mapping used for inference.
- `history.json` – per-epoch training/validation metrics.
- `test_metrics.json` – final test split metrics and confusion matrix.
- `checkpoints/` – best model (by monitored metric) plus optional periodic snapshots.

Console output mirrors epoch metrics and prints the final test-set summary.

### 5. Publish the model to the Hugging Face Hub

After training, use `src/utils/push_to_hf.py` to bundle the best checkpoint and
`labels.json` into a Hub repository:

```bash
python -m src.utils.push_to_hf \
  --model-name vit_base_patch16_224 \
  --checkpoint out/checkpoints/best.pt \
  --labels out/labels.json \
  --repo-id your-username/your-model \
  --token hf_your_token_here
```

The helper:

- Re-creates the TIMM architecture to validate the checkpoint.
- Uploads a `pytorch_model.bin`, `config.json`, `labels.json`, and README to the Hub.
- Can target private repositories (`--private`) or update an existing repo (`--allow-existing`).

Provide the drop-path rate used during training via `--drop-path-rate` if it was
enabled in your configuration.

## Development tips

- Enable deterministic behaviour by leaving `out.seed` at a fixed value. The script seeds Python, NumPy (via scikit-learn), and PyTorch.
- To resume from a checkpoint, load the saved state dict into a compatible model manually before calling `run_training` (a resume helper is not yet implemented).
- Configure `model.models_dir` to control where TIMM and Hugging Face caches store pretrained weights (useful on shared clusters with limited home quotas).
- When Neptune logging is active, confusion matrices are logged both as raw data and rendered figures (requires `matplotlib`).

## Troubleshooting

| Symptom | Possible fix |
| ------- | ------------- |
| `RuntimeError: No images found...` | Confirm filenames follow `<class>_<id>.<ext>` and the directory path in `config.yaml` is correct. |
| `All classes were filtered out by min_samples_per_class` | Reduce `data.min_samples_per_class` or add more samples for the affected classes. |
| `CUDA device requested but CUDA is not available` | Remove the device override or install CUDA-enabled PyTorch. |
| Neptune import errors | Install the `neptune` package (`pip install neptune`) or disable Neptune logging. |

## Repository layout

```
├── train.py                # Entry point that wires configuration, data, model, and training loop
├── src/
│   ├── data/               # Dataset indexing, splitting, and PyTorch dataset definitions
│   ├── engine/             # Training loop, schedulers, losses, evaluation utilities
│   ├── loggers/            # Optional Neptune logger wrapper
│   ├── models/             # TIMM model builder
│   ├── transforms.py       # Image augmentations for train/validation
│   └── utils/              # Configuration helpers and the Hugging Face uploader
└── config.example.yaml     # Reference configuration with documented defaults
```

## License

This project is released under the terms of the [MIT License](LICENSE).
