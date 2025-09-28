# vit_tune

## Configuration highlights

- **Neptune logging** – enable via the `neptune` section. Provide `project`, optional `api_token` (or rely on `NEPTUNE_API_TOKEN` env var), `tags`, and `name`. When enabled the training pipeline logs metrics, confusion matrices, dataset statistics, and checkpoint info to Neptune.
- **Model cache directory** – control where pretrained weights are downloaded from Hugging Face/TIMM with `model.models_dir`.
- **Checkpointing** – configure with the `checkpoint` section. Specify output directory, frequency (`save_every_epochs`), the metric to monitor (`monitor`), and whether to maximize or minimize it (`mode`). Best checkpoints and periodic checkpoints are saved there.
- **Class filtering** – classes with fewer than `data.min_samples_per_class` samples are dropped. The retained class distribution is saved to `class_stats.json` and, when Neptune logging is enabled, uploaded as metadata.
- **Metrics** – weighted F1 score and normalized confusion matrices are computed for validation/test splits. Confusion matrices are also logged to Neptune.

Refer to `config.yaml` for a complete example configuration.
