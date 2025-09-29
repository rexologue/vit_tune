"""Utilities for publishing trained TIMM classifiers to the Hugging Face Hub.

The module exposes a small CLI that accepts the pieces produced by the
training pipeline (``labels.json`` and a checkpoint containing the model
``state_dict``) and pushes a Hub-ready repository that can be pulled back with
``timm.create_model`` and the stored weights.
"""

from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(sys.path)

import argparse
import json
from pathlib import Path
import shutil
import textwrap
import tempfile
from typing import Dict, List, Tuple

import torch
from huggingface_hub import HfApi

from models.build import build_model


def _load_labels(labels_path: Path) -> Tuple[List[str], Dict[str, int]]:
    """Load an ``idx -> label`` mapping from ``labels.json``.

    The training script stores the mapping as a JSON object where the keys are
    stringified integers.  This helper accepts that format (or an explicit list)
    and returns both the ordered label list and a ``label -> idx`` dictionary.
    """

    with labels_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        try:
            sorted_items = sorted(((int(idx), name) for idx, name in raw.items()), key=lambda item: item[0])
        except ValueError as exc:  # pragma: no cover - defensive, shouldn't happen with the training output
            raise ValueError("labels.json keys must be integer indices") from exc
        labels = [name for _, name in sorted_items]
    elif isinstance(raw, list):
        labels = list(raw)
    else:  # pragma: no cover - defensive, keeps the CLI explicit about bad inputs
        raise TypeError("labels.json must be either a mapping of index to class name or a list of class names")

    label_to_id = {name: idx for idx, name in enumerate(labels)}
    return labels, label_to_id


def _load_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    """Extract the model state dict from the training checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a mapping or contain a 'model'/'state_dict' entry. "
            "Received object of type %s" % type(checkpoint).__name__
        )
    return checkpoint


def _build_card(repo_id: str, model_name: str, labels: List[str]) -> str:
    """Generate a minimal model card for the Hub repository."""

    labels_block = "\n".join(f"- {label}" for label in labels)
    return textwrap.dedent(
        f"""
        ---
        tags:
          - timm
          - vision
        library_name: timm
        ---

        # {repo_id}

        This repository hosts a fine-tuned `{model_name}` classifier exported from the
        [`vit_tune`](https://github.com/) training pipeline.

        ## Labels

        {labels_block}

        ## Usage

        ```python
        import timm
        import torch

        model = timm.create_model(
            "{model_name}",
            num_classes={len(labels)},
            pretrained=False,
        )
        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/{repo_id}/resolve/main/pytorch_model.bin",
            map_location="cpu",
            file_name="{repo_id.replace('/', '--')}.bin",
        )
        model.load_state_dict(state_dict)
        model.eval()
        ```
        """
    ).strip()


def _resolve_token(cli_token: str | None) -> str:
    token = cli_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "A Hugging Face token is required. Pass --token or set the HF_TOKEN/HUGGINGFACE_TOKEN environment variable."
        )
    return token


def _prepare_payload(
    temp_dir: Path,
    model_name: str,
    labels: List[str],
    label_to_id: Dict[str, int],
    state_dict: Dict[str, torch.Tensor],
    labels_source: Path,
) -> None:
    """Write Hub artifacts (weights, config, labels, README) into ``temp_dir``."""

    weights_path = temp_dir / "pytorch_model.bin"
    torch.save(state_dict, weights_path)

    config_path = temp_dir / "config.json"
    config = {
        "architectures": [model_name],
        "model_type": "timm",
        "timm_model_name": model_name,
        "num_labels": len(labels),
        "id2label": {str(idx): label for idx, label in enumerate(labels)},
        "label2id": label_to_id,
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    shutil.copy(labels_source, temp_dir / "labels.json")


def push_to_hub(args: argparse.Namespace) -> None:
    repo_id = args.repo_id
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found: {labels_path}")

    labels, label_to_id = _load_labels(labels_path)
    state_dict = _load_state_dict(checkpoint_path)

    model = build_model(
        args.model_name,
        num_classes=len(labels),
        pretrained=False,
        drop_path_rate=float(args.drop_path_rate),
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing keys: {sorted(missing)}")
        if unexpected:
            problems.append(f"unexpected keys: {sorted(unexpected)}")
        raise RuntimeError(
            "Checkpoint could not be loaded cleanly into the model (" + "; ".join(problems) + ")"
        )

    token = _resolve_token(args.token)
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=args.allow_existing,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _prepare_payload(tmp_path, args.model_name, labels, label_to_id, model.state_dict(), labels_path)
        card = _build_card(repo_id, args.model_name, labels)
        (tmp_path / "README.md").write_text(card, encoding="utf-8")

        commit_message = args.commit_message or "Add TIMM checkpoint"
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="model",
            revision=args.revision,
            commit_message=commit_message,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Push a fine-tuned TIMM model to the Hugging Face Hub")
    parser.add_argument("--model-name", required=True, help="Model identifier understood by timm.create_model")
    parser.add_argument("--checkpoint", required=True, help="Path to the training checkpoint (.pt)")
    parser.add_argument("--labels", required=True, help="Path to labels.json produced during training")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repository identifier (e.g. username/model-name)")
    parser.add_argument("--token", default=None, help="Hugging Face token (defaults to HF_TOKEN env var)")
    parser.add_argument("--drop-path-rate", default=0.0, type=float, help="Drop path rate used during training")
    parser.add_argument("--revision", default="main", help="Target branch on the Hub")
    parser.add_argument("--commit-message", default=None, help="Commit message for the upload")
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private")
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow uploading to an existing repository instead of requiring a fresh repo",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    push_to_hub(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main(sys.argv[1:])
