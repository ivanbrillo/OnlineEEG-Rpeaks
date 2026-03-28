import os
import json
from datetime import datetime

import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
SAVED_MODELS_DIR = os.path.join(_PROJECT_ROOT, "saved_models")


def save_trained_model(model_name, config):
    """
    Persist the currently trained model to disk.

    Args:
        model_name (str): Human-readable name for the saved model.
        config (dict): Configuration used to train the model.

    Returns:
        dict: {"status": "success" | "error", "message": str, "model_id": str}
    """
    # Import here to avoid circular import at module load time
    from callbacks.training import _last_trained

    if _last_trained["state_dict"] is None:
        raise ValueError("No trained model available. Run training first.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for use as a directory
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
    model_id = f"{safe_name}_{timestamp}"

    model_dir = os.path.join(SAVED_MODELS_DIR, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # Build config dict (including architecture params stored during training)
    config_data = dict(config)
    config_data["_in_channels"] = _last_trained.get("in_channels")
    config_data["_in_samples"] = _last_trained.get("in_samples")

    # Save weights + config bundled together so the .pt is self-contained
    weights_path = os.path.join(model_dir, "model.pt")
    torch.save({"state_dict": _last_trained["state_dict"], "config": config_data}, weights_path)

    # Also write config.json for human inspection
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config_data, f, indent=2)

    # Save metadata
    metadata = {
        "model_id": model_id,
        "name": model_name,
        "saved_at": datetime.now().isoformat(),
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "status": "success",
        "message": f"Model '{model_name}' saved as '{model_id}'.",
        "model_id": model_id,
    }


def load_model_bundle(model_id_or_path):
    """
    Load a model's state_dict and config from a bundled .pt file.

    Args:
        model_id_or_path (str): Either:
            - An absolute path to a .pt file (manual upload)
            - A model_id (directory name under saved_models/)

    Returns:
        dict: {"state_dict": OrderedDict, "config": dict}

    Raises:
        ValueError: If the file is not found or not in the new bundled format.
    """
    if os.path.isabs(model_id_or_path) and model_id_or_path.endswith(".pt"):
        pt_path = model_id_or_path
    else:
        pt_path = os.path.join(SAVED_MODELS_DIR, model_id_or_path, "model.pt")

    if not os.path.exists(pt_path):
        raise ValueError(f"Model file not found: {pt_path}")

    bundle = torch.load(pt_path, map_location="cpu", weights_only=False)

    if not isinstance(bundle, dict) or "state_dict" not in bundle or "config" not in bundle:
        raise ValueError(
            "Model file is in old format (state_dict only). "
            "Please re-save the model from the Training page to update it."
        )

    return bundle


def list_available_models():
    """
    Return all saved models available for inference.

    Returns:
        list[dict]: Each entry has {"id", "name", "saved_at"}, sorted newest first.
    """
    if not os.path.isdir(SAVED_MODELS_DIR):
        return []

    models = []
    for entry in os.scandir(SAVED_MODELS_DIR):
        if not entry.is_dir():
            continue
        meta_path = os.path.join(entry.path, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            models.append({
                "id": meta["model_id"],
                "name": meta["name"],
                "saved_at": meta["saved_at"],
            })
        except Exception:
            continue

    # Sort newest first
    models.sort(key=lambda m: m["saved_at"], reverse=True)
    return models
