import os
import io
import json
import tempfile
import threading
import traceback
import zipfile
from datetime import datetime, timezone

from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_file
from werkzeug.utils import secure_filename

from callbacks.preprocessing import handle_training_upload, get_saved_subjects, reset_dataset
from callbacks.training import start_training
from callbacks.model import save_trained_model, list_available_models
from callbacks.inference import run_inference, generate_window_plot, recompute_metrics, build_inference_export
from callbacks.results import get_results
from utils.ui_helpers import allowed_file, allowed_model_file, format_metric, safe_json_parse

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-in-production"
app.config["MAX_CONTENT_LENGTH"] = 10000 * 1024 * 1024  # 10GB upload limit

# Use the OS temp directory so uploaded files never land inside the project
# folder, which would trigger Flask's file watcher and restart the server.
UPLOAD_FOLDER = tempfile.gettempdir()

# ── In-process state ──────────────────────────────────────────────────────────
_state = {
    "training_logs": "",
    "training_status": "",
    "training_running": False,
    "training_metrics": None,
    "training_plot": None,
    "last_model_name": "",
    "last_config": {},
    "inference_logs": "",
    "inference_status": "",
    "inference_running": False,
    "inference_summary": None,
    "inference_plot_files": [],
    "inference_results": None,
    "training_form_values": None,
    "inference_form_values": None,
}
_state_lock = threading.Lock()

import json

DEFAULT_HYPERPARAMS = json.dumps({
    "general": {
        "SEED": 1234,
        "BATCH_SIZE": 32,
        "NUM_EPOCHS": 50
    },
    "optimization": {
        "LEARN_RATE": 0.001,
        "WEIGHT_DECAY": 0.0001,
        "CLIP_NORM": 1
    },
    "scheduler": {
        "SCHED_FACTOR": 0.7,
        "SCHED_PATIENCE": 5
    },
    "early_stopping": {
        "EARLY_STOP_PATIENCE": 15,
        "MIN_EPOCH_IMPROVEMENT": 3
    },
    "model": {
        "dim_feedforward": 512,
        "num_layers": 8,
        "num_heads": 4,
        "drop_rate": 0.1,
        "skip_type": "SE",
        "conv_type": "InceptionSE",
        "skip_concat": True
    },
    "evaluation": {
        "height_threshold": -0.4,
        "min_dist": 200,
        "tol_ms": 150,
        "prominence": 0.035
    },
    "augmentation": {
        "use_augmentation": False,
        "warp_factor_range_min": 0.85,
        "warp_factor_range_max": 1.15,
        "n_augmented_per_segment": 5
    },
    "channels_64_map": [
        "E22", "E17", "E9", "E19", "E4", "E26", "E2", "E16", "E12", "E5",
        "E24", "E124", "E27", "E111", "E33", "E122", "E11", "E13", "E112", "E28",
        "E117", "E31", "E105", "E6", "E29", "E110", "E36", "E104", "E41", "E103",
        "E54", "E34", "E123", "E45", "E108", "E46", "E102", "E42", "E93", "E43",
        "E80", "E47", "E98", "E55", "E53", "E86", "E52", "E92", "E51", "E97",
        "E58", "E96", "E62", "E60", "E85", "E65", "E90", "E72", "E70", "E83",
        "E75", "E82", "E57", "E100"
    ]
}, indent=2)

# ── Template helper ────────────────────────────────────────────────────────────

app.jinja_env.globals["format_metric"] = format_metric

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Upload ─────────────────────────────────────────────────────────────────────

@app.route("/upload", methods=["GET", "POST"])
def upload():
    upload_result = None

    if request.method == "POST":
        file = request.files.get("zip_file")

        # Validate file
        if not file or file.filename == "":
            flash("No ZIP file selected.", "error")
            return redirect(url_for("upload"))
        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload a .zip archive.", "error")
            return redirect(url_for("upload"))

        # Save file
        filename = secure_filename(file.filename)
        zip_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(zip_path)

        # Delegate to callback
        try:
            result = handle_training_upload(zip_path)
            upload_result = result
            flash(result.get("message", "Upload successful."), "success")
        except (zipfile.BadZipFile, zipfile.LargeZipFile):
            flash("Invalid ZIP file. Please upload a valid .zip archive.", "error")
        except NotImplementedError:
            flash("handle_training_upload: not yet implemented.", "info")
        except ValueError as e:
            traceback.print_exc()
            flash(str(e), "error")
        except Exception as e:
            traceback.print_exc()
            flash(f"Upload error: {e}", "error")

    return render_template("upload.html", upload_result=upload_result)


# ── Training ───────────────────────────────────────────────────────────────────

def _run_training_thread(config, excluded_subjects):
    """Background thread that runs start_training and streams logs via log_fn."""
    with _state_lock:
        _state["training_running"] = True
        _state["training_logs"] = ""
        _state["training_status"] = "Running…"
        _state["training_metrics"] = None
        _state["training_plot"] = None

    def log_fn(line):
        with _state_lock:
            _state["training_logs"] += line + "\n"

    try:
        result = start_training(config, excluded_subjects, log_fn=log_fn)
        with _state_lock:
            if result.get("logs"):
                _state["training_logs"] += result["logs"]
            _state["training_status"] = result.get("status", "Complete")
            _state["training_metrics"] = result.get("metrics")
            model_name = config.get("model_name", "model")
            plot_filename = f"{model_name}_training_curves.png"
            if os.path.exists(os.path.join("static", "plots", plot_filename)):
                _state["training_plot"] = plot_filename
    except NotImplementedError:
        with _state_lock:
            _state["training_status"] = "start_training: not yet implemented."
    except Exception as e:
        traceback.print_exc()
        with _state_lock:
            _state["training_status"] = f"Error: {e}"
    finally:
        with _state_lock:
            _state["training_running"] = False


@app.route("/training")
def training():
    subjects = []
    try:
        subjects = get_saved_subjects()
    except NotImplementedError:
        pass  # silently show empty list; user will upload data first
    except Exception as e:
        flash(f"Could not load subjects: {e}", "error")

    form_values = _state.get("training_form_values")

    return render_template(
        "training.html",
        subjects=subjects,
        default_hyperparams=DEFAULT_HYPERPARAMS,
        training_logs=_state["training_logs"],
        training_status=_state["training_status"],
        training_running=_state["training_running"],
        training_metrics=_state["training_metrics"],
        training_plot=_state["training_plot"],
        last_model_name=_state["last_model_name"],
        form_values=form_values,
    )


@app.route("/training/status")
def training_status_api():
    """JSON endpoint polled by the training page to stream live log updates."""
    with _state_lock:
        data = {
            "logs": _state["training_logs"],
            "running": _state["training_running"],
            "status": _state["training_status"],
            "metrics": _state["training_metrics"],
            "plot": _state["training_plot"],
        }
    return jsonify(data)


@app.route("/training/start", methods=["POST"])
def training_start():
    if _state["training_running"]:
        flash("Training is already running.", "info")
        return redirect(url_for("training"))

    form_values = {
        "model_name": request.form.get("model_name", "").strip(),
        "channels": request.form.get("channels", "64"),
        "overlap_percentage": request.form.get("overlap_percentage", "0"),
        "butterworth": bool(request.form.get("butterworth")),
        "f_min": request.form.get("f_min", "5"),
        "f_max": request.form.get("f_max", "22.5"),
        "validation_ratio": request.form.get("validation_ratio", "0.3"),
        "time_window_length": request.form.get("time_window_length", "10"),
        "hyperparams": request.form.get("hyperparams", "{}"),
        "excluded_subjects": request.form.getlist("excluded_subjects"),
    }
    _state["training_form_values"] = form_values

    model_name = request.form.get("model_name", "").strip()
    if not model_name:
        flash("Model name is required.", "error")
        return redirect(url_for("training"))

    hyperparams = safe_json_parse(request.form.get("hyperparams", "{}"))
    if not hyperparams:
        flash("Invalid JSON in the hyperparameters field.", "error")
        return redirect(url_for("training"))

    butterworth = bool(request.form.get("butterworth"))

    # Inject UI-controlled fields into hyperparams so training.py finds them normally
    hyperparams.setdefault("general", {})["VALIDATION_RATIO"] = float(request.form.get("validation_ratio", 0.3))
    hyperparams.setdefault("windowing", {})["time_window_length"] = float(request.form.get("time_window_length", 10))
    hyperparams.setdefault("windowing", {})["overlap_percentage"] = float(request.form.get("overlap_percentage", 0.0))

    config = {
        "model_name": model_name,
        "channels": int(request.form.get("channels", 64)),
        "butterworth": butterworth,
        "f_min": float(request.form.get("f_min", 5.0)) if butterworth else None,
        "f_max": float(request.form.get("f_max", 22.5)) if butterworth else None,
        "hyperparams": hyperparams,
    }
    excluded_subjects = request.form.getlist("excluded_subjects")

    _state["last_model_name"] = model_name
    _state["last_config"] = config

    t = threading.Thread(target=_run_training_thread, args=(config, excluded_subjects), daemon=True)
    t.start()

    flash("Training started. Logs will update below.", "info")
    return redirect(url_for("training"))


@app.route("/training/download")
def training_download():
    model_name = _state.get("last_model_name", "")
    if not model_name:
        flash("No trained model available. Run training first.", "error")
        return redirect(url_for("training"))
    checkpoint = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "saved_models", f"{model_name}_best_model.pt"
    )
    if not os.path.exists(checkpoint):
        flash("Checkpoint file not found. Run training first.", "error")
        return redirect(url_for("training"))
    return send_file(checkpoint, as_attachment=True,
                     download_name=f"{model_name}_best_model.pt")


@app.route("/training/save", methods=["POST"])
def training_save():
    model_name = request.form.get("model_name", "").strip()
    if not model_name:
        flash("Model name is required to save.", "error")
        return redirect(url_for("training"))

    try:
        result = save_trained_model(model_name, _state["last_config"])
        flash(result.get("message", "Model saved successfully."), "success")
    except NotImplementedError:
        flash("save_trained_model: not yet implemented.", "info")
    except Exception as e:
        flash(f"Save error: {e}", "error")

    return redirect(url_for("training"))


# ── Inference ──────────────────────────────────────────────────────────────────

def _run_inference_thread(model_id, zip_path=None, included_subjects=None):
    """Background thread that runs run_inference and streams logs via log_fn."""
    with _state_lock:
        _state["inference_running"] = True
        _state["inference_logs"] = ""
        _state["inference_status"] = "Running…"
        _state["inference_summary"] = None
        _state["inference_plot_files"] = []

    def log_fn(line):
        with _state_lock:
            _state["inference_logs"] += line + "\n"

    try:
        result = run_inference(
            model_id,
            zip_path=zip_path,
            included_subjects=included_subjects,
            log_fn=log_fn,
        )
        with _state_lock:
            _state["inference_status"] = result.get("status", "complete")
            _state["inference_summary"] = result.get("summary")
            _state["inference_plot_files"] = result.get("plot_files", [])
    except ValueError as e:
        with _state_lock:
            _state["inference_status"] = f"Error: {e}"
    except Exception as e:
        traceback.print_exc()
        with _state_lock:
            _state["inference_status"] = f"Error: {e}"
    finally:
        with _state_lock:
            _state["inference_running"] = False


@app.route("/inference/status")
def inference_status_api():
    """JSON endpoint polled by the inference page to stream live log updates."""
    with _state_lock:
        data = {
            "logs": _state["inference_logs"],
            "running": _state["inference_running"],
            "status": _state["inference_status"],
        }
    return jsonify(data)


@app.route("/inference", methods=["GET", "POST"])
def inference():
    models = []
    subjects = []
    try:
        models = list_available_models()
    except NotImplementedError:
        pass  # silently show empty dropdown
    except Exception as e:
        flash(f"Could not load models: {e}", "error")

    try:
        subjects = get_saved_subjects()
    except NotImplementedError:
        pass
    except Exception as e:
        flash(f"Could not load subjects: {e}", "error")

    form_values = _state.get("inference_form_values") or {}

    if request.method == "POST":
        if _state["inference_running"]:
            flash("Inference is already running.", "info")
            return redirect(url_for("inference"))

        model_id = request.form.get("model_id", "").strip()
        data_source = request.form.get("data_source", "zip").strip().lower()
        pt_file = request.files.get("pt_file")
        zip_file = request.files.get("zip_file")
        included_subjects = request.form.getlist("included_subjects")

        _state["inference_form_values"] = {
            "model_id": model_id,
            "data_source": data_source,
            "included_subjects": included_subjects,
        }

        # Resolve model source: uploaded .pt takes priority over dropdown
        if pt_file and pt_file.filename:
            if not allowed_model_file(pt_file.filename):
                flash("Invalid model file. Please upload a .pt file.", "error")
                return redirect(url_for("inference"))
            pt_filename = secure_filename(pt_file.filename)
            pt_path = os.path.join(UPLOAD_FOLDER, pt_filename)
            pt_file.save(pt_path)
            model_id = pt_path  # absolute path signals manual upload
        elif not model_id:
            flash("Please select a saved model or upload a .pt file.", "error")
            return redirect(url_for("inference"))

        zip_path = None
        if data_source == "cached":
            if not included_subjects:
                flash("Select at least one cached subject or switch to ZIP upload.", "error")
                return redirect(url_for("inference"))
        else:
            if not zip_file or zip_file.filename == "":
                flash("No ZIP file selected.", "error")
                return redirect(url_for("inference"))
            if not allowed_file(zip_file.filename):
                flash("Invalid file type. Please upload a .zip archive.", "error")
                return redirect(url_for("inference"))

            filename = secure_filename(zip_file.filename)
            zip_path = os.path.join(UPLOAD_FOLDER, filename)
            zip_file.save(zip_path)

        t = threading.Thread(
            target=_run_inference_thread,
            args=(model_id, zip_path, included_subjects if data_source == "cached" else None),
            daemon=True,
        )
        t.start()

        flash("Inference started. Logs will update below.", "info")
        return redirect(url_for("inference"))

    # Load stored results (from previous or current inference run)
    results_data = None
    try:
        results_data = get_results()
    except NotImplementedError:
        pass  # no results yet — page will show placeholder
    except Exception as e:
        flash(f"Could not load results: {e}", "error")

    return render_template(
        "inference.html",
        models=models,
        subjects=subjects,
        inference_form_values=form_values,
        inference_logs=_state["inference_logs"],
        inference_status=_state["inference_status"],
        inference_running=_state["inference_running"],
        inference_summary=_state["inference_summary"],
        results=results_data,
    )


# ── Inference AJAX plot endpoint ───────────────────────────────────────────────

@app.route("/inference/plot/<subject_id>/<int:window_idx>")
def inference_plot(subject_id, window_idx):
    """Return JSON with the URL of the requested inference window plot."""
    try:
        plot_fname = generate_window_plot(subject_id, window_idx)
        return jsonify({"plot_url": f"/static/plots/{plot_fname}"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Plot generation failed: {e}"}), 500


@app.route("/inference/recompute", methods=["POST"])
def inference_recompute():
    """Re-run peak extraction and metric computation with new eval hyperparameters."""
    new_ep = {
        "height_threshold": request.form.get("height_threshold", type=float),
        "min_dist":         request.form.get("min_dist", type=int),
        "tol_ms":           request.form.get("tol_ms", type=float),
        "prominence":       request.form.get("prominence", type=float),
    }
    # Drop None values so recompute_metrics falls back to stored defaults
    new_ep = {k: v for k, v in new_ep.items() if v is not None}
    try:
        recompute_metrics(new_ep)
        flash("Metrics recomputed with new evaluation parameters.", "success")
    except ValueError as e:
        flash(str(e), "error")
    except Exception as e:
        flash(f"Recompute error: {e}", "error")
    return redirect(url_for("inference"))


@app.route("/inference/export/json")
def inference_export_json():
    """Download a JSON export with per-subject and per-segment inference details."""
    try:
        payload = build_inference_export()
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("inference"))
    except Exception as e:
        flash(f"Export error: {e}", "error")
        return redirect(url_for("inference"))

    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_name = f"inference_export_{ts}.json"
    out_bytes = json.dumps(payload, indent=2).encode("utf-8")

    return send_file(
        io.BytesIO(out_bytes),
        mimetype="application/json",
        as_attachment=True,
        download_name=out_name,
    )


# ── Reset ──────────────────────────────────────────────────────────────────────

@app.route("/reset", methods=["GET", "POST"])
def reset():
    if request.method == "POST":
        try:
            result = reset_dataset()
            # Clear in-process state too
            for key in _state:
                _state[key] = "" if isinstance(_state[key], str) else None
            _state["inference_plot_files"] = []
            flash(result.get("message", "Dataset reset successfully."), "success")
        except NotImplementedError:
            flash("reset_dataset: not yet implemented.", "info")
        except Exception as e:
            flash(f"Reset error: {e}", "error")
        return redirect(url_for("index"))

    # GET: show confirmation page
    return render_template("reset.html")


# ── Results (redirect to inference page) ──────────────────────────────────────

@app.route("/results")
def results():
    return redirect(url_for("inference"))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
