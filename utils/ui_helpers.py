import json


def allowed_file(filename):
    """Return True if the filename has a .zip extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "zip"


def allowed_model_file(filename):
    """Return True if the filename has a .pt extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pt"


def format_metric(value, decimals=4):
    """Format a numeric metric for display. Returns '—' for None or non-numeric values."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def safe_json_parse(text, default=None):
    """Parse a JSON string, returning default on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}
