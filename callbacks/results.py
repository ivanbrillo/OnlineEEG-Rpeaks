def get_results():
    """
    Retrieve the most recent inference results for display.

    Returns:
        dict: {
            "global_metrics": dict | None,
            "per_subject": list[dict],
            "plots": {"per_subject": dict[str, list[str]]},
            "subject_details": dict[str, dict]
        }

    Raises:
        NotImplementedError: If no inference has been run yet.
    """
    from callbacks.inference import _inference_results

    if _inference_results is None:
        raise NotImplementedError("No inference results available yet.")

    return _inference_results
