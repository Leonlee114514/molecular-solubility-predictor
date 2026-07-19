"""DisSolve FastAPI backend package.

Serves the framework-free prediction service (services/) plus analysis,
visualization, molecule search, and AI explanation endpoints for the React
frontend. Independent of Streamlit at runtime.
"""

import dataclasses

import numpy as np


def to_jsonable(obj):
    """Recursively convert dataclasses, numpy scalars/arrays, tuples and NaN
    into JSON-safe Python types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, float) and obj != obj:  # NaN is not valid JSON
        return None
    return obj


def prediction_result_to_dict(result):
    """services.prediction.PredictionResult (or a batch error dict) -> JSON-safe dict."""
    return to_jsonable(result)
