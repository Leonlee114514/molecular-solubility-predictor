"""DisSolve - framework-free unified prediction service package."""

from .prediction import PredictionResult, predict_batch, run_prediction

__all__ = ["PredictionResult", "run_prediction", "predict_batch"]
