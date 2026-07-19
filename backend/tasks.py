"""In-memory batch prediction task registry.

Single-user local app: a plain dict + lock + daemon threads is sufficient.
Each task runs services.prediction.predict_batch in a background thread and
stores JSON-safe results (PredictionResult dicts and/or per-row error dicts).
"""

import logging
import threading
import time
import uuid

from backend import prediction_result_to_dict

logger = logging.getLogger(__name__)


class BatchTaskRegistry:
    """Tracks background batch prediction tasks."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: dict[str, dict] = {}

    def create(self, smiles_list: list[str], mode: str) -> str:
        """Start a background batch task; returns its task_id."""
        from services.prediction import predict_batch

        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._tasks[task_id] = {
                "status": "running",
                "total": len(smiles_list),
                "done": 0,
                "results": None,
                "error": None,
                "created_at": time.time(),
            }

        def _worker():
            try:
                results = predict_batch(smiles_list, mode=mode)
                safe_results = [prediction_result_to_dict(r) for r in results]
                with self._lock:
                    task = self._tasks.get(task_id)
                    if task is not None:
                        task["results"] = safe_results
                        task["done"] = len(smiles_list)
                        task["status"] = "done"
            except Exception as e:  # predict_batch isolates row errors; this is catastrophic failure
                logger.exception("Batch task %s failed", task_id)
                with self._lock:
                    task = self._tasks.get(task_id)
                    if task is not None:
                        task["status"] = "error"
                        task["error"] = str(e)

        threading.Thread(target=_worker, daemon=True, name=f"batch-{task_id}").start()
        return task_id

    def get(self, task_id: str) -> dict | None:
        """Public view: {status, progress: {done, total}, results, error}."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return {
                "status": task["status"],
                "progress": {"done": task["done"], "total": task["total"]},
                "results": task["results"],
                "error": task["error"],
            }


# Shared singleton used by the routes layer.
registry = BatchTaskRegistry()
