"""Model training and inference modules."""

from ml.src.models.xgboost_model import XGBoostTrainer, XGBoostPrediction
from ml.src.models.calibration import Calibrator, CalibrationCoefficients

# MLP is imported lazily — it requires torch which is a heavy dependency
# not always available (e.g., in the backend API server).
# Use: from ml.src.models.mlp_model import MLPTrainer, MLPPrediction


def _get_mlp_classes():
    """Lazy accessor for MLP classes (requires torch)."""
    from ml.src.models.mlp_model import MLPTrainer, NBA_MLP, MLPPrediction
    return MLPTrainer, NBA_MLP, MLPPrediction
