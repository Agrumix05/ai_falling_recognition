"""MiniRocket and ordinal binary-decomposition models used by the benchmark."""
from __future__ import annotations

import numpy as np


def make_transformer():
    """Create the same full MiniRocket multivariate transformer as the project."""
    import config as cfg
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

    return MiniRocketMultivariate(
        num_kernels=cfg.MINIROCKET_NUM_KERNELS,
        random_state=cfg.SEED,
        n_jobs=1,
    )


def make_multiclass_classifier():
    """Create the nominal MiniRocket Ridge classifier used by the project."""
    from sklearn.linear_model import RidgeClassifierCV

    return RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))


def make_binary_classifier(target: np.ndarray):
    """Create one binary Ridge classifier for an ordinal threshold."""
    return make_multiclass_classifier()


def fit_obd(x_train, ranked_y, n_classes, binary_estimator_factory):
    """Fit K-1 binary classifiers using the cumulative ordinal decomposition."""
    ranked_y = np.asarray(ranked_y, dtype=int)
    models = []
    for threshold in range(n_classes - 1):
        binary_y = (ranked_y > threshold).astype(int)
        model = binary_estimator_factory(binary_y)
        model.fit(x_train, binary_y)
        models.append(model)
    return models


def predict_obd(models, x_eval, n_classes):
    """Convert binary threshold scores into monotone ordinal probabilities."""
    if len(models) != n_classes - 1:
        raise ValueError("OBD requires exactly n_classes - 1 binary models")
    scores = np.column_stack([model.decision_function(x_eval) for model in models])
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    exceedance = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
    exceedance = np.minimum.accumulate(exceedance, axis=1)
    probabilities = np.empty((exceedance.shape[0], n_classes), dtype=float)
    probabilities[:, 0] = 1.0 - exceedance[:, 0]
    probabilities[:, 1:-1] = exceedance[:, :-1] - exceedance[:, 1:]
    probabilities[:, -1] = exceedance[:, -1]
    probabilities = np.clip(probabilities, 0.0, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities /= np.where(row_sums > 0.0, row_sums, 1.0)
    predicted_ranks = probabilities.argmax(axis=1)
    return probabilities, predicted_ranks
