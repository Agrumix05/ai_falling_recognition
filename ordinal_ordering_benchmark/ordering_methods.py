"""In-memory, leakage-safe label-ordering benchmark helpers for Jupyter."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


RANDOM_METHOD = "Random order"
NEW_METHODS = (
    RANDOM_METHOD,
    "Confusion-based",
    "Feature-based",
    "Embedding-based",
    "Optimization-based",
)
ALL_METHODS = ("Band power", *NEW_METHODS)
NOMINAL_METHOD = "MiniRocket nominal"


@dataclass
class DatasetContext:
    dataset: str
    X: np.ndarray
    X_features: np.ndarray
    y: np.ndarray
    record_ids: np.ndarray
    subjects: np.ndarray
    lengths: np.ndarray
    class_names: list[str]
    splits: list[tuple[np.ndarray, np.ndarray]]
    bandpower_orders: dict[int, np.ndarray]
    nominal_folds: pd.DataFrame
    nominal_predictions: pd.DataFrame



def load_dataset_context(project_dir: str | Path, dataset: str) -> DatasetContext:
    """Load project inputs and validate matching prior nominal CV artifacts."""
    project_dir = Path(project_dir).resolve()
    if str(project_dir) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(project_dir))
    import advanced_models
    import config as cfg
    import utils

    if dataset not in cfg.DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    raw_result_dir = project_dir / "results" / "advanced"
    import os

    run_setup_error = "Existing nominal MiniRocket results do not match the full stratified CV setup."
    if os.environ.get("FD_QUICK_TEST", "0") == "1":
        raise ValueError("Unset FD_QUICK_TEST to run this benchmark against the full existing folds.")

    X, y, record_ids, subjects, _, lengths = advanced_models._load_data(
        "minirocket", dataset, "raw"
    )
    meta = pd.read_csv(cfg.DATA_DIR / f"raw_{dataset}_meta.csv")
    if not np.array_equal(record_ids.astype(str), meta["record_id"].astype(str).to_numpy()):
        raise ValueError(f"Raw record ordering does not match metadata for {dataset}")
    if lengths is None:
        raise ValueError(f"Raw recording lengths are unavailable for {dataset}")

    feature_df = pd.read_csv(cfg.DATA_DIR / f"features_{dataset}.csv")
    if not np.array_equal(record_ids.astype(str), feature_df["record_id"].astype(str).to_numpy()):
        raise ValueError(f"Feature record ordering does not match raw records for {dataset}")
    if not np.array_equal(y, feature_df["label"].to_numpy(dtype=int)):
        raise ValueError(f"Feature labels do not match raw metadata for {dataset}")
    feature_cols = [
        column for column in feature_df.columns
        if column not in {"record_id", "subject_id", "label", "label_name"}
    ]
    if not feature_cols:
        raise ValueError(f"No extracted signal features found for {dataset}")
    X_features = feature_df[feature_cols].to_numpy(dtype=float)

    splits, mode = utils.make_splitter(dataset, y, subjects)
    splits = list(splits)
    if mode != "stratified" or len(splits) != cfg.N_SPLITS:
        raise ValueError(run_setup_error)

    nominal_prefix = f"minirocket_{dataset}_raw_{mode}"
    nominal_folds = pd.read_csv(raw_result_dir / f"{nominal_prefix}_folds.csv")
    nominal_predictions = pd.read_csv(raw_result_dir / f"{nominal_prefix}_preds.csv")
    class_names = list(cfg.DATASETS[dataset]["class_names"])
    fixed_order = fixed_bandpower_order(dataset, class_names)
    bandpower_orders = {fold: fixed_order.copy() for fold in range(1, len(splits) + 1)}

    _validate_prior_artifacts(
        dataset, splits, record_ids, y, nominal_folds, nominal_predictions
    )
    return DatasetContext(
        dataset=dataset,
        X=np.asarray(X, dtype=np.float32),
        X_features=X_features,
        y=np.asarray(y, dtype=int),
        record_ids=record_ids.astype(str),
        subjects=np.asarray(subjects, dtype=float),
        lengths=np.asarray(lengths, dtype=int),
        class_names=class_names,
        splits=splits,
        bandpower_orders=bandpower_orders,
        nominal_folds=nominal_folds,
        nominal_predictions=nominal_predictions,
    )


def _validate_prior_artifacts(
    dataset: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    record_ids: np.ndarray,
    y: np.ndarray,
    nominal_folds: pd.DataFrame,
    nominal_predictions: pd.DataFrame,
) -> None:
    expected_fold = np.empty(len(y), dtype=int)
    for fold, (_, test_idx) in enumerate(splits, start=1):
        expected_fold[test_idx] = fold
    expected = pd.DataFrame({
        "record_id": record_ids,
        "y_true": y,
        "fold": expected_fold,
    }).sort_values("record_id").reset_index(drop=True)
    required = {"record_id", "y_true", "fold", "y_pred"}
    if not required.issubset(nominal_predictions.columns):
        raise ValueError(f"{dataset} nominal MiniRocket predictions lack expected columns")
    observed = nominal_predictions[["record_id", "y_true", "fold"]].copy()
    observed["record_id"] = observed["record_id"].astype(str)
    observed = observed.sort_values("record_id").reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError(
            f"{dataset} nominal MiniRocket artifacts do not use the same record-to-fold assignment."
        )
    if len(nominal_folds) != len(splits):
        raise ValueError(f"{dataset} nominal fold metrics do not cover all folds")


def fixed_bandpower_order(dataset: str, class_names: list[str]) -> np.ndarray:
    """Return the experiment's fixed a-priori band-power class order."""
    named_orders = {
        "UniMiB": [
            "Walking", "GoingDownS", "GoingUpS", "StandingUpFL", "StandingUpFS",
            "LyingDownFS", "SittingDown", "Jumping", "FallingWithPS", "Running",
            "FallingBackSC", "FallingForw", "Syncope", "FallingRight", "FallingLeft",
            "FallingBack", "HittingObstacle",
        ],
        "UniZgFall": ["OT", "LD", "WU", "MW", "FB", "WD", "FF", "FS", "MJ", "MR"],
    }
    if dataset not in named_orders:
        raise ValueError(f"No fixed band-power order is defined for {dataset}")
    if set(named_orders[dataset]) != set(class_names):
        raise ValueError(f"Fixed band-power names do not match {dataset} classes")
    positions = {name: index for index, name in enumerate(class_names)}
    return np.asarray([positions[name] for name in named_orders[dataset]], dtype=int)


def limit_context(context: DatasetContext, n_folds: int) -> DatasetContext:
    """Trim a context to a small outer-CV prefix for quick smoke tests."""
    if n_folds < 1:
        raise ValueError("n_folds must be positive")
    allowed = set(range(1, n_folds + 1))
    nominal_folds = context.nominal_folds[
        context.nominal_folds["fold"].astype(int).isin(allowed)
    ].copy()
    nominal_predictions = context.nominal_predictions[
        context.nominal_predictions["fold"].astype(int).isin(allowed)
    ].copy()
    return replace(
        context,
        splits=context.splits[:n_folds],
        nominal_folds=nominal_folds,
        nominal_predictions=nominal_predictions,
    )


def class_order_from_distance(distance: np.ndarray) -> np.ndarray:
    """Return a linear class order from a symmetric, non-negative distance matrix."""
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1] or distance.shape[0] < 2:
        raise ValueError(f"Expected a square class-distance matrix, got {distance.shape}")
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("Class distances must be finite and non-negative")
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=True)
    tree = linkage(condensed, method="average", optimal_ordering=True)
    return leaves_list(tree).astype(int)


def feature_distance_order(x_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> np.ndarray:
    """Cluster diagonal-Gaussian feature distributions using Bhattacharyya distance."""
    x_train = np.asarray(x_train, dtype=float)
    medians = np.nanmedian(np.where(np.isfinite(x_train), x_train, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x = np.where(np.isfinite(x_train), x_train, medians)
    scale = x.std(axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    x = (x - x.mean(axis=0)) / scale
    means = np.vstack([x[y_train == label].mean(axis=0) for label in range(n_classes)])
    variances = np.vstack([
        x[y_train == label].var(axis=0, ddof=1) for label in range(n_classes)
    ])
    floor = 1e-8
    variances = np.maximum(variances, floor)
    distances = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            var_i, var_j = variances[i], variances[j]
            avg_var = (var_i + var_j) / 2.0
            log_term = 0.25 * np.sum(
                np.log(np.maximum(avg_var, floor) / np.sqrt(var_i * var_j))
            )
            mean_term = 0.125 * np.sum((means[i] - means[j]) ** 2 / avg_var)
            distances[i, j] = distances[j, i] = max(0.0, float(log_term + mean_term))
    return class_order_from_distance(distances)


def embedding_distance_order(
    x_embedding_train: np.ndarray, y_train: np.ndarray, n_classes: int
) -> np.ndarray:
    """Cluster training-only class centroids in the fitted MiniRocket feature space."""
    centroids = np.vstack([
        np.asarray(x_embedding_train)[y_train == label].mean(axis=0)
        for label in range(n_classes)
    ])
    distances = squareform(pdist(centroids, metric="cosine"))
    distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=0.0)
    return class_order_from_distance(distances)


def random_class_order(n_classes: int, seed: int) -> np.ndarray:
    """Return a reproducible uniform permutation used as an ordering baseline."""
    if n_classes < 2:
        raise ValueError(f"At least two classes are required, got {n_classes}")
    return np.random.RandomState(seed).permutation(n_classes).astype(int)


def confusion_distance_order(
    y_train: np.ndarray,
    oof_predictions: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster classes that are mutually confused in inner-CV nominal predictions."""
    matrix = confusion_matrix(
        y_train, oof_predictions, labels=np.arange(n_classes), normalize="true"
    )
    affinity = (matrix + matrix.T) / 2.0
    distance = np.clip(1.0 - affinity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return class_order_from_distance(distance), matrix


def _rank_targets(y: np.ndarray, order: np.ndarray) -> np.ndarray:
    label_to_rank = np.empty(len(order), dtype=int)
    label_to_rank[order] = np.arange(len(order))
    return label_to_rank[np.asarray(y, dtype=int)]


def _inner_oof_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    seed: int,
    n_splits: int,
    make_transformer: Callable[[], Any],
    make_nominal_classifier: Callable[[], Any],
) -> np.ndarray:
    """Build a training-only nominal confusion matrix from inner OOF predictions."""
    inner_splits = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    ).split(np.zeros(len(y)), y)
    predictions = np.full(len(y), -1, dtype=int)
    for train_idx, valid_idx in inner_splits:
        transformer = make_transformer()
        x_train = transformer.fit_transform(X[train_idx], y[train_idx])
        x_valid = transformer.transform(X[valid_idx])
        estimator = make_nominal_classifier()
        estimator.fit(x_train, y[train_idx])
        predictions[valid_idx] = np.asarray(estimator.predict(x_valid), dtype=int).reshape(-1)
    if np.any(predictions < 0):
        raise AssertionError("Inner OOF predictions do not cover all outer-training records.")
    return predictions


def _order_probabilities(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    order: np.ndarray,
    n_classes: int,
    fit_obd: Callable[..., Any],
    predict_obd: Callable[..., Any],
    binary_estimator_factory: Callable[[np.ndarray], Any],
) -> tuple[np.ndarray, np.ndarray]:
    rank_y = _rank_targets(y_train, order)
    models = fit_obd(
        x_train, rank_y, n_classes, binary_estimator_factory
    )
    probabilities, predicted_ranks = predict_obd(models, x_eval, n_classes)
    predicted_labels = order[predicted_ranks]
    return probabilities, predicted_labels


def optimize_order(
    x_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    x_inner_valid: np.ndarray,
    y_inner_valid: np.ndarray,
    initial_order: np.ndarray,
    n_classes: int,
    seed: int,
    iterations: int,
    fit_obd: Callable[..., Any],
    predict_obd: Callable[..., Any],
    binary_estimator_factory: Callable[[np.ndarray], Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Use bounded simulated annealing, selecting order by inner-validation macro-F1."""
    if iterations < 1:
        raise ValueError("Optimization iterations must be positive.")
    rng = np.random.RandomState(seed)
    cache: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
    trace: list[dict[str, Any]] = []

    def score(order: np.ndarray) -> float:
        key = tuple(int(value) for value in order)
        if key not in cache:
            _, prediction = _order_probabilities(
                x_inner_train,
                y_inner_train,
                x_inner_valid,
                order,
                n_classes,
                fit_obd,
                predict_obd,
                binary_estimator_factory,
            )
            value = float(f1_score(
                y_inner_valid, prediction, labels=np.arange(n_classes),
                average="macro", zero_division=0,
            ))
            cache[key] = (value, prediction)
        return cache[key][0]

    current = np.asarray(initial_order, dtype=int).copy()
    if not np.array_equal(np.sort(current), np.arange(n_classes)):
        raise ValueError("Initial optimization order is not a class permutation.")
    current_score = score(current)
    best, best_score = current.copy(), current_score
    trace.append({"step": 0, "inner_macro_f1": current_score, "order": current.copy()})
    for step in range(1, iterations + 1):
        proposal = current.copy()
        i, j = rng.choice(n_classes, size=2, replace=False)
        proposal[i], proposal[j] = proposal[j], proposal[i]
        proposal_score = score(proposal)
        temperature = 0.05 * (0.01 / 0.05) ** ((step - 1) / max(iterations - 1, 1))
        delta = proposal_score - current_score
        if delta >= 0 or rng.rand() < np.exp(delta / temperature):
            current, current_score = proposal, proposal_score
        if proposal_score > best_score:
            best, best_score = proposal.copy(), proposal_score
        trace.append({
            "step": step,
            "inner_macro_f1": proposal_score,
            "current_macro_f1": current_score,
            "best_macro_f1": best_score,
            "accepted": bool(np.array_equal(current, proposal)),
            "order": proposal.copy(),
        })
    return best, pd.DataFrame(trace)


def _make_inner_context(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_fraction: float,
    seed: int,
    transformer_factory: Callable[[], Any],
) -> dict[str, np.ndarray]:
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=inner_fraction, random_state=seed
    )
    inner_train, inner_valid = next(split.split(np.zeros(len(y_train)), y_train))
    transformer = transformer_factory()
    x_inner_train = transformer.fit_transform(X_train[inner_train], y_train[inner_train])
    x_inner_valid = transformer.transform(X_train[inner_valid])
    return {
        "x_train": x_inner_train,
        "y_train": y_train[inner_train],
        "x_valid": x_inner_valid,
        "y_valid": y_train[inner_valid],
        "train_indices": inner_train,
        "valid_indices": inner_valid,
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    order: np.ndarray,
) -> dict[str, float]:
    true_rank = _rank_targets(y_true, order)
    pred_rank = _rank_targets(y_pred, order)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "ordinal_mae": float(mean_absolute_error(true_rank, pred_rank)),
    }


def run_benchmark(
    context: DatasetContext,
    *,
    transformer_factory: Callable[[], Any],
    nominal_classifier_factory: Callable[[], Any],
    binary_estimator_factory: Callable[[np.ndarray], Any],
    fit_obd: Callable[..., Any],
    predict_obd: Callable[..., Any],
    inner_confusion_splits: int = 3,
    inner_validation_fraction: float = 0.25,
    optimization_iterations: int = 4,
    seed: int = 89,
) -> dict[str, Any]:
    """Run all new outer-fold orderings; keep all results in memory."""
    n_classes = len(context.class_names)
    method_metrics: list[dict[str, Any]] = []
    method_orders: list[dict[str, Any]] = []
    optimization_traces: list[pd.DataFrame] = []
    confusion_matrices: list[np.ndarray] = []
    evaluation_predictions: list[dict[str, Any]] = []

    for fold, (outer_train, outer_test) in enumerate(context.splits, start=1):
        print(
            f"[{context.dataset} | fold {fold}/{len(context.splits)}] "
            f"building training-only orders...",
            flush=True,
        )
        y_train = context.y[outer_train]
        y_test = context.y[outer_test]
        x_raw_train = context.X[outer_train]

        transformer = transformer_factory()
        x_embedding_train = transformer.fit_transform(x_raw_train, y_train)
        x_embedding_test = transformer.transform(context.X[outer_test])

        feature_order = feature_distance_order(
            context.X_features[outer_train], y_train, n_classes
        )
        embedding_order = embedding_distance_order(
            x_embedding_train, y_train, n_classes
        )

        inner_oof = _inner_oof_baseline(
            x_raw_train,
            y_train,
            n_classes,
            seed + 1009 * fold,
            inner_confusion_splits,
            transformer_factory,
            nominal_classifier_factory,
        )
        confusion_order, confusion_matrix_train = confusion_distance_order(
            y_train, inner_oof, n_classes
        )
        confusion_matrices.append(confusion_matrix_train)

        inner = _make_inner_context(
            x_raw_train,
            y_train,
            inner_validation_fraction,
            seed + 2017 * fold,
            transformer_factory,
        )
        optimization_order, trace = optimize_order(
            inner["x_train"],
            inner["y_train"],
            inner["x_valid"],
            inner["y_valid"],
            feature_order,
            n_classes,
            seed + 3011 * fold,
            optimization_iterations,
            fit_obd,
            predict_obd,
            binary_estimator_factory,
        )
        trace.insert(0, "fold", fold)
        trace.insert(0, "dataset", context.dataset)
        optimization_traces.append(trace)

        orders = {
            "Band power": context.bandpower_orders[fold],
            RANDOM_METHOD: random_class_order(n_classes, seed + 4019 * fold),
            "Confusion-based": confusion_order,
            "Feature-based": feature_order,
            "Embedding-based": embedding_order,
            "Optimization-based": optimization_order,
        }
        for method, order in orders.items():
            if not np.array_equal(np.sort(order), np.arange(n_classes)):
                raise AssertionError(f"{method} returned invalid class order on fold {fold}")
            method_orders.extend({
                "dataset": context.dataset,
                "fold": fold,
                "method": method,
                "position": position,
                "label": int(label),
                "class_name": context.class_names[int(label)],
            } for position, label in enumerate(order))

            start = time.perf_counter()
            _, predictions = _order_probabilities(
                x_embedding_train,
                y_train,
                x_embedding_test,
                order,
                n_classes,
                fit_obd,
                predict_obd,
                binary_estimator_factory,
            )
            elapsed = time.perf_counter() - start
            method_metrics.append({
                "dataset": context.dataset,
                "fold": fold,
                "method": method,
                "fit_predict_seconds": elapsed,
                **evaluate_predictions(y_test, predictions, order),
            })
            evaluation_predictions.extend({
                "dataset": context.dataset,
                "fold": fold,
                "method": method,
                "record_id": record_id,
                "y_true": int(true_label),
                "y_pred": int(predicted_label),
            } for record_id, true_label, predicted_label in zip(
                context.record_ids[outer_test], y_test, predictions
            ))

        # Include the existing nominal MiniRocket predictions and score their
        # ordinal MAE relative to this fold's band-power order for reference.
        nominal_frame = context.nominal_predictions[
            context.nominal_predictions["fold"] == fold
        ].set_index("record_id").loc[context.record_ids[outer_test]]
        nominal_pred = nominal_frame["y_pred"].to_numpy(dtype=int)
        evaluation_predictions.extend({
            "dataset": context.dataset,
            "fold": fold,
            "method": NOMINAL_METHOD,
            "record_id": record_id,
            "y_true": int(true_label),
            "y_pred": int(predicted_label),
        } for record_id, true_label, predicted_label in zip(
            context.record_ids[outer_test], y_test, nominal_pred
        ))
        print(
            f"[{context.dataset} | fold {fold}] orders and held-out predictions complete.",
            flush=True,
        )

    metrics = pd.DataFrame(method_metrics)
    orders = pd.DataFrame(method_orders)
    predictions = pd.DataFrame(evaluation_predictions)
    traces = pd.concat(optimization_traces, ignore_index=True)
    return {
        "dataset": context.dataset,
        "metrics": metrics,
        "orders": orders,
        "predictions": predictions,
        "optimization_trace": traces,
        "inner_oof_confusion": np.stack(confusion_matrices),
    }


def run_single_method(
    context: DatasetContext,
    method: str,
    *,
    transformer_factory: Callable[[], Any],
    nominal_classifier_factory: Callable[[], Any],
    binary_estimator_factory: Callable[[np.ndarray], Any],
    fit_obd: Callable[..., Any],
    predict_obd: Callable[..., Any],
    inner_confusion_splits: int = 3,
    inner_validation_fraction: float = 0.25,
    optimization_iterations: int = 4,
    seed: int = 89,
    outer_fold_limit: int | None = None,
) -> dict[str, Any]:
    """Train and test one ordering method independently on all outer folds."""
    if method not in ALL_METHODS:
        raise ValueError(f"Unknown ordering method: {method}")
    n_classes = len(context.class_names)
    metric_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    traces: list[pd.DataFrame] = []
    confusion_matrices: list[np.ndarray] = []

    folds = context.splits if outer_fold_limit is None else context.splits[:outer_fold_limit]
    for fold, (outer_train, outer_test) in enumerate(folds, start=1):
        print(f"[{context.dataset} | {method} | fold {fold}/{len(folds)}]", flush=True)
        x_raw_train = context.X[outer_train]
        y_train = context.y[outer_train]
        y_test = context.y[outer_test]

        transformer = transformer_factory()
        x_embedding_train = transformer.fit_transform(x_raw_train, y_train)
        x_embedding_test = transformer.transform(context.X[outer_test])

        trace = pd.DataFrame()
        if method == "Band power":
            order = context.bandpower_orders[fold]
        elif method == RANDOM_METHOD:
            order = random_class_order(n_classes, seed + 4019 * fold)
        elif method == "Feature-based":
            order = feature_distance_order(context.X_features[outer_train], y_train, n_classes)
        elif method == "Embedding-based":
            order = embedding_distance_order(x_embedding_train, y_train, n_classes)
        elif method == "Confusion-based":
            inner_oof = _inner_oof_baseline(
                x_raw_train, y_train, n_classes, seed + 1009 * fold,
                inner_confusion_splits, transformer_factory, nominal_classifier_factory,
            )
            order, matrix = confusion_distance_order(y_train, inner_oof, n_classes)
            confusion_matrices.append(matrix)
        elif method == "Optimization-based":
            initial_order = feature_distance_order(
                context.X_features[outer_train], y_train, n_classes
            )
            inner = _make_inner_context(
                x_raw_train, y_train, inner_validation_fraction,
                seed + 2017 * fold, transformer_factory,
            )
            order, trace = optimize_order(
                inner["x_train"], inner["y_train"], inner["x_valid"], inner["y_valid"],
                initial_order, n_classes, seed + 3011 * fold, optimization_iterations,
                fit_obd, predict_obd, binary_estimator_factory,
            )
            trace.insert(0, "fold", fold)
            trace.insert(0, "dataset", context.dataset)
            traces.append(trace)
        else:
            raise AssertionError(f"Unhandled method {method}")

        if not np.array_equal(np.sort(order), np.arange(n_classes)):
            raise AssertionError(f"{method} returned invalid class order on fold {fold}")
        order_rows.extend({
            "dataset": context.dataset,
            "fold": fold,
            "method": method,
            "position": position,
            "label": int(label),
            "class_name": context.class_names[int(label)],
        } for position, label in enumerate(order))

        start = time.perf_counter()
        _, predictions = _order_probabilities(
            x_embedding_train, y_train, x_embedding_test, order, n_classes,
            fit_obd, predict_obd, binary_estimator_factory,
        )
        elapsed = time.perf_counter() - start
        metric_rows.append({
            "dataset": context.dataset,
            "fold": fold,
            "method": method,
            "fit_predict_seconds": elapsed,
            **evaluate_predictions(y_test, predictions, order),
        })
        prediction_rows.extend({
            "dataset": context.dataset,
            "fold": fold,
            "method": method,
            "record_id": record_id,
            "y_true": int(true_label),
            "y_pred": int(predicted_label),
        } for record_id, true_label, predicted_label in zip(
            context.record_ids[outer_test], y_test, predictions
        ))

    return {
        "dataset": context.dataset,
        "method": method,
        "metrics": pd.DataFrame(metric_rows),
        "orders": pd.DataFrame(order_rows),
        "predictions": pd.DataFrame(prediction_rows),
        "optimization_trace": (
            pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
        ),
        "inner_oof_confusion": (
            np.stack(confusion_matrices) if confusion_matrices else np.empty((0, n_classes, n_classes))
        ),
    }


def run_random_baseline(
    context: DatasetContext,
    *,
    transformer_factory: Callable[[], Any],
    binary_estimator_factory: Callable[[np.ndarray], Any],
    fit_obd: Callable[..., Any],
    predict_obd: Callable[..., Any],
    n_repetitions: int = 20,
    seed: int = 89,
    outer_fold_limit: int | None = None,
) -> dict[str, Any]:
    """Train a reproducible population of independent random ordinal orders."""
    if n_repetitions < 2:
        raise ValueError("n_repetitions must be at least 2")
    n_classes = len(context.class_names)
    metric_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    folds = context.splits if outer_fold_limit is None else context.splits[:outer_fold_limit]
    for fold, (outer_train, outer_test) in enumerate(folds, start=1):
        print(
            f"[{context.dataset} | Random order | fold {fold}/{len(folds)} | "
            f"{n_repetitions} repetitions]",
            flush=True,
        )
        y_train = context.y[outer_train]
        y_test = context.y[outer_test]

        transformer = transformer_factory()
        x_embedding_train = transformer.fit_transform(context.X[outer_train], y_train)
        x_embedding_test = transformer.transform(context.X[outer_test])

        for repetition in range(n_repetitions):
            random_seed = seed + 100000 * repetition + 4019 * fold
            order = random_class_order(n_classes, random_seed)
            random_label = f"Random order seed {repetition + 1}"

            _, predictions = _order_probabilities(
                x_embedding_train,
                y_train,
                x_embedding_test,
                order,
                n_classes,
                fit_obd,
                predict_obd,
                binary_estimator_factory,
            )
            metric_rows.append({
                "dataset": context.dataset,
                "fold": fold,
                "method": RANDOM_METHOD,
                "random_rep": repetition,
                "random_seed": random_seed,
                **evaluate_predictions(y_test, predictions, order),
            })
            order_rows.extend({
                "dataset": context.dataset,
                "fold": fold,
                "method": RANDOM_METHOD,
                "random_rep": repetition,
                "random_seed": random_seed,
                "random_label": random_label,
                "position": position,
                "label": int(label),
                "class_name": context.class_names[int(label)],
            } for position, label in enumerate(order))
            prediction_rows.extend({
                "dataset": context.dataset,
                "fold": fold,
                "method": RANDOM_METHOD,
                "random_rep": repetition,
                "random_seed": random_seed,
                "record_id": record_id,
                "y_true": int(true_label),
                "y_pred": int(predicted_label),
            } for record_id, true_label, predicted_label in zip(
                context.record_ids[outer_test], y_test, predictions
            ))

    metrics = pd.DataFrame(metric_rows)
    robust_metrics = (
        metrics.groupby(["dataset", "fold", "method"], as_index=False)
        .agg({
            column: "mean"
            for column in (
                "accuracy", "f1_macro", "precision_macro",
                "recall_macro", "ordinal_mae", "fit_predict_seconds"
            )
            if column in metrics.columns
        })
    )
    return {
        "dataset": context.dataset,
        "method": RANDOM_METHOD,
        "metrics": robust_metrics,
        "metrics_repetitions": metrics,
        "orders": pd.DataFrame(order_rows),
        "predictions": pd.DataFrame(prediction_rows),
        "optimization_trace": pd.DataFrame(),
        "inner_oof_confusion": np.empty((0, n_classes, n_classes)),
    }


def robust_random_comparison(
    method_metrics: pd.DataFrame,
    random_repetitions: pd.DataFrame,
    *,
    metric: str = "accuracy",
    n_bootstrap: int = 10000,
    seed: int = 89,
) -> pd.DataFrame:
    """Bootstrap fold and random-seed uncertainty and run paired Wilcoxon tests."""
    from scipy.stats import wilcoxon

    rng = np.random.RandomState(seed)
    rows: list[dict[str, Any]] = []
    for dataset, method_group in method_metrics.groupby("dataset", sort=False):
        random_group = random_repetitions[random_repetitions["dataset"] == dataset]
        if random_group.empty:
            continue
        random_by_fold = (
            random_group.groupby(["random_rep", "fold"], as_index=False)[metric]
            .mean()
            .pivot(index="random_rep", columns="fold", values=metric)
        )
        random_fold_mean = random_by_fold.mean(axis=0)

        for method, part in method_group.groupby("method", sort=False):
            if method in (NOMINAL_METHOD, RANDOM_METHOD):
                continue
            method_by_fold = part.set_index("fold")[metric]
            folds = sorted(set(method_by_fold.index).intersection(random_fold_mean.index))
            if not folds:
                continue
            method_values = method_by_fold.loc[folds].to_numpy(dtype=float)
            baseline_values = random_fold_mean.loc[folds].to_numpy(dtype=float)
            paired_delta = method_values - baseline_values

            bootstrap_delta = np.empty(n_bootstrap, dtype=float)
            for iteration in range(n_bootstrap):
                sampled_folds = rng.choice(folds, size=len(folds), replace=True)
                method_sample = method_by_fold.loc[sampled_folds].to_numpy(dtype=float)
                random_sample = []
                for fold in sampled_folds:
                    repetition = rng.randint(0, len(random_by_fold))
                    random_sample.append(random_by_fold.iloc[repetition].loc[fold])
                bootstrap_delta[iteration] = (
                    float(method_sample.mean()) - float(np.mean(random_sample))
                )

            ci_low, ci_high = np.percentile(bootstrap_delta, [2.5, 97.5])
            if np.all(paired_delta == 0):
                wilcoxon_stat = np.nan
                wilcoxon_p = np.nan
            else:
                try:
                    test = wilcoxon(paired_delta, alternative="two-sided", method="auto")
                    wilcoxon_stat = float(test.statistic)
                    wilcoxon_p = float(test.pvalue)
                except ValueError:
                    wilcoxon_stat = np.nan
                    wilcoxon_p = np.nan
            rows.append({
                "dataset": dataset,
                "method": method,
                "n_random_repetitions": int(len(random_by_fold)),
                "n_folds": len(folds),
                f"{metric}_method_mean": float(method_values.mean()),
                f"{metric}_random_mean": float(baseline_values.mean()),
                f"{metric}_paired_delta_mean": float(paired_delta.mean()),
                f"{metric}_bootstrap_ci_low": float(ci_low),
                f"{metric}_bootstrap_ci_high": float(ci_high),
                "folds_better_than_random_mean": int((paired_delta > 0).sum()),
                "folds_tied_with_random_mean": int((paired_delta == 0).sum()),
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_p_value": wilcoxon_p,
                "bootstrap_positive_probability": float((bootstrap_delta > 0).mean()),
            })
    return pd.DataFrame(rows)


def save_method_csvs(result: dict[str, Any], output_dir: str | Path) -> None:
    """Save one independently runnable method as CSV files only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{result['dataset']}_{result['method'].lower().replace(' ', '_').replace('-', '_')}"
    result["metrics"].to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
    if "metrics_repetitions" in result:
        result["metrics_repetitions"].to_csv(
            output_dir / f"{prefix}_metrics_repetitions.csv", index=False
        )
    result["orders"].to_csv(output_dir / f"{prefix}_orders.csv", index=False)
    result["predictions"].to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    if not result["optimization_trace"].empty:
        trace = result["optimization_trace"].copy()
        if "order" in trace:
            trace["order"] = trace["order"].map(lambda order: ",".join(map(str, order)))
        trace.to_csv(output_dir / f"{prefix}_optimization_trace.csv", index=False)
    if result["inner_oof_confusion"].size:
        rows = []
        for fold, matrix in enumerate(result["inner_oof_confusion"], start=1):
            rows.extend({
                "dataset": result["dataset"], "method": result["method"], "fold": fold,
                "true_label": i, "predicted_label": j, "confusion_probability": value,
            } for i, row in enumerate(matrix) for j, value in enumerate(row))
        pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_inner_oof_confusion.csv", index=False)


def existing_reference_metrics(context: DatasetContext) -> pd.DataFrame:
    """Reuse only prior nominal MiniRocket metrics; Band power is retrained here."""
    nominal = context.nominal_folds.copy()
    nominal["method"] = NOMINAL_METHOD
    order = context.bandpower_orders[1]
    nominal_ordered_mae = {}
    for fold in range(1, len(context.splits) + 1):
        frame = context.nominal_predictions[
            context.nominal_predictions["fold"] == fold
        ].sort_values("record_id")
        nominal_ordered_mae[fold] = evaluate_predictions(
            frame["y_true"].to_numpy(dtype=int),
            frame["y_pred"].to_numpy(dtype=int),
            order,
        )["ordinal_mae"]
    rows = []
    for _, row in nominal.iterrows():
        fold = int(row["fold"])
        rows.append({
            "dataset": context.dataset,
            "fold": fold,
            "method": NOMINAL_METHOD,
            "accuracy": float(row["accuracy"]),
            "f1_macro": float(row["f1_macro"]),
            "precision_macro": float(row["precision_macro"]),
            "recall_macro": float(row["recall_macro"]),
            "fit_predict_seconds": float(row.get("fit_time", np.nan))
            + float(row.get("predict_time", np.nan)),
            "ordinal_mae": nominal_ordered_mae[fold],
        })
    return pd.DataFrame(rows)


def summarize_benchmark(
    context: DatasetContext,
    result: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join computed methods to prior references and calculate paired deltas."""
    metrics = pd.concat(
        [existing_reference_metrics(context), result["metrics"]],
        ignore_index=True,
    )
    rows = []
    nominal = metrics[metrics["method"] == NOMINAL_METHOD].set_index("fold")
    for method in (NOMINAL_METHOD, "Band power", *NEW_METHODS):
        part = metrics[metrics["method"] == method].set_index("fold").sort_index()
        if len(part) != len(context.splits):
            raise ValueError(f"{method} has incomplete fold metrics")
        row: dict[str, Any] = {
            "dataset": context.dataset,
            "method": method,
            "n_folds": len(part),
        }
        for metric in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "ordinal_mae"):
            values = part[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
            if method != NOMINAL_METHOD:
                base = nominal[metric].to_numpy(dtype=float)
                row[f"{metric}_delta_vs_nominal"] = float(np.nanmean(values - base))
                baseline_mean = float(np.nanmean(base))
                row[f"{metric}_relative_delta_pct"] = (
                    100.0 * row[f"{metric}_delta_vs_nominal"] / abs(baseline_mean)
                    if baseline_mean > 0 else np.nan
                )
        rows.append(row)
    summary = pd.DataFrame(rows)
    return metrics, summary


def baseline_comparison_table(
    all_fold_metrics: pd.DataFrame,
    baseline_method: str,
    *,
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Calculate fold-paired differences against an ordering baseline."""
    rows = []
    for dataset, group in all_fold_metrics.groupby("dataset", sort=False):
        baseline = group[group["method"] == baseline_method].set_index("fold")[metric]
        if baseline.empty:
            raise ValueError(f"Missing baseline {baseline_method!r} for {dataset}")
        for method, part in group.groupby("method", sort=False):
            values = part.set_index("fold")[metric]
            if set(values.index) != set(baseline.index):
                raise ValueError(f"Paired fold indices differ for {dataset}, {method}")
            delta = values - baseline
            base_mean = float(baseline.mean())
            rows.append({
                "dataset": dataset,
                "method": method,
                "baseline_method": baseline_method,
                f"{metric}_mean": float(values.mean()),
                f"{metric}_std": float(values.std(ddof=1)),
                f"{metric}_absolute_delta_vs_baseline": float(delta.mean()),
                f"{metric}_relative_delta_pct_vs_baseline": (
                    100 * float(delta.mean()) / abs(base_mean) if base_mean else np.nan
                ),
                "folds_better_than_baseline": int((delta > 0).sum()),
                "folds_tied_with_baseline": int((delta == 0).sum()),
            })
    return pd.DataFrame(rows)


def order_stability(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return position frequencies, mean pairwise Kendall tau, and per-class ranks."""
    from scipy.stats import kendalltau

    frequency_rows = []
    stability_rows = []
    rank_rows = []
    for (dataset, method), part in orders.groupby(["dataset", "method"], sort=False):
        part = part.copy()
        if (
            method == RANDOM_METHOD
            and "random_rep" in part.columns
            and part["random_rep"].notna().all()
        ):
            folds_per_instance = int(part["fold"].nunique())
            random_rep = pd.to_numeric(part["random_rep"], errors="raise")
            fold = pd.to_numeric(part["fold"], errors="raise")
            part["instance"] = (
                random_rep.astype(int) * folds_per_instance
                + fold.astype(int) - 1
            )
            duplicate_keys = part.duplicated(["instance", "label"])
            if duplicate_keys.any():
                raise ValueError(
                    "Random-order stability data contain duplicate instance/label entries"
                )
            pivot = part.pivot(
                index="instance", columns="label", values="position"
            ).sort_index()
        else:
            pivot = part.pivot(
                index="fold", columns="label", values="position"
            ).sort_index()
        classes = part.sort_values("label")["class_name"].drop_duplicates().tolist()
        for label, name in enumerate(classes):
            for position in range(pivot.shape[1]):
                frequency_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "label": label,
                    "class_name": name,
                    "position": position,
                    "count": int((pivot[label] == position).sum()),
                    "frequency": float((pivot[label] == position).mean()),
                })
            rank_rows.append({
                "dataset": dataset,
                "method": method,
                "label": label,
                "class_name": name,
                "mean_position": float(pivot[label].mean()),
                "position_std": float(pivot[label].std(ddof=1)),
                "unique_positions": int(pivot[label].nunique()),
            })
        taus = [
            kendalltau(a, b).statistic
            for i, a in enumerate(pivot.to_numpy())
            for b in pivot.to_numpy()[i + 1:]
        ]
        stability_rows.append({
            "dataset": dataset,
            "method": method,
            "mean_pairwise_kendall_tau": float(np.nanmean(taus)) if taus else np.nan,
            "std_pairwise_kendall_tau": float(np.nanstd(taus, ddof=1)) if len(taus) > 1 else np.nan,
            "mean_rank_std": float(pivot.std(axis=0, ddof=1).mean()),
        })
    return (
        pd.DataFrame(frequency_rows),
        pd.DataFrame(stability_rows),
        pd.DataFrame(rank_rows),
    )


def metric_comparison_table(
    all_fold_metrics: pd.DataFrame,
    *,
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Make fold-paired absolute and relative metric differences to nominal."""
    rows = []
    for dataset, group in all_fold_metrics.groupby("dataset", sort=False):
        nominal = group[group["method"] == NOMINAL_METHOD].set_index("fold")[metric]
        for method, part in group.groupby("method", sort=False):
            values = part.set_index("fold")[metric]
            if set(values.index) != set(nominal.index):
                raise ValueError(f"Paired fold indices differ for {dataset}, {method}")
            delta = values - nominal
            base_mean = float(nominal.mean())
            rows.append({
                "dataset": dataset,
                "method": method,
                f"{metric}_mean": float(values.mean()),
                f"{metric}_std": float(values.std(ddof=1)),
                f"{metric}_absolute_delta_vs_nominal": float(delta.mean()),
                f"{metric}_relative_delta_pct_vs_nominal": (
                    100 * float(delta.mean()) / abs(base_mean) if base_mean else np.nan
                ),
                "folds_better_than_nominal": int((delta > 0).sum()),
                "folds_tied_with_nominal": int((delta == 0).sum()),
            })
    return pd.DataFrame(rows)
