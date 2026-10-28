"""Experiments aggiuntivi, isolati dagli esperimenti SVM/RF/CatBoost esistenti."""
from __future__ import annotations

import copy
import importlib.metadata
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config as cfg
import utils


MODEL_SPECS = {
    "xgboost": {"input": "features", "datasets": ("UniZgFall", "UniMiB")},
    "minirocket": {"input": "raw", "datasets": ("UniZgFall", "UniMiB")},
    "multirocket": {"input": "raw", "datasets": ("UniZgFall", "UniMiB")},
    "cnn1d": {"input": "raw", "datasets": ("UniZgFall", "UniMiB")},
    "inceptiontime": {"input": "raw", "datasets": ("UniZgFall", "UniMiB")},
    "resnet1d": {"input": "raw", "datasets": ("UniZgFall", "UniMiB")},
}
TORCH_MODELS = {"cnn1d", "inceptiontime", "resnet1d"}
ROCKET_MODELS = {"minirocket", "multirocket"}


def print_versions():
    """Mostra versioni e disponibilità delle dipendenze avanzate senza importarle."""
    import platform
    import sys

    print(f"Python: {sys.version.split()[0]} | {platform.platform()}")
    for package in ("xgboost", "sktime", "torch"):
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: non installato")


def _output_dir():
    root = Path(cfg.ADVANCED_RESULTS_DIR)
    if cfg.QUICK_TEST:
        root = root / "quick"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _prefix(model, dataset):
    return f"{model}_{dataset}_{cfg.CV_MODE}"


def _load_data(dataset, kind):
    """Carica via utils e ricostruisce i raw come (n_casi, 3 assi, tempo)."""
    X, y, record_id, subject_id, feature_names = utils.load_experiment_data(dataset, kind)
    lengths = None
    if kind == "raw":
        X = X.reshape(len(X), -1, 3).transpose(0, 2, 1)
        meta = pd.read_csv(cfg.DATA_DIR / f"raw_{dataset}_meta.csv")
        lookup = meta.set_index("record_id")["n_samples"]
        lengths = lookup.loc[record_id].to_numpy(dtype=int)
        if X.shape[1] != 3 or np.any(lengths <= 0) or np.any(lengths > X.shape[2]):
            raise ValueError(f"Forma o lunghezze raw non valide per {dataset}: {X.shape}")
    return X, y, record_id, subject_id, feature_names, lengths


def _quick_subset_note(model, dataset, n):
    if cfg.QUICK_TEST:
        print(f"[{model} | {dataset}] QUICK_TEST: {n} esempi dopo il sottoinsieme per classe.")


def _binary_metrics(dataset, y_true, y_pred):
    """Usa senza modificarla la mappatura binaria configurata nel progetto."""
    return utils.binary_fall_metrics(dataset, y_true, y_pred)


def _split(dataset, y, subjects):
    splits, effective_mode = utils.make_splitter(dataset, y, subjects)
    return list(splits), effective_mode


def _xgboost_estimator(n_classes):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("XGBoost non è installato. Aggiungere le dipendenze con requirements.txt.") from exc
    params = dict(cfg.XGBOOST_PARAMS)
    if cfg.QUICK_TEST:
        params.update(cfg.QUICK_XGBOOST_PARAMS)
    params["num_class"] = n_classes
    return XGBClassifier(**params)


def _rocket_transformer(model):
    try:
        from sktime.transformations.compose import FitInTransform
        from sktime.transformations.panel.rocket import (
            MiniRocketMultivariate,
            MultiRocketMultivariate,
        )
    except ImportError as exc:
        raise ImportError("MiniRocket/MultiRocket richiedono sktime; consultare requirements.txt.") from exc
    n_kernels = cfg.QUICK_ROCKET_NUM_KERNELS if cfg.QUICK_TEST else (
        cfg.MINIROCKET_NUM_KERNELS if model == "minirocket" else cfg.MULTIROCKET_NUM_KERNELS
    )
    transformer_type = MiniRocketMultivariate if model == "minirocket" else MultiRocketMultivariate
    transformer = transformer_type(num_kernels=n_kernels, random_state=cfg.SEED, n_jobs=1)
    # Queste implementazioni stimano parametri per serie; FitInTransform le rifitta
    # indipendentemente su ciascun batch, senza apprendere statistiche tra casi.
    return FitInTransform(transformer)


def _fit_rocket(model, X_train, y_train, X_test):
    from sklearn.linear_model import RidgeClassifierCV

    transformer = _rocket_transformer(model)
    start = time.perf_counter()
    Xt_train = transformer.fit_transform(X_train, y_train)
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))
    classifier.fit(Xt_train, y_train)
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    Xt_test = transformer.transform(X_test)
    y_pred = classifier.predict(Xt_test)
    predict_time = time.perf_counter() - start
    return y_pred, fit_time, predict_time


def _torch_components(model, channels, n_classes):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("CNN 1D/InceptionTime/ResNet1D richiedono PyTorch; consultare requirements.txt.") from exc

    class MaskedPool(nn.Module):
        def forward(self, x, lengths):
            valid = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
            mask = valid[:, None, :]
            mean = (x * mask).sum(dim=-1) / lengths.clamp(min=1).to(x.dtype)[:, None]
            maximum = x.masked_fill(~mask, torch.finfo(x.dtype).min).amax(dim=-1)
            return torch.cat((mean, maximum), dim=1)

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            )
            self.skip = nn.Conv1d(in_channels, out_channels, 1, bias=False)
            self.pool = nn.MaxPool1d(2, ceil_mode=True)

        def forward(self, x):
            return self.pool(self.layers(x) + self.skip(x))

    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                ConvBlock(3, channels[0]),
                ConvBlock(channels[0], channels[1]),
                ConvBlock(channels[1], channels[2]),
            ])
            self.pool = MaskedPool()
            self.head = nn.Linear(channels[-1] * 2, n_classes)

        def forward(self, x, lengths):
            for block in self.blocks:
                x = block(x)
                lengths = (lengths + 1) // 2
            return self.head(self.pool(x, lengths))

    class InceptionBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            branch = out_channels // 4
            bottleneck = max(8, in_channels // 4)
            self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False)
            kernels = (9, 19, 39)
            self.branches = nn.ModuleList([
                nn.Conv1d(bottleneck, branch, kernel_size=k, padding=k // 2, bias=False)
                for k in kernels
            ])
            self.pool_branch = nn.Sequential(
                nn.MaxPool1d(3, stride=1, padding=1),
                nn.Conv1d(in_channels, branch, kernel_size=1, bias=False),
            )
            self.norm = nn.BatchNorm1d(branch * 4)
            self.skip = nn.Conv1d(in_channels, branch * 4, kernel_size=1, bias=False)
            self.pool = nn.MaxPool1d(2, ceil_mode=True)

        def forward(self, x):
            b = self.bottleneck(x)
            y = torch.cat([conv(b) for conv in self.branches] + [self.pool_branch(x)], dim=1)
            return self.pool(torch.nn.functional.gelu(self.norm(y) + self.skip(x)))

    class InceptionTime(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                InceptionBlock(3, channels[0]),
                InceptionBlock(channels[0], channels[1]),
                InceptionBlock(channels[1], channels[2]),
            ])
            self.pool = MaskedPool()
            self.head = nn.Linear((channels[-1] // 4) * 8, n_classes)

        def forward(self, x, lengths):
            for block in self.blocks:
                x = block(x)
                lengths = (lengths + 1) // 2
            return self.head(self.pool(x, lengths))

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, 7, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, 7, padding=3, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.skip = nn.Conv1d(in_channels, out_channels, 1, bias=False)
            self.pool = nn.MaxPool1d(2, ceil_mode=True)

        def forward(self, x):
            y = torch.nn.functional.gelu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            return self.pool(torch.nn.functional.gelu(y + self.skip(x)))

    class ResNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                ResidualBlock(3, channels[0]),
                ResidualBlock(channels[0], channels[1]),
                ResidualBlock(channels[1], channels[2]),
            ])
            self.pool = MaskedPool()
            self.head = nn.Linear(channels[-1] * 2, n_classes)

        def forward(self, x, lengths):
            for block in self.blocks:
                x = block(x)
                lengths = (lengths + 1) // 2
            return self.head(self.pool(x, lengths))

    factories = {"cnn1d": CNN1D, "inceptiontime": InceptionTime, "resnet1d": ResNet1D}
    return torch, factories[model]()


def _torch_validation_indices(train_idx, y, subjects):
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

    if cfg.CV_MODE == "group" and not np.isnan(subjects).all():
        splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.TORCH_VALIDATION_FRACTION,
                                     random_state=cfg.SEED)
        local_train, local_val = next(splitter.split(train_idx, y[train_idx], subjects[train_idx]))
        return train_idx[local_train], train_idx[local_val]
    y_train = y[train_idx]
    counts = np.unique(y_train, return_counts=True)[1]
    if len(counts) > 1 and counts.min() >= 2:
        test_fraction = max(cfg.TORCH_VALIDATION_FRACTION, len(counts) / len(train_idx))
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=cfg.SEED)
        local_train, local_val = next(splitter.split(np.zeros(len(train_idx)), y_train))
        return train_idx[local_train], train_idx[local_val]
    raise ValueError("Il training fold è troppo piccolo per creare una validation stratificata.")


def _train_torch(model, X, lengths, y, train_idx, test_idx, subjects, dataset):
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score

    channels = cfg.QUICK_TORCH_CHANNELS if cfg.QUICK_TEST else cfg.TORCH_CHANNELS
    torch, network = _torch_components(model, channels, len(cfg.DATASETS[dataset]["class_names"]))
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)

    fit_idx, val_idx = _torch_validation_indices(train_idx, y, subjects)
    valid_train = np.arange(X.shape[2])[None, :] < lengths[fit_idx, None]
    train_vals = X[fit_idx].transpose(0, 2, 1)[valid_train]
    mean = train_vals.mean(axis=0).astype(np.float32)
    scale = train_vals.std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    Xn = np.asarray(X, dtype=np.float32).copy()
    for i, n in enumerate(lengths):
        Xn[i, :, :n] = (Xn[i, :, :n] - mean[:, None]) / scale[:, None]

    x_tensor = torch.from_numpy(Xn)
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.int64))
    lengths_tensor = torch.from_numpy(np.asarray(lengths, dtype=np.int64))
    optimizer = torch.optim.AdamW(network.parameters(), lr=cfg.TORCH_LEARNING_RATE,
                                  weight_decay=cfg.TORCH_WEIGHT_DECAY)
    batch_size = cfg.QUICK_TORCH_BATCH_SIZE if cfg.QUICK_TEST else cfg.TORCH_BATCH_SIZE
    max_epochs = cfg.QUICK_TORCH_MAX_EPOCHS if cfg.QUICK_TEST else cfg.TORCH_MAX_EPOCHS
    history = []
    best_loss, best_state, stale = float("inf"), None, 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for epoch in range(max_epochs):
        network.train()
        rng = np.random.RandomState(cfg.SEED + epoch)
        order = rng.permutation(fit_idx)
        epoch_losses = []
        for start_i in range(0, len(order), batch_size):
            batch = order[start_i:start_i + batch_size]
            xb = x_tensor[batch].to(device)
            yb = y_tensor[batch].to(device)
            lb = lengths_tensor[batch].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(network(xb, lb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        network.eval()
        with torch.no_grad():
            val_logits = network(x_tensor[val_idx].to(device), lengths_tensor[val_idx].to(device))
            val_loss = float(F.cross_entropy(val_logits, y_tensor[val_idx].to(device)).cpu())
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "val_loss": val_loss,
            "val_accuracy": accuracy_score(y[val_idx], val_pred),
        })
        if val_loss < best_loss:
            best_loss, best_state, stale = val_loss, copy.deepcopy(network.state_dict()), 0
        else:
            stale += 1
        print(f"  epoch {epoch + 1}/{max_epochs}: train_loss={history[-1]['train_loss']:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={history[-1]['val_accuracy']:.4f}", flush=True)
        if stale >= cfg.TORCH_EARLY_STOPPING_PATIENCE:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_time = time.perf_counter() - start
    if best_state is not None:
        network.load_state_dict(best_state)
    network.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(test_idx), batch_size):
            batch = test_idx[i:i + batch_size]
            logits = network(x_tensor[batch].to(device), lengths_tensor[batch].to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    predict_time = time.perf_counter() - start
    return np.concatenate(predictions), fit_time, predict_time, pd.DataFrame(history), device.type


def run_experiment(model, dataset):
    """Esegue un singolo dataset in CV, salvando fold, predizioni e curve."""
    from tqdm.auto import tqdm

    if model not in MODEL_SPECS:
        raise ValueError(f"Modello avanzato sconosciuto: {model}")
    kind = MODEL_SPECS[model]["input"]
    out = _output_dir()
    prefix = _prefix(model, dataset)
    folds_path = out / f"{prefix}_folds.csv"
    if folds_path.exists() and not cfg.FORCE_RERUN:
        print(f"[{model} | {dataset}] risultati già presenti, salto ({folds_path.name}).")
        return pd.read_csv(folds_path)

    X, y, record_ids, subjects, feature_names, lengths = _load_data(dataset, kind)
    _quick_subset_note(model, dataset, len(y))
    splits, cv_mode = _split(dataset, y, subjects)
    n_classes = len(cfg.DATASETS[dataset]["class_names"])
    labels = np.arange(n_classes)
    rows, predictions, confusion_matrices, importances, histories = [], [], [], [], []

    print("=" * 76)
    print(f"[{model} | {dataset} | {kind}] X={X.shape}, classi={n_classes}, "
          f"CV={cv_mode}, fold={len(splits)}")
    if model == "xgboost":
        params = dict(cfg.XGBOOST_PARAMS)
        if cfg.QUICK_TEST:
            params.update(cfg.QUICK_XGBOOST_PARAMS)
        print("Parametri:", params)
    elif model in ROCKET_MODELS:
        n_kernels = cfg.QUICK_ROCKET_NUM_KERNELS if cfg.QUICK_TEST else (
            cfg.MINIROCKET_NUM_KERNELS if model == "minirocket" else cfg.MULTIROCKET_NUM_KERNELS
        )
        print(f"Parametri: num_kernels={n_kernels}, RidgeClassifierCV, seed={cfg.SEED}")
    else:
        print(f"Parametri: max_epochs={cfg.QUICK_TORCH_MAX_EPOCHS if cfg.QUICK_TEST else cfg.TORCH_MAX_EPOCHS}, "
              f"batch_size={cfg.QUICK_TORCH_BATCH_SIZE if cfg.QUICK_TEST else cfg.TORCH_BATCH_SIZE}, "
              f"lr={cfg.TORCH_LEARNING_RATE}, weight_decay={cfg.TORCH_WEIGHT_DECAY}")

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(splits, desc=f"{model}|{dataset}"), start=1
    ):
        tag = f"[{model} | {dataset} | {kind} | fold {fold}/{len(splits)}]"
        print(f"{tag} train su {len(train_idx)} esempi, test su {len(test_idx)}")
        print("fit in corso...", flush=True)
        if model == "xgboost":
            estimator = _xgboost_estimator(n_classes)
            start = time.perf_counter()
            estimator.fit(X[train_idx], y[train_idx], verbose=False)
            fit_time = time.perf_counter() - start
            start = time.perf_counter()
            y_pred = np.asarray(estimator.predict(X[test_idx])).ravel().astype(int)
            predict_time = time.perf_counter() - start
            importances.append(np.asarray(estimator.feature_importances_, dtype=float))
        elif model in ROCKET_MODELS:
            y_pred, fit_time, predict_time = _fit_rocket(
                model, X[train_idx], y[train_idx], X[test_idx]
            )
            y_pred = np.asarray(y_pred).ravel().astype(int)
        else:
            y_pred, fit_time, predict_time, history, device = _train_torch(
                model, X, lengths, y, train_idx, test_idx, subjects, dataset
            )
            history.insert(0, "fold", fold)
            histories.append(history)

        row = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "fit_time": fit_time,
            "predict_time": predict_time,
            "accuracy": accuracy_score(y[test_idx], y_pred),
            "f1_macro": f1_score(y[test_idx], y_pred, average="macro", zero_division=0),
            "precision_macro": precision_score(y[test_idx], y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y[test_idx], y_pred, average="macro", zero_division=0),
        }
        row.update(_binary_metrics(dataset, y[test_idx], y_pred))
        rows.append(row)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            confusion_matrices.append(
                confusion_matrix(y[test_idx], y_pred, labels=labels, normalize="true")
            )
        predictions.append(pd.DataFrame({
            "record_id": record_ids[test_idx],
            "y_true": y[test_idx],
            "y_pred": y_pred,
            "fold": fold,
        }))
        print(f"{tag} fit {fit_time:.2f}s | predict {predict_time:.2f}s | "
              f"acc {row['accuracy']:.4f} | F1 macro {row['f1_macro']:.4f}", flush=True)

    folds = pd.DataFrame(rows)
    folds.to_csv(folds_path, index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / f"{prefix}_preds.csv", index=False)
    np.save(out / f"{prefix}_confmat.npy", np.stack(confusion_matrices))
    if importances:
        pd.DataFrame({
            "feature": feature_names,
            "importance_mean": np.mean(importances, axis=0),
            "importance_std": np.std(importances, axis=0),
        }).sort_values("importance_mean", ascending=False).to_csv(
            out / f"{prefix}_importance.csv", index=False
        )
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(out / f"{prefix}_history.csv", index=False)
    print(f"[{model} | {dataset}] media ± std:")
    for metric in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "fit_time", "predict_time"):
        print(f"  {metric:16}: {folds[metric].mean():.4f} ± {folds[metric].std(ddof=1):.4f}")
    return folds


def run_all(model):
    """Esegue i due dataset appropriati al nuovo modello."""
    for dataset in MODEL_SPECS[model]["datasets"]:
        run_experiment(model, dataset)


def summary_table(model):
    out = _output_dir()
    rows = []
    for dataset in MODEL_SPECS[model]["datasets"]:
        path = out / f"{_prefix(model, dataset)}_folds.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        row = {"dataset": dataset, "input": MODEL_SPECS[model]["input"]}
        for metric in ("accuracy", "f1_macro", "precision_macro", "recall_macro",
                       "fit_time", "predict_time", "fall_sensitivity",
                       "fall_specificity", "fall_precision"):
            values = frame[metric]
            row[metric] = "n/c" if values.isna().all() else (
                f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out / f"{model}_{cfg.CV_MODE}_summary.csv", index=False)
    return result


def plot_all(model):
    """Crea figure distinte per gli esperimenti avanzati già salvati."""
    import matplotlib.pyplot as plt

    out = _output_dir()
    for dataset in MODEL_SPECS[model]["datasets"]:
        prefix = _prefix(model, dataset)
        folds_path = out / f"{prefix}_folds.csv"
        if not folds_path.exists():
            continue
        folds = pd.read_csv(folds_path)
        for metric, title in (("accuracy", "Accuracy"), ("f1_macro", "F1 macro")):
            fig, ax = plt.subplots(figsize=(5, 4))
            values = folds[metric].to_numpy()
            ax.boxplot([values], tick_labels=[MODEL_SPECS[model]["input"]], showfliers=False)
            ax.scatter(np.random.RandomState(cfg.SEED).normal(1, 0.035, len(values)),
                       values, alpha=0.75, zorder=3)
            ax.set_title(f"{model} | {dataset} | {title} per fold")
            ax.set_ylabel(title)
            fig.savefig(out / f"{prefix}_boxplot_{metric}.png", dpi=150, bbox_inches="tight")
            plt.show()
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(folds["fold"], folds["fit_time"], marker="o", label="training")
        ax.plot(folds["fold"], folds["predict_time"], marker="s", label="inferenza")
        ax.set_xlabel("fold")
        ax.set_ylabel("tempo (s)")
        ax.legend()
        ax.set_title(f"{model} | {dataset} | tempi per fold")
        fig.savefig(out / f"{prefix}_timing_seconds.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        class_names = cfg.DATASETS[dataset]["class_names"]
        matrices = np.load(out / f"{prefix}_confmat.npy")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_matrix = np.nanmean(matrices, axis=0) * 100
        size = 6 if len(class_names) <= 10 else 12
        fig, ax = plt.subplots(figsize=(size + 1, size))
        image = ax.imshow(np.nan_to_num(mean_matrix), cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=60, ha="right")
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlabel("predetta")
        ax.set_ylabel("vera")
        ax.set_title(f"{model} | {dataset} | confusion matrix media (%)")
        fig.colorbar(image, ax=ax, fraction=0.046)
        fig.savefig(out / f"{prefix}_confusion-matrix_percent.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        importance_path = out / f"{prefix}_importance.csv"
        if importance_path.exists():
            importance = pd.read_csv(importance_path).head(20).iloc[::-1]
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(importance["feature"], importance["importance_mean"],
                    xerr=importance["importance_std"])
            ax.set_xlabel("importanza media sui fold")
            ax.set_title(f"{model} | {dataset} | top 20 feature")
            fig.savefig(out / f"{prefix}_importance_top20.png", dpi=150, bbox_inches="tight")
            plt.show()
            plt.close(fig)

        history_path = out / f"{prefix}_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            fig, ax = plt.subplots(figsize=(6, 4))
            for fold, part in history.groupby("fold"):
                ax.plot(part["epoch"], part["train_loss"], alpha=0.5,
                        label=f"train fold {fold}")
                ax.plot(part["epoch"], part["val_loss"], linestyle="--", alpha=0.7,
                        label=f"val fold {fold}")
            ax.set_xlabel("epoca")
            ax.set_ylabel("cross-entropy")
            ax.set_title(f"{model} | {dataset} | curve training/validation")
            ax.legend(fontsize=7)
            fig.savefig(out / f"{prefix}_learning-curves.png", dpi=150, bbox_inches="tight")
            plt.show()
            plt.close(fig)
