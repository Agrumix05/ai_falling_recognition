"""Codice condiviso: caricamento dati, feature extraction, CV, metriche, plot."""
import json
import platform
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

ACC_AXES = ["x", "y", "z"]
TEMPORAL_STATS = [
    "mean", "std", "aad", "min", "max", "maxmin_diff", "median", "mad", "IQR",
    "neg_count", "pos_count", "above_mean", "peak_count", "skewness", "kurtosis", "energy",
]
FFT_STATS = [s for s in TEMPORAL_STATS if s not in ("neg_count", "pos_count")]
FEATURE_NAMES = (
    [f"{a}_{s}" for s in TEMPORAL_STATS for a in ACC_AXES] + ["avg_result_accl", "sma"]
    + [f"{a}_{s}_fft" for s in FFT_STATS for a in ACC_AXES] + ["avg_result_accl_fft", "sma_fft"]
)
assert len(FEATURE_NAMES) == 94, len(FEATURE_NAMES)


# --------------------------------------------------------------------------- ambiente
def print_versions():
    import catboost, matplotlib, natsort, scipy, sklearn, tqdm
    print("Python      :", sys.version.split()[0], "|", platform.platform())
    for m in (np, pd, scipy, sklearn, catboost, matplotlib, natsort, tqdm):
        print(f"{m.__name__:12}: {m.__version__}")


def print_config():
    print(f"SEED={cfg.SEED}  N_SPLITS={n_splits()}  CV_MODE={cfg.CV_MODE}  QUICK_TEST={cfg.QUICK_TEST}")
    print("FALL_CLASSES:", cfg.FALL_CLASSES)


def n_splits():
    return cfg.QUICK_N_SPLITS if cfg.QUICK_TEST else cfg.N_SPLITS


# --------------------------------------------------------------------------- caricamento
def list_files(dataset):
    """File in ordinamento naturale (come natsorted(os.listdir()) nei notebook originali)."""
    from natsort import natsorted
    import os
    d = cfg.DATASETS[dataset]["data_dir"]
    return natsorted(os.listdir(d))


def load_labels(dataset):
    return np.loadtxt(cfg.DATASETS[dataset]["labels_file"]).astype(int)  # '#' = commento


def load_recording(dataset, filename):
    """Array (n, 3) con x, y, z. UniZgFall: 4a colonna (tempo) ignorata."""
    path = cfg.DATASETS[dataset]["data_dir"] / filename
    a = np.genfromtxt(path, delimiter=",")  # righe '#' ignorate (UniMiB)
    return a[:, :3]


def parse_subject(dataset, filename):
    rx = cfg.DATASETS[dataset]["subject_regex"]
    if rx is None:
        return np.nan
    m = re.search(rx, filename)
    assert m, f"nome file non conforme: {filename}"
    return int(m.group(1))


# --------------------------------------------------------------------------- feature extraction
def _stats(s, with_counts):
    from scipy.signal import find_peaks
    from scipy.stats import kurtosis, skew
    n = len(s)
    mean = np.mean(s)
    q75, q25 = np.percentile(s, [75, 25])
    med = np.median(s)
    d = {
        "mean": mean, "std": np.std(s), "aad": np.mean(np.abs(s - mean)),
        "min": np.min(s), "max": np.max(s), "maxmin_diff": np.max(s) - np.min(s),
        "median": med, "mad": np.median(np.abs(s - med)), "IQR": q75 - q25,
        "neg_count": np.sum(s < 0), "pos_count": np.sum(s > 0),
        "above_mean": np.sum(s > mean), "peak_count": len(find_peaks(s)[0]),
        "skewness": skew(s), "kurtosis": kurtosis(s), "energy": np.sum(s ** 2) / n,
    }
    if not with_counts:
        d.pop("neg_count"); d.pop("pos_count")
    return d


def _block(sig, with_counts):
    """sig: (n, 3). Ritorna lista di valori ordinata per statistica e poi per asse + avg_result + sma."""
    stats = TEMPORAL_STATS if with_counts else FFT_STATS
    per_axis = [_stats(sig[:, i], with_counts) for i in range(3)]
    out = [per_axis[i][st] for st in stats for i in range(3)]
    out.append(np.mean(np.sqrt(np.sum(sig ** 2, axis=1))))
    out.append(sum(np.sum(np.abs(sig[:, i])) / len(sig) for i in range(3)))
    return out


def extract_features(sig):
    """94 feature: 50 temporali + 44 sul modulo FFT (bin 1..n//2-1, senza DC)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # skew/kurtosis di segnali costanti -> NaN (poi riportati)
        fft = np.stack([np.abs(np.fft.fft(sig[:, i]))[1:len(sig) // 2] for i in range(3)], axis=1)
        feats = _block(sig, True) + _block(fft, False)
    assert len(feats) == 94
    return feats


# --------------------------------------------------------------------------- creazione dataset
def build_dataset_files(dataset):
    """Carica un dataset, estrae le feature e salva CSV (feature) + npy/csv (raw)."""
    from tqdm.auto import tqdm
    info = cfg.DATASETS[dataset]
    files = list_files(dataset)
    y = load_labels(dataset)
    assert len(files) == len(y), f"{dataset}: {len(files)} file vs {len(y)} label"
    names = info["class_names"]
    assert y.min() >= 0 and y.max() < len(names)
    label_name = [names[i] for i in y]
    if dataset == "UniZgFall":  # i suffissi dei file devono coincidere con le label per posizione
        suffix = [re.search(r"_([A-Z]+)\.csv$", f).group(1) for f in files]
        assert suffix == label_name, "suffissi dei file e label non coincidono"
    subject = [parse_subject(dataset, f) for f in files]

    print(f"=== {dataset}: {len(files)} registrazioni; feature attese: {len(FEATURE_NAMES)} ===")
    signals, feats = [], []
    for f in tqdm(files, desc=f"Caricamento {dataset}"):
        s = load_recording(dataset, f)
        signals.append(s)
        feats.append(extract_features(s))

    lengths = np.array([len(s) for s in signals])
    X = np.array(feats, dtype=float)
    assert X.shape == (len(files), 94)
    print("Numero di feature:", X.shape[1])

    ids = pd.DataFrame({"record_id": files, "subject_id": pd.array(subject, dtype="Int64")})
    df = pd.concat([ids, pd.DataFrame(X, columns=FEATURE_NAMES)], axis=1)
    df["label"] = y.astype(int)
    df["label_name"] = label_name
    cfg.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(cfg.DATA_DIR / f"features_{dataset}.csv", index=False)

    max_len = cfg.RAW_MAX_LENGTH.get(dataset) if cfg.RAW_MAX_LENGTH else None
    if max_len is None or lengths.max() > max_len:
        print(f"Padding: uso il massimo reale {lengths.max()} (configurato: {max_len})")
        max_len = int(lengths.max())
    raw = np.zeros((len(signals), max_len, 3))
    for i, s in enumerate(signals):
        raw[i, :len(s)] = s
    raw = raw.reshape(len(signals), -1)
    np.save(cfg.DATA_DIR / f"raw_{dataset}_X.npy", raw)
    meta = ids.copy()
    meta["label"] = y.astype(int)
    meta["label_name"] = label_name
    meta["n_samples"] = lengths
    meta.to_csv(cfg.DATA_DIR / f"raw_{dataset}_meta.csv", index=False)

    print("X feature:", X.shape, "| y:", y.shape, "| X raw:", raw.shape, "(max_length =", max_len, ")")
    print("Distribuzione classi:")
    print(df.groupby(["label", "label_name"]).size().to_string())
    print("NaN nelle feature:", int(np.isnan(X).sum()), "| inf:", int(np.isinf(X).sum()))
    if np.isnan(X).any():
        cols = df[FEATURE_NAMES].columns[np.isnan(X).any(axis=0)].tolist()
        print("  colonne con NaN:", cols)
    print(f"Lunghezze registrazioni: min={lengths.min()}  mediana={int(np.median(lengths))}  max={lengths.max()}")
    return df, meta


# --------------------------------------------------------------------------- dati per gli esperimenti
def load_experiment_data(dataset, kind):
    """Ritorna X, y, record_id, subject_id (subset se QUICK_TEST)."""
    if kind == "features":
        df = pd.read_csv(cfg.DATA_DIR / f"features_{dataset}.csv")
        X = df[FEATURE_NAMES].to_numpy(float)
        meta = df
        feat_names = FEATURE_NAMES
    else:
        meta = pd.read_csv(cfg.DATA_DIR / f"raw_{dataset}_meta.csv")
        X = np.load(cfg.DATA_DIR / f"raw_{dataset}_X.npy", mmap_mode="r")
        feat_names = None
    y = meta["label"].to_numpy(int)
    rec = meta["record_id"].to_numpy()
    subj = meta["subject_id"].to_numpy(float)
    if cfg.QUICK_TEST:
        rng = np.random.RandomState(cfg.SEED)
        idx = np.sort(np.concatenate([
            rng.choice(np.where(y == c)[0], min(cfg.QUICK_PER_CLASS, (y == c).sum()), replace=False)
            for c in np.unique(y)]))
        X, y, rec, subj = X[idx], y[idx], rec[idx], subj[idx]
    X = np.asarray(X, dtype=float)
    return X, y, rec, subj, feat_names


# --------------------------------------------------------------------------- modelli e CV
def build_estimator(model):
    if model == "svm":
        from sklearn.svm import SVC
        return SVC(**cfg.SVM_PARAMS)
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        n = cfg.QUICK_RF_ESTIMATORS if cfg.QUICK_TEST else cfg.RF_N_ESTIMATORS
        return RandomForestClassifier(n_estimators=n, **cfg.RF_PARAMS)
    if model == "catboost":
        from catboost import CatBoostClassifier
        params = dict(cfg.CATBOOST_PARAMS)
        if cfg.QUICK_TEST:
            params["iterations"] = cfg.QUICK_CATBOOST_ITERATIONS
        elif cfg.CATBOOST_ITERATIONS:
            params["iterations"] = cfg.CATBOOST_ITERATIONS
        return CatBoostClassifier(**params)
    raise ValueError(model)


def make_splitter(dataset, y, subj):
    """Ritorna (generatore di split, modalita' effettiva)."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    mode = cfg.CV_MODE
    assert mode in ("stratified", "group"), mode
    k = n_splits()
    if mode == "group" and np.isnan(subj).all():
        print(f"[{dataset}] CV_MODE='group' non applicabile (subject_id non disponibile): uso 'stratified'.")
        mode = "stratified"
    if mode == "group":
        sp = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg.SEED)
        return sp.split(np.zeros(len(y)), y, groups=subj), mode
    sp = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.SEED)
    return sp.split(np.zeros(len(y)), y), mode


def binary_fall_metrics(dataset, y_true, y_pred):
    fall = cfg.FALL_CLASSES.get(dataset)
    if not fall:
        return dict(fall_sensitivity=np.nan, fall_specificity=np.nan, fall_precision=np.nan)
    t, p = np.isin(y_true, fall), np.isin(y_pred, fall)
    tp, fn, fp, tn = (t & p).sum(), (t & ~p).sum(), (~t & p).sum(), (~t & ~p).sum()
    div = lambda a, b: a / b if b else np.nan
    return dict(fall_sensitivity=div(tp, tp + fn), fall_specificity=div(tn, tn + fp),
                fall_precision=div(tp, tp + fp))


def results_dir():
    d = cfg.RESULTS_DIR / "quick" if cfg.QUICK_TEST else cfg.RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prefix(model, dataset, kind):
    return f"{model}_{dataset}_{kind}_{cfg.CV_MODE}"


def run_experiment(model, dataset, kind):
    """Train + evaluation in CV. Salva metriche per fold, predizioni, confmat, importanze."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    from tqdm.auto import tqdm
    out = results_dir()
    pre = _prefix(model, dataset, kind)
    if (out / f"{pre}_folds.csv").exists() and not cfg.FORCE_RERUN:
        print(f"[{model} | {dataset} | {kind}] risultati gia' presenti ({pre}), salto (FORCE_RERUN=False).")
        return pd.read_csv(out / f"{pre}_folds.csv")

    X, y, rec, subj, feat_names = load_experiment_data(dataset, kind)
    n_cls = len(cfg.DATASETS[dataset]["class_names"])
    labels = np.arange(n_cls)
    splits, mode = make_splitter(dataset, y, subj)
    splits = list(splits)
    k = len(splits)
    print("=" * 80)
    print(f"[{model} | {dataset} | {kind}] X={X.shape}, classi={n_cls}, CV={mode}, fold={k}")
    print("Parametri:", build_estimator(model).get_params())
    if not cfg.FALL_CLASSES.get(dataset):
        print(f"Metrica binaria caduta/ADL: non calcolata (FALL_CLASSES['{dataset}'] non definita)")

    rows, preds, cms, imps = [], [], [], []
    for i, (tr, te) in enumerate(tqdm(splits, desc=f"{model}|{dataset}|{kind}"), start=1):
        tag = f"[{model} | {dataset} | {kind} | fold {i}/{k}]"
        print(f"{tag} train su {len(tr)} esempi, test su {len(te)}")
        est = build_estimator(model)
        print("fit in corso...", flush=True)
        t0 = time.time()
        if model == "catboost":
            every = 10 if cfg.QUICK_TEST else cfg.CATBOOST_LOG_EVERY
            est.fit(X[tr], y[tr], verbose=every if every else False)
        else:
            est.fit(X[tr], y[tr])
        fit_t = time.time() - t0
        t0 = time.time()
        yp = np.asarray(est.predict(X[te])).ravel().astype(int)
        pred_t = time.time() - t0
        row = dict(fold=i, n_train=len(tr), n_test=len(te), fit_time=fit_t, predict_time=pred_t,
                   accuracy=accuracy_score(y[te], yp),
                   f1_macro=f1_score(y[te], yp, average="macro", zero_division=0),
                   precision_macro=precision_score(y[te], yp, average="macro", zero_division=0),
                   recall_macro=recall_score(y[te], yp, average="macro", zero_division=0))
        row.update(binary_fall_metrics(dataset, y[te], yp))
        rows.append(row)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cms.append(confusion_matrix(y[te], yp, labels=labels, normalize="true"))  # righe vuote -> NaN
        preds.append(pd.DataFrame({"record_id": rec[te], "y_true": y[te], "y_pred": yp, "fold": i}))
        if model in ("rf", "catboost") and kind == "features":
            imp = est.feature_importances_ if model == "rf" else est.get_feature_importance()
            imps.append(np.asarray(imp, dtype=float))
        print(f"{tag} fit {fit_t:.1f}s | predict {pred_t:.2f}s | acc {row['accuracy']:.4f} | "
              f"F1 macro {row['f1_macro']:.4f}", flush=True)

    folds = pd.DataFrame(rows)
    folds.to_csv(out / f"{pre}_folds.csv", index=False)
    pd.concat(preds).to_csv(out / f"{pre}_preds.csv", index=False)
    np.save(out / f"{pre}_confmat.npy", np.stack(cms))
    if imps:
        pd.DataFrame({"feature": feat_names, "importance_mean": np.mean(imps, axis=0),
                      "importance_std": np.std(imps, axis=0)}
                     ).sort_values("importance_mean", ascending=False
                                   ).to_csv(out / f"{pre}_importance.csv", index=False)
    print(f"[{model} | {dataset} | {kind}] RIEPILOGO (media ± std su {k} fold)")
    for m in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "fit_time"):
        print(f"  {m:16}: {folds[m].mean():.4f} ± {folds[m].std(ddof=1):.4f}")
    return folds


EXPERIMENTS = [("UniZgFall", "features"), ("UniZgFall", "raw"), ("UniMiB", "features"), ("UniMiB", "raw")]


def run_all(model):
    for ds, kind in EXPERIMENTS:
        run_experiment(model, ds, kind)


def summary_table(model):
    out = results_dir()
    rows = []
    for ds, kind in EXPERIMENTS:
        p = out / f"{_prefix(model, ds, kind)}_folds.csv"
        if not p.exists():
            continue
        f = pd.read_csv(p)
        r = {"dataset": ds, "input": kind}
        for m in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "fit_time",
                  "fall_sensitivity", "fall_specificity", "fall_precision"):
            r[m] = "n/c" if f[m].isna().all() else f"{f[m].mean():.4f} ± {f[m].std(ddof=1):.4f}"
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / f"{model}_{cfg.CV_MODE}_summary.csv", index=False)
    return t


# --------------------------------------------------------------------------- plot
def _save(fig, name):
    import matplotlib.pyplot as plt
    fig.savefig(results_dir() / name, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_all(model):
    import matplotlib.pyplot as plt
    out = results_dir()
    cv = cfg.CV_MODE
    for ds in cfg.DATASETS:
        have = {k: pd.read_csv(out / f"{_prefix(model, ds, k)}_folds.csv")
                for k in ("features", "raw") if (out / f"{_prefix(model, ds, k)}_folds.csv").exists()}
        if not have:
            continue
        for metric, title in (("accuracy", "Accuracy"), ("f1_macro", "F1 macro")):
            fig, ax = plt.subplots(figsize=(5, 4))
            kinds = list(have)
            ax.boxplot([have[k][metric] for k in kinds], tick_labels=kinds, showfliers=False)
            for j, k in enumerate(kinds, start=1):
                v = have[k][metric].to_numpy()
                ax.scatter(np.random.RandomState(cfg.SEED).normal(j, 0.04, len(v)), v, alpha=.7, zorder=3)
            ax.set_title(f"{model} | {ds} | {title} per fold ({cv})")
            ax.set_ylabel(title)
            _save(fig, f"{model}_{ds}_boxplot_{metric}.png")
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, f in have.items():
            ax.plot(f["fold"], f["fit_time"], marker="o", label=k)
        ax.set_xlabel("fold"); ax.set_ylabel("tempo di fit (s)"); ax.legend()
        ax.set_title(f"{model} | {ds} | tempi di training ({cv})")
        _save(fig, f"{model}_{ds}_fit-time_seconds.png")
        names = cfg.DATASETS[ds]["class_names"]
        for k in have:
            cm = np.load(out / f"{_prefix(model, ds, k)}_confmat.npy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = np.nanmean(cm, axis=0) * 100
            n = len(names)
            s = 6 if n <= 10 else 12
            fig, ax = plt.subplots(figsize=(s + 1, s))
            im = ax.imshow(np.nan_to_num(m), cmap="Blues", vmin=0, vmax=100)
            ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=60, ha="right")
            ax.set_yticks(range(n)); ax.set_yticklabels(names)
            for i in range(n):
                for j in range(n):
                    v = np.nan_to_num(m[i, j])
                    if v >= 0.5:
                        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7 if n > 10 else 9,
                                color="white" if v > 50 else "black")
            ax.set_xlabel("predetta"); ax.set_ylabel("vera")
            ax.set_title(f"{model} | {ds} | {k} | confusion matrix media (% per riga, {cv})")
            fig.colorbar(im, ax=ax, fraction=0.046)
            _save(fig, f"{model}_{ds}_{k}_confusion-matrix_percent.png")
        p = out / f"{_prefix(model, ds, 'features')}_importance.csv"
        if p.exists():
            imp = pd.read_csv(p).head(20)[::-1]
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"])
            ax.set_xlabel("importanza media sui fold")
            ax.set_title(f"{model} | {ds} | top 20 feature importance ({cv})")
            _save(fig, f"{model}_{ds}_features_importance_top20.png")
