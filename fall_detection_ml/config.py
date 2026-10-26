"""Configurazione centrale: percorsi, seed, parametri dei modelli e dei protocolli."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = (PROJECT_DIR.parent / "ml_accelerometers").resolve()  # solo lettura
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

SEED = 89
N_SPLITS = 10

# "stratified" = StratifiedKFold (baseline dei notebook originali)
# "group"      = StratifiedGroupKFold con subject_id (solo UniMiB; UniZgFall ricade su "stratified")
CV_MODE = os.environ.get("FD_CV_MODE", "stratified")

# Verifica rapida: 2 fold, pochi record per classe, modelli ridotti. Risultati in results/quick/.
QUICK_TEST = os.environ.get("FD_QUICK_TEST", "0") == "1"
QUICK_N_SPLITS = 2
QUICK_PER_CLASS = 6
QUICK_RF_ESTIMATORS = 10
QUICK_CATBOOST_ITERATIONS = 20

# Se True, riesegue gli esperimenti anche se i risultati esistono gia' in results/
FORCE_RERUN = False

# Parametri dei modelli
RF_N_ESTIMATORS = 100  # i notebook originali usano 10000 solo per features/UniZgFall
CATBOOST_ITERATIONS = None  # None = default CatBoost (1000)
CATBOOST_LOG_EVERY = 100  # stampa ogni N iterazioni; 0 = nessun log
SVM_PARAMS = dict(probability=True)  # resto: default scikit-learn
RF_PARAMS = dict(random_state=SEED)
CATBOOST_PARAMS = dict(random_seed=SEED, allow_writing_files=False)

# Lunghezza di padding dei dati grezzi (valori dei notebook originali).
# Se la lunghezza reale massima la supera, viene usata quella reale. None = sempre il massimo reale.
RAW_MAX_LENGTH = {"UniZgFall": 5150, "UniMiB": 3208}

UNIMIB_CLASSES = [
    "Walking", "Running", "GoingUpS", "GoingDownS", "Jumping", "SittingDown",
    "StandingUpFS", "LyingDownFS", "StandingUpFL", "FallingForw", "FallingBack",
    "FallingRight", "FallingLeft", "Syncope", "FallingBackSC", "FallingWithPS",
    "HittingObstacle",
]
UNIZGFALL_CLASSES = ["MW", "MR", "MJ", "WD", "WU", "FF", "FS", "FB", "LD", "OT"]

DATASETS = {
    "UniZgFall": dict(
        data_dir=SOURCE_DIR / "UniZgFall",
        labels_file=SOURCE_DIR / "y_UniZgFall.csv",
        class_names=UNIZGFALL_CLASSES,
        subject_regex=None,  # nessun ID soggetto nei nomi dei file
    ),
    "UniMiB": dict(
        data_dir=SOURCE_DIR / "UniMiB_raw",
        labels_file=SOURCE_DIR / "y_UniMiB_SHAR.csv",
        class_names=UNIMIB_CLASSES,
        subject_regex=r"subject_(\d+)_file_(\d+)\.csv$",
    ),
}

# Classi di caduta per la metrica binaria caduta vs ADL.
# UniMiB: 9-16 (Falling*, Syncope, HittingObstacle). UniZgFall: NON verificabile -> None (metrica non calcolata).
# Per abilitarla inserire una lista di codici label, es. [5, 6, 7].
FALL_CLASSES = {"UniMiB": list(range(9, 17)), "UniZgFall": [5, 6, 7]}