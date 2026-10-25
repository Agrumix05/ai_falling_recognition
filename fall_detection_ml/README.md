# fall_detection_ml

Riclassificazione di cadute / attività quotidiane da accelerometri (UniZgFall, UniMiB SHAR) con SVM, Random Forest e CatBoost, su feature ingegnerizzate e su dati grezzi. Replica la logica di `../ml_accelerometers/` (che non viene mai modificata, solo letta).

## Esecuzione (in ordine)
1. `creating_csv_features_dataset.ipynb` → genera i file in `data/` (qualche minuto).
2. `model_svm.ipynb`, `model_rf.ipynb`, `model_catboost.ipynb` (indipendenti tra loro; CatBoost e SVM sui dati grezzi sono lenti).

Verifica rapida senza addestramento completo: `QUICK_TEST = True` in `config.py` oppure variabile d'ambiente `FD_QUICK_TEST=1` (2 fold, 6 record per classe, RF 10 alberi, CatBoost 20 iterazioni; risultati in `results/quick/`, ignorata da git).
Modalità CV: `CV_MODE = "stratified"` (baseline) o `"group"` (env `FD_CV_MODE`). I risultati sono salvati con il nome della modalità nel file; se esistono già vengono saltati (`FORCE_RERUN = True` per riaddestrare). Per rifare solo i plot basta rieseguire l'ultima cella.

## Contenuto
| File | Descrizione |
|---|---|
| `config.py` | percorsi, seed (89), n_splits, `CV_MODE`, `QUICK_TEST`, parametri modelli, mappatura classi di caduta |
| `utils.py` | caricamento, 94 feature, CV, metriche, salvataggio risultati, plot |
| `data/features_{UniZgFall,UniMiB}.csv` | `record_id`, `subject_id`, 94 feature, `label`, `label_name` |
| `data/raw_{ds}_X.npy` | registrazioni con zero-padding in coda, flatten (N × L·3, ordine x,y,z per campione) |
| `data/raw_{ds}_meta.csv` | `record_id`, `subject_id`, `label`, `label_name`, `n_samples` (lunghezza reale) |
| `results/{modello}_{ds}_{features\|raw}_{cv}_folds.csv` | metriche per fold (accuracy, F1/precision/recall macro, tempi, metrica binaria caduta) |
| `..._preds.csv`, `..._confmat.npy`, `..._importance.csv` | predizioni per record, confusion matrix per fold (normalizzate per riga), importanza media (solo RF/CatBoost su feature) |
| `results/*.png` | boxplot, confusion matrix media (%), tempi di fit, top‑20 importanza |

## Assunzioni
- Label assegnate per posizione dopo ordinamento naturale (come negli originali); `assert` sul numero di file e, per UniZgFall, sulla coincidenza tra suffisso del file e label (verificata).
- UniZgFall: usate le prime 3 colonne (la 4ª è il tempo). UniMiB: intestazione `# x,y,z` ignorata.
- 94 feature nello stesso ordine degli originali (50 temporali + 44 sul modulo FFT, bin 1..n//2−1, senza DC). Deviazione standard con ddof=0.
- Padding raw: 5150 (UniZgFall) e 3208 (UniMiB), come negli originali; se un file fosse più lungo si usa il massimo reale.
- F1 macro, precision, recall: `sklearn` con `zero_division=0`. Gli originali usavano un F1 custom che vale 1.0 per le classi assenti sia nel vero sia nel predetto: i valori possono differire leggermente.
- RF: 100 alberi (gli originali usano 10000 solo per feature/UniZgFall; modificabile in `config.py`).
- CatBoost: oltre a `random_seed=89`, `verbose` a intervalli (`CATBOOST_LOG_EVERY`) invece di `False` e `allow_writing_files=False` (nessuna cartella `catboost_info`); nessun effetto sul modello.
- Nessuno scaling né selezione di feature (come negli originali): quindi nessuna possibilità di leakage da preprocessing.
- `CV_MODE="group"`: `StratifiedGroupKFold(shuffle=True, random_state=89)` su `subject_id` (solo UniMiB); per UniZgFall viene segnalato e usato `stratified`. Con questo metodo i fold possono avere dimensioni diverse.
- UniMiB: classi di caduta 9–16 (Falling*, Syncope, HittingObstacle). Sensibilità/specificità/precision binarie in `*_folds.csv` (`n/c` dove non definite).
- Con scikit-learn recente `SVC(probability=True)` emette un `FutureWarning` (parametro deprecato): è atteso.

## Informazioni non verificabili
- Frequenza di campionamento e unità di misura (g o m/s²) di entrambi i dataset: non documentate nei file.
- Legenda dei suffissi UniZgFall (MW, MR, MJ, WD, WU, FF, FS, FB, LD, OT): non documentata; i nomi sono quelli usati nei notebook originali. Per questo `FALL_CLASSES["UniZgFall"] = None` e la metrica binaria risulta **non calcolata**; va compilata in `config.py` solo dopo averla verificata.
- `subject_id` per UniZgFall: non disponibile (colonna vuota); non è possibile una CV per soggetto.
