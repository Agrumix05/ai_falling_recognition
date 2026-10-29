# fall_detection_ml

Riclassificazione di cadute / attività quotidiane da accelerometri (UniZgFall, UniMiB SHAR) con SVM, Random Forest e CatBoost, su feature ingegnerizzate e su dati grezzi. Replica la logica di `../ml_accelerometers/` (che non viene mai modificata, solo letta).

## Esecuzione (in ordine)
1. `creating_csv_features_dataset.ipynb` → genera i file in `data/` (qualche minuto).
2. `model_svm.ipynb`, `model_svm_scaled.ipynb`, `model_rf.ipynb`, `model_catboost.ipynb` (indipendenti tra loro; CatBoost e SVM sui dati grezzi sono lenti). `model_svm_scaled.ipynb` è la variante SVM con `StandardScaler` (`utils.build_estimator("svm_scaled")`): risultati con prefisso `svm_scaled_`, senza toccare quelli della SVM originale.

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
| `results/{modello}_{ds}_{features\|raw}_{cv}_folds.csv` | metriche per fold (accuracy, F1/precision/recall macro, tempi, metrica binaria caduta). Stesso schema per i modelli avanzati in `results/advanced/`; i quick test stanno in `quick/` sotto ciascuna delle due cartelle |
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
- Nessuno scaling né selezione di feature (come negli originali) per SVM, RF e CatBoost: quindi nessuna possibilità di leakage da preprocessing. Unica eccezione: la variante `svm_scaled` (Pipeline `StandardScaler` + `SVC`), dove lo scaler è fittato solo sul training di ciascun fold; sul raw scala per colonna il vettore piatto con padding (le colonne costanti a zero restano a zero, quelle quasi sempre a zero possono avere valori scalati elevati).
- `CV_MODE="group"`: `StratifiedGroupKFold(shuffle=True, random_state=89)` su `subject_id` (solo UniMiB); per UniZgFall viene segnalato e usato `stratified`. Con questo metodo i fold possono avere dimensioni diverse.
- UniMiB: classi di caduta 9–16 (Falling*, Syncope, HittingObstacle). Sensibilità/specificità/precision binarie in `*_folds.csv` (`n/c` dove non definite).
- Con scikit-learn recente `SVC(probability=True)` emette un `FutureWarning` (parametro deprecato): è atteso.

## Informazioni non verificabili
- Frequenza di campionamento e unità di misura (g o m/s²) di entrambi i dataset: non documentate nei file.
- Legenda dei suffissi UniZgFall (MW, MR, MJ, WD, WU, FF, FS, FB, LD, OT): non documentata; i nomi sono quelli usati nei notebook originali. La configurazione corrente contiene `FALL_CLASSES["UniZgFall"] = [5, 6, 7]`; questa mappatura non è verificata dal progetto e va considerata un'impostazione utente, non una conclusione scientifica.
- `subject_id` per UniZgFall: non disponibile (colonna vuota); non è possibile una CV per soggetto.

## Esperimenti aggiuntivi

I nuovi notebook sono additivi e usano `advanced_models.py` (solo i dettagli specifici dei modelli); caricamento, fold, metriche, riepilogo e grafici sono quelli di `utils.py` (`utils.summary_table/plot_all(model, advanced=True)`). I file sono separati in `results/advanced/` (o `results/advanced/quick/`) e hanno lo stesso prefisso dei modelli classici: `{modello}_{ds}_{features|raw}_{cv}_*`.

### Matrice modello × rappresentazione

| Modello | features (94, `(n, 94)`) | raw |
|---|---|---|
| SVM, `svm_scaled`, RF, CatBoost | sì | sì, vettore piatto `(n, 3·T)` |
| XGBoost | sì | sì, vettore piatto `(n, 3·T)` |
| MiniRocket, MultiRocket | no: le feature non hanno asse temporale | sì, `(n, 3, T)` |
| CNN 1D, InceptionTime, ResNet 1D | no: convoluzioni su feature tabellari senza significato fisico | sì, `(n, 3, T)` |

### Preprocessing e protocollo per modello

| Modello / input | Normalizzazione | Padding e lunghezze | Validation interna |
|---|---|---|---|
| Classici e XGBoost / features | nessuna (`svm_scaled`: `StandardScaler` sul training fold) | feature calcolate sul segnale originale senza padding | no |
| Classici e XGBoost / raw piatto | nessuna (`svm_scaled`: `StandardScaler` per colonna sul training fold) | zeri di padding visibili; `n_samples` non usato, ma la lunghezza è codificata implicitamente dalla posizione dei campioni non nulli | no |
| MiniRocket, MultiRocket / raw | nessuna dal progetto; parametri Rocket stimati per serie (`FitInTransform`) | zeri visibili, lunghezze non usate | no (Ridge con CV interna sugli alpha) |
| CNN 1D, InceptionTime, ResNet 1D / raw | z-score per asse, stimato solo sui campioni validi del fit set (esclusi validation e test) | pooling finale e normalizzazione mascherati con `n_samples`; le convoluzioni vedono comunque gli zeri a destra ai bordi | sì: 20% del training fold, early stopping (patience 5) su val loss, mai sul test |

Fold e record sono gli stessi per tutti i modelli (`utils.make_splitter`, seed 89). La CV di default è stratificata per registrazione, non per soggetto.

| Notebook | Modello e input | Dipendenza aggiunta |
|---|---|---|
| `model_xgboost.ipynb` | XGBoost multiclasse su 94 feature e su raw piatto, entrambi i dataset; salva feature importance (solo features) | `xgboost` |
| `model_minirocket.ipynb` | MiniRocket multivariato + Ridge sui raw, entrambi i dataset | `sktime` |
| `model_multirocket.ipynb` | MultiRocket multivariato + Ridge sui raw, entrambi i dataset | `sktime` |
| `model_cnn1d.ipynb` | CNN 1D sui raw, entrambi i dataset | `torch` |
| `model_inceptiontime.ipynb` | Blocchi 1D ispirati a InceptionTime sui raw; implementazione compatta, non una riproduzione certificata della configurazione del paper | `torch` |
| `model_resnet1d.ipynb` | ResNet 1D residuale per serie temporali sui raw | `torch` |

Dipendenze: `xgboost`, `sktime` e `torch` sono in `requirements.txt` senza versioni. Nel venv del progetto risultano installati (xgboost 3.4.1, sktime 1.2.0, torch 2.14.1); non è stato installato nulla per questa estensione. I notebook indicano la dipendenza mancante in caso di esecuzione. Il test rapido (`FD_QUICK_TEST=1`) è stato eseguito per tutti e sei i modelli avanzati e per `svm_scaled`; gli esperimenti completi dei modelli avanzati non sono stati eseguiti.

Uso: eseguire il notebook desiderato dopo `creating_csv_features_dataset.ipynb`. Esempi da terminale, dalla cartella del progetto:

```bash
FD_QUICK_TEST=1 jupyter nbconvert --execute --to notebook --inplace model_xgboost.ipynb
FD_QUICK_TEST=1 jupyter nbconvert --execute --to notebook --inplace model_minirocket.ipynb
```

In `QUICK_TEST` sono usati i 2 fold e il sottoinsieme definiti dalla configurazione esistente; sono ridotti gli alberi XGBoost, i kernel Rocket e batch/epoche Torch. In esecuzione completa, i fold seguono `CV_MODE` e riutilizzano `utils.make_splitter`. Metriche per fold, predizioni, tempi, confusion matrix, grafici e curve di apprendimento (per i modelli Torch) sono salvati con prefissi specifici per modello e dataset. Le trasformazioni Rocket sono apprese/trasformate per serie tramite `FitInTransform`; la matrice trasformata non rappresenta assi fisici.

Raw input: i file sono ricostruiti dall'ordine di flatten preesistente in `(N, 3, L)`, mantenendo le colonne x/y/z e il padding a destra già generato. MiniRocket/MultiRocket richiedono lunghezze uniformi: il padding viene mantenuto (nessun ricampionamento, dunque i tratti di padding possono influire sulle feature Rocket). Per le reti Torch, `n_samples` maschera il pooling sui campioni padding; la normalizzazione per asse viene stimata solo nel training fold e l'early stopping usa un validation split interno al training (group-aware quando `CV_MODE="group"`). L'inferenza e la valutazione restano sul test fold esterno. I risultati raw non sono perfettamente confrontabili con gli esperimenti sulle feature per la differenza di input/preprocessing.

### AccNet24

Non è stato creato un notebook AccNet24. Il paper ([Farrahi et al., 2023](https://doi.org/10.1016/j.ijmedinf.2023.105004)) descrive la classificazione di quattro categorie di comportamento giornaliero da finestre non sovrapposte di 30 secondi di accelerometria triassiale al polso; la pipeline converte le finestre in immagini GAF, estrae rappresentazioni con una CNN e usa un BiLSTM. Il paper è open access CC BY 4.0, ma questo **non** è una licenza software. Non è stato trovato un repository ufficiale pubblico né codice/pesi con licenza; l'assenza di risultati nelle ricerche non prova che non esistano archivi non pubblici. I dettagli esatti del backbone CNN e della riduzione dimensionale non sono stati verificati nel full text accessibile. Le registrazioni qui disponibili sono clip brevi etichettate per attività/caduta, non sequenze comportamentali di 24 ore: tassonomia, granularità temporale e contesto sono diversi. Adattare la pipeline a fall detection sarebbe un esperimento distinto, non una replica scientifica di AccNet24. Nessuna sequenza di 24 ore è stata simulata e il modello è lasciato non implementato per questi motivi metodologici. Riferimenti: [DOI del paper](https://doi.org/10.1016/j.ijmedinf.2023.105004), [PubMed](https://pubmed.ncbi.nlm.nih.gov/36724729/), [profilo GitHub del corresponding author](https://github.com/farrahi) (nessun repository AccNet24 identificato).

### Validazione dell'estensione

È stato verificato che `advanced_models.py` e `utils.py` si importano, che i notebook sono JSON/nbformat validi e che il caricamento raw produce la forma `(N, 3, L)` coerente con `n_samples` (modelli sequenziali) o `(N, 3·L)` (XGBoost raw). Con `FD_QUICK_TEST=1` sono stati eseguiti XGBoost (features e raw), MiniRocket, MultiRocket, CNN 1D, InceptionTime, ResNet 1D e `svm_scaled` su entrambi i dataset, controllando che i fold coincidano con quelli dei modelli classici. L'accuratezza dei quick test (pochi record, 2 epoche, 84 kernel) non è significativa. Non sono stati eseguiti gli esperimenti completi dei modelli avanzati. La configurazione corrente `FALL_CLASSES` viene riutilizzata senza modificarla; la metrica binaria riflette quindi la mappatura impostata dall'utente, anche dove non verificata.
