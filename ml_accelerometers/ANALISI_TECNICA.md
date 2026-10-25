# Analisi tecnica e metodologica del progetto di riconoscimento delle cadute

## Ambito e metodo

Questa relazione descrive il contenuto attuale dei quattro notebook, dei CSV, delle label e delle figure salvate nel repository. I notebook non sono stati rieseguiti: i numeri dei risultati riportati sotto provengono dagli output già serializzati nei file `.ipynb`. Le forme di `X` e le statistiche dei CSV sono state ricostruite staticamente e con una lettura sequenziale dei file; non è stato eseguito alcun addestramento.

Al momento dell'analisi il worktree conteneva già modifiche non committate ai quattro notebook e a sei immagini. L'analisi descrive quindi la versione presente, non la versione del commit di base. Le modifiche preesistenti sono state lasciate intatte.

## 1. Executive summary

Il progetto confronta classificatori convenzionali su registrazioni di accelerometro provenienti da UniZgFall e UniMiB SHAR. Per ogni file viene assegnata una singola label e viene prodotto un singolo esempio di training: i notebook `features` calcolano 94 statistiche temporali e spettrali per registrazione; i notebook `raw` passano al classificatore i tre assi grezzi, azzerati in coda fino a una lunghezza fissa e poi appiattiti in un vettore.

Sono istanziati e addestrati una SVM, una Random Forest e un CatBoost. È inoltre valutato un predittore denominato `ALL`, che sceglie per ogni esempio il modello con la massima probabilità della propria classe più probabile. La valutazione è una cross-validation stratificata casuale a 10 fold sui file, non una validazione indipendente per soggetto. Non c'è un test set finale.

Gli output salvati mostrano, in sintesi, che:

| Notebook | Accuracy media salvata | F1 macro media salvata |
|---|---:|---:|
| `training_features.ipynb` | CatBoost 0,92; ALL 0,92 | CatBoost 0,91; ALL 0,91 |
| `training_features_UniMiB.ipynb` | CatBoost 0,70; ALL 0,70 | CatBoost 0,74; ALL 0,74 |
| `training_raw.ipynb` | SVM 0,78; ALL 0,78 | SVM 0,68; ALL 0,69 |
| `training_raw_UniMiB.ipynb` | CatBoost 0,56; ALL 0,56 | CatBoost 0,59; ALL 0,59 |

Sono medie di metriche calcolate per fold negli output del notebook, non prestazioni su un test set indipendente. Non sono confrontabili direttamente tra dataset: cambiano classi, numero di esempi, codifica e rappresentazione del segnale.

La criticità metodologica più importante è la ripartizione dei dati a livello di registrazione anziché di soggetto. In UniMiB i nomi dei file identificano 30 soggetti con 66 registrazioni ciascuno; la cross-validation casuale può quindi mettere registrazioni dello stesso soggetto sia nel training sia nel test. Le prestazioni non dimostrano generalizzazione a soggetti mai visti.

## 2. Architettura del repository e dipendenze

### Notebook analizzati

1. [`training_features.ipynb`](./training_features.ipynb): UniZgFall, 94 feature ingegnerizzate.
2. [`training_features_UniMiB.ipynb`](./training_features_UniMiB.ipynb): UniMiB, 94 feature ingegnerizzate.
3. [`training_raw.ipynb`](./training_raw.ipynb): UniZgFall, segnale grezzo appiattito.
4. [`training_raw_UniMiB.ipynb`](./training_raw_UniMiB.ipynb): UniMiB, segnale grezzo appiattito.

### File e relazioni

- [`README.md`](./README.md) presenta il lavoro di A. Antonietti, indica il paper del 2023 IEEE Signal Processing in Medicine and Biology Symposium e spiega la distinzione tra notebook `features` e `raw`. Il README dichiara 94 feature estratte.
- [`UniZgFall/`](./UniZgFall/) contiene 468 CSV; [`y_UniZgFall.csv`](./y_UniZgFall.csv) contiene 468 label numeriche.
- [`UniMiB_raw/`](./UniMiB_raw/) contiene 1.980 CSV; [`y_UniMiB_SHAR.csv`](./y_UniMiB_SHAR.csv) contiene 1.980 label numeriche, oltre all'intestazione commentata.
- I notebook generano i dodici PNG nella directory di lavoro. Le figure `acc_*` e `F1_*` mostrano i punteggi per fold di SVM, RF, CAT e ALL. Ciascuna figura `Confusion_CAT_*` salva invece la matrice di CatBoost.
- `catboost_info/` contiene artefatti di training CatBoost, non è una dipendenza richiesta esplicitamente dal codice. È esclusa dal controllo versione dalle regole locali.
- Il solo documento di progetto è il README; non sono presenti `requirements.txt`, `pyproject.toml` o file di ambiente con dipendenze bloccate. La licenza presente è GPL-3.0; la licenza specifica dei dati non è documentata localmente.

I notebook usano percorsi relativi (per esempio `./UniZgFall` e `y_UniZgFall.csv`): per eseguirli dall'inizio occorre impostare la directory corrente su `ml_accelerometers/`. Non sono notebook autonomi se lanciati dalla root del repository senza adattare la directory di lavoro. Il kernel è Python 3.12.13 nei metadata della maggior parte dei notebook; `training_raw_UniMiB.ipynb` riporta invece il nome generico `Python 3 (ipykernel)`.

Versioni osservate nell'ambiente locale al momento dell'analisi: Python 3.12.13, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, Matplotlib 3.11.2, scikit-learn 1.9.1, CatBoost 1.2.10, natsort 8.4.0, tqdm 4.70.1 e scikit-posthocs 0.17.1. Sono versioni installate nell'ambiente corrente, non una specifica riproducibile del progetto.

## 3. Dataset, file e label

### 3.1 UniZgFall

La directory contiene 468 CSV senza intestazione: ciascuna riga ha quattro valori numerici, commentati nel codice come `X Y Z time`. Le prime tre colonne sono usate come accelerazioni; la quarta è ignorata sia nel calcolo delle feature sia nel preprocessing raw. Una scansione sequenziale ha verificato 4 campi numerici finiti per tutte le righe lette.

Le registrazioni contengono da 388 a 5.146 campioni (mediana 1.525). Il campo temporale di alcuni record avanza di circa 0,005 tra righe, ma il repository non documenta unità, frequenza nominale, calibrazione o orientamento del sensore: non è quindi corretto dichiarare una frequenza o una durata assoluta come specifica garantita.

I nomi sono del tipo `Folder_fileNNN_SUFFIX.csv`. Ordinando naturalmente i nomi e associando in sequenza il vettore di label, il suffisso determina la codifica osservata:

| Label | Suffisso | File/campioni |
|---:|---|---:|
| 0 | MW | 68 |
| 1 | MR | 68 |
| 2 | MJ | 68 |
| 3 | WD | 13 |
| 4 | WU | 13 |
| 5 | FF | 34 |
| 6 | FS | 34 |
| 7 | FB | 34 |
| 8 | LD | 34 |
| 9 | OT | 102 |

Le abbreviazioni sono anche le etichette usate dalle matrici di confusione. Il README descrive il tema generale come classificazione di attività quotidiane e tipi di caduta, ma nei documenti locali non è presente una legenda completa che espanda con certezza ogni suffisso. Non si attribuisce quindi un significato più specifico a `MW`, `MR`, `MJ`, `WD`, `WU`, `FF`, `FS`, `FB`, `LD` e `OT`.

### 3.2 UniMiB SHAR

`UniMiB_raw` contiene 1.980 CSV con nomi `subject_0_file_0.csv` ... `subject_29_file_65.csv`: si verificano 30 identificativi di soggetto, ciascuno con 66 file. I CSV hanno tre colonne `x,y,z`; l'intestazione è una riga commentata (`# x,y,z`), ignorata da `np.genfromtxt`. Le righe di segnale sono numeriche e finite. Le lunghezze osservate vanno da 85 a 1.863 campioni, con mediana 206. Non sono presenti timestamp nel file e il repository non specifica la frequenza di campionamento.

La label non è codificata nel nome del file: viene caricata dal CSV separato e associata per posizione dopo `natsorted(os.listdir(...))`. Le quantità coincidono (1.980 file e 1.980 label); la correttezza dell'allineamento dipende pertanto dall'ordine dei file previsto dal dataset e non viene verificata mediante metadata per singolo record.

La lista delle classi mostrata nei notebook stabilisce la corrispondenza fra codici e nomi:

| Codice | Nome visualizzato | Campioni |
|---:|---|---:|
| 0 | Walking | 60 |
| 1 | Running | 60 |
| 2 | GoingUpS | 60 |
| 3 | GoingDownS | 60 |
| 4 | Jumping | 60 |
| 5 | SittingDown | 60 |
| 6 | StandingUpFS | 60 |
| 7 | LyingDownFS | 60 |
| 8 | StandingUpFL | 60 |
| 9 | FallingForw | 180 |
| 10 | FallingBack | 180 |
| 11 | FallingRight | 180 |
| 12 | FallingLeft | 180 |
| 13 | Syncope | 180 |
| 14 | FallingBackSC | 180 |
| 15 | FallingWithPS | 180 |
| 16 | HittingObstacle | 180 |

I nomi provengono dalla lista `classes` nei notebook; non vengono espansi ulteriormente perché alcune sigle interne (per esempio `FS`, `FL`, `SC`, `PS`) non sono definite nel repository. Nei dati le classi 0–8 hanno 60 campioni ciascuna e le classi 9–16 ne hanno 180 ciascuna.

## 4. Feature extraction

L'estrazione si trova nella cella di training dei due notebook `features` (cella 6 in UniZgFall, cella 7 in UniMiB). Non si caricano feature precalcolate: per ogni file il codice legge le prime tre colonne, mette le sequenze in `x_list`, `y_list`, `z_list`, calcola le statistiche e costruisce `X`, un `DataFrame` con 94 colonne e una riga per file. Non vengono usate finestre mobili: ogni feature riassume l'intera registrazione.

Sia \(s=(s_1,\ldots,s_n)\) la sequenza di un asse. Per ciascuno di x, y e z sono calcolate le seguenti 16 statistiche temporali:

| Suffisso del nome | Calcolo | Interpretazione |
|---|---|---|
| `_mean` | \(\frac{1}{n}\sum s_i\) | livello medio dell'asse |
| `_std` | deviazione standard NumPy, `ddof=0` | dispersione |
| `_aad` | \(\frac{1}{n}\sum |s_i-\bar{s}|\) | scarto assoluto medio |
| `_min`, `_max` | minimo e massimo | estremi |
| `_maxmin_diff` | `max - min` | escursione |
| `_median` | mediana | posizione centrale robusta |
| `_mad` | mediana di \(|s_i-\mathrm{median}(s)|\) | dispersione robusta |
| `_IQR` | percentile 75 meno percentile 25 | intervallo interquartile |
| `_neg_count` | conteggio di \(s_i<0\) | numero di campioni sotto zero |
| `_pos_count` | conteggio di \(s_i>0\) | numero di campioni sopra zero |
| `_above_mean` | conteggio di \(s_i>\bar{s}\) | numero di campioni sopra la media |
| `_peak_count` | `len(find_peaks(s)[0])` | massimi locali secondo i default SciPy |
| `_skewness` | `scipy.stats.skew(s)` | asimmetria |
| `_kurtosis` | `scipy.stats.kurtosis(s)` | curtosi di Fisher (default SciPy) |
| `_energy` | \(\sum s_i^2/n\) | media del quadrato del segnale |

Queste 16 colonne per asse producono 48 feature temporali. Si aggiungono:

- `avg_result_accl`: media della magnitudine risultante \(\sqrt{x_i^2+y_i^2+z_i^2}\).
- `sma`: somma delle medie dei valori assoluti dei tre assi. È un descrittore di attività aggregata, non un calcolo di area tramite finestre.

Per ogni asse viene poi calcolato `abs(np.fft.fft(s))[1:n//2]`: si scartano il bin DC e la metà superiore dello spettro. La trasformata non è normalizzata e non è convertita in frequenza fisica perché non viene passata la frequenza di campionamento. Sul modulo dei bin selezionati si calcolano 14 statistiche: media, deviazione standard, scarto assoluto medio, minimo, massimo, range, mediana, MAD, IQR, conteggio sopra la media, picchi, skewness, kurtosis ed energia. I nomi corrispondono alle feature sopra con suffisso `_fft`; non esistono `neg_count_fft` o `pos_count_fft`. Sono 42 colonne. Si aggiungono `avg_result_accl_fft` e `sma_fft`.

Il totale è dunque \(48+2+42+2=94\) scalari per record. In termini di nomi, ogni statistica per asse è prefissata da `x_`, `y_` o `z_`; le versioni spettrali sono suffissate da `_fft`. Non vengono calcolate correlazioni tra assi, RMS esplicito, frequenze dominanti, bande di frequenza, filtri, trasformate wavelet o feature per sottofinestre.

Alcune feature dipendono dalla durata: conteggi di segni, conteggio sopra la media e picchi sono conteggi assoluti, mentre la FFT non normalizzata ha ampiezze dipendenti dalla lunghezza. Questo è rilevante perché la lunghezza delle registrazioni varia molto.

## 5. Costruzione di X e y e pipeline dei notebook

In tutti i notebook `y` è caricato con `np.loadtxt(...).astype(int)`; i file hanno una colonna di codici numerici, senza encoding one-hot. I CSV sono letti con `np.genfromtxt(..., delimiter=',')`. Le label vengono associate ai file per indice nell'ordine naturale di `FILES`; il codice non controlla esplicitamente che le lunghezze coincidano prima dell'addestramento, anche se le cardinalità dei file e delle label verificate sono uguali.

Nei notebook `features`, `X` è un `pandas.DataFrame`; nei notebook `raw` è un array NumPy 2D dopo padding e flatten. Non sono creati tensori. Non vengono applicati standardizzazione/normalizzazione, filtri, bilanciamento o pesi di classe, imputazione, selezione di feature o riduzione dimensionale. La scansione dei CSV non ha rilevato righe malformate o valori non finiti nei campi numerici, ma il codice dei notebook non contiene un controllo o una gestione esplicita dei missing value. Lo shuffle è quello di `StratifiedKFold`; non c'è un ulteriore shuffle manuale.

### 5.1 `training_features.ipynb` — UniZgFall

Ordine operativo delle celle:

1. Cella 0 importa NumPy, pandas, SciPy, Matplotlib, scikit-learn, CatBoost, tqdm e scikit-posthocs; imposta `FOLDNAME="UniZgFall"` e ordina naturalmente i file della directory.
2. Cella 1 definisce `CI95`, che calcola due statistiche d'ordine attorno alla mediana; non è un intervallo t o bootstrap sulla media.
3. Cella 2 definisce F1 macro su 10 classi, indicizzate da 1 a 10.
4. Celle 3–5 impostano lo stile dei grafici.
5. Cella 6 carica le label 0–9, legge x/y/z da ogni registrazione e costruisce le 94 colonne descritte sopra.
6. Cella 7 addestra/valuta SVC, Random Forest, CatBoost e il selettore `ALL` con 10 fold. In questo notebook `X` è un DataFrame.
7. Cella 8 stampa medie, deviazioni standard, mediane e intervalli d'ordine per accuracy, F1 e tempi.
8. Cella 9 applica Kruskal–Wallis e Dunn post-hoc ai vettori dei fold.
9. Cella 10 salva le figure di accuracy e F1; la disposizione mostra i punteggi di ciascun fold e un boxplot.
10. Cella 11 costruisce le confusion matrix medie. Salva solo quella CatBoost con nome `Confusion_CAT_features.png`; il codice crea anche i grafici SVM, RF e ALL ma non li salva con altri nomi.
11. Celle 12–13 calcolano e stampano importanze RF/CatBoost medie sui fold e un coefficiente di correlazione fra i due vettori.

Forma ricostruita: `X` = 468 × 94, `y` = 468. Le label sono codici 0–9; per la F1 il codice aggiunge 1 a predizioni e target in modo da allinearsi agli indici 1–10 della funzione custom.

### 5.2 `training_features_UniMiB.ipynb` — UniMiB SHAR

La pipeline è pressoché la stessa del notebook precedente, ma la cella 0 seleziona `UniMiB_raw`; la cella 3 definisce `MAA`; la cella 7 carica `y_UniMiB_SHAR.csv` e crea `X`; la cella 8 addestra i modelli. `MAA(TARGET, RESULT)` calcola il richiamo per ciascuna delle 17 classi e ne fa la media, cioè macro-recall. Le label sono già 0–16, quindi non sono traslate per F1.

Ordine delle celle: setup e helper nelle celle 0–3, figure nelle celle 4–6, caricamento/feature extraction in 7, CV in 8, riepilogo in 9, test statistici in 10, grafici di score in 11, confusion matrix in 12 e importanza feature in 13. La cella 12 salva solo la matrice CatBoost come `Confusion_CAT_features_UniMiB.png`.

Forma ricostruita: `X` = 1.980 × 94, `y` = 1.980. I dati sono letti dal CSV grezzo; le feature non sono importate da file esterni.

### 5.3 `training_raw.ipynb` — UniZgFall

Le celle di setup/helper sono come in `training_features.ipynb`. La cella 6 carica ogni registrazione e aggiunge a `X` la matrice `D[:, 0:3]`. Imposta `max_length=5150`, `input_dim=3`, crea `X_padded` di forma 468 × 5150 × 3, copia il segnale all'inizio e lascia zeri in coda; infine appiattisce con `reshape` in `X` di forma 468 × 15.450.

Non è feature extraction: il reshape conserva i valori dei campioni e li converte in un vettore ordinato nel tempo (x,y,z per timestep). Il timestamp della quarta colonna viene scartato. La lunghezza massima osservata è 5.146, quindi la costante copre gli esempi presenti, con al più quattro righe di padding per il record più lungo.

Le celle successive eseguono la stessa CV a 10 fold, riepilogano accuracy/F1/tempi, eseguono test statistici e salvano `acc_raw.png`, `F1_raw.png` e la matrice CatBoost `Confusion_CAT_raw.png`. Non viene calcolata feature importance per i vettori grezzi.

### 5.4 `training_raw_UniMiB.ipynb` — UniMiB SHAR

La cella 6 carica gli assi x/y/z, imposta `max_length=3208` e costruisce `X_padded` di forma 1.980 × 3.208 × 3, poi la appiattisce in `X` di forma 1.980 × 9.624. La lunghezza effettiva massima verificata è 1.863; la costante aggiunge quindi un considerevole tratto nullo anche al file più lungo. Il segnale non ha maschera di validità, centratura, riscalamento o normalizzazione. Il resto della pipeline segue quella raw UniZgFall e genera gli artefatti `*_raw_UniMiB.png`.

## 6. Modelli e iperparametri

| Modello | Stato nel codice | Parametri espliciti | Implicazioni/default rilevanti |
|---|---|---|---|
| `SVC` | Importato, istanziato, addestrato e valutato in tutti i notebook | `probability=True` | Gli altri parametri sono default scikit-learn (kernel RBF, `C=1`, `gamma='scale'`). La probabilità è richiesta dal predittore ALL. `random_state` non è fissato; gli output serializzati riportano anche un FutureWarning relativo a `probability=True` nell'ambiente corrente. |
| `RandomForestClassifier` | Importato, istanziato, addestrato e valutato in tutti i notebook | `n_estimators=10000` solo in `training_features.ipynb`; `n_estimators=100` negli altri tre | `random_state` non è fissato. Non vengono impostati pesi di classe, profondità o criteri: si applicano i default della versione installata. |
| `CatBoostClassifier` | Importato, istanziato, addestrato e valutato in tutti i notebook | `random_seed=89`, `verbose=False` | Iterazioni, profondità e learning rate non sono impostati esplicitamente: sono i default CatBoost. |
| `ALL` | Non è una classe importata; è un ensemble scritto nel ciclo | Non ha parametri addestrabili aggiuntivi | Per ogni campione sceglie il modello con probabilità massima massima e copia la sua classe prevista. |
| `KNeighborsClassifier` | Importato soltanto | Nessuno | Non viene istanziato o valutato. |

Non sono presenti Grid Search, Random Search, tuning annidato, selezione automatica delle feature o selezione del modello basata su un validation set separato. Tutti i tre modelli e l'ensemble ricevono lo stesso split nei rispettivi notebook.

L'ensemble confronta `max(p)` fra modelli diversi, non una probabilità per una medesima classe. Le probabilità di SVC, RF e CatBoost non sono calibrate in comune; la regola di selezione è quindi un'euristica, e il confronto di confidenza tra algoritmi può essere distorto.

## 7. Cross-validation e metriche

Tutti e quattro i notebook dichiarano:

```python
StratifiedKFold(n_splits=10, shuffle=True, random_state=89)
```

Il fold è creato sul singolo record (`X`, `y`). Ogni iterazione addestra i classificatori su 9 fold e calcola le metriche sul fold rimanente. La medesima sequenza di indici è usata dai tre modelli e dall'ensemble in quel notebook. Non esiste un test set indipendente, né una seconda CV per il tuning.

Le metriche sono calcolate fold per fold e poi riassunte sui dieci valori con media, deviazione standard e mediana:

- **Accuracy**: \(\frac{\#\{\hat{y}=y\}}{N}\), tramite `.score` (per i classificatori sklearn/CatBoost) o media di uguaglianza per ALL.
- **F1 macro**: il helper calcola \(F1_c=2P_cR_c/(P_c+R_c)\) per ogni classe e ne fa la media non pesata. I calcoli di precision e recall servono alla F1 ma non vengono presentati come metriche indipendenti.
- **MAA**: presente solo in `training_features_UniMiB.ipynb`; media dei recall per classe (macro-recall/balanced accuracy multiclasse).
- **Confusion matrix**: `confusion_matrix(..., normalize="true")`, quindi righe normalizzate; il notebook calcola la media delle dieci matrici e moltiplica per 100 per il grafico. Non è la matrice dei conteggi aggregati su tutte le predizioni.
- **Tempi**: misurazione con `time.time()` dall'inizio del fit fino al termine delle predizioni/probabilità del singolo modello, per fold. I tempi dipendono fortemente dall'ambiente e non sono metriche di accuratezza.
- **Test statistici**: Kruskal–Wallis e Dunn con correzione Bonferroni post-hoc sui punteggi dei fold e, separatamente, sui tempi. I test trattano i gruppi come campioni indipendenti, mentre i punteggi provengono dagli stessi fold e sono appaiati; inoltre ALL è costruito a partire dalle uscite dei modelli confrontati. I p-value sono pertanto da interpretare con cautela, non come prova indipendente che un algoritmo sia migliore.

La funzione chiamata `CI95` usa gli indici d'ordine basati sulla posizione della mediana e una costante 1,645. Non calcola un intervallo standard al 95% per la media (come un intervallo t o bootstrap); le parentesi quadre stampate non vanno interpretate come un CI convenzionale. Le etichette “media ± deviazione standard” descrivono la variabilità dei dieci fold, non l'incertezza su un test indipendente.

Non sono calcolate specificità, ROC-AUC, precision/recall pubblicate separatamente, curve ROC o metriche binarie caduta-vs-ADL. Di conseguenza, i notebook non quantificano direttamente il compromesso fra rilevamento delle cadute e falsi allarmi. L'accuracy può nascondere errori nelle classi minoritarie; F1 macro e MAA sono più sensibili a classi con pochi campioni, ma restano metriche di classificazione offline.

## 8. Output sperimentali salvati

I valori seguenti sono trascritti dagli output già memorizzati nei notebook. Sono proporzioni su scala 0–1, media ± deviazione standard dei fold; non una nuova esecuzione.

| Notebook | Modello | Accuracy | F1 macro | MAA |
|---|---|---:|---:|---:|
| features UniZgFall | SVM | 0,78 ± 0,03 | 0,68 ± 0,07 | — |
|  | RF (10.000 alberi) | 0,91 ± 0,04 | 0,90 ± 0,05 | — |
|  | CatBoost | 0,92 ± 0,04 | 0,91 ± 0,04 | — |
|  | ALL | 0,92 ± 0,04 | 0,91 ± 0,05 | — |
| features UniMiB | SVM | 0,33 ± 0,02 | 0,30 ± 0,02 | 0,33 ± 0,02 |
|  | RF (100 alberi) | 0,67 ± 0,03 | 0,72 ± 0,03 | 0,72 ± 0,03 |
|  | CatBoost | 0,70 ± 0,04 | 0,74 ± 0,04 | 0,74 ± 0,04 |
|  | ALL | 0,70 ± 0,04 | 0,74 ± 0,04 | 0,75 ± 0,03 |
| raw UniZgFall | SVM | 0,78 ± 0,07 | 0,68 ± 0,11 | — |
|  | RF (100 alberi) | 0,73 ± 0,07 | 0,66 ± 0,08 | — |
|  | CatBoost | 0,76 ± 0,07 | 0,67 ± 0,08 | — |
|  | ALL | 0,78 ± 0,08 | 0,69 ± 0,10 | — |
| raw UniMiB | SVM | 0,48 ± 0,04 | 0,51 ± 0,04 | — |
|  | RF (100 alberi) | 0,55 ± 0,04 | 0,57 ± 0,04 | — |
|  | CatBoost | 0,56 ± 0,04 | 0,59 ± 0,04 | — |
|  | ALL | 0,56 ± 0,04 | 0,59 ± 0,05 | — |

I tempi medi di fit e predizione salvati (secondi per fold, media ± deviazione standard) sono:

| Notebook | SVM | RF | CatBoost |
|---|---:|---:|---:|
| features UniZgFall | 0,05 ± 0,02 | 19,72 ± 0,49 | 27,08 ± 0,76 |
| features UniMiB | 0,81 ± 0,01 | 0,89 ± 0,02 | 53,19 ± 1,23 |
| raw UniZgFall | 190,75 ± 88,89 | 1,78 ± 0,22 | 1.517,10 ± 90,55 |
| raw UniMiB | 194,10 ± 30,32 | 5,34 ± 0,73 | 1.721,02 ± 375,91 |

I test omnibus Kruskal–Wallis salvati confrontano i punteggi dei quattro gruppi (SVM, RF, CAT, ALL) e i tempi dei tre modelli:

| Notebook | p accuracy | p F1 | p tempi |
|---|---:|---:|---:|
| features UniZgFall | 6,588 × 10⁻⁵ | 6,071 × 10⁻⁵ | 2,490 × 10⁻⁶ |
| features UniMiB | 2,449 × 10⁻⁵ | 2,635 × 10⁻⁵ | 2,490 × 10⁻⁶ |
| raw UniZgFall | 0,416 | 0,967 | 2,490 × 10⁻⁶ |
| raw UniMiB | 0,001592 | 0,001995 | 2,490 × 10⁻⁶ |

Gli output includono anche tabelle Dunn post-hoc Bonferroni. I p-value non correggono i limiti del protocollo appaiato descritti nella sezione precedente.

I risultati suggeriscono, limitatamente a questi split casuali, una differenza fra le rappresentazioni su UniMiB e una maggiore variabilità dei punteggi raw UniZgFall. Non autorizzano a concludere che uno specifico algoritmo generalizzi meglio a soggetti nuovi.

Le figure `acc_features.png`, `acc_features_UniMiB.png`, `acc_raw.png`, `acc_raw_UniMiB.png` e i corrispondenti file `F1_*.png` sono boxplot con i dieci punteggi per modello e punti dei fold. Le figure `Confusion_CAT_features.png`, `Confusion_CAT_features_UniMiB.png`, `Confusion_CAT_raw.png` e `Confusion_CAT_raw_UniMiB.png` sono matrici CatBoost: il codice mostra SVM, RF, CAT e ALL, ma esegue `savefig` mentre è attiva la matrice CAT; il nome del file lo conferma. Le righe rappresentano la classe vera, le colonne la classe predetta e ogni cella è una percentuale, non un conteggio di record aggregati. Le confusion matrix UniMiB hanno molte etichette lunghe e risultano visivamente tagliate ai margini.

Sei immagini (le tre coppie `acc_features`, `F1_features` e `Confusion_CAT_features`) erano già modificate nel worktree all'inizio dell'analisi. Sono state esaminate nella forma presente; la loro uguaglianza byte-per-byte con i risultati che genererebbe oggi il notebook non è stata verificata. Le immagini raw non risultavano modificate. I numeri della tabella, in ogni caso, provengono dagli output testuali incorporati nei notebook, non da stime ricavate dai pixel delle figure.

## 9. Feature importance e interpretabilità

Solo i notebook `features` raccolgono `rf_clf.feature_importances_` e `cat_clf.feature_importances_` dopo il fit di ciascun fold e ne fanno la media componente per componente. Le importanze vengono ordinate in modo decrescente; per RF sono moltiplicate per 100, mentre quelle CatBoost sono mostrate nella scala fornita dal modello. Non sono importanze dell'ensemble e non esiste importanza per SVM.

Nell'output salvato di UniZgFall, tra le feature RF più alte figurano `y_energy` (3,47), `y_aad_fft` (2,86), `y_energy_fft` (2,81) e `avg_result_accl` (2,54). Per CatBoost compaiono `y_energy` (5,00), `y_mean_fft` (3,76) e `avg_result_accl` (2,85). La correlazione stampata fra i vettori RF e CatBoost è 0,750.

Per UniMiB, fra le prime RF figurano `z_mean` (1,80), `y_aad_fft` (1,74) e `y_above_mean_fft` (1,56); per CatBoost compaiono `z_above_mean` (2,73), `y_above_mean` (2,51) e `z_std_fft` (2,15). La correlazione fra importanze è 0,617.

Questi numeri derivano da importanze native dei modelli ad albero (RF: riduzione impurità; CatBoost: importanza predefinita CatBoost), mediate fra fold. Non dimostrano causalità né stabilità per soggetto: feature correlate possono spartirsi l'importanza, le statistiche di lunghezza possono essere proxy, e gli output stampati mostrano solo parte della tabella. Nei notebook raw le coordinate appiattite sono input, non feature semantiche; non viene calcolata né visualizzata la loro importanza.

## 10. Confronto sistematico

| Notebook | Dataset | Input e dimensione ricostruita | Estrazione feature | Classi | Modelli addestrati | Preprocessing | Validazione | Metriche e output |
|---|---|---|---|---:|---|---|---|---|
| `training_features.ipynb` | UniZgFall | 468 × 94 | Statistiche per registrazione su x/y/z e FFT | 10 | SVC, RF 10.000, CatBoost, ALL | Nessuno scaling; ignora timestamp | Stratified 10-fold casuale | Accuracy, F1 macro, tempi; grafici e importanze RF/CAT |
| `training_features_UniMiB.ipynb` | UniMiB SHAR | 1.980 × 94 | Stesse 94 statistiche | 17 | SVC, RF 100, CatBoost, ALL | Nessuno scaling | Stratified 10-fold casuale | Accuracy, F1 macro, MAA, tempi; grafici e importanze RF/CAT |
| `training_raw.ipynb` | UniZgFall | 468 × 15.450 | Nessuna; padding a 5.150 × 3 e flatten | 10 | SVC, RF 100, CatBoost, ALL | Zeri aggiunti in coda; ignora tempo | Stratified 10-fold casuale | Accuracy, F1 macro, tempi; boxplot e matrice CAT |
| `training_raw_UniMiB.ipynb` | UniMiB SHAR | 1.980 × 9.624 | Nessuna; padding a 3.208 × 3 e flatten | 17 | SVC, RF 100, CatBoost, ALL | Zeri aggiunti in coda; nessuna normalizzazione | Stratified 10-fold casuale | Accuracy, F1 macro, tempi; boxplot e matrice CAT |

La comparazione features-vs-raw all'interno di un dataset utilizza, in linea generale, lo stesso vettore `y` e la stessa configurazione StratifiedKFold, ma diversa dimensionalità e preprocessing, oltre a una diversa configurazione RF in UniZgFall features. I fold sono riproducibili solo parzialmente perché gli estimator non hanno tutti un seed fissato. I confronti UniZgFall-vs-UniMiB non sono omogenei: variano il numero delle classi, il numero/identità dei soggetti, le frequenze di classe e il contenuto delle registrazioni. Perciò le differenze di score non isolano l'effetto del dataset o dell'algoritmo.

## 11. Criticità metodologiche e rischio di leakage

| Evidenza | Conseguenza | Gravità | Stato / correzione |
|---|---|---|---|
| `StratifiedKFold` divide i record individuali; `UniMiB_raw` ha 30 soggetti × 66 registrazioni | Record dello stesso soggetto possono apparire sia nel training sia nel test; stima ottimistica per uso su persona non osservata | Alta | Rischio dimostrato dalla struttura e dal protocollo. Usare `StratifiedGroupKFold` o `GroupKFold` con il soggetto come gruppo; verificare se la stratificazione è fattibile. |
| UniZgFall non fornisce un ID soggetto esplicito nei nomi mostrati e non è documentato localmente | Non è possibile attestare indipendenza per soggetto | Alta per la pretesa di generalizzazione; leakage non dimostrato | Da verificare tramite metadata/fonti del dataset; non etichettare questa situazione come leakage certo. |
| Esempio = intero CSV, nessun overlap di finestre creato dal codice | Il codice non introduce sovrapposizione artificiale di finestre; la dipendenza intra-soggetto resta per UniMiB | Media | Confermato che non c'è segmentazione in finestre nel notebook. Group split resta necessario per test cross-soggetto. |
| Raw padding con zeri senza maschera; lunghezze variabili e padding UniMiB fino a 3.208 contro un massimo osservato di 1.863 | Durata e pattern di padding possono diventare indizi; il vettore ha molte colonne rispetto ai campioni | Media | Confermato. Usare lunghezze/mask, finestre definite in modo controllato o rappresentazioni invarianti alla durata; verificare l'effetto con ablazioni. |
| Feature calcolate prima del fold, ma per registrazione indipendentemente e senza stime globali o uso di y | Non c'è evidenza di leakage di preprocessing supervisionato; nessuno scaler è fit sull'intero dataset | Bassa | Non è un leakage dimostrato. Se si introducono selezione supervisionata o scaling, inserirli in una `Pipeline` dentro la CV. |
| Nessun test set indipendente e nessun nested CV | Gli score sono stime CV riutilizzate anche per confronto/statistica; non c'è stima finale non selezionata | Alta per la validità della conclusione finale | Confermato. Riservare un test set per soggetto o usare nested CV a gruppi. |
| RF e SVC non hanno `random_state`; solo CatBoost e splitter fissano 89 | Riesecuzioni possono produrre score e importanze diverse | Media | Confermato. Fissare i seed, registrare versioni e ambiente, e riportare la variabilità tra seed. |
| ALL seleziona per massimo di probabilità tra modelli diversi | Le probabilità non sono necessariamente comparabili/calibrate; il metodo può favorire il modello più overconfident | Media | Confermato. Calibrare solo dentro i fold e validare la regola, oppure usare stacking con training out-of-fold. |
| Test Kruskal–Wallis/Dunn su punteggi di fold riutilizzati e dipendenti; ALL dipende dai tre modelli | Assunzioni di indipendenza non rispettate; p-value non sostengono inferenza forte | Media | Confermato. Preferire analisi appaiate appropriate e un protocollo di confronto predefinito; evitare di trattare dieci fold come dieci studi indipendenti. |
| UniZgFall ha classi con 13, 34, 68 e 102 campioni | Le classi piccole producono stime per fold instabili; accuracy globale può sottopesarle | Media | Confermato. Riportare metriche per classe, intervalli adeguati e confusion matrix aggregata; valutare macro-recall e costi di errore. |
| Non sono calcolate metriche binarie caduta/ADL, specificità o sensibilità di caduta | Non si può quantificare il tasso di falsi allarmi o la rilevazione delle cadute | Alta rispetto a una pretesa di sistema di allarme | Confermato. Definire mapping binario documentato e riportare sensibilità, specificità, precision-recall, falsi allarmi/ora e latenza. |
| Non sono documentati sample rate, unità, orientamento, protocollo di acquisizione o ambiente reale per entrambi i dataset | Trasferibilità al dispositivo e comportamento online non sono dimostrati | Alta per l'uso clinico/operativo | Mancanza documentale locale confermata. Recuperare documentazione ufficiale e validare su acquisizioni realistiche. |
| Il helper `CI95` e i p-value sono salvati, ma il nome “CI95” non corrisponde a un CI convenzionale per la media | Rischio di sovrainterpretare la precisione degli score | Media | Confermato dal calcolo. Sostituire con bootstrap o intervallo appropriato e descrivere l'unità statistica. |

Non è stato dimostrato che due file contengano campioni duplicati, che siano segmenti sovrapposti della stessa registrazione o che label errate siano effettivamente presenti. Questi aspetti richiedono metadata o un controllo dedicato; non vanno presentati come problemi certi.

## 12. Riproducibilità

Per ripetere l'esecuzione in modo coerente con il codice attuale:

1. Usare Python 3.12 e installare le dipendenze richieste dagli import; il repository non le blocca in un manifest.
2. Avviare Jupyter con working directory `ml_accelerometers/`.
3. Verificare che `UniZgFall/`, `UniMiB_raw/` e i due `y_*.csv` siano presenti e ordinati come atteso.
4. Eseguire le celle dall'inizio in ordine. Gli output inclusi non provano che tutte le celle siano ripetibili in un ambiente pulito.
5. Evitare di lanciare i notebook nella directory di lavoro che contiene le figure esistenti se si vuole conservarle: `savefig` usa gli stessi nomi e le sovrascriverebbe.
6. Registrare dipendenze/versioni, hardware e tempi; fissare anche i seed degli estimator.

I percorsi sono relativi e la generazione di immagini sovrascrive file. Gli output `catboost_info/` possono essere sovrascritti/aggiornati durante l'addestramento. Il codice non crea una cartella separata per run, non registra un manifest degli split e non esporta predizioni per record. Il notebook raw UniMiB ha kernel metadata diverso dagli altri.

Non è garantita la riproducibilità bit-a-bit: `random_state` manca per SVC e RF, gli import/export dipendono da versioni non fissate e i grafici aggiungono jitter tramite `np.random.normal` senza seed. Le statistiche di score incorporate sono risultati salvati, non una prova che l'esecuzione corrente li riproduca.

## 13. Conclusioni scientifiche e miglioramenti prioritari

Il lavoro implementa una comparazione esplorativa di feature statistiche su registrazione completa e vettori raw appiattiti, con tre classificatori e una regola di ensemble. Il codice e i dati consentono di ricostruire la pipeline, ma gli score rappresentano classificazione tra record all'interno dei dataset, non un sistema validato su soggetti indipendenti o in tempo reale.

Priorità consigliate:

1. **Validazione per soggetto**: usare split a gruppi e riportare i soggetti di ogni fold. Per UniZgFall occorre prima recuperare l'ID soggetto.
2. **Definire l'obiettivo operativo**: classificazione multiclass di attività/tipi di caduta oppure rilevamento binario; documentare il mapping e le classi escluse.
3. **Valutare l'obiettivo fall detection**: sensibilità delle cadute, specificità sulle attività quotidiane, precision-recall, falsi allarmi per ora e latenza, oltre alle metriche macro.
4. **Correggere la comparazione statistica**: mantenere pairing dei fold e non assumere indipendenza; usare un protocollo di confronto adatto a CV ripetuta/a gruppi.
5. **Rendere la pipeline riproducibile**: dipendenze bloccate, seed per tutti i modelli, split salvati, cartella per run e output tabellari per esempio/fold.
6. **Rivedere la rappresentazione raw**: trattare esplicitamente lunghezza e maschere, controllare l'effetto dei padding e confrontare con finestre coerenti e/o modelli temporali.
7. **Documentare dati e feature**: legenda label UniZgFall, espansione delle sigle UniMiB, frequenza/unità, modalità e soggetti; valutare stabilità delle feature per soggetto e importanza permutation/SHAP o analisi per classe.
8. **Migliorare le figure**: ridimensionare o ruotare le etichette lunghe nelle confusion matrix e distinguere chiaramente matrici di fold medie da conteggi aggregati.

### Domande aperte da risolvere con documentazione esterna o metadata

- Qual è la legenda ufficiale dei suffissi UniZgFall e quale label è fall/ADL?
- Quali sono unità, sample rate, orientamento del sensore e protocollo dei due dataset?
- Qual è il mapping ufficiale delle sigle `GoingUpS`, `StandingUpFS`, `FallingBackSC` e `FallingWithPS`?
- Qual è l'esatto ordine di serializzazione delle label UniMiB rispetto ai file per soggetto/attività?
- Quali soggetti e registrazioni sono presenti in UniZgFall e quali record condividono sessioni?
- Le figure modificate in working tree corrispondono all'ultima esecuzione dei notebook modificati?
- In quali condizioni reali si vuole misurare un falso allarme (per ora, per attività o per registrazione)?

## Riferimenti interni

- [README del progetto](./README.md)
- [Notebook con feature UniZgFall](./training_features.ipynb)
- [Notebook con feature UniMiB](./training_features_UniMiB.ipynb)
- [Notebook raw UniZgFall](./training_raw.ipynb)
- [Notebook raw UniMiB](./training_raw_UniMiB.ipynb)
- [Label UniZgFall](./y_UniZgFall.csv)
- [Label UniMiB SHAR](./y_UniMiB_SHAR.csv)
