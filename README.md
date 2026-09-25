# Misure di inconsistenza per Data-Centric AI

Codice e risultati della tesi di laurea triennale in Ingegneria Informatica (Università della Calabria, DIMES), relatore Prof. Francesco Parisi.

La tesi studia **quanto l'inconsistenza dei dati di training peggiori le predizioni di un modello di machine learning** usato poi su dati corretti. L'inconsistenza è definita rispetto a dipendenze funzionali (FD) e misurata con tre indici del framework di Parisi e Grant: **IM**, **IP** e **IH**.

---

## Indice

- [La domanda e l'esperimento](#la-domanda-e-lesperimento)
- [Risultati principali](#risultati-principali)
- [Installazione](#installazione)
- [Riprodurre i risultati](#riprodurre-i-risultati)
- [Struttura del repository](#struttura-del-repository)
- [Come funziona il codice](#come-funziona-il-codice)
- [Verifiche automatiche](#verifiche-automatiche)
- [Documenti](#documenti)
- [Limiti noti](#limiti-noti)

---

## La domanda e l'esperimento

> **Quanto incide sulla qualità delle predizioni di un modello AI la qualità (consistenza) dei dati in input?**

Lo scenario è quello di un modello **addestrato su dati storici contaminati ma usato su dati corretti**. Per riprodurlo:

1. si prende un campione di **30.000 voli** nazionali statunitensi di gennaio 2025;
2. l'obiettivo è classificare la **durata schedulata del volo** in 5 fasce (`DurataBucket`: `<90m`, `90-150m`, `150-210m`, `210-300m`, `>300m`);
3. si corrompono le colonne di alcune **dipendenze funzionali** su una quota crescente di righe del training: **0, 5, 10, 20, 30, 40%**;
4. si addestrano **4 modelli con iperparametri di default** (Logistic Regression, Random Forest, Decision Tree, rete neurale MLP) in **cross-validation stratificata a 5 fold**;
5. si valutano **sulle stesse righe di test prese dal campione pulito**, e per confronto su quelle sporcate;
6. si misura l'inconsistenza del training con **IM** (coppie di righe in conflitto), **IP** (righe coinvolte in almeno un conflitto) e **IH** (numero minimo di righe da correggere).

L'unica variabile controllata è la **percentuale di righe alterate**. IM, IP e IH sono **osservati** dopo l'iniezione del rumore, come la F1 dei modelli.

Gli esperimenti sono due:

| | Cosa si corrompe | Script |
|---|---|---|
| **Blocco 1** | una FD, `Origin + Dest → Distance`, insieme alle 17 colonne che ripetono la stessa informazione (aeroporti, città, stati, fasce di distanza), così che il modello non possa recuperarla | `blocco1_esperimento.py` |
| **Blocco 2** | 1, 2 e 4 FD scelte per rilevanza rispetto all'obiettivo: `Distance → DistanceGroup`, `Origin + Dest → Distance`, `OriginAirportID → Origin`, `DestAirportID → Dest` | `blocco2_scaling_fd.py` |

Ogni configurazione è ripetuta **5 volte con seed diversi**. Il seed della replica governa il rumore iniettato, il bilanciamento delle classi e la divisione in fold.

---

## Risultati principali

I numeri completi sono in [`numeri_tesi.md`](numeri_tesi.md), generato dai CSV delle analisi.

| | Risultato |
|---|---|
| Baseline (rumore 0%) | F1 media **0,9214** (caso puro: 0,20) |
| Blocco 1, 40% di righe alterate | calo di F1 di **8,3 punti**; Logistic Regression **23,4**, gli altri tre modelli in media **3,3** |
| Blocco 2, 40% di rumore | calo di **7,8 / 11,2 / 17,1 punti** con 1 / 2 / 4 FD |
| Significatività (t-test di Welch contro il baseline) | **18 confronti su 20** nel Blocco 1, **58 su 60** nel Blocco 2; non significativi solo Decision Tree e Random Forest al 5% di rumore con una FD |
| Valutazione sul test sporco | sovrastima il danno da circa **2** a **3,7 volte** |
| IH 2-approssimato contro esatto | sulle istanze reali l'approssimato supera il minimo del **45–70%** |

In sintesi:
- **l'inconsistenza nel training peggiora le predizioni su dati corretti**, in modo regolare;
- **l'entità dipende da quale informazione è inconsistente** — due colonne sulla distanza fanno quasi il danno di venti colonne geografiche — e **da quanto il modello ne dipende**;
- **le tre misure crescono con la corruzione ma non predicono il danno**: con 4 FD, IM smette di crescere oltre il 30% e IP satura già al 10%, mentre il danno continua ad aumentare. Descrivono le violazioni dei vincoli, non il degrado del modello.

Discussione completa in [`relazione_finale_progetto.md`](relazione_finale_progetto.md).

---

## Installazione

Serve **Python 3.13** (i risultati sono stati prodotti con 3.13.2).

```bash
git clone https://github.com/Segreeee/ProgettoTesi28-04.git
cd ProgettoTesi28-04
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### Dati

- **`flight_sample_30000.csv`** è il campione usato da tutti gli esperimenti ed è **incluso nel repository**: per riprodurre i risultati non serve altro.
- **Il dataset originale** (232 MB) **non è incluso**, perché supera il limite di GitHub. Serve solo per rigenerare il campione con `creaCampione.py`. Si scarica da [BTS TranStats](https://www.transtats.bts.gov/), tabella *Reporting Carrier On-Time Performance (1987-present)*, mese di gennaio 2025, e va salvato nella cartella del progetto come `On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_1.csv`.

---

## Riprodurre i risultati

Gli script vanno lanciati dalla cartella del progetto, nell'ordine indicato. Tutti leggono e scrivono file nella cartella corrente.

| # | Comando | Cosa fa | Durata indicativa |
|---|---|---|---|
| 0 | `python creaCampione.py` | *(facoltativo)* rigenera `flight_sample_30000.csv` dal dataset originale, con seed fisso | 1 min |
| 1 | `python discoverFDs.py` | *(facoltativo)* scoperta automatica delle FD su `flight_sample_1000.csv` → `lista_FD_trovate.csv` | pochi min |
| 2 | `python blocco1_esperimento.py` | Blocco 1: 6 livelli × 5 repliche → `blocco1_risultati_raw.csv` | 30–60 min |
| 3 | `python blocco1_analisi.py` | controlli, aggregati, scomposizione del danno, test di Welch | < 1 min |
| 4 | `python analisi_rilevanza_fd.py` | rilevanza di ogni FD per l'obiettivo (40% di rumore, 3 repliche); richiede i risultati del Blocco 1 | ~1 h |
| 5 | `python blocco2_scaling_fd.py` | Blocco 2: 3 configurazioni × 6 livelli × 5 repliche → `blocco2_risultati_raw.csv` | 1–3 h |
| 6 | `python blocco2_analisi.py` | analisi del Blocco 2, inclusi confronti fra configurazioni e comportamento delle misure | ~5 min |
| 7 | `python verifica_ih.py` | IH 2-approssimato contro ottimo esatto (ILP) su istanze ridotte | pochi min |
| 8 | `python ih_esatto_blocco2.py` | IH esatto sulle istanze reali del Blocco 2, fin dove il risolutore chiude | ~15 min |
| 9 | `python genera_plot_finali.py` | grafici e tabelle riassuntive | < 1 min |
| 10 | `python numeri_tesi.py` | genera `numeri_tesi.md` con tutti i numeri da citare | < 1 min |

**Esecuzione parallela e ripresa.** Gli esperimenti (passi 2, 4 e 5) scelgono da soli quanti processi usare in base alla RAM libera. Il numero si può forzare con `--workers N`. Ogni lavoro completato viene salvato subito nel CSV dei risultati: se l'esecuzione si interrompe, rilanciando lo stesso comando riparte dai lavori mancanti. Per ripartire da zero basta cancellare il CSV dei risultati grezzi.

**Riproducibilità.** Tutti i seed sono fissati: `random_state=42` per i modelli e seed `100 + replica` per rumore, bilanciamento e fold. A parità di versioni delle librerie, i risultati sono identici. Il numero di processi paralleli cambia solo i tempi, non i risultati.

---

## Struttura del repository

### Codice

| File | Ruolo |
|---|---|
| `progettoTesi_v2.py` | **Motore**: iniezione del rumore, grafo dei conflitti e calcolo di IM/IP/IH, IH esatto per una FD, preparazione dei dati, addestramento e doppia valutazione |
| `esecuzione_parallela.py` | Esecuzione parallela con checkpoint e scelta dei processi in base alla RAM |
| `creaCampione.py` | Estrazione del campione da 30.000 righe |
| `discoverFDs.py` | Scoperta empirica delle dipendenze funzionali |
| `analisi_rilevanza_fd.py` | Rilevanza delle FD candidate per l'obiettivo predittivo |
| `blocco1_esperimento.py`, `blocco1_analisi.py` | Blocco 1: esperimento e analisi |
| `blocco2_scaling_fd.py`, `blocco2_analisi.py` | Blocco 2: esperimento e analisi |
| `verifica_ih.py`, `ih_esatto_blocco2.py` | IH approssimato contro esatto |
| `genera_plot_finali.py` | Grafici e tabelle |
| `numeri_tesi.py` | Raccolta dei numeri verificati per la tesi |
| `controllo_tesi.py` | Confronta i numeri citati nel PDF della tesi con quelli delle analisi |

### Risultati

| File | Contenuto |
|---|---|
| `blocco1_risultati_raw.csv`, `blocco2_risultati_raw.csv` | Una riga per modello × livello × replica (× configurazione): metriche sul test pulito e sporco, quota di righe sporche, IM/IP/IH per fold |
| `blocco*_aggregato.csv`, `blocco*_scomposizione.csv`, `blocco*_test_degrado.csv` | Medie, scomposizione del danno (apprendimento contro input corrotto), test di Welch |
| `blocco2_test_configurazioni.csv`, `blocco2_quota_normalizzata.csv`, `blocco2_quota_punti_confrontabili.csv` | Confronti fra 1, 2 e 4 FD: a parità di livello e a parità di righe sporche |
| `blocco2_meccanismo_indici.csv` | Come si comportano IM, IP e IH al crescere del rumore, per FD |
| `blocco2_verifica_fd.csv` | Certificazione delle 4 FD: zero violazioni, colonne che arrivano al modello |
| `rilevanza_fd.csv`, `rilevanza_colonne.csv` | Rilevanza delle FD e delle singole colonne per l'obiettivo |
| `verifica_ih.csv`, `blocco2_ih_esatto.csv` | IH approssimato contro esatto |
| `*.log` | Log delle esecuzioni e delle analisi |
| `plot_*.png`, `tabella_riassuntiva_*.png/.csv` | Figure e tabelle |

### Altre cartelle

| Cartella | Contenuto |
|---|---|
| `blocco2_csv_completo/` | Progetto separato e autonomo: una versione del Blocco 2 su un campione più grande (45% del dataset originale). Ha un proprio [README](blocco2_csv_completo/README.md) |
| `archivio_prima_revisione_fd/`, `archivio_prima_revisione_prof/` | Risultati delle versioni precedenti, conservati per confronto |

---

## Come funziona il codice

**Iniezione del rumore** (`inject_multiple_fd_noise`). Per ogni FD si scelgono a caso le righe da corrompere, pari al livello di rumore. Su quelle righe si sostituiscono **sia il lato sinistro sia il lato destro** della FD, più le colonne che ne ripetono l'informazione, con un valore diverso preso dalla stessa colonna. Il target non viene mai toccato.

**Grafo dei conflitti** (`get_global_inconsistency_metrics`). Un nodo per riga, un arco per ogni coppia di righe con lo stesso lato sinistro e un lato destro diverso. Da qui:
- **IM** = numero di archi;
- **IP** = numero di nodi con almeno un arco;
- **IH** = vertex cover del grafo, calcolato con l'algoritmo **2-approssimato** di NetworkX: è un limite superiore del minimo, al più doppio. Con una sola FD il grafo è un'unione di grafi multipartiti completi, e il minimo esatto si ottiene in forma chiusa (`ih_esatto_una_fd`). Con più FD il minimo esatto si calcola con la programmazione lineare intera (`verifica_ih.py`, `ih_esatto_blocco2.py`).

**Valutazione** (`ml_preparation`, `fold_di_valutazione`, `indici_per_fold`):
- il dataset sporcato e quello pulito passano per la stessa preparazione (blacklist anti-leakage, bilanciamento delle classi per sottocampionamento);
- poiché il target non viene mai corrotto, le righe selezionate sono **identiche** nei due casi;
- per ogni fold il modello si addestra sulle righe sporcate e si valuta due volte, sul test pulito e su quello sporco;
- IM, IP e IH sono calcolati sulle **stesse righe di training** di ciascun fold, così indici e metriche descrivono la stessa popolazione.

---

## Verifiche automatiche

Gli script si **interrompono** invece di produrre risultati non interpretabili se:
- le righe o le etichette del test pulito e di quello sporco non coincidono;
- a rumore 0% il training non è pulito, o le due valutazioni differiscono;
- la quota di righe sporche non cresce con il rumore; nel Blocco 2 deve anche seguire 1 − (1 − p)^N entro 0,01;
- il baseline non varia fra le repliche, condizione necessaria per il test di Welch;
- una FD usata ha violazioni sul campione pulito, valori mancanti o colonne escluse dal modello;
- con una FD, i conflitti ricalcolati dall'analisi non coincidono con IM dell'esperimento.

I log delle analisi riportano inoltre il numero di righe su cui sono calcolati gli indici, che deve coincidere con le righe di training di un fold (6.472).

---

## Documenti

| File | Contenuto |
|---|---|
| [`relazione_finale_progetto.md`](relazione_finale_progetto.md) | Relazione scientifica: disegno, risultati, risposta alla domanda di ricerca, limiti |
| [`relazione_revisione_sperimentale.md`](relazione_revisione_sperimentale.md) | Storia delle revisioni metodologiche e del loro effetto sui risultati |
| [`numeri_tesi.md`](numeri_tesi.md) | Tutti i numeri da citare nella tesi, generati dai CSV |
| [`correzioni_tesi.md`](correzioni_tesi.md) | Correzioni da riportare nel testo della tesi dopo le osservazioni del relatore |
| [`bibliografia.bib`](bibliografia.bib) | Bibliografia in formato BibTeX |

---

## Limiti noti

- **Un solo mese di dati** (gennaio 2025) e **un solo obiettivo predittivo**.
- **Rumore casuale uniforme**: gli errori reali tendono a essere sistematici.
- **Modelli con iperparametri di default**, per isolare l'effetto dei dati da quello del tuning.
- **IH con più FD è un limite superiore**: il valore esatto si calcola solo fino al 20% di rumore con 2 FD e fino al 10% con 4 FD.
- **Cinque repliche per livello**: i test a rumore basso hanno poca potenza.

---

## Riferimenti

- F. Parisi, J. Grant, *On measuring inconsistency in definite and indefinite databases with denial constraints*, Artificial Intelligence, 2023.
- Dati: Bureau of Transportation Statistics, *Reporting Carrier On-Time Performance (1987-present)*, gennaio 2025.

Bibliografia completa in [`bibliografia.bib`](bibliografia.bib).
