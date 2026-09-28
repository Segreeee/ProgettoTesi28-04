# Revisione sperimentale — cosa è stato verificato, cosa è cambiato, come funziona adesso

Documento di chiusura del lavoro di revisione svolto dopo il confronto con il relatore. Descrive le richieste ricevute, l'esito delle verifiche, le modifiche applicate e il funzionamento attuale del progetto.

Le conclusioni scientifiche sono in [`relazione_finale_progetto.md`](relazione_finale_progetto.md).

---

## 1. Le richieste del relatore

1. Aggiungere più dipendenze funzionali — cosa succede con 1, 5, 10.
2. Verificare che le FD usate abbiano senso nel contesto reale e non solo per quell'insieme di dati.
3. Portare il rumore massimo al 40% invece che al 100%, con granularità più fine.
4. Sporcare gradualmente il training, ma fare il test su dati puliti, non su quelli già sporcati.
5. Provare la cross-validation K-Fold.
6. Valutare F1 e accuratezza usando i dati di training per costruire il modello e dati di test non sporcati.

In aggiunta: **aumentare le dimensioni del dataset** rispetto al campione da 10.000 righe.

## 2. Esito delle verifiche sul progetto precedente

### Il test era fatto su dati sporchi

La funzione di valutazione riceveva solo il dataframe corrotto e lo divideva internamente in training e test: anche il test set conteneva righe sporche. Al 20% di rumore, 102 righe di test su 543 avevano feature corrotte.

Le etichette invece non erano mai corrotte: il target deriva da `CRSElapsedTime`, che non è fra le colonne sporcate. La contaminazione riguardava quindi solo le *feature*.

### Una proprietà che ha reso semplice la correzione

Gli indici delle righe di training e test risultano **identici** fra dataset pulito e dataset sporcato, a qualunque livello di rumore: il bilanciamento delle classi opera sul target, che non viene mai corrotto, e usa un seed fisso. È quindi possibile valutare il modello sulle *stesse* righe prese dal dataset pulito, senza alcun riallineamento.

### Le FD: il problema era reale

La scoperta automatica delle FD girava su `flight_sample_1000.csv`, mentre gli esperimenti usavano un campione diverso e più grande. Due FD valide sul campione piccolo si rompono su quello grande: `Tail_Number → WeatherDelay` (47 violazioni) e `Flight_Number → CarrierDelay` (225), entrambe spurie perché generate dall'81,4% di valori mancanti.

Inoltre i dati coprono **un solo mese** (dal 2025-01-01 al 2025-01-31): questo rende valide sui dati ma false nella realtà `DayofMonth → DayOfWeek` e `DayofMonth → FlightDate`.

### Non tutte le colonne arrivano al modello

La blacklist scarta ogni colonna il cui nome contiene `ID` (eccetto `OriginAirportID` e `DestAirportID`), più `Tail_Number`. Una FD costruita su colonne scartate muoverebbe IM/IP/IH senza toccare le metriche del modello: il vincolo ha guidato la scelta delle FD.

### Gli altri punti

Rumore fino al 100%, con livelli grossolani; nessuna cross-validation (un solo split 80/20); una sola FD corrotta.

## 3. Cosa è stato modificato

| # | Modifica | File | Perché |
|---|---|---|---|
| 1 | **Valutazione su dati puliti** | `progettoTesi_v2.py`, `blocco1_esperimento.py`, `blocco2_scaling_fd.py` | Richieste 4 e 6: separa "il modello ha imparato peggio" da "gli stiamo dando input illeggibili" |
| 2 | **Rumore fino al 40%**: 0, 5, 10, 20, 30, 40% | `blocco1_esperimento.py`, `blocco2_scaling_fd.py` | Richiesta 3 |
| 3 | **Cross-validation stratificata a 5 fold** | `progettoTesi_v2.py` | Richiesta 5 |
| 4 | **Esperimento con 1, 5 e 10 FD** | `blocco2_scaling_fd.py` | Richiesta 1 |
| 5 | **Verifica delle FD nel dominio reale**, salvata in `blocco2_verifica_fd.csv` | `blocco2_scaling_fd.py`, `blocco1_esperimento.py` | Richiesta 2; controllo automatico all'avvio |
| 6 | **Campione da 10.000 a 30.000 righe** | `creaCampione.py` | Dataset più grande, entro il limite di memoria del grafo dei conflitti (§3.2) |
| 7 | **Calcolo di IM/IP/IH più rapido** | `progettoTesi_v2.py` | Con più FD e più righe il confronto di tutte le coppie diventava troppo lento |
| 8 | **One-hot in formato sparso** | `progettoTesi_v2.py` | Memoria; predizioni identiche al formato denso |
| 9 | **Esecuzione parallela con checkpoint** | `esecuzione_parallela.py` (nuovo) | Tempi; i risultati non cambiano |

### 3.1 La valutazione su dati puliti

`ml_preparation` accetta un parametro `df_eval`: il dataset **pulito**. Per ogni fold il modello è addestrato sulle righe sporcate e valutato due volte — sulle righe di test prese dal dataset sporco (`*_test_sporco`, per confronto) e sulle stesse righe prese da quello pulito (`*_test_pulito`, la misura richiesta).

Due assert bloccanti proteggono la validità: gli indici delle due matrici devono coincidere e le etichette devono essere identiche. Per documentare che il training è davvero sporcato, ogni risultato riporta anche la **quota di righe di training che differiscono dal dataset pulito** (`Quota_train_sporca`): è zero a rumore 0% e cresce con il rumore. L'analisi del Blocco 1 si rifiuta di procedere se questo non accade, o se a rumore 0% le due valutazioni non coincidono.

### 3.2 Perché 30.000 righe e non di più

Le misure di inconsistenza sono calcolate costruendo il grafo dei conflitti con networkx, che va tenuto interamente in memoria. Misurato sul progetto:

- ogni arco del grafo occupa **~160 byte**;
- il numero di archi cresce circa **con il quadrato delle righe**: nel caso peggiore dell'esperimento (10 FD corrotte al 20% di rumore) si passa da 391 mila archi a 10.000 righe a 1,48 milioni a 20.000 e a **3,27 milioni a 30.000**.

A 30.000 righe il grafo più grande occupa **0,54 GB** di memoria reale del processo, compatibile con l'esecuzione di più processi in parallelo sulla RAM disponibile. A 150.000 righe il numero di archi sarebbe circa 25 volte maggiore (~80 milioni, oltre 13 GB per un solo grafo): non costruibile in memoria.

### 3.3 Calcolo di IM/IP/IH

Il calcolo originale confrontava tutte le coppie di righe dentro ogni gruppo. La nuova versione raggruppa le righe per valore del lato destro della FD: i conflitti sono allora tutte le coppie di righe con valori diversi, e si aggiungono al grafo in blocco senza enumerarle una per una. Il grafo prodotto è **lo stesso** (IM e IP identici, verificato); IH, calcolato da networkx come approssimazione del minimo, può variare di poche unità a seconda dell'ordine degli archi.

### 3.4 One-hot sparso ed esecuzione parallela

Il formato sparso dell'one-hot e il numero di thread del Random Forest cambiano solo memoria e velocità: verificato su un fold, le predizioni sono **identiche riga per riga** per tutti e 4 i modelli. L'esecuzione parallela sceglie il numero di processi in base alla RAM libera; ogni lavoro completato è salvato subito, e un'esecuzione interrotta riprende dai lavori mancanti.

## 4. Come funziona il progetto adesso

### La pipeline

1. **`creaCampione.py`** estrae 30.000 righe dal dataset completo (`flight_sample_30000.csv`).
2. **`discoverFDs.py`** scopre empiricamente le dipendenze funzionali.
3. Le FD candidate vengono **filtrate**: 0 violazioni sul campione usato, valide nel dominio reale (non artefatti del mese singolo o dei valori mancanti), con colonne che superano la blacklist.
4. **`progettoTesi_v2.py`** è il motore: iniezione del rumore, calcolo di IM/IP/IH, preparazione dei dati, addestramento dei 4 modelli RAW in cross-validation e doppia valutazione.
5. **`blocco1_esperimento.py`**: una FD (rotta → distanza) con le colonne ridondanti, 6 livelli di rumore, 5 repliche.
6. **`blocco2_scaling_fd.py`**: 1, 5 e 10 FD, 6 livelli di rumore, 5 repliche.
7. **`blocco1_analisi.py`**: controlli, aggregati, scomposizione del danno, correlazioni, significatività del degrado.
8. **`blocco2_analisi.py`**: le stesse analisi del Blocco 1 per ciascun numero di FD, più il confronto fra configurazioni a parità di livello e a parità di righe sporche e la misura del meccanismo che fa calare IM.
9. **`genera_plot_finali.py`**: grafici e tabelle, con figure speculari fra i due blocchi.

### Configurazione sperimentale

| Parametro | Valore |
|---|---|
| Obiettivo predittivo | `DurataBucket` — 5 fasce di durata schedulata del volo |
| Campione | 30.000 righe; 8.090 dopo il bilanciamento delle classi |
| Livelli di rumore | 0, 5, 10, 20, 30, 40% |
| Repliche per configurazione | 5, con seed diversi |
| Cross-validation | stratificata, 5 fold (~6.470 righe di training per fold) |
| Modelli | Logistic Regression, Random Forest, Decision Tree, Neural Network — RAW |
| Valutazione | sul test pulito; il test sporco è riportato per confronto |

### I file

| File | Ruolo |
|---|---|
| `creaCampione.py` | Estrazione del campione di lavoro |
| `discoverFDs.py` | Scoperta empirica delle dipendenze funzionali |
| `progettoTesi_v2.py` | Motore: rumore, indici di inconsistenza, valutazione ML |
| `esecuzione_parallela.py` | Esecuzione parallela con checkpoint |
| `blocco1_esperimento.py` | Blocco 1: una FD con le colonne ridondanti |
| `blocco2_scaling_fd.py` | Blocco 2: 1, 5 e 10 FD |
| `blocco1_analisi.py` | Analisi statistica del Blocco 1 |
| `blocco2_analisi.py` | Analisi statistica del Blocco 2 |
| `genera_plot_finali.py` | Grafici e tabelle finali |
| `blocco1_risultati_raw.csv`, `blocco2_risultati_raw.csv` | Risultati grezzi |
| `blocco2_verifica_fd.csv` | Certificazione delle 10 FD |
| `relazione_finale_progetto.md` | Relazione scientifica |
| `relazione_revisione_sperimentale.md` | Questo documento |

## 5. Cosa è cambiato nei risultati

### Il danno era sovrastimato di circa quattro volte

È la conseguenza più importante della correzione richiesta ai punti 4 e 6. Con il metodo precedente — test sulle stesse righe sporcate — al 40% di rumore il calo di F1 risultava di **10,7 punti**. Valutando gli stessi modelli sul test pulito, il calo è di **2,9 punti**: tutto il resto era il costo di interrogare il modello su input corrotti, non un apprendimento peggiore. La proporzione è stabile a ogni livello di rumore: fra il 24% e il 28% del danno apparente riguarda davvero il modello. Nel Blocco 2, dove per ogni riga sporca sono corrotte molte meno colonne, la quota sale al 40–53% e la sovrastima è di circa due volte.

### Il degrado resta reale, regolare e significativo

Con una FD violata, l'F1 sul test pulito scende da 0,9233 a 0,8948 al 40% di rumore, in modo monotono su tutti e 4 i modelli; il calo è significativo in 19 confronti su 20. Le correlazioni fra inconsistenza misurata e qualità delle predizioni vanno da −0,91 a −0,99.

### Più FD violate: più danno a parità di livello, meno a parità di righe sporche

Il calo dal baseline al 40% passa da **0,0199** con una FD a **0,0333** con cinque e **0,0510** con dieci. Ma ogni FD sceglie le proprie righe: allo stesso livello, più FD sporcano molte più righe di training (al 40%: 39,8%, 92,3% e 99,4%). Confrontando configurazioni con la stessa quota di righe sporche il rapporto si inverte: più FD danno una F1 più alta in 6 coppie su 6, e nella regressione *F1 ~ quota + quota² + N_FD* il coefficiente di N_FD è positivo e significativo per tutti e 4 i modelli nell'intervallo di quota comune alle tre configurazioni. Il danno dipende da quanta informazione predittiva viene distrutta, non dal numero di vincoli violati.

### Un risultato nuovo: IM smette di misurare il danno oltre una soglia

Con 10 FD corrotte, IM cresce fino al 20% di rumore (3.268.942 conflitti) e poi **cala** (2.917.019 al 40%), mentre l'F1 continua a peggiorare; dal 20% in poi la sua correlazione con la F1 diventa positiva (da +0,85 a +0,87), mentre quella di IH resta negativa (da −0,87 a −0,90). Con 1 e 5 FD, invece, IM cresce sempre. Il meccanismo è stato misurato: sporcando il lato sinistro le righe si ridistribuiscono uniformemente fra gli aeroporti, le coppie di righe con lo stesso aeroporto — il tetto dei conflitti possibili — scendono da 16,9 a 3,0 milioni, e quando quasi tutte sono in conflitto IM può solo seguire il tetto verso il basso. È un limite del campo di validità dell'indice, da dichiarare quando lo si usa come criterio di qualità dei dati.

### Il campione più grande alza il livello di partenza

Con 30.000 righe il baseline è 0,9233 di F1 (8.090 righe dopo il bilanciamento, circa 6.470 di training per fold). Prima della revisione, sul campione da 10.000 righe, era intorno a 0,87: con più dati i modelli predicono meglio la durata del volo, e il degrado va letto su questa base più alta.
