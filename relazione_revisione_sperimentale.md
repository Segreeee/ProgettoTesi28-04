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
| 4 | **Esperimento con più FD: 1, 2 e 4** (in una prima versione 1, 5 e 10) | `blocco2_scaling_fd.py` | Richiesta 1; il numero massimo è limitato dalle FD rilevanti per l'obiettivo (§3.5) |
| 5 | **Verifica delle FD nel dominio reale**, salvata in `blocco2_verifica_fd.csv` | `blocco2_scaling_fd.py`, `blocco1_esperimento.py` | Richiesta 2; controllo automatico all'avvio |
| 6 | **Campione da 10.000 a 30.000 righe** | `creaCampione.py` | Dataset più grande, entro il limite di memoria del grafo dei conflitti (§3.2) |
| 7 | **Calcolo di IM/IP/IH più rapido** | `progettoTesi_v2.py` | Con più FD e più righe il confronto di tutte le coppie diventava troppo lento |
| 8 | **One-hot in formato sparso** | `progettoTesi_v2.py` | Memoria; predizioni identiche al formato denso |
| 9 | **Esecuzione parallela con checkpoint** | `esecuzione_parallela.py` (nuovo) | Tempi; i risultati non cambiano |
| 10 | **Selezione delle FD per rilevanza rispetto all'obiettivo** | `analisi_rilevanza_fd.py` (nuovo) | Le 10 FD della prima versione del Blocco 2 erano valide ma quasi tutte irrilevanti per la durata del volo (§3.5) |
| 11 | **`DistanceGroup` sporcata insieme alla distanza** | `blocco1_esperimento.py`, `blocco2_scaling_fd.py` | Era una via di fuga aperta: il modello recuperava la distanza corrotta dalle fasce di distanza rimaste pulite |
| 12 | **Analisi statistica del Blocco 2**, speculare a quella del Blocco 1 | `blocco2_analisi.py` (nuovo) | Simmetria fra i due capitoli sperimentali |

### 3.1 La valutazione su dati puliti

`ml_preparation` accetta un parametro `df_eval`: il dataset **pulito**. Per ogni fold il modello è addestrato sulle righe sporcate e valutato due volte — sulle righe di test prese dal dataset sporco (`*_test_sporco`, per confronto) e sulle stesse righe prese da quello pulito (`*_test_pulito`, la misura richiesta).

Due assert bloccanti proteggono la validità: gli indici delle due matrici devono coincidere e le etichette devono essere identiche. Per documentare che il training è davvero sporcato, ogni risultato riporta anche la **quota di righe di training che differiscono dal dataset pulito** (`Quota_train_sporca`): è zero a rumore 0% e cresce con il rumore. Le analisi di entrambi i blocchi si rifiutano di procedere se questo non accade, o se a rumore 0% le due valutazioni non coincidono.

### 3.2 Perché 30.000 righe e non di più

Le misure di inconsistenza sono calcolate costruendo il grafo dei conflitti con networkx, che va tenuto interamente in memoria. Misurato sul progetto:

- ogni arco del grafo occupa **~160 byte**;
- il numero di archi cresce circa **con il quadrato delle righe**: nel caso peggiore della prima versione (10 FD corrotte al 20% di rumore) si passa da 391 mila archi a 10.000 righe a 1,48 milioni a 20.000 e a **3,27 milioni a 30.000**.

A 30.000 righe quel grafo occupava **0,54 GB** di memoria reale del processo. A 150.000 righe il numero di archi sarebbe circa 25 volte maggiore (~80 milioni, oltre 13 GB per un solo grafo): non costruibile in memoria.

Con le 4 FD rilevanti il caso peggiore sale a **5,9 milioni di archi** (4 FD al 40% di rumore), circa 0,9 GB stimati con lo stesso costo per arco: l'esecuzione riserva 1,5 GB a ogni processo parallelo.

### 3.3 Calcolo di IM/IP/IH

Il calcolo originale confrontava tutte le coppie di righe dentro ogni gruppo. La nuova versione raggruppa le righe per valore del lato destro della FD: i conflitti sono allora tutte le coppie di righe con valori diversi, e si aggiungono al grafo in blocco senza enumerarle una per una. Il grafo prodotto è **lo stesso** (IM e IP identici, verificato); IH, calcolato da networkx come approssimazione del minimo, può variare di poche unità a seconda dell'ordine degli archi.

### 3.4 One-hot sparso ed esecuzione parallela

Il formato sparso dell'one-hot e il numero di thread del Random Forest cambiano solo memoria e velocità: verificato su un fold, le predizioni sono **identiche riga per riga** per tutti e 4 i modelli. L'esecuzione parallela sceglie il numero di processi in base alla RAM libera; ogni lavoro completato è salvato subito, e un'esecuzione interrotta riprende dai lavori mancanti.

### 3.5 Le FD devono essere rilevanti per l'obiettivo

Una FD valida nel mondo reale non è per questo utile all'esperimento: se le sue colonne portano poca informazione sulla durata del volo, o se quell'informazione resta disponibile in colonne pulite, sporcarla non misura l'effetto dell'inconsistenza sulle predizioni. `analisi_rilevanza_fd.py` sporca al 40% una configurazione alla volta (training sporco, test pulito, 3 repliche) e misura il calo di F1:

| Configurazione | Calo F1 medio | Modelli con calo significativo |
|---|---|---|
| `Distance → DistanceGroup` | 7,9 punti | 4/4 |
| `Origin+Dest → Distance` | 2,0 punti | 4/4 |
| Aeroporto di origine, con città, stato, FIPS, nome dello stato, WAC | 1,3 punti | 4/4 |
| Aeroporto di destinazione, con le stesse colonne | 1,1 punti | 4/4 |
| Compagnia, orario di partenza, singole FD aeroporto → stato | 0,1–0,2 punti | 0 o 1 su 4 |

Il Blocco 2 usa quindi le **4 FD rilevanti**, nell'ordine della tabella, con configurazioni da 1, 2 e 4 FD. Ogni FD viene sporcata insieme alle colonne che ne ripetono l'informazione, come nel Blocco 1: le FD sugli aeroporti con città, stato, FIPS, nome dello stato e WAC, la FD sulla rotta con `DistanceGroup`. Dieci FD rilevanti e indipendenti non esistono per questo obiettivo predittivo: raggiungerle avrebbe richiesto di aggiungere FD irrilevanti.

La stessa analisi ha mostrato che `DistanceGroup`, non sporcata nella prima versione del Blocco 1, predice da sola la durata quasi quanto la distanza (F1 0,80 contro 0,83 con un albero su una sola colonna): era una via di fuga aperta. I risultati della prima versione sono conservati in `archivio_prima_revisione_fd/`.

## 4. Come funziona il progetto adesso

### La pipeline

1. **`creaCampione.py`** estrae 30.000 righe dal dataset completo (`flight_sample_30000.csv`).
2. **`discoverFDs.py`** scopre empiricamente le dipendenze funzionali.
3. Le FD candidate vengono **filtrate**: 0 violazioni sul campione usato, valide nel dominio reale (non artefatti del mese singolo o dei valori mancanti), con colonne che superano la blacklist.
4. **`analisi_rilevanza_fd.py`** misura quali FD contano per l'obiettivo predittivo e quali colonne ne ripetono l'informazione.
5. **`progettoTesi_v2.py`** è il motore: iniezione del rumore, calcolo di IM/IP/IH, preparazione dei dati, addestramento dei 4 modelli RAW in cross-validation e doppia valutazione.
6. **`blocco1_esperimento.py`**: una FD (rotta → distanza) con 17 colonne ridondanti, `DistanceGroup` compresa; 6 livelli di rumore, 5 repliche.
7. **`blocco2_scaling_fd.py`**: 1, 2 e 4 FD rilevanti, 6 livelli di rumore, 5 repliche.
8. **`blocco1_analisi.py`**: controlli, aggregati, scomposizione del danno, correlazioni, significatività del degrado.
9. **`blocco2_analisi.py`**: le stesse analisi del Blocco 1 per ciascun numero di FD, più il confronto fra configurazioni a parità di livello e a parità di righe sporche e la misura del meccanismo di saturazione di IM.
10. **`genera_plot_finali.py`**: grafici e tabelle, con figure speculari fra i due blocchi.

### Configurazione sperimentale

| Parametro | Valore |
|---|---|
| Obiettivo predittivo | `DurataBucket` — 5 fasce di durata schedulata del volo |
| Campione | 30.000 righe; 8.090 dopo il bilanciamento delle classi |
| FD del Blocco 1 | `Origin+Dest → Distance` + 17 colonne ridondanti |
| FD del Blocco 2 | `Distance → DistanceGroup`; `Origin+Dest → Distance` (+ `DistanceGroup`); `OriginAirportID → Origin` e `DestAirportID → Dest`, ciascuna con 5 colonne ridondanti |
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
| `analisi_rilevanza_fd.py` | Rilevanza delle FD per l'obiettivo predittivo |
| `progettoTesi_v2.py` | Motore: rumore, indici di inconsistenza, valutazione ML |
| `esecuzione_parallela.py` | Esecuzione parallela con checkpoint |
| `blocco1_esperimento.py` | Blocco 1: una FD con le colonne ridondanti |
| `blocco2_scaling_fd.py` | Blocco 2: 1, 2 e 4 FD rilevanti |
| `blocco1_analisi.py` | Analisi statistica del Blocco 1 |
| `blocco2_analisi.py` | Analisi statistica del Blocco 2 |
| `genera_plot_finali.py` | Grafici e tabelle finali |
| `blocco1_risultati_raw.csv`, `blocco2_risultati_raw.csv` | Risultati grezzi |
| `rilevanza_colonne.csv`, `rilevanza_fd.csv` | Risultati dell'analisi di rilevanza |
| `blocco2_verifica_fd.csv` | Certificazione delle 4 FD del Blocco 2 |
| `archivio_prima_revisione_fd/` | Risultati, grafici e relazioni della versione con `DistanceGroup` pulita e 10 FD |
| `relazione_finale_progetto.md` | Relazione scientifica |
| `relazione_revisione_sperimentale.md` | Questo documento |

## 5. Cosa è cambiato nei risultati

### Il danno misurato sul test sporco è sovrastimato

È la conseguenza della correzione richiesta ai punti 4 e 6. Nel Blocco 1, al 40% di rumore, il calo di F1 misurato sulle stesse righe sporcate è di **31,2 punti**; valutando gli stessi modelli sul test pulito è di **8,4 punti**. Il resto è il costo di interrogare il modello su input corrotti, non un apprendimento peggiore. La proporzione è stabile: a ogni livello il danno dovuto all'apprendimento è fra il 27% e il 32% di quello apparente. Nel Blocco 2, dove ogni riga sporca ha meno colonne corrotte, la quota sale al 36–56% e la sovrastima va da circa due volte (1 e 2 FD) a due volte e mezzo (4 FD).

### Chiudere la via di fuga `DistanceGroup` triplica il danno

Nella prima versione del Blocco 1 la F1 sul test pulito scendeva da 0,9233 a 0,8948 al 40% di rumore (−2,9 punti). Sporcando anche `DistanceGroup` scende a **0,8390 (−8,4 punti)**, in modo monotono e significativo in 20 confronti su 20. IM, IP e IH sono **identici** nelle due versioni, perché `DistanceGroup` non fa parte della FD dichiarata: gli indici non vedono le copie pulite dell'informazione corrotta.

Il calo varia molto fra modelli: 2,9 punti Random Forest, 3,0 Decision Tree, 4,9 Neural Network, 22,9 Logistic Regression, che usa la distanza come valore numerico.

Nel Blocco 2 la stessa via di fuga era rimasta aperta in parte: `DistanceGroup` era sporcata solo dalla FD `Distance → DistanceGroup`, mentre la FD sulla rotta corrompeva la distanza su altre righe lasciando lì la fascia pulita (7.179 righe su 30.000 al 40%). Aggiungendo `DistanceGroup` alle colonne ridondanti della FD sulla rotta, il calo al 40% passa da 10,0 a **11,0 punti** con 2 FD e da 15,3 a **17,0** con 4.

### Con FD rilevanti, più FD fanno più danno a parità di livello

Il calo dal baseline al 40% passa da **7,8 punti** con 1 FD a **11,0** con 2 e **17,0** con 4. Nella prima versione, con 10 FD quasi tutte irrilevanti, il calo massimo era di 5,1 punti con il 99% delle righe sporche. A parità di livello di rumore il passaggio 1 → 2 FD è significativo in 20 confronti su 20, il passaggio 2 → 4 in 19 su 20.

### A parità di righe sporche, il risultato dipende dal modello

Nella prima versione, a parità di righe sporche, più FD facevano meno danno per tutti e 4 i modelli. Con le FD rilevanti non è più una regola generale. Nella regressione *F1 ~ quota + quota² + N_FD*, nell'intervallo di quota comune alle tre configurazioni, il coefficiente di N_FD è:
- positivo per Logistic Regression e Decision Tree (p < 0,001), che soffrono soprattutto la distanza corrotta;
- negativo per Neural Network (p < 0,001), che soffre di più gli aeroporti corrotti;
- non significativo per Random Forest (p = 0,34).

Conta quale informazione è inconsistente e quanto il modello ne dipende, non il numero di vincoli violati.

### IM non è confrontabile fra insiemi di FD diversi

Dentro ogni configurazione IM e IH correlano col danno (Spearman fra −0,91 e −0,99). Fra configurazioni diverse, invece, al 40% IM è 32 volte più grande con 4 FD che con 1 a fronte di un danno 2,2 volte maggiore, perché è dominato dalle FD sugli aeroporti, che hanno gruppi grandi e contano poco per la previsione. IH cresce di 1,7 volte, molto più vicino al rapporto fra i danni.

Con 4 FD, inoltre, IM si satura: fra il 30% e il 40% cresce solo dell'1,1% mentre la F1 perde altri 4,0 punti. Il meccanismo, misurato, è lo stesso che nella prima versione con 10 FD faceva calare IM: sporcando l'ID dell'aeroporto i gruppi grandi si svuotano e il numero di coppie di righe che possono entrare in conflitto crolla.

### Il campione più grande alza il livello di partenza

Con 30.000 righe il baseline è 0,9233 di F1 (8.090 righe dopo il bilanciamento, circa 6.470 di training per fold). Prima della revisione, sul campione da 10.000 righe, era intorno a 0,87: con più dati i modelli predicono meglio la durata del volo, e il degrado va letto su questa base più alta.
