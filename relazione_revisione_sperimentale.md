# Revisione sperimentale — cosa è stato verificato, cosa è cambiato, come funziona adesso

Documento di chiusura del lavoro di revisione svolto dopo il confronto con il relatore. Descrive le richieste ricevute, l'esito delle verifiche, le modifiche applicate e il funzionamento attuale del progetto.

Le conclusioni scientifiche sono in [`relazione_finale_progetto.md`](relazione_finale_progetto.md); l'approfondimento sul braccio di controllo è in [`relazione_braccio_C.md`](relazione_braccio_C.md).

---

## 1. Le richieste del relatore

1. Aggiungere più dipendenze funzionali — cosa succede per 1, 5, 10.
2. Verificare che le FD usate abbiano senso nel contesto reale e non solo per quell'insieme di dati.
3. Portare il rumore massimo al 40% invece che al 100%, con granularità più fine.
4. Sporcare gradualmente il training, ma fare il test su dati puliti.
5. Provare la cross-validation K-Fold.
6. Valutare F1 e accuratezza usando i dati di training per costruire il modello e dati di test non sporcati.

Più una domanda specifica: **il braccio C, come il braccio A, fa training sul dataset sporcato e test su quello pulito?**

## 2. Esito delle verifiche

### La domanda sul braccio C: la risposta era NO

Entrambi i bracci addestravano **e** testavano su dati sporcati. La funzione di valutazione riceveva solo il dataframe corrotto e lo divideva internamente in training e test, quindi il test set era contaminato in entrambi i casi: al 20% di rumore, 102 righe di test su 543 nel braccio A e 113 su 543 nel braccio C.

Le etichette invece non erano mai corrotte: il target deriva da `CRSElapsedTime`, che non è fra le colonne sporcate. La contaminazione riguardava quindi solo le *feature*.

### Una proprietà che ha reso semplice la correzione

Gli indici delle righe di training e test sono risultati **identici** fra dataset pulito e dataset sporcati, a tutti i livelli di rumore: il bilanciamento delle classi opera sul target non corrotto e lo split usa un seed fisso. È stato quindi possibile valutare sulle *stesse* righe prese dal dataset pulito senza alcun riallineamento.

### Un difetto di disegno non previsto: il braccio C non era un controllo equo

Confronto colonna per colonna al 40% di rumore: il braccio A modificava **19 colonne**, il braccio C **una sola** (`Distance`), lasciando `Origin` e `Dest` intatti — proprio le più informative per predire la durata di un volo. Il confronto fra i due bracci non isolava quindi la coerenza, ma la confondeva con "19 colonne distrutte contro 1 perturbata".

Dettagli e conseguenze in [`relazione_braccio_C.md`](relazione_braccio_C.md).

### Le FD: il problema era reale

La scoperta automatica girava su `flight_sample_1000.csv` mentre gli esperimenti usano `flight_sample_10000.csv`. Due FD valide sul campione piccolo si rompono su quello grande: `Tail_Number → WeatherDelay` (47 violazioni) e `Flight_Number → CarrierDelay` (225), entrambe spurie perché generate dall'81,4% di valori mancanti.

Inoltre il campione copre **un solo mese** (dal 2025-01-01 al 2025-01-31): questo rende valide sui dati ma false nella realtà `DayofMonth → DayOfWeek` e `DayofMonth → FlightDate`.

### Un vincolo scoperto: non tutte le colonne arrivano al modello

La blacklist scarta ogni colonna il cui nome contiene la sottostringa `ID` (eccetto `OriginAirportID` e `DestAirportID`), più `Tail_Number`. Quindi `DOT_ID_Reporting_Airline`, `OriginCityMarketID`, `DestCityMarketID`, `OriginAirportSeqID`, `DestAirportSeqID` non raggiungono mai il modello: una FD costruita su di esse muoverebbe IM/IP/IH senza toccare le metriche ML. Il vincolo ha determinato la selezione delle FD per l'esperimento di scaling.

### Stato precedente sugli altri punti

Rumore fino al 100% con 6 livelli grossolani; nessuna cross-validation (un singolo split 80/20 con seed fisso); una sola FD corrotta.

## 3. Cosa è stato modificato

| # | Modifica | File | Perché |
|---|---|---|---|
| 1 | **Valutazione su dati puliti** per entrambi i bracci | `progettoTesi_v2.py`, `blocco1_esperimento.py` | Separa "il modello ha imparato peggio" da "gli stiamo dando input illeggibili" |
| 2 | **Braccio C reso controllo equo** | `blocco1_esperimento.py` | Elimina il confound: stesse 19 colonne del braccio A, ma coerenti |
| 3 | **Rumore fino al 40%** con granularità fine | `blocco1_esperimento.py` | Richiesta 3 |
| 4 | **Cross-validation stratificata a 5 fold** | `progettoTesi_v2.py` | Richiesta 5; particolarmente utile su un dataset di 2.715 righe dopo il bilanciamento |
| 5 | **Esperimento di scaling 1 / 5 / 10 FD** | `blocco2_scaling_fd.py` (nuovo) | Richiesta 1 |
| 6 | **Verifica delle FD nel dominio reale** | `blocco2_scaling_fd.py` | Richiesta 2; controllo automatico all'avvio |
| 7 | **Ottimizzazione del calcolo di IM/IP/IH** | `progettoTesi_v2.py` | Necessaria: con 10 FD servivano 58s per chiamata, ore complessive |

### Dettaglio della modifica 1

`ml_preparation` accetta ora un parametro `df_eval`: il dataset **pulito**. Per ogni fold il modello è addestrato sulle righe sporcate e valutato due volte — sulle righe di test prese dal dataset sporco (`*_test_sporco`) e sulle stesse righe prese da quello pulito (`*_test_pulito`).

La chiamata è collocata **fuori** dal ramo che sceglie quale corruzione applicare, così è strutturalmente impossibile che i due bracci vengano valutati diversamente. Due assert bloccanti proteggono la validità: gli indici delle due matrici devono coincidere e le etichette devono essere identiche.

### Dettaglio della modifica 2

`inject_fd_preserving_noise` riassegna ora a interi gruppi di righe il **profilo completo di un'altra rotta realmente presente** nel dataset: nuovo `Origin`, nuovo `Dest`, la distanza vera di quella rotta e tutte le colonne geografiche corrispondenti. Ogni gruppo resta internamente uniforme e riceve una combinazione realmente esistente, quindi tutte le FD restano soddisfatte e IM vale 0 per costruzione.

### Dettaglio della modifica 7

Il calcolo dell'inconsistenza confrontava tutte le coppie di righe dentro ogni gruppo. Con 10 FD e gruppi grandi il costo esplodeva. La nuova versione raggruppa le righe per valore del RHS: i conflitti sono allora il grafo multipartito completo fra i gruppi, senza enumerare le coppie una per una.

Risultato: da **58s a 4,9s** per chiamata, con IM e IP **identici** e IH entro lo 0,06% (differenza dovuta all'approssimazione del vertex cover, non a un errore).

## 4. Come funziona il progetto adesso

### La pipeline

1. **`creaCampione.py`** estrae 10.000 righe dal dataset completo.
2. **`discoverFDs.py`** scopre empiricamente le dipendenze funzionali presenti nel campione.
3. Le FD candidate vengono **filtrate**: 0 violazioni sulle 10.000 righe, valide nel dominio reale (non artefatti del mese singolo o dei valori mancanti), con tutte le colonne che superano la blacklist.
4. **`progettoTesi_v2.py`** fornisce il motore condiviso: iniezione del rumore, calcolo di IM/IP/IH, preparazione dei dati e valutazione dei 4 modelli RAW in cross-validation.
5. **`blocco1_esperimento.py`** esegue il confronto a due bracci: per ogni livello di rumore e replica genera il dataset del braccio A (incoerente) e del braccio C (coerente), misura l'inconsistenza e valuta i modelli **su test sporco e su test pulito**.
6. **`blocco2_scaling_fd.py`** esegue lo scaling 1 → 5 → 10 FD sul solo braccio A.
7. **`blocco1_analisi.py`** produce aggregati, correlazioni e t-test fra i bracci.
8. **`genera_plot_finali.py`** produce grafici e tabelle per la relazione.

### Configurazione sperimentale corrente

| Parametro | Valore |
|---|---|
| Obiettivo predittivo | `DurataBucket` — 5 classi di durata del volo |
| Baseline (rumore 0%) | 0.8723 di F1, contro un caso puro di 0.20 |
| Livelli di rumore | 0, 5%, 10%, 20%, 30%, 40% |
| Repliche per configurazione | 5, con seed diversi |
| Cross-validation | stratificata, 5 fold |
| Modelli | Logistic Regression, Random Forest, Decision Tree, Neural Network — RAW |
| Valutazione | doppia: test sporco e test pulito |

### I file

| File | Ruolo |
|---|---|
| `creaCampione.py` | Estrazione del campione di lavoro |
| `discoverFDs.py` | Scoperta empirica delle dipendenze funzionali |
| `progettoTesi_v2.py` | Motore condiviso: rumore, indici di inconsistenza, valutazione ML |
| `blocco1_esperimento.py` | Esperimento a due bracci (A incoerente, C coerente) |
| `blocco2_scaling_fd.py` | Esperimento di scaling 1 / 5 / 10 FD |
| `blocco1_analisi.py` | Aggregati, correlazioni, t-test fra bracci |
| `genera_plot_finali.py` | Grafici e tabelle finali |
| `blocco1_risultati_raw.csv`, `blocco2_risultati_raw.csv` | Risultati grezzi |
| `blocco1_aggregato.csv`, `blocco1_correlazioni.csv`, `blocco1_test_bracci.csv` | Analisi del Blocco 1 |
| `tabella_riassuntiva_blocco1/2.csv` e `.png` | Tabelle riassuntive |
| `plot_blocco1_*.png`, `plot_blocco2_*.png` | Grafici |
| `relazione_finale_progetto.md` | Relazione scientifica principale |
| `relazione_braccio_C.md` | Approfondimento sul braccio di controllo |
| `relazione_revisione_sperimentale.md` | Questo documento |
| `variante_solo_braccioA/` | Ramo alternativo che considera il solo braccio A |

## 5. Cosa è cambiato nei risultati

### La conclusione precedente era un artefatto

Prima si concludeva che *violare una dipendenza funzionale danneggia più che avere valori sbagliati ma coerenti*. Valutando su dati puliti — e con il braccio C reso controllo equo — il rapporto si **inverte**:

| Rumore | A test sporco | C test sporco | A test pulito | C test pulito |
|---|---|---|---|---|
| 20% | 0.7992 | 0.8295 | 0.8518 | 0.8400 |
| 40% | 0.7540 | 0.8088 | 0.8296 | 0.8131 |

Sul test sporco il braccio A appare peggiore (differenza significativa in 18 confronti su 20); sul test pulito è il braccio C a risultare peggiore (13 confronti su 20).

### Il danno era sovrastimato di quasi tre volte

Al 40% di rumore, del calo totale di 0.1183 osservato con il metodo precedente solo il **36%** (0.0427) è attribuibile a un modello che ha imparato peggio. Il restante **64%** (0.0756) era l'effetto di interrogare il modello su input corrotti.

### Cosa invece regge

La risposta alla domanda di ricerca è confermata: l'inconsistenza nel training degrada le predizioni su dati puliti, con correlazioni fra −0.89 e −0.94 (p<0.001) su tutti e 4 i modelli. E il degrado cresce col numero di FD corrotte: il calo dal baseline al 40% passa da 0.0270 (1 FD) a 0.0498 (5 FD) a 0.0697 (10 FD).

### Due risultati nuovi

**Gli indici sono ciechi verso l'errore coerente.** Nel braccio C, IM = IP = IH = 0 a tutti i livelli di rumore, anche quando il 40% delle righe ha la rotta sbagliata — ed è proprio quel dataset a produrre i modelli peggiori.

**Gli indici invertono segno oltre una soglia.** Con 10 FD corrotte, IM cresce fino al 20% di rumore e poi cala, mentre l'F1 continua a peggiorare: nel regime ≥20% la correlazione IM ↔ F1 diventa **positiva** (da +0.84 a +0.97, significativa su tutti e 4 i modelli). La causa è che corrompendo anche il determinante i gruppi si frammentano e i conflitti smettono di essere rilevabili.

---

## In parole più semplici

**Il problema principale che è stato corretto.** Prima si sporcavano i dati e poi si chiedeva all'intelligenza artificiale di indovinare *sempre partendo da dati sporchi*. Così però si misuravano due cose insieme: quanto l'IA avesse imparato male, e quanto fosse difficile rispondere avendo davanti informazioni sbagliate. Ora l'IA impara sui dati sporchi ma viene interrogata su dati corretti. Risultato: il danno vero è circa un terzo di quello che sembrava.

**Un errore nel confronto.** L'esperimento metteva a confronto due gruppi che dovevano differire per una cosa sola. In realtà a uno si rovinavano diciannove colonne e all'altro una soltanto: non era un confronto alla pari. È stato sistemato, e la conclusione si è capovolta — la storia completa è in [`relazione_braccio_C.md`](relazione_braccio_C.md).

**Le regole usate sono state controllate una per una.** Alcune "regole" che il programma aveva trovato nei dati erano vere solo per caso: per esempio "il giorno del mese determina il giorno della settimana" vale soltanto perché i dati coprono un mese solo. Altre erano vere su un campione piccolo ma false su quello grande. Sono state tutte scartate, tenendo solo quelle che hanno senso nella realtà (un aeroporto sta in una sola città, la distanza fra due città è fissa, e simili).

**Cosa è stato aggiunto.** La validazione incrociata (si ripete la prova cinque volte dividendo i dati in modo diverso, così il risultato non dipende da come è stato fatto il taglio), un livello di rumore più fine e fermo al 40%, e un nuovo esperimento che sporca una, cinque e dieci regole per volta per vedere se il danno cresce — cresce.

**Una cosa da sapere sugli strumenti di misura.** Gli indici che contano le contraddizioni nei dati hanno due punti ciechi: non vedono affatto gli errori "ordinati" (quelli che non si contraddicono, e che fanno più danno), e oltre una certa soglia di sporcizia iniziano a scendere mentre i dati continuano a peggiorare. Non sono strumenti sbagliati, ma è importante sapere fin dove arrivano.
