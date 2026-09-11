# Relazione finale — Misure di inconsistenza per Data-Centric AI

Relazione scientifica del progetto di tesi. Riporta la domanda di ricerca, il disegno sperimentale, i risultati e le conclusioni.

Documenti collegati: [`relazione_braccio_C.md`](relazione_braccio_C.md) per l'approfondimento sul braccio di controllo; [`relazione_revisione_sperimentale.md`](relazione_revisione_sperimentale.md) per il dettaglio delle modifiche metodologiche; [`variante_solo_braccioA/`](variante_solo_braccioA/) per una variante semplificata della tesi.

---

## 1. Domanda di ricerca

> **Quanto incide sulla qualità delle predizioni di un modello AI la qualità (consistenza) dei dati in input?**

**Vincoli metodologici** imposti dal relatore e rispettati in tutto il lavoro:
- i modelli devono restare **RAW**: iperparametri di default, nessun tuning per condizione sperimentale, nessun `class_weight`;
- l'obiettivo predittivo deve essere esplicito e le colonne corrotte devono essere coerenti con esso;
- il training va sporcato gradualmente, ma **il test va fatto su dati puliti**.

## 2. Framework di misura dell'inconsistenza

L'inconsistenza è definita rispetto a **dipendenze funzionali (FD)** della forma `LHS → RHS`: un insieme di colonne che determina univocamente un'altra colonna. Si costruisce un grafo dei conflitti — nodo = riga, arco = coppia di righe con stesso LHS ma RHS diverso — e se ne ricavano tre indici:

- **IM**: numero di conflitti a coppie (archi del grafo);
- **IP**: numero di tuple coinvolte in almeno un conflitto;
- **IH**: dimensione minima dell'insieme di tuple da correggere per eliminare tutte le violazioni (vertex cover minimo approssimato).

## 3. Obiettivo predittivo e dati

**Task**: classificazione a 5 classi della durata schedulata del volo (`DurataBucket`, da `CRSElapsedTime`: `<90m`, `90-150m`, `150-210m`, `210-300m`, `>300m`).

**Dati**: campione di 10.000 voli dal dataset "On-Time Reporting Carrier On-Time Performance" (gennaio 2025). Le classi sono bilanciate per undersampling, quindi il caso puro vale **0.20**.

**Baseline verificato** a rumore 0%, su dati puliti:

| Modello | F1 |
|---|---|
| Decision Tree | 0.899 |
| Random Forest | 0.892 |
| Logistic Regression | 0.877 |
| Neural Network | 0.821 |

Media: **0.8723**, nettamente sopra il caso puro. È una condizione necessaria: senza un baseline solido un eventuale degrado non sarebbe interpretabile.

## 4. Le dipendenze funzionali usate, e perché sono valide

Le FD non sono ipotizzate ma **scoperte empiricamente** da `discoverFDs.py`. Poiché la scoperta girava su un campione da 1.000 righe mentre gli esperimenti usano 10.000, e poiché il campione copre un solo mese, ogni FD candidata è stata verificata: 0 violazioni sulle 10.000 righe, **validità nel dominio reale**, e colonne che superano la blacklist raggiungendo davvero il modello.

| # | FD | Giustificazione nel mondo reale |
|---|---|---|
| 1 | `Origin+Dest → Distance` | la distanza fra due aeroporti è una quantità fisica fissa |
| 2 | `OriginAirportID → OriginCityName` | un aeroporto sta in una sola città |
| 3 | `DestAirportID → DestCityName` | simmetrico |
| 4 | `OriginAirportID → OriginState` | un aeroporto sta in un solo stato |
| 5 | `DestAirportID → DestState` | simmetrico |
| 6 | `OriginAirportID → Origin` | l'ID identifica il codice IATA dell'aeroporto |
| 7 | `DestAirportID → Dest` | simmetrico |
| 8 | `OriginAirportID → OriginWac` | il world area code è determinato dallo stato |
| 9 | `DestAirportID → DestWac` | simmetrico |
| 10 | `OriginAirportID → OriginStateName` | il nome dello stato è determinato dallo stato |

**FD scartate, con motivazione:**

| FD | Perché scartata |
|---|---|
| `DayofMonth → DayOfWeek`, `DayofMonth → FlightDate` | valide sui dati ma **false nella realtà**: dipendono dal fatto che il campione copre un solo mese |
| `Tail_Number → WeatherDelay`, `Flight_Number → CarrierDelay` | spurie, generate dall'81,4% di valori mancanti; si rompono già sulle 10.000 righe (47 e 225 violazioni) |
| `OriginAirportID → OriginAirportSeqID` | il SeqID è progettato per cambiare nel tempo |
| `Reporting_Airline → DOT_ID_...`, `Tail_Number → Reporting_Airline` | una colonna viene scartata dalla blacklist: il rumore muoverebbe IM/IP/IH senza toccare le metriche ML |
| `Reporting_Airline → IATA_CODE_...` | valida, ma il LHS ha solo 14 valori distinti: da sola genera 2.039.767 conflitti contro i 367.368 di tutte le altre nove, dominando gli indici |

## 5. Disegno sperimentale

### I due bracci

- **Braccio A (incoerente)**: per le righe selezionate corrompe in modo indipendente tutte le colonne del concetto "rotta" (`Origin`, `Dest`, `Distance` e 16 colonne ridondanti). Le righe diventano internamente contraddittorie: IM > 0.
- **Braccio C (coerente, controllo)**: riassegna a interi gruppi di righe il **profilo completo di un'altra rotta reale**. Stesse 19 colonne, numero di righe allineato, ma ogni gruppo resta internamente uniforme con valori realmente esistenti: **IM = 0 per costruzione**.

I due bracci differiscono quindi **solo per la coerenza interna**, che è la variabile da isolare.

### La valutazione

Il modello è sempre addestrato sulle righe **sporcate** e valutato due volte: sulle righe di test prese dal dataset sporco (`test_sporco`) e sulle **stesse** righe prese dal dataset pulito (`test_pulito`). Questo separa "il modello ha imparato peggio" da "gli stiamo dando input illeggibili".

### Configurazione

Livelli di rumore 0, 5%, 10%, 20%, 30%, 40%; 5 repliche seedate per configurazione; cross-validation stratificata a 5 fold; 4 modelli RAW.

## 6. Risultati

### 6.1 L'inconsistenza nel training degrada le predizioni

Braccio A, valutato su dati puliti:

| Rumore | IM | IH | F1 | Calo dal baseline |
|---|---|---|---|---|
| 0% | 0 | 0 | 0.8723 | 0.0000 |
| 5% | 55 | 30 | 0.8650 | 0.0073 |
| 10% | 113 | 64 | 0.8600 | 0.0123 |
| 20% | 190 | 121 | 0.8518 | 0.0205 |
| 30% | 307 | 202 | 0.8415 | 0.0308 |
| 40% | 378 | 280 | 0.8296 | 0.0427 |

Correlazione fra inconsistenza misurata e F1, per modello (tutte con p<0.001):

| Modello | IM vs F1 | IH vs F1 |
|---|---|---|
| Decision Tree | −0.889 | −0.894 |
| Logistic Regression | −0.928 | −0.935 |
| Neural Network | −0.940 | −0.963 |
| Random Forest | −0.933 | −0.955 |

### 6.2 Il degrado cresce col numero di FD corrotte

| Rumore | 1 FD | 5 FD | 10 FD |
|---|---|---|---|
| 0% | 0.8723 | 0.8723 | 0.8723 |
| 20% | 0.8556 | 0.8434 | 0.8308 |
| 40% | 0.8453 | 0.8225 | 0.8026 |

Calo dal baseline al 40%: **0.0270** → **0.0498** → **0.0697**.

### 6.3 Dove si misura cambia quanto danno si vede

| Rumore | A test sporco | C test sporco | A test pulito | C test pulito |
|---|---|---|---|---|
| 20% | 0.7992 | 0.8295 | 0.8518 | 0.8400 |
| 40% | 0.7540 | 0.8088 | 0.8296 | 0.8131 |

Al 40% di rumore, del calo totale di 0.1183 misurato sul test sporco solo il **36%** (0.0427) è dovuto a un modello che ha imparato peggio; il restante **64%** (0.0756) è il costo di interrogarlo su input corrotti. Valutare sul test sporco sovrastimava il danno di quasi tre volte.

### 6.4 L'errore coerente danneggia più di quello incoerente

Sul test pulito il confronto fra i bracci si **inverte** rispetto al test sporco: è il braccio C — quello coerente — a produrre i modelli peggiori (delta A−C medio +0.0102, significativo in 13 confronti su 20, sistematico dal 20-30% in su).

Isolando la sola componente di apprendimento, al 40% di rumore: braccio A **0.0427**, braccio C **0.0592**. Un errore coerente è indistinguibile da un dato vero e insegna al modello una regola falsa ma plausibile; un errore incoerente si auto-segnala come implausibile e viene in parte diluito come rumore.

Analisi completa in [`relazione_braccio_C.md`](relazione_braccio_C.md).

### 6.5 Gli indici hanno due punti ciechi

**Sono ciechi verso l'errore coerente.** Nel braccio C, IM = IP = IH = 0 a tutti i livelli di rumore, anche con il 40% delle righe che ha la rotta sbagliata. Gli indici certificano come perfettamente pulito il dataset che produce i modelli peggiori.

**Invertono segno oltre una soglia.** Con 10 FD corrotte IM cresce fino al 20% di rumore (391.837) e poi cala (362.421 al 40%), mentre l'F1 continua a peggiorare. Nel regime ≥20% la correlazione IM ↔ F1 diventa **positiva**:

| Modello | 1 FD | 5 FD | 10 FD |
|---|---|---|---|
| Decision Tree | −0.246 | −0.642* | **+0.839*** |
| Logistic Regression | −0.550* | −0.866* | **+0.853*** |
| Neural Network | −0.809* | −0.927* | **+0.970*** |
| Random Forest | −0.699* | −0.792* | **+0.894*** |

*(\* = p<0.05)*. Corrompendo anche il determinante della FD i gruppi si frammentano, quindi i conflitti rilevabili diminuiscono pur essendo i dati sempre più corrotti.

## 7. Risposta alla domanda di ricerca

> **Sì, la consistenza dei dati incide sulla qualità delle predizioni — ma meno di quanto sembrasse, e la consistenza non coincide con la qualità dei dati.**

Tre affermazioni, ciascuna supportata dai numeri:

**1. L'effetto esiste ed è sistematico.** L'inconsistenza nel training degrada le predizioni su dati puliti, con correlazioni fra −0.89 e −0.94 su tutti e 4 i modelli, e il degrado cresce col numero di dipendenze funzionali coinvolte. L'entità va però dichiarata con onestà: corrompere il 40% delle righe su dieci FD costa circa **7 punti di F1** su un baseline di 87.

**2. Dove si misura conta quanto si sporca.** Valutare su dati sporchi confonde l'effetto sull'apprendimento con il costo di predire da input corrotti, e sovrastima il danno di quasi tre volte. Nel confronto fra bracci arriva a **invertire il segno** della conclusione.

**3. Le misure di inconsistenza sono strumenti validi ma parziali.** Intercettano il tipo di sporcizia meno dannoso (l'errore incoerente) e sono cieche verso quello più dannoso (l'errore coerente); inoltre oltre una soglia di corruzione smettono di tracciare il danno e lo tracciano al contrario. Una pipeline data-centric che validasse i dati solo controllando le FD promuoverebbe a "puliti" proprio i dati che degradano di più il modello.

Questa terza affermazione è il contributo più solido del lavoro rispetto al titolo della tesi. Non invalida gli indici IM/IP/IH — misurano correttamente ciò che dichiarano — ma ne **delimita il campo di validità**, che è una conclusione più forte e più difendibile del claim iniziale, perché dice quando quel claim vale e quando no.

## 8. Limiti

- Un solo dataset, un solo obiettivo predittivo, un solo mese di dati (gennaio 2025).
- Il dataset effettivo dopo il bilanciamento è di 2.715 righe: piccolo.
- I concetti realmente indipendenti disponibili sono pochi (rotta, aeroporto di origine, aeroporto di destinazione); il decimo FD estende una famiglia già presente.
- Quattro modelli con iperparametri di default: un modello ottimizzato potrebbe reagire diversamente.
- La differenza A vs C sul test pulito è significativa in 13 confronti su 20, e di entità moderata (massimo +0.0165 di F1).
- Il meccanismo proposto al §6.4 è un'interpretazione coerente con i dati, non una misura diretta: verificarla richiederebbe di ispezionare cosa i modelli hanno appreso.

## 9. Materiale allegato

| File | Contenuto |
|---|---|
| `plot_blocco1_f1_test_pulito.png` | F1 sul test pulito, bracci A e C |
| `plot_blocco1_sporco_vs_pulito.png` | Perché il dataset di test cambia la conclusione |
| `plot_blocco1_f1_per_modello.png` | Dettaglio per ciascuno dei 4 modelli |
| `plot_blocco1_correlazioni.png` | Correlazione IM/IH vs F1 |
| `plot_blocco2_scaling_fd.png` | Effetto del numero di FD corrotte |
| `plot_blocco2_im_inversione.png` | L'inversione di segno di IM |
| `tabella_riassuntiva_blocco1.csv` / `.png` | F1 per livello di rumore, entrambi i bracci e le due valutazioni |
| `tabella_riassuntiva_blocco2.csv` / `.png` | F1 e IM per numero di FD |

---

## In parole più semplici

**La domanda.** Se i dati che diamo in pasto a un'intelligenza artificiale sono sporchi, l'IA sbaglia di più? E quanto?

**Come si è misurato.** Si prendono 10.000 voli e si chiede a quattro programmi di IA di indovinare quanto durerà un volo. Con dati corretti ci riescono bene: azzeccano circa 87 volte su 100, contro le 20 che otterrebbero tirando a caso. Poi si sporcano di proposito i dati di *addestramento*, in dosi crescenti, e si guarda quanto peggiorano — chiedendo però loro di indovinare su dati **corretti**, per capire quanto hanno davvero imparato male.

**Prima risposta: sì, ma poco.** Sporcando il 40% delle righe su dieci regole diverse, l'IA scende da 87 a 80. Un peggioramento reale e costante, ma non un crollo.

**Seconda risposta: dove si misura conta.** Nella versione precedente del progetto si interrogava l'IA usando anch'essi dati sporchi, e così il danno sembrava quasi tre volte più grande. Due terzi di quel danno non erano "l'IA ha imparato male", ma semplicemente "le stiamo facendo una domanda scritta male".

**La scoperta più interessante.** Si sono confrontati due modi di sporcare i dati. Nel primo le righe si contraddicono a vicenda (stessa rotta, distanze diverse). Nel secondo le righe sono sbagliate ma **tutte d'accordo fra loro** (un intero gruppo di voli diventa una rotta diversa, ma con la sua distanza giusta e le sue città giuste). Risultato: il secondo tipo fa **più** danno. Un errore ordinato è un bugiardo credibile — l'IA non ha modo di accorgersene e impara una regola falsa. Un errore contraddittorio invece si tradisce da solo.

**Perché questo conta per la tesi.** Gli strumenti che misurano le contraddizioni nei dati, applicati al secondo caso, dicono che i dati sono *perfettamente puliti* — eppure sono quelli che rovinano di più l'IA. E c'è un secondo punto cieco: oltre una certa soglia di sporcizia, questi strumenti iniziano addirittura a segnare *meno* contraddizioni mentre i dati peggiorano ancora.

La conclusione non è che questi strumenti siano sbagliati: misurano bene ciò che dicono di misurare. È che sono **parziali**, e sapere esattamente dove si fermano è un risultato più utile del semplice "dati sporchi uguale modello peggiore".
