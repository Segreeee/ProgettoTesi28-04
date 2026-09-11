# Variante di tesi: il solo braccio A

Versione alternativa e semplificata dell'esperimento, che considera **unicamente il braccio A** — la condizione in cui il rumore viola le dipendenze funzionali. Il braccio di controllo è deliberatamente escluso.

Nessun esperimento è stato rieseguito: i dati sono gli stessi del progetto principale, filtrati sul braccio A (`../blocco1_risultati_raw.csv`) e sull'esperimento di scaling, che è già solo braccio A per costruzione (`../blocco2_risultati_raw.csv`).

---

## 1. Domanda e obiettivo predittivo

**Domanda**: quanto incide la qualità dei dati di input sulla qualità delle predizioni di un modello AI?

**Obiettivo predittivo**: classificazione a 5 classi della durata schedulata del volo (`DurataBucket`, derivata da `CRSElapsedTime`: `<90m`, `90-150m`, `150-210m`, `210-300m`, `>300m`), su un campione di 10.000 voli del dataset "On-Time Reporting Carrier On-Time Performance".

**Disegno**: si corrompe progressivamente il training set violando una o più dipendenze funzionali reali, si misura l'inconsistenza risultante con gli indici IM/IP/IH, e si valuta quanto peggiorano le predizioni **su dati puliti**.

## 2. Premesse metodologiche

### Adottate prima del confronto con il relatore

- **Modelli RAW**: `LogisticRegression`, `RandomForestClassifier`, `DecisionTreeClassifier`, `MLPClassifier` con iperparametri di default, identici in ogni condizione sperimentale. Nessun `class_weight`, nessun tuning per livello di rumore. L'unica deviazione dai default è `max_iter=1000` per Logistic Regression e Neural Network, scelta neutra che garantisce solo la convergenza.
- **Blacklist anti-leakage**: rimosse tutte le colonne calcolate a posteriori o che rivelerebbero il target (`ArrDelay`, `AirTime`, `ActualElapsedTime`, ecc.), più `CRSElapsedTime` che è la fonte diretta del target e `CRSArrTime`/`ArrTimeBlk` che lo approssimano.
- **Bilanciamento delle classi** via undersampling casuale: stesso numero di righe per ciascuna delle 5 classi, così che l'accuratezza non premi la classe maggioritaria. Il caso puro vale quindi 0.20.
- **Dipendenze funzionali reali**, scoperte empiricamente da `discoverFDs.py` e non ipotizzate a priori.
- **Chiusura delle vie di fuga**: corrompendo una FD si corrompono anche le colonne ridondanti che codificano la stessa informazione (es. `OriginAirportID` è in corrispondenza biunivoca con `Origin`), altrimenti il modello ricostruirebbe l'informazione persa da una colonna rimasta pulita.
- **Seed fisso** per la riproducibilità della selezione delle righe da corrompere.
- **Baseline verificato**: a rumore 0% l'accuratezza è 0.87 contro un caso puro di 0.20. Condizione necessaria: senza un baseline solido un eventuale degrado non sarebbe interpretabile.

### Adottate dopo il confronto con il relatore

- **Training sporcato, test su dati puliti.** Il modello è addestrato sulle righe corrotte e valutato sulle *stesse* righe di test prese dal dataset pulito. È la correzione più importante: prima anche il test set era corrotto, il che confondeva "il modello ha imparato peggio" con "al modello si stanno dando input illeggibili".
- **Cross-validation stratificata a 5 fold**, al posto di un singolo split fisso 80/20.
- **Rumore fino al 40%** con granularità fine (`0, 5%, 10%, 20%, 30%, 40%`), invece che fino al 100%.
- **5 repliche seedate** per ogni configurazione, per distinguere l'effetto del rumore dalla variabilità statistica.
- **FD verificate nel dominio reale**, non solo sul campione (§3).
- **Scaling del numero di FD**: 1 → 5 → 10.

## 3. Le dipendenze funzionali usate, e perché

La scoperta automatica girava su un campione da 1.000 righe mentre gli esperimenti usano 10.000: due FD "valide" sul campione piccolo si rompono su quello grande (`Tail_Number → WeatherDelay`, 47 violazioni; `Flight_Number → CarrierDelay`, 225). Inoltre il campione copre **un solo mese** (gennaio 2025), il che genera FD vere sui dati ma false nella realtà.

Le 10 FD usate sono state verificate una per una: 0 violazioni sulle 10.000 righe, valide nel dominio reale, e con tutte le colonne che superano la blacklist e raggiungono davvero il modello.

| # | FD | Concetto | Giustificazione nel mondo reale |
|---|---|---|---|
| 1 | `Origin+Dest → Distance` | rotta | la distanza fra due aeroporti è una quantità fisica fissa |
| 2 | `OriginAirportID → OriginCityName` | aeroporto origine | un aeroporto sta in una sola città |
| 3 | `DestAirportID → DestCityName` | aeroporto destinazione | idem, simmetrico |
| 4 | `OriginAirportID → OriginState` | aeroporto origine | un aeroporto sta in un solo stato |
| 5 | `DestAirportID → DestState` | aeroporto destinazione | idem |
| 6 | `OriginAirportID → Origin` | aeroporto origine | l'ID identifica il codice IATA dell'aeroporto |
| 7 | `DestAirportID → Dest` | aeroporto destinazione | idem |
| 8 | `OriginAirportID → OriginWac` | aeroporto origine | il world area code è determinato dallo stato |
| 9 | `DestAirportID → DestWac` | aeroporto destinazione | idem |
| 10 | `OriginAirportID → OriginStateName` | aeroporto origine | il nome dello stato è determinato dallo stato |

**Escluse, con motivazione:**

| FD scartata | Perché |
|---|---|
| `DayofMonth → DayOfWeek`, `DayofMonth → FlightDate` | 0 violazioni sul campione, ma solo perché copre un mese singolo: nella realtà il 15 del mese può essere qualunque giorno della settimana |
| `Tail_Number → WeatherDelay`, `Flight_Number → CarrierDelay` | spurie, generate dall'81,4% di valori mancanti; si rompono già sulle 10.000 righe |
| `OriginAirportID → OriginAirportSeqID` | il SeqID è progettato per cambiare nel tempo: vale solo entro il mese osservato |
| `Reporting_Airline → DOT_ID_...`, `Tail_Number → Reporting_Airline` | una colonna della FD viene scartata dalla blacklist, quindi il rumore muoverebbe IM/IP/IH senza toccare le metriche del modello |
| `Reporting_Airline → IATA_CODE_...` | valida e sensata, ma il LHS ha solo 14 valori distinti: da sola genera 2.039.767 conflitti contro i 367.368 di tutte le altre nove messe insieme, dominando IM/IP/IH e rendendoli non confrontabili fra configurazioni |

## 4. Risultati

### 4.1 Degrado in funzione del rumore (1 FD)

| Rumore | IM | IH | F1 test sporco | F1 test pulito | Calo dal baseline |
|---|---|---|---|---|---|
| 0% | 0 | 0 | 0.8723 | 0.8723 | 0.0000 |
| 5% | 55 | 30 | 0.8477 | 0.8650 | 0.0073 |
| 10% | 113 | 64 | 0.8309 | 0.8600 | 0.0123 |
| 20% | 190 | 121 | 0.7992 | 0.8518 | 0.0205 |
| 30% | 307 | 202 | 0.7783 | 0.8415 | 0.0308 |
| 40% | 378 | 280 | 0.7540 | 0.8296 | 0.0427 |

Tutti e 4 i modelli peggiorano all'aumentare del rumore. Correlazione fra inconsistenza misurata e F1 sul test pulito:

| Modello | IM vs F1 | IH vs F1 |
|---|---|---|
| Decision Tree | −0.889 | −0.894 |
| Logistic Regression | −0.928 | −0.935 |
| Neural Network | −0.940 | −0.963 |
| Random Forest | −0.933 | −0.955 |

Tutte con p < 0.001.

### 4.2 Perché il dataset di test conta

Al 40% di rumore il modello ottiene 0.7540 se valutato sui dati sporchi e 0.8296 sui dati puliti. La differenza — 0.0756 di F1 — **non** misura un apprendimento peggiore, ma il costo di dare al modello input corrotti al momento della predizione.

Scomposto: del calo totale di 0.1183 osservato con il metodo precedente, solo il **36%** (0.0427) è attribuibile a un modello peggiore; il restante **64%** (0.0756) era l'effetto di interrogarlo su dati illeggibili. Misurare sul test sporco sovrastimava quindi il danno di quasi tre volte.

### 4.3 Effetto del numero di FD corrotte

| Rumore | F1 con 1 FD | F1 con 5 FD | F1 con 10 FD |
|---|---|---|---|
| 0% | 0.8723 | 0.8723 | 0.8723 |
| 5% | 0.8652 | 0.8622 | 0.8575 |
| 10% | 0.8619 | 0.8575 | 0.8491 |
| 20% | 0.8556 | 0.8434 | 0.8308 |
| 30% | 0.8497 | 0.8325 | 0.8164 |
| 40% | 0.8453 | 0.8225 | 0.8026 |

Il calo dal baseline al 40% di rumore cresce col numero di FD coinvolte: **0.0270** (1 FD) → **0.0498** (5 FD) → **0.0697** (10 FD). Corrompere più dipendenze funzionali contemporaneamente danneggia il modello più che corromperne una sola, come atteso.

### 4.4 Un limite di IM: oltre una soglia inverte segno

Con 10 FD corrotte, IM cresce fino al 20% di rumore (391.837 conflitti) e poi **cala** (362.421 al 40%), mentre l'F1 continua a peggiorare. Limitando la correlazione al regime ≥20% di rumore:

| Modello | 1 FD | 5 FD | 10 FD |
|---|---|---|---|
| Decision Tree | −0.246 | −0.642* | **+0.839*** |
| Logistic Regression | −0.550* | −0.866* | **+0.853*** |
| Neural Network | −0.809* | −0.927* | **+0.970*** |
| Random Forest | −0.699* | −0.792* | **+0.894*** |

*(\* = p<0.05)*

La causa è meccanica: corrompendo anche il determinante della FD, le righe si sparpagliano su gruppi diversi. I gruppi si **frammentano**, diventano più piccoli e più omogenei, e i conflitti misurabili diminuiscono — pur essendo i dati sempre più corrotti. IM smette quindi di tracciare il danno e, oltre quella soglia, lo traccia al contrario.

## 5. Conclusione

> La corruzione dei dati di training degrada le predizioni su dati puliti in modo sistematico e crescente col numero di dipendenze funzionali coinvolte. Gli indici IM/IP/IH ne forniscono una misura quantitativa che correla fortemente con il degrado (da −0.89 a −0.94, p<0.001) **entro un regime limitato**, oltre il quale la correlazione si indebolisce e può invertirsi.

L'entità va dichiarata con onestà: corrompere il 40% delle righe su dieci dipendenze funzionali costa circa **7 punti di F1** (0.8723 → 0.8026). L'effetto è reale, sistematico e riproducibile, ma non drammatico.

## 6. Limiti di questa variante

Due limiti sono intrinseci al disegno scelto e vanno dichiarati, perché ne delimitano il claim.

**Primo — nessuna attribuzione causale all'incoerenza.** Nel solo braccio A, IM cresce insieme al livello di rumore: la correlazione IM ↔ F1 è quindi in larga parte una correlazione fra *"quanto ho corrotto"* e *"quanto peggiora"*. Questi dati mostrano che **corrompere i dati danneggia il modello**, non che sia l'**incoerenza in quanto tale** a farlo — la spiegazione alternativa (è la perdita di informazione a fare danno, e l'incoerenza è solo il suo sintomo misurabile) resta pienamente compatibile con i risultati. Distinguere le due cose richiede un braccio di controllo con dati altrettanto sbagliati ma coerenti, che questa variante non include.

*Il progetto principale include quel braccio di controllo, e il suo esito è documentato in [`../relazione_braccio_C.md`](../relazione_braccio_C.md). Anticipazione: il risultato non conferma l'interpretazione intuitiva.*

**Secondo — IM non è un indicatore monotono.** Il risultato del §4.4 limita direttamente la conclusione del §5: gli indici di inconsistenza sono un buon proxy del danno solo finché il determinante della dipendenza resta sufficientemente integro. Oltre quella soglia il loro valore cala mentre il danno cresce, e usarli come criterio di qualità porterebbe a conclusioni sbagliate.

**Altri limiti**: un solo dataset, un solo obiettivo predittivo, un solo mese di dati, un numero ristretto di concetti realmente indipendenti (rotta, aeroporto di origine, aeroporto di destinazione), dataset effettivo di 2.715 righe dopo il bilanciamento, quattro soli modelli con iperparametri di default.

---

## In parole più semplici

**Cosa fa questo esperimento.** Prende 10.000 voli e chiede a quattro programmi di intelligenza artificiale di indovinare quanto durerà un volo. Con i dati corretti ci riescono bene: azzeccano circa 87 volte su 100, contro le 20 che otterrebbero tirando a caso. Poi si sporcano di proposito i dati di addestramento, in dosi crescenti fino al 40% delle righe, e si guarda quanto peggiorano.

**La regola più importante.** Si sporcano solo i dati con cui l'IA *impara*. Quando poi le si chiede di indovinare, le si danno dati **corretti**. Serve a separare due cose diverse: se l'IA sbaglia perché ha imparato male, oppure semplicemente perché le stiamo dando in pasto informazioni sbagliate al momento della domanda. Nella versione precedente del progetto le due cose erano mescolate, e questo faceva sembrare il danno quasi tre volte più grande di quello che è.

**Cosa succede.** Più si sporcano i dati, più l'IA peggiora: da 0.87 a 0.83 di punteggio. E più regole si violano contemporaneamente (una, cinque, dieci), più il danno cresce: fino a 0.80 con dieci regole violate. L'effetto è chiaro e costante, ma contenuto: circa 7 punti su 87 nel caso peggiore.

**Una sorpresa.** Gli indici che contano le contraddizioni nei dati funzionano bene finché la sporcizia è moderata. Ma oltre una certa soglia iniziano a *scendere* mentre i dati continuano a peggiorare — perché rompendo anche le colonne che fanno da "chiave", le righe si sparpagliano e le contraddizioni smettono di essere visibili. Insomma, lo strumento di misura si tara male proprio quando la situazione è più grave.

**Cosa questa variante non può dire.** Qui i dati vengono sporcati in un solo modo, quindi possiamo dire che *sporcare i dati fa danno*, ma non possiamo dire se il danno venga dal fatto che i dati **si contraddicono** oppure semplicemente dal fatto che **abbiamo cancellato informazione utile**. Per distinguerlo servirebbe un secondo esperimento di confronto — che esiste nel progetto principale ed è raccontato in [`../relazione_braccio_C.md`](../relazione_braccio_C.md).
