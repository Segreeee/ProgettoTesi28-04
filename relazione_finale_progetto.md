# Relazione finale — Misure di inconsistenza per Data-Centric AI

Relazione scientifica del progetto di tesi: domanda di ricerca, disegno sperimentale, risultati e conclusioni.

Il dettaglio delle modifiche metodologiche è in [`relazione_revisione_sperimentale.md`](relazione_revisione_sperimentale.md).

---

## 1. Domanda di ricerca

> **Quanto incide sulla qualità delle predizioni di un modello AI la qualità (consistenza) dei dati in input?**

Nella forma resa operativa dopo il confronto con il relatore: *se addestro un modello su dati contaminati ma poi lo uso su dati corretti, quanta capacità predittiva perdo, e come cresce questa perdita al crescere dell'inconsistenza?*

**Vincoli metodologici** rispettati in tutto il lavoro:
- i modelli restano **RAW**: iperparametri di default, `random_state=42`, nessun `class_weight`, nessun tuning per condizione sperimentale;
- l'obiettivo predittivo è esplicito e le colonne corrotte sono coerenti con esso;
- il training viene sporcato gradualmente, ma **il test è sempre su dati puliti**.

## 2. Framework di misura dell'inconsistenza

L'inconsistenza è definita rispetto a **dipendenze funzionali (FD)** della forma `LHS → RHS`: un insieme di colonne che determina univocamente un'altra colonna. Si costruisce un grafo dei conflitti — nodo = riga, arco = coppia di righe con stesso LHS ma RHS diverso — e se ne ricavano tre indici:

- **IM**: numero di conflitti a coppie (archi del grafo);
- **IP**: numero di tuple coinvolte in almeno un conflitto;
- **IH**: dimensione minima dell'insieme di tuple da correggere per eliminare tutte le violazioni (vertex cover minimo approssimato).

## 3. Dati e obiettivo predittivo

**Task**: classificazione a 5 classi della durata schedulata del volo (`DurataBucket`, da `CRSElapsedTime`: `<90m`, `90-150m`, `150-210m`, `210-300m`, `>300m`).

**Dati**: campione di **30.000 voli** dal dataset "On-Time Reporting Carrier On-Time Performance" (gennaio 2025). Le classi sono bilanciate per undersampling → **8.090 righe**, circa 6.470 di training per fold con K=5. Il caso puro vale quindi **0,20**.

**Baseline** a rumore 0%, su dati puliti (F1 weighted):

| Modello | F1 |
|---|---|
| Random Forest | 0,9345 |
| Decision Tree | 0,9218 |
| Logistic Regression | 0,9218 |
| Neural Network | 0,9152 |

Media: **0,9233**, nettamente sopra il caso puro — condizione necessaria perché un degrado sia interpretabile.

## 4. Le dipendenze funzionali usate, e perché sono valide

Le FD non sono ipotizzate ma **scoperte empiricamente** (`discoverFDs.py`) e poi filtrate con tre criteri: 0 violazioni sul campione effettivamente usato, **validità nel dominio reale**, e colonne che superano la blacklist anti-leakage arrivando davvero al modello. La certificazione è in `blocco2_verifica_fd.csv`.

**Blocco 1** corrompe una sola FD, `Origin + Dest → Distance` (una rotta ha una distanza fissa), insieme alle **16 colonne ridondanti** che codificano la stessa informazione (`OriginAirportID`, `DestAirportID`, City, State, StateName, StateFips, Wac, CityMarketID, AirportSeqID per partenza e arrivo): tutte verificate a 0 violazioni, altrimenti il modello ricostruirebbe da lì l'informazione persa.

**Blocco 2** usa un insieme di 10 FD, tutte con 0 violazioni:

| # | FD | Giustificazione nel mondo reale |
|---|---|---|
| 1 | `Origin+Dest → Distance` | la distanza fra due aeroporti è una quantità fisica fissa |
| 2 | `OriginAirportID → OriginCityName` | un aeroporto sta in una sola città |
| 3 | `DestAirportID → DestCityName` | idem, simmetrico |
| 4 | `OriginAirportID → OriginState` | un aeroporto sta in un solo stato |
| 5 | `DestAirportID → DestState` | idem |
| 6 | `OriginAirportID → Origin` | l'ID identifica il codice IATA dell'aeroporto |
| 7 | `DestAirportID → Dest` | idem |
| 8 | `OriginAirportID → OriginWac` | il World Area Code è determinato dallo stato |
| 9 | `DestAirportID → DestWac` | idem |
| 10 | `OriginAirportID → OriginStateName` | il nome dello stato è determinato dallo stato |

**FD scartate, con motivazione:**

| FD | Perché scartata |
|---|---|
| `DayofMonth → DayOfWeek`, `DayofMonth → FlightDate` | valide sui dati ma **false nella realtà**: dipendono dal fatto che il dataset copre un solo mese |
| `Tail_Number → WeatherDelay`, `Flight_Number → CarrierDelay` | spurie, generate dall'81,4% di valori mancanti; si rompono già su un campione più grande |
| `Reporting_Airline → DOT_ID_...`, `Tail_Number → Reporting_Airline` | una colonna viene scartata dalla blacklist: il rumore muoverebbe IM/IP/IH senza toccare le metriche del modello |
| `Reporting_Airline → IATA_CODE_...` | valida, ma il LHS ha solo 14 valori distinti: genererebbe da sola molti più conflitti di tutte le altre, dominando gli indici |

## 5. Disegno sperimentale

Per ogni livello di rumore e replica: si corrompono le colonne delle FD scelte sul campione, poi si addestrano i 4 modelli RAW in **cross-validation stratificata a 5 fold** sulle righe sporcate, e si valuta **sulle stesse righe di test prese dal campione pulito** (oltre che, per confronto, su quelle sporcate).

Livelli di rumore: **0, 5, 10, 20, 30, 40%**; 5 repliche con seed diversi per livello.

**Verifica della garanzia.** Ogni risultato riporta la quota di righe di training che differiscono dal campione pulito, misurata riga per riga: vale **0 a rumore 0%** e coincide con il livello richiesto (0,0495 · 0,0977 · 0,1977 · 0,2982 · 0,3976). A rumore 0% le due valutazioni danno numeri identici a sei decimali. L'analisi si interrompe se uno di questi controlli fallisce.

## 6. Risultati

### 6.1 L'inconsistenza nel training degrada le predizioni

F1 sul test pulito, Blocco 1:

| Rumore | IM | IH | Decision Tree | Logistic Regression | Neural Network | Random Forest | Media |
|---|---|---|---|---|---|---|---|
| 0% | 0 | 0 | 0,9218 | 0,9218 | 0,9152 | 0,9345 | 0,9233 |
| 5% | 411 | 117 | 0,9216 | 0,9047 | 0,9122 | 0,9295 | 0,9170 |
| 10% | 787 | 260 | 0,9200 | 0,9004 | 0,9084 | 0,9285 | 0,9143 |
| 20% | 1.540 | 588 | 0,9121 | 0,8927 | 0,9007 | 0,9238 | 0,9073 |
| 30% | 2.257 | 984 | 0,9083 | 0,8870 | 0,8949 | 0,9178 | 0,9020 |
| 40% | 2.816 | 1.421 | 0,9046 | 0,8821 | 0,8830 | 0,9093 | 0,8948 |

Il degrado è **regolare e monotono** su tutti e 4 i modelli. Al 40% di rumore il calo va da **1,7 punti** (Decision Tree) a **4,0** (Logistic Regression), con una media di **2,9 punti**. Il calo è statisticamente significativo in **19 confronti su 20** (t-test sulle 5 repliche contro il baseline; al 40% tutti con p < 0,001).

Correlazione di Spearman fra inconsistenza e F1 sul test pulito, per modello:

| Modello | IM | IP | IH |
|---|---|---|---|
| Logistic Regression | −0,985 | −0,985 | −0,987 |
| Neural Network | −0,970 | −0,971 | −0,974 |
| Random Forest | −0,960 | −0,961 | −0,960 |
| Decision Tree | −0,907 | −0,917 | −0,906 |

### 6.2 Dove si misura cambia quanto danno si vede

Valutando lo stesso modello sul test sporcato invece che su quello pulito:

| Rumore | F1 test pulito | F1 test sporco | Perdita da apprendimento | Perdita da input corrotto | Quota da apprendimento |
|---|---|---|---|---|---|
| 10% | 0,9143 | 0,8851 | 0,0090 | 0,0292 | 23,6% |
| 20% | 0,9073 | 0,8575 | 0,0160 | 0,0498 | 24,3% |
| 30% | 0,9020 | 0,8369 | 0,0213 | 0,0650 | 24,7% |
| 40% | 0,8948 | 0,8164 | 0,0286 | 0,0784 | 26,7% |

Al 40% il calo misurato sul test sporcato è di **10,7 punti**, ma solo **2,9 sono dovuti a un apprendimento peggiore**: gli altri 7,8 sono il costo di interrogare il modello su input corrotti. **Misurare sul test sporco sovrastima il danno di circa quattro volte**, e la proporzione è stabile a ogni livello (fra il 24% e il 27%).

### 6.3 Più dipendenze funzionali violate, più danno

F1 sul test pulito al variare del numero di FD corrotte (Blocco 2):

| Rumore | 1 FD | 5 FD | 10 FD |
|---|---|---|---|
| 0% | 0,9233 | 0,9233 | 0,9233 |
| 10% | 0,9141 | 0,9104 | 0,9071 |
| 20% | 0,9109 | 0,9039 | 0,8961 |
| 30% | 0,9070 | 0,8966 | 0,8848 |
| 40% | 0,9034 | 0,8900 | 0,8724 |

Calo dal baseline al 40%: **0,0199** (1 FD) → **0,0333** (5 FD) → **0,0510** (10 FD). L'ordine è rispettato a ogni livello di rumore: violare più dipendenze funzionali contemporaneamente danneggia il modello più che violarne una sola, e il danno cresce in modo regolare.

### 6.4 Oltre una soglia, IM smette di misurare il danno

| Rumore | IM con 1 FD | IM con 5 FD | IM con 10 FD |
|---|---|---|---|
| 5% | 411 | 519.945 | 2.065.172 |
| 10% | 787 | 966.149 | 2.958.915 |
| 20% | 1.540 | 1.667.317 | **3.268.942** |
| 30% | 2.257 | 2.131.686 | 3.070.086 |
| 40% | 2.816 | 2.434.518 | 2.917.019 |

Con 1 e 5 FD, IM cresce sempre con il rumore. Con **10 FD** raggiunge il massimo al **20%** e poi **cala**, mentre l'F1 continua a peggiorare (da 0,8961 a 0,8724 fra il 20% e il 40%).

La causa è meccanica: corrompendo anche il lato sinistro delle dipendenze, le righe si sparpagliano su determinanti diversi. I gruppi si **frammentano**, diventano più piccoli e più omogenei, e i conflitti rilevabili diminuiscono — pur essendo i dati sempre più corrotti. Oltre quella soglia, quindi, IM non traccia più il danno.

## 7. Risposta alla domanda di ricerca

> **Sì, l'inconsistenza dei dati di training riduce la qualità delle predizioni su dati corretti, in modo regolare e misurabile — ma l'entità è moderata, e le misure di inconsistenza descrivono bene il danno solo entro un certo regime.**

Tre affermazioni, ciascuna sostenuta dai numeri:

**1. L'effetto esiste, è sistematico e cresce col numero di vincoli violati.** Con il 40% delle righe di training corrotte si perdono 2,9 punti di F1 con una FD e 5,1 con dieci, su un baseline di 92,3. Le correlazioni fra inconsistenza e qualità delle predizioni vanno da −0,91 a −0,99, e il degrado è significativo in 19 confronti su 20.

**2. Dove si misura conta quanto si sporca.** Valutare sul test sporcato confonde l'effetto sull'apprendimento con il costo di predire da input corrotti, e sovrastima il danno di circa quattro volte: solo un quarto del calo osservato con quel metodo riguarda davvero il modello.

**3. IM è un indicatore valido solo entro un regime limitato.** Con poche FD segue fedelmente il degrado; con dieci FD raggiunge il massimo al 20% di rumore e poi diminuisce mentre il danno continua a crescere. Usarlo come criterio di qualità dei dati oltre quella soglia porterebbe a conclusioni sbagliate.

## 8. Limiti

- **Un solo mese di dati** (gennaio 2025): il dataset originale non copre altro periodo.
- **Campione di 30.000 righe**, scelto come massimo compatibile con la costruzione in memoria del grafo dei conflitti (il numero di archi cresce circa col quadrato delle righe).
- Dopo il bilanciamento il dataset effettivo è di **8.090 righe**: i modelli lavorano su circa 6.470 righe per fold.
- **Rumore casuale uniforme**: i valori sporchi sono estratti a caso fra quelli possibili, mentre gli errori reali tendono a essere sistematici (refusi, codici scambiati, valori vicini al vero).
- **IM misura solo le FD dichiarate**: le violazioni introdotte sporcando le colonne ridondanti non vengono conteggiate.
- **Modelli con iperparametri di default**, come richiesto: un modello ottimizzato potrebbe reagire diversamente.
- Il meccanismo proposto per la saturazione di IM (§6.4) è coerente con i dati ma non è stato verificato ispezionando direttamente la struttura dei gruppi a ogni livello.

## 9. Materiale allegato

| File | Contenuto |
|---|---|
| `plot_blocco1_f1_test_pulito.png` | F1 sul test pulito per modello, al crescere del rumore |
| `plot_blocco1_sporco_vs_pulito.png` | Perché il dataset di test cambia la conclusione |
| `plot_blocco1_correlazioni.png` | Correlazione IM/IH con F1, per modello |
| `plot_blocco2_scaling_fd.png` | Effetto del numero di FD corrotte su F1 e su IM |
| `plot_blocco2_im_e_f1.png` | La saturazione di IM con 10 FD |
| `tabella_riassuntiva_blocco1.csv` / `.png`, `tabella_riassuntiva_blocco2.csv` / `.png` | Tabelle riassuntive |
| `blocco2_verifica_fd.csv` | Certificazione delle 10 FD |
| `blocco1_scomposizione.csv`, `blocco1_correlazioni.csv`, `blocco1_test_degrado.csv` | Analisi statistica |

---

## In parole più semplici

**La domanda.** Se un programma di intelligenza artificiale impara da dati pieni di errori, ma poi lo si usa su dati corretti, quanto sbaglia in più?

**Come si è misurato.** Si prendono 30.000 voli e si chiede a quattro programmi di indovinare quanto durerà un volo. Con dati corretti ci riescono bene: azzeccano circa 92 volte su 100, contro le 20 che otterrebbero tirando a caso. Poi si rovinano di proposito i dati con cui imparano — la rotta, la distanza, la città, lo stato — su una quota crescente di voli, fino al 40%. Le domande di verifica, invece, si fanno sempre con dati corretti, e il programma controlla a ogni passaggio che sia davvero così.

**Cosa si è scoperto.**

- Rovinare i dati fa peggiorare le previsioni, in modo regolare: con il 40% dei voli rovinati si perdono circa 3 punti su 92. Più regole si violano insieme, più il danno cresce: da 2 punti con una regola a 5 con dieci.
- **Il modo di misurare conta più di quanto si pensi.** Se anche le domande di verifica si fanno con dati rovinati, il danno sembra di 11 punti invece che di 3. Tre quarti di quel danno non è "l'IA ha imparato male", ma semplicemente "le stiamo facendo una domanda scritta male".
- **Gli strumenti che contano le contraddizioni hanno un punto cieco.** Quando si violano dieci regole insieme, il conteggio delle contraddizioni cresce fino al 20% di dati rovinati e poi *diminuisce*, mentre l'IA continua a peggiorare. Succede perché rompendo anche le colonne che fanno da chiave le righe si sparpagliano, e le contraddizioni smettono di essere visibili. Insomma, il misuratore si tara male proprio quando la situazione peggiora.
