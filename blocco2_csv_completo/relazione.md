# Blocco 2 su un campione del 45% del CSV originale — relazione sul lavoro

Questo documento descrive lo sviluppo della roadmap del Blocco 2 (*addestramento su dati sporchi, test su dati puliti*) su un campione ampio del dataset originale: cosa è stato seguito così com'era, cosa è stato cambiato e perché, come è stato verificato, e cosa ne è emerso.

Tutti i numeri provengono dai file prodotti in questa cartella.

---

## 1. Obiettivo

> *Se addestro un modello su dati contaminati ma poi lo uso nel mondo reale, dove i dati sono corretti, quanta capacità predittiva perdo — e come cresce questa perdita al crescere dell'inconsistenza?*

Rispetto al lavoro precedente cambiano due cose: il **test è sempre su dati puliti**, e i dati non sono più il campione da 10.000 righe ma **242.887 voli**, il 45% del CSV originale.

## 2. Cosa della roadmap è stato seguito così com'è

- **Obiettivo predittivo**: `DurataBucket`, le stesse 5 fasce di durata schedulata del volo.
- **Blacklist anti-leakage** identica a quella del progetto principale, inclusi `CRSElapsedTime`, `CRSArrTime`, `ArrTimeBlk`.
- **Quattro modelli RAW**: Logistic Regression, Random Forest, Decision Tree, Neural Network, con iperparametri di default, `random_state=42`, nessun `class_weight`, nessun tuning.
- **Bilanciamento una volta sola**, sul dataset pulito, prima di tutto (§4.2): i fold contengono le stesse righe a ogni livello di rumore.
- **Prima si divide, poi si sporca**, e solo il training (§4.1).
- **Livelli di rumore** 0, 5, 10, …, 40% (§4.3).
- **Configurazioni cumulative** di FD (§4.4).
- **IM/IP/IH sul training sporcato effettivo** (§6.2), non sul dataset intero.
- **Preprocessore addestrato solo sul training**, con `handle_unknown='ignore'` (§5.2).
- **F1 weighted e F1 macro** registrati entrambi (§7), e certificazione delle FD su file (§3.3).
- **Assert sul test** a ogni fold (§10), rafforzato come descritto al §5.

## 3. Deviazioni dalla roadmap, e perché

| # | Deviazione | Motivo |
|---|---|---|
| 1 | **Campione del 45% del CSV** invece di 10.000 righe | Richiesta. Campionamento stratificato per classe di durata (proporzioni identiche al CSV), seed 42 |
| 2 | **FD5 esclusa e K=3** invece di 5 | Le due regole di riduzione della roadmap stessa (§4.5). Misurato: sul CSV intero la sola rete neurale richiede 580 s per fold, e il disegno completo avrebbe richiesto ~35 ore in sequenza |
| 3 | **Stesse righe sporcate per tutte le FD** | La roadmap chiede di riusare una funzione che sceglie righe diverse per ogni FD (§5.4), ma formula l'ipotesi "a parità di percentuale di righe sporcate" (§4.4). Con righe indipendenti, al 40% le righe toccate sarebbero passate dal 40% (C1) all'87% (C4), confondendo "più FD" con "più righe" |
| 4 | **Geografia dell'aeroporto di arrivo sporcata con FD1** | Nella roadmap la sporcava FD5. Senza FD5 sarebbe rimasta sempre pulita: `DestCityName` identifica l'aeroporto quasi uno a uno, quindi il modello avrebbe continuato a conoscere la destinazione |
| 5 | **IM/IP/IH calcolati senza costruire il grafo** | La roadmap chiede di usare la funzione esistente basata su networkx (§6.1). Su questi volumi il grafo arriva a milioni di archi per ogni addestramento: non è costruibile in memoria (dettagli al §4) |
| 6 | **One-hot in formato sparso** | In formato denso la matrice di training occuperebbe centinaia di MB per fold e per processo. Predizioni identiche (§4) |
| 7 | **Sporcatura vettoriale, un generatore per colonna** | La versione a ciclo cella per cella è lenta su decine di migliaia di righe. Il nuovo schema produce la stessa distribuzione e rende le configurazioni esattamente annidate |
| 8 | **Baseline condiviso a rumore 0%** | A 0% non si sporca nulla e le 4 configurazioni ricevono dati identici: un solo addestramento per fold, registrato per ciascuna. 99 addestramenti invece di 108 |
| 9 | **Esecuzione parallela** | Più addestramenti contemporanei, numero di processi adattato alla RAM. Non cambia i risultati (verificato) |

**Conseguenza da tenere presente**: poiché IH è ora calcolato in modo diverso (§4), i suoi valori **non sono confrontabili** con quelli dei progetti precedenti.

## 4. Verifiche eseguite prima dell'esperimento

### 4.1 I dati

| | |
|---|---|
| Righe del CSV originale | 539.747 (un solo mese: gennaio 2025) |
| Campione del 45% | 242.887 |
| Classe più rara (`>300m`) nel campione | 13.354 |
| Dataset bilanciato | **66.770** righe (13.354 × 5 classi) |
| Training / test per fold | ~44.513 / ~22.257 |
| Colonne interamente vuote eliminate | 21 (`Div2…`–`Div5…`, `Unnamed: 109`) — l'imputer le scartava comunque |

### 4.2 Le dipendenze funzionali

Certificazione in `blocco2_verifica_fd.csv`: **0 violazioni** e nessun valore mancante, sia sul CSV completo sia sul dataset bilanciato, per le 4 FD e per le 11 FD che giustificano le colonne ridondanti. Tutte le colonne coinvolte arrivano al modello.

| FD | Dipendenza | Gruppi (CSV completo) | Significato, verificato sui dati |
|---|---|---|---|
| FD1 | `Origin + Dest → Distance` | 5.649 | una rotta ha una distanza fissa |
| FD2 | `Distance → DistanceGroup` | 1.487 | `DistanceGroup` sono fasce esatte di 250 miglia (1 = 31–249, 2 = 250–497, …, 11 = ≥ 2.500) |
| FD3 | `CRSDepTime → DepTimeBlk` | 1.209 | `DepTimeBlk` è la fascia oraria di `CRSDepTime` (19 fasce) |
| FD4 | `Origin → OriginWac` | 329 | un aeroporto sta in una sola area geografica |

### 4.3 Il calcolo delle misure di inconsistenza

- **IM** — coppie di righe in conflitto — è calcolato **esattamente** per inclusione-esclusione su conteggi di coppie per gruppo, senza enumerare gli archi.
- **IP** — righe coinvolte in almeno un conflitto — è **esatto**: una riga è in conflitto se e solo se il suo gruppo ha più di un valore del lato destro.
- **IH** — righe da correggere — è l'**unione delle coperture minime delle singole FD** (in ogni gruppo si tiene il valore più frequente e si correggono le altre righe). È una copertura valida del grafo complessivo, **esattamente minima con una sola FD**; è riportato anche un limite inferiore (`IH_min`).

Validazione (`test_motore.py`, sul campione da 10.000 righe): in tutte le 8 combinazioni di configurazione e rumore provate, IM e IP coincidono **sia** con un grafo networkx costruito esplicitamente **sia** con il confronto di tutte le coppie di righe una per una. In tutti i casi, togliendo le righe della copertura IH non resta alcun conflitto. In un confronto preliminare IH è sempre risultato più basso della vecchia approssimazione basata su networkx (es. 217 contro 279), quindi più vicino al minimo reale.

### 4.4 Formato sparso e parallelismo

Sul campione da 10.000 righe, stesso fold: con one-hot denso e sparso le predizioni sono **identiche riga per riga** per tutti e 4 i modelli (0 differenze su 543). Idem per il Random Forest con numero di thread diverso.

### 4.5 La garanzia: training sporcato, test pulito

Tre livelli indipendenti:

1. **Struttura**: il test di ogni fold è estratto dal dataset pulito e non viene mai passato alla funzione di sporcatura, che riceve solo una copia del training.
2. **Assert a ogni addestramento**: righe di training modificate = ⌊livello × righe di training⌋ (0 a rumore 0%); test identico, riga per riga, al dataset pulito e privo di violazioni; training e test disgiunti. Le righe modificate sono **misurate** confrontando l'impronta (hash) di ogni riga con quella del dataset pulito originale, non dichiarate da chi ha sporcato.
3. **Verifica a posteriori**: le stesse due quantità sono salvate in ogni riga dei risultati, e l'analisi si rifiuta di procedere se anche una non torna.

I controlli preliminari (`test_motore.py`) hanno superato **36 verifiche su 36**: ad esempio, al 40% di rumore le righe di training modificate sono esattamente 17.805 = ⌊0,40 × 44.513⌋ in tutte e 4 le configurazioni, 0 a rumore 0%; ogni cella sporcata ha un valore diverso dall'originale; il test resta identico dopo tutte le sporcature; e ogni configurazione coincide con la precedente sulle colonne comuni.

## 5. Il disegno sperimentale

**Configurazioni e colonne sporcate** (cumulative, sulle stesse righe):

| Config | Colonne aggiunte | Totale |
|---|---|---|
| C1 = FD1 | `Origin`, `Dest`, `Distance` + `OriginAirportID`, `DestAirportID`, `DestCityName`, `DestState`, `DestStateName`, `DestStateFips`, `DestWac` | 10 |
| C2 = + FD2 | `DistanceGroup` | 11 |
| C3 = + FD3 | `CRSDepTime`, `DepTimeBlk` | 13 |
| C4 = + FD4 | `OriginWac`, `OriginCityName`, `OriginState`, `OriginStateName`, `OriginStateFips` | 18 |

Ogni colonna è sporcata indipendentemente dalle altre, sostituendo il valore con uno diverso estratto fra quelli che la colonna assume: le righe sporcate diventano incoerenti e violano le FD. Poiché righe e valori sporchi di ogni colonna sono gli stessi in tutte le configurazioni, **C(k+1) differisce da C(k) solo per le colonne aggiunte**: il confronto fra configurazioni adiacenti isola esattamente il loro effetto.

Proprietà del disegno da tenere presente: la geografia dell'aeroporto di **partenza** resta pulita fino a C4 (come nella roadmap), quella di **arrivo** è sporcata già da C1. Le configurazioni chiudono progressivamente le vie attraverso cui il modello può ricostruire l'informazione persa.

**Dimensione**: 3 fold × (1 baseline + 8 livelli × 4 configurazioni) = **99 addestramenti**, ciascuno dei 4 modelli.

## 6. Esecuzione

99 addestramenti in **1 ora e 53 minuti**, con 4 processi paralleli. La rete neurale ha richiesto in media 232 s per addestramento (132 iterazioni); gli altri tre modelli fra 4 e 20 s.

**La garanzia è stata rispettata su tutte le 432 righe dei risultati**: a rumore maggiore di zero le righe di training sporcate vanno da 2.225 (5%) a 17.805 (40%), sempre pari a ⌊livello × righe⌋; a rumore 0% sono zero; le righe di test modificate sono **zero in ogni riga**.

## 7. Risultati

### 7.1 Il baseline

A rumore 0%, su dati puliti, l'accuracy vale **0,9717** (Decision Tree), **0,9534** (Logistic Regression), **0,9712** (Neural Network), **0,9766** (Random Forest), contro un caso puro di 0,20. F1 weighted e F1 macro coincidono (differenza massima 0,000004): il bilanciamento ha funzionato.

### 7.2 La curva di degrado

F1 sul test pulito, media dei 4 modelli (`plot_f1_per_configurazione.png`, `tabella_riassuntiva.csv`):

| Rumore | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| 0% | 0,9682 | 0,9682 | 0,9682 | 0,9682 |
| 10% | 0,9610 | 0,9439 | 0,9444 | 0,9436 |
| 20% | 0,9578 | 0,9259 | 0,9258 | 0,9248 |
| 30% | 0,9549 | 0,9113 | 0,9112 | 0,9095 |
| 40% | 0,9512 | 0,8979 | 0,8978 | 0,8955 |

Il degrado cresce col rumore in tutte le configurazioni. Quasi tutta la differenza fra configurazioni sta nel passaggio **C1 → C2**; C2, C3 e C4 sono praticamente sovrapposte.

### 7.3 La media nasconde due comportamenti opposti

F1 sul test pulito per modello (`plot_f1_per_modello.png`):

| Modello | Baseline | C1 al 40% | C2 al 40% | C4 al 40% | Calo al 40% |
|---|---|---|---|---|---|
| Random Forest | 0,9765 | 0,9662 | 0,9657 | 0,9660 | 1,0–1,1 punti |
| Decision Tree | 0,9716 | 0,9582 | 0,9542 | 0,9553 | 1,3–1,7 punti |
| Neural Network | 0,9711 | 0,9602 | 0,9544 | 0,9538 | 1,1–1,7 punti |
| **Logistic Regression** | 0,9533 | 0,9200 | **0,7171** | **0,7067** | **3,3–24,7 punti** |

**Tre modelli su quattro sono robusti**: con il 40% delle righe di training sporcate su 18 colonne, perdono meno di 2 punti di F1.

**La Logistic Regression è fragile, e in modo molto specifico**: in C1 perde 3,3 punti, ma quando si sporca anche `DistanceGroup` (C2) ne perde 23,6. Il salto C1 → C2 è significativo per la Logistic Regression a **tutti gli 8 livelli di rumore** (t-test appaiato sui 3 fold: p < 0,005 a ogni livello, p < 0,001 dal 10% in su), mentre per gli altri modelli lo è in al più 1 livello su 8.

Una spiegazione coerente con i dati — ma non una misura diretta — è la natura lineare del modello. La durata schedulata dipende quasi linearmente dalla distanza: `Distance` e `DistanceGroup` sono il suo segnale principale. Sostituire valori numerici con valori casuali attenua il coefficiente stimato (è il fenomeno noto come *attenuazione* dei predittori rumorosi). Finché `DistanceGroup` resta pulita, il modello ha ancora un segnale lineare corretto; quando si sporcano entrambe, lo perde. Alberi e foreste, invece, dividono i dati per soglie e combinano `Origin` e `Dest`: riescono a imparare la relazione dalla maggioranza pulita delle righe.

### 7.4 Conta quale informazione si distrugge, non quante FD si violano

Confronto fra configurazioni adiacenti, sulle 96 coppie modello × livello × fold con rumore > 0:

| Passaggio | Cosa si aggiunge | IM al 40% | F1 peggiora / migliora | Delta medio di F1 |
|---|---|---|---|---|
| C1 → C2 | fascia di distanza | 5.998 → 407.108 | 77 / 19 | −0,0332 |
| C2 → C3 | fascia oraria di partenza | 407.108 → 921.185 | 47 / 49 | +0,0000 |
| C3 → C4 | geografia di partenza | 921.185 → 2.826.212 | 58 / 38 | −0,0012 |

Aggiungere la fascia oraria **più che raddoppia IM ma non ha alcun effetto** sul modello (47 coppie peggiorano, 49 migliorano: puro rumore): l'orario di partenza non dice nulla sulla durata schedulata di un volo. Aggiungere la geografia di partenza **triplica IM** con un effetto di un decimo di punto. Da C2 a C4, al 40% di rumore, **IM cresce di 6,9 volte mentre F1 passa da 0,8979 a 0,8955**.

Il danno dipende da *quanto è predittiva l'informazione distrutta* (la distanza lo è moltissimo, l'orario per nulla), non dal numero di FD violate né dal numero di coppie in conflitto.

### 7.5 IM, IP e IH non misurano la stessa cosa

(`plot_inconsistenza.png`, `plot_correlazioni.png`)

- **IM** cresce di ordini di grandezza fra configurazioni ed è dominato dalle FD con gruppi grandi: in C4 al 40% il 67,7% dei conflitti viene da FD4 (`Origin → OriginWac`: nel training i suoi gruppi contano in media 138 righe, contro 9 di FD1), lo 0,2% da FD1. La sua dimensione riflette la **struttura dei gruppi**, non la gravità del danno.
- **IP** si satura: tutte le righe di training risultano coinvolte in almeno un conflitto già dal 5% di rumore in C4, dal 20% in C3 e dal 35% in C2 (solo in C1 non si satura mai). Da lì in poi smette di distinguere i livelli.
- **IH** — le righe da correggere — cresce in modo quasi lineare col rumore ed è simile fra C2, C3 e C4 (al 40%: 16.302, 18.122, 18.369): è la misura che più rispecchia la quantità di dati effettivamente guasti.

Correlazione di Spearman con F1 sul test pulito (99 punti per modello, tutte con p < 0,001):

| Modello | IM | IP | IH |
|---|---|---|---|
| Logistic Regression | −0,839 | −0,817 | **−0,994** |
| Neural Network | −0,546 | −0,545 | −0,760 |
| Decision Tree | −0,486 | −0,478 | −0,721 |
| Random Forest | −0,425 | −0,400 | −0,659 |

Tutte negative e significative, ma **IH è sempre la misura più legata al degrado**, e IM la meno affidabile fra quelle non sature. Nessuna inversione di segno di IM è stata osservata in questo disegno: IM e IH crescono in modo monotono col rumore in tutte le configurazioni.

### 7.6 Esiste una soglia di contaminazione tollerabile?

Livello di rumore fino al quale il calo di F1 rispetto al baseline resta sotto 1 punto (`blocco2_soglie.csv`):

| Modello | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| Random Forest | 35% | 35% | 35% | 35% |
| Decision Tree | 30% | 25% | 25% | 25% |
| Neural Network | 30% | 25% | 25% | 25% |
| Logistic Regression | 0% | 0% | 0% | 0% |
| *media dei 4 modelli* | *15%* | *0%* | *0%* | *0%* |

Una risposta pratica: **sotto il 25% di righe contaminate, Random Forest, Decision Tree e rete neurale perdono meno di un punto di F1** in ogni configurazione (il Random Forest regge fino al 35%). Per la Logistic Regression non esiste soglia: già al 5% perde più di un punto. La soglia "media" è fuorviante, perché è trascinata dal solo modello fragile.

### 7.7 Una nota sui test statistici

Il t-test aggregato sui 4 modelli non risulta significativo in nessun confronto fra configurazioni (0 su 24; per C1 → C2 i p-value stanno fra 0,066 e 0,086). **Non va letto come assenza di effetto**: aggrega un modello con un effetto enorme e costante (Logistic Regression, p < 0,001 a ogni livello) e tre con effetto quasi nullo, e questa eterogeneità gonfia la varianza delle differenze. Con K = 3 i test per modello hanno inoltre solo 2 gradi di libertà: la potenza statistica è bassa, e ciò che si osserva di significativo va inteso come un effetto grande, non come l'unico effetto presente.

## 8. Conclusioni

1. **Addestrare su dati contaminati e prevedere su dati corretti costa poco ai modelli robusti.** Con il 40% delle righe di training sporcate su 18 colonne, Random Forest, Decision Tree e rete neurale perdono meno di 2 punti di F1 su un baseline di 97. Con decine di migliaia di righe, la maggioranza pulita basta a questi modelli per imparare la relazione corretta.

2. **La vulnerabilità dipende dal modello.** La Logistic Regression perde fino a 25 punti: l'effetto dei dati sporchi non è una proprietà dei dati soltanto, ma dell'interazione fra dati e modello.

3. **Il danno dipende da quale informazione viene distrutta, non da quante dipendenze funzionali vengono violate.** L'ipotesi della roadmap (§4.4) — *più FD violate → più IM/IP/IH → più degrado* — è confermata solo a metà: aggiungere FD fa crescere sempre le misure di inconsistenza, ma fa crescere il degrado solo se l'informazione colpita è predittiva per l'obiettivo. Da C2 a C4 IM cresce di quasi 7 volte e F1 resta ferma.

4. **Fra le misure di inconsistenza, IH è la più informativa.** IM è dominato dalla dimensione dei gruppi delle FD e IP si satura presto; IH conta le righe da correggere e segue da vicino il degrado, fino a una correlazione di −0,994 con la F1 della Logistic Regression.

## 9. Limiti

- **Un solo mese di dati** (gennaio 2025): il CSV originale non copre altro periodo.
- **Campione del 45%** e **K = 3**: 3 stime per cella, test per modello con 2 gradi di libertà, nessuna ripetizione con seed diversi.
- **Rumore casuale uniforme**: i valori sporchi sono estratti a caso fra quelli possibili; errori reali tendono a essere sistematici (refusi, codici scambiati, valori vicini al vero), e potrebbero produrre effetti diversi.
- **IM misura solo le FD dichiarate**: le violazioni introdotte sporcando le colonne ridondanti (per esempio `Dest → DestCityName`) non vengono contate.
- **Asimmetria del disegno**: la geografia di arrivo è sporcata da C1, quella di partenza solo da C4.
- **Il meccanismo proposto per la Logistic Regression** (§7.3) è un'interpretazione coerente con i dati, non verificata ispezionando i coefficienti del modello.
- **Modelli con iperparametri di default**, come richiesto: un modello ottimizzato potrebbe reagire diversamente.
- **IH non è confrontabile** con i valori dei progetti precedenti, perché calcolato con un metodo diverso.

---

## In parole semplici

**La domanda.** Se un programma di intelligenza artificiale impara da dati pieni di errori, ma poi viene usato su dati corretti, quanto peggiora?

**Come si è provato.** Si sono presi circa 67.000 voli, con la stessa quantità di voli per ciascuna delle 5 fasce di durata. Si divide ogni volta in una parte per imparare e una per la prova. Nella parte per imparare si rovinano di proposito alcune informazioni — la rotta, la distanza, l'orario, la zona geografica — su una quota crescente di voli, fino al 40%. La parte per la prova resta sempre corretta, e il programma ha controllato a ogni passaggio che fosse davvero così.

**Cosa si è scoperto.**

- Tre programmi su quattro se la cavano benissimo: anche con 4 voli su 10 rovinati, sbagliano appena di più (meno di 2 punti su 97). Avendo tanti esempi, la parte corretta basta per imparare bene.
- Il quarto programma, il più semplice (la regressione logistica), crolla: perde fino a 25 punti. Il motivo è che si appoggia quasi tutto sulla distanza del volo; quando si rovina sia la distanza sia la "fascia di distanza", non ha più su cosa contare.
- Rovinare un'informazione inutile non fa danno. Rovinare l'orario di partenza, che non c'entra con la durata del volo, non cambia nulla, anche se il numero di contraddizioni nei dati raddoppia.

**Cosa vuol dire.** Non basta contare quanti errori ci sono nei dati per sapere quanto faranno male: conta *quali* informazioni sono rovinate e *quale* programma le usa. Fra i modi di misurare la sporcizia, il più utile è il numero di righe da correggere, che segue da vicino il danno reale. Il numero di coppie di righe in contraddizione, invece, può esplodere anche quando il danno è nullo.

## 10. File prodotti

| File | Contenuto |
|---|---|
| `blocco2_verifica_fd.csv` | Certificazione: 0 violazioni per ogni FD usata |
| `dataset_pulito_bilanciato.csv.gz` | Il dataset pulito, bilanciato, con i fold |
| `blocco2_risultati_raw.csv` | 432 righe: una per configurazione × livello × fold × modello |
| `blocco2_aggregato.csv` | Medie e deviazioni standard sui fold |
| `blocco2_correlazioni.csv` | Pearson e Spearman fra IM/IP/IH e F1, per modello e configurazione |
| `blocco2_test_configurazioni.csv` | t-test appaiati fra configurazioni adiacenti |
| `blocco2_soglie.csv` | Soglie di contaminazione tollerabile |
| `plot_f1_per_configurazione.png` | Grafico principale: curva di degrado per configurazione |
| `plot_f1_per_modello.png` | Curve per modello, con variabilità fra i fold |
| `plot_inconsistenza.png` | IM e IH in funzione del rumore |
| `plot_correlazioni.png` | Correlazioni IM/IH ↔ F1 per modello |
| `tabella_riassuntiva.csv` / `.png` | F1 e IM per livello e configurazione |
| `esperimento.log` | Registro dell'esecuzione |
