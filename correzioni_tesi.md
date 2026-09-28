# Correzioni alla tesi dopo le osservazioni del relatore

Documento di lavoro per aggiornare la tesi (`Template_Tesi_DIMES_Unical (2).pdf`, 48 pagine). Le correzioni sono ordinate per osservazione del relatore; seguono l'elenco dei refusi e dei rinvii errati trovati con una rilettura completa, e le sezioni i cui numeri cambiano dopo la riesecuzione.

**Come usarlo.** Ogni voce indica la sezione, il testo attuale (come compare nel PDF) e il testo da mettere al suo posto. I numeri nuovi vanno presi **solo** da `numeri_tesi.md`, generato dai CSV: dove una voce dice *[numeri_tesi]*, il valore esatto è in quel file.

---

## Osservazione 1 — Le misure sono variabili osservate, non manipolate

Il relatore osserva che l'unica variabile controllata è la percentuale di righe alterate: IM, IP e IH sono misurate dopo l'iniezione del rumore e sono quindi variabili dipendenti. E che quando IM cala il database è davvero meno inconsistente rispetto al vincolo, perché gruppi più piccoli generano meno coppie in conflitto: la misura funziona correttamente, e non ha mai avuto il compito di predire il degrado di un modello.

**Introduzione, terzo obiettivo.**
- *Attuale:* "Valutare il regime di validità delle misure di inconsistenza usate per quantificare la corruzione dei dati, verificando se continuino a riflettere in modo fedele il danno osservato anche quando il numero di vincoli violati cresce."
- *Nuovo:* "Osservare come si comportano le tre misure di inconsistenza al crescere della corruzione e del numero di vincoli violati, e in che rapporto stanno con il degrado predittivo osservato: una misura di inconsistenza quantifica le violazioni di un vincolo, non il danno subito da un modello."

**§2.3, in apertura del disegno sperimentale — aggiungere:**
> "L'unica variabile controllata dall'esperimento è la percentuale di righe alterate, fissata a priori su sei livelli. Le misure IM, IP e IH sono calcolate dopo l'iniezione del rumore: sono quindi variabili osservate, al pari della F1 dei modelli. Il rapporto fra le due si studia confrontandone l'andamento al variare del rumore, non facendo variare l'inconsistenza."

**§3.7, dove si descrive il ciclo di ogni esecuzione.**
- *Attuale:* "inietta il rumore, misura IM, IP e IH e passa il campione pulito nella preparazione dei modelli."
- *Nuovo:* "inietta il rumore (la variabile controllata), misura IM, IP e IH sulle righe di training di ciascun fold (variabili osservate) e passa il campione pulito nella preparazione dei modelli."

**§4.6, frase conclusiva sul Blocco 1.**
- *Attuale:* "la qualità dei dati in input incide sulla qualità delle predizioni in modo reale, statisticamente significativo e ben tracciato dalle misure di inconsistenza."
- *Nuovo:* "la qualità dei dati in input incide sulla qualità delle predizioni in modo reale e statisticamente significativo; al crescere del rumore le misure di inconsistenza crescono insieme al degrado."

**§5.6 — da riscrivere per intero.** Titolo nuovo: "5.6 Il comportamento delle misure di inconsistenza". Contenuto (numeri *[numeri_tesi]*, sezione "Comportamento delle misure"):
1. Con 4 FD, IM cresce rapidamente fino al 20% e poi rallenta; IP raggiunge il massimo possibile (tutte le righe di training) già a basso rumore; IH [andamento dai numeri nuovi].
2. Il meccanismo, misurato: corrompendo il lato sinistro i gruppi più grandi si svuotano, e le coppie di righe che condividono il lato sinistro — il massimo numero di conflitti possibili — diminuiscono. Il database, rispetto al vincolo, **è** meno inconsistente di quanto sarebbe a gruppi invariati: IM sta misurando correttamente.
3. IP si comporta diversamente da IM: la frammentazione riduce le coppie in conflitto ma non il numero di tuple coinvolte [confermare con la colonna `Tuple_coinvolte_FD`], come ipotizzato dal relatore.
4. Chiusura, al posto di "IM smette di riflettere fedelmente il danno reale": "Nessuna delle tre misure è costruita per predire il degrado di un modello: quantificano le violazioni di un vincolo. Il fatto che il danno continui a crescere mentre IM rallenta indica che, a parità di violazioni misurate, il degrado dipende da quale informazione è stata alterata e da quanto il modello ne dipende."

**§5.7, secondo punto.**
- *Attuale:* "Le misure di inconsistenza basate sul conteggio dei conflitti hanno regime di validità limitato [...]. Oltre una certa soglia di corruzione, IM smette di crescere in proporzione al danno reale, proprio nella regione in cui il danno è più severo."
- *Nuovo:* "Le tre misure seguono il rumore in modo diverso. IM conta le coppie in conflitto e dipende dalla dimensione dei gruppi del lato sinistro, che la corruzione riduce: oltre una certa soglia cresce poco, perché ci sono meno coppie che possono entrare in conflitto. IP e IH contano tuple e ne risentono in modo diverso. Nessuna delle tre è un predittore del degrado: descrivono l'inconsistenza del dato rispetto al vincolo."

**Conclusioni, punto 3 della risposta alla domanda di ricerca — da riscrivere.**
- *Attuale (inizio):* "Le misure di inconsistenza basate sul conteggio dei conflitti hanno un regime di validità circoscritto. [...] IM, che conta queste coppie, può quindi smettere di crescere in proporzione al danno reale, o persino invertire la rotta [...] dove il danno reale è più severo e la misura servirebbe di più."
- *Nuovo:* "Le misure di inconsistenza misurano ciò per cui sono definite: le violazioni di un vincolo, non il danno di un modello. Al crescere del rumore crescono insieme al degrado, ma con andamenti diversi fra loro: IM, che conta coppie, rallenta quando la corruzione del lato sinistro svuota i gruppi più grandi — il dato è davvero meno inconsistente rispetto al vincolo; IP [e IH] contano tuple e [comportamento dai numeri nuovi]. A parità di misura, il danno dipende da quale informazione è inconsistente e da quanto il modello ne dipende."

**Limiti — voce sul pool di quattro FD.** Togliere "meccanismo di saturazione di IM, osservato con una dipendenza su quattro che si appiattisce e inverte"; sostituire con "comportamento delle tre misure con un numero maggiore di dipendenze violate simultaneamente".

**Sviluppi futuri — terza voce.** "verificare se il meccanismo di saturazione di IM osservato si accentui" → "verificare come si comportano IM, IP e IH con un numero maggiore di dipendenze violate".

---

## Osservazione 2 — Su quali righe sono calcolate le misure

Il difetto era reale: le misure erano calcolate sulle 30.000 righe del campione sporcato, mentre i modelli si addestrano su circa 6.470 righe per fold. **Nel codice ora sono calcolate sulle righe di training di ciascun fold** (6.472 righe), e se ne riportano media e deviazione standard sui 5 fold.

**§3.4 — aggiungere in fondo:**
> "Le misure non sono calcolate sull'intero campione, ma sulle righe di training di ciascun fold, cioè sulle stesse righe su cui i modelli vengono addestrati (6.472 righe per fold). La funzione `fold_di_valutazione` restituisce esattamente la partizione usata dalla valutazione, e `indici_per_fold` calcola IM, IP e IH su ciascun fold; i risultati riportano media e deviazione standard sui cinque fold. In questo modo indici e metriche descrivono la stessa popolazione."

**§3.4, ultima frase.**
- *Attuale:* "Il costo resta comunque legato alla dimensione del grafo: ogni arco occupa circa 160 byte, ed è questo vincolo a limitare il campione a 30.000 righe, come discusso nel paragrafo 3.2."
- *Nuovo:* "Il costo resta legato alla dimensione del grafo, che cresce circa con il quadrato delle righe: calcolare le misure sui fold di training (circa 6.500 righe) invece che sull'intero campione riduce il grafo di oltre un ordine di grandezza."

**§3.2, ultimo capoverso (limite dei 30.000).**
- *Attuale:* "È un limite di memoria del grafo, non un vincolo sui tempi di calcolo, a fissare la soglia delle 30.000 righe." (con il calcolo dei 5,9 milioni di archi)
- *Nuovo:* "La dimensione del campione è stata fissata in una prima versione del progetto, quando le misure erano calcolate sull'intero campione e il grafo dei conflitti ne limitava la dimensione. Ora che le misure sono calcolate sui fold di training quel vincolo si allenta; la soglia è stata mantenuta per confrontabilità con i risultati precedenti e per il costo dei 480 addestramenti (4 modelli × 5 fold × 120 esecuzioni)." Togliere il rinvio "con lo stesso calcolo già anticipato nel Capitolo 2", che non ha riscontro nel Capitolo 2.

**§2.1 — aggiungere dopo "ogni fold di training contiene circa 6.470 righe":** "Su queste righe sono calcolate anche le misure di inconsistenza (Paragrafo 3.4)."

**Figure e tabelle con IM, IP, IH (Figura 4.2, Figure 5.1, 5.2, 5.6, 5.7).** Didascalie: aggiungere "misurati sulle righe di training di ciascun fold, media sulle repliche". I valori cambiano di ordine di grandezza (per esempio, con 1 FD al 40% IM passa da migliaia a circa 150): vanno rigenerati da `genera_plot_finali.py`.

**Limiti — voce "La scala del campione è vincolata dalla misura stessa".** Da riscrivere: "La scala del campione. Il campione di 30.000 righe (Paragrafo 3.2) è stato mantenuto per confrontabilità; con le misure calcolate sui fold di training il limite di memoria si allenta, e resterebbe da verificare se i risultati si mantengano su campioni più grandi, dove il costo principale diventano gli addestramenti."

---

## Osservazione 3 — IH è un limite superiore

IH è calcolata con `min_weighted_vertex_cover` di NetworkX, un algoritmo 2-approssimato: il valore restituito è una copertura valida, quindi un **limite superiore** del minimo, al più doppio. Nel progetto ora ci sono due verifiche.

- **Con una sola FD il minimo esatto ha forma chiusa.** Il grafo dei conflitti di una FD è l'unione disgiunta, sui gruppi del lato sinistro, di grafi multipartiti completi (le parti sono i valori del lato destro). Il massimo insieme indipendente di un multipartito completo è la parte più numerosa, quindi per ogni gruppo il vertex cover minimo vale (righe del gruppo − righe della classe più numerosa), e IH esatto è la somma sui gruppi. È implementato in `ih_esatto_una_fd` e riportato accanto al valore approssimato in tutto il Blocco 1.
- **Con più FD** i grafi si sovrappongono e il problema resta NP-difficile: `verifica_ih.py` confronta il valore approssimato con l'ottimo calcolato con un modello di programmazione lineare intera (`scipy.optimize.milp`) su istanze da 250 a 2.000 righe, per 1, 2 e 4 FD e cinque livelli di rumore.

**Risultati della verifica** (`verifica_ih.csv`, 59 istanze su 60 risolte all'ottimo; una ha superato il limite di tempo):
- con 1 FD la formula chiusa coincide con l'ottimo dell'ILP in tutte le istanze;
- rapporto approssimato/esatto da **1,000 a 1,473**, entro la garanzia teorica di 2;
- lo scarto cresce con il numero di FD: da 1,00 a 1,08 con una FD, da 1,04 a 1,22 con due, da 1,31 a 1,47 con quattro;
- negli esperimenti del Blocco 1 (una FD, fold di training di 6.472 righe) l'approssimato supera l'esatto del 12–44% (rapporto da 1,12 a 1,44);
- scarto assoluto da 0 a 443 tuple.

**Verifica sulle istanze reali del Blocco 2** (`ih_esatto_blocco2.py`, `blocco2_ih_esatto.csv`): sulle righe di training di ciascun fold (6.472), per tutte le 5 repliche × 5 fold, il minimo esatto è calcolato con la formula chiusa (1 FD) o con l'ILP (2 e 4 FD, limite di 150 secondi per istanza).

| Configurazione | Livelli risolti all'ottimo | Rapporto approssimato / esatto |
|---|---|---|
| 1 FD | tutti (formula chiusa) | da 1,63 (5%) a 1,45 (40%) |
| 2 FD | 5, 10, 20% (25/25); 40% solo 8/25; 30% non risolto | da 1,69 (5%) a 1,54 (20%) |
| 4 FD | 5 e 10% (25/25); dal 20% non risolto | 1,70 (5%) e 1,60 (10%) |

**Questo cambia il testo da scrivere:** sulle istanze reali il 2-approssimato supera il minimo del **45–70%**, molto più che sulle istanze ridotte (al più 47%). La verifica su istanze piccole sottostimava l'errore, e va detto esplicitamente. Frase proposta per §3.4: "Sulle istanze degli esperimenti il valore restituito dall'algoritmo 2-approssimato supera il minimo esatto del 45–70%, pur restando sotto il limite teorico del doppio; lo scarto diminuisce al crescere del rumore. Il minimo esatto è stato calcolato con la formula chiusa per una sola FD e con la programmazione lineare intera per 2 e 4 FD, fin dove il risolutore chiude in tempi ragionevoli (20% di rumore con 2 FD, 10% con 4 FD). Le due grandezze hanno lo stesso andamento, per cui le conclusioni qualitative su IH non cambiano, ma il valore assoluto riportato va letto come un limite superiore largo." Le figure 5.1 e 5.6 riportano il minimo esatto come linea tratteggiata.

**Modifiche alla tesi**
- **§1.2.3 — aggiungere in fondo:** "Nel seguito IH è calcolata con un algoritmo 2-approssimato [Bar-Yehuda ed Even]: il valore riportato è quindi un limite superiore del minimo, che nel caso peggiore ne vale il doppio. Lo scarto effettivamente osservato è misurato nel Paragrafo 3.4."
- **§3.1, voce networkx:** "calcolo approssimato del minimum vertex cover" → "calcolo 2-approssimato del minimum vertex cover (limite superiore di IH)".
- **§3.4, voce IH:** "IH è la dimensione della copertura approssimata dei vertici" → "IH_approx è la dimensione della copertura restituita da `min_weighted_vertex_cover`, un limite superiore del minimo al più doppio. Con una sola FD si calcola anche il minimo esatto in forma chiusa [spiegazione sopra]." Aggiungere un paragrafo con la tabella degli scarti (da `numeri_tesi.md`, sezione "IH: approssimato contro esatto").
- **Figure e tabelle:** rinominare la grandezza in "IH (2-approssimato)", e dove c'è anche il valore esatto riportare entrambi.

---

## Osservazione 4 — La validazione di Torchia riguardava i tempi

**§1.3, secondo capoverso.**
- *Attuale:* "Si è inoltre affrontato direttamente il problema della complessità computazionale del calcolo esatto del minimum vertex cover [...], proponendo e validando in modo sperimentale un algoritmo di calcolo 2-approssimato. I risultati ottenuti sono rilevanti anche per la presente tesi: i tempi di esecuzione dell'algoritmo 2-approssimato sono risultati sostanzialmente costanti [...], a differenza dell'algoritmo esatto, la cui crescita è risultata quasi esponenziale [4]."
- *Nuovo:* "Si è inoltre affrontato il problema della complessità computazionale del calcolo esatto del minimum vertex cover (Paragrafo 1.2.3), confrontando sperimentalmente i tempi di un algoritmo 2-approssimato con quelli dell'algoritmo esatto: i primi sono risultati sostanzialmente costanti al crescere dell'inconsistenza, i secondi in crescita quasi esponenziale [4]. Quel confronto riguarda i tempi di calcolo, non la qualità dell'approssimazione: la distanza fra il valore approssimato e il minimo è misurata direttamente in questa tesi (Paragrafo 3.4)."

Nessun altro punto della tesi deve usare [4] a sostegno della qualità dell'approssimazione.

---

## Osservazione 5 — Il baseline non è una costante

A rumore 0% le cinque repliche erano identiche, perché bilanciamento e fold avevano seed fisso: il baseline aveva varianza nulla per costruzione. **Ora il seed di ogni replica governa anche il sottocampionamento di bilanciamento e la partizione in fold**, quindi ogni replica lavora su righe e fold propri e il baseline ha una sua varianza; il confronto è un **t-test di Welch a due campioni** (5 repliche del livello contro 5 del baseline, varianze non assunte uguali).

**Modifiche alla tesi**
- **§2.1, ultimo capoverso** ("con seed fisso, di ottenere partizioni di training e di test i cui indici di riga coincidono"): "con lo stesso seed" al posto di "con seed fisso", e aggiungere: "Il seed è quello della replica: repliche diverse lavorano su righe bilanciate e fold diversi, e anche il risultato a rumore 0% varia da una replica all'altra."
- **§2.3, "il bilanciamento delle classi opera sul target (mai corrotto) con seed fisso":** → "con il seed della replica, lo stesso per la versione pulita e per quella sporcata".
- **§2.4:** aggiungere che le cinque repliche differiscono per rumore iniettato, righe bilanciate e fold.
- **§3.1, voce scipy.stats:** "t-test a un campione" → "t-test di Welch a due campioni (confronto con il baseline) e t-test appaiato (confronto fra configurazioni)".
- **§3.6:** "StratifiedFold(n_splits=5, shuffle=True, random_state=42)" → "StratifiedKFold(n_splits=5, shuffle=True, random_state=seed della replica)" (anche il nome della classe è errato nel testo attuale).
- **§3.8, terza voce:** "un test t a un campione [...] contro il valore di baseline, con gestione esplicita del caso di varianza nulla fra le repliche" → "un t-test di Welch a due campioni, per modello e livello di rumore, fra le cinque repliche del livello e le cinque del baseline: il baseline è una stima, con la sua varianza, e trattarlo come costante sottostimerebbe la varianza complessiva e sopravvaluterebbe la significatività."
- **§4.2:** "test t a un campione contro il baseline, p < 0,05 ovunque" → risultati del test di Welch *[numeri_tesi]*.
- **Conclusioni, punto 1:** "statisticamente significativo in tutte le 20 combinazioni [...] e in tutte le 60 del Blocco 2" → conteggi del test di Welch *[numeri_tesi]*.

---

## Osservazione 6 — IP e IH nei risultati

Ora analisi, tabelle e grafici riportano tutte e tre le misure, e lo studio del meccanismo misura anche le tuple coinvolte per FD.

**Modifiche alla tesi**
- **Figura 4.2 (tabella del Blocco 1):** aggiungere le colonne IP e IH esatto accanto a IM e IH 2-approssimato. Commentarla nel testo di §4.2: con una FD tutte e tre crescono a ogni livello insieme al degrado; IH esatto e approssimato differiscono di [rapporto da numeri_tesi].
- **Figura 5.1:** ora ha un pannello per il danno e uno per ciascuna misura; didascalia "Effetto sul modello e indici osservati (IM, IP, IH), per numero di FD corrotte".
- **Figura 5.2 (tabella):** aggiunge le colonne IP e IH per 1, 2 e 4 FD.
- **Figura 5.6:** diventa `plot_blocco2_indici_e_f1.png`, con danno, IM, IP e IH; commentare tutte e tre le misure.
- **Figura 5.7 e §5.6:** oltre ai conflitti per FD, riportare le tuple coinvolte per FD e verificare l'ipotesi del relatore (IP non si satura come IM).
- **Conclusioni, punto 3:** parlare di tutte e tre le misure (testo nell'osservazione 1).

---

## Osservazione 7 — Bibliografia

Il file `bibliografia.bib` contiene 24 voci verificate, raggruppate per tema. Dove citarle:

| Dove | Cosa citare |
|---|---|
| §2.1, nome del dataset | `bts2025ontime` (il dataset oggi non compare in bibliografia) |
| §3.1, elenco delle librerie | `mckinney2010pandas`, `harris2020numpy`, `hagberg2008networkx`, `pedregosa2011scikit`, `virtanen2020scipy`, `hunter2007matplotlib` |
| §1.2 e §1.2.2, framework e misure | `parisi2019inconsistency`, `parisi2023denial` (già [1]), `parisi2021properties`, `grant2020general` |
| §1.2.1, dipendenze funzionali | `fan2012foundations`, `abedjan2015profiling`, `kivinen1995approximate` |
| §2.2 e §3.3, scoperta delle FD | `papenbrock2015fd`, `abedjan2015profiling` |
| §1.1, Data-Centric AI | `zha2023datacentric` (già [2]), `sambasivan2021everyone`, `whang2023datacollection` |
| §1.1 o §2.3, effetto degli errori sull'apprendimento | `li2021cleanml`, `krishnan2016activeclean`, `frenay2014labelnoise`, `northcutt2021pervasive` |
| §1.2.3, vertex cover e approssimazione | `garey1979computers`, `baryehuda1981linear` |
| Limiti, "L'inconsistenza è simulata" | `ilyas2019datacleaning`, `li2021cleanml` (errori reali contro errori iniettati) |

CleanML in particolare è il lavoro più vicino a questa tesi (effetto della pulizia dei dati sui modelli, con test statistici): va citato e discusso brevemente nello stato dell'arte, con la differenza principale — CleanML usa errori reali, questa tesi errori iniettati in modo controllato.

---

## Osservazione 8 — Refusi, rinvii errati, numeri discordanti

Rilettura completa del PDF. Voci ordinate per sezione.

### Rinvii e riferimenti incrociati errati

| Sezione | Problema | Correzione |
|---|---|---|
| §2.2.1, didascalia Figura 2.1 | "Le **dieci** dipendenze funzionali usate nel Blocco 2" | "Le **quattro**" |
| §2.2.1 e §2.2.3 | il testo rimanda a "Tabella 2.1" ma l'elemento è etichettato "Figura 2.1" | uniformare: se è una tabella, usare l'ambiente `table` e "Tabella 2.1" ovunque |
| §2.2.1 | "insieme a **16** colonne ridondanti" | "17" (come in §3.5 e §3.7; la diciassettesima è `DistanceGroup`) |
| §2.3, ultimo capoverso | "Stesso principio che nel **Capitolo 1** (Paragrafo 2.2)" | "nel Capitolo 2 (Paragrafo 2.2)" |
| §3.2 | "con lo stesso calcolo già anticipato nel Capitolo 2" | il Capitolo 2 non contiene quel calcolo: togliere il rinvio |
| Introduzione al Capitolo 3 | l'elenco dei paragrafi si ferma a 3.8 | aggiungere "la verifica e validazione dei meccanismi (Paragrafo 3.9)" |
| §3.8 | "l'analisi produce **quattro** outputs" seguito da tre voci | "tre output" (le correlazioni sono state tolte) |
| §3.9, seconda voce | "segue la formula di unione del paragrafo 2.3" | la formula compare in §5.3, non in §2.3: rinviare a §5.3 o riportarla in §2.3 |
| §4.1 | "un'accuracy F1 compresa tra..." | "una F1 compresa tra..." (accuracy e F1 sono metriche diverse) |
| §4.1 | "(ANN)" | "(Neural Network)", come nel resto della tesi |
| §5.1 | l'elenco delle FD mette per prima `Origin+Dest → Distance` | l'ordine del codice (e quindi delle configurazioni) è: 1) `Distance → DistanceGroup`, 2) `Origin+Dest → Distance`, 3) `OriginAirportID → Origin`, 4) `DestAirportID → Dest`. Con "1 FD" si intende la prima di questo elenco |
| §5.6 | "Le **altre due** FD ad alta cardinalità del lato sinistro" | "le due FD sugli aeroporti" (le FD sono quattro, e il testo ne ha nominata una) |
| Conclusioni, "Sguardo d'insieme" | "per le configurazioni del **Blocco 1** a 1 e 2 FD è sostanzialmente più alta" | "del **Blocco 2** a 1 e 2 FD" |
| Conclusioni, sintesi | "rispetto a un **indice** di dipendenze funzionali violate" | "rispetto a un **insieme** di dipendenze funzionali" |

### Criterio di rilevanza delle FD (§2.2.3) — da riformulare dopo il test di Welch

Con il test di Welch l'analisi di rilevanza (`rilevanza_fd.csv`) conferma la classifica, ma il concetto "aeroporto di origine" ha un calo significativo per **3 modelli su 4**, non per tutti e quattro come richiede il criterio scritto in tesi. Si mantengono le 4 FD e si riformula il criterio.

- *Attuale:* "fra le FD valide superstiti, si sporca ciascuna candidata al 40% da sola e si verifica se il calo di F1 sul test pulito è significativo per tutti e quattro i modelli. Le FD scartate in questo passaggio hanno un effetto nullo o non significativo e vengono escluse; le quattro che superano entrambi i filtri sono quelle di Tabella 2.1."
- *Nuovo:* "fra le FD valide superstiti, si sporca ciascuna candidata al 40% da sola (3 repliche) e se ne misura il calo di F1 sul test pulito, con un t-test di Welch contro il baseline per ciascun modello. È considerata rilevante una FD con calo medio di almeno un punto di F1 e significativo per la maggioranza dei modelli. Il criterio separa due gruppi netti: le quattro FD selezionate hanno cali fra 1,1 e 8,0 punti, significativi per 3 o 4 modelli su 4; tutte le altre hanno cali di al più 0,3 punti e non sono significative per nessun modello. Il caso limite è l'aeroporto di origine: calo di 1,2 punti, significativo per 3 modelli su 4."

Valori esatti in `numeri_tesi.md`, sezione "Rilevanza delle FD".

### Contenuti superati dal codice attuale

| Sezione | Problema | Correzione |
|---|---|---|
| §3.5, secondo capoverso | "Il Blocco 2 invece non forza alcuna colonna ridondante `redundant_cols_map=None` [...] le colonne che in un esperimento a una sola FD sarebbero rimaste ridondanti e pulite diventano esse stesse il lato destro di una delle FD aggiuntive" | descrive il vecchio Blocco 2 a 10 FD. Ora: "Anche il Blocco 2 usa colonne ridondanti: la FD sulla rotta porta con sé `DistanceGroup`, le FD sugli aeroporti portano città, stato, FIPS, nome dello stato e WAC. Ogni FD sporca le proprie colonne e le proprie copie sulle righe che sceglie." |
| §3.6 | `StratifiedFold(...random_state=42)` | vedi osservazione 5 |
| §5.6 | "Corrompere il lato sinistro frammenta i gruppi in sottogruppi più piccoli e numerosi" | per le FD sugli aeroporti i gruppi restano lo stesso numero (322 e 315): si **svuotano i più grandi**, non si frammentano. Si frammentano solo i gruppi della rotta |

### Numeri incoerenti fra capitoli (anche prima della riesecuzione)

| Dove | Problema |
|---|---|
| §4.2 | "crolla di 22,9 punti al 40%, **sei volte e mezzo** il calo medio degli altri tre modelli (3,6 punti)": 22,9 / 3,6 = 6,4, non 6,5 |
| §4.5, §4.6, Conclusioni | lo stesso rapporto è scritto "6,4", "fattore sei", "sei volte più F1": usarne uno solo |
| §4.3 contro Conclusioni | quota di apprendimento "fra il 27% e il 32%" in §4.3 e "tra il 26,8% e il 32,2%" nelle Conclusioni: stesso dato, arrotondamenti diversi |
| Conclusioni, punto 2 | "Nel Blocco 1, al 40% **40** di rumore" | numero di pagina finito nel testo: "al 40% di rumore" |

Tutti i valori numerici dei Capitoli 4 e 5 e delle Conclusioni cambiano con la riesecuzione: vanno ripresi da `numeri_tesi.md` (vedi la sezione finale).

### Refusi

| Sezione | Attuale | Corretto |
|---|---|---|
| §1.1.2 | miglioamento | miglioramento |
| §1.2.2 | "IM e **IM** si ottengono contando i conflitti e le tuple coinvolte" | "IM e **IP**" |
| §1.2.3, §3.1 | calcoolo | calcolo |
| Introduzione al Capitolo 2 | "è disponibili" | "è disponibile" |
| §2.1 | "dal dataset pubblic On-Time Reporting" | "pubblico" (e citare il dataset) |
| §2.2, criterio 2 | parentesi aperta "(in particolare, del fatto che..." mai chiusa | chiudere la parentesi |
| §2.2, criterio 3 | "sopravvivenza" minuscolo a inizio voce | "Sopravvivenza" |
| §2.2.3 | quell edi | quelle di |
| §2.3 | contenevanofeature | contenevano feature |
| §2.3 | ciasun risltato | ciascun risultato |
| §2.3 | eslicitamente | esplicitamente |
| §2.3 | nessna | nessuna |
| §2.3 | "il secondo controllo la intercetterebbe" | la frase non dice qual è il secondo controllo: "il secondo verifica che le etichette coincidano riga per riga; se una delle due condizioni cadesse, l'esecuzione si fermerebbe invece di..." |
| §2.4 | valutazoine | valutazione |
| §3.1 | esecuzine | esecuzione |
| §3.2 | esecuzine | esecuzione |
| §3.3 | risulato | risultato |
| §3.4 | esplicitamemnte | esplicitamente |
| §3.4 | correttamene | correttamente |
| §3.5 | "tanto informativo **quando** il lato destro" | "quanto" |
| §3.5 | corrzione | correzione |
| §3.5 | clonne | colonne |
| §3.6 | rimozine | rimozione |
| §3.7 | Entrmabi | Entrambi |
| §3.7 | "il propri seed" | "il proprio seed" |
| §3.8 | silenziosamento | silenziosamente |
| §3.9 | posibbile | possibile |
| Introduzione al Capitolo 4 | riportto | riportato |
| §4.2 | "si tacca" | "si stacca" |
| §4.2 | "nessuno dei due comportamento" | "nessuno dei due comportamenti" |
| §4.4 | illegibile | illeggibile |
| §4.4 | "Gli altri tre modello" | "modelli" |
| §4.4 | condati | con dati |
| §4.5 | "di questo blocc:" | "di questo blocco:" |
| §4.5 | sviilupp o | sviluppo |
| §5.3 | ciasuna | ciascuna |
| §5.3 | veriicato | verificato |
| §5.7 | "due precisazione" | "due precisazioni" |
| Conclusioni | esplicitamento | esplicitamente |
| Conclusioni, punto 3 | "la dimensione dei gruppo" | "dei gruppi" |
| Conclusioni, punto 3 | qando | quando |
| Conclusioni | aggiune | aggiunge |
| Conclusioni | specfico | specifico |
| Conclusioni | "tesi:la" | "tesi: la" |
| Conclusioni | dipenenza | dipendenza |
| Conclusioni | artiolato | articolato |
| Conclusioni | "La quota di pende" | "La quota dipende" |
| Limiti | "L'inconsistenza è stimata, non naturale" | "è simulata, non naturale" |
| Limiti | "un meccanismo di corruzioni uniforme" | "di corruzione uniforme" |
| Limiti | foni | fonti |
| Sviluppi futuri | verficare / vericate | verificare |
| Sviluppi futuri | meccanismmo | meccanismo |
| Sviluppi futuri | "dela tesi" | "della tesi" |

Nota: alcune spaziature strane nell'estratto del PDF ("mo dello", "p er") sono artefatti dell'estrazione del testo da pdflatex, non refusi. Le voci sopra sono quelle in cui la parola è effettivamente sbagliata.

---

## Numeri da aggiornare dopo la riesecuzione

Valori presi da `numeri_tesi.md`, generato dai CSV dopo la riesecuzione del 21–22 settembre (indici calcolati sui fold di training, baseline variabile, test di Welch). Tutte le figure e le tabelle vanno sostituite con le versioni rigenerate.

### Capitolo 2

| Dove | Attuale | Nuovo |
|---|---|---|
| §2.1 | "ogni fold di training contiene circa 6.470 righe" | 6.472 righe (valore esatto) |
| §2.2.3 / Tabella 2.1, rilevanza delle FD | valori della versione precedente | cali al 40%: `Distance → DistanceGroup` 8,0 punti (4/4 modelli), `Origin+Dest → Distance` 2,0 (4/4), aeroporto di origine 1,2 (3/4), aeroporto di destinazione 1,1 (4/4); FD escluse: al più 0,3 punti, 0/4 |

### Capitolo 3

| Dove | Attuale | Nuovo |
|---|---|---|
| §3.2 | "quasi 5,9 milioni di archi, circa 0,8 GB" | da togliere con il paragrafo sul limite di memoria (osservazione 2). Con gli indici calcolati sui fold, il grafo più grande (4 FD, 30% di rumore) ha circa 290.000 archi |

### Capitolo 4 — Blocco 1

| Dove | Attuale | Nuovo |
|---|---|---|
| §4.1, baseline | DT 0,9218, LR 0,9218, NN 0,9152, RF 0,9345; media 0,9233 | DT 0,9212, LR 0,9231, NN 0,9107, RF 0,9306; **media 0,9214**. Aggiungere che ora il baseline varia fra repliche (deviazione standard da 0,0013 a 0,0044) |
| §4.1 | "F1 compresa tra 0,9152 e 0,9345" | tra 0,9107 e 0,9306 |
| §4.2 | "un calo contenuto fra 2,9 e 4,9 punti" (RF, DT, NN) | fra **2,6 e 4,6 punti** (RF 2,6; DT 2,7; NN 4,6) |
| §4.2 | "crolla di 22,9 punti al 40%" | **23,4 punti** |
| §4.2 | "sei volte e mezzo il calo medio degli altri tre modelli (3,6 punti)" | **7,0 volte** il calo medio degli altri tre (**3,3 punti**) |
| §4.2 | "significativo in tutte e 20 le combinazioni modello × livello (test t a un campione [...] p < 0,05 ovunque, nella maggioranza dei casi p < 0,001)" | "significativo in **18 combinazioni su 20** (test di Welch a due campioni). Non lo sono Decision Tree e Random Forest al 5% di rumore (cali di 0,36 e 0,19 punti, p = 0,26): a rumore basso il loro degrado non si distingue dalla variabilità fra repliche" |
| §4.2, Figura 4.2 | IM e IH | tabella con IM, IP, IH 2-approssimato e IH esatto. Al 40%: IM 135, IP 218, IH approssimato 102, IH esatto 86 |
| §4.3 | "31,2 punti di F1 (da 0,9233 a 0,6117)" | **31,0 punti (da 0,9214 a 0,6115)** |
| §4.3 | "solo 8,4 punti sono dovuti a un apprendimento effettivamente peggiore" | **8,3 punti** |
| §4.3 | "fra il 27% e il 32%" | fra il **27% e il 33%** |
| §4.4, Figura 4.4 | quote per modello (LR oltre la metà, gli altri pochi punti) | LR **58–63%**; DT 8–10%; NN 14–16%; RF 5–10% |
| §4.5 | "un fattore 6,4 rispetto alla media degli altri tre modelli" | **fattore 7,0** |
| §4.6 | "raggiunge un fattore sei" | "raggiunge un fattore sette" |

### Capitolo 5 — Blocco 2

| Dove | Attuale | Nuovo |
|---|---|---|
| §5.2 | "da 0,8451 con 1 FD a 0,8238 con 2 FD a 0,7701 con 4" | da **0,8430** con 1 FD a **0,8099** con 2 FD a **0,7507** con 4 |
| §5.2 | "da 1 a 2 FD è significativo in 18 confronti su 20, da 2 a 4 FD in 19 su 20" | da 1 a 2 FD in **19 su 20**, da 2 a 4 FD in **18 su 20** (non significativi solo al 5% di rumore: NN per 1→2, DT e RF per 2→4) |
| §5.2 | "peggioramento medio aggiuntivo rispettivamente di 1,3 e 2,3 punti" | **2,2 e 2,6 punti** |
| §5.2, nuovo | — | calo del degrado significativo in **58 confronti su 60** (Welch): non lo sono DT e RF con 1 FD al 5% |
| §5.3 | "39,8% con 1 FD, 63,7% con 2 FD e 87,0% con 4 FD" | **39,7%**, **64,1%** e **87,2%** |
| §5.3 | "scarto massimo verificato di 0,0043" | **0,0040** |
| §5.4, Figura 5.4 | "Per Random Forest l'effetto non è distinguibile dal rumore statistico (p = 0,456)" | p = **0,26** su tutti i punti (p = 0,81 nell'intervallo di quota comune). DT e LR: più FD, meno danno a parità di righe sporche (p < 0,001); NN: più FD, più danno (p < 0,001) |
| §5.4 | "in quattro comparabili su cinque, la configurazione di più dipendenze funzionali ottiene un F1 uguale o migliore" | in **5 coppie comparabili su 5** la configurazione con più FD ha F1 media maggiore; per modello: 3/4, 2/4, 3/4, 4/4, 3/4 |
| §5.5 | "48–53% con 1 FD, 47–55% con 2 FD, 36–39% con 4 FD" | **51–53%** con 1 FD, **47–56%** con 2 FD, **36–40%** con 4 FD |
| §5.6 | "da 5,07 milioni al 20% a 5,92 milioni al 40%" | IM con 4 FD: **251.133** al 20%, **290.361** al 30%, **289.586** al 40%: cresce fino al 30% e poi resta sostanzialmente fermo (lieve calo) |
| §5.6, nuovo — IP | — | IP con 4 FD vale 6.467 già al 5% e **6.472, cioè tutte le righe di training, dal 10% in poi**: satura prima e in modo più netto di IM. Con 1 FD invece cresce a ogni livello (da 1.390 a 5.553, l'86% delle righe al 40%). Contrariamente all'ipotesi del relatore, con 4 FD IP satura, ma per un motivo diverso da IM: basta una riga corrotta in un gruppo grande per rendere "coinvolte" tutte le righe del gruppo, e con le FD sugli aeroporti già al 5% sono coinvolte 6.223 righe su 6.472 per la sola origine |
| §5.6, nuovo — IH | — | IH (2-approssimato) con 4 FD continua a crescere a ogni livello (da 4.960 al 20% a 6.122 al 40%), avvicinandosi al numero di righe: è l'ultima delle tre misure a saturare |
| §5.6, Figura 5.7 | conflitti per FD | FD sulla rotta: massimo al 20% (1.683 conflitti) e poi calo (745 al 40%), perché le rotte si frammentano (gruppo più grande da 38 a 8 righe). FD sugli aeroporti: conflitti quasi fermi fra 30% e 40% |
| §5.6 | "da 1.419 a 901 righe per il gruppo più grande di OriginAirportID → Origin" | sui fold di training: **da 276 a 177 righe** |

### Conclusioni

| Dove | Attuale | Nuovo |
|---|---|---|
| Punto 1 | "significativo in tutte le 20 combinazioni [...] del Blocco 1, e in tutte le 60 del Blocco 2" | in **18 su 20** nel Blocco 1 e **58 su 60** nel Blocco 2; non lo sono solo DT e RF al 5% di rumore con una FD |
| Punto 1 | "la Logistic Regression perde 22,9 punti [...] contro un calo medio di 3,6 punti degli altri tre modelli (un fattore 6,4)" | **23,4 punti** contro **3,3** (fattore **7,0**) |
| Punto 1 | "il 57–61% del danno osservato è un genuino apprendimento peggiore, contro il 7–17% degli altri tre" | **58–63%** contro **5–16%** |
| Punto 2 | "31,2 punti di F1 contro gli 8,4 [...] circa 3,7 volte" | **31,0 contro 8,3**, circa **3,7 volte** |
| Punto 2 | "tra il 26,8% e il 32,2%" | tra il **26,9% e il 32,7%** |
| Sguardo d'insieme | "fra il 27% e il 32%" (Blocco 1); "fra il 46% e il 55%" (Blocco 2, 1 e 2 FD); "fra il 36% e il 39%" (4 FD) | 27–33%; **47–56%**; **36–40%** |
| Risultati non attesi | "perdere sei volte più F1 degli altri tre modelli" | "circa sette volte" |

### Controllo finale

Dopo aver inserito i numeri, rigenerare il PDF ed eseguire:

```
python controllo_tesi.py "percorso/della/tesi.pdf"
```

Segnala i numeri decimali della tesi che non corrispondono a nessun valore delle analisi. Quelli che restano vanno verificati a mano: rapporti e medie calcolati nel testo, oppure valori da aggiornare.
