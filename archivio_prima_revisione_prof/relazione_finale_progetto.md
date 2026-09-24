# Relazione finale — Misure di inconsistenza per Data-Centric AI

Relazione scientifica del progetto di tesi: domanda di ricerca, disegno sperimentale, risultati e conclusioni.

Il dettaglio delle modifiche metodologiche è in [`relazione_revisione_sperimentale.md`](relazione_revisione_sperimentale.md).

---

## 1. Domanda di ricerca

> **Quanto incide sulla qualità delle predizioni di un modello AI la qualità (consistenza) dei dati in input?**

Nella forma resa operativa dopo il confronto con il relatore: *se addestro un modello su dati contaminati ma poi lo uso su dati corretti, quanta capacità predittiva perdo, e come cresce questa perdita al crescere dell'inconsistenza?*

**Vincoli metodologici** rispettati in tutto il lavoro:
- i modelli restano **RAW**: iperparametri di default, `random_state=42`, nessun `class_weight`, nessun tuning per condizione sperimentale;
- l'obiettivo predittivo è unico ed esplicito, e le dipendenze sporcate sono scelte **in base alla loro rilevanza per quell'obiettivo**;
- il training viene sporcato gradualmente, ma **il test è sempre su dati puliti**.

## 2. Framework di misura dell'inconsistenza

L'inconsistenza è definita rispetto a **dipendenze funzionali (FD)** della forma `LHS → RHS`: un insieme di colonne che determina univocamente un'altra colonna. Si costruisce un grafo dei conflitti — nodo = riga, arco = coppia di righe con stesso LHS ma RHS diverso — e se ne ricavano tre indici:

- **IM**: numero di conflitti a coppie (archi del grafo);
- **IP**: numero di tuple coinvolte in almeno un conflitto;
- **IH**: dimensione minima dell'insieme di tuple da correggere per eliminare tutte le violazioni (vertex cover minimo approssimato).

## 3. Dati e obiettivo predittivo

**Task**: classificazione a 5 classi della durata schedulata del volo (`DurataBucket`, da `CRSElapsedTime`: `<90m`, `90-150m`, `150-210m`, `210-300m`, `>300m`).

**Dati**: campione di **30.000 voli** dal dataset "On-Time Reporting Carrier On-Time Performance" (gennaio 2025). Le classi sono bilanciate per undersampling → **8.090 righe**, circa 6.470 di training per fold con cross-validation stratificata a **5 fold**. Il caso puro vale **0,20**.

**Baseline** a rumore 0%, su dati puliti (F1 weighted), comune ai due blocchi:

| Modello | F1 |
|---|---|
| Random Forest | 0,9345 |
| Decision Tree | 0,9218 |
| Logistic Regression | 0,9218 |
| Neural Network | 0,9152 |

Media: **0,9233**, nettamente sopra il caso puro — condizione necessaria perché un degrado sia interpretabile.

## 4. Quali dipendenze funzionali contano per l'obiettivo

Una FD valida non è necessariamente rilevante: se le sue colonne portano poca informazione sulla durata del volo, o se quella informazione è ripetuta in colonne rimaste pulite, sporcarla non dice nulla sull'effetto dell'inconsistenza sulle predizioni. Le FD sono quindi state selezionate in due passi (`analisi_rilevanza_fd.py`).

**Passo 1 — FD candidate.** Tutte le FD con 0 violazioni sul campione fra le colonne che arrivano al modello, valide nel dominio reale. Si raggruppano in cinque famiglie: distanza (`Distance → DistanceGroup`), rotta (`Origin+Dest → Distance`), aeroporto di origine e di destinazione (ID → codice, città, stato, FIPS, nome dello stato, WAC), compagnia (`Reporting_Airline → IATA_CODE`), orario (`CRSDepTime → DepTimeBlk`). Scartate perché false nella realtà: `DayofMonth → DayOfWeek` e `DayofMonth → FlightDate`, valide solo perché i dati coprono un solo mese.

**Passo 2 — rilevanza misurata.** Ogni configurazione è stata sporcata al 40% con la pipeline dell'esperimento (training sporco, test pulito, 3 repliche):

| Cosa si sporca | Colonne | Calo F1 medio | Modelli con calo significativo |
|---|---|---|---|
| Rotta e distanza, con tutte le copie | 16 | 8,4 punti | 4/4 |
| **FD `Distance → DistanceGroup`** | 2 | **7,9 punti** | 4/4 |
| **FD `Origin+Dest → Distance`** | 3 | **2,0 punti** | 4/4 |
| **Aeroporto di origine**, con le copie | 7 | **1,3 punti** | 4/4 |
| **Aeroporto di destinazione**, con le copie | 7 | **1,1 punti** | 4/4 |
| FD `Reporting_Airline → IATA_CODE` | 2 | 0,2 punti | 1/4 |
| FD `DestAirportID → DestState` (da sola) | 2 | 0,2 punti | 0/4 |
| FD `OriginAirportID → OriginState` (da sola) | 2 | 0,1 punti | 1/4 |
| FD `CRSDepTime → DepTimeBlk` | 2 | 0,1 punti | 1/4 |

Coerentemente, addestrando un albero su una sola colonna, `Distance` (F1 0,83) e `DistanceGroup` (0,80) predicono quasi da sole la durata; gli attributi degli aeroporti arrivano a 0,33–0,39; compagnia e orario sono vicini al caso puro.

**Due conseguenze per il disegno sperimentale:**

1. **Si usano solo le 4 FD rilevanti**: distanza, rotta, aeroporto di origine, aeroporto di destinazione. Le FD sugli aeroporti sono rilevanti solo come concetto intero, quindi vengono sporcate insieme alle colonne che ne ripetono l'informazione. Le altre FD valide del dataset hanno effetto nullo o non significativo: un esperimento con 10 FD avrebbe dovuto includerne sei irrilevanti.
2. **`DistanceGroup` era una via di fuga aperta.** Nella versione precedente del Blocco 1 non veniva sporcata: il modello recuperava da lì la distanza corrotta. Ora è inclusa fra le colonne ridondanti.

La certificazione delle FD usate è in `blocco2_verifica_fd.csv`.

## 5. Disegno sperimentale

Per ogni livello di rumore e replica: si corrompono le colonne delle FD scelte sul campione, poi si addestrano i 4 modelli RAW in **cross-validation stratificata a 5 fold** sulle righe sporcate, e si valuta **sulle stesse righe di test prese dal campione pulito** (oltre che, per confronto, su quelle sporcate).

- **Blocco 1** — una FD, `Origin + Dest → Distance`, sporcata insieme alle **17 colonne ridondanti** che codificano la stessa informazione (ID, codici di sequenza e di mercato, città, stato, FIPS, nome dello stato, WAC degli aeroporti, e `DistanceGroup`): tutte le vie di fuga sono chiuse.
- **Blocco 2** — le 4 FD rilevanti, aggiunte in ordine di rilevanza: **1 FD** (`Distance → DistanceGroup`), **2 FD** (+ `Origin+Dest → Distance`), **4 FD** (+ aeroporto di origine e di destinazione). Ogni FD sporca anche le colonne che ne ripetono l'informazione: la FD sulla rotta porta con sé `DistanceGroup`, quelle sugli aeroporti città, stato, FIPS, nome dello stato e WAC.

Livelli di rumore: **0, 5, 10, 20, 30, 40%**; 5 repliche con seed diversi per livello. Il seed di ogni replica è lo stesso in tutte le configurazioni, e righe bilanciate e fold sono identici in ogni condizione: le configurazioni si confrontano a coppie.

**Verifica della garanzia.** Ogni risultato riporta la quota di righe di training che differiscono dal campione pulito. Vale **0 a rumore 0%**, dove le due valutazioni coincidono. Nel Blocco 1 coincide con il livello richiesto (0,0495 · 0,0977 · 0,1977 · 0,2982 · 0,3976); nel Blocco 2, poiché ogni FD sceglie le proprie righe, vale 1 − (1 − p)^N, con uno scarto massimo di 0,003 dall'atteso. Le analisi di entrambi i blocchi si interrompono se uno di questi controlli fallisce.

---

# Risultati — Blocco 1: una dipendenza funzionale, vie di fuga chiuse

## 6.1 L'inconsistenza nel training degrada le predizioni

| Rumore | IM | IH | Decision Tree | Logistic Regression | Neural Network | Random Forest | Media |
|---|---|---|---|---|---|---|---|
| 0% | 0 | 0 | 0,9218 | 0,9218 | 0,9152 | 0,9345 | 0,9233 |
| 5% | 411 | 117 | 0,9189 | 0,8714 | 0,9089 | 0,9292 | 0,9071 |
| 10% | 787 | 260 | 0,9164 | 0,8352 | 0,9042 | 0,9279 | 0,8959 |
| 20% | 1.540 | 588 | 0,9105 | 0,7802 | 0,8943 | 0,9226 | 0,8769 |
| 30% | 2.257 | 984 | 0,9035 | 0,7288 | 0,8843 | 0,9153 | 0,8580 |
| 40% | 2.816 | 1.421 | 0,8921 | 0,6924 | 0,8660 | 0,9056 | 0,8390 |

*(F1 sul test pulito)*. Il degrado è **regolare e monotono** su tutti e 4 i modelli e significativo in **20 confronti su 20**. Al 40% di rumore la media perde **8,4 punti**, con forti differenze fra modelli: **2,9** Random Forest, **3,0** Decision Tree, **4,9** Neural Network, **22,9** Logistic Regression. Logistic Regression usa la distanza come valore numerico, e una distanza sostituita a caso sposta la sua previsione in modo proporzionale all'errore; gli alberi, che dividono per soglie, ne risentono molto meno.

**Confronto con la versione precedente**, identica tranne `DistanceGroup` lasciata pulita: al 40% la media perdeva 2,9 punti. **Gli indici IM e IH sono identici nelle due versioni** — `DistanceGroup` non fa parte della FD dichiarata — **ma il danno è quasi triplo.**

## 6.2 Dove si misura cambia quanto danno si vede

| Rumore | F1 test pulito | F1 test sporco | Perdita da apprendimento | Perdita da input corrotto | Quota da apprendimento |
|---|---|---|---|---|---|
| 5% | 0,9071 | 0,8729 | 0,0162 | 0,0342 | 32,2% |
| 10% | 0,8959 | 0,8309 | 0,0274 | 0,0650 | 29,6% |
| 20% | 0,8769 | 0,7524 | 0,0464 | 0,1245 | 27,2% |
| 30% | 0,8580 | 0,6791 | 0,0654 | 0,1789 | 26,8% |
| 40% | 0,8390 | 0,6117 | 0,0843 | 0,2273 | 27,1% |

Al 40% il calo misurato sul test sporcato è di **31,2 punti**, di cui **8,4 dovuti a un apprendimento peggiore**: misurare sul test sporco sovrastima il danno di **quasi quattro volte** (3,7). Su ogni riga sporca sono corrotte 20 colonne, e un input così degradato costa molto più dell'apprendimento peggiore.

## 6.3 Gli indici di inconsistenza seguono il degrado

Con una sola FD dichiarata, IM, IP e IH crescono a ogni aumento del rumore (IM da 411 a 2.816, IH da 117 a 1.421) mentre la F1 sul test pulito cala a ogni livello, per tutti e 4 i modelli: i tre indici ordinano i livelli di rumore esattamente come li ordina il danno.

---

# Risultati — Blocco 2: 1, 2 e 4 dipendenze funzionali rilevanti

## 7.1 Il degrado per numero di FD corrotte

F1 sul test pulito, media dei 4 modelli, e quota di righe di training effettivamente sporcate:

| Rumore | F1 · 1 FD | F1 · 2 FD | F1 · 4 FD | Righe sporche · 1 FD | · 2 FD | · 4 FD |
|---|---|---|---|---|---|---|
| 0% | 0,9233 | 0,9233 | 0,9233 | 0,0% | 0,0% | 0,0% |
| 5% | 0,9049 | 0,8933 | 0,8885 | 5,0% | 9,7% | 18,7% |
| 10% | 0,8901 | 0,8728 | 0,8655 | 9,8% | 18,8% | 34,2% |
| 20% | 0,8700 | 0,8490 | 0,8297 | 19,8% | 36,0% | 59,1% |
| 30% | 0,8572 | 0,8291 | 0,7934 | 29,8% | 50,9% | 76,1% |
| 40% | 0,8451 | 0,8137 | 0,7533 | 39,8% | 63,7% | 87,0% |

Il degrado è significativo in **60 confronti su 60**. Calo dal baseline al 40%, per modello:

| Configurazione | Decision Tree | Logistic Regression | Neural Network | Random Forest | Media |
|---|---|---|---|---|---|
| 1 FD | 3,7 | 23,6 | 2,1 | 1,9 | 7,8 |
| 2 FD | 7,4 | 28,1 | 4,3 | 4,1 | 11,0 |
| 4 FD | 11,0 | 31,6 | 15,0 | 10,5 | 17,0 |

**Una sola FD da due colonne fa quasi quanto il Blocco 1 con venti.** Sporcando solo `Distance` e `DistanceGroup` al 40% la media perde 7,8 punti, contro gli 8,4 del Blocco 1, che sporca anche rotta, aeroporti, città e stati sulle stesse righe. Nella versione precedente del Blocco 2, con 10 FD sugli attributi geografici degli aeroporti, il calo massimo era di 5,1 punti con il **99%** delle righe sporche.

## 7.2 Dove si misura cambia quanto danno si vede

| Configurazione (40%) | Perdita da apprendimento | Perdita da input corrotto | Quota da apprendimento |
|---|---|---|---|
| 1 FD | 0,0782 | 0,0698 | 52,8% |
| 2 FD | 0,1097 | 0,0870 | 55,8% |
| 4 FD | 0,1700 | 0,2589 | 39,6% |

La quota di danno dovuta all'apprendimento sta fra il **45% e il 56%** con 1 e 2 FD e fra il **36% e il 40%** con 4 FD, a ogni livello: misurare sul test sporco sovrastima il danno di **circa due volte** con 1 e 2 FD e di **due volte e mezzo** con 4 FD, contro quasi quattro del Blocco 1. La sovrastima cresce con il numero di colonne corrotte per riga.

## 7.3 Quanto gli indici seguono il danno

**Dentro ogni configurazione** IM e IH crescono a ogni livello di rumore, mentre la F1 cala: ordinano i livelli come li ordina il danno. Due limiti però emergono.

**Primo: con 4 FD gli indici si saturano.** Fra il 30% e il 40% di rumore la F1 perde altri 4,0 punti, mentre IM cresce solo dell'1,1% (da 5.862.349 a 5.924.540) e IH del 4,3% (da 28.099 a 29.316). L'ordine dei livelli resta corretto, ma la dimensione dell'aumento non dice più nulla sull'entità del danno.

**Secondo: fra configurazioni diverse IM non è confrontabile col danno.**

| Al 40% | 1 FD | 4 FD | Rapporto |
|---|---|---|---|
| IM | 186.614 | 5.924.540 | ×31,7 |
| IH | 17.498 | 29.316 | ×1,7 |
| Calo di F1 | 7,8 punti | 17,0 punti | ×2,2 |

IM è dominato dalle due FD sugli aeroporti, che hanno gruppi grandi (fino a 1.419 righe per aeroporto) e quindi moltissime coppie in conflitto: nella replica 0 al 40% producono **5,68 milioni** dei 5,92 milioni di conflitti, ma sono le FD che contano meno per la previsione. IH, che conta le righe da correggere e non le coppie, resta proporzionato al danno.

## 7.4 A parità di livello: più FD, più danno

T-test appaiato per replica fra configurazioni allo stesso livello di rumore: passando da 1 a 2 FD la F1 cala in media di **2,2 punti** (significativo in 20 confronti su 20), da 2 a 4 FD di **2,6 punti** (19 su 20).

A parità di livello, però, più FD sporcano anche più righe (al 20%: il 20% con una FD, il 59% con quattro). Il confronto va quindi ripetuto a parità di righe sporche.

## 7.5 A parità di righe sporche: dipende da quale FD e da quale modello

Coppie di configurazioni con quota di righe sporche confrontabile (entro 3,5 punti percentuali):

| Meno FD | Più FD | Righe sporche | F1 meno FD | F1 più FD | Modelli con più FD migliore |
|---|---|---|---|---|---|
| 1 FD al 10% | 2 FD al 5% | 9,8% / 9,7% | 0,8901 | 0,8933 | 3/4 |
| 1 FD al 20% | 2 FD al 10% | 19,8% / 18,8% | 0,8700 | 0,8728 | 2/4 |
| 1 FD al 20% | 4 FD al 5% | 19,8% / 18,7% | 0,8700 | 0,8885 | 3/4 |
| 2 FD al 10% | 4 FD al 5% | 18,8% / 18,7% | 0,8728 | 0,8885 | 3/4 |
| 2 FD al 20% | 4 FD al 10% | 36,0% / 34,2% | 0,8490 | 0,8655 | 3/4 |

Regressione per modello *F1 ~ quota di righe sporche + quota² + N_FD*, nell'intervallo di quota comune alle tre configurazioni (≤ 0,41, 55 punti):

| Modello | Coefficiente di N_FD | t | p | Lettura |
|---|---|---|---|---|
| Logistic Regression | +0,0202 | 18,5 | < 0,001 | più FD, meno danno |
| Decision Tree | +0,0029 | 7,1 | < 0,001 | più FD, meno danno |
| Random Forest | +0,0002 | 1,0 | 0,34 | nessun effetto |
| Neural Network | −0,0022 | −5,2 | < 0,001 | più FD, **più** danno |

Sull'intero intervallo il segno è lo stesso per Logistic Regression, Decision Tree e Neural Network (p < 0,001), mentre per Random Forest il coefficiente resta non significativo (p = 0,39).

**Interpretazione.** A parità di righe sporche, aggiungere FD significa sporcare meno la FD sulla distanza e più quelle su rotta e aeroporti. Il risultato dipende da quanto ciascun modello è sensibile a ciascuna informazione:
- **Logistic Regression e Decision Tree** soffrono soprattutto la distanza corrotta (rispettivamente 23,6 e 3,7 punti con la sola FD sulla distanza al 40%), quindi diluirla con altre FD riduce il danno;
- **Neural Network** soffre di più gli aeroporti corrotti: il suo calo al 40% passa da 4,3 punti con 2 FD a 15,0 con 4, quindi aggiungere le FD sugli aeroporti aumenta il danno anche a parità di righe;
- **Random Forest** non mostra alcun effetto del numero di FD a parità di righe sporche.

Non esiste quindi una regola generale "più vincoli violati, più (o meno) danno": **conta quale informazione è inconsistente, e quanto il modello dipende da quell'informazione.**

## 7.6 Perché IM si satura con 4 FD

Rigenerando i training sporcati con lo stesso seed (verificato: con 1 FD i conflitti ricalcolati coincidono con IM a ogni livello), per 4 FD e replica 0:

- le due FD sugli aeroporti producono **oltre il 95%** dei conflitti a ogni livello;
- sporcando l'ID dell'aeroporto le righe si ridistribuiscono fra gli aeroporti e i gruppi più grandi si svuotano (da 1.419 a 908 righe): le coppie di righe con lo stesso aeroporto di origine, cioè il massimo numero di conflitti possibili, scendono da 8,4 a 3,9 milioni;
- intanto la quota di quelle coppie in conflitto sale fino al 72%: i due effetti si compensano e i conflitti delle FD sugli aeroporti quasi smettono di crescere (da 2,794 a 2,817 milioni per l'origine fra il 30% e il 40%, cioè +0,8%);
- i conflitti della FD sulla rotta arrivano al massimo al 10% e poi calano (da 27.693 a 13.421), perché le rotte si frammentano: da 4.624 combinazioni distinte a 22.000.

Le coppie in conflitto su più FD insieme sono poche (somma dei conflitti per FD / IM = 1,01): la saturazione dipende dal tetto dei conflitti possibili, non dalla sovrapposizione fra FD. Rispetto al Blocco 2 precedente, con 10 FD, IM non arriva a calare, ma il meccanismo è lo stesso.

---

## 8. Risposta alla domanda di ricerca

> **Sì: l'inconsistenza dei dati di training riduce la qualità delle predizioni su dati corretti, in modo regolare, significativo e — quando colpisce informazione predittiva — rilevante. L'entità non dipende da quanta inconsistenza si misura, ma da quale informazione è inconsistente, da quante copie pulite di quell'informazione restano disponibili e da quanto il modello ne dipende.**

**1. L'effetto esiste ed è sistematico.** In entrambi i blocchi il degrado è monotono e significativo in tutti gli 80 confronti col baseline. Con le vie di fuga chiuse, il 40% di righe inconsistenti costa **8,4 punti di F1** nel Blocco 1 e fino a **17,0 punti** nel Blocco 2 con 4 FD.

**2. Conta quale informazione è inconsistente.** Due colonne sulla distanza fanno quasi il danno di venti colonne geografiche; FD valide ma irrilevanti per l'obiettivo, anche su quasi tutte le righe, fanno poco danno. Selezionare le FD per rilevanza è una condizione necessaria per studiare l'effetto dell'inconsistenza.

**3. Conta la ridondanza.** Lasciando pulita una sola colonna che ripete l'informazione corrotta, il danno si riduce a un terzo, **con gli stessi valori di IM, IP e IH**. Le misure di inconsistenza basate sulle FD non vedono le vie di fuga.

**4. Conta il modello.** Con la stessa inconsistenza Logistic Regression perde oltre 20 punti, Random Forest meno di 3 (Blocco 1, 40%). A parità di righe sporche, aggiungere FD riduce il danno per alcuni modelli e lo aumenta per altri.

**5. Dove si misura conta.** Valutare sul test sporcato confonde l'apprendimento peggiore con il costo di predire da input corrotti, e sovrastima il danno **da due a quasi quattro volte**.

**6. Le misure di inconsistenza tracciano il danno solo entro una configurazione, e IH è più affidabile di IM.** Con un insieme di FD fissato, IM e IH crescono a ogni livello di rumore mentre la F1 cala. Ma IM non è confrontabile fra insiemi di FD diversi (×32 fra 1 e 4 FD a fronte di un danno ×2,2), perché è dominato dalle FD con gruppi grandi, e con 4 FD quasi smette di crescere oltre il 30% di rumore mentre il danno continua. IH resta proporzionato al danno fra configurazioni.

## 9. Limiti

- **Un solo mese di dati** (gennaio 2025): il dataset originale non copre altro periodo.
- **Campione di 30.000 righe**, massimo compatibile con la costruzione in memoria del grafo dei conflitti (fino a 5,9 milioni di archi con 4 FD); dopo il bilanciamento i modelli lavorano su circa 6.470 righe per fold.
- **Rumore casuale uniforme**: gli errori reali tendono a essere sistematici (refusi, codici scambiati, valori vicini al vero). Una distanza sostituita con un valore qualsiasi è un errore più grave di uno realistico, e pesa soprattutto sui modelli che la usano come numero.
- **Poche FD rilevanti**: con questo obiettivo predittivo le FD rilevanti e indipendenti sono 4, non 10. La rilevanza è stata misurata a un solo livello di rumore (40%) con 3 repliche.
- **Nel Blocco 2 ogni FD sceglie righe diverse**: il confronto a parità di righe sporche (§7.5) separa i due effetti per via statistica, non per costruzione, e assume una relazione quadratica fra quota di righe sporche e F1.
- **Nel Blocco 2 le vie di fuga sono chiuse dentro ogni FD, non fra FD diverse.** Ogni FD sporca le proprie colonne e le proprie copie sulle righe che sceglie: sulle righe toccate solo dalla FD sulla distanza la rotta resta pulita, e il modello può in parte ricostruire la distanza da lì. Nel Blocco 1, dove tutte le colonne del concetto sono sporcate sulle stesse righe, questo non accade: è una delle ragioni per cui i due blocchi non sono confrontabili riga per riga.
- **IP satura nel Blocco 2**: conta le righe coinvolte in almeno un conflitto, e con 4 FD vale 30.000 — tutte le righe — già al 5% di rumore; con 1 FD ci arriva al 30%. Per questo le analisi del Blocco 2 si concentrano su IM e IH.
- **Modelli con iperparametri di default**, come richiesto: un modello ottimizzato potrebbe reagire diversamente.
- **Il disegno non separa incoerenza e perdita di informazione**: corrompere i dati produce sempre entrambe, e un confronto con dati altrettanto sbagliati ma coerenti richiederebbe un gruppo di controllo.

## 10. Materiale allegato

| Blocco 1 | Blocco 2 | Contenuto |
|---|---|---|
| `plot_blocco1_f1_test_pulito.png` | `plot_blocco2_scaling_fd.png` | Degrado al crescere del rumore |
| `plot_blocco1_sporco_vs_pulito.png` | `plot_blocco2_sporco_vs_pulito.png` | Test sporco contro test pulito |
| `tabella_riassuntiva_blocco1.csv` / `.png` | `tabella_riassuntiva_blocco2.csv` / `.png` | Tabella riassuntiva |
| — | `plot_blocco2_quota_sporca.png` | F1 a parità di righe sporche |
| — | `plot_blocco2_im_e_f1.png` | Andamento di IM con 4 FD |
| `blocco1_aggregato.csv`, `blocco1_scomposizione.csv`, `blocco1_test_degrado.csv` | `blocco2_aggregato.csv`, `blocco2_scomposizione.csv`, `blocco2_test_degrado.csv` | Analisi statistica comune |
| — | `blocco2_test_configurazioni.csv`, `blocco2_quota_normalizzata.csv`, `blocco2_quota_punti_confrontabili.csv`, `blocco2_meccanismo_im.csv`, `blocco2_verifica_fd.csv` | Analisi specifiche del Blocco 2 |
| `rilevanza_colonne.csv`, `rilevanza_fd.csv`, `rilevanza_fd.log` | | Selezione delle FD per rilevanza (§4) |
| `archivio_prima_revisione_fd/` | | Risultati con `DistanceGroup` pulita e 10 FD irrilevanti, per confronto |

---

## In parole più semplici

**La domanda.** Se un programma di intelligenza artificiale impara da dati pieni di errori, ma poi lo si usa su dati corretti, quanto sbaglia in più?

**Come si è misurato.** Si prendono 30.000 voli e si chiede a quattro programmi di indovinare quanto durerà un volo. Con dati corretti ci riescono bene: azzeccano circa 92 volte su 100, contro le 20 che otterrebbero tirando a caso. Poi si rovinano di proposito i dati con cui imparano, su una quota crescente di voli, fino al 40%. Le domande di verifica, invece, si fanno sempre con dati corretti.

**Prima di tutto: cosa rovinare.** Nel dataset ci sono molte regole sempre vere (un aeroporto sta in un solo stato, una rotta ha una sola distanza, e così via). Ma non tutte servono a indovinare la durata del volo. Si è misurato quali contano: la distanza conta moltissimo, la rotta e gli aeroporti un po', la compagnia e l'orario quasi nulla. Si sono rovinate solo quelle che contano.

**Cosa si è scoperto.**

- **Rovinare i dati giusti fa peggiorare molto le previsioni**: rovinando la distanza sul 40% dei voli si perdono circa 8 punti su 92, e rovinando anche rotta e aeroporti fino a 15.
- **Basta una copia pulita per salvare il programma.** Nel dataset la distanza compare due volte: in chilometri e in fasce. Se si rovina solo la prima, il programma usa la seconda e il danno si riduce a un terzo. Il conteggio delle contraddizioni, però, resta identico: non si accorge che c'era una copia di riserva.
- **Non tutti i programmi soffrono allo stesso modo.** Quello che fa i conti con i numeri (Logistic Regression) crolla quando la distanza è sbagliata; quelli che ragionano per soglie (gli alberi) se la cavano molto meglio.
- **Il modo di misurare conta.** Se anche le domande di verifica si fanno con dati rovinati, il danno sembra da due a quattro volte più grande.
- **Contare le contraddizioni non basta.** Le regole sugli aeroporti producono milioni di contraddizioni perché ogni aeroporto ha migliaia di voli, ma sono le regole che contano meno. La distanza produce poche contraddizioni e fa il danno maggiore. Contare quante righe andrebbero corrette (IH) rispecchia il danno meglio che contare le coppie di righe in contraddizione (IM).
