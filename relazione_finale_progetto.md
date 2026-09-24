# Relazione finale — Misure di inconsistenza per Data-Centric AI

Relazione scientifica del progetto di tesi: domanda di ricerca, disegno sperimentale, risultati e conclusioni. Versione aggiornata dopo le otto osservazioni del relatore (settembre 2026): misure calcolate sulle righe di training, baseline variabile e test di Welch, IH dichiarato come limite superiore e confrontato con il minimo esatto, IP e IH analizzati accanto a IM.

Il dettaglio delle modifiche metodologiche è in [`relazione_revisione_sperimentale.md`](relazione_revisione_sperimentale.md); le correzioni da riportare nel testo della tesi sono in [`correzioni_tesi.md`](correzioni_tesi.md); tutti i numeri citati qui sono in [`numeri_tesi.md`](numeri_tesi.md), generato dai CSV.

---

## 1. Domanda di ricerca

> **Quanto incide sulla qualità delle predizioni di un modello AI la qualità (consistenza) dei dati in input?**

Nella forma resa operativa: *se addestro un modello su dati contaminati ma poi lo uso su dati corretti, quanta capacità predittiva perdo, e come cresce questa perdita al crescere della corruzione?*

**Variabili.** L'unica variabile controllata è la **percentuale di righe alterate** nel training, fissata su sei livelli. Le misure di inconsistenza IM, IP e IH sono calcolate dopo l'iniezione del rumore: sono **variabili osservate**, al pari della F1 dei modelli. Il loro rapporto con il degrado si studia confrontandone l'andamento al variare del rumore.

**Vincoli metodologici** rispettati in tutto il lavoro:
- i modelli restano **RAW**: iperparametri di default, `random_state=42`, nessun `class_weight`, nessun tuning per condizione sperimentale;
- l'obiettivo predittivo è unico ed esplicito, e le dipendenze sporcate sono scelte **in base alla loro rilevanza per quell'obiettivo**;
- il training viene sporcato gradualmente, ma **il test è sempre su dati puliti**.

## 2. Framework di misura dell'inconsistenza

L'inconsistenza è definita rispetto a **dipendenze funzionali (FD)** della forma `LHS → RHS`. Si costruisce un grafo dei conflitti — nodo = riga, arco = coppia di righe con stesso LHS ma RHS diverso — e se ne ricavano tre misure:

- **IM**: numero di conflitti a coppie (archi del grafo);
- **IP**: numero di tuple coinvolte in almeno un conflitto;
- **IH**: numero minimo di tuple da eliminare per rimuovere tutte le violazioni, cioè il vertex cover minimo del grafo.

**IH è calcolato come limite superiore.** Il vertex cover minimo è NP-difficile; il progetto usa `min_weighted_vertex_cover` di NetworkX, un algoritmo **2-approssimato**: restituisce una copertura valida, quindi un valore **maggiore o uguale** al minimo e al più doppio. Due verifiche ne misurano lo scarto:

- **con una sola FD il minimo esatto ha forma chiusa.** Il grafo è l'unione disgiunta, sui gruppi del lato sinistro, di grafi multipartiti completi (le parti sono i valori del lato destro). Il massimo insieme indipendente di un multipartito completo è la parte più numerosa, quindi il vertex cover minimo di ogni gruppo vale righe del gruppo meno righe della classe più numerosa. Nel Blocco 1 si riporta anche questo valore esatto;
- **con più FD** si confronta l'approssimato con l'ottimo di un modello di programmazione lineare intera (`verifica_ih.py`, 60 istanze da 250 a 2.000 righe, 59 risolte all'ottimo):

| | Rapporto approssimato / esatto |
|---|---|
| Istanze ridotte (250–2.000 righe), 1 FD | da 1,00 a 1,08 (formula chiusa = ottimo ILP in tutte) |
| Istanze ridotte, 2 FD | da 1,04 a 1,22 |
| Istanze ridotte, 4 FD | da 1,31 a 1,47 |
| Esperimenti del Blocco 1 (1 FD, fold di training da 6.472 righe) | da 1,12 a 1,44 |
| Esperimenti del Blocco 2, 1 FD (formula chiusa) | da 1,45 a 1,63 |
| Esperimenti del Blocco 2, 2 FD (ILP, fino al 20%) | da 1,54 a 1,69 |
| Esperimenti del Blocco 2, 4 FD (ILP, fino al 10%) | da 1,60 a 1,70 |

**Sulle istanze reali lo scarto è molto più grande che sulle istanze ridotte.** Sui fold di training del Blocco 2 il valore approssimato supera il minimo del 45–70%, contro al più il 47% sulle istanze da poche centinaia di righe: la verifica su istanze piccole sottostimava l'errore. Lo scarto resta sotto il limite teorico di 2 e diminuisce al crescere del rumore. Il minimo esatto è calcolabile con l'ILP solo fino a un certo livello di corruzione (fino al 20% con 2 FD, fino al 10% con 4 FD, entro 150 secondi per istanza); oltre, il risolutore non chiude. L'andamento delle due curve è lo stesso (`plot_blocco2_indici_e_f1.png`, `plot_blocco2_scaling_fd.png`), quindi le conclusioni qualitative su IH non cambiano; il valore assoluto dell'approssimato va però letto come un limite superiore largo.

## 3. Dati e obiettivo predittivo

**Task**: classificazione a 5 classi della durata schedulata del volo (`DurataBucket`, da `CRSElapsedTime`: `<90m`, `90-150m`, `150-210m`, `210-300m`, `>300m`).

**Dati**: campione di **30.000 voli** dal dataset *Reporting Carrier On-Time Performance* del Bureau of Transportation Statistics (gennaio 2025). Le classi sono bilanciate per undersampling → **8.090 righe**, **6.472 righe di training per fold** con cross-validation stratificata a **5 fold**. Il caso puro vale **0,20**.

**Baseline** a rumore 0%, su dati puliti (F1 weighted, media delle 5 repliche), comune ai due blocchi:

| Modello | F1 | Deviazione standard fra repliche |
|---|---|---|
| Random Forest | 0,9306 | 0,0013 |
| Logistic Regression | 0,9231 | 0,0022 |
| Decision Tree | 0,9212 | 0,0044 |
| Neural Network | 0,9107 | 0,0016 |

Media: **0,9214**. Il baseline ha una sua variabilità perché ogni replica usa righe bilanciate e fold propri (§5).

## 4. Quali dipendenze funzionali contano per l'obiettivo

Una FD valida non è necessariamente rilevante: se le sue colonne portano poca informazione sulla durata del volo, o se quell'informazione è ripetuta in colonne rimaste pulite, sporcarla non dice nulla sull'effetto dell'inconsistenza sulle predizioni. Le FD sono state selezionate in due passi (`analisi_rilevanza_fd.py`).

**Passo 1 — FD candidate.** Tutte le FD con 0 violazioni sul campione fra le colonne che arrivano al modello, valide nel dominio reale, raggruppate in cinque famiglie: distanza, rotta, aeroporto di origine, aeroporto di destinazione, compagnia, orario. Scartate perché false nella realtà: `DayofMonth → DayOfWeek` e `DayofMonth → FlightDate`, valide solo perché i dati coprono un solo mese.

**Passo 2 — rilevanza misurata.** Ogni configurazione è sporcata al 40% con la pipeline dell'esperimento (training sporco, test pulito, 3 repliche); il calo è confrontato con le 5 repliche del baseline con un **t-test di Welch** per ciascun modello.

| Cosa si sporca | Colonne | Calo F1 medio | Modelli con calo significativo |
|---|---|---|---|
| Rotta e distanza, con tutte le copie | 16 | 8,3 punti | 4/4 |
| **FD `Distance → DistanceGroup`** | 2 | **8,0 punti** | 4/4 |
| **FD `Origin+Dest → Distance`** | 3 | **2,0 punti** | 4/4 |
| **Aeroporto di origine**, con le copie | 7 | **1,2 punti** | 3/4 |
| **Aeroporto di destinazione**, con le copie | 7 | **1,1 punti** | 4/4 |
| FD `Reporting_Airline → IATA_CODE` | 2 | 0,3 punti | 0/4 |
| FD `CRSDepTime → DepTimeBlk` | 2 | 0,1 punti | 0/4 |
| FD `DestAirportID → DestState` (da sola) | 2 | 0,1 punti | 0/4 |
| FD `OriginAirportID → OriginState` (da sola) | 2 | 0,0 punti | 0/4 |

**Criterio di rilevanza**: calo medio di almeno un punto di F1, significativo per la maggioranza dei modelli. Separa due gruppi netti — le quattro FD scelte (da 1,1 a 8,0 punti, 3 o 4 modelli su 4) e tutte le altre (al più 0,3 punti, nessun modello). Il caso limite è l'aeroporto di origine, significativo per 3 modelli su 4.

**Due conseguenze per il disegno sperimentale:**

1. **Si usano solo le 4 FD rilevanti**; le FD sugli aeroporti sono rilevanti solo come concetto intero, quindi vengono sporcate insieme alle colonne che ne ripetono l'informazione.
2. **`DistanceGroup` è una via di fuga** e viene sporcata insieme alla distanza, sia nel Blocco 1 sia con la FD sulla rotta nel Blocco 2.

## 5. Disegno sperimentale

Per ogni livello di rumore e replica: si corrompono le colonne delle FD scelte sul campione, si addestrano i 4 modelli RAW in **cross-validation stratificata a 5 fold** sulle righe sporcate, e si valuta **sulle stesse righe di test prese dal campione pulito** (oltre che, per confronto, su quelle sporcate).

- **Blocco 1** — una FD, `Origin + Dest → Distance`, sporcata insieme alle **17 colonne ridondanti** (ID, codici di sequenza e di mercato, città, stato, FIPS, nome dello stato, WAC degli aeroporti, e `DistanceGroup`): tutte le vie di fuga sono chiuse.
- **Blocco 2** — le 4 FD rilevanti, aggiunte in ordine di rilevanza: **1 FD** (`Distance → DistanceGroup`), **2 FD** (+ `Origin+Dest → Distance`, con `DistanceGroup`), **4 FD** (+ aeroporto di origine e di destinazione, con città, stato, FIPS, nome dello stato e WAC).

Livelli di rumore: **0, 5, 10, 20, 30, 40%**; 5 repliche.

**Il seed della replica governa rumore, bilanciamento e fold.** Ogni replica lavora su righe bilanciate e fold propri, quindi anche il baseline varia: il degrado si valuta con un **t-test di Welch a due campioni** (5 repliche del livello contro 5 del baseline), senza trattare il baseline come una costante. Il seed dipende solo dalla replica: a parità di replica le configurazioni del Blocco 2 condividono righe e fold e si confrontano a coppie.

**Le misure sono calcolate sulle righe di training di ciascun fold** (6.472 righe), le stesse su cui si addestrano i modelli; se ne riportano media e deviazione standard sui 5 fold. Indici e metriche descrivono così la stessa popolazione.

**Verifiche automatiche.** Ogni risultato riporta la quota di righe di training che differiscono dal campione pulito: vale **0 a rumore 0%**, dove le due valutazioni coincidono, e cresce con il rumore (nel Blocco 2 segue 1 − (1 − p)^N, scarto massimo 0,004). Le analisi si interrompono anche se gli indici non risultano calcolati sulle righe di training, o se il baseline non ha varianza.

---

# Risultati — Blocco 1: una dipendenza funzionale, vie di fuga chiuse

## 6.1 L'inconsistenza nel training degrada le predizioni

| Rumore | Decision Tree | Logistic Regression | Neural Network | Random Forest | Media |
|---|---|---|---|---|---|
| 0% | 0,9212 | 0,9231 | 0,9107 | 0,9306 | 0,9214 |
| 5% | 0,9176 | 0,8685 | 0,9049 | 0,9287 | 0,9049 |
| 10% | 0,9133 | 0,8301 | 0,8978 | 0,9271 | 0,8921 |
| 20% | 0,9095 | 0,7763 | 0,8870 | 0,9229 | 0,8739 |
| 30% | 0,9009 | 0,7295 | 0,8779 | 0,9157 | 0,8560 |
| 40% | 0,8940 | 0,6896 | 0,8644 | 0,9046 | 0,8381 |

*(F1 sul test pulito, media delle 5 repliche)*. Il degrado è monotono per tutti e 4 i modelli. Al 40% la media perde **8,3 punti**: 2,6 Random Forest, 2,7 Decision Tree, 4,6 Neural Network, **23,4 Logistic Regression**, circa **7 volte** il calo medio degli altri tre (3,3 punti).

**Significatività (Welch): 18 confronti su 20.** Non sono significativi Decision Tree e Random Forest al 5% di rumore (cali di 0,36 e 0,19 punti, p = 0,26): a rumore basso il loro degrado non si distingue dalla variabilità fra repliche. Tutti gli altri lo sono.

**Perché Logistic Regression.** L'analisi di rilevanza mostra che sporcando **solo** `Distance` e `DistanceGroup`, due colonne numeriche, Logistic Regression perde già 24 punti. La sua fragilità dipende quindi soprattutto dall'uso della distanza come valore numerico: una distanza sostituita a caso sposta la previsione in proporzione all'errore, mentre gli alberi dividono per soglie e ne risentono molto meno.

## 6.2 Dove si misura cambia quanto danno si vede

| Rumore | F1 test pulito | F1 test sporco | Perdita da apprendimento | Perdita da input corrotto | Quota da apprendimento |
|---|---|---|---|---|---|
| 5% | 0,9049 | 0,8710 | 0,0165 | 0,0339 | 32,7% |
| 10% | 0,8921 | 0,8272 | 0,0293 | 0,0649 | 31,2% |
| 20% | 0,8739 | 0,7505 | 0,0475 | 0,1234 | 27,8% |
| 30% | 0,8560 | 0,6783 | 0,0654 | 0,1777 | 26,9% |
| 40% | 0,8381 | 0,6115 | 0,0833 | 0,2266 | 26,9% |

Al 40% il calo misurato sul test sporcato è di **31,0 punti**, di cui **8,3 dovuti a un apprendimento peggiore**: misurare sul test sporco sovrastima il danno di **3,7 volte**. Per modello la quota dovuta all'apprendimento è 58–63% per Logistic Regression e 5–16% per gli altri tre.

## 6.3 Le tre misure al crescere del rumore

Misure sulle righe di training di ciascun fold (media sulle repliche):

| Rumore | IM | IP | IH 2-approssimato | IH esatto |
|---|---|---|---|---|
| 5% | 16 | 24 | 10 | 7 |
| 10% | 37 | 55 | 22 | 18 |
| 20% | 75 | 114 | 50 | 40 |
| 30% | 107 | 165 | 74 | 60 |
| 40% | 135 | 218 | 102 | 86 |

Con una FD tutte e tre le misure crescono a ogni livello, mentre la F1 cala. IP è la più grande: con questa FD le tuple coinvolte sono più delle coppie in conflitto, perché i gruppi della rotta sono piccoli (una riga sporcata entra in conflitto con poche altre). IH approssimato supera l'esatto del 12–44%.

---

# Risultati — Blocco 2: 1, 2 e 4 dipendenze funzionali rilevanti

## 7.1 Il degrado per numero di FD corrotte

| Rumore | F1 · 1 FD | F1 · 2 FD | F1 · 4 FD | Righe sporche · 1 FD | · 2 FD | · 4 FD |
|---|---|---|---|---|---|---|
| 0% | 0,9214 | 0,9214 | 0,9214 | 0,0% | 0,0% | 0,0% |
| 5% | 0,9015 | 0,8887 | 0,8852 | 4,8% | 9,7% | 18,5% |
| 10% | 0,8861 | 0,8697 | 0,8618 | 9,8% | 18,9% | 34,2% |
| 20% | 0,8684 | 0,8472 | 0,8272 | 19,6% | 35,7% | 58,9% |
| 30% | 0,8554 | 0,8299 | 0,7926 | 29,7% | 50,8% | 75,9% |
| 40% | 0,8430 | 0,8099 | 0,7507 | 39,7% | 64,1% | 87,2% |

Calo dal baseline al 40%, per modello:

| Configurazione | Decision Tree | Logistic Regression | Neural Network | Random Forest | Media |
|---|---|---|---|---|---|
| 1 FD | 3,9 | 24,0 | 2,0 | 1,5 | 7,8 |
| 2 FD | 8,0 | 28,8 | 4,1 | 3,7 | 11,2 |
| 4 FD | 11,6 | 32,0 | 14,5 | 10,3 | 17,1 |

Il degrado è significativo (Welch) in **58 confronti su 60**: non lo sono Decision Tree e Random Forest con 1 FD al 5% di rumore. Una sola FD da due colonne (`Distance → DistanceGroup`, 7,8 punti) fa quasi quanto il Blocco 1 con venti colonne (8,3 punti).

## 7.2 Dove si misura cambia quanto danno si vede

La quota di danno dovuta all'apprendimento sta fra il **51% e il 53%** con 1 FD, fra il **47% e il 56%** con 2 FD e fra il **36% e il 40%** con 4 FD. Al 40% la sovrastima del test sporco è di circa **1,9 volte** con 1 FD (14,8 contro 7,8 punti) e **2,5 volte** con 4 FD (43,2 contro 17,1), contro 3,7 del Blocco 1: cresce con il numero di colonne corrotte per riga.

## 7.3 Le tre misure al crescere del rumore

| Rumore | IM · 1 FD | IM · 2 FD | IM · 4 FD | IP · 1 FD | IP · 2 FD | IP · 4 FD | IH · 1 FD | IH · 2 FD | IH · 4 FD |
|---|---|---|---|---|---|---|---|---|---|
| 5% | 1.306 | 3.604 | 92.275 | 1.390 | 3.093 | 6.467 | 403 | 916 | 1.868 |
| 10% | 2.596 | 6.579 | 161.630 | 2.538 | 4.592 | 6.472 | 803 | 1.729 | 3.244 |
| 20% | 4.813 | 10.761 | 251.133 | 4.026 | 5.780 | 6.472 | 1.549 | 2.997 | 4.960 |
| 30% | 6.957 | 13.492 | 290.361 | 4.995 | 6.172 | 6.472 | 2.292 | 3.917 | 5.750 |
| 40% | 8.665 | 14.742 | 289.586 | 5.553 | 6.312 | 6.472 | 2.930 | 4.487 | 6.122 |

*(IH 2-approssimato; righe di training per fold: 6.472)*

Tre comportamenti diversi:

- **IM**, che conta coppie, cresce fino al 30% con 4 FD e poi si ferma (289.586 al 40%, lievemente sotto il 30%).
- **IP**, che conta tuple, con 4 FD raggiunge il suo massimo — **tutte le 6.472 righe** — già al 10% di rumore. Con 1 FD invece cresce a ogni livello, fino all'86% delle righe.
- **IH** cresce a ogni livello anche con 4 FD (da 4.960 al 20% a 6.122 al 40%): è l'ultima delle tre misure a fermarsi.

**Fra configurazioni diverse le misure non sono confrontabili fra loro allo stesso modo.** Al 40%, passando da 1 a 4 FD, IM cresce di 33 volte, IH di 2,1 volte, IP di 1,2 volte (è saturato), mentre il danno cresce di 2,2 volte. IM è dominato dalle FD sugli aeroporti, che hanno gruppi grandi e quindi moltissime coppie possibili.

## 7.4 A parità di livello: più FD, più danno

T-test appaiato per replica: passando da 1 a 2 FD la F1 cala in media di **2,2 punti** (significativo in 19 confronti su 20), da 2 a 4 FD di **2,6 punti** (18 su 20); i confronti non significativi sono tutti al 5% di rumore. A parità di livello, però, più FD sporcano anche più righe (al 20%: 19,6% con una FD, 58,9% con quattro).

## 7.5 A parità di righe sporche: dipende da quale FD e da quale modello

Coppie di configurazioni con quota di righe sporche confrontabile (entro 3,5 punti percentuali):

| Meno FD | Più FD | F1 meno FD | F1 più FD | Modelli con più FD migliore |
|---|---|---|---|---|
| 1 FD al 10% | 2 FD al 5% | 0,8861 | 0,8887 | 3/4 |
| 1 FD al 20% | 2 FD al 10% | 0,8684 | 0,8697 | 2/4 |
| 1 FD al 20% | 4 FD al 5% | 0,8684 | 0,8852 | 3/4 |
| 2 FD al 10% | 4 FD al 5% | 0,8697 | 0,8852 | 4/4 |
| 2 FD al 20% | 4 FD al 10% | 0,8472 | 0,8618 | 3/4 |

Regressione per modello *F1 ~ quota di righe sporche + quota² + N_FD*, nell'intervallo di quota comune (≤ 0,41):

| Modello | Coefficiente di N_FD | p | Lettura |
|---|---|---|---|
| Logistic Regression | +0,0188 | < 0,001 | più FD, meno danno |
| Decision Tree | +0,0029 | < 0,001 | più FD, meno danno |
| Random Forest | −0,0001 | 0,81 | nessun effetto |
| Neural Network | −0,0023 | < 0,001 | più FD, **più** danno |

A parità di righe sporche, aggiungere FD significa sporcare meno la distanza e più rotta e aeroporti: Logistic Regression e Decision Tree soffrono soprattutto la distanza e ne beneficiano; Neural Network soffre di più gli aeroporti (4,1 punti con 2 FD, 14,5 con 4) e ne risente; Random Forest è indifferente. Non esiste una regola generale "più vincoli violati, più (o meno) danno": **conta quale informazione è inconsistente, e quanto il modello ne dipende.**

## 7.6 Perché le misure si fermano: un comportamento corretto

Rigenerando il training sporcato della replica 0 sulle righe del primo fold (verificato: con 1 FD i conflitti ricalcolati coincidono con IM a ogni livello), con 4 FD:

- **IM si ferma perché diminuiscono le coppie possibili.** Corrompendo l'ID dell'aeroporto le righe si ridistribuiscono fra gli aeroporti e i gruppi più grandi si svuotano (il più grande di `OriginAirportID → Origin` passa da 276 a 177 righe): le coppie di righe che condividono il lato sinistro, cioè il massimo di conflitti possibili, diminuiscono. I conflitti della FD sulla rotta arrivano al massimo al 20% (1.683) e poi calano (745 al 40%), perché le rotte si frammentano (gruppo più grande da 38 a 8 righe).
- **IP si ferma perché non ci sono più tuple da aggiungere.** Basta una riga corrotta in un gruppo per rendere "coinvolte" tutte le righe del gruppo: con le FD sugli aeroporti, i cui gruppi contengono centinaia di righe, già al 5% di rumore sono coinvolte 6.223 righe su 6.472 per la sola origine.
- **IH si avvicina al numero di righe** ma continua a crescere fino al 40%.

In tutti e tre i casi **la misura si comporta correttamente**: quando IM cala il dato è davvero meno inconsistente rispetto al vincolo, perché gruppi più piccoli generano meno coppie in conflitto; quando IP si ferma tutte le righe sono davvero coinvolte. Le misure quantificano le violazioni di un vincolo, non il danno subito da un modello: il fatto che il danno continui a crescere mentre IM e IP si fermano dice che, a parità di violazioni misurate, il degrado dipende da quale informazione è stata alterata.

---

## 8. Risposta alla domanda di ricerca

> **L'inconsistenza dei dati di training riduce la qualità delle predizioni su dati corretti, in modo regolare e — quando colpisce informazione predittiva — rilevante. L'entità dipende da quale informazione è inconsistente, da quante copie pulite di quell'informazione restano disponibili e da quanto il modello ne dipende. Le misure di inconsistenza crescono con la corruzione, ma quantificano le violazioni dei vincoli, non il degrado del modello.**

**1. L'effetto esiste ed è sistematico, ma non a ogni livello.** Il degrado è monotono in entrambi i blocchi e significativo in 76 confronti su 80 con il test di Welch. Con le vie di fuga chiuse, il 40% di righe inconsistenti costa **8,3 punti di F1** nel Blocco 1 e fino a **17,1 punti** nel Blocco 2 con 4 FD. Al 5% di rumore, con una sola FD, il degrado di Decision Tree e Random Forest non si distingue dalla variabilità fra repliche.

**2. Conta quale informazione è inconsistente.** Due colonne sulla distanza fanno quasi il danno di venti colonne geografiche; FD valide ma irrilevanti per l'obiettivo fanno poco danno anche su quasi tutte le righe. Selezionare le FD per rilevanza è una condizione necessaria per studiare l'effetto dell'inconsistenza.

**3. Conta il modello.** Con la stessa inconsistenza Logistic Regression perde 23,4 punti, gli altri tre in media 3,3 (Blocco 1, 40%): la differenza nasce soprattutto dall'uso della distanza come valore numerico. A parità di righe sporche, aggiungere FD riduce il danno per alcuni modelli e lo aumenta per altri.

**4. Dove si misura conta.** Valutare sul test sporcato confonde l'apprendimento peggiore con il costo di predire da input corrotti, e sovrastima il danno **da circa 2 a 3,7 volte**.

**5. Le misure di inconsistenza descrivono l'inconsistenza, non il danno.** Con un insieme di FD fissato crescono tutte al crescere del rumore, come il degrado. Ma ciascuna ha un proprio tetto — le coppie possibili per IM, le righe per IP e IH — e con 4 FD IM e IP lo raggiungono prima che il danno smetta di crescere. Fra insiemi di FD diversi il valore assoluto non è confrontabile: IM cresce di 33 volte passando da 1 a 4 FD, a fronte di un danno doppio, mentre IH cresce in modo molto più vicino al danno (2,1 volte).

## 9. Limiti

- **Un solo mese di dati** (gennaio 2025) e un solo obiettivo predittivo.
- **Campione di 30.000 righe**, mantenuto per confrontabilità con le versioni precedenti. Con le misure calcolate sui fold di training il grafo dei conflitti non è più il vincolo principale; il costo è dominato dai 480 addestramenti per blocco.
- **IH è un limite superiore** del minimo: sulle istanze reali il 2-approssimato lo supera del 45–70%. Il valore esatto è disponibile con una FD (formula chiusa) e, con più FD, solo fino al 20% (2 FD) e al 10% (4 FD) di rumore.
- **Rumore casuale uniforme**: gli errori reali tendono a essere sistematici. Una distanza sostituita con un valore qualsiasi è un errore più grave di uno realistico, e pesa soprattutto sui modelli che la usano come numero.
- **Poche FD rilevanti**: con questo obiettivo le FD rilevanti e indipendenti sono 4. La rilevanza è misurata a un solo livello di rumore (40%) con 3 repliche; l'aeroporto di origine è un caso limite (3 modelli su 4).
- **Nel Blocco 2 ogni FD sceglie righe diverse**: il confronto a parità di righe sporche separa i due effetti per via statistica, non per costruzione, e assume una relazione quadratica fra quota e F1. Le vie di fuga sono chiuse dentro ogni FD, non fra FD diverse.
- **Cinque repliche per livello**: i test a rumore basso hanno poca potenza, come mostrano i due confronti non significativi al 5%.
- **Modelli con iperparametri di default**, come richiesto.
- **Il disegno non separa incoerenza e perdita di informazione**: corrompere i dati produce sempre entrambe.

## 10. Materiale allegato

| Blocco 1 | Blocco 2 | Contenuto |
|---|---|---|
| `plot_blocco1_f1_test_pulito.png` | `plot_blocco2_scaling_fd.png` | Degrado al crescere del rumore (Blocco 2: con IM, IP e IH) |
| `plot_blocco1_sporco_vs_pulito.png` | `plot_blocco2_sporco_vs_pulito.png` | Test sporco contro test pulito |
| `tabella_riassuntiva_blocco1.csv` / `.png` | `tabella_riassuntiva_blocco2.csv` / `.png` | Tabelle con F1, righe sporche, IM, IP e IH |
| — | `plot_blocco2_quota_sporca.png` | F1 a parità di righe sporche |
| — | `plot_blocco2_indici_e_f1.png` | Danno e tre misure con 4 FD |
| `blocco1_aggregato.csv`, `blocco1_scomposizione.csv`, `blocco1_test_degrado.csv` | `blocco2_aggregato.csv`, `blocco2_scomposizione.csv`, `blocco2_test_degrado.csv` | Analisi statistica comune |
| — | `blocco2_test_configurazioni.csv`, `blocco2_quota_normalizzata.csv`, `blocco2_quota_punti_confrontabili.csv`, `blocco2_meccanismo_indici.csv`, `blocco2_verifica_fd.csv` | Analisi specifiche del Blocco 2 |
| `rilevanza_colonne.csv`, `rilevanza_fd.csv`, `rilevanza_fd.log` | | Selezione delle FD (§4) |
| `verifica_ih.csv`, `verifica_ih.log` | `blocco2_ih_esatto.csv`, `ih_esatto_blocco2.log` | IH approssimato contro esatto (§2): istanze ridotte e istanze reali |
| `numeri_tesi.md` | | Tutti i numeri da citare in tesi |
| `archivio_prima_revisione_prof/` | | Risultati prima delle osservazioni del relatore |

---

## In parole più semplici

**La domanda.** Se un programma di intelligenza artificiale impara da dati pieni di errori, ma poi lo si usa su dati corretti, quanto sbaglia in più?

**Come si è misurato.** Si prendono 30.000 voli e si chiede a quattro programmi di indovinare quanto durerà un volo. Con dati corretti azzeccano circa 92 volte su 100, contro le 20 che otterrebbero tirando a caso. Poi si rovinano di proposito i dati con cui imparano, su una quota crescente di voli, fino al 40%. Le domande di verifica si fanno sempre con dati corretti. Ogni prova è ripetuta cinque volte con dati diversi, per distinguere un peggioramento vero da una semplice fluttuazione.

**Cosa si è scoperto.**

- **Rovinare i dati giusti fa peggiorare molto le previsioni**: rovinando la distanza sul 40% dei voli si perdono circa 8 punti su 92, e rovinando anche rotta e aeroporti fino a 17. Con pochissimi errori (5%), invece, due programmi su quattro non peggiorano in modo distinguibile dal caso.
- **Non tutti i programmi soffrono allo stesso modo.** Quello che fa i conti con i numeri (Logistic Regression) crolla quando la distanza è sbagliata; quelli che ragionano per soglie (gli alberi) se la cavano molto meglio.
- **Il modo di misurare conta.** Se anche le domande di verifica si fanno con dati rovinati, il danno sembra da due a quasi quattro volte più grande.
- **Contare le contraddizioni non basta per prevedere il danno.** I tre contatori di contraddizioni crescono insieme agli errori, ma ciascuno a un certo punto si ferma: uno perché le coppie di voli in contraddizione non possono più aumentare, un altro perché tutti i voli sono già coinvolti. Il programma, intanto, continua a peggiorare. I contatori misurano bene quanto i dati violano le regole; quanto questo danneggi il programma dipende da quale informazione è stata rovinata.
