# Sottoprogetto — Blocco 1 con la FD `Distance → DistanceGroup` e obiettivo `AirTime`

Variante del Blocco 1 del progetto principale. Cambiano la dipendenza funzionale dichiarata e l'obiettivo predittivo; la metodologia è identica. Il progetto principale non è stato modificato: gli script di questa cartella ne importano in sola lettura il motore (`progettoTesi_v2.py`), l'esecuzione parallela (`esecuzione_parallela.py`), le funzioni di analisi (`blocco1_analisi.py`) e il campione (`flight_sample_30000.csv`).

---

## 1. Perché questa variante

L'analisi di rilevanza del progetto principale (`analisi_rilevanza_fd.py`) ha mostrato che, per `DurataBucket`, la FD più importante è `Distance → DistanceGroup` (7,9 punti di calo di F1 al 40% di rumore), non `Origin+Dest → Distance` (2,0 punti) usata nel Blocco 1. La domanda è: **cosa succede rifacendo il Blocco 1 con la FD più importante, su un obiettivo predittivo per cui quella FD conta?**

## 2. Scelta dell'obiettivo predittivo

**Screening degli obiettivi candidati** (`selezione_obiettivi.csv`). Per ogni obiettivo, F1 di un Decision Tree addestrato su una sola colonna, con classi bilanciate:

| Obiettivo candidato | Caso puro | `Distance` | `DistanceGroup` | Origin | Dest | Compagnia | CRSDepTime |
|---|---|---|---|---|---|---|---|
| **AirTime, 5 fasce** | 0,20 | **0,779** | **0,689** | 0,343 | 0,340 | 0,293 | 0,243 |
| Compagnia aerea | 0,07 | 0,371 | 0,115 | 0,395 | 0,398 | — | 0,129 |
| Ritardo in arrivo > 15 min | 0,50 | 0,522 | 0,489 | 0,540 | 0,549 | 0,496 | 0,508 |
| Taxi-out, 3 fasce | 0,33 | 0,388 | 0,313 | 0,460 | 0,385 | 0,446 | 0,356 |
| Fascia oraria di arrivo | 0,05 | 0,093 | 0,053 | 0,087 | 0,089 | 0,036 | 0,365 |
| Volo cancellato | 0,50 | 0,538 | 0,531 | 0,594 | 0,625 | 0,564 | 0,486 |

L'informazione sulla distanza è la più predittiva solo per **`AirTime`**, il tempo effettivo in volo (dal decollo all'atterraggio). Per la compagnia contano di più gli aeroporti e `DistanceGroup` quasi nulla; ritardi e cancellazioni non sono predetti da nessuna colonna; la fascia di arrivo dipende dall'orario di partenza.

**Obiettivo `AirTimeBucket`**: `AirTime` in 5 fasce con confini ai quintili del campione (≤58, 59–86, 87–120, 121–164, >164 minuti), fissati nel codice. `AirTime` non viene mai sporcato, quindi le etichette del training e del test coincidono sempre. Le righe senza `AirTime` (3,1%: voli cancellati o dirottati) sono escluse dal training e dalla valutazione, non dal calcolo degli indici di inconsistenza.

**Blacklist anti-leakage**: oltre alle colonne già escluse dal motore (`AirTime`, `ActualElapsedTime`, `WheelsOff/On`, `TaxiIn/Out`, `ArrTime`, `DepTime`), si escludono `CRSElapsedTime`, `CRSArrTime` e `ArrTimeBlk`, come nel progetto principale. `CRSElapsedTime`, la durata schedulata, ha correlazione 0,98 con la distanza: lasciarla sarebbe una via di fuga.

## 3. Verifica della premessa: la FD è la più importante per `AirTime`

Stesso metodo di `analisi_rilevanza_fd.py`: si sporca al 40% una configurazione alla volta, training sporco e test pulito, 3 repliche (`rilevanza_fd_airtime.csv`).

| Configurazione | Colonne sporcate | Calo F1 medio | Modelli con calo significativo |
|---|---|---|---|
| Distanza e rotta con tutte le copie (setup dell'esperimento) | 20 | 7,9 punti | 4/4 |
| **FD `Distance → DistanceGroup`** | 2 | **7,5 punti** | 4/4 |
| FD `Origin+Dest → Distance` | 3 | 1,7 punti | 3/4 |
| Aeroporto di origine con le copie | 7 | 0,7 punti | 3/4 |
| FD `Reporting_Airline → IATA_CODE` | 2 | −0,2 punti | 2/4 |
| FD `CRSDepTime → DepTimeBlk` | 2 | −0,2 punti | 3/4 |

`Distance → DistanceGroup` è la FD più importante anche per `AirTime`, e da sola vale quasi quanto sporcare tutte le 20 colonne dell'esperimento. Sporcare compagnia e orario peggiora poco o nulla: il segno negativo indica un lieve miglioramento, significativo in alcuni modelli ma inferiore a 0,4 punti.

## 4. Disegno sperimentale

Identico al Blocco 1 del progetto principale:

| Parametro | Valore |
|---|---|
| FD dichiarata | `Distance → DistanceGroup` |
| Colonne ridondanti sporcate sulle stesse righe | `Origin`, `Dest` e le 16 colonne sugli aeroporti (ID, SeqID, CityMarketID, città, stato, FIPS, nome dello stato, WAC) |
| Colonne sporcate in totale | 20, **le stesse del Blocco 1 principale** |
| Validità | 0 violazioni per la FD dichiarata, per `Origin+Dest → Distance` e per le FD aeroporto → colonne ridondanti |
| Campione | 30.000 righe; **28.680 dopo il bilanciamento** (Blocco 1 principale: 8.090) |
| Livelli di rumore | 0, 5, 10, 20, 30, 40%; 5 repliche con seed 100–104 |
| Valutazione | cross-validation stratificata a 5 fold, 4 modelli RAW, training sporco e test pulito (e sporco per confronto) |

**Controlli superati.** La quota di righe di training sporcate vale 0 a rumore 0% e poi 0,0501 · 0,1002 · 0,2001 · 0,3001 · 0,4003. A rumore 0% il test sporco coincide con quello pulito.

## 5. Risultati

### 5.1 Degrado

F1 sul test pulito:

| Rumore | IM | IH | Decision Tree | Logistic Regression | Neural Network | Random Forest | Media |
|---|---|---|---|---|---|---|---|
| 0% | 0 | 0 | 0,8419 | 0,8714 | 0,8351 | 0,8662 | 0,8536 |
| 5% | 28.475 | 2.504 | 0,8398 | 0,8081 | 0,8330 | 0,8653 | 0,8366 |
| 10% | 55.661 | 4.909 | 0,8395 | 0,7607 | 0,8317 | 0,8642 | 0,8240 |
| 20% | 105.792 | 9.508 | 0,8361 | 0,6942 | 0,8264 | 0,8612 | 0,8045 |
| 30% | 149.280 | 13.678 | 0,8330 | 0,6424 | 0,8211 | 0,8580 | 0,7886 |
| 40% | 186.614 | 17.498 | 0,8296 | 0,5977 | 0,8145 | 0,8538 | 0,7739 |

Il degrado è monotono per tutti e 4 i modelli e **significativo in 20 confronti su 20**. Al 40% la media perde **8,0 punti**: 1,2 Decision Tree, 1,2 Random Forest, 2,1 Neural Network, **27,4 Logistic Regression**.

Il baseline (0,8536) è più basso di quello di `DurataBucket` (0,9233): il tempo effettivo in volo dipende anche da vento, traffico e rotte di attesa, che non compaiono fra i dati.

### 5.2 Dove si misura

| Rumore | F1 test pulito | F1 test sporco | Perdita da apprendimento | Perdita da input corrotto | Quota da apprendimento |
|---|---|---|---|---|---|
| 5% | 0,8366 | 0,8051 | 0,0171 | 0,0315 | 35,2% |
| 10% | 0,8240 | 0,7638 | 0,0296 | 0,0602 | 33,0% |
| 20% | 0,8045 | 0,6909 | 0,0492 | 0,1136 | 30,2% |
| 30% | 0,7886 | 0,6263 | 0,0650 | 0,1624 | 28,6% |
| 40% | 0,7739 | 0,5657 | 0,0797 | 0,2082 | 27,7% |

Al 40% il test sporco mostra un calo di 28,8 punti, di cui 8,0 dovuti all'apprendimento: misurare sul test sporco sovrastima il danno di **3,6 volte**.

### 5.3 Correlazioni

Spearman fra indici e F1 sul test pulito:

| Modello | IM | IP | IH |
|---|---|---|---|
| Logistic Regression | −0,978 | −0,987 | −0,975 |
| Neural Network | −0,975 | −0,973 | −0,971 |
| Random Forest | −0,964 | −0,968 | −0,966 |
| Decision Tree | −0,940 | −0,946 | −0,948 |

IP si satura: al 40% vale 29.997 su 30.000 righe, e già al 20% 29.365. La correlazione di Pearson di IP con la F1 scende così fra −0,68 e −0,86, mentre quella di IM e IH resta fra −0,95 e −0,99.

## 6. Confronto con il Blocco 1 principale

Nei due esperimenti **le colonne sporcate sono le stesse 20**, sporcate sulle stesse quote di righe. Cambiano l'obiettivo predittivo (`AirTime` al posto di `DurataBucket`) e la FD su cui si misurano gli indici.

| Al 40% di rumore | Blocco 1 principale | Sottoprogetto | Rapporto |
|---|---|---|---|
| Obiettivo · FD dichiarata | DurataBucket · `Origin+Dest → Distance` | AirTime · `Distance → DistanceGroup` | |
| Baseline (media) | 0,9233 | 0,8536 | |
| Calo di F1 (media) | 8,4 punti (9,1% del baseline) | 8,0 punti (9,3% del baseline) | ≈ 1 |
| Decision Tree / Logistic Regression / Neural Network / Random Forest | 3,0 / 22,9 / 4,9 / 2,9 | 1,2 / 27,4 / 2,1 / 1,2 | |
| Sovrastima sul test sporco | 3,7 volte | 3,6 volte | ≈ 1 |
| IM | 2.816 | 186.614 | **×66** |
| IH | 1.421 | 17.498 | **×12** |

**1. Stessa corruzione, danno complessivo quasi identico.** Il calo medio relativo al baseline è del 9,1% e del 9,3%, e la sovrastima sul test sporco è la stessa. Sporcare l'informazione sulla distanza fa lo stesso danno ai due obiettivi, entrambi legati alla durata del volo.

**2. Ma cambia la distribuzione fra modelli.** Con `AirTime` Logistic Regression perde di più (27,4 contro 22,9 punti), mentre Decision Tree, Neural Network e Random Forest perdono meno della metà (in media 1,5 contro 3,6 punti). Non si può attribuire la differenza al solo obiettivo: con `AirTime` le classi sono quasi uguali e i modelli si addestrano su circa 3,5 volte più righe (28.680 contro 8.090), e un training più grande può rendere gli alberi e la rete più robusti al rumore.

**3. Gli indici dipendono dalla FD dichiarata, non dal danno.** Con le stesse 20 colonne sporcate e un danno quasi identico, IM è 66 volte più grande e IH 12 volte più grande dichiarando `Distance → DistanceGroup` al posto di `Origin+Dest → Distance`. Il motivo è la struttura dei gruppi del lato sinistro. I valori distinti di `Distance` sono 1.403, con gruppi fino a 154 righe; le rotte sono 4.624, con gruppi fino a 66 righe. Gruppi più grandi contengono più coppie di righe che condividono il lato sinistro, e quindi più conflitti possibili quando una riga sporcata vi finisce dentro con un lato destro diverso. **Il valore assoluto di IM e IH non è confrontabile fra FD diverse**, anche quando i dati sono corrotti allo stesso modo; resta significativo solo il suo andamento con una FD fissata, dove la correlazione col danno è fra −0,94 e −0,99 in entrambi gli esperimenti.

**Controllo di coerenza.** Gli IM di questo esperimento coincidono con quelli della configurazione a 1 FD del Blocco 2 principale (28.475 al 5%, 105.792 al 20%, 186.614 al 40%), che usa la stessa FD e gli stessi seed: l'iniezione del rumore sulle colonne della FD è identica, e le colonne ridondanti in più non entrano nel calcolo di IM.

## 7. Conclusioni

1. **L'effetto dell'inconsistenza si replica su un secondo obiettivo.** Con la FD più rilevante e tutte le vie di fuga chiuse, il 40% di righe inconsistenti costa 8,0 punti di F1 su `AirTime`, significativo in tutti i 20 confronti, con la stessa sovrastima di circa 3,6 volte se si misura sul test sporco.
2. **Il danno dipende da quale informazione è corrotta, non da quale FD si dichiara.** Le stesse colonne sporcate producono lo stesso danno relativo nei due esperimenti.
3. **Gli indici di inconsistenza dipendono fortemente dalla FD dichiarata.** A parità di corruzione e di danno, IM cambia di 66 volte e IH di 12. Usati per confrontare dataset o insiemi di vincoli diversi, gli indici darebbero indicazioni fuorvianti; usati per seguire una stessa FD al crescere del rumore, tracciano bene il degrado. IH resta il più stabile dei due.
4. **La sensibilità dipende dal modello**: Logistic Regression, che usa la distanza come valore numerico, è di gran lunga la più colpita in entrambi gli esperimenti.

## 8. Limiti

- **Obiettivo vicino a `DurataBucket`**: il tempo effettivo in volo e la durata schedulata sono entrambi determinati soprattutto dalla distanza. Per gli altri obiettivi del dataset questa FD non è rilevante, quindi una replica su un compito di natura diversa non è possibile con questa FD.
- **Dimensioni diverse dopo il bilanciamento** (28.680 righe contro 8.090): il confronto per modello con il Blocco 1 principale mescola l'effetto dell'obiettivo con quello della quantità di dati.
- **Rilevanza misurata a un solo livello** (40%) con 3 repliche.
- Valgono i limiti del progetto principale: un solo mese di dati, rumore casuale uniforme, modelli con iperparametri di default.

## 9. File

| File | Contenuto |
|---|---|
| `config.py` | FD, colonne ridondanti, obiettivo e fasce, blacklist, livelli, seed |
| `selezione_obiettivo.py` | Screening degli obiettivi e rilevanza delle FD per `AirTime` |
| `esperimento.py` | Esperimento: 6 livelli × 5 repliche, esecuzione parallela con checkpoint |
| `analisi.py` | Analisi statistica (funzioni di `blocco1_analisi.py`) |
| `genera_plot.py`, `stile_grafici.py` | Grafici e tabella |
| `selezione_obiettivi.csv`, `rilevanza_colonne_airtime.csv`, `rilevanza_fd_airtime.csv`, `rilevanza_fd_airtime_raw.csv`, `selezione_obiettivo.log` | Risultati della selezione |
| `risultati_raw.csv`, `run.log`, `aggregato.csv`, `scomposizione.csv`, `correlazioni.csv`, `test_degrado.csv`, `analisi.log` | Risultati dell'esperimento |
| `plot_f1_test_pulito.png`, `plot_sporco_vs_pulito.png`, `plot_correlazioni.png`, `tabella_riassuntiva.png` | Figure, speculari a quelle del Blocco 1 principale |
| `plot_confronto_calo_f1.png`, `plot_confronto_im_ih.png` | Confronto con il Blocco 1 principale |

---

## In parole più semplici

Nel progetto principale si rovinava la rotta dei voli (partenza, arrivo e distanza) e si guardava quanto peggiorava un programma che indovina la durata del volo. Ma misurando si è scoperto che la regola più importante non era "la rotta decide la distanza", bensì "la distanza decide la sua fascia".

Qui si è rifatto lo stesso esperimento con quella regola, e con una domanda diversa: indovinare **quanto tempo l'aereo resta davvero in volo**, l'unica domanda del dataset per cui la distanza è davvero decisiva.

- **Il danno è quasi lo stesso**: rovinando gli stessi dati sul 40% dei voli, il programma perde circa il 9% della sua bravura in entrambi i casi.
- **Il conteggio delle contraddizioni, invece, cambia moltissimo**: con la nuova regola se ne contano 66 volte di più, pur avendo rovinato esattamente le stesse colonne. Il numero dipende da quale regola si sceglie di controllare, non da quanto sono rovinati i dati.
- **Il programma che fa i conti con i numeri** (Logistic Regression) resta di gran lunga il più fragile quando la distanza è sbagliata.
