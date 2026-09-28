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

**Baseline** a rumore 0%, su dati puliti (F1 weighted), comune ai due blocchi:

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

Livelli di rumore: **0, 5, 10, 20, 30, 40%**; 5 repliche con seed diversi per livello. Il seed di ogni replica è lo stesso nelle configurazioni a 1, 5 e 10 FD, e righe bilanciate e fold sono identici in ogni condizione: le configurazioni si confrontano quindi a coppie.

**Verifica della garanzia.** Ogni risultato riporta la quota di righe di training che differiscono dal campione pulito, misurata riga per riga. Vale **0 a rumore 0%**, dove le due valutazioni coincidono. Nel Blocco 1 coincide con il livello richiesto (0,0495 · 0,0977 · 0,1977 · 0,2982 · 0,3976); nel Blocco 2, poiché ogni FD sceglie le proprie righe, vale 1 − (1 − p)^N, con uno scarto massimo di 0,003 dall'atteso. Le analisi di entrambi i blocchi si interrompono se uno di questi controlli fallisce.

---

# Risultati — Blocco 1: una dipendenza funzionale, vie di fuga chiuse

## 6.1 L'inconsistenza nel training degrada le predizioni

| Rumore | IM | IH | Decision Tree | Logistic Regression | Neural Network | Random Forest | Media |
|---|---|---|---|---|---|---|---|
| 0% | 0 | 0 | 0,9218 | 0,9218 | 0,9152 | 0,9345 | 0,9233 |
| 5% | 411 | 117 | 0,9216 | 0,9047 | 0,9122 | 0,9295 | 0,9170 |
| 10% | 787 | 260 | 0,9200 | 0,9004 | 0,9084 | 0,9285 | 0,9143 |
| 20% | 1.540 | 588 | 0,9121 | 0,8927 | 0,9007 | 0,9238 | 0,9073 |
| 30% | 2.257 | 984 | 0,9083 | 0,8870 | 0,8949 | 0,9178 | 0,9020 |
| 40% | 2.816 | 1.421 | 0,9046 | 0,8821 | 0,8830 | 0,9093 | 0,8948 |

*(F1 sul test pulito)*. Il degrado è **regolare e monotono** su tutti e 4 i modelli. Al 40% di rumore il calo va da **1,7 punti** (Decision Tree) a **4,0** (Logistic Regression), con una media di **2,9 punti**.

## 6.2 Dove si misura cambia quanto danno si vede

| Rumore | F1 test pulito | F1 test sporco | Perdita da apprendimento | Perdita da input corrotto | Quota da apprendimento |
|---|---|---|---|---|---|
| 10% | 0,9143 | 0,8851 | 0,0090 | 0,0292 | 23,6% |
| 20% | 0,9073 | 0,8575 | 0,0160 | 0,0498 | 24,3% |
| 30% | 0,9020 | 0,8369 | 0,0213 | 0,0650 | 24,7% |
| 40% | 0,8948 | 0,8164 | 0,0286 | 0,0784 | 26,7% |

Al 40% il calo misurato sul test sporcato è di **10,7 punti**, ma solo **2,9 sono dovuti a un apprendimento peggiore**. Qui misurare sul test sporco sovrastima il danno di **circa quattro volte**: su ogni riga sporca sono corrotte 19 colonne, e un input così degradato costa molto più dell'apprendimento peggiore.

## 6.3 Correlazioni e significatività

Il calo è statisticamente significativo in **19 confronti su 20** (t-test sulle 5 repliche contro il baseline; al 40% tutti con p < 0,001). Correlazione di Spearman fra inconsistenza e F1 sul test pulito:

| Modello | IM | IP | IH |
|---|---|---|---|
| Logistic Regression | −0,985 | −0,985 | −0,987 |
| Neural Network | −0,970 | −0,971 | −0,974 |
| Random Forest | −0,960 | −0,961 | −0,960 |
| Decision Tree | −0,907 | −0,917 | −0,906 |

Con una sola FD, i tre indici tracciano il degrado allo stesso modo.

---

# Risultati — Blocco 2: 1, 5 e 10 dipendenze funzionali

## 7.1 Il degrado per numero di FD corrotte

F1 sul test pulito, media dei 4 modelli, e quota di righe di training effettivamente sporcate:

| Rumore | F1 · 1 FD | F1 · 5 FD | F1 · 10 FD | Righe sporche · 1 FD | · 5 FD | · 10 FD |
|---|---|---|---|---|---|---|
| 0% | 0,9233 | 0,9233 | 0,9233 | 0,0% | 0,0% | 0,0% |
| 5% | 0,9169 | 0,9151 | 0,9129 | 5,0% | 22,8% | 40,3% |
| 10% | 0,9141 | 0,9104 | 0,9071 | 9,8% | 40,7% | 64,8% |
| 20% | 0,9109 | 0,9039 | 0,8961 | 19,8% | 67,3% | 89,3% |
| 30% | 0,9070 | 0,8966 | 0,8848 | 29,8% | 83,3% | 97,2% |
| 40% | 0,9034 | 0,8900 | 0,8724 | 39,8% | 92,3% | 99,4% |

Il degrado è significativo in **19 confronti su 20 in ciascuna configurazione**. Calo dal baseline al 40%: **0,0199** (1 FD), **0,0333** (5 FD), **0,0510** (10 FD).

## 7.2 Dove si misura cambia quanto danno si vede

| Configurazione (40%) | Perdita da apprendimento | Perdita da input corrotto | Quota da apprendimento |
|---|---|---|---|
| 1 FD | 0,0199 | 0,0179 | 52,6% |
| 5 FD | 0,0333 | 0,0353 | 48,6% |
| 10 FD | 0,0510 | 0,0492 | 50,9% |

A tutti i livelli la quota di danno dovuta all'apprendimento sta fra il **40% e il 53%**: misurare sul test sporco sovrastima il danno di **circa due volte**, contro le quattro del Blocco 1. La sovrastima cresce con il numero di colonne corrotte per riga, che nel Blocco 2 è molto più basso.

## 7.3 Correlazioni: IM smette di tracciare il danno, IH no

Correlazione di Spearman con F1 sul test pulito:

| Configurazione | Indice | Decision Tree | Logistic Regression | Neural Network | Random Forest |
|---|---|---|---|---|---|
| 1 FD | IM | −0,895 | −0,967 | −0,952 | −0,973 |
| 1 FD | IH | −0,895 | −0,972 | −0,955 | −0,973 |
| 5 FD | IM | −0,925 | −0,961 | −0,981 | −0,966 |
| 5 FD | IH | −0,926 | −0,964 | −0,974 | −0,976 |
| 10 FD, tutti i livelli | IM | −0,581 | −0,615 | −0,606 | −0,608 |
| 10 FD, tutti i livelli | IH | −0,937 | −0,972 | −0,974 | −0,972 |
| **10 FD, dal 20% in poi** | **IM** | **+0,868** | **+0,861** | **+0,871** | **+0,846** |
| 10 FD, dal 20% in poi | IH | −0,869 | −0,865 | −0,885 | −0,899 |

Con 1 e 5 FD i due indici si equivalgono. Con 10 FD **IM cambia segno** dopo il picco al 20% di rumore — cala mentre il danno cresce — mentre **IH resta fortemente negativo** a ogni livello.

## 7.4 A parità di livello: più FD, più danno

T-test appaiato per replica fra configurazioni allo stesso livello di rumore: passando da 1 a 5 FD la F1 cala in media di 0,0073 (significativo in 13 confronti su 20), da 5 a 10 FD di 0,0086 (15 su 20).

**Ma questo confronto mescola due cause.** A parità di livello, più FD sporcano molte più righe (al 20%: il 20% con una FD, l'89% con dieci; §7.1). Il calo può dipendere dal numero di FD violate o semplicemente dal numero di righe sporche.

## 7.5 A parità di righe sporche: più FD, **meno** danno

Coppie di configurazioni con quota di righe sporche confrontabile (entro 3,5 punti percentuali):

| Meno FD | Più FD | Righe sporche | F1 meno FD | F1 più FD | Modelli con più FD migliore |
|---|---|---|---|---|---|
| 1 FD al 40% | 5 FD al 10% | 39,8% / 40,7% | 0,9034 | 0,9104 | 4/4 |
| 1 FD al 40% | 10 FD al 5% | 39,8% / 40,3% | 0,9034 | 0,9129 | 4/4 |
| 5 FD al 10% | 10 FD al 5% | 40,7% / 40,3% | 0,9104 | 0,9129 | 4/4 |
| 1 FD al 20% | 5 FD al 5% | 19,8% / 22,8% | 0,9109 | 0,9151 | 4/4 |
| 5 FD al 20% | 10 FD al 10% | 67,3% / 64,8% | 0,9039 | 0,9071 | 3/4 |
| 5 FD al 40% | 10 FD al 20% | 92,3% / 89,3% | 0,8900 | 0,8961 | 4/4 |

In **6 coppie su 6** la configurazione con più FD ha F1 più alta, in 23 confronti su 24 per modello. Per separare formalmente i due effetti, per ogni modello si stima la regressione *F1 ~ quota di righe sporche + quota² + N_FD*:

| Modello | Coefficiente di N_FD (intervallo di quota comune, ≤ 0,41) | t | p |
|---|---|---|---|
| Decision Tree | +0,00124 | 10,03 | < 0,001 |
| Neural Network | +0,00129 | 8,81 | < 0,001 |
| Random Forest | +0,00093 | 8,88 | < 0,001 |
| Logistic Regression | +0,00098 | 4,20 | < 0,001 |

Nell'intervallo di quota in cui tutte e tre le configurazioni hanno dati, **a parità di righe sporche ogni FD in più aumenta la F1**, in modo significativo per tutti e 4 i modelli: passare da 1 a 10 FD vale circa un punto di F1 (da 0,8 a 1,2 punti). Sull'intero intervallo — che richiede di estrapolare, perché con una FD le righe sporche non superano il 40% — il segno resta positivo per tutti i modelli, significativo per Decision Tree e Random Forest.

**Interpretazione.** Il danno non cresce col numero di vincoli violati, ma con **quanta informazione predittiva viene distrutta**. Una sola FD al 40% concentra la corruzione sulla rotta e sulla distanza, l'informazione più legata alla durata del volo, su 4 righe su 10. Dieci FD al 5% toccano altrettante righe, ma distribuiscono la corruzione soprattutto su attributi geografici degli aeroporti, meno predittivi e in larga parte ridondanti fra loro.

## 7.6 Perché IM cala: il meccanismo misurato

Rigenerando i training sporcati con lo stesso seed (verificato: con 1 FD i conflitti ricalcolati coincidono con IM a ogni livello), per 10 FD:

| Rumore | Coppie con lo stesso aeroporto (tetto dei conflitti) | IM | Somma dei conflitti delle 10 FD / IM |
|---|---|---|---|
| 5% | 11.762.082 | 2.106.448 | 2,72 |
| 20% | 4.763.811 | 3.274.677 | 3,53 |
| 40% | 2.977.926 | 2.913.826 | 4,26 |

*(replica 0; il tetto somma le coppie con lo stesso `OriginAirportID` e quelle con lo stesso `DestAirportID`)*

1. **Il tetto crolla.** Sporcando il lato sinistro, le righe si ridistribuiscono in modo uniforme fra gli aeroporti: i gruppi restano 322, ma i più grandi si svuotano (il massimo passa da 1.419 a 199 righe). Le coppie di righe che condividono lo stesso aeroporto — il massimo numero di conflitti possibili — scendono da 16,9 a 3,0 milioni.
2. **I conflitti delle singole FD continuano a crescere.** A ogni livello aumentano: al 40% le nove FD sugli aeroporti coinvolgono fra l'88% e il 98% delle coppie possibili. La stessa coppia di righe però confligge su sempre più FD insieme: il rapporto fra somma dei conflitti e IM sale da 2,7 a 4,3.
3. **IM viene schiacciato sul tetto.** IM conta ogni coppia una volta sola: quando quasi tutte le coppie possibili sono già in conflitto, può solo seguire il tetto verso il basso. Al 40% IM vale 2,91 milioni e il tetto 2,98.

Solo per la FD sulla rotta (lato sinistro a due attributi) i gruppi si frammentano davvero: da 4.624 a 22.077 rotte distinte, perché combinazioni casuali di partenza e arrivo creano rotte nuove.

---

## 8. Risposta alla domanda di ricerca

> **Sì, l'inconsistenza dei dati di training riduce la qualità delle predizioni su dati corretti, in modo regolare e misurabile. Ma l'entità è moderata, dipende da quale informazione viene colpita più che da quanti vincoli vengono violati, e le misure di inconsistenza la descrivono bene solo entro un certo regime.**

**1. L'effetto esiste ed è sistematico.** In entrambi i blocchi il degrado è monotono e significativo in 19 confronti su 20 per ogni configurazione. Con una FD e le vie di fuga chiuse si perdono 2,9 punti di F1 al 40% di rumore, su un baseline di 92,3.

**2. Dove si misura conta.** Valutare sul test sporcato confonde l'apprendimento peggiore con il costo di predire da input corrotti, e sovrastima il danno **da due a quattro volte**, tanto più quante più colonne sono corrotte per riga.

**3. Conta quale informazione si distrugge, non quante FD si violano.** A parità di livello di rumore più FD fanno più danno, ma solo perché sporcano più righe. A parità di righe sporche l'effetto si inverte: ogni FD in più aumenta la F1, in modo significativo per tutti e 4 i modelli. Concentrare la corruzione sull'informazione più predittiva fa più danno che distribuirla su molte dipendenze secondarie.

**4. IM è un indicatore valido solo entro un regime limitato; IH è più affidabile.** Con dieci FD IM raggiunge il massimo al 20% di rumore e poi scende mentre il danno cresce, fino a correlare *positivamente* con la qualità delle predizioni. Il motivo è misurato: la corruzione del determinante svuota i gruppi più grandi, il numero di coppie che possono entrare in conflitto crolla, e IM ne viene limitato. IH, che conta le righe da correggere, resta fortemente legato al degrado in ogni configurazione.

## 9. Limiti

- **Un solo mese di dati** (gennaio 2025): il dataset originale non copre altro periodo.
- **Campione di 30.000 righe**, massimo compatibile con la costruzione in memoria del grafo dei conflitti (il numero di archi cresce circa col quadrato delle righe); dopo il bilanciamento i modelli lavorano su circa 6.470 righe per fold.
- **Rumore casuale uniforme**: gli errori reali tendono a essere sistematici (refusi, codici scambiati, valori vicini al vero).
- **Nel Blocco 2 ogni FD sceglie righe diverse**: il confronto a parità di righe sporche (§7.5) separa i due effetti per via statistica, non per costruzione. Il modello di regressione assume una relazione quadratica fra quota di righe sporche e F1.
- **IM misura solo le FD dichiarate**: le violazioni introdotte sporcando le colonne ridondanti del Blocco 1 non vengono conteggiate.
- **Modelli con iperparametri di default**, come richiesto: un modello ottimizzato potrebbe reagire diversamente.
- **Il disegno non separa incoerenza e perdita di informazione**: corrompere i dati produce sempre entrambe, e un confronto con dati altrettanto sbagliati ma coerenti richiederebbe un gruppo di controllo.

## 10. Materiale allegato

| Blocco 1 | Blocco 2 | Contenuto |
|---|---|---|
| `plot_blocco1_f1_test_pulito.png` | `plot_blocco2_scaling_fd.png` | Degrado al crescere del rumore |
| `plot_blocco1_sporco_vs_pulito.png` | `plot_blocco2_sporco_vs_pulito.png` | Test sporco contro test pulito |
| `plot_blocco1_correlazioni.png` | `plot_blocco2_correlazioni.png` | Correlazioni IM/IH ↔ F1 |
| `tabella_riassuntiva_blocco1.csv` / `.png` | `tabella_riassuntiva_blocco2.csv` / `.png` | Tabella riassuntiva |
| — | `plot_blocco2_quota_sporca.png` | F1 a parità di righe sporche |
| — | `plot_blocco2_im_e_f1.png` | La saturazione di IM |
| `blocco1_aggregato.csv`, `blocco1_scomposizione.csv`, `blocco1_correlazioni.csv`, `blocco1_test_degrado.csv` | `blocco2_aggregato.csv`, `blocco2_scomposizione.csv`, `blocco2_correlazioni.csv`, `blocco2_test_degrado.csv` | Analisi statistica comune |
| — | `blocco2_test_configurazioni.csv`, `blocco2_quota_normalizzata.csv`, `blocco2_quota_punti_confrontabili.csv`, `blocco2_meccanismo_im.csv`, `blocco2_verifica_fd.csv` | Analisi specifiche del Blocco 2 |

---

## In parole più semplici

**La domanda.** Se un programma di intelligenza artificiale impara da dati pieni di errori, ma poi lo si usa su dati corretti, quanto sbaglia in più?

**Come si è misurato.** Si prendono 30.000 voli e si chiede a quattro programmi di indovinare quanto durerà un volo. Con dati corretti ci riescono bene: azzeccano circa 92 volte su 100, contro le 20 che otterrebbero tirando a caso. Poi si rovinano di proposito i dati con cui imparano — la rotta, la distanza, la città, lo stato — su una quota crescente di voli, fino al 40%. Le domande di verifica, invece, si fanno sempre con dati corretti, e il programma controlla a ogni passaggio che sia davvero così.

**Cosa si è scoperto.**

- **Rovinare i dati fa peggiorare le previsioni**, in modo regolare: rovinando per bene la rotta sul 40% dei voli si perdono circa 3 punti su 92.
- **Il modo di misurare conta.** Se anche le domande di verifica si fanno con dati rovinati, il danno sembra da due a quattro volte più grande. Buona parte di quel danno non è "l'IA ha imparato male", ma "le stiamo facendo una domanda scritta male".
- **Non conta quante regole si violano, conta quale informazione si rovina.** A prima vista violare più regole sembra fare più danno. Ma violando più regole si rovinano anche molti più voli. Confrontando a parità di voli rovinati, succede il contrario: rovinare poco su tante regole secondarie fa meno danno che rovinare di più l'unica informazione che conta davvero, cioè la rotta con la sua distanza.
- **Uno strumento di misura si inganna.** Quando si violano dieci regole insieme, il conteggio delle contraddizioni cresce fino al 20% di dati rovinati e poi scende, mentre l'IA continua a peggiorare. Il motivo è che rovinando l'aeroporto i voli si sparpagliano in modo uniforme fra tutti gli aeroporti: i grandi aeroporti si svuotano, e restano molte meno coppie di voli che *possono* contraddirsi. Il conteggio delle righe da correggere, invece, non si fa ingannare e segue il danno fino in fondo.
