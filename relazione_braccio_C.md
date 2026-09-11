# Il braccio C — disegno, correzione e risultato

Relazione dedicata al braccio di controllo dell'esperimento. Documenta a cosa serve, un difetto di disegno individuato e corretto, e il risultato controintuitivo che ne è emerso — risultato che ha ribaltato la conclusione precedente della tesi.

Tutti i numeri citati provengono da `blocco1_risultati_raw.csv`, `blocco1_test_bracci.csv` e `blocco1_correlazioni.csv`.

---

## 1. A cosa serve un braccio di controllo

L'esperimento corrompe i dati di training e misura quanto peggiorano le predizioni. Con il solo braccio sperimentale (braccio A) si osserva che, aumentando il rumore, il modello peggiora. Ma questa osservazione **non basta** a rispondere alla domanda della tesi, perché confonde due cause diverse:

- i dati sono diventati **incoerenti** (violano una dipendenza funzionale), oppure
- i dati hanno semplicemente **perso informazione** (i valori non sono più quelli veri).

Corrompere i dati produce sempre entrambe le cose insieme. Per attribuire il danno specificamente all'incoerenza serve un secondo braccio che produca la **seconda condizione senza la prima**: dati altrettanto sbagliati, ma internamente coerenti. È questo il ruolo del braccio C.

Il criterio di correttezza del confronto è quindi: **i due bracci devono differire solo per la coerenza interna**, e per nient'altro.

## 2. Il disegno originale e il confound individuato

Nella versione iniziale il braccio C selezionava interi gruppi di righe con la stessa rotta e assegnava a tutto il gruppo lo stesso valore alternativo di `Distance`. Il gruppo restava internamente uniforme, quindi IM = 0: formalmente corretto.

Il problema emerge confrontando **quante colonne** ciascun braccio modificava davvero. Verifica eseguita al 40% di rumore, con confronto colonna per colonna che tratta correttamente i valori mancanti:

| Braccio | Colonne realmente modificate |
|---|---|
| **A** | **19** — `Origin`, `Dest`, `Distance` più 16 colonne ridondanti (AirportID, CityName, State, StateName, StateFips, Wac, CityMarketID, AirportSeqID per origine e destinazione) |
| **C** | **1** — solo `Distance` |

Nel braccio C `Origin` e `Dest` restavano **intatti** (verificato). Ed è decisivo, perché l'obiettivo predittivo è la durata del volo: conoscere la rotta permette di stimare la durata anche senza `Distance`. Il braccio C lasciava quindi al modello quasi tutta l'informazione utile, mentre il braccio A gliela distruggeva.

**Il confronto A vs C non isolava dunque la coerenza**, ma la confondeva con "19 colonne distrutte contro 1 colonna perturbata". Qualunque differenza osservata era inattribuibile: poteva derivare dall'incoerenza oppure semplicemente dal fatto che un braccio riceveva molta più informazione dell'altro.

## 3. La correzione

Il braccio C è stato riscritto perché corrompa **le stesse 19 colonne del braccio A**, mantenendo però la coerenza interna. Il meccanismo: per ogni gruppo di righe selezionato (un'intera rotta), si riassegna il **profilo completo di un'altra rotta realmente presente nel dataset** — nuovo `Origin`, nuovo `Dest`, la distanza reale di quella rotta, e tutte le colonne geografiche corrispondenti.

Poiché ogni gruppo riceve una combinazione di valori realmente esistente e resta internamente uniforme, **tutte le dipendenze funzionali restano soddisfatte**: IM = 0 per costruzione, a qualunque livello di rumore. Cambia soltanto *quali* valori ci sono, non la loro coerenza reciproca.

Verifica eseguita su tutti i livelli di rumore:

| Rumore | IM braccio A | Colonne A | Righe A | IM braccio C | Colonne C | Righe C |
|---|---|---|---|---|---|---|
| 0% | 0 | 0 | 0 | 0 | 0 | 0 |
| 5% | 52 | 19 | 500 | **0** | 19 | 500 |
| 10% | 149 | 19 | 1.000 | **0** | 19 | 1.004 |
| 20% | 207 | 19 | 2.000 | **0** | 19 | 2.000 |
| 30% | 315 | 19 | 3.000 | **0** | 19 | 3.000 |
| 40% | 386 | 19 | 4.000 | **0** | 19 | 4.001 |

Colonne identiche, righe allineate, coerenza preservata: il confound è eliminato e il confronto isola ora la sola variabile di interesse.

## 4. La verifica su training e test

Entrambi i bracci addestrano su dati corrotti e vengono valutati su dati **puliti** — requisito introdotto dopo il confronto con il relatore. Verifica diretta sulle matrici effettivamente usate (braccio C, 40% di rumore, primo fold):

| Controllo | Esito |
|---|---|
| Indici ed etichette allineati fra matrice sporca e pulita | ✅ |
| Il training contiene dati corrotti | 843 / 2.172 righe con `Distance` diverso dal vero |
| Il test usa i dati originali puliti | 543 / 543 righe coincidono col dataset pulito |
| Righe di test che erano corrotte nella matrice di training | 206 / 543 |

Quelle 206 righe sono esattamente ciò che il metodo precedente valutava su valori sbagliati. Strutturalmente, la chiamata di valutazione è collocata **fuori** dal ramo che sceglie quale corruzione applicare, quindi è impossibile che i due bracci vengano trattati diversamente.

## 5. Il risultato: il segno si inverte

F1 medio sui 4 modelli, per livello di rumore:

| Rumore | A test sporco | C test sporco | A test pulito | C test pulito | Delta A−C (pulito) |
|---|---|---|---|---|---|
| 0% | 0.8723 | 0.8723 | 0.8723 | 0.8723 | 0.0000 |
| 5% | 0.8477 | 0.8563 | 0.8650 | 0.8624 | +0.0026 |
| 10% | 0.8309 | 0.8448 | 0.8600 | 0.8541 | +0.0059 |
| 20% | 0.7992 | 0.8295 | 0.8518 | 0.8400 | +0.0118 |
| 30% | 0.7783 | 0.8175 | 0.8415 | 0.8276 | +0.0139 |
| 40% | 0.7540 | 0.8088 | 0.8296 | 0.8131 | +0.0165 |

Valutando sul **test sporco** il braccio A appare nettamente peggiore del braccio C (delta medio −0.0294, differenza significativa in 18 confronti su 20). Valutando sul **test pulito** il rapporto si **inverte**: è il braccio C a risultare peggiore (delta medio +0.0102, significativo in 13 confronti su 20, sistematico dal 20-30% in su — 3 modelli su 4 al 20%, 4 su 4 al 30%).

La conclusione precedente della tesi — *"violare la dipendenza funzionale danneggia più che avere valori sbagliati ma coerenti"* — era quindi un **artefatto del dataset di test**.

## 6. Perché il risultato precedente era un artefatto

La scomposizione del danno chiarisce il meccanismo. Il calo rispetto al baseline si divide in due parti: quanto il modello ha **imparato peggio** (visibile sul test pulito) e quanto costa **dargli input corrotti** al momento della predizione (la differenza fra le due valutazioni).

| Braccio (40% rumore) | Perdita da apprendimento | Perdita da input corrotto | Quota da apprendimento |
|---|---|---|---|
| A (incoerente) | 0.0427 | **0.0756** | 36% |
| C (coerente) | **0.0592** | 0.0043 | 93% |

Nel braccio C la penalità da input corrotto è quasi nulla (0.0043) perché **le righe di test erano corrotte nello stesso modo coerente del training**: il modello ritrovava al test esattamente la mappatura sbagliata che aveva imparato, e quindi "ci azzeccava". Era un vantaggio fittizio, che sparisce appena si valuta sui dati veri.

Nel braccio A invece quasi due terzi del danno apparente (0.0756 su 0.1183) non derivavano da un apprendimento peggiore, ma dal chiedere al modello di predire a partire da input incoerenti. Misurando correttamente, il braccio A risulta **meno** danneggiato del braccio C.

## 7. Il meccanismo

Guardando la sola colonna "perdita da apprendimento" — l'unica che misura davvero la qualità del modello appreso — l'errore coerente fa più danno: 0.0592 contro 0.0427 al 40% di rumore.

La spiegazione è che un errore **coerente è indistinguibile da un dato vero**. Una riga che afferma di essere un volo Chicago-Miami, con la distanza corretta di Chicago-Miami e tutte le colonne geografiche coerenti, è a tutti gli effetti un dato plausibile: il modello non ha alcun appiglio per sospettarne e impara da essa una regola falsa, applicandola poi con sicurezza.

Un errore **incoerente** invece si auto-segnala: una riga con `Origin` di un aeroporto, città di un altro e distanza di un terzo è una combinazione implausibile, che i modelli — in particolare alberi e foreste — trattano in parte come rumore e diluiscono, conservando più segnale vero.

Va precisato che questa è un'interpretazione coerente con i dati, non una misura diretta: verificarla richiederebbe di ispezionare cosa i modelli hanno effettivamente appreso.

## 8. La conseguenza per la tesi

Il punto più rilevante non è quale braccio peggiori di più, ma questo: **nel braccio C gli indici valgono IM = IP = IH = 0 a tutti i livelli di rumore**, anche quando il 40% delle righe ha la rotta sbagliata. Le misure di inconsistenza certificano quel dataset come perfettamente pulito. Ed è proprio quel dataset a produrre i modelli peggiori dell'intero esperimento.

Ne segue una delimitazione precisa, sostenuta dai numeri:

> Le misure di inconsistenza basate su dipendenze funzionali intercettano il tipo di sporcizia **meno** dannoso, e sono cieche verso quello **più** dannoso.

Una pipeline data-centric che validasse i dati controllando solo le dipendenze funzionali promuoverebbe a "puliti" proprio i dati che degradano di più il modello, e scarterebbe quelli che lo degradano di meno.

Questo **non invalida** gli indici IM/IP/IH: misurano correttamente ciò che dichiarano di misurare, e nel braccio A la loro correlazione con il degrado è forte (da −0.89 a −0.94, p<0.001 su tutti e 4 i modelli). Ne delimita il campo di validità — che è una conclusione più forte e più difendibile del claim iniziale, perché dice *quando* quel claim vale e *quando no*.

## 9. Limiti

- La differenza A vs C sul test pulito è significativa in **13 confronti su 20**: non è universale. A rumore basso (5%) l'effetto non è distinguibile dal rumore statistico, e la Logistic Regression al 40% resta non significativa (p=0.066).
- L'entità è **moderata**: il delta massimo è +0.0165 di F1, su un baseline di 0.8723.
- Un solo dataset, un solo obiettivo predittivo, un solo mese di dati (gennaio 2025), un solo concetto corrotto (la rotta del volo). Nulla garantisce che le proporzioni osservate valgano altrove.
- Il meccanismo del §7 è un'interpretazione, non una misura.

---

## In parole più semplici

**Perché esiste il braccio C.** Se sporco i dati e il modello peggiora, non so *perché* è peggiorato: può essere perché i dati ora si contraddicono, oppure semplicemente perché ho cancellato informazione utile. Per distinguere le due cose serve un secondo esperimento in cui i dati sono altrettanto sbagliati ma **non si contraddicono**. Quello è il braccio C.

**Cosa non andava.** Il braccio C cambiava una sola colonna (la distanza), mentre il braccio A ne cambiava diciannove — comprese quelle che dicono da dove a dove vola l'aereo. Siccome per indovinare la durata di un volo sapere la rotta è quasi tutto, il braccio C stava giocando con un vantaggio enorme. Il confronto non era alla pari.

**Come è stato sistemato.** Ora il braccio C cambia le stesse diciannove colonne, ma in modo sensato: prende un gruppo di voli e li trasforma *tutti insieme* in un'altra rotta vera, con la sua distanza vera e le sue città vere. I dati sono sbagliati rispetto alla realtà, ma raccontano una storia coerente — e infatti gli indici di contraddizione segnano zero.

**Cosa è venuto fuori.** Con il vecchio metodo di valutazione sembrava che i dati contraddittori facessero più danno. Ma quel confronto era truccato senza volerlo: il braccio C veniva interrogato su dati sbagliati *nello stesso modo* in cui era stato addestrato, quindi rispondeva bene per il motivo sbagliato. Valutando tutti e due su dati veri, il risultato si capovolge: **l'errore coerente fa più danno di quello contraddittorio**.

**Perché ha senso.** Un errore coerente è un bugiardo credibile: il modello non ha modo di accorgersene e impara una regola falsa che poi applica con sicurezza. Un errore contraddittorio invece si tradisce da solo, e il modello tende a scartarlo come rumore.

**La conclusione che conta.** Nel braccio C tutti gli indici di inconsistenza valgono zero — i dati risultano "perfettamente puliti" — eppure sono quelli che producono i modelli peggiori. Quindi gli strumenti che misurano le contraddizioni nei dati vedono benissimo il tipo di sporcizia meno pericoloso, e non vedono affatto quello più pericoloso. Non sono strumenti sbagliati: sono strumenti **parziali**, e sapere dove si fermano è di per sé un risultato utile.
