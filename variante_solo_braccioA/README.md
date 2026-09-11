# Variante "solo braccio A"

Ramo alternativo del progetto di tesi. Considera **unicamente il braccio A** dell'esperimento — la condizione in cui il rumore viola le dipendenze funzionali — escludendo deliberatamente il braccio di controllo.

## Cos'è

Una versione più semplice della tesi: *si sporca progressivamente il training set, si misura l'inconsistenza risultante, si valuta quanto peggiorano le predizioni su dati puliti.* Un'unica condizione sperimentale, una narrazione lineare.

Tutte le premesse metodologiche del progetto principale sono mantenute — modelli RAW, blacklist anti-leakage, bilanciamento delle classi, dipendenze funzionali reali e verificate nel dominio, chiusura delle vie di fuga, seed fissi, training sporcato con test su dati puliti, cross-validation a 5 fold, rumore fino al 40%, scaling 1/5/10 FD.

## In cosa differisce dal progetto principale

| | Progetto principale | Questa variante |
|---|---|---|
| Bracci sperimentali | A (incoerente) + C (coerente, controllo) | solo A |
| Claim | l'incoerenza in quanto tale ha un effetto attribuibile | la corruzione dei dati degrada il modello |
| Attribuzione causale | possibile (il controllo isola la variabile) | **non possibile** |
| Complessità | maggiore, ma conclusione controintuitiva e più forte | minore, conclusione lineare |

**Il limite da tenere presente**: senza braccio di controllo, la correlazione fra inconsistenza misurata e degrado è in larga parte una correlazione fra *quanto si è corrotto* e *quanto peggiora*. Questi dati mostrano che corrompere i dati fa danno, non che sia l'**incoerenza in quanto tale** a farlo.

Cosa viene qui deliberatamente omesso, e il suo esito, è documentato in **[`../relazione_braccio_C.md`](../relazione_braccio_C.md)** — vale la pena leggerlo prima di scegliere quale delle due versioni portare in tesi, perché il braccio di controllo ha prodotto un risultato che ribalta l'interpretazione intuitiva.

## Contenuto

| File | Cosa contiene |
|---|---|
| `relazione_solo_braccioA.md` | **La relazione della variante**: domanda, premesse, FD usate e scartate, risultati, conclusione, limiti, spiegazione in parole semplici |
| `analisi_solo_braccioA.py` | Analisi e grafici; legge i risultati grezzi dalla cartella superiore |
| `plot_A_f1_per_modello.png` | F1 sul test pulito in funzione del rumore, per ciascuno dei 4 modelli |
| `plot_A_sporco_vs_pulito.png` | Confronto fra valutazione su dati sporchi e su dati puliti |
| `plot_A_scaling_fd.png` | Effetto del numero di FD corrotte (1 / 5 / 10) su F1 e su IM |
| `A_tabella_blocco1.csv` / `.png` | Inconsistenza e F1 per livello di rumore |
| `A_tabella_blocco2.csv` / `.png` | F1 e IM per numero di FD corrotte |
| `A_correlazioni.csv` | Correlazione IM/IH vs F1 per modello |
| `A_correlazioni_scaling.csv` | Le stesse correlazioni sull'intero range e nel regime ≥20% di rumore |

## Come rigenerare gli output

```
cd variante_solo_braccioA
../.venv/Scripts/python.exe analisi_solo_braccioA.py
```

**Nessun esperimento viene rieseguito.** Lo script legge i risultati grezzi già prodotti dal progetto principale (`../blocco1_risultati_raw.csv` filtrato sul braccio A, e `../blocco2_risultati_raw.csv` che è già solo braccio A) e scrive i propri output esclusivamente in questa cartella: il progetto principale non viene toccato.
