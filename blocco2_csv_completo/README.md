# Blocco 2 su un campione del 45% del CSV originale

Progetto autonomo che sviluppa la roadmap del **Blocco 2** — *addestramento su dati sporchi, test su dati puliti* — su **242.887 voli** (45% del CSV originale, stratificato per classe di durata) invece che sul campione da 10.000 righe.

Non modifica nulla del progetto principale: legge soltanto `../On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_1.csv` e scrive esclusivamente in questa cartella. Non importa codice dalla cartella superiore.

La relazione sul lavoro svolto è in **[`relazione.md`](relazione.md)**.

## L'esperimento in breve

- Il dataset pulito viene campionato, bilanciato **una volta sola** e diviso in **3 fold**.
- Per ogni fold, il **test resta pulito**; il **training viene sporcato** a livelli crescenti (0%, 5%, …, 40% delle righe; a 0% resta pulito).
- Il rumore viola un numero crescente di dipendenze funzionali, in modo cumulativo: **C1** = FD1, **C2** = +FD2, **C3** = +FD3, **C4** = +FD4.
- L'inconsistenza (IM, IP, IH) è misurata sul training sporcato effettivo; 4 modelli RAW vengono addestrati sul training sporcato e valutati sul test pulito.

| FD | Dipendenza | Significato |
|---|---|---|
| FD1 | `Origin + Dest → Distance` | una rotta ha una distanza fissa |
| FD2 | `Distance → DistanceGroup` | la fascia di 250 miglia deriva dalla distanza |
| FD3 | `CRSDepTime → DepTimeBlk` | la fascia oraria deriva dall'orario |
| FD4 | `Origin → OriginWac` | un aeroporto sta in una sola area geografica |

## Come si esegue

Dalla cartella `blocco2_csv_completo/`, con l'ambiente del progetto:

```
../.venv/Scripts/python.exe verifica_fd.py      # certifica 0 violazioni e crea il dataset bilanciato
../.venv/Scripts/python.exe test_motore.py      # controlli automatici del motore
../.venv/Scripts/python.exe esperimento.py      # esperimento (parallelo, con checkpoint)
../.venv/Scripts/python.exe analisi.py          # analisi statistica
../.venv/Scripts/python.exe grafici.py          # grafici e tabella
```

`esperimento.py` sceglie il numero di processi in base alla RAM libera (massimo 4; `--workers N` per forzarlo). Se viene interrotto, rilanciandolo riprende dai lavori mancanti. Prima esegue un pilota a rumore 0%: si ferma se il baseline non è ben sopra il caso puro o se la proiezione dei tempi supera 10 ore (`--ignora-soglia` per proseguire comunque).

## Garanzia: training sporcato, test pulito

1. **Struttura**: il test è estratto dal dataset pulito e non viene mai passato alla funzione di sporcatura.
2. **Assert a ogni lavoro**: righe di training sporcate = ⌊livello × righe⌋ (0 a rumore 0%), test identico al dataset pulito e senza violazioni. Il programma si ferma se uno fallisce.
3. **Verifica a posteriori**: le colonne `Righe_train_sporcate` e `Righe_test_modificate` del CSV dei risultati sono misurate contro l'impronta del dataset pulito; `analisi.py` si rifiuta di procedere se anche una riga non torna.

## File

| File | Contenuto |
|---|---|
| `config.py` | Tutti i parametri |
| `motore.py` | Preparazione del dataset, sporcatura, misure di inconsistenza, addestramento |
| `verifica_fd.py` → `blocco2_verifica_fd.csv` | Certificazione delle FD |
| `test_motore.py` | Controlli automatici |
| `esperimento.py` → `blocco2_risultati_raw.csv`, `esperimento.log` | Esperimento |
| `analisi.py` → `blocco2_aggregato.csv`, `blocco2_correlazioni.csv`, `blocco2_test_configurazioni.csv`, `blocco2_soglie.csv` | Analisi |
| `grafici.py` → `plot_*.png`, `tabella_riassuntiva.csv/.png` | Grafici e tabella |
| `dataset_pulito_bilanciato.csv.gz` | Il dataset pulito, bilanciato e con i fold assegnati, usato da tutti gli script |
| `relazione.md` | Relazione sul lavoro |
