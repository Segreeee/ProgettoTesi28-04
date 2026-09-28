"""
IH esatto nel Blocco 2, accanto al valore 2-approssimato degli esperimenti.

Con una sola FD il minimo esatto ha forma chiusa (ih_esatto_una_fd) ed e' gia'
nei risultati del Blocco 2. Con 2 e 4 FD i grafi delle diverse FD si
sovrappongono e il minimo va calcolato risolvendo il problema di vertex cover
con la programmazione lineare intera (ih_esatto_ilp di verifica_ih.py).

Le istanze sono le stesse degli esperimenti: training sporcato con il seed
della replica, righe di training di ciascun fold (6.472). Il problema e'
NP-difficile e il tempo del risolutore cresce in modo irregolare con il
rumore: ogni istanza gira in un processo separato con un limite di tempo, e
quelle non risolte entro il limite sono registrate come tali.

Livelli calcolati (dalla prova dei tempi su un fold): 2 FD al 5, 10, 20 e 40%, 4 FD
al 5 e 10%. Il 30% con 2 FD e i livelli dal 20% con 4 FD non chiudono entro tre
minuti.

Output: blocco2_ih_esatto.csv (per replica e fold) e riepilogo a video.
Uso:  python ih_esatto_blocco2.py [--tempo-massimo 120]
"""
import argparse
import multiprocessing as mp
import time
import warnings

import numpy as np
import pandas as pd

RISULTATI_FILE = 'blocco2_ih_esatto.csv'
# Il 30% con 2 FD e i livelli oltre il 10% con 4 FD non chiudono entro qualche
# minuto nella prova dei tempi (un fold): sono esclusi e dichiarati come non risolti.
CONFIGURAZIONI = {2: [5, 10, 20, 40], 4: [5, 10]}
N_REPS = 5


def _risolvi(n_fd, livello, rep, fold, coda):
    warnings.filterwarnings('ignore')
    from progettoTesi_v2 import inject_multiple_fd_noise, fold_di_valutazione, get_global_inconsistency_metrics
    from blocco1_esperimento import carica_campione, EXTRA_BLACKLIST, TARGET_COL
    from blocco2_scaling_fd import FD_POOL, REDUNDANT_COLS_MAP, SEED_BASE
    from verifica_ih import ih_esatto_ilp

    seed = SEED_BASE + rep
    campione = carica_campione()
    righe = fold_di_valutazione(campione, TARGET_COL, EXTRA_BLACKLIST, seed_valutazione=seed)[fold]
    sporco = inject_multiple_fd_noise(campione, FD_POOL[:n_fd], livello / 100, corrupt_lhs=True,
                                      redundant_cols_map=REDUNDANT_COLS_MAP, seed=seed).loc[righe]
    im, ip, ih_approx, grafo = get_global_inconsistency_metrics(sporco, FD_POOL[:n_fd])
    esatto, secondi, risolto = ih_esatto_ilp(grafo, tempo_massimo=10 ** 6)
    coda.put({'IM': im, 'IP': ip, 'IH_approx': ih_approx, 'IH_esatto': esatto,
              'Secondi_ilp': round(secondi, 1), 'Risolto': risolto})


NON_RISOLTO = {'IM': np.nan, 'IP': np.nan, 'IH_approx': np.nan, 'IH_esatto': np.nan,
               'Secondi_ilp': np.nan, 'Risolto': False}


def risolvi_in_parallelo(istanze, tempo_massimo, processi):
    """Esegue le istanze a gruppi di `processi`, ciascuna in un processo
    separato; quelle che superano il limite di tempo vengono terminate."""
    esiti = []
    for inizio in range(0, len(istanze), processi):
        gruppo = istanze[inizio:inizio + processi]
        attivi = []
        for istanza in gruppo:
            coda = mp.Queue()
            processo = mp.Process(target=_risolvi, args=(*istanza, coda))
            processo.start()
            attivi.append((istanza, processo, coda))
        scadenza = time.time() + tempo_massimo
        for istanza, processo, coda in attivi:
            processo.join(max(0.0, scadenza - time.time()))
            if processo.is_alive():
                processo.terminate()
                processo.join()
                esito = dict(NON_RISOLTO, Secondi_ilp=tempo_massimo)
            else:
                esito = coda.get() if not coda.empty() else dict(NON_RISOLTO)
            n_fd, livello, rep, fold = istanza
            esiti.append({'N_FD': n_fd, 'Rumore_%': livello, 'Rep': rep, 'Fold': fold, **esito})
            print(f"  {n_fd} FD, {livello}%, replica {rep}, fold {fold}: "
                  f"approx {esito['IH_approx']}, esatto {esito['IH_esatto']}, "
                  f"{esito['Secondi_ilp']} s, risolto {esito['Risolto']}", flush=True)
        pd.DataFrame(esiti).to_csv(RISULTATI_FILE, index=False)
    return esiti


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tempo-massimo', type=int, default=120)
    parser.add_argument('--processi', type=int, default=4)
    args = parser.parse_args()
    pd.set_option('display.width', 200)

    istanze = [(n_fd, livello, rep, fold)
               for n_fd, livelli in CONFIGURAZIONI.items() for livello in livelli
               for rep in range(N_REPS) for fold in range(5)]
    righe = risolvi_in_parallelo(istanze, args.tempo_massimo, args.processi)

    tabella = pd.DataFrame(righe)
    tabella.to_csv(RISULTATI_FILE, index=False)
    riepilogo = (tabella.groupby(['N_FD', 'Rumore_%'])
                 .agg(Istanze=('Risolto', 'size'), Risolte=('Risolto', 'sum'),
                      IH_approx=('IH_approx', 'mean'), IH_esatto=('IH_esatto', 'mean')))
    riepilogo['Rapporto'] = (riepilogo['IH_approx'] / riepilogo['IH_esatto']).round(3)
    print(f"\nSalvato in '{RISULTATI_FILE}'.")
    print("Medie sulle istanze risolte (IH_approx sulle stesse istanze):")
    print(riepilogo.round(1).to_string())


if __name__ == "__main__":
    main()
