"""
Quanto e' lontano dal minimo il valore di IH riportato negli esperimenti?

IH e' calcolato con min_weighted_vertex_cover di NetworkX, un algoritmo
2-approssimato: restituisce un vertex cover valido, quindi un LIMITE SUPERIORE
del minimo, garantito al piu' doppio dell'ottimo. Questo script misura lo
scarto effettivo su istanze di dimensione ridotta, confrontando tre valori:

  - IH_approx: il 2-approssimato usato negli esperimenti;
  - IH_esatto_formula: per UNA sola FD il grafo e' un'unione disgiunta di
    multipartiti completi, quindi il minimo ha forma chiusa (righe del gruppo
    meno la classe piu' numerosa del lato destro);
  - IH_esatto_ilp: l'ottimo calcolato risolvendo il problema di
    programmazione lineare intera (una variabile binaria per nodo, un vincolo
    per arco), con scipy.optimize.milp.

Uso:  python verifica_ih.py [--righe 250 500 1000] [--livelli 10 20 40]
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import coo_matrix

from progettoTesi_v2 import get_global_inconsistency_metrics, ih_esatto_una_fd, inject_multiple_fd_noise
from blocco1_esperimento import carica_campione
from blocco2_scaling_fd import FD_POOL, REDUNDANT_COLS_MAP

RISULTATI_FILE = 'verifica_ih.csv'
SEED = 100


def ih_esatto_ilp(grafo, tempo_massimo=120):
    """Vertex cover minimo esatto: minimizza la somma delle variabili binarie
    x_v con il vincolo x_u + x_v >= 1 per ogni arco (u, v)."""
    nodi = [n for n in grafo.nodes() if grafo.degree(n) > 0]
    if not nodi:
        return 0, 0.0, True
    posizione = {n: i for i, n in enumerate(nodi)}
    archi = list(grafo.edges())
    righe = np.repeat(np.arange(len(archi)), 2)
    colonne = np.array([[posizione[u], posizione[v]] for u, v in archi]).ravel()
    A = coo_matrix((np.ones(len(righe)), (righe, colonne)), shape=(len(archi), len(nodi)))
    vincoli = LinearConstraint(A, lb=1, ub=np.inf)
    t0 = time.time()
    esito = milp(c=np.ones(len(nodi)), constraints=vincoli, integrality=np.ones(len(nodi)),
                 bounds=(0, 1), options={'time_limit': tempo_massimo})
    durata = time.time() - t0
    if not esito.success:
        return np.nan, durata, False
    return int(round(esito.fun)), durata, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--righe', type=int, nargs='+', default=[250, 500, 1000])
    parser.add_argument('--livelli', type=int, nargs='+', default=[10, 20, 40])
    parser.add_argument('--tempo-massimo', type=int, default=120)
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    pd.set_option('display.width', 220)

    campione = carica_campione()
    righe_risultato = []
    for n_fd in [1, 2, 4]:
        fds = FD_POOL[:n_fd]
        for livello in args.livelli:
            sporco = inject_multiple_fd_noise(campione, fds, noise_level=livello / 100, corrupt_lhs=True,
                                              redundant_cols_map=REDUNDANT_COLS_MAP, seed=SEED)
            for n_righe in args.righe:
                sotto = sporco.sample(n=n_righe, random_state=SEED)
                t0 = time.time()
                im, ip, ih_approx, grafo = get_global_inconsistency_metrics(sotto, fds)
                tempo_approx = time.time() - t0
                ih_ilp, tempo_ilp, risolto = ih_esatto_ilp(grafo, args.tempo_massimo)
                formula = ih_esatto_una_fd(sotto, fds[0][0], fds[0][1]) if n_fd == 1 else np.nan
                riga = {
                    'N_FD': n_fd, 'Rumore_%': livello, 'Righe': n_righe, 'IM': im, 'IP': ip,
                    'IH_approx': ih_approx, 'IH_esatto_ilp': ih_ilp, 'IH_esatto_formula': formula,
                    'Scarto_assoluto': ih_approx - ih_ilp if risolto else np.nan,
                    'Rapporto_approx_su_esatto': round(ih_approx / ih_ilp, 3) if risolto and ih_ilp else np.nan,
                    'Secondi_approx': round(tempo_approx, 2), 'Secondi_ilp': round(tempo_ilp, 2),
                    'ILP_risolto': risolto,
                }
                righe_risultato.append(riga)
                print(f"  {n_fd} FD, rumore {livello}%, {n_righe} righe: IH_approx={ih_approx}, "
                      f"esatto={ih_ilp}, rapporto={riga['Rapporto_approx_su_esatto']}", flush=True)

    tabella = pd.DataFrame(righe_risultato)
    tabella.to_csv(RISULTATI_FILE, index=False)
    print(f"\nSalvato in '{RISULTATI_FILE}'.")
    print(tabella.to_string(index=False))

    risolti = tabella[tabella['ILP_risolto']]
    if len(risolti):
        print(f"\nRapporto approssimato / esatto: da {risolti['Rapporto_approx_su_esatto'].min():.3f} "
              f"a {risolti['Rapporto_approx_su_esatto'].max():.3f} (garanzia teorica: al piu' 2).")
        print(f"Scarto assoluto: da {int(risolti['Scarto_assoluto'].min())} a "
              f"{int(risolti['Scarto_assoluto'].max())} tuple.")
    una_fd = tabella[(tabella['N_FD'] == 1) & tabella['ILP_risolto']]
    if len(una_fd):
        coincide = bool((una_fd['IH_esatto_formula'] == una_fd['IH_esatto_ilp']).all())
        print(f"Con 1 FD la formula chiusa coincide con l'ottimo dell'ILP: {coincide}")


if __name__ == "__main__":
    main()
