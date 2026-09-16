"""
Blocco 2 — quante FD? Scaling 1 -> 2 -> 4.

Stesso disegno del Blocco 1 (training sporcato, test pulito in
cross-validation), ma corrompendo un numero crescente di dipendenze
funzionali: 1, 2 e 4.

Si usano solo FD RILEVANTI per l'obiettivo predittivo, individuate da
analisi_rilevanza_fd.py: sporcate al 40%, fanno calare la F1 sul test pulito
in modo significativo per tutti e 4 i modelli. Le altre FD valide del dataset
(stato, citta', WAC dell'aeroporto presi singolarmente, compagnia, orario)
hanno un effetto nullo o non significativo e sono escluse. L'ordine segue la
rilevanza misurata.

Le FD sugli aeroporti sporcano anche le colonne che ripetono la stessa
informazione (citta', stato, FIPS, nome dello stato, WAC): senza, il modello
le userebbe come via di fuga e la FD non avrebbe effetto.

Esecuzione parallela con checkpoint (vedi esecuzione_parallela.py).
Uso:  python blocco2_scaling_fd.py [--workers N]
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import sys
import argparse
import warnings

import pandas as pd

from progettoTesi_v2 import (
    ml_preparation,
    get_global_inconsistency_metrics,
    inject_multiple_fd_noise,
)
from blocco1_esperimento import (
    carica_campione, conta_violazioni, EXTRA_BLACKLIST, TARGET_COL, FILE_CAMPIONE,
)
from esecuzione_parallela import numero_processi, esegui_lavori, Registro

FD_POOL = [
    (['Distance'], 'DistanceGroup'),
    (['Origin', 'Dest'], 'Distance'),
    (['OriginAirportID'], 'Origin'),
    (['DestAirportID'], 'Dest'),
]

SIGNIFICATO = [
    "la fascia di distanza e' definita dalla distanza",
    "la distanza fra due aeroporti e' una quantita' fisica fissa",
    "l'ID identifica il codice IATA dell'aeroporto; citta' e stato ne seguono",
    "l'ID identifica il codice IATA dell'aeroporto; citta' e stato ne seguono (arrivo)",
]

REDUNDANT_COLS_MAP = {
    (('Origin', 'Dest'), 'Distance'): ['DistanceGroup'],
    (('OriginAirportID',), 'Origin'): ['OriginCityName', 'OriginState', 'OriginStateFips',
                                       'OriginStateName', 'OriginWac'],
    (('DestAirportID',), 'Dest'): ['DestCityName', 'DestState', 'DestStateFips',
                                   'DestStateName', 'DestWac'],
}
FD_RIDONDANTI = [(list(lhs), c) for (lhs, _), cols in REDUNDANT_COLS_MAP.items() for c in cols]

FD_COUNTS = [1, 2, 4]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
N_REPS = 5
SEED_BASE = 100

RAW_RESULTS_FILE = 'blocco2_risultati_raw.csv'
VERIFICA_FD_FILE = 'blocco2_verifica_fd.csv'
LOG_FILE = 'blocco2_run.log'
RAM_PER_PROCESSO_GB = 1.5


def verifica_fd(df, log):
    """
    Controllo preliminare obbligatorio: ogni FD del pool deve avere 0 violazioni
    sul campione effettivamente usato e tutte le sue colonne devono superare la
    blacklist di ml_preparation (altrimenti il rumore inciderebbe su IM/IP/IH
    ma non sulle metriche ML). Il risultato e' salvato in blocco2_verifica_fd.csv.
    """
    cols_to_drop = [
        'ArrDelay', 'ArrDelayMinutes', 'ArrDel15', 'ArrTime', 'ActualElapsedTime',
        'AirTime', 'TaxiIn', 'TaxiOut', 'WheelsOff', 'WheelsOn',
        'DepTime', 'DepDel15', 'ArrivalDelayGroups', 'DepartureDelayGroups',
        'FlightDate', 'Tail_Number', 'Flight_Number_Reporting_Airline',
        'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
        'Cancelled', 'CancellationCode', 'Diverted', 'FirstDepTime', 'TotalAddGTime',
        'LongestAddGTime', 'DepDelayMinutes', 'DelayGroups',
    ] + list(EXTRA_BLACKLIST)
    cols_to_drop += [c for c in df.columns if 'ID' in c and c not in ['OriginAirportID', 'DestAirportID']]
    scartate = set(cols_to_drop)

    righe = []
    for i, ((lhs, rhs), significato) in enumerate(zip(FD_POOL, SIGNIFICATO), 1):
        extra = REDUNDANT_COLS_MAP.get((tuple(lhs), rhs), [])
        fuori = [c for c in lhs + [rhs] + extra if c in scartate]
        righe.append({
            'N': i, 'FD': f"{'+'.join(lhs)} -> {rhs}", 'Significato_nel_mondo_reale': significato,
            'Gruppi': int(df.groupby(lhs).ngroups),
            'Colonne_ridondanti': ', '.join(extra),
            'Violazioni': conta_violazioni(df, [(lhs, rhs)] + [(lhs, c) for c in extra]),
            'Valori_mancanti': int(df[lhs + [rhs] + extra].isna().sum().sum()),
            'Colonne_al_modello': not fuori,
        })
    tabella = pd.DataFrame(righe)
    tabella.to_csv(VERIFICA_FD_FILE, index=False)
    for r in righe:
        log(f"  FD #{r['N']:>2} {r['FD']:<40} gruppi={r['Gruppi']:>5} violazioni={r['Violazioni']} "
            f"NaN={r['Valori_mancanti']} colonne al modello={r['Colonne_al_modello']}")
    problemi = tabella[(tabella['Violazioni'] > 0) | (tabella['Valori_mancanti'] > 0) | (~tabella['Colonne_al_modello'])]
    if len(problemi):
        log(f"STOP: FD non valide: {problemi['FD'].tolist()}")
        sys.exit(1)
    log(f"Verifica superata: le {len(FD_POOL)} FD e le loro colonne ridondanti hanno 0 violazioni "
        f"sul campione. Salvata in {VERIFICA_FD_FILE}")


_DF = None


def _inizializza_processo():
    global _DF
    warnings.filterwarnings('ignore')
    _DF = carica_campione()


def esegui_lavoro(n_fd, level, rep, n_jobs_rf):
    fds = FD_POOL[:n_fd]
    seed = SEED_BASE + rep
    df_noisy = inject_multiple_fd_noise(
        _DF, fds, noise_level=level,
        corrupt_lhs=True, redundant_cols_map=REDUNDANT_COLS_MAP, seed=seed,
    )
    im, ip, ih, _ = get_global_inconsistency_metrics(df_noisy, fds)

    ml_scores = ml_preparation(
        df_noisy, TARGET_COL,
        extra_blacklist=EXTRA_BLACKLIST,
        df_eval=_DF,
        n_jobs_rf=n_jobs_rf,
    )

    righe = []
    for model_name, metrics in ml_scores.items():
        row = {
            'N_FD': n_fd,
            'Rumore_%': int(round(level * 100)),
            'Rep': rep,
            'Seed': seed,
            'Modello': model_name,
            'IM': im,
            'IP': ip,
            'IH': ih,
        }
        row.update({k: round(v, 4) for k, v in metrics.items()})
        righe.append(row)
    return righe



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    log = Registro(LOG_FILE)

    df = carica_campione()
    log("=" * 60)
    log(f"Blocco 2 — campione {FILE_CAMPIONE}: {len(df):,} righe")
    verifica_fd(df, log)

    im0, _, _, _ = get_global_inconsistency_metrics(df, FD_POOL)
    if im0 != 0:
        log(f"STOP: il campione di riferimento non e' pulito (IM={im0}).")
        sys.exit(1)
    log(f"Campione di valutazione sulle {len(FD_POOL)} FD: IM={im0} (pulito, OK)")

    n_processi, ram = numero_processi(RAM_PER_PROCESSO_GB, args.workers)
    n_jobs_rf = max(1, (os.cpu_count() or 1) // n_processi)
    log(f"RAM disponibile {ram:.1f} GB -> {n_processi} processi paralleli, Random Forest con {n_jobs_rf} thread")
    log(f"Configurazioni: {FD_COUNTS} FD | Livelli di rumore: {NOISE_LEVELS} | Repliche: {N_REPS}")

    lavori = [(n_fd, level, rep, n_jobs_rf) for level in NOISE_LEVELS for n_fd in FD_COUNTS for rep in range(N_REPS)]
    esegui_lavori(esegui_lavoro, lavori, lambda a: (a[0], int(round(a[1] * 100)), a[2]),
                  _inizializza_processo, RAW_RESULTS_FILE, ['N_FD', 'Rumore_%', 'Rep'], n_processi, log)

    risultati = pd.read_csv(RAW_RESULTS_FILE)
    log("Sintesi — F1 medio sul test pulito, per numero di FD:\n" +
        risultati.pivot_table(index='Rumore_%', columns='N_FD', values='F1_Score_test_pulito',
                              aggfunc='mean').round(4).to_string())
    log(f"FINE. Righe nel CSV: {len(risultati)} (attese {len(FD_COUNTS) * len(NOISE_LEVELS) * N_REPS * 4})")


if __name__ == "__main__":
    main()
