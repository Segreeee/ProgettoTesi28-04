"""
Blocco 1 — effetto dell'inconsistenza dei dati sulle predizioni.

Obiettivo predittivo: DurataBucket, 5 fasce di durata schedulata del volo.

Il rumore viola la FD Origin + Dest -> Distance: per le righe scelte vengono
corrotte in modo indipendente tutte le colonne del concetto "rotta", comprese
le colonne ridondanti che codificano la stessa informazione (aeroporti, citta',
stati e fasce di distanza), cosi' che il
modello non possa recuperarla da una colonna rimasta pulita.

Il modello e' addestrato sui dati sporcati (a rumore 0% restano puliti) e
valutato in cross-validation sul TEST PULITO — le stesse righe prese dal
campione originale — e, per confronto, sul test sporcato.

Esecuzione parallela con checkpoint (vedi esecuzione_parallela.py).
Uso:  python blocco1_esperimento.py [--workers N]
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import sys
import argparse
import warnings

import numpy as np
import pandas as pd

from progettoTesi_v2 import (
    ml_preparation,
    get_global_inconsistency_metrics,
    inject_multiple_fd_noise,
)
from esecuzione_parallela import numero_processi, esegui_lavori, Registro

FILE_CAMPIONE = 'flight_sample_30000.csv'

FDS_LIST = [(['Origin', 'Dest'], 'Distance')]

REDUNDANT_COLS_MAP = {
    (('Origin', 'Dest'), 'Distance'): [
        'OriginAirportID', 'DestAirportID',
        'OriginCityName', 'DestCityName',
        'OriginState', 'DestState',
        'OriginStateFips', 'DestStateFips',
        'OriginStateName', 'DestStateName',
        'OriginWac', 'DestWac',
        'OriginCityMarketID', 'DestCityMarketID',
        'OriginAirportSeqID', 'DestAirportSeqID',
        'DistanceGroup',
    ]
}

FD_RIDONDANTI = (
    [(['Origin'], c) for c in REDUNDANT_COLS_MAP[(('Origin', 'Dest'), 'Distance')] if c.startswith('Origin')]
    + [(['Dest'], c) for c in REDUNDANT_COLS_MAP[(('Origin', 'Dest'), 'Distance')] if c.startswith('Dest')]
    + [(['Distance'], 'DistanceGroup')]
)

EXTRA_BLACKLIST = ['CRSElapsedTime', 'CRSArrTime', 'ArrTimeBlk']

TARGET_COL = 'DurataBucket'
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
N_REPS = 5
SEED_BASE = 100

RAW_RESULTS_FILE = 'blocco1_risultati_raw.csv'
LOG_FILE = 'blocco1_run.log'
RAM_PER_PROCESSO_GB = 0.6


def create_durata_bucket(df):
    """
    Deriva il target multiclasse DurataBucket dalla durata schedulata del volo
    (CRSElapsedTime, in minuti), suddivisa in 5 fasce.
    """
    df = df.copy()
    bins = [0, 90, 150, 210, 300, np.inf]
    labels = ['<90m', '90-150m', '150-210m', '210-300m', '>300m']
    df[TARGET_COL] = pd.cut(df['CRSElapsedTime'], bins=bins, labels=labels)
    return df


def carica_campione():
    return create_durata_bucket(pd.read_csv(FILE_CAMPIONE, low_memory=False))


def conta_violazioni(df, fds):
    """Numero di gruppi LHS che violano almeno una FD (0 = dati coerenti)."""
    return int(sum((df.groupby(list(lhs))[rhs].nunique() > 1).sum() for lhs, rhs in fds))


_DF = None


def _inizializza_processo():
    """Ogni processo carica una volta il campione pulito."""
    global _DF
    warnings.filterwarnings('ignore')
    _DF = carica_campione()


def esegui_lavoro(level, rep, n_jobs_rf):
    seed = SEED_BASE + rep
    df_noisy = inject_multiple_fd_noise(
        _DF, FDS_LIST, noise_level=level,
        corrupt_lhs=True, redundant_cols_map=REDUNDANT_COLS_MAP, seed=seed,
    )
    im, ip, ih, _ = get_global_inconsistency_metrics(df_noisy, FDS_LIST)

    ml_scores = ml_preparation(
        df_noisy, TARGET_COL,
        extra_blacklist=EXTRA_BLACKLIST,
        df_eval=_DF,
        n_jobs_rf=n_jobs_rf,
    )

    righe = []
    for model_name, metrics in ml_scores.items():
        row = {
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
    log(f"Blocco 1 — campione {FILE_CAMPIONE}: {len(df):,} righe")
    log(f"Classi del target: {df[TARGET_COL].value_counts().to_dict()}")

    violazioni = conta_violazioni(df, FDS_LIST + FD_RIDONDANTI)
    if violazioni:
        log(f"STOP: il campione pulito ha {violazioni} violazioni della FD o delle colonne ridondanti.")
        sys.exit(1)
    log(f"FD {FDS_LIST} e {len(FD_RIDONDANTI)} colonne ridondanti: 0 violazioni sul campione pulito.")

    n_processi, ram = numero_processi(RAM_PER_PROCESSO_GB, args.workers)
    n_jobs_rf = max(1, (os.cpu_count() or 1) // n_processi)
    log(f"RAM disponibile {ram:.1f} GB -> {n_processi} processi paralleli, Random Forest con {n_jobs_rf} thread")
    log(f"Livelli di rumore: {NOISE_LEVELS} | Repliche: {N_REPS}")

    lavori = [(level, rep, n_jobs_rf) for level in NOISE_LEVELS for rep in range(N_REPS)]
    esegui_lavori(esegui_lavoro, lavori, lambda a: (int(round(a[0] * 100)), a[1]),
                  _inizializza_processo, RAW_RESULTS_FILE, ['Rumore_%', 'Rep'], n_processi, log)

    risultati = pd.read_csv(RAW_RESULTS_FILE)
    base = risultati[risultati['Rumore_%'] == 0]
    acc = base.groupby('Modello')['Accuracy_test_pulito'].mean()
    log("Baseline (rumore 0%), accuracy sul test pulito: " + ", ".join(f"{m} {v:.4f}" for m, v in acc.items()))
    if (acc < 0.5).any():
        log("ATTENZIONE: baseline vicino al caso puro (0.20 per 5 classi): non interpretare l'effetto del rumore.")

    diff = (base['Accuracy_test_sporco'] - base['Accuracy_test_pulito']).abs().max()
    quota0 = base['Quota_train_sporca'].max()
    log(f"Sanity check (rumore 0%): max |test_sporco - test_pulito| = {diff:.6f}, quota training sporca = {quota0}")
    if diff > 1e-9 or quota0 != 0:
        log("STOP: a rumore 0% le due valutazioni devono coincidere e il training essere pulito.")
        sys.exit(1)
    log(f"FINE. Righe nel CSV: {len(risultati)} (attese {len(NOISE_LEVELS) * N_REPS * 4})")


if __name__ == "__main__":
    main()
