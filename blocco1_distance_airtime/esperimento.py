"""
Blocco 1 con la FD Distance -> DistanceGroup e obiettivo AirTimeBucket.

Stessa metodologia di blocco1_esperimento.py del progetto principale: per le
righe scelte vengono corrotte in modo indipendente le colonne della FD e tutte
le colonne ridondanti che codificano la stessa informazione (vie di fuga
chiuse). Il modello e' addestrato sui dati sporcati e valutato in
cross-validation sul TEST PULITO (stesse righe prese dal campione originale)
e, per confronto, sul test sporcato.

Esecuzione parallela con checkpoint (esecuzione_parallela.py del progetto).
Uso:  python esperimento.py [--workers N]
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import sys
import warnings

import pandas as pd

import config as C
from progettoTesi_v2 import ml_preparation, get_global_inconsistency_metrics, inject_multiple_fd_noise
from esecuzione_parallela import numero_processi, esegui_lavori, Registro

RAW_RESULTS_FILE = C.percorso('risultati_raw.csv')
LOG_FILE = C.percorso('run.log')
RAM_PER_PROCESSO_GB = 1.0

_DF = None


def _inizializza_processo():
    global _DF
    warnings.filterwarnings('ignore')
    _DF = C.carica_campione()


def esegui_lavoro(level, rep, n_jobs_rf):
    seed = C.SEED_BASE + rep
    df_noisy = inject_multiple_fd_noise(
        _DF, C.FDS_LIST, noise_level=level,
        corrupt_lhs=True, redundant_cols_map=C.REDUNDANT_COLS_MAP, seed=seed,
    )
    im, ip, ih, _ = get_global_inconsistency_metrics(df_noisy, C.FDS_LIST)

    # Training sulle righe sporcate, test sulle STESSE righe del campione pulito.
    ml_scores = ml_preparation(
        df_noisy, C.TARGET_COL,
        extra_blacklist=C.EXTRA_BLACKLIST,
        df_eval=_DF,
        n_jobs_rf=n_jobs_rf,
    )

    righe = []
    for model_name, metrics in ml_scores.items():
        row = {'Rumore_%': int(round(level * 100)), 'Rep': rep, 'Seed': seed, 'Modello': model_name,
               'IM': im, 'IP': ip, 'IH': ih}
        row.update({k: round(v, 4) for k, v in metrics.items()})
        righe.append(row)
    return righe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    log = Registro(LOG_FILE)

    df = C.carica_campione()
    log("=" * 60)
    log(f"Blocco 1 (Distance -> DistanceGroup, obiettivo {C.TARGET_COL}) — {len(df):,} righe")
    log(f"Classi del target: {df[C.TARGET_COL].value_counts().sort_index().to_dict()}")

    violazioni = C.conta_violazioni(df, C.FDS_LIST + C.FD_RIDONDANTI)
    if violazioni:
        log(f"STOP: il campione pulito ha {violazioni} violazioni della FD o delle colonne ridondanti.")
        sys.exit(1)
    n_extra = len(C.REDUNDANT_COLS_MAP[(('Distance',), 'DistanceGroup')])
    log(f"FD {C.FDS_LIST} e {n_extra} colonne ridondanti: 0 violazioni sul campione pulito.")

    n_processi, ram = numero_processi(RAM_PER_PROCESSO_GB, args.workers)
    n_jobs_rf = max(1, (os.cpu_count() or 1) // n_processi)
    log(f"RAM disponibile {ram:.1f} GB -> {n_processi} processi paralleli, Random Forest con {n_jobs_rf} thread")
    log(f"Livelli di rumore: {C.NOISE_LEVELS} | Repliche: {C.N_REPS}")

    lavori = [(level, rep, n_jobs_rf) for level in C.NOISE_LEVELS for rep in range(C.N_REPS)]
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
    log(f"FINE. Righe nel CSV: {len(risultati)} (attese {len(C.NOISE_LEVELS) * C.N_REPS * 4})")


if __name__ == "__main__":
    main()
