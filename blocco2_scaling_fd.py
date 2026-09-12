"""
Blocco 2 — quante FD? Scaling 1 -> 5 -> 10.

Stesso disegno del Blocco 1 (training sporcato, test pulito in
cross-validation), ma corrompendo un numero crescente di dipendenze
funzionali: 1, 5 e 10.

Qui NON si usano colonne ridondanti forzate: le colonne che codificano la
stessa informazione compaiono come RHS delle FD aggiuntive, quindi vengono
sporcate solo nella misura in cui la FD che le riguarda entra nel set. E'
questo che rende l'esperimento informativo: con 1 sola FD il modello puo'
ancora aggirare il danno, con 10 le vie di fuga si chiudono progressivamente.

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

# Pool di FD: tutte verificate a 0 violazioni sul campione, valide nel
# dominio reale e con TUTTE le colonne che superano la blacklist (verifica
# automatica all'avvio, salvata in blocco2_verifica_fd.csv). Ordine scelto
# per massimizzare la diversita' di concetto nei primi step: rotta ->
# aeroporto di origine -> aeroporto di destinazione, poi si estende
# all'interno delle famiglie.
FD_POOL = [
    (['Origin', 'Dest'], 'Distance'),                 # 1  rotta -> distanza
    (['OriginAirportID'], 'OriginCityName'),          # 2  aeroporto origine
    (['DestAirportID'], 'DestCityName'),              # 3  aeroporto destinazione
    (['OriginAirportID'], 'OriginState'),             # 4
    (['DestAirportID'], 'DestState'),                 # 5
    (['OriginAirportID'], 'Origin'),                  # 6
    (['DestAirportID'], 'Dest'),                      # 7
    (['OriginAirportID'], 'OriginWac'),               # 8
    (['DestAirportID'], 'DestWac'),                   # 9
    (['OriginAirportID'], 'OriginStateName'),         # 10
]

SIGNIFICATO = [
    "la distanza fra due aeroporti e' una quantita' fisica fissa",
    "un aeroporto sta in una sola citta'",
    "un aeroporto sta in una sola citta' (arrivo)",
    "un aeroporto sta in un solo stato",
    "un aeroporto sta in un solo stato (arrivo)",
    "l'ID identifica il codice IATA dell'aeroporto",
    "l'ID identifica il codice IATA dell'aeroporto (arrivo)",
    "il World Area Code e' determinato dallo stato",
    "il World Area Code e' determinato dallo stato (arrivo)",
    "il nome dello stato e' determinato dallo stato",
]

# ESCLUSA deliberatamente: Reporting_Airline -> IATA_CODE_Reporting_Airline.
# E' valida e sensata nel mondo reale, ma il suo LHS ha solo 14 valori
# distinti: i gruppi risultanti sono enormi e da sola genererebbe molti piu'
# conflitti di tutte le altre nove FD messe insieme, dominando IM/IP/IH e
# rendendoli non confrontabili tra le configurazioni a 1, 5 e 10 FD.

FD_COUNTS = [1, 5, 10]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
N_REPS = 5
SEED_BASE = 100

RAW_RESULTS_FILE = 'blocco2_risultati_raw.csv'
VERIFICA_FD_FILE = 'blocco2_verifica_fd.csv'
LOG_FILE = 'blocco2_run.log'
RAM_PER_PROCESSO_GB = 1.0   # il grafo di networkx arriva a ~0,5 GB nel caso peggiore


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
        fuori = [c for c in lhs + [rhs] if c in scartate]
        righe.append({
            'N': i, 'FD': f"{'+'.join(lhs)} -> {rhs}", 'Significato_nel_mondo_reale': significato,
            'Gruppi': int(df.groupby(lhs).ngroups),
            'Violazioni': conta_violazioni(df, [(lhs, rhs)]),
            'Valori_mancanti': int(df[lhs + [rhs]].isna().sum().sum()),
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
    log(f"Verifica superata: le 10 FD hanno 0 violazioni sul campione. Salvata in {VERIFICA_FD_FILE}")


# ---------------------------------------------------------------
# Lato processo di lavoro
# ---------------------------------------------------------------
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
        corrupt_lhs=True, redundant_cols_map=None, seed=seed,
    )
    # IM/IP/IH misurati sullo STESSO set di FD che viene corrotto.
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


# ---------------------------------------------------------------
# Lato processo principale
# ---------------------------------------------------------------

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
    log(f"Campione di valutazione sulle 10 FD: IM={im0} (pulito, OK)")

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
