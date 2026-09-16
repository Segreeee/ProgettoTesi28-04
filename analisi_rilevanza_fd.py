"""
Quali dipendenze funzionali contano per l'obiettivo predittivo (DurataBucket)?

Parte 1 — Capacita' predittiva di ogni colonna da sola: Decision Tree RAW
addestrato su UNA sola colonna, cross-validation stratificata a 5 fold sulle
righe bilanciate (caso puro: 0,20). Misura quanta informazione sulla durata
porta la colonna, indipendentemente dalle altre.

Parte 2 — Danno nel contesto dell'esperimento: si sporca al 40% una sola
configurazione, si addestrano i 4 modelli RAW sui dati sporchi e si valuta sul
test pulito (stessa pipeline dei Blocchi 1 e 2). Due tipi di configurazione:
  - FD singola: si sporcano solo le colonne della FD. Le copie ridondanti
    della stessa informazione restano pulite: misura il contributo NON
    recuperabile dal resto dei dati.
  - Concetto: si sporcano sulle stesse righe tutte le colonne che codificano
    la stessa informazione (vie di fuga chiuse): misura quanto conta
    l'informazione in se'.

Output: rilevanza_colonne.csv, rilevanza_fd_raw.csv, rilevanza_fd.csv,
rilevanza_fd.log
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from progettoTesi_v2 import ml_preparation, inject_multiple_fd_noise, _prepare_xy
from blocco1_esperimento import carica_campione, conta_violazioni, EXTRA_BLACKLIST, TARGET_COL
from esecuzione_parallela import numero_processi, esegui_lavori, Registro

LIVELLO = 0.40
N_REPS = 3
SEED_BASE = 100

ORIGINE = ['Origin', 'OriginCityName', 'OriginState', 'OriginStateFips', 'OriginStateName', 'OriginWac']
DESTINAZIONE = ['Dest', 'DestCityName', 'DestState', 'DestStateFips', 'DestStateName', 'DestWac']

CONFIGURAZIONI = {
    'FD Origin+Dest -> Distance':            ((['Origin', 'Dest'], 'Distance'), []),
    'FD Distance -> DistanceGroup':          ((['Distance'], 'DistanceGroup'), []),
    'FD CRSDepTime -> DepTimeBlk':           ((['CRSDepTime'], 'DepTimeBlk'), []),
    'FD Reporting_Airline -> IATA_CODE':     ((['Reporting_Airline'], 'IATA_CODE_Reporting_Airline'), []),
    'FD OriginAirportID -> OriginState':     ((['OriginAirportID'], 'OriginState'), []),
    'FD DestAirportID -> DestState':         ((['DestAirportID'], 'DestState'), []),
    'Concetto distanza':                     ((['Distance'], 'DistanceGroup'), []),
    'Concetto aeroporto di origine':         ((['OriginAirportID'], 'Origin'), ORIGINE[1:]),
    'Concetto aeroporto di destinazione':    ((['DestAirportID'], 'Dest'), DESTINAZIONE[1:]),
    'Concetto rotta e distanza':             ((['Origin', 'Dest'], 'Distance'),
                                              ['OriginAirportID', 'DestAirportID', 'DistanceGroup']
                                              + ORIGINE[1:] + DESTINAZIONE[1:]),
}
DUPLICATI = {'Concetto distanza': 'FD Distance -> DistanceGroup'}

COLONNE_UNIVARIATE = [
    'Distance', 'DistanceGroup', 'Origin', 'Dest', 'OriginAirportID', 'DestAirportID',
    'OriginCityName', 'DestCityName', 'OriginState', 'DestState', 'OriginStateFips', 'DestStateFips',
    'OriginStateName', 'DestStateName', 'OriginWac', 'DestWac',
    'CRSDepTime', 'DepTimeBlk', 'Reporting_Airline', 'IATA_CODE_Reporting_Airline',
    'DayofMonth', 'DayOfWeek',
]

RAW_FILE = 'rilevanza_fd_raw.csv'
LOG_FILE = 'rilevanza_fd.log'


def build_univariata(df):
    X, y = _prepare_xy(df, TARGET_COL, EXTRA_BLACKLIST)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    righe = []
    for col in COLONNE_UNIVARIATE:
        numerica = pd.api.types.is_numeric_dtype(X[col]) and col in ('Distance', 'DistanceGroup', 'CRSDepTime')
        if numerica:
            modello = DecisionTreeClassifier(random_state=42)
        else:
            modello = Pipeline([
                ('oh', ColumnTransformer([('c', OneHotEncoder(handle_unknown='ignore'), [col])])),
                ('dt', DecisionTreeClassifier(random_state=42)),
            ])
        f1 = cross_val_score(modello, X[[col]].astype(float if numerica else str), y,
                             cv=skf, scoring='f1_weighted')
        righe.append({'Colonna': col, 'Valori_distinti': int(X[col].nunique()),
                      'F1_da_sola': round(f1.mean(), 4), 'F1_sd_fold': round(f1.std(), 4)})
    return pd.DataFrame(righe).sort_values('F1_da_sola', ascending=False)


_DF = None


def _inizializza_processo():
    global _DF
    warnings.filterwarnings('ignore')
    _DF = carica_campione()


def esegui_lavoro(nome, rep, n_jobs_rf):
    fd, extra = CONFIGURAZIONI[nome]
    seed = SEED_BASE + rep
    df_noisy = inject_multiple_fd_noise(
        _DF, [fd], noise_level=LIVELLO, corrupt_lhs=True,
        redundant_cols_map={(tuple(fd[0]), fd[1]): extra}, seed=seed,
    )
    scores = ml_preparation(df_noisy, TARGET_COL, extra_blacklist=EXTRA_BLACKLIST,
                            df_eval=_DF, n_jobs_rf=n_jobs_rf)
    righe = []
    for modello, m in scores.items():
        righe.append({'Configurazione': nome, 'Rep': rep, 'Seed': seed, 'Modello': modello,
                      'Colonne_sporcate': len(fd[0]) + 1 + len(extra),
                      'F1_test_pulito': round(m['F1_Score_test_pulito'], 4),
                      'F1_test_sporco': round(m['F1_Score_test_sporco'], 4),
                      'Quota_train_sporca': round(m['Quota_train_sporca'], 4)})
    return righe


def build_sintesi(raw, baseline):
    righe = []
    for (nome, modello), g in raw.groupby(['Configurazione', 'Modello']):
        base = baseline[modello]
        calo = base - g['F1_test_pulito']
        t, p = stats.ttest_1samp(g['F1_test_pulito'], base) if g['F1_test_pulito'].std() > 0 else (np.nan, np.nan)
        righe.append({'Configurazione': nome, 'Modello': modello, 'Colonne_sporcate': g['Colonne_sporcate'].iloc[0],
                      'F1_baseline': base, 'F1_test_pulito': round(g['F1_test_pulito'].mean(), 4),
                      'Calo_F1': round(calo.mean(), 4), 'p_value': p,
                      'Significativo': bool(p < 0.05) if not np.isnan(p) else False,
                      'Quota_train_sporca': round(g['Quota_train_sporca'].mean(), 4)})
    per_modello = pd.DataFrame(righe)
    media = (per_modello.groupby('Configurazione')
             .agg(Colonne_sporcate=('Colonne_sporcate', 'first'),
                  F1_test_pulito=('F1_test_pulito', 'mean'),
                  Calo_F1_medio=('Calo_F1', 'mean'),
                  Calo_F1_min=('Calo_F1', 'min'),
                  Calo_F1_max=('Calo_F1', 'max'),
                  Modelli_significativi=('Significativo', 'sum'))
             .round(4).sort_values('Calo_F1_medio', ascending=False).reset_index())
    return per_modello, media


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    pd.set_option('display.width', 220)
    log = Registro(LOG_FILE)

    df = carica_campione()
    log("=" * 60)
    log(f"Rilevanza delle FD per {TARGET_COL} — {len(df):,} righe, rumore {LIVELLO:.0%}, {N_REPS} repliche")

    fds = [fd for fd, _ in CONFIGURAZIONI.values()]
    fds += [(['OriginAirportID'], c) for c in ORIGINE[1:]] + [(['DestAirportID'], c) for c in DESTINAZIONE[1:]]
    violazioni = conta_violazioni(df, fds)
    if violazioni:
        log(f"STOP: {violazioni} violazioni sul campione pulito.")
        sys.exit(1)
    log("Tutte le FD e le colonne ridondanti usate hanno 0 violazioni sul campione pulito.")

    log("\nParte 1 — F1 di un Decision Tree addestrato su una sola colonna (caso puro 0,20)")
    uni = build_univariata(df)
    uni.to_csv('rilevanza_colonne.csv', index=False)
    log("\n" + uni.to_string(index=False))

    n_processi, ram = numero_processi(1.0, args.workers)
    n_jobs_rf = max(1, (os.cpu_count() or 1) // n_processi)
    log(f"\nParte 2 — RAM {ram:.1f} GB -> {n_processi} processi, Random Forest con {n_jobs_rf} thread")
    nomi = [n for n in CONFIGURAZIONI if n not in DUPLICATI]
    lavori = [(n, rep, n_jobs_rf) for n in nomi for rep in range(N_REPS)]
    esegui_lavori(esegui_lavoro, lavori, lambda a: (a[0], a[1]), _inizializza_processo,
                  RAW_FILE, ['Configurazione', 'Rep'], n_processi, log)

    raw = pd.read_csv(RAW_FILE)
    for dup, orig in DUPLICATI.items():
        raw = pd.concat([raw, raw[raw['Configurazione'] == orig].assign(Configurazione=dup)])

    q = raw.groupby('Configurazione')['Quota_train_sporca'].mean()
    log("\nQuota di righe di training sporcate: " + ", ".join(f"{k} {v:.3f}" for k, v in q.items()))
    if ((q - LIVELLO).abs() > 0.01).any():
        log("STOP: la quota di righe sporcate non corrisponde al livello di rumore.")
        sys.exit(1)

    b = pd.read_csv('blocco1_risultati_raw.csv')
    baseline = b[b['Rumore_%'] == 0].groupby('Modello')['F1_Score_test_pulito'].mean().round(4).to_dict()
    per_modello, media = build_sintesi(raw, baseline)
    per_modello.to_csv('rilevanza_fd.csv', index=False)
    log("\nCalo di F1 sul test pulito al 40% (media 4 modelli)")
    log("\n" + media.to_string(index=False))
    log("\nPer modello")
    log("\n" + per_modello.pivot(index='Configurazione', columns='Modello', values='Calo_F1').to_string())


if __name__ == "__main__":
    main()
