"""
Selezione dell'obiettivo predittivo e verifica della rilevanza della FD.

Parte 1 — Screening degli obiettivi candidati: per ogni obiettivo, F1 di un
Decision Tree addestrato su UNA sola colonna (classi bilanciate, CV a 5 fold).
Serve a trovare un obiettivo per cui l'informazione della FD
Distance -> DistanceGroup sia la piu' predittiva.

Parte 2 — Rilevanza delle FD per AirTimeBucket, con la stessa pipeline
dell'esperimento (training sporcato al 40%, test pulito, 3 repliche), come
analisi_rilevanza_fd.py del progetto principale.

Output: selezione_obiettivi.csv, rilevanza_colonne_airtime.csv,
rilevanza_fd_airtime_raw.csv, rilevanza_fd_airtime.csv, selezione_obiettivo.log
Uso:  python selezione_obiettivo.py [--workers N]
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

import config as C
from progettoTesi_v2 import ml_preparation, inject_multiple_fd_noise, _prepare_xy
from esecuzione_parallela import numero_processi, esegui_lavori, Registro

LIVELLO = 0.40
N_REPS_RILEVANZA = 3
NUMERICHE = {'Distance', 'DistanceGroup', 'CRSDepTime', 'DayOfWeek', 'DayofMonth'}

COLONNE_SCREENING = ['Distance', 'DistanceGroup', 'Origin', 'Dest', 'OriginState',
                     'Reporting_Airline', 'DepTimeBlk', 'DayOfWeek', 'CRSDepTime']

COLONNE_UNIVARIATE = [
    'Distance', 'DistanceGroup', 'Origin', 'Dest', 'OriginAirportID', 'DestAirportID',
    'OriginCityName', 'DestCityName', 'OriginState', 'DestState', 'OriginWac', 'DestWac',
    'CRSDepTime', 'DepTimeBlk', 'Reporting_Airline', 'DayofMonth', 'DayOfWeek', 'DepDelay',
]

ORIGINE = ['OriginCityName', 'OriginState', 'OriginStateFips', 'OriginStateName', 'OriginWac']

# Nome -> (FD, colonne ridondanti sporcate sulle stesse righe)
CONFIGURAZIONI = {
    'Nessun rumore (baseline)':            ((['Distance'], 'DistanceGroup'), []),
    'FD Distance -> DistanceGroup':        ((['Distance'], 'DistanceGroup'), []),
    'FD Origin+Dest -> Distance':          ((['Origin', 'Dest'], 'Distance'), []),
    'Concetto aeroporto di origine':       ((['OriginAirportID'], 'Origin'), ORIGINE),
    'FD Reporting_Airline -> IATA_CODE':   ((['Reporting_Airline'], 'IATA_CODE_Reporting_Airline'), []),
    'FD CRSDepTime -> DepTimeBlk':         ((['CRSDepTime'], 'DepTimeBlk'), []),
    'Concetto distanza e rotta (esperimento)': (C.FDS_LIST[0], C.REDUNDANT_COLS_MAP[(('Distance',), 'DistanceGroup')]),
}


def _f1_una_colonna(d, col, y, skf):
    if col in NUMERICHE:
        modello, X = DecisionTreeClassifier(random_state=42, min_samples_leaf=5), d[[col]].astype(float)
    else:
        modello = make_pipeline(OneHotEncoder(handle_unknown='ignore'),
                                DecisionTreeClassifier(random_state=42, min_samples_leaf=5))
        X = d[[col]].astype(str)
    return cross_val_score(modello, X, y, cv=skf, scoring='f1_weighted').mean()


def build_screening(df):
    """Obiettivi candidati: classi bilanciate (al massimo 1.500 righe per classe)."""
    candidati = {
        'AirTime (5 fasce, quintili)': df[C.TARGET_COL],
        'Compagnia aerea (Reporting_Airline)': df['Reporting_Airline'],
        'Ritardo in arrivo > 15 min (ArrDel15)': df['ArrDel15'],
        'Taxi-out (3 fasce, terzili)': pd.qcut(df['TaxiOut'], 3, labels=False, duplicates='drop'),
        'Fascia oraria di arrivo (ArrTimeBlk)': df['ArrTimeBlk'],
        'Volo cancellato (Cancelled)': df['Cancelled'],
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    righe = []
    for nome, y in candidati.items():
        d = df.assign(_y=y).dropna(subset=['_y'])
        n = min(int(d['_y'].value_counts().min()), 1500)
        idx = np.concatenate([g.sample(n, random_state=42).index for _, g in d.groupby('_y', observed=True)])
        d = d.loc[idx]
        riga = {'Obiettivo': nome, 'Classi': int(d['_y'].nunique()), 'Righe': len(d),
                'Caso_puro': round(1 / d['_y'].nunique(), 2)}
        for col in COLONNE_SCREENING:
            if col in nome:
                continue
            riga[col] = round(_f1_una_colonna(d, col, d['_y'].astype(str), skf), 3)
        righe.append(riga)
    return pd.DataFrame(righe)


def build_univariata(df):
    X, y = _prepare_xy(df, C.TARGET_COL, C.EXTRA_BLACKLIST)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    righe = []
    for col in COLONNE_UNIVARIATE:
        d = X[[col]].copy()
        if col == 'DepDelay':
            d[col] = d[col].fillna(d[col].median())
            NUMERICHE.add(col)
        righe.append({'Colonna': col, 'Valori_distinti': int(X[col].nunique()),
                      'F1_da_sola': round(_f1_una_colonna(d, col, y, skf), 4)})
    return pd.DataFrame(righe).sort_values('F1_da_sola', ascending=False)


_DF = None


def _inizializza_processo():
    global _DF
    warnings.filterwarnings('ignore')
    _DF = C.carica_campione()


def esegui_lavoro(nome, rep, n_jobs_rf):
    fd, extra = CONFIGURAZIONI[nome]
    livello = 0.0 if nome.startswith('Nessun rumore') else LIVELLO
    seed = C.SEED_BASE + rep
    df_noisy = inject_multiple_fd_noise(_DF, [fd], noise_level=livello, corrupt_lhs=True,
                                        redundant_cols_map={(tuple(fd[0]), fd[1]): extra}, seed=seed)
    scores = ml_preparation(df_noisy, C.TARGET_COL, extra_blacklist=C.EXTRA_BLACKLIST,
                            df_eval=_DF, n_jobs_rf=n_jobs_rf)
    return [{'Configurazione': nome, 'Rep': rep, 'Seed': seed, 'Modello': modello,
             'Colonne_sporcate': 0 if livello == 0 else len(fd[0]) + 1 + len(extra),
             'F1_test_pulito': round(m['F1_Score_test_pulito'], 4),
             'F1_test_sporco': round(m['F1_Score_test_sporco'], 4),
             'Quota_train_sporca': round(m['Quota_train_sporca'], 4)}
            for modello, m in scores.items()]


def build_sintesi(raw):
    baseline = raw[raw['Configurazione'].str.startswith('Nessun rumore')].set_index('Modello')['F1_test_pulito']
    righe = []
    for (nome, modello), g in raw[~raw['Configurazione'].str.startswith('Nessun rumore')].groupby(
            ['Configurazione', 'Modello']):
        base = baseline[modello]
        valori = g['F1_test_pulito']
        t, p = stats.ttest_1samp(valori, base) if valori.std() > 0 else (np.nan, np.nan)
        righe.append({'Configurazione': nome, 'Modello': modello, 'Colonne_sporcate': g['Colonne_sporcate'].iloc[0],
                      'F1_baseline': base, 'F1_test_pulito': round(valori.mean(), 4),
                      'Calo_F1': round(base - valori.mean(), 4), 'p_value': p,
                      'Significativo': bool(p < 0.05) if p == p else False})
    per_modello = pd.DataFrame(righe)
    media = (per_modello.groupby('Configurazione')
             .agg(Colonne_sporcate=('Colonne_sporcate', 'first'), Calo_F1_medio=('Calo_F1', 'mean'),
                  Calo_F1_min=('Calo_F1', 'min'), Calo_F1_max=('Calo_F1', 'max'),
                  Modelli_significativi=('Significativo', 'sum'))
             .round(4).sort_values('Calo_F1_medio', ascending=False).reset_index())
    return per_modello, media


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    pd.set_option('display.width', 220)
    log = Registro(C.percorso('selezione_obiettivo.log'))

    df = C.carica_campione()
    log("=" * 60)
    log(f"Selezione dell'obiettivo — {len(df):,} righe; fasce di AirTime: {C.BINS[1:-1]}")
    log(f"Classi di {C.TARGET_COL}: {df[C.TARGET_COL].value_counts().sort_index().to_dict()}")

    log("\nParte 1 — F1 di un Decision Tree addestrato su una sola colonna, per obiettivo candidato")
    screening = build_screening(df)
    screening.to_csv(C.percorso('selezione_obiettivi.csv'), index=False)
    log("\n" + screening.to_string(index=False))

    uni = build_univariata(df)
    uni.to_csv(C.percorso('rilevanza_colonne_airtime.csv'), index=False)
    log(f"\nColonne per {C.TARGET_COL} (dopo blacklist e bilanciamento):\n" + uni.to_string(index=False))

    violazioni = conta_violazioni_config(df)
    if violazioni:
        log(f"STOP: {violazioni} violazioni sul campione pulito.")
        sys.exit(1)
    log("\nTutte le FD e le colonne ridondanti usate hanno 0 violazioni sul campione pulito.")

    n_processi, ram = numero_processi(1.0, args.workers)
    n_jobs_rf = max(1, (os.cpu_count() or 1) // n_processi)
    log(f"\nParte 2 — RAM {ram:.1f} GB -> {n_processi} processi, Random Forest con {n_jobs_rf} thread")
    lavori = [('Nessun rumore (baseline)', 0, n_jobs_rf)]
    lavori += [(n, rep, n_jobs_rf) for n in CONFIGURAZIONI if not n.startswith('Nessun rumore')
               for rep in range(N_REPS_RILEVANZA)]
    raw_file = C.percorso('rilevanza_fd_airtime_raw.csv')
    esegui_lavori(esegui_lavoro, lavori, lambda a: (a[0], a[1]), _inizializza_processo,
                  raw_file, ['Configurazione', 'Rep'], n_processi, log)

    raw = pd.read_csv(raw_file)
    q = raw.groupby('Configurazione')['Quota_train_sporca'].mean()
    log("\nQuota di righe di training sporcate: " + ", ".join(f"{k} {v:.3f}" for k, v in q.items()))
    base = raw[raw['Configurazione'].str.startswith('Nessun rumore')]
    if (base['F1_test_pulito'] != base['F1_test_sporco']).any() or (base['Quota_train_sporca'] != 0).any():
        log("STOP: a rumore 0% il training deve essere pulito e le due valutazioni coincidere.")
        sys.exit(1)
    rumorose = q.drop(index=[k for k in q.index if k.startswith('Nessun rumore')])
    if ((rumorose - LIVELLO).abs() > 0.01).any():
        log("STOP: la quota di righe sporcate non corrisponde al livello di rumore.")
        sys.exit(1)

    per_modello, media = build_sintesi(raw)
    per_modello.to_csv(C.percorso('rilevanza_fd_airtime.csv'), index=False)
    log("\nBaseline (rumore 0%): " + ", ".join(f"{r.Modello} {r.F1_test_pulito:.4f}" for r in base.itertuples()))
    log("\nCalo di F1 sul test pulito al 40% (media 4 modelli)\n" + media.to_string(index=False))
    log("\nPer modello\n" + per_modello.pivot(index='Configurazione', columns='Modello', values='Calo_F1').to_string())


def conta_violazioni_config(df):
    fds = C.FDS_LIST + C.FD_RIDONDANTI
    fds += [fd for fd, _ in CONFIGURAZIONI.values()]
    fds += [(['OriginAirportID'], c) for c in ORIGINE]
    return C.conta_violazioni(df, fds)


if __name__ == "__main__":
    main()
