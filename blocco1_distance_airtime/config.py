"""
Configurazione del sottoprogetto: Blocco 1 con la FD Distance -> DistanceGroup
e obiettivo predittivo AirTimeBucket (tempo effettivo in volo, 5 fasce).

Stessa metodologia del Blocco 1 del progetto principale (training sporcato,
test pulito, cross-validation stratificata a 5 fold, 4 modelli RAW, vie di
fuga chiuse). Il motore e l'esecuzione parallela sono importati dalla cartella
padre in sola lettura: nessun file del progetto principale viene modificato.
"""
import os
import sys

import numpy as np
import pandas as pd

QUI = os.path.dirname(os.path.abspath(__file__))
PADRE = os.path.dirname(QUI)
if PADRE not in sys.path:
    sys.path.insert(0, PADRE)

FILE_CAMPIONE = os.path.join(PADRE, 'flight_sample_30000.csv')


def percorso(nome):
    """Percorso di un file di output dentro la sottocartella."""
    return os.path.join(QUI, nome)


# FD dichiarata: la fascia di distanza e' definita dalla distanza. E' la FD
# piu' rilevante per la durata del volo (analisi_rilevanza_fd.py del progetto
# principale) e per AirTime (selezione_obiettivo.py).
FDS_LIST = [(['Distance'], 'DistanceGroup')]

# Colonne che codificano la stessa informazione della distanza: la rotta la
# determina (Origin + Dest -> Distance) e le colonne degli aeroporti
# determinano la rotta. Sono le stesse colonne sporcate dal Blocco 1 del
# progetto principale: cambia solo la FD dichiarata.
COLONNE_AEROPORTI = [
    'OriginAirportID', 'DestAirportID',
    'OriginCityName', 'DestCityName',
    'OriginState', 'DestState',
    'OriginStateFips', 'DestStateFips',
    'OriginStateName', 'DestStateName',
    'OriginWac', 'DestWac',
    'OriginCityMarketID', 'DestCityMarketID',
    'OriginAirportSeqID', 'DestAirportSeqID',
]
REDUNDANT_COLS_MAP = {
    (('Distance',), 'DistanceGroup'): ['Origin', 'Dest'] + COLONNE_AEROPORTI,
}

# FD che giustificano le colonne ridondanti: 0 violazioni sul campione pulito.
FD_RIDONDANTI = (
    [(['Origin', 'Dest'], 'Distance')]
    + [(['Origin'], c) for c in COLONNE_AEROPORTI if c.startswith('Origin')]
    + [(['Dest'], c) for c in COLONNE_AEROPORTI if c.startswith('Dest')]
)

# Obiettivo: AirTime (minuti effettivi in volo) in 5 fasce. Confini ai
# quintili del campione pulito (58, 86, 120, 164), fissati qui perche' siano
# identici in ogni esecuzione. AirTime non viene mai sporcato.
TARGET_COL = 'AirTimeBucket'
TARGET_SORGENTE = 'AirTime'
BINS = [0, 58, 86, 120, 164, np.inf]
LABELS = ['<=58m', '59-86m', '87-120m', '121-164m', '>164m']

# _prepare_xy scarta gia' AirTime, ActualElapsedTime, WheelsOff/On, TaxiIn/Out,
# ArrTime, DepTime. CRSElapsedTime (durata schedulata, correlazione 0,98 con
# la distanza), CRSArrTime e ArrTimeBlk (da cui si ricava la durata
# schedulata) sarebbero copie dell'informazione: stessa blacklist del progetto.
EXTRA_BLACKLIST = ['CRSElapsedTime', 'CRSArrTime', 'ArrTimeBlk']

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
N_REPS = 5
SEED_BASE = 100


def crea_target(df):
    df = df.copy()
    df[TARGET_COL] = pd.cut(df[TARGET_SORGENTE], bins=BINS, labels=LABELS)
    return df


def carica_campione():
    return crea_target(pd.read_csv(FILE_CAMPIONE, low_memory=False))


def conta_violazioni(df, fds):
    """Numero di gruppi LHS che violano almeno una FD (0 = dati coerenti)."""
    return int(sum((df.groupby(list(lhs))[rhs].nunique() > 1).sum() for lhs, rhs in fds))
