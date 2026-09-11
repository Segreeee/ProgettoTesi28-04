"""
Configurazione del Blocco 2 su un campione del 45% del CSV originale.

Tutti i parametri dell'esperimento stanno qui, cosi' che motore, esperimento,
analisi e grafici leggano gli stessi valori.
"""
import os
import numpy as np

QUI = os.path.dirname(os.path.abspath(__file__))
CARTELLA_PROGETTO = os.path.dirname(QUI)

# ---------------------------------------------------------------
# Dati. Il CSV originale e' letto dalla cartella del progetto
# principale SOLO IN LETTURA: nessun file di quella cartella viene
# scritto o modificato.
# ---------------------------------------------------------------
CSV_ORIGINALE = os.path.join(
    CARTELLA_PROGETTO,
    'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_1.csv',
)
CAMPIONE_10K = os.path.join(CARTELLA_PROGETTO, 'flight_sample_10000.csv')  # usato solo dai test

# Output: tutti in questa cartella.
DATASET_PULITO = os.path.join(QUI, 'dataset_pulito_bilanciato.csv.gz')
VERIFICA_FD_CSV = os.path.join(QUI, 'blocco2_verifica_fd.csv')
RISULTATI_RAW = os.path.join(QUI, 'blocco2_risultati_raw.csv')
LOG_ESPERIMENTO = os.path.join(QUI, 'esperimento.log')

# ---------------------------------------------------------------
# Campione e bilanciamento
# ---------------------------------------------------------------
QUOTA_CAMPIONE = 0.45        # 45% delle righe del CSV, stratificato per classe
SEED_CAMPIONE = 42
SEED_BILANCIAMENTO = 42

# ---------------------------------------------------------------
# Target: 5 fasce di durata schedulata (identico al Blocco 1)
# ---------------------------------------------------------------
TARGET_SORGENTE = 'CRSElapsedTime'
TARGET_COL = 'DurataBucket'
FASCE_TARGET = [0, 90, 150, 210, 300, np.inf]
ETICHETTE_TARGET = ['<90m', '90-150m', '150-210m', '210-300m', '>300m']

# Colonne di servizio: MAI usate come feature.
COL_FOLD = 'Fold'
COL_RIGA_ORIGINALE = 'Riga_originale'
COLONNE_NON_FEATURE = [TARGET_COL, COL_FOLD, COL_RIGA_ORIGINALE]

# ---------------------------------------------------------------
# Blacklist anti-leakage: identica al Blocco 1
# (progettoTesi_v2._prepare_xy + EXTRA_BLACKLIST di blocco1_esperimento).
# CRSElapsedTime e' la fonte del target: viene caricata per costruirlo e
# poi eliminata. Si scartano inoltre tutte le colonne con 'ID' nel nome,
# tranne OriginAirportID e DestAirportID.
# ---------------------------------------------------------------
BLACKLIST = [
    'ArrDelay', 'ArrDelayMinutes', 'ArrDel15', 'ArrTime', 'ActualElapsedTime',
    'AirTime', 'TaxiIn', 'TaxiOut', 'WheelsOff', 'WheelsOn',
    'DepTime', 'DepDel15', 'ArrivalDelayGroups', 'DepartureDelayGroups',
    'FlightDate', 'Tail_Number', 'Flight_Number_Reporting_Airline',
    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
    'Cancelled', 'CancellationCode', 'Diverted', 'FirstDepTime', 'TotalAddGTime',
    'LongestAddGTime', 'DepDelayMinutes', 'DelayGroups',
    'CRSElapsedTime', 'CRSArrTime', 'ArrTimeBlk',
]
ID_AMMESSI = ('OriginAirportID', 'DestAirportID')

# ---------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------
K_FOLD = 3
SEED_FOLD = 42

# ---------------------------------------------------------------
# Rumore: in punti percentuali interi (5 = 5%), cosi' che il numero di
# righe sporcate sia esattamente (righe * livello) // 100 senza errori
# di arrotondamento in virgola mobile.
# ---------------------------------------------------------------
LIVELLI_RUMORE = [0, 5, 10, 15, 20, 25, 30, 35, 40]
SEED_BASE_RUMORE = 1000

# ---------------------------------------------------------------
# Dipendenze funzionali (roadmap, FD5 esclusa)
# ---------------------------------------------------------------
FD = {
    'FD1': (['Origin', 'Dest'], 'Distance'),
    'FD2': (['Distance'], 'DistanceGroup'),
    'FD3': (['CRSDepTime'], 'DepTimeBlk'),
    'FD4': (['Origin'], 'OriginWac'),
}

SIGNIFICATO_FD = {
    'FD1': "una rotta ha una distanza fissa",
    'FD2': "la fascia di distanza (250 miglia) deriva dalla distanza",
    'FD3': "la fascia oraria deriva dall'orario programmato",
    'FD4': "un aeroporto sta in una sola area geografica (World Area Code)",
}

# Colonne ridondanti: codificano la stessa informazione della FD e vanno
# sporcate con essa, altrimenti il modello ricostruisce da li' cio' che si
# e' corrotto. FD1 include anche la geografia dell'aeroporto di ARRIVO
# (decisione presa: senza FD5 resterebbe pulita per sempre); FD4 include
# la geografia di PARTENZA (roadmap par. 5.4).
RIDONDANTI = {
    'FD1': ['OriginAirportID', 'DestAirportID',
            'DestCityName', 'DestState', 'DestStateName', 'DestStateFips', 'DestWac'],
    'FD2': [],
    'FD3': [],
    'FD4': ['OriginCityName', 'OriginState', 'OriginStateName', 'OriginStateFips'],
}

# FD che giustificano ciascuna colonna ridondante: devono avere 0 violazioni
# sui dati puliti (verificato da verifica_fd.py e all'avvio dell'esperimento).
FD_RIDONDANTI = [
    (['Origin'], 'OriginAirportID'), (['Dest'], 'DestAirportID'),
    (['Dest'], 'DestCityName'), (['Dest'], 'DestState'), (['Dest'], 'DestStateName'),
    (['Dest'], 'DestStateFips'), (['Dest'], 'DestWac'),
    (['Origin'], 'OriginCityName'), (['Origin'], 'OriginState'),
    (['Origin'], 'OriginStateName'), (['Origin'], 'OriginStateFips'),
]

# Configurazioni cumulative
CONFIGURAZIONI = {
    'C1': ['FD1'],
    'C2': ['FD1', 'FD2'],
    'C3': ['FD1', 'FD2', 'FD3'],
    'C4': ['FD1', 'FD2', 'FD3', 'FD4'],
}


def fds_di(config):
    """Lista di (LHS, RHS) delle FD di una configurazione."""
    return [FD[nome] for nome in CONFIGURAZIONI[config]]


def colonne_da_sporcare(config):
    """
    Unione ordinata e senza duplicati di LHS, RHS e colonne ridondanti di
    tutte le FD della configurazione. Ogni colonna viene sporcata una sola
    volta (le FD condividono colonne: Origin e' in FD1 e FD4, Distance in
    FD1 e FD2).
    """
    colonne = []
    for nome in CONFIGURAZIONI[config]:
        lhs, rhs = FD[nome]
        for c in list(lhs) + [rhs] + RIDONDANTI[nome]:
            if c not in colonne:
                colonne.append(c)
    return colonne


def seed_rumore(fold, livello):
    """Seed della scelta delle righe: dipende solo da fold e livello, NON dalla
    configurazione, cosi' che tutte le configurazioni sporchino le stesse righe."""
    return SEED_BASE_RUMORE + 100 * fold + livello


# ---------------------------------------------------------------
# Esecuzione
# ---------------------------------------------------------------
MAX_PROCESSI = 4
RAM_PER_PROCESSO_GB = 0.5
RISERVA_RAM_GB = 1.0
SOGLIA_ORE = 10                  # oltre questa proiezione il pilota si ferma
SOGLIA_ACCURACY_BASELINE = 0.5   # caso puro = 0.20
