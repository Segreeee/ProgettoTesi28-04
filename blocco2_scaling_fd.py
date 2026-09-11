import pandas as pd
import numpy as np

from progettoTesi_v2 import (
    ml_preparation,
    get_global_inconsistency_metrics,
    inject_multiple_fd_noise,
)
from blocco1_esperimento import create_durata_bucket, EXTRA_BLACKLIST, TARGET_COL

# ==========================================
# BLOCCO 2: quante FD? Scaling 1 -> 5 -> 10
# ==========================================
# Risponde alla domanda del relatore "cosa succede per 1, 5, 10 FD".
# Si usa il solo braccio A (rumore che viola le FD): la domanda riguarda
# l'effetto del NUMERO di FD corrotte, non richiede il braccio di controllo.
#
# Come nel Blocco 1, il modello e' addestrato sui dati sporcati e valutato sia
# sul test sporco sia sul test PULITO (df_eval=df_sample).
#
# NOTA DI DISEGNO: qui NON si usa redundant_cols_map. Nel Blocco 1 le colonne
# ridondanti venivano forzatamente sporcate per chiudere le "vie di fuga" del
# modello; qui invece quelle stesse colonne compaiono come RHS delle FD
# aggiuntive, quindi vengono sporcate solo nella misura in cui la FD che le
# riguarda entra nel set. E' esattamente questo che rende l'esperimento
# informativo: con 1 sola FD il modello puo' ancora aggirare il danno, con 10
# le vie di fuga si chiudono progressivamente.

# Pool di FD: tutte verificate a 0 violazioni sul campione da 10.000 righe,
# valide nel dominio reale e con TUTTE le colonne che superano la blacklist
# (verifica automatica all'avvio). Ordine scelto per massimizzare la diversita'
# di concetto nei primi step: rotta -> aeroporto origine -> aeroporto
# destinazione, poi si estende all'interno delle famiglie.
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

# ESCLUSA deliberatamente: Reporting_Airline -> IATA_CODE_Reporting_Airline.
# E' una FD valida e sensata nel mondo reale, ma il suo LHS ha solo 14 valori
# distinti: i gruppi risultanti sono ~50 volte piu' grandi di quelli delle FD
# aeroportuali e da sola genera 2.039.767 conflitti contro i 367.368 di tutte
# le altre nove messe insieme (misurato). Dominerebbe IM/IP/IH rendendoli non
# confrontabili tra le configurazioni a 1, 5 e 10 FD.

FD_COUNTS = [1, 5, 10]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
N_REPS = 5
SEED_BASE = 100

RAW_RESULTS_FILE = 'blocco2_risultati_raw.csv'


def verifica_fd(df, fd_pool):
    """
    Controllo preliminare obbligatorio: ogni FD del pool deve avere 0 violazioni
    sul campione effettivamente usato e tutte le sue colonne devono superare la
    blacklist di ml_preparation (altrimenti il rumore inciderebbe su IM/IP/IH
    ma non sulle metriche ML).
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

    print("Verifica preliminare del pool di FD:")
    for i, (lhs, rhs) in enumerate(fd_pool, 1):
        sub = df.dropna(subset=lhs + [rhs])
        viol = int((sub.groupby(lhs)[rhs].nunique() > 1).sum())
        fuori = [c for c in lhs + [rhs] if c in scartate]
        if viol > 0:
            raise ValueError(f"FD #{i} {lhs}->{rhs}: {viol} violazioni sul campione.")
        if fuori:
            raise ValueError(f"FD #{i} {lhs}->{rhs}: colonne scartate dalla blacklist {fuori}.")
        print(f"  #{i:>2} {'+'.join(lhs) + ' -> ' + rhs:<48} 0 violazioni, colonne OK")
    print()


def run_experiment(df_sample):
    all_results = []

    for n_fd in FD_COUNTS:
        fds = FD_POOL[:n_fd]
        for level in NOISE_LEVELS:
            for rep in range(N_REPS):
                seed = SEED_BASE + rep

                df_noisy = inject_multiple_fd_noise(
                    df_sample, fds, noise_level=level,
                    corrupt_lhs=True, redundant_cols_map=None, seed=seed,
                )

                # IM/IP/IH misurati sullo STESSO set di FD che viene corrotto.
                im, ip, ih, _ = get_global_inconsistency_metrics(df_noisy, fds)

                ml_scores = ml_preparation(
                    df_noisy, TARGET_COL,
                    extra_blacklist=EXTRA_BLACKLIST,
                    df_eval=df_sample,
                )

                print(f"[{n_fd:>2} FD] Rumore={level*100:.0f}% Rep={rep} "
                      f"-> IM={im:,} IP={ip:,} IH={ih:,}")

                for model_name, metrics in ml_scores.items():
                    row = {
                        'N_FD': n_fd,
                        'Rumore_%': int(level * 100),
                        'Rep': rep,
                        'Seed': seed,
                        'Modello': model_name,
                        'IM': im,
                        'IP': ip,
                        'IH': ih,
                    }
                    row.update({k: round(v, 4) for k, v in metrics.items()})
                    all_results.append(row)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    print("Caricamento campione e creazione target DurataBucket...")
    df_sample = pd.read_csv('flight_sample_10000.csv', low_memory=False)
    df_sample = create_durata_bucket(df_sample)

    verifica_fd(df_sample, FD_POOL)

    im0, _, _, _ = get_global_inconsistency_metrics(df_sample, FD_POOL)
    if im0 != 0:
        raise ValueError(f"Il dataset di riferimento non e' pulito (IM={im0}).")
    print(f"Verifica dataset di valutazione sulle 10 FD: IM={im0} (pulito, OK)\n")

    print(f"Configurazioni: {FD_COUNTS} FD | Livelli di rumore: {NOISE_LEVELS} | Repliche: {N_REPS}")
    print("-" * 60)

    df_results = run_experiment(df_sample)
    df_results.to_csv(RAW_RESULTS_FILE, index=False)
    print(f"\nRisultati grezzi salvati in '{RAW_RESULTS_FILE}'.")

    print("\n=== SINTESI: F1 medio (test pulito) per numero di FD e livello di rumore ===")
    sintesi = df_results.pivot_table(
        index='Rumore_%', columns='N_FD', values='F1_Score_test_pulito', aggfunc='mean'
    ).round(4)
    print(sintesi.to_string())

    print("\n=== IM medio per numero di FD e livello di rumore ===")
    print(df_results.pivot_table(index='Rumore_%', columns='N_FD', values='IM', aggfunc='mean').round(0).to_string())
