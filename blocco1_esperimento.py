import pandas as pd
import numpy as np

from progettoTesi_v2 import (
    ml_preparation,
    get_global_inconsistency_metrics,
    inject_multiple_fd_noise,
)

# ==========================================
# BLOCCO 1: target a segnale predittivo forte (DurataBucket)
# ==========================================
# Disegno a due bracci, entrambi addestrati sui dati sporcati e valutati sia
# sul test sporco sia sul test PULITO (vedi ml_preparation, parametro df_eval):
#
#   - Braccio A: rumore che VIOLA la FD. Per le righe scelte corrompe in modo
#     indipendente tutte le colonne del concetto "rotta" -> IM > 0.
#   - Braccio C: rumore di controllo che PRESERVA la FD. Corrompe le STESSE
#     colonne del braccio A, ma riassegnando a interi gruppi LHS (intere rotte)
#     il profilo completo di un'ALTRA rotta reale: ogni gruppo resta
#     internamente uniforme e i valori sono una combinazione realmente
#     esistente, quindi IM = 0 per costruzione.
#
# I due bracci corrompono cosi' lo stesso numero di colonne e un numero
# confrontabile di righe: la sola differenza e' la coerenza interna, che e'
# esattamente la variabile che l'esperimento vuole isolare.

FDS_LIST = [(['Origin', 'Dest'], 'Distance')]

# Colonne ridondanti verificate a 0 violazioni con Origin/Dest sul campione
# (vanno sporcate insieme a Distance per evitare che il modello le usi come
# via di fuga). Usate da ENTRAMBI i bracci.
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
    ]
}

# CRSElapsedTime e' la fonte diretta del target; CRSArrTime/ArrTimeBlk sono
# leakage (CRSArrTime - CRSDepTime approssima CRSElapsedTime, a meno del fuso
# orario tra origine e destinazione: corr. 0.67, errore medio ~44 min).
EXTRA_BLACKLIST = ['CRSElapsedTime', 'CRSArrTime', 'ArrTimeBlk']

TARGET_COL = 'DurataBucket'
# Granularita' concentrata nell'intervallo di interesse (max 40%).
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
N_REPS = 5
SEED_BASE = 100

RAW_RESULTS_FILE = 'blocco1_risultati_raw.csv'


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


def inject_fd_preserving_noise(df, fds_list, noise_level, seed=42, redundant_cols_map=None):
    """
    Braccio C (controllo equo): corrompe le STESSE colonne del braccio A
    (LHS + RHS + colonne ridondanti) ma in modo coerente per gruppo.

    Per ogni FD, sceglie interi gruppi LHS (es. intere rotte Origin+Dest) fino
    a coprire circa lo stesso numero di righe corrotte dal braccio A allo
    stesso noise_level, e riassegna a TUTTE le righe del gruppo il profilo
    completo di un'ALTRA rotta realmente presente nel dataset (nuovo LHS, nuovo
    RHS, nuove colonne ridondanti, tutti coerenti tra loro).

    Poiche' ogni gruppo riceve una combinazione di valori realmente esistente e
    resta internamente uniforme, tutte le FD restano soddisfatte: IM = 0 per
    costruzione, qualunque sia il livello di rumore. Cambia solo QUALI valori
    ci sono, non la loro coerenza reciproca.
    """
    df_noisy = df.copy()
    if noise_level == 0.0:
        return df_noisy

    n_to_corrupt = int(len(df) * noise_level)
    rng = np.random.RandomState(seed)
    redundant_cols_map = redundant_cols_map or {}

    for lhs, rhs in fds_list:
        extra_cols = redundant_cols_map.get((tuple(lhs), rhs), [])
        other_cols = [c for c in ([rhs] + list(extra_cols)) if c in df.columns]

        # Profilo di ciascun gruppo LHS: il valore (unico) di ogni colonna
        # coinvolta. Le FD garantiscono che sia effettivamente unico.
        if df.groupby(lhs)[other_cols].nunique().max().max() > 1:
            raise ValueError(
                f"Il gruppo LHS {lhs} non determina univocamente {other_cols}: "
                "il braccio C non puo' costruire profili coerenti."
            )
        profiles = df.groupby(lhs)[other_cols].first()
        keys = list(profiles.index)

        groups = list(df.groupby(lhs).groups.items())
        rng.shuffle(groups)

        covered = 0
        for key, idx in groups:
            if covered >= n_to_corrupt:
                break
            idx_list = list(idx)

            # Scegli un'ALTRA rotta reale come profilo sostitutivo.
            new_key = keys[rng.randint(len(keys))]
            tentativi = 0
            while new_key == key and tentativi < 50:
                new_key = keys[rng.randint(len(keys))]
                tentativi += 1
            if new_key == key:
                continue

            # Riassegna il LHS (dalla chiave) e tutte le altre colonne (dal
            # profilo), colonna per colonna per preservare i dtype.
            key_vals = new_key if isinstance(new_key, tuple) else (new_key,)
            for col, val in zip(lhs, key_vals):
                df_noisy.loc[idx_list, col] = val
            for col in other_cols:
                df_noisy.loc[idx_list, col] = profiles.loc[new_key, col]

            covered += len(idx_list)

    return df_noisy


def run_experiment(df_sample):
    all_results = []

    for arm in ['A', 'C']:
        for level in NOISE_LEVELS:
            for rep in range(N_REPS):
                seed = SEED_BASE + rep

                if arm == 'A':
                    df_noisy = inject_multiple_fd_noise(
                        df_sample, FDS_LIST, noise_level=level,
                        corrupt_lhs=True, redundant_cols_map=REDUNDANT_COLS_MAP,
                        seed=seed,
                    )
                else:
                    df_noisy = inject_fd_preserving_noise(
                        df_sample, FDS_LIST, noise_level=level,
                        seed=seed, redundant_cols_map=REDUNDANT_COLS_MAP,
                    )

                im, ip, ih, _ = get_global_inconsistency_metrics(df_noisy, FDS_LIST)

                # NOTA: questa chiamata e' FUORI dai rami if/else, quindi vale
                # identicamente per il braccio A e per il braccio C. Il modello
                # e' addestrato sulle righe sporcate di df_noisy e valutato sia
                # sul test sporco sia sul test PULITO (df_eval=df_sample).
                ml_scores = ml_preparation(
                    df_noisy, TARGET_COL,
                    extra_blacklist=EXTRA_BLACKLIST,
                    df_eval=df_sample,
                )

                print(f"[Arm {arm}] Rumore={level*100:.0f}% Rep={rep} Seed={seed} "
                      f"-> IM={im} IP={ip} IH={ih}")

                for model_name, metrics in ml_scores.items():
                    row = {
                        'Arm': arm,
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
    print(df_sample[TARGET_COL].value_counts())

    # Il dataset di valutazione deve essere realmente pulito: nessuna
    # violazione della FD monitorata.
    im0, _, _, _ = get_global_inconsistency_metrics(df_sample, FDS_LIST)
    if im0 != 0:
        raise ValueError(
            f"Il dataset di riferimento non e' pulito (IM={im0}): non puo' essere "
            "usato come test set pulito."
        )
    print(f"\nVerifica dataset di valutazione: IM={im0} (pulito, OK)")

    print(f"FD sporcata: {FDS_LIST}")
    print(f"Blacklist aggiuntiva: {EXTRA_BLACKLIST}")
    print(f"Livelli di rumore: {NOISE_LEVELS} | Repliche per configurazione: {N_REPS}")
    print("-" * 60)

    df_results = run_experiment(df_sample)
    df_results.to_csv(RAW_RESULTS_FILE, index=False)
    print(f"\nRisultati grezzi salvati in '{RAW_RESULTS_FILE}'.")

    # Verifica baseline richiesta: a Rumore_%=0 l'accuracy deve essere
    # nettamente sopra il caso puro (1/5 = 0.20 per 5 classi bilanciate).
    baseline = df_results[df_results['Rumore_%'] == 0]
    baseline_acc = baseline.groupby('Modello')['Accuracy_test_pulito'].mean()
    print("\n=== VERIFICA BASELINE (Rumore 0%) ===")
    print(baseline_acc)
    print("Caso puro atteso per 5 classi bilanciate: ~0.20")
    if (baseline_acc < 0.5).any():
        print("ATTENZIONE: almeno un modello ha baseline vicino al caso puro — "
              "rivalutare prima di interpretare l'effetto del rumore.")
    else:
        print("Baseline nettamente sopra il caso puro: si puo' procedere "
              "con l'interpretazione dell'effetto del rumore.")

    # Sanity check M1: a rumore 0% i due dataset coincidono, quindi le metriche
    # su test sporco e test pulito devono essere identiche.
    diff = (baseline['Accuracy_test_sporco'] - baseline['Accuracy_test_pulito']).abs().max()
    print(f"\nSanity check (rumore 0%): max |test_sporco - test_pulito| = {diff:.6f}")
    if diff > 1e-9:
        raise ValueError("A rumore 0% le due valutazioni devono coincidere: allineamento rotto.")
    print("Sanity check superato.")
