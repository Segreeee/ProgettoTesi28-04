"""
Blocco 2 — esperimento principale.

Per ogni fold, livello di rumore e configurazione di FD:
  - il TEST e' il fold, preso dal dataset pulito e mai sporcato;
  - il TRAINING sono gli altri fold, sporcati al livello richiesto
    (a rumore 0% il training resta pulito);
  - IM/IP/IH sono misurati sul training sporcato effettivo;
  - i 4 modelli RAW sono addestrati sul training sporcato e valutati sul test pulito.

Esecuzione parallela, adattata alla RAM disponibile, con checkpoint: ogni
lavoro completato e' aggiunto subito al CSV dei risultati e, se il programma
si interrompe, alla ripartenza i lavori gia' fatti vengono saltati.

Uso:  python esperimento.py [--workers N] [--ignora-soglia]
"""
import os

# Un thread BLAS per processo: il parallelismo e' fra processi, non dentro.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import sys
import time
import argparse
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config as C
import motore as M

COLONNE_OUTPUT = [
    'Config_FD', 'N_FD', 'Rumore_%', 'Fold', 'Seed', 'Modello',
    'IM', 'IP', 'IH', 'IH_min', 'IM_FD1', 'IM_FD2', 'IM_FD3', 'IM_FD4',
    'Colonne_sporcate', 'Righe_train', 'Righe_train_sporcate', 'Righe_test', 'Righe_test_modificate',
    'Accuracy', 'Precision', 'Recall', 'F1_weighted', 'F1_macro',
    'Tempo_s', 'Iterazioni', 'Tempo_lavoro_s', 'Baseline_condivisa',
]
BASELINE = 'baseline'

# ---------------------------------------------------------------
# Lato processo di lavoro
# ---------------------------------------------------------------
_DF = None
_IMPRONTA = None


def _inizializza_processo():
    """Ogni processo carica una volta il dataset pulito e l'impronta delle sue righe."""
    global _DF, _IMPRONTA
    warnings.filterwarnings('ignore')
    _DF, _ = M.carica_dataset_pulito()
    _IMPRONTA = M.impronta_righe(_DF)


def esegui_lavoro(fold, livello, config, n_jobs_rf):
    t0 = time.time()
    train_pulito, test = M.dividi_fold(_DF, fold)

    # --- struttura: training e test disgiunti e complementari
    if set(train_pulito.index) & set(test.index) or len(train_pulito) + len(test) != len(_DF):
        raise AssertionError(f"Fold {fold}: training e test non sono una partizione del dataset.")

    baseline = (config == BASELINE)
    colonne = [] if baseline else C.colonne_da_sporcare(config)
    seed = C.seed_rumore(fold, livello)

    # --- la funzione di sporcatura riceve SOLO il training
    train_sporco, _ = M.sporca_training(train_pulito, colonne, livello, seed)

    # --- garanzia, misurata contro l'impronta del dataset pulito originale
    attese = (len(train_pulito) * livello) // 100
    train_sporcate = M.righe_modificate(train_sporco, _IMPRONTA)
    test_modificate = M.righe_modificate(test, _IMPRONTA)
    if train_sporcate != attese:
        raise AssertionError(f"Fold {fold}, {livello}%, {config}: righe di training sporcate "
                             f"{train_sporcate}, attese {attese}.")
    if test_modificate != 0:
        raise AssertionError(f"Fold {fold}, {livello}%, {config}: {test_modificate} righe di TEST modificate.")
    if M.conta_violazioni(test, C.fds_di('C4')) != 0:
        raise AssertionError(f"Fold {fold}: il test contiene violazioni delle FD.")

    risultati_ml = M.allena_e_valuta(train_sporco, test, n_jobs_rf)

    # A rumore 0% il training e' identico per tutte le configurazioni: si
    # addestra una volta e si registra il risultato per ciascuna.
    configurazioni = list(C.CONFIGURAZIONI) if baseline else [config]
    righe = []
    for cfg in configurazioni:
        fds = C.fds_di(cfg)
        m = M.metriche_inconsistenza(train_sporco, fds)
        if baseline and (m['IM'] or m['IP'] or m['IH']):
            raise AssertionError(f"Fold {fold}: inconsistenza non nulla sul training pulito ({cfg}).")
        im_fd = dict(zip(C.CONFIGURAZIONI[cfg], m['IM_per_FD']))
        comune = {
            'Config_FD': cfg, 'N_FD': len(C.CONFIGURAZIONI[cfg]), 'Rumore_%': livello,
            'Fold': fold, 'Seed': seed,
            'IM': m['IM'], 'IP': m['IP'], 'IH': m['IH'], 'IH_min': m['IH_min'],
            **{f'IM_{nome}': im_fd.get(nome, np.nan) for nome in C.FD},
            'Colonne_sporcate': len(colonne),
            'Righe_train': len(train_pulito), 'Righe_train_sporcate': train_sporcate,
            'Righe_test': len(test), 'Righe_test_modificate': test_modificate,
            'Baseline_condivisa': baseline,
        }
        for r in risultati_ml:
            righe.append({**comune, **r})

    durata = round(time.time() - t0, 1)
    for r in righe:
        r['Tempo_lavoro_s'] = durata
    return righe


# ---------------------------------------------------------------
# Lato processo principale
# ---------------------------------------------------------------

def ram_disponibile_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        pass
    if os.name == 'nt':
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        stato = _Mem()
        stato.dwLength = ctypes.sizeof(_Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stato))
        return stato.ullAvailPhys / 1e9
    return 4.0


def elenco_lavori():
    lavori = [(fold, 0, BASELINE) for fold in range(C.K_FOLD)]
    for livello in C.LIVELLI_RUMORE[1:]:
        for config in C.CONFIGURAZIONI:
            for fold in range(C.K_FOLD):
                lavori.append((fold, livello, config))
    return lavori


def lavori_completati():
    """Chiavi dei lavori gia' presenti (e completi) nel CSV dei risultati."""
    if not os.path.exists(C.RISULTATI_RAW):
        return set()
    r = pd.read_csv(C.RISULTATI_RAW)
    fatti = set()
    for (fold, livello, cfg, base), g in r.groupby(['Fold', 'Rumore_%', 'Config_FD', 'Baseline_condivisa']):
        if base:
            fatti.add((int(fold), 0, BASELINE))   # registrato per ogni configurazione
        elif len(g) == 4:
            fatti.add((int(fold), int(livello), cfg))
    return fatti


def salva(righe):
    nuovo = not os.path.exists(C.RISULTATI_RAW)
    pd.DataFrame(righe)[COLONNE_OUTPUT].to_csv(C.RISULTATI_RAW, mode='a', header=nuovo, index=False)


class Registro:
    def __init__(self, percorso):
        self.f = open(percorso, 'a', encoding='utf-8')

    def __call__(self, msg):
        riga = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(riga, flush=True)
        self.f.write(riga + '\n')
        self.f.flush()


def esegui(esecutore, lavori, n_jobs_rf, log, etichetta):
    """Esegue i lavori in parallelo; si ferma al primo errore (assert inclusi)."""
    futuri = {esecutore.submit(esegui_lavoro, f, l, c, n_jobs_rf): (f, l, c) for f, l, c in lavori}
    t0, fatti, durate = time.time(), 0, []
    for fut in as_completed(futuri):
        fold, livello, config = futuri[fut]
        try:
            righe = fut.result()
        except Exception as e:
            log(f"ERRORE nel lavoro fold={fold} rumore={livello}% config={config}: {e}")
            for altro in futuri:
                altro.cancel()
            raise
        salva(righe)
        fatti += 1
        durate.append(righe[0]['Tempo_lavoro_s'])
        trascorso = time.time() - t0
        eta = trascorso / fatti * (len(lavori) - fatti)
        acc = np.mean([r['Accuracy'] for r in righe])
        log(f"{etichetta} {fatti}/{len(lavori)}  fold={fold} rumore={livello:>2}% {config:<8} "
            f"IM={righe[0]['IM']:>10,} acc media={acc:.4f}  lavoro {righe[0]['Tempo_lavoro_s']:.0f}s  "
            f"fine stimata tra {eta/60:.0f} min")
    return durate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--ignora-soglia', action='store_true')
    args = parser.parse_args()
    log = Registro(C.LOG_ESPERIMENTO)
    warnings.filterwarnings('ignore')

    log("=" * 70)
    log("Avvio esperimento Blocco 2 (campione 45% del CSV originale)")
    df, info = M.carica_dataset_pulito()
    if info:
        log(f"Dataset pulito creato: {info}")
    log(f"Dataset pulito e bilanciato: {len(df):,} righe, {df.shape[1]} colonne, "
        f"{C.K_FOLD} fold da {df[C.COL_FOLD].value_counts().min():,}-{df[C.COL_FOLD].value_counts().max():,} righe")

    violazioni = M.conta_violazioni(df, list(C.FD.values()) + C.FD_RIDONDANTI)
    if violazioni:
        log(f"STOP: il dataset pulito ha {violazioni} violazioni delle FD.")
        sys.exit(1)
    log("Dataset pulito: 0 violazioni su FD principali e ridondanti.")

    ram = ram_disponibile_gb()
    n_workers = args.workers or max(1, min(C.MAX_PROCESSI, int((ram - C.RISERVA_RAM_GB) / C.RAM_PER_PROCESSO_GB)))
    n_jobs_rf = max(1, (os.cpu_count() or 1) // n_workers)
    log(f"RAM disponibile {ram:.1f} GB -> {n_workers} processi paralleli, Random Forest con {n_jobs_rf} thread")

    fatti = lavori_completati()
    da_fare = [l for l in elenco_lavori() if l not in fatti]
    log(f"Lavori totali {len(elenco_lavori())}, gia' completati {len(fatti)}, da eseguire {len(da_fare)}")
    pilota = [l for l in da_fare if l[2] == BASELINE]
    resto = [l for l in da_fare if l[2] != BASELINE]

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_inizializza_processo) as esecutore:
        durate_pilota = esegui(esecutore, pilota, n_jobs_rf, log, "PILOTA") if pilota else []

        risultati = pd.read_csv(C.RISULTATI_RAW)
        base = risultati[risultati['Rumore_%'] == 0]
        acc_min = base['Accuracy'].min()
        log(f"Baseline (rumore 0%): accuracy minima {acc_min:.4f}, media per modello: "
            + ", ".join(f"{m} {v:.4f}" for m, v in base.groupby('Modello')['Accuracy'].mean().items()))
        if acc_min < C.SOGLIA_ACCURACY_BASELINE:
            log(f"STOP: baseline sotto {C.SOGLIA_ACCURACY_BASELINE} (caso puro 0.20). Non interpretare nulla.")
            sys.exit(1)

        if resto:
            riferimento = durate_pilota or list(base.groupby('Fold')['Tempo_lavoro_s'].first())
            ore = np.mean(riferimento) * len(resto) / n_workers / 3600
            log(f"Proiezione: {len(resto)} lavori x {np.mean(riferimento):.0f}s / {n_workers} processi "
                f"= ~{ore:.1f} ore")
            if ore > C.SOGLIA_ORE and not args.ignora_soglia:
                log(f"STOP: la proiezione supera {C.SOGLIA_ORE} ore. Rilanciare con --ignora-soglia per procedere.")
                sys.exit(2)
            esegui(esecutore, resto, n_jobs_rf, log, "LAVORO")

    finale = pd.read_csv(C.RISULTATI_RAW)
    attese = C.K_FOLD * len(C.LIVELLI_RUMORE) * len(C.CONFIGURAZIONI) * 4
    mancanti = int(finale[['Accuracy', 'F1_weighted', 'F1_macro', 'IM', 'IP', 'IH']].isna().sum().sum())
    log(f"FINE. Righe nel CSV: {len(finale)} (attese {attese}), valori mancanti nelle metriche: {mancanti}")


if __name__ == "__main__":
    main()
