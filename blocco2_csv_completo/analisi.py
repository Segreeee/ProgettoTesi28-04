"""
Blocco 2 — analisi statistica (roadmap par. 8).

0. Verifica a posteriori della garanzia training sporco / test pulito: si
   ferma se anche una sola riga dei risultati non la rispetta.
1. Aggregato: media e deviazione standard sui fold.
2. Baseline e confronto F1 weighted / F1 macro.
3. Correlazioni Pearson e Spearman fra IM/IP/IH e F1 sul test pulito.
4. Effetto del numero di FD: t-test appaiato fra configurazioni adiacenti.
5. Soglia: fino a che livello di rumore il calo di F1 resta sotto 1 punto.
"""
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel

import config as C

AGGREGATO = os.path.join(C.QUI, 'blocco2_aggregato.csv')
CORRELAZIONI = os.path.join(C.QUI, 'blocco2_correlazioni.csv')
TEST_CONFIG = os.path.join(C.QUI, 'blocco2_test_configurazioni.csv')
SOGLIE = os.path.join(C.QUI, 'blocco2_soglie.csv')
METRICA = 'F1_weighted'
CONFIGS = list(C.CONFIGURAZIONI)


def verifica_garanzia(df):
    attese = (df['Righe_train'] * df['Rumore_%']) // 100
    errori_train = df[df['Righe_train_sporcate'] != attese]
    errori_test = df[df['Righe_test_modificate'] != 0]
    errori_zero = df[(df['Rumore_%'] == 0) & (df['Righe_train_sporcate'] != 0)]
    n_attese = C.K_FOLD * len(C.LIVELLI_RUMORE) * len(CONFIGS) * 4
    print("0. Verifica a posteriori: training sporcato, test pulito")
    print(f"   righe del CSV: {len(df)} (attese {n_attese})")
    print(f"   righe con training sporcato diverso da (righe x livello)//100: {len(errori_train)}")
    print(f"   righe con test modificato: {len(errori_test)}")
    print(f"   righe a rumore 0% con training sporcato: {len(errori_zero)}")
    sporcate = df[df['Rumore_%'] > 0]
    print(f"   a rumore > 0: training sporcato fra {sporcate['Righe_train_sporcate'].min():,} e "
          f"{sporcate['Righe_train_sporcate'].max():,} righe; test modificato: sempre "
          f"{int(df['Righe_test_modificate'].max())}")
    if len(df) != n_attese or len(errori_train) or len(errori_test) or len(errori_zero):
        print("   VERIFICA FALLITA: i risultati non possono essere interpretati.")
        sys.exit(1)
    print("   VERIFICA SUPERATA su tutte le righe.\n")


def punti_distinti(df):
    """Il baseline e' registrato per ogni configurazione ma e' un solo addestramento
    per fold: nelle analisi fra configurazioni diverse va contato una volta."""
    return df[~(df['Baseline_condivisa'] & (df['Config_FD'] != CONFIGS[0]))]


def aggregato(df):
    agg = df.groupby(['Config_FD', 'Modello', 'Rumore_%']).agg(
        Accuracy=('Accuracy', 'mean'), Accuracy_sd=('Accuracy', 'std'),
        F1_weighted=('F1_weighted', 'mean'), F1_weighted_sd=('F1_weighted', 'std'),
        F1_macro=('F1_macro', 'mean'),
        IM=('IM', 'mean'), IP=('IP', 'mean'), IH=('IH', 'mean'),
        Righe_train_sporcate=('Righe_train_sporcate', 'mean'),
    ).reset_index()
    return agg.round(4)


def correlazioni(df):
    righe = []
    distinti = punti_distinti(df)
    for modello, sub in distinti.groupby('Modello'):
        for ambito, dati in [('tutte le configurazioni', sub)] + \
                            [(cfg, df[(df['Modello'] == modello) & (df['Config_FD'] == cfg)]) for cfg in CONFIGS]:
            for metrica in ['IM', 'IP', 'IH']:
                if dati[metrica].nunique() < 2:
                    continue
                r, p = pearsonr(dati[metrica], dati[METRICA])
                rho, p_s = spearmanr(dati[metrica], dati[METRICA])
                righe.append({'Modello': modello, 'Ambito': ambito, 'Metrica': metrica, 'N_punti': len(dati),
                              'Pearson_r': round(r, 4), 'Pearson_p': p, 'Spearman_rho': round(rho, 4),
                              'Spearman_p': p_s})
    return pd.DataFrame(righe)


def test_configurazioni(df):
    """t-test appaiato: a parita' di fold e livello, i fold sono le stesse righe."""
    righe = []
    for livello in C.LIVELLI_RUMORE[1:]:
        sub = df[df['Rumore_%'] == livello]
        for a, b in zip(CONFIGS[:-1], CONFIGS[1:]):
            coppie_tutte = []
            for modello, g in sub.groupby('Modello'):
                xa = g[g['Config_FD'] == a].sort_values('Fold')[METRICA].to_numpy()
                xb = g[g['Config_FD'] == b].sort_values('Fold')[METRICA].to_numpy()
                coppie_tutte.append((xa, xb))
                diff = xb - xa
                t, p = ttest_rel(xb, xa) if np.std(diff) > 0 else (np.nan, np.nan)
                righe.append({'Rumore_%': livello, 'Confronto': f'{a} -> {b}', 'Modello': modello,
                              'N_coppie': len(diff), 'Delta_F1': round(diff.mean(), 4),
                              't': t, 'p_value': p})
            xa = np.concatenate([x for x, _ in coppie_tutte])
            xb = np.concatenate([y for _, y in coppie_tutte])
            t, p = ttest_rel(xb, xa)
            righe.append({'Rumore_%': livello, 'Confronto': f'{a} -> {b}', 'Modello': 'TUTTI (aggregato)',
                          'N_coppie': len(xa), 'Delta_F1': round((xb - xa).mean(), 4), 't': t, 'p_value': p})
    return pd.DataFrame(righe)


def soglie(df):
    """Per ogni configurazione: il livello di rumore piu' alto fino al quale il calo
    medio di F1 rispetto al baseline resta sotto 1 punto (0.01) a tutti i livelli."""
    righe = []
    for cfg in CONFIGS:
        sub = df[df['Config_FD'] == cfg]
        for modello, g in [('media dei 4 modelli', sub)] + list(sub.groupby('Modello')):
            curva = g.groupby('Rumore_%')[METRICA].mean()
            calo = curva.loc[0] - curva
            soglia = 0
            for livello in C.LIVELLI_RUMORE[1:]:
                if calo.loc[livello] < 0.01:
                    soglia = livello
                else:
                    break
            righe.append({'Config_FD': cfg, 'Modello': modello, 'Soglia_%': soglia,
                          'Calo_al_40%': round(calo.loc[40], 4)})
    return pd.DataFrame(righe)


def main():
    df = pd.read_csv(C.RISULTATI_RAW)
    verifica_garanzia(df)

    agg = aggregato(df)
    agg.to_csv(AGGREGATO, index=False)
    print("1. Aggregato salvato:", AGGREGATO)

    base = df[df['Rumore_%'] == 0].groupby('Modello')['Accuracy'].mean()
    print("\n2. Baseline (rumore 0%), accuracy media sui fold:")
    print(base.round(4).to_string())
    if (base < C.SOGLIA_ACCURACY_BASELINE).any():
        print("   ATTENZIONE: baseline vicino al caso puro (0.20). Non interpretare.")
        sys.exit(1)
    div = (df['F1_weighted'] - df['F1_macro']).abs().max()
    print(f"   massima differenza F1 weighted / F1 macro: {div:.6f}"
          f" -> {'coincidono (bilanciamento corretto)' if div < 0.005 else 'DIVERGONO: controllare il bilanciamento'}")

    print(f"\n   {METRICA} medio sui 4 modelli, per configurazione:")
    print(df.pivot_table(index='Rumore_%', columns='Config_FD', values=METRICA, aggfunc='mean').round(4).to_string())

    corr = correlazioni(df)
    corr.to_csv(CORRELAZIONI, index=False)
    print("\n3. Correlazioni con F1 (tutte le configurazioni):")
    vista = corr[corr['Ambito'] == 'tutte le configurazioni']
    print(vista[['Modello', 'Metrica', 'N_punti', 'Pearson_r', 'Pearson_p', 'Spearman_rho', 'Spearman_p']]
          .to_string(index=False))

    tc = test_configurazioni(df)
    tc.to_csv(TEST_CONFIG, index=False)
    print("\n4. Configurazioni adiacenti, test aggregato sui 4 modelli (12 coppie):")
    agg_tc = tc[tc['Modello'] == 'TUTTI (aggregato)']
    print(agg_tc[['Rumore_%', 'Confronto', 'Delta_F1', 't', 'p_value']].to_string(index=False))
    per_modello = tc[tc['Modello'] != 'TUTTI (aggregato)']
    print(f"   per modello (3 coppie, 2 gradi di liberta'): significativi {int((per_modello['p_value'] < 0.05).sum())}"
          f"/{len(per_modello)}  |  aggregati: {int((agg_tc['p_value'] < 0.05).sum())}/{len(agg_tc)}")

    sg = soglie(df)
    sg.to_csv(SOGLIE, index=False)
    print("\n5. Soglia di rumore sotto la quale il calo di F1 resta < 1 punto:")
    print(sg[sg['Modello'] == 'media dei 4 modelli'].to_string(index=False))


if __name__ == "__main__":
    main()
