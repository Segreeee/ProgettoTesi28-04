"""
Genera 'numeri_tesi.md': l'unica fonte dei numeri da citare nella tesi.

Legge i CSV prodotti dalle analisi e stampa, in forma di tabelle pronte da
copiare, tutti i valori che compaiono nel testo: baseline, cali per modello e
livello, quote di righe sporche, indici osservati, esiti dei test, scarto fra
IH approssimato ed esatto. Serve a evitare numeri discordanti fra capitoli.

Uso:  python numeri_tesi.py
"""
import os

import numpy as np
import pandas as pd

USCITA = 'numeri_tesi.md'


def tabella(df, titolo, indice=False):
    """Tabella markdown senza dipendenze esterne (niente tabulate)."""
    df = df.reset_index() if indice else df
    colonne = [str(c) for c in df.columns]
    righe = ["| " + " | ".join(colonne) + " |",
             "|" + "|".join(['---'] * len(colonne)) + "|"]
    for _, r in df.iterrows():
        valori = ["" if pd.isna(v) else (f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v))
                  for v in r]
        righe.append("| " + " | ".join(valori) + " |")
    return f"\n### {titolo}\n\n" + "\n".join(righe) + "\n"


def sezione_blocco1(parti):
    agg = pd.read_csv('blocco1_aggregato.csv')
    scomp = pd.read_csv('blocco1_scomposizione.csv')
    test = pd.read_csv('blocco1_test_degrado.csv')
    raw = pd.read_csv('blocco1_risultati_raw.csv')

    parti.append("\n## Blocco 1 — una FD (rotta -> distanza) con le colonne ridondanti\n")
    parti.append(f"- Righe su cui sono calcolati gli indici (training di ogni fold): "
                 f"{sorted(raw['Righe_indici'].unique())}\n")
    base = agg[agg['Rumore_%'] == 0].set_index('Modello')['F1_test_pulito_mean']
    parti.append(f"- Baseline F1 sul test pulito: " +
                 ", ".join(f"{m} {v:.4f}" for m, v in base.items()) +
                 f"; media {base.mean():.4f}\n")

    f1 = agg.pivot(index='Rumore_%', columns='Modello', values='F1_test_pulito_mean').round(4)
    f1['Media'] = f1.mean(axis=1).round(4)
    parti.append(tabella(f1.reset_index(), "F1 sul test pulito per modello e livello"))

    calo = (f1.iloc[0] - f1).round(4)
    parti.append(tabella(calo.reset_index(), "Calo di F1 rispetto al baseline (punti/100)"))

    indici = agg.groupby('Rumore_%')[['IM_mean', 'IP_mean', 'IH_approx_mean', 'IH_esatto_mean']].mean().round(1)
    parti.append(tabella(indici.reset_index(), "Indici osservati (media sulle repliche)"))
    rapporto = (agg['IH_approx_mean'] / agg['IH_esatto_mean']).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rapporto):
        parti.append(f"\n- IH approssimato / IH esatto negli esperimenti: da {rapporto.min():.2f} "
                     f"a {rapporto.max():.2f}\n")

    parti.append(tabella(scomp.round(4), "Scomposizione del danno (media dei 4 modelli)"))
    parti.append(f"\n- Confronti con calo significativo (Welch, p < 0,05): "
                 f"{int(test['Significativo'].sum())} su {len(test)}\n")
    parti.append(tabella(test[['Modello', 'Rumore_%', 'F1_baseline', 'F1_medio', 'Calo', 'p_value',
                               'Significativo']].round(4), "Test di Welch per modello e livello"))


def sezione_blocco2(parti):
    agg = pd.read_csv('blocco2_aggregato.csv')
    scomp = pd.read_csv('blocco2_scomposizione.csv')
    test = pd.read_csv('blocco2_test_degrado.csv')
    conf = pd.read_csv('blocco2_test_configurazioni.csv')
    reg = pd.read_csv('blocco2_quota_normalizzata.csv')
    punti = pd.read_csv('blocco2_quota_punti_confrontabili.csv')

    parti.append("\n## Blocco 2 — 1, 2 e 4 FD rilevanti\n")
    f1 = scomp.pivot(index='Rumore_%', columns='N_FD', values='F1_test_pulito').round(4)
    parti.append(tabella(f1.reset_index(), "F1 sul test pulito (media dei 4 modelli) per numero di FD"))
    quota = scomp.pivot(index='Rumore_%', columns='N_FD', values='Quota_train_sporca').round(4)
    parti.append(tabella(quota.reset_index(), "Quota di righe di training effettivamente sporcate"))

    for indice in ['IM_mean', 'IP_mean', 'IH_approx_mean']:
        vista = agg.pivot_table(index='Rumore_%', columns='N_FD', values=indice, aggfunc='mean').round(1)
        parti.append(tabella(vista.reset_index(), f"{indice.replace('_mean', '')} osservato per numero di FD"))

    per_modello = agg.pivot_table(index=['N_FD', 'Modello'], columns='Rumore_%',
                                  values='F1_test_pulito_mean').round(4)
    parti.append(tabella(per_modello.reset_index(), "F1 sul test pulito per configurazione e modello"))

    parti.append(tabella(scomp.round(4), "Scomposizione del danno per numero di FD"))
    parti.append(f"\n- Confronti con calo significativo (Welch): "
                 f"{int(test['Significativo'].sum())} su {len(test)}; "
                 f"per configurazione: " +
                 ", ".join(f"{n} FD {int(g['Significativo'].sum())}/{len(g)}"
                           for n, g in test.groupby('N_FD')) + "\n")
    sintesi_conf = conf.groupby('Confronto').agg(significativi=('Significativo', 'sum'),
                                                 confronti=('Significativo', 'count'),
                                                 delta_medio=('Delta_F1', 'mean')).round(4)
    parti.append(tabella(sintesi_conf.reset_index(), "Confronto fra configurazioni a parita' di livello"))
    parti.append(tabella(reg[['Modello', 'Intervallo', 'N_punti', 'R2', 'Coef_N_FD', 't_N_FD',
                              'p_N_FD', 'Significativo']].round(5),
                         "Regressione F1 ~ quota + quota^2 + numero di FD"))
    parti.append(tabella(punti.round(4), "Punti a quota di righe sporche confrontabile"))


def sezione_meccanismo(parti):
    if not os.path.exists('blocco2_meccanismo_indici.csv'):
        return
    mecc = pd.read_csv('blocco2_meccanismo_indici.csv')
    massimo = mecc['N_FD'].max()
    vista = mecc[mecc['N_FD'] == massimo]
    parti.append(f"\n## Comportamento delle misure con {massimo} FD (replica 0, primo fold)\n")
    for grandezza in ['Gruppi_LHS', 'Dimensione_massima', 'Coppie_stesso_LHS', 'Conflitti_FD',
                      'Quota_coppie_in_conflitto', 'Tuple_coinvolte_FD']:
        piv = vista.pivot_table(index=['FD_n', 'FD'], columns='Rumore_%', values=grandezza)
        parti.append(tabella(piv.reset_index(), grandezza.replace('_', ' ')))
    totali = vista.drop_duplicates('Rumore_%').set_index('Rumore_%')[
        ['IM_totale', 'IP_totale', 'IH_approx_totale']]
    parti.append(tabella(totali.reset_index(), "Misure complessive sulle stesse righe"))


def sezione_ih(parti):
    if not os.path.exists('verifica_ih.csv'):
        return
    ver = pd.read_csv('verifica_ih.csv')
    risolti = ver[ver['ILP_risolto']]
    parti.append("\n## IH: approssimato contro esatto (verifica_ih.py)\n")
    parti.append(f"\n- Rapporto approssimato / esatto: da {risolti['Rapporto_approx_su_esatto'].min():.3f} "
                 f"a {risolti['Rapporto_approx_su_esatto'].max():.3f} (garanzia teorica: al piu' 2)\n")
    parti.append(f"- Scarto assoluto: da {int(risolti['Scarto_assoluto'].min())} a "
                 f"{int(risolti['Scarto_assoluto'].max())} tuple\n")
    una = risolti[risolti['N_FD'] == 1]
    parti.append(f"- Con 1 FD la formula chiusa coincide con l'ottimo dell'ILP: "
                 f"{bool((una['IH_esatto_formula'] == una['IH_esatto_ilp']).all())}\n")
    parti.append(tabella(risolti[['N_FD', 'Rumore_%', 'Righe', 'IM', 'IP', 'IH_approx', 'IH_esatto_ilp',
                                  'Scarto_assoluto', 'Rapporto_approx_su_esatto']],
                         "Confronto su istanze ridotte"))


def sezione_ih_blocco2(parti):
    """IH esatto sulle istanze reali del Blocco 2: formula chiusa con 1 FD,
    ILP con 2 e 4 FD (ih_esatto_blocco2.py)."""
    righe = []
    agg = pd.read_csv('blocco2_aggregato.csv').drop_duplicates(['N_FD', 'Rumore_%'])
    una = agg[(agg['N_FD'] == 1) & (agg['Rumore_%'] > 0)]
    for _, r in una.iterrows():
        righe.append({'N_FD': 1, 'Rumore_%': r['Rumore_%'], 'Metodo': 'formula chiusa',
                      'Istanze_risolte': '25/25', 'IH_approx': round(r['IH_approx_mean'], 1),
                      'IH_esatto': round(r['IH_esatto_mean'], 1),
                      'Rapporto': round(r['IH_approx_mean'] / r['IH_esatto_mean'], 3)})
    if os.path.exists('blocco2_ih_esatto.csv'):
        e = pd.read_csv('blocco2_ih_esatto.csv')
        for (n, lv), g in e.groupby(['N_FD', 'Rumore_%']):
            ok = g[g['Risolto']]
            righe.append({'N_FD': n, 'Rumore_%': lv, 'Metodo': 'ILP',
                          'Istanze_risolte': f"{len(ok)}/{len(g)}",
                          'IH_approx': round(ok['IH_approx'].mean(), 1) if len(ok) else np.nan,
                          'IH_esatto': round(ok['IH_esatto'].mean(), 1) if len(ok) else np.nan,
                          'Rapporto': round((ok['IH_approx'] / ok['IH_esatto']).mean(), 3) if len(ok) else np.nan})
    parti.append("\n## IH esatto sulle istanze del Blocco 2 (fold di training, 6.472 righe)\n")
    parti.append("\nCon 1 FD formula chiusa; con 2 e 4 FD programmazione lineare intera "
                 "(`ih_esatto_blocco2.py`, limite di 150 s per istanza). Non risolti: 2 FD al 30% "
                 "e 4 FD dal 20% (prova dei tempi); 2 FD al 40% risolto solo in parte.\n")
    parti.append(tabella(pd.DataFrame(righe), "IH 2-approssimato contro esatto (medie su repliche e fold)"))


def sezione_rilevanza(parti):
    if not os.path.exists('rilevanza_fd.csv'):
        return
    ril = pd.read_csv('rilevanza_fd.csv')
    media = (ril.groupby('Configurazione')
             .agg(Colonne=('Colonne_sporcate', 'first'), Calo_medio=('Calo_F1', 'mean'),
                  Modelli_significativi=('Significativo', 'sum'))
             .round(4).sort_values('Calo_medio', ascending=False))
    parti.append("\n## Rilevanza delle FD per l'obiettivo (analisi_rilevanza_fd.py)\n")
    parti.append(tabella(media.reset_index(), "Calo di F1 al 40% sporcando una configurazione alla volta"))


if __name__ == "__main__":
    parti = ["# Numeri verificati per la tesi\n",
             "\nGenerato da `numeri_tesi.py` a partire dai CSV delle analisi. "
             "Ogni numero citato nella tesi deve corrispondere a uno di questi valori.\n",
             "\nConvenzioni: il rumore e' l'unica variabile controllata; IM, IP e IH sono osservati "
             "sulle righe di training di ciascun fold; IH e' riportato come 2-approssimato "
             "(limite superiore) e, con una sola FD, anche esatto.\n"]
    sezione_blocco1(parti)
    sezione_blocco2(parti)
    sezione_meccanismo(parti)
    sezione_ih(parti)
    sezione_ih_blocco2(parti)
    sezione_rilevanza(parti)
    with open(USCITA, 'w', encoding='utf-8') as f:
        f.write(''.join(parti))
    print(f"Scritto {USCITA} ({sum(len(p) for p in parti):,} caratteri).")
