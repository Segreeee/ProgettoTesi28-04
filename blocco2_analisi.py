"""
Blocco 2 — analisi statistica dei risultati.

Parte comune, speculare a blocco1_analisi.py (stratificata per numero di FD):
0. Controllo: il training e' sporcato e il test e' pulito. A rumore 0% la
   quota di righe sporche e' nulla e le due valutazioni coincidono; a rumore
   p la quota vale 1 - (1 - p)^N, perche' ogni FD sceglie le proprie righe.
1. Aggregato per N_FD x Modello x Rumore (media e deviazione standard sulle
   repliche), per il test sporco e per il test pulito.
2. Scomposizione del danno per N_FD: apprendimento peggiore (baseline - test
   pulito) contro input corrotto (test pulito - test sporco).
3. Significativita' del degrado: t-test di Welch a due campioni fra le
   repliche del livello e quelle del baseline, per N_FD x modello x livello.
   Il baseline e' una stima con una sua varianza, non una costante.

Parte specifica del Blocco 2:
4. Configurazioni a parita' di LIVELLO di rumore: t-test appaiato per replica
   (stesso seed, stesse righe bilanciate, stessi fold) fra configurazioni
   consecutive (1 -> 2 e 2 -> 4 FD).
5. Configurazioni a parita' di RIGHE SPORCHE: a parita' di livello, piu' FD
   sporcano molte piu' righe, quindi il confronto del punto 4 mescola "piu' FD"
   e "piu' righe sporche". Si separano i due effetti con una regressione
   F1 ~ quota + quota^2 + N_FD, e confrontando i punti a quota simile.
6. Comportamento di IM, IP e IH al crescere del rumore: si rigenerano i
   training sporcati (stesso seed della replica 0) e si misura, per ogni FD,
   come cambiano i gruppi del lato sinistro, le coppie in conflitto e le tuple
   coinvolte, e come si comportano di conseguenza le tre misure.

Uso:  python blocco2_analisi.py [--senza-meccanismo]
"""
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
from scipy.stats import t as distribuzione_t

RAW_RESULTS_FILE = 'blocco2_risultati_raw.csv'
AGGREGATO_FILE = 'blocco2_aggregato.csv'
SCOMPOSIZIONE_FILE = 'blocco2_scomposizione.csv'
TEST_DEGRADO_FILE = 'blocco2_test_degrado.csv'
TEST_CONFIGURAZIONI_FILE = 'blocco2_test_configurazioni.csv'
QUOTA_REGRESSIONE_FILE = 'blocco2_quota_normalizzata.csv'
QUOTA_PUNTI_FILE = 'blocco2_quota_punti_confrontabili.csv'
MECCANISMO_FILE = 'blocco2_meccanismo_indici.csv'
INDICI = ['IM_mean', 'IP_mean', 'IH_approx_mean', 'IH_esatto_mean']
TEST_TYPES = ['test_sporco', 'test_pulito']
TOLLERANZA_QUOTA_TEORICA = 0.01
TOLLERANZA_PUNTI_CONFRONTABILI = 0.035


def punti_distinti(df):
    """A rumore 0% nessuna FD e' sporcata: il baseline e' identico in ogni
    configurazione. Nelle analisi che mescolano le configurazioni va contato una volta."""
    return df[~((df['Rumore_%'] == 0) & (df['N_FD'] != df['N_FD'].min()))]


def controlla_training_e_test(df):
    print("0. Controllo: training sporcato, test pulito")
    quota = df.groupby(['N_FD', 'Rumore_%'])['Quota_train_sporca'].mean().unstack(0)
    attesa = pd.DataFrame({n: [1 - (1 - p / 100) ** n for p in quota.index] for n in quota.columns},
                          index=quota.index)
    print("   quota media di righe di training sporche (misurata):")
    print('   ' + quota.round(4).to_string().replace('\n', '\n   '))
    scarto = float((quota - attesa).abs().max().max())

    base = df[df['Rumore_%'] == 0]
    zero_pulito = (base['Quota_train_sporca'] == 0).all()
    positivo = (df[df['Rumore_%'] > 0]['Quota_train_sporca'] > 0).all()
    cresce_col_rumore = all(quota[n].is_monotonic_increasing for n in quota.columns)
    cresce_con_fd = all(quota.loc[p].is_monotonic_increasing for p in quota.index if p > 0)
    diff = (base['F1_Score_test_sporco'] - base['F1_Score_test_pulito']).abs().max()
    print(f"   rumore 0%: training pulito={zero_pulito}, test sporco = test pulito: {diff == 0}")
    print(f"   rumore > 0: training sporco={positivo}; quota crescente col rumore={cresce_col_rumore} "
          f"e col numero di FD={cresce_con_fd}")
    print(f"   scarto massimo dalla quota attesa 1-(1-p)^N: {scarto:.4f} (tolleranza {TOLLERANZA_QUOTA_TEORICA})")
    righe_indici = sorted(df['Righe_indici'].unique())
    print(f"   righe su cui sono calcolati gli indici (training di ogni fold): {righe_indici}")
    sd_baseline = base.groupby(['N_FD', 'Modello'])['F1_Score_test_pulito'].std()
    baseline_variabile = bool((sd_baseline > 0).all())
    print(f"   baseline variabile fra repliche: {baseline_variabile} "
          f"(deviazione standard da {sd_baseline.min():.4f} a {sd_baseline.max():.4f})")

    if not (zero_pulito and positivo and cresce_col_rumore and cresce_con_fd and diff == 0
            and baseline_variabile and scarto <= TOLLERANZA_QUOTA_TEORICA):
        print("   CONTROLLO FALLITO: i risultati non possono essere interpretati.")
        sys.exit(1)
    print("   CONTROLLO SUPERATO.\n")


def build_aggregato(df):
    spec = {'Quota_train_sporca': ('Quota_train_sporca', 'mean')}
    spec.update({i: (i, 'mean') for i in INDICI})
    for tt in TEST_TYPES:
        spec[f'Accuracy_{tt}_mean'] = (f'Accuracy_{tt}', 'mean')
        spec[f'Accuracy_{tt}_sd'] = (f'Accuracy_{tt}', 'std')
        spec[f'F1_{tt}_mean'] = (f'F1_Score_{tt}', 'mean')
        spec[f'F1_{tt}_sd'] = (f'F1_Score_{tt}', 'std')
    return df.groupby(['N_FD', 'Modello', 'Rumore_%']).agg(**spec).reset_index().round(4)


def build_scomposizione(df):
    """Media dei 4 modelli, per N_FD: perdita da apprendimento e da input corrotto."""
    righe = []
    for n_fd, sub in df.groupby('N_FD'):
        curve = sub.groupby('Rumore_%')[['F1_Score_test_pulito', 'F1_Score_test_sporco', 'Quota_train_sporca']].mean()
        base = curve.loc[0, 'F1_Score_test_pulito']
        for livello, r in curve.iterrows():
            apprendimento = base - r['F1_Score_test_pulito']
            input_corrotto = r['F1_Score_test_pulito'] - r['F1_Score_test_sporco']
            totale = base - r['F1_Score_test_sporco']
            righe.append({'N_FD': n_fd, 'Rumore_%': livello,
                          'Quota_train_sporca': round(r['Quota_train_sporca'], 4),
                          'F1_test_pulito': round(r['F1_Score_test_pulito'], 4),
                          'F1_test_sporco': round(r['F1_Score_test_sporco'], 4),
                          'Perdita_apprendimento': round(apprendimento, 4),
                          'Perdita_input_corrotto': round(input_corrotto, 4),
                          'Perdita_totale_test_sporco': round(totale, 4),
                          'Quota_apprendimento_%': round(100 * apprendimento / totale, 1) if totale > 0 else np.nan})
    return pd.DataFrame(righe)


def build_test_degrado(df):
    """Ogni replica ha righe bilanciate e fold propri, quindi anche il baseline
    a rumore 0% varia: si confrontano due campioni di repliche con il t-test di
    Welch, che non assume varianze uguali."""
    righe = []
    for (n_fd, modello), sub in df.groupby(['N_FD', 'Modello']):
        base = sub[sub['Rumore_%'] == 0]['F1_Score_test_pulito']
        for livello in sorted(sub['Rumore_%'].unique()):
            if livello == 0:
                continue
            valori = sub[sub['Rumore_%'] == livello]['F1_Score_test_pulito']
            t, p = ttest_ind(valori, base, equal_var=False)
            righe.append({'N_FD': n_fd, 'Modello': modello, 'Rumore_%': livello,
                          'F1_baseline': round(base.mean(), 4), 'F1_baseline_sd': round(base.std(), 4),
                          'F1_medio': round(valori.mean(), 4), 'F1_sd': round(valori.std(), 4),
                          'Calo': round(base.mean() - valori.mean(), 4),
                          't': t, 'p_value': p, 'Significativo': bool(p < 0.05) if p == p else False})
    return pd.DataFrame(righe)


def build_test_configurazioni(df):
    """A parita' di livello di rumore: F1 con piu' FD contro F1 con meno FD,
    appaiati per replica (stesso seed, stesse righe, stessi fold)."""
    righe = []
    configurazioni = sorted(df['N_FD'].unique())
    for livello in sorted(df['Rumore_%'].unique()):
        if livello == 0:
            continue
        for modello, sub in df[df['Rumore_%'] == livello].groupby('Modello'):
            for a, b in zip(configurazioni[:-1], configurazioni[1:]):
                xa = sub[sub['N_FD'] == a].sort_values('Rep')['F1_Score_test_pulito'].to_numpy()
                xb = sub[sub['N_FD'] == b].sort_values('Rep')['F1_Score_test_pulito'].to_numpy()
                diff = xb - xa
                t, p = ttest_rel(xb, xa) if np.std(diff) > 0 else (np.nan, np.nan)
                righe.append({'Rumore_%': livello, 'Confronto': f'{a} -> {b} FD', 'Modello': modello,
                              'F1_meno_FD': round(xa.mean(), 4), 'F1_piu_FD': round(xb.mean(), 4),
                              'Delta_F1': round(diff.mean(), 4), 't': t, 'p_value': p,
                              'Significativo': bool(p < 0.05) if p == p else False})
    return pd.DataFrame(righe)


def _ols(X, y):
    """Minimi quadrati con errori standard classici."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residui = y - X @ beta
    n, k = X.shape
    gradi = n - k
    varianza = residui @ residui / gradi
    errori = np.sqrt(np.diag(varianza * np.linalg.inv(X.T @ X)))
    t = beta / errori
    p = 2 * distribuzione_t.sf(np.abs(t), gradi)
    r2 = 1 - (residui @ residui) / ((y - y.mean()) @ (y - y.mean()))
    return beta, errori, t, p, r2, n


def build_quota_normalizzata(df):
    """
    Regressione, per modello: F1 (test pulito) ~ quota + quota^2 + N_FD.
    Il coefficiente di N_FD dice come cambia la F1 aggiungendo una FD A PARITA'
    di righe sporche. Il termine quadratico assorbe la curvatura della relazione
    fra quota e F1, che altrimenti potrebbe confondersi con l'effetto di N_FD.
    Due stime: su tutti i punti, e sul solo intervallo di quota comune alle tre
    configurazioni (dove il confronto non richiede di estrapolare).
    """
    distinti = punti_distinti(df)
    quota_max_comune = distinti.groupby('N_FD')['Quota_train_sporca'].max().min() + 0.01
    righe = []
    for modello, sub in distinti.groupby('Modello'):
        for intervallo, dati in [('tutti i punti', sub),
                                 (f'quota <= {quota_max_comune:.2f} (comune alle configurazioni)',
                                  sub[sub['Quota_train_sporca'] <= quota_max_comune])]:
            q = dati['Quota_train_sporca'].to_numpy()
            X = np.column_stack([np.ones(len(dati)), q, q ** 2, dati['N_FD'].to_numpy(dtype=float)])
            beta, errori, t, p, r2, n = _ols(X, dati['F1_Score_test_pulito'].to_numpy())
            righe.append({'Modello': modello, 'Intervallo': intervallo, 'N_punti': n, 'R2': round(r2, 4),
                          'Coef_quota': round(beta[1], 5), 'Coef_quota2': round(beta[2], 5),
                          'Coef_N_FD': round(beta[3], 6), 'SE_N_FD': round(errori[3], 6),
                          't_N_FD': round(t[3], 3), 'p_N_FD': p[3], 'Significativo': bool(p[3] < 0.05)})
    return pd.DataFrame(righe)


def build_punti_confrontabili(df):
    """Coppie di configurazioni con quota di righe sporche simile (entro la
    tolleranza) ma numero di FD diverso: quale fa piu' danno?"""
    punti = df.groupby(['N_FD', 'Rumore_%']).agg(
        Quota=('Quota_train_sporca', 'mean'), F1=('F1_Score_test_pulito', 'mean')).reset_index()
    per_modello = df.groupby(['N_FD', 'Rumore_%', 'Modello'])['F1_Score_test_pulito'].mean()
    punti = punti[punti['Rumore_%'] > 0]
    righe = []
    for _, a in punti.iterrows():
        for _, b in punti.iterrows():
            if b['N_FD'] <= a['N_FD'] or abs(a['Quota'] - b['Quota']) > TOLLERANZA_PUNTI_CONFRONTABILI:
                continue
            modelli = df['Modello'].unique()
            migliori = sum(per_modello[(b['N_FD'], b['Rumore_%'], m)] > per_modello[(a['N_FD'], a['Rumore_%'], m)]
                           for m in modelli)
            righe.append({'Meno_FD': int(a['N_FD']), 'Rumore_meno_FD_%': int(a['Rumore_%']),
                          'Quota_meno_FD': round(a['Quota'], 4), 'F1_meno_FD': round(a['F1'], 4),
                          'Piu_FD': int(b['N_FD']), 'Rumore_piu_FD_%': int(b['Rumore_%']),
                          'Quota_piu_FD': round(b['Quota'], 4), 'F1_piu_FD': round(b['F1'], 4),
                          'Delta_quota': round(b['Quota'] - a['Quota'], 4),
                          'Delta_F1_piu_meno': round(b['F1'] - a['F1'], 4),
                          'Modelli_con_piu_FD_migliore': f'{migliori}/{len(modelli)}'})
    return pd.DataFrame(righe)


def build_meccanismo_indici(df):
    """
    Come si comportano IM, IP e IH al crescere del rumore, e perche'.

    Rigenera il training sporcato della replica 0 (stesso seed
    dell'esperimento) con 1 FD e con il numero massimo di FD, e misura sulle
    righe del primo fold — le stesse su cui l'esperimento calcola gli indici:
      - per ogni FD: gruppi del lato sinistro, dimensione media e massima,
        coppie di righe con lo stesso lato sinistro (i conflitti possibili),
        coppie in conflitto e tuple coinvolte;
      - per l'insieme delle FD: IM, IP, IH approssimato ed esatto.
    Serve a distinguere il comportamento delle tre misure: la frammentazione
    dei gruppi riduce le coppie in conflitto, ma non necessariamente il numero
    di tuple coinvolte.

    Con 1 FD i conflitti della FD coincidono per costruzione con IM: e' la
    verifica che la rigenerazione sia fedele all'esperimento.
    """
    from progettoTesi_v2 import inject_multiple_fd_noise, fold_di_valutazione, indici_inconsistenza
    from blocco1_esperimento import carica_campione, EXTRA_BLACKLIST, TARGET_COL
    from blocco2_scaling_fd import FD_POOL, SEED_BASE, REDUNDANT_COLS_MAP

    campione = carica_campione()
    seed = SEED_BASE + 0
    righe_fold = fold_di_valutazione(campione, TARGET_COL, EXTRA_BLACKLIST, seed_valutazione=seed)[0]
    righe = []
    for n_fd in [1, max(df['N_FD'])]:
        fds = FD_POOL[:n_fd]
        for livello in sorted(df['Rumore_%'].unique()):
            sporco = inject_multiple_fd_noise(campione, fds, noise_level=livello / 100,
                                              corrupt_lhs=True, redundant_cols_map=REDUNDANT_COLS_MAP,
                                              seed=seed).loc[righe_fold]
            complessivi = indici_inconsistenza(sporco, fds) if livello > 0 else {
                'IM': 0, 'IP': 0, 'IH_approx': 0, 'IH_esatto': 0 if n_fd == 1 else np.nan}
            for i, (lhs, rhs) in enumerate(fds, 1):
                dim = sporco.groupby(lhs).size().to_numpy(np.int64)
                dim_valore = sporco.groupby(lhs + [rhs]).size().to_numpy(np.int64)
                coppie = int((dim * (dim - 1) // 2).sum())
                coppie_stesso_valore = int((dim_valore * (dim_valore - 1) // 2).sum())
                conflitti = coppie - coppie_stesso_valore
                # Tuple coinvolte in almeno un conflitto di QUESTA FD: tutte le
                # righe dei gruppi con piu' di un valore del lato destro.
                gruppi_misti = sporco.groupby(lhs)[rhs].nunique() > 1
                tuple_coinvolte = int(sporco.groupby(lhs).size()[gruppi_misti].sum())
                righe.append({
                    'N_FD': n_fd, 'Rumore_%': livello, 'FD_n': i, 'FD': f"{'+'.join(lhs)} -> {rhs}",
                    'Gruppi_LHS': len(dim), 'Dimensione_media': round(dim.mean(), 1),
                    'Dimensione_massima': int(dim.max()),
                    'Coppie_stesso_LHS': coppie,
                    'Conflitti_FD': conflitti,
                    'Quota_coppie_in_conflitto': round(conflitti / coppie, 4) if coppie else 0.0,
                    'Tuple_coinvolte_FD': tuple_coinvolte,
                    'IM_totale': complessivi['IM'], 'IP_totale': complessivi['IP'],
                    'IH_approx_totale': complessivi['IH_approx'], 'IH_esatto_totale': complessivi['IH_esatto'],
                })
            print(f"   meccanismo: {n_fd} FD, rumore {livello}% misurato", flush=True)
    tabella = pd.DataFrame(righe)

    una_fd = tabella[tabella['N_FD'] == 1]
    fedele = bool((una_fd['Conflitti_FD'] == una_fd['IM_totale']).all())
    print(f"   verifica di fedelta' (1 FD, replica 0): conflitti della FD = IM a ogni livello: {fedele}")
    if not fedele:
        print("   VERIFICA FALLITA: la rigenerazione non riproduce l'esperimento.")
        sys.exit(1)
    return tabella


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--senza-meccanismo', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(RAW_RESULTS_FILE)
    controlla_training_e_test(df)

    agg = build_aggregato(df)
    agg.to_csv(AGGREGATO_FILE, index=False)
    print(f"1. Aggregato salvato in '{AGGREGATO_FILE}'. F1 sul test pulito, media dei 4 modelli:")
    print(df.pivot_table(index='Rumore_%', columns='N_FD', values='F1_Score_test_pulito',
                         aggfunc='mean').round(4).to_string())
    for indice in INDICI[:3]:
        print(f"\n   {indice.replace('_mean', '')} osservato (media sulle repliche, training di ogni fold):")
        print(df.pivot_table(index='Rumore_%', columns='N_FD', values=indice, aggfunc='mean')
              .round(1).to_string())

    scomp = build_scomposizione(df)
    scomp.to_csv(SCOMPOSIZIONE_FILE, index=False)
    print(f"\n2. Scomposizione del danno per N_FD, salvata in '{SCOMPOSIZIONE_FILE}':")
    print(scomp.to_string(index=False))

    test = build_test_degrado(df)
    test.to_csv(TEST_DEGRADO_FILE, index=False)
    print(f"\n3. Significativita' del degrado, salvata in '{TEST_DEGRADO_FILE}':")
    print(test.groupby('N_FD')['Significativo'].agg(['sum', 'count']).rename(
        columns={'sum': 'significativi', 'count': 'confronti'}).to_string())

    conf = build_test_configurazioni(df)
    conf.to_csv(TEST_CONFIGURAZIONI_FILE, index=False)
    print(f"\n4. Configurazioni a parita' di livello (t-test appaiato), salvato in '{TEST_CONFIGURAZIONI_FILE}':")
    print(conf.groupby('Confronto').agg(significativi=('Significativo', 'sum'), confronti=('Significativo', 'count'),
                                        delta_medio=('Delta_F1', 'mean')).round(4).to_string())

    reg = build_quota_normalizzata(df)
    reg.to_csv(QUOTA_REGRESSIONE_FILE, index=False)
    punti = build_punti_confrontabili(df)
    punti.to_csv(QUOTA_PUNTI_FILE, index=False)
    print(f"\n5. Configurazioni a parita' di righe sporche, salvato in '{QUOTA_REGRESSIONE_FILE}' "
          f"e '{QUOTA_PUNTI_FILE}':")
    print(reg[['Modello', 'Intervallo', 'N_punti', 'R2', 'Coef_N_FD', 't_N_FD', 'p_N_FD', 'Significativo']]
          .to_string(index=False))
    print("   punti a quota confrontabile:")
    print(punti.to_string(index=False))

    if not args.senza_meccanismo:
        print("\n6. Comportamento di IM, IP e IH (rigenerazione dei training sporcati):")
        mecc = build_meccanismo_indici(df)
        mecc.to_csv(MECCANISMO_FILE, index=False)
        massimo = mecc['N_FD'].max()
        vista = mecc[mecc['N_FD'] == massimo]
        print(f"   salvato in '{MECCANISMO_FILE}'. Con {massimo} FD, per FD e livello:")
        for grandezza in ['Conflitti_FD', 'Tuple_coinvolte_FD', 'Dimensione_massima']:
            print(f"   {grandezza}:")
            print(vista.pivot_table(index=['FD_n', 'FD'], columns='Rumore_%', values=grandezza).to_string())
        print("   misure complessive sulle stesse righe:")
        print(vista.drop_duplicates('Rumore_%').set_index('Rumore_%')[
            ['IM_totale', 'IP_totale', 'IH_approx_totale']].to_string())
