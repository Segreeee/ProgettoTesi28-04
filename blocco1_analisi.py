"""
Blocco 1 — analisi statistica dei risultati.

0. Controllo: il training e' sporcato (quota di righe sporche nulla a rumore
   0%, positiva e crescente col rumore) e il test pulito coincide con quello
   sporco a rumore 0%.
1. Aggregato per Modello x Rumore (media e deviazione standard sulle repliche),
   per il test sporco e per il test pulito.
2. Scomposizione del danno: quanto e' dovuto a un apprendimento peggiore
   (baseline - test pulito) e quanto al dare al modello input corrotti
   (test pulito - test sporco).
3. Significativita' del degrado: per ogni modello e livello, t-test di Welch
   a due campioni fra le repliche del livello e quelle del baseline. Il
   baseline e' esso stesso una stima, con una sua varianza: trattarlo come
   costante sottostimerebbe la varianza complessiva.

Gli indici IM, IP e IH sono variabili OSSERVATE: l'unica variabile controllata
e' la percentuale di righe alterate. Sono misurati sulle righe di training di
ciascun fold (colonne *_mean e *_sd nei risultati grezzi).
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

RAW_RESULTS_FILE = 'blocco1_risultati_raw.csv'
AGGREGATO_FILE = 'blocco1_aggregato.csv'
SCOMPOSIZIONE_FILE = 'blocco1_scomposizione.csv'
TEST_DEGRADO_FILE = 'blocco1_test_degrado.csv'
TEST_TYPES = ['test_sporco', 'test_pulito']


def controlla_training_e_test(df):
    print("0. Controllo: training sporcato, test pulito")
    quota = df.groupby('Rumore_%')['Quota_train_sporca'].mean()
    for livello, q in quota.items():
        print(f"   rumore {livello:>2}%: quota media di righe di training sporche = {q:.4f}")
    base = df[df['Rumore_%'] == 0]
    zero_pulito = (base['Quota_train_sporca'] == 0).all()
    positivo = (df[df['Rumore_%'] > 0]['Quota_train_sporca'] > 0).all()
    crescente = quota.is_monotonic_increasing
    diff = (base['F1_Score_test_sporco'] - base['F1_Score_test_pulito']).abs().max()
    print(f"   rumore 0%: training pulito={zero_pulito}; rumore > 0: training sporco={positivo}; "
          f"quota crescente col rumore={crescente}; test sporco = test pulito a 0%: {diff == 0}")

    # Gli indici devono essere misurati sulle righe di training dei fold, non
    # sull'intero campione: senza questo, indici e metriche descriverebbero
    # popolazioni diverse e il confronto non sarebbe interpretabile.
    righe_indici = sorted(df['Righe_indici'].unique())
    print(f"   righe su cui sono calcolati gli indici (training di ogni fold): {righe_indici}")

    # Il baseline deve variare fra repliche, altrimenti il t-test a due
    # campioni non avrebbe senso: ogni replica ha bilanciamento e fold propri.
    sd_baseline = base.groupby('Modello')['F1_Score_test_pulito'].std()
    baseline_variabile = bool((sd_baseline > 0).all())
    print("   deviazione standard del baseline fra repliche: "
          + ", ".join(f"{m} {v:.4f}" for m, v in sd_baseline.items()))

    if not (zero_pulito and positivo and crescente and diff == 0 and baseline_variabile):
        print("   CONTROLLO FALLITO: i risultati non possono essere interpretati.")
        sys.exit(1)
    print("   CONTROLLO SUPERATO.\n")


INDICI = ['IM_mean', 'IP_mean', 'IH_approx_mean', 'IH_esatto_mean']


def build_aggregato(df):
    spec = {i: (i, 'mean') for i in INDICI}
    for tt in TEST_TYPES:
        spec[f'Accuracy_{tt}_mean'] = (f'Accuracy_{tt}', 'mean')
        spec[f'Accuracy_{tt}_sd'] = (f'Accuracy_{tt}', 'std')
        spec[f'F1_{tt}_mean'] = (f'F1_Score_{tt}', 'mean')
        spec[f'F1_{tt}_sd'] = (f'F1_Score_{tt}', 'std')
    return df.groupby(['Modello', 'Rumore_%']).agg(**spec).reset_index().round(4)


def build_scomposizione(df):
    """Media dei 4 modelli: perdita da apprendimento e da input corrotto."""
    curve = df.groupby('Rumore_%')[['F1_Score_test_pulito', 'F1_Score_test_sporco']].mean()
    base = curve.loc[0, 'F1_Score_test_pulito']
    righe = []
    for livello, r in curve.iterrows():
        apprendimento = base - r['F1_Score_test_pulito']
        input_corrotto = r['F1_Score_test_pulito'] - r['F1_Score_test_sporco']
        totale = base - r['F1_Score_test_sporco']
        righe.append({'Rumore_%': livello, 'F1_test_pulito': round(r['F1_Score_test_pulito'], 4),
                      'F1_test_sporco': round(r['F1_Score_test_sporco'], 4),
                      'Perdita_apprendimento': round(apprendimento, 4),
                      'Perdita_input_corrotto': round(input_corrotto, 4),
                      'Perdita_totale_test_sporco': round(totale, 4),
                      'Quota_apprendimento_%': round(100 * apprendimento / totale, 1) if totale > 0 else np.nan})
    return pd.DataFrame(righe)


def build_test_degrado(df):
    """Il calo di F1 sul test pulito rispetto al baseline e' significativo?
    Ogni replica ha righe bilanciate e fold propri, quindi anche il baseline a
    rumore 0% varia: si confrontano due campioni di repliche con il t-test di
    Welch, che non assume varianze uguali."""
    righe = []
    for modello, sub in df.groupby('Modello'):
        base = sub[sub['Rumore_%'] == 0]['F1_Score_test_pulito']
        for livello in sorted(sub['Rumore_%'].unique()):
            if livello == 0:
                continue
            valori = sub[sub['Rumore_%'] == livello]['F1_Score_test_pulito']
            t, p = ttest_ind(valori, base, equal_var=False)
            righe.append({'Modello': modello, 'Rumore_%': livello,
                          'F1_baseline': round(base.mean(), 4), 'F1_baseline_sd': round(base.std(), 4),
                          'F1_medio': round(valori.mean(), 4), 'F1_sd': round(valori.std(), 4),
                          'Calo': round(base.mean() - valori.mean(), 4),
                          't': t, 'p_value': p, 'Significativo': bool(p < 0.05) if p == p else False})
    return pd.DataFrame(righe)


if __name__ == "__main__":
    df = pd.read_csv(RAW_RESULTS_FILE)
    controlla_training_e_test(df)

    agg = build_aggregato(df)
    agg.to_csv(AGGREGATO_FILE, index=False)
    print(f"1. Aggregato salvato in '{AGGREGATO_FILE}'.")
    print(df.pivot_table(index='Rumore_%', columns='Modello', values='F1_Score_test_pulito',
                         aggfunc='mean').round(4).to_string())
    print("\n   Indici osservati (media sulle repliche, misurati sul training di ogni fold):")
    print(df.groupby('Rumore_%')[INDICI + ['IM_sd', 'IP_sd', 'IH_approx_sd']].mean().round(1).to_string())
    rapporto = (df['IH_approx_mean'] / df['IH_esatto_mean']).replace([np.inf, -np.inf], np.nan).dropna()
    if len(rapporto):
        print(f"   IH approssimato / IH esatto: da {rapporto.min():.2f} a {rapporto.max():.2f} "
              f"(garanzia teorica: al piu' 2)")

    scomp = build_scomposizione(df)
    scomp.to_csv(SCOMPOSIZIONE_FILE, index=False)
    print(f"\n2. Scomposizione del danno (media dei 4 modelli), salvata in '{SCOMPOSIZIONE_FILE}':")
    print(scomp.to_string(index=False))

    test = build_test_degrado(df)
    test.to_csv(TEST_DEGRADO_FILE, index=False)
    print(f"\n3. Significativita' del degrado (t-test di Welch a due campioni), salvata in '{TEST_DEGRADO_FILE}':")
    print(test.to_string(index=False))
    print(f"   Livelli con calo significativo: {int(test['Significativo'].sum())}/{len(test)}")
