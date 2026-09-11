import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind
import matplotlib.pyplot as plt

RAW_RESULTS_FILE = 'blocco1_risultati_raw.csv'
AGGREGATO_FILE = 'blocco1_aggregato.csv'
CORRELAZIONI_FILE = 'blocco1_correlazioni.csv'
TEST_BRACCI_FILE = 'blocco1_test_bracci.csv'
PLOT_FILE = 'blocco1_plot_f1_vs_rumore.png'

# I due regimi di valutazione: il modello e' sempre addestrato sui dati
# sporcati, cambia solo il dataset su cui viene misurato.
TEST_TYPES = ['test_sporco', 'test_pulito']


def build_aggregato(df):
    """Media e deviazione standard di Accuracy/F1 per Modello x Rumore_% x Arm,
    per entrambi i regimi di valutazione (test sporco e test pulito)."""
    agg_spec = {}
    for tt in TEST_TYPES:
        agg_spec[f'Accuracy_{tt}_mean'] = (f'Accuracy_{tt}', 'mean')
        agg_spec[f'Accuracy_{tt}_sd'] = (f'Accuracy_{tt}', 'std')
        agg_spec[f'F1_{tt}_mean'] = (f'F1_Score_{tt}', 'mean')
        agg_spec[f'F1_{tt}_sd'] = (f'F1_Score_{tt}', 'std')
    agg = df.groupby(['Modello', 'Rumore_%', 'Arm']).agg(**agg_spec).reset_index()
    return agg.round(4)


def build_correlazioni(df):
    """
    Correlazione di Pearson e Spearman (con p-value) tra IM/IH e F1, calcolata
    solo sul braccio A (nel braccio C IM/IH sono costanti a 0 per costruzione),
    per entrambi i regimi di valutazione.
    """
    dirty_a = df[df['Arm'] == 'A']
    rows = []
    for modello, sub in dirty_a.groupby('Modello'):
        for tt in TEST_TYPES:
            for metrica in ['IM', 'IH']:
                pearson_r, pearson_p = pearsonr(sub[metrica], sub[f'F1_Score_{tt}'])
                spearman_rho, spearman_p = spearmanr(sub[metrica], sub[f'F1_Score_{tt}'])
                rows.append({
                    'Modello': modello,
                    'Valutazione': tt,
                    'Metrica_Inconsistenza': metrica,
                    'Pearson_r': round(pearson_r, 4),
                    'Pearson_p': round(pearson_p, 4),
                    'Spearman_rho': round(spearman_rho, 4),
                    'Spearman_p': round(spearman_p, 4),
                })
    return pd.DataFrame(rows)


def build_test_bracci(df):
    """
    T-test tra le repliche di F1 del braccio A e del braccio C, per ciascun
    Modello x Rumore_% con rumore > 0 (a 0% i due bracci coincidono), per
    entrambi i regimi di valutazione.

    La riga con Valutazione == 'test_pulito' e' quella che decide se il braccio
    C discrimina ancora: e' il criterio del GATE previsto dal piano.
    """
    rows = []
    for (modello, livello), sub in df[df['Rumore_%'] > 0].groupby(['Modello', 'Rumore_%']):
        for tt in TEST_TYPES:
            f1_a = sub[sub['Arm'] == 'A'][f'F1_Score_{tt}']
            f1_c = sub[sub['Arm'] == 'C'][f'F1_Score_{tt}']
            t_stat, p_value = ttest_ind(f1_a, f1_c)
            rows.append({
                'Modello': modello,
                'Valutazione': tt,
                'Rumore_%': livello,
                'F1_medio_A': round(f1_a.mean(), 4),
                'F1_medio_C': round(f1_c.mean(), 4),
                'Delta_A_meno_C': round(f1_a.mean() - f1_c.mean(), 4),
                't_stat': round(t_stat, 4),
                'p_value': round(p_value, 4),
                'Significativo': bool(p_value < 0.05),
            })
    return pd.DataFrame(rows)


def stampa_gate(test_bracci):
    """Riassume l'esito del GATE: il braccio C discrimina ancora sul test pulito?"""
    print("\n" + "=" * 68)
    print("GATE — il braccio C discrimina ancora, valutando sul TEST PULITO?")
    print("=" * 68)
    for tt in TEST_TYPES:
        sub = test_bracci[test_bracci['Valutazione'] == tt]
        n_sig = int(sub['Significativo'].sum())
        n_tot = len(sub)
        print(f"  {tt:<12}: {n_sig}/{n_tot} confronti con differenza significativa "
              f"(p<0.05) | delta medio A-C = {sub['Delta_A_meno_C'].mean():+.4f}")
    pulito = test_bracci[test_bracci['Valutazione'] == 'test_pulito']
    quota = pulito['Significativo'].mean()
    print()
    if quota > 0.5:
        print(f"  ESITO: TENERE il braccio C — discrimina nel {quota*100:.0f}% dei confronti.")
    else:
        print(f"  ESITO: il braccio C discrimina solo nel {quota*100:.0f}% dei confronti "
              "=> valutare con l'utente se eliminarlo e tenere solo il braccio A.")


def plot_f1_vs_rumore(agg, filename=PLOT_FILE):
    """Griglia 4 modelli x 2 regimi di valutazione: F1 vs rumore, bracci A e C."""
    modelli = sorted(agg['Modello'].unique())
    fig, axes = plt.subplots(len(modelli), 2, figsize=(11, 4 * len(modelli)),
                             sharex=True, sharey=True)

    for r, modello in enumerate(modelli):
        for c, tt in enumerate(TEST_TYPES):
            ax = axes[r][c]
            sub = agg[agg['Modello'] == modello]
            for arm, stile in [('A', {'color': 'tab:red', 'marker': 'o', 'label': 'Braccio A (viola FD)'}),
                               ('C', {'color': 'tab:blue', 'marker': 's', 'label': 'Braccio C (preserva FD)'})]:
                sa = sub[sub['Arm'] == arm].sort_values('Rumore_%')
                ax.errorbar(sa['Rumore_%'], sa[f'F1_{tt}_mean'], yerr=sa[f'F1_{tt}_sd'],
                            capsize=3, **stile)
            ax.set_title(f'{modello} — {tt.replace("_", " ")}')
            ax.grid(alpha=0.3)
            if r == len(modelli) - 1:
                ax.set_xlabel('Rumore %')
            if c == 0:
                ax.set_ylabel('F1 Score')
            if r == 0 and c == 0:
                ax.legend()

    fig.suptitle('Blocco 1 — F1 vs rumore (media ± std su 5 repliche x 5 fold)')
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in '{filename}'.")


if __name__ == "__main__":
    df = pd.read_csv(RAW_RESULTS_FILE)

    # Controllo M1: il test pulito deve essere popolato per ENTRAMBI i bracci.
    for arm in ['A', 'C']:
        mancanti = df[df['Arm'] == arm]['F1_Score_test_pulito'].isna().sum()
        print(f"Braccio {arm}: valori mancanti in F1_Score_test_pulito = {mancanti}")
        if mancanti:
            raise ValueError(f"Il braccio {arm} non e' stato valutato sul test pulito.")

    agg = build_aggregato(df)
    agg.to_csv(AGGREGATO_FILE, index=False)
    print(f"\nTabella aggregata salvata in '{AGGREGATO_FILE}'.")
    print(agg.to_string())

    correlazioni = build_correlazioni(df)
    correlazioni.to_csv(CORRELAZIONI_FILE, index=False)
    print(f"\nCorrelazioni salvate in '{CORRELAZIONI_FILE}'.")
    print(correlazioni.to_string())

    test_bracci = build_test_bracci(df)
    test_bracci.to_csv(TEST_BRACCI_FILE, index=False)
    print(f"\nT-test tra bracci salvati in '{TEST_BRACCI_FILE}'.")
    print(test_bracci.to_string())

    stampa_gate(test_bracci)
    plot_f1_vs_rumore(agg)
