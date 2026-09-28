"""
Grafici e tabella del sottoprogetto, con lo stesso stile di genera_plot_finali.py
del progetto principale, piu' due confronti con il Blocco 1 principale
(obiettivo DurataBucket, FD Origin + Dest -> Distance), letto in sola lettura.

Le colonne sporcate sono le stesse nei due esperimenti; cambiano l'obiettivo
predittivo e la FD dichiarata su cui si misurano IM, IP e IH.
"""
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from stile_grafici import (BLUE, ORANGE, PALETTE, stile, linea, formato_migliaia,
                           barre_correlazioni, salva_tabella_immagine)

P = C.percorso
df = pd.read_csv(P('risultati_raw.csv'))
principale = pd.read_csv(C.os.path.join(C.PADRE, 'blocco1_risultati_raw.csv'))
modelli = sorted(df['Modello'].unique())
COLORI_MODELLI = dict(zip(modelli, PALETTE))
ETICHETTA_NUOVO = "AirTime · FD Distance → DistanceGroup"
ETICHETTA_PRINCIPALE = "DurataBucket · FD Origin+Dest → Distance"

# --- F1 sul test pulito, per modello (banda = std sulle repliche dello stesso modello) ---
fig, ax = plt.subplots(figsize=(8, 5.2))
for m in modelli:
    s = df[df['Modello'] == m].groupby('Rumore_%')['F1_Score_test_pulito'].agg(['mean', 'std'])
    ax.fill_between(s.index, s['mean'] - s['std'], s['mean'] + s['std'],
                    color=COLORI_MODELLI[m], alpha=0.12, linewidth=0)
    linea(ax, s.index, s['mean'].values, COLORI_MODELLI[m], m)
ax.set_xlabel("Rumore nel training (%)")
ax.set_ylabel("F1 sul test pulito (± std sulle repliche)")
ax.set_title("AirTime: training sporcato, test su dati puliti", loc="left", pad=62)
ax.legend(frameon=False, loc="upper left", ncols=2, bbox_to_anchor=(0, 1.22))
stile(ax)
fig.tight_layout()
fig.savefig(P("plot_f1_test_pulito.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Test sporco vs test pulito (media dei 4 modelli) ---
fig, ax = plt.subplots(figsize=(8, 5.2))
for tt, colore, marker, etichetta in [
        ('test_sporco', ORANGE, 's', 'Test su dati sporchi (il modello riceve input corrotti)'),
        ('test_pulito', BLUE, 'o', 'Test su dati puliti (misura quanto il modello ha imparato male)')]:
    s = df.groupby('Rumore_%')[f'F1_Score_{tt}'].mean()
    linea(ax, s.index, s.values, colore, etichetta, marker)
ax.set_xlabel("Rumore nel training (%)")
ax.set_ylabel("F1 (media dei 4 modelli)")
ax.set_title("AirTime: dove si misura cambia quanto danno si vede", loc="left", pad=52)
ax.legend(frameon=False, loc="upper left", ncols=1, bbox_to_anchor=(0, 1.20))
stile(ax)
fig.tight_layout()
fig.savefig(P("plot_sporco_vs_pulito.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Correlazioni IM/IH vs F1 (Spearman, test pulito) ---
corr = pd.read_csv(P('correlazioni.csv'))
corr = corr[corr['Valutazione'] == 'test_pulito']
fig, ax = plt.subplots(figsize=(8, 5.5))
barre_correlazioni(ax, corr, modelli, "")
ax.set_ylabel("Correlazione di Spearman con F1 sul test pulito")
ax.set_title("AirTime: inconsistenza del training e qualita' delle predizioni", loc="left", pad=48, fontsize=12)
ax.legend(frameon=False, loc="upper left", ncols=2, bbox_to_anchor=(0, 1.14))
fig.tight_layout()
fig.savefig(P("plot_correlazioni.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Confronto 1: calo di F1 sul test pulito, un pannello per modello ---
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
for ax, m in zip(axes, modelli):
    for dati, colore, marker, etichetta in [(df, BLUE, 'o', ETICHETTA_NUOVO),
                                            (principale, ORANGE, 's', ETICHETTA_PRINCIPALE)]:
        s = dati[dati['Modello'] == m].groupby('Rumore_%')['F1_Score_test_pulito'].mean()
        linea(ax, s.index, 100 * (s.iloc[0] - s.values), colore, etichetta, marker)
    ax.set_title(m, loc="left", fontsize=11)
    ax.set_xlabel("Rumore nel training (%)")
    stile(ax)
axes[0].set_ylabel("Calo di F1 sul test pulito (punti)")
fig.suptitle("Stesse colonne sporcate, obiettivo e FD diversi: calo di F1 per modello",
             x=0.01, ha="left", fontsize=13, y=1.12)
fig.tight_layout()
maniglie, etichette = axes[0].get_legend_handles_labels()
fig.legend(maniglie, etichette, frameon=False, loc="upper left", ncols=2, bbox_to_anchor=(0.01, 1.06))
fig.savefig(P("plot_confronto_calo_f1.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Confronto 2: IM e IH nei due esperimenti ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, metrica, titolo in [(axes[0], 'IM', "IM — conflitti a coppie"),
                            (axes[1], 'IH', "IH — tuple minime da correggere")]:
    for dati, colore, marker, etichetta in [(df, BLUE, 'o', ETICHETTA_NUOVO),
                                            (principale, ORANGE, 's', ETICHETTA_PRINCIPALE)]:
        s = dati.drop_duplicates(['Rumore_%', 'Rep']).groupby('Rumore_%')[metrica].mean()
        s = s[s.index > 0]    # scala logaritmica: il livello 0% (indici nulli) non e' rappresentabile
        linea(ax, s.index, s.values, colore, etichetta, marker)
    ax.set_yscale('log')
    ax.set_title(titolo + " (scala logaritmica)", loc="left", fontsize=11)
    ax.set_xlabel("Rumore nel training (%)")
    formato_migliaia(ax)
    stile(ax)
fig.suptitle("Stesse colonne sporcate, FD dichiarata diversa: inconsistenza misurata",
             x=0.01, ha="left", fontsize=13, y=1.12)
fig.tight_layout()
maniglie, etichette = axes[0].get_legend_handles_labels()
fig.legend(maniglie, etichette, frameon=False, loc="upper left", ncols=2, bbox_to_anchor=(0.01, 1.06))
fig.savefig(P("plot_confronto_im_ih.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Tabella riassuntiva ---
tab = df.groupby('Rumore_%').agg(
    IM=('IM', 'mean'), IH=('IH', 'mean'),
    F1_sporco=('F1_Score_test_sporco', 'mean'), F1_pulito=('F1_Score_test_pulito', 'mean'),
).reset_index()
tab['Calo'] = tab.loc[tab['Rumore_%'] == 0, 'F1_pulito'].iloc[0] - tab['F1_pulito']
tab['IM'] = tab['IM'].round(0).astype(int)
tab['IH'] = tab['IH'].round(0).astype(int)
tab[['F1_sporco', 'F1_pulito', 'Calo']] = tab[['F1_sporco', 'F1_pulito', 'Calo']].round(4)
tab.to_csv(P('tabella_riassuntiva.csv'), index=False)
vista = tab.copy()
vista['IM'] = [f"{v:,}".replace(",", ".") for v in tab['IM']]
vista['IH'] = [f"{v:,}".replace(",", ".") for v in tab['IH']]
for c in ['F1_sporco', 'F1_pulito', 'Calo']:
    vista[c] = [f"{v:.4f}" for v in tab[c]]
salva_tabella_immagine(
    vista, P("tabella_riassuntiva.png"),
    "AirTime · FD Distance → DistanceGroup — inconsistenza e F1 (media dei 4 modelli)",
    ["Rumore\n%", "IM", "IH", "F1\ntest sporco", "F1\ntest pulito", "Calo\n(test pulito)"],
    [0.7, 1.0, 0.9, 1.0, 1.0, 1.0],
)
print("Salvati: plot_f1_test_pulito.png, plot_sporco_vs_pulito.png, plot_correlazioni.png, "
      "plot_confronto_calo_f1.png, plot_confronto_im_ih.png, tabella_riassuntiva.csv/.png")
