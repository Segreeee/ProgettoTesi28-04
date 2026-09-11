"""
Variante "solo braccio A" — analisi e grafici.

Legge i risultati grezzi gia' prodotti dal progetto principale (cartella
superiore) e li rianalizza considerando il SOLO braccio A, cioe' la condizione
in cui il rumore viola le dipendenze funzionali:

  ../blocco1_risultati_raw.csv  -> filtrato su Arm == 'A'
  ../blocco2_risultati_raw.csv  -> gia' solo braccio A per costruzione

Nessun esperimento viene rieseguito: i dati sono gli stessi del progetto
principale, cambia solo cosa se ne mostra. Tutti gli output (CSV e PNG) sono
scritti in questa sottocartella e non toccano il progetto principale.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr

QUI = os.path.dirname(os.path.abspath(__file__))
SU = os.path.dirname(QUI)
B1 = os.path.join(SU, 'blocco1_risultati_raw.csv')
B2 = os.path.join(SU, 'blocco2_risultati_raw.csv')

# ---------------------------------------------------------------
# Stile (stesso del progetto principale: palette dataviz, light)
# ---------------------------------------------------------------
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY, "text.color": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "grid.color": GRID,
    "font.family": "sans-serif", "font.size": 10.5,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def style_axes(ax):
    ax.grid(axis="y", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(length=0)


def salva_tabella_immagine(df, filename, titolo, col_labels, col_widths):
    fig, ax = plt.subplots(figsize=(11, 0.9 + 0.5 * len(df)))
    ax.axis("off")
    ax.set_title(titolo, loc="left", fontsize=12, pad=16)
    table = ax.table(cellText=df.values, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    for ci, w in enumerate(col_widths):
        for ri in range(len(df) + 1):
            table[ri, ci].set_width(w / sum(col_widths))
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#eef2f8")
            cell.set_text_props(color=INK, fontweight="bold", wrap=True)
        else:
            cell.set_facecolor(SURFACE if r % 2 else "#f6f6f4")
            cell.set_text_props(color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(QUI, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


# =================================================================
# 1. BLOCCO 1 — solo braccio A
# =================================================================
d1 = pd.read_csv(B1)
a1 = d1[d1['Arm'] == 'A'].copy()
print(f"Blocco 1: {len(d1)} righe totali -> {len(a1)} righe del braccio A")

modelli = sorted(a1['Modello'].unique())
COLORI = dict(zip(modelli, [BLUE, ORANGE, AQUA, VIOLET]))

# --- Grafico 1: F1 sul test pulito, per modello ---
fig, ax = plt.subplots(figsize=(7.5, 5))
for m in modelli:
    s = (a1[a1['Modello'] == m].groupby('Rumore_%')['F1_Score_test_pulito']
         .agg(['mean', 'std']).reset_index())
    ax.fill_between(s['Rumore_%'], s['mean'] - s['std'], s['mean'] + s['std'],
                    color=COLORI[m], alpha=0.12, linewidth=0)
    ax.plot(s['Rumore_%'], s['mean'], color=COLORI[m], linewidth=2, marker='o',
            markersize=5, markeredgecolor=SURFACE, markeredgewidth=1, label=m, zorder=3)
ax.set_xlabel("Rumore nel training (%)")
ax.set_ylabel("F1 sul test pulito (± std su 5 repliche)")
ax.set_title("Training sporcato, test su dati puliti:\ntutti i modelli peggiorano all'aumentare del rumore",
             loc="left", pad=64)
ax.legend(frameon=False, loc="upper left", ncols=2, bbox_to_anchor=(0, 1.26))
style_axes(ax)
fig.tight_layout()
fig.savefig(os.path.join(QUI, "plot_A_f1_per_modello.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Grafico 2: perche' il test pulito cambia la misura ---
fig, ax = plt.subplots(figsize=(7.5, 5))
for tt, color, marker, lab in [('test_sporco', ORANGE, 's', 'Test su dati sporchi (metodo precedente)'),
                               ('test_pulito', BLUE, 'o', 'Test su dati puliti (metodo corretto)')]:
    s = a1.groupby('Rumore_%')[f'F1_Score_{tt}'].mean().reset_index()
    ax.plot(s['Rumore_%'], s[f'F1_Score_{tt}'], color=color, linewidth=2, marker=marker,
            markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, label=lab, zorder=3)
ax.set_xlabel("Rumore nel training (%)")
ax.set_ylabel("F1 (media dei 4 modelli)")
ax.set_title("Dove si misura cambia quanto danno si vede", loc="left", pad=52)
ax.legend(frameon=False, loc="upper left", ncols=1, bbox_to_anchor=(0, 1.20))
style_axes(ax)
fig.tight_layout()
fig.savefig(os.path.join(QUI, "plot_A_sporco_vs_pulito.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Correlazioni per modello ---
righe = []
for m in modelli:
    s = a1[a1['Modello'] == m]
    for metrica in ['IM', 'IH']:
        r, p = pearsonr(s[metrica], s['F1_Score_test_pulito'])
        righe.append({'Modello': m, 'Metrica': metrica,
                      'Pearson_r': round(r, 4), 'p_value': round(p, 6)})
corr1 = pd.DataFrame(righe)
corr1.to_csv(os.path.join(QUI, 'A_correlazioni.csv'), index=False)

# --- Tabella Blocco 1 ---
t1 = a1.groupby('Rumore_%').agg(
    IM=('IM', 'mean'), IH=('IH', 'mean'),
    F1_sporco=('F1_Score_test_sporco', 'mean'),
    F1_pulito=('F1_Score_test_pulito', 'mean'),
).reset_index()
base = t1.loc[t1['Rumore_%'] == 0, 'F1_pulito'].iloc[0]
t1['Calo_dal_baseline'] = (base - t1['F1_pulito']).round(4)
t1['IM'] = t1['IM'].round(0).astype(int)
t1['IH'] = t1['IH'].round(0).astype(int)
t1['F1_sporco'] = t1['F1_sporco'].round(4)
t1['F1_pulito'] = t1['F1_pulito'].round(4)
t1.to_csv(os.path.join(QUI, 'A_tabella_blocco1.csv'), index=False)
salva_tabella_immagine(
    t1, "A_tabella_blocco1.png",
    "Braccio A — inconsistenza e F1 per livello di rumore",
    ["Rumore\n%", "IM", "IH", "F1\ntest sporco", "F1\ntest pulito", "Calo\ndal baseline"],
    col_widths=[0.7, 0.9, 0.9, 1.0, 1.0, 1.0],
)

# =================================================================
# 2. BLOCCO 2 — scaling del numero di FD (gia' solo braccio A)
# =================================================================
d2 = pd.read_csv(B2)
counts = sorted(d2['N_FD'].unique())
COL2 = dict(zip(counts, [BLUE, ORANGE, AQUA]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for n in counts:
    s = d2[d2['N_FD'] == n].groupby('Rumore_%')['F1_Score_test_pulito'].mean().reset_index()
    axes[0].plot(s['Rumore_%'], s['F1_Score_test_pulito'], color=COL2[n], linewidth=2,
                 marker='o', markersize=6, markeredgecolor=SURFACE, markeredgewidth=1,
                 label=f'{n} FD corrotte', zorder=3)
    si = d2[d2['N_FD'] == n].groupby('Rumore_%')['IM'].mean().reset_index()
    axes[1].plot(si['Rumore_%'], si['IM'], color=COL2[n], linewidth=2, marker='s',
                 markersize=6, markeredgecolor=SURFACE, markeredgewidth=1,
                 label=f'{n} FD corrotte', zorder=3)
axes[0].set_ylabel("F1 sul test pulito (media dei 4 modelli)")
axes[0].set_title("Effetto sul modello", loc="left", fontsize=11)
axes[1].set_ylabel("IM — conflitti a coppie")
axes[1].set_title("Inconsistenza misurata", loc="left", fontsize=11)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
for ax in axes:
    ax.set_xlabel("Rumore nel training (%)")
    ax.legend(frameon=False)
    style_axes(ax)
fig.suptitle("Piu' dipendenze funzionali si corrompono, piu' il modello peggiora",
             x=0.02, ha="left", fontsize=13, y=1.04)
fig.tight_layout()
fig.savefig(os.path.join(QUI, "plot_A_scaling_fd.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Correlazioni Blocco 2: intero range e regime >= 20% ---
righe2 = []
for n in counts:
    for m in modelli:
        s = d2[(d2['N_FD'] == n) & (d2['Modello'] == m)]
        r_tot, p_tot = pearsonr(s['IM'], s['F1_Score_test_pulito'])
        sa = s[s['Rumore_%'] >= 20]
        r_alt, p_alt = pearsonr(sa['IM'], sa['F1_Score_test_pulito'])
        righe2.append({'N_FD': n, 'Modello': m,
                       'r_intero_range': round(r_tot, 3), 'p_intero_range': round(p_tot, 6),
                       'r_rumore_ge20': round(r_alt, 3), 'p_rumore_ge20': round(p_alt, 6)})
corr2 = pd.DataFrame(righe2)
corr2.to_csv(os.path.join(QUI, 'A_correlazioni_scaling.csv'), index=False)

# --- Tabella Blocco 2 ---
t2 = d2.pivot_table(index='Rumore_%', columns='N_FD',
                    values='F1_Score_test_pulito', aggfunc='mean').round(4).reset_index()
im2 = d2.pivot_table(index='Rumore_%', columns='N_FD', values='IM', aggfunc='mean').round(0)
for n in counts:
    t2[f'IM_{n}'] = im2[n].astype(int).values
t2.columns = ['Rumore_%'] + [f'F1_{n}FD' for n in counts] + [f'IM_{n}FD' for n in counts]
t2.to_csv(os.path.join(QUI, 'A_tabella_blocco2.csv'), index=False)
salva_tabella_immagine(
    t2, "A_tabella_blocco2.png",
    "Braccio A — F1 sul test pulito e inconsistenza, per numero di FD corrotte",
    ["Rumore\n%"] + [f"F1\n{n} FD" for n in counts] + [f"IM\n{n} FD" for n in counts],
    col_widths=[0.6] + [0.9] * len(counts) + [1.0] * len(counts),
)

# =================================================================
# 3. SINTESI A SCHERMO
# =================================================================
print("\n=== BLOCCO 1 — braccio A, test pulito ===")
print(t1.to_string(index=False))
print("\n=== Correlazione IM/IH vs F1 (test pulito), per modello ===")
print(corr1.to_string(index=False))
print("\n=== BLOCCO 2 — F1 per numero di FD ===")
print(t2.to_string(index=False))
print("\n=== Correlazione IM vs F1: intero range vs regime rumore >= 20% ===")
print(corr2.to_string(index=False))
print("\nOutput scritti in:", QUI)
