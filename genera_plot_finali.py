"""
Genera i grafici esplicativi e le tabelle riassuntive per la relazione.

Blocco 1 (blocco1_risultati_raw.csv): una FD (rotta -> distanza) con le
colonne ridondanti; modello addestrato sui dati sporcati e valutato sul test
pulito e, per confronto, sul test sporcato.
Blocco 2 (blocco2_risultati_raw.csv, se presente): 1, 5 e 10 FD corrotte.

Palette e regole di stile seguono lo skill "dataviz" (palette categoriale
validata assegnata in ordine fisso, niente doppio asse, griglia recessiva).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
PALETTE = [BLUE, ORANGE, AQUA, YELLOW]      # ordine fisso, mai ciclato

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY, "text.color": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "grid.color": GRID,
    "font.family": "sans-serif", "font.size": 10.5,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def stile(ax):
    ax.grid(axis="y", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(length=0)


def linea(ax, x, y, colore, etichetta=None, marker='o'):
    ax.plot(x, y, color=colore, linewidth=2, marker=marker, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1, label=etichetta, zorder=3)


def formato_migliaia(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))


def salva_tabella_immagine(df, filename, titolo, col_labels, col_widths):
    fig, ax = plt.subplots(figsize=(11, 0.9 + 0.5 * len(df)))
    ax.axis("off")
    ax.set_title(titolo, loc="left", fontsize=12, pad=16)
    table = ax.table(cellText=df.values, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    for col_idx, w in enumerate(col_widths):
        for row_idx in range(len(df) + 1):
            table[row_idx, col_idx].set_width(w / sum(col_widths))
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#eef2f8")
            cell.set_text_props(color=INK, fontweight="bold", wrap=True)
        else:
            cell.set_facecolor(SURFACE if row % 2 else "#f6f6f4")
            cell.set_text_props(color=INK)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =================================================================
# BLOCCO 1
# =================================================================
df1 = pd.read_csv('blocco1_risultati_raw.csv')
modelli = sorted(df1['Modello'].unique())
COLORI_MODELLI = dict(zip(modelli, PALETTE))

# --- Grafico 1: F1 sul test pulito, per modello (banda = std sulle repliche) ---
fig, ax = plt.subplots(figsize=(8, 5.2))
for m in modelli:
    s = df1[df1['Modello'] == m].groupby('Rumore_%')['F1_Score_test_pulito'].agg(['mean', 'std'])
    ax.fill_between(s.index, s['mean'] - s['std'], s['mean'] + s['std'],
                    color=COLORI_MODELLI[m], alpha=0.12, linewidth=0)
    linea(ax, s.index, s['mean'].values, COLORI_MODELLI[m], m)
ax.set_xlabel("Rumore nel training (%)")
ax.set_ylabel("F1 sul test pulito (± std sulle repliche)")
ax.set_title("Training sporcato, test su dati puliti: F1 per modello", loc="left", pad=62)
ax.legend(frameon=False, loc="upper left", ncols=2, bbox_to_anchor=(0, 1.22))
stile(ax)
fig.tight_layout()
fig.savefig("plot_blocco1_f1_test_pulito.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Grafico 2: test sporco vs test pulito (media dei 4 modelli) ---
# Nessuna banda: la media e' su 4 modelli diversi e la loro dispersione non e'
# un'incertezza della misura.
fig, ax = plt.subplots(figsize=(8, 5.2))
for tt, colore, marker, etichetta in [
        ('test_sporco', ORANGE, 's', 'Test su dati sporchi (il modello riceve input corrotti)'),
        ('test_pulito', BLUE, 'o', 'Test su dati puliti (misura quanto il modello ha imparato male)')]:
    s = df1.groupby('Rumore_%')[f'F1_Score_{tt}'].mean()
    linea(ax, s.index, s.values, colore, etichetta, marker)
ax.set_xlabel("Rumore nel training (%)")
ax.set_ylabel("F1 (media dei 4 modelli)")
ax.set_title("Dove si misura cambia quanto danno si vede", loc="left", pad=52)
ax.legend(frameon=False, loc="upper left", ncols=1, bbox_to_anchor=(0, 1.20))
stile(ax)
fig.tight_layout()
fig.savefig("plot_blocco1_sporco_vs_pulito.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Grafico 3: correlazione IM/IH vs F1 (test pulito), per modello ---
corr = pd.read_csv('blocco1_correlazioni.csv')
corr = corr[corr['Valutazione'] == 'test_pulito']
x = range(len(modelli))
fig, ax = plt.subplots(figsize=(8, 5.5))
for i, (metrica, colore, etichetta) in enumerate([('IM', BLUE, "IM (conflitti a coppie)"),
                                                  ('IH', AQUA, "IH (tuple minime da correggere)")]):
    valori = [corr[(corr['Modello'] == m) & (corr['Metrica_Inconsistenza'] == metrica)]['Pearson_r'].iloc[0]
              for m in modelli]
    ax.bar([j + (i - 0.5) * 0.35 for j in x], valori, width=0.35, color=colore, label=etichetta, zorder=3)
ax.axhline(0, color=AXIS, linewidth=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels(modelli)
ax.set_ylim(-1.05, 1.05)
ax.set_ylabel("Correlazione di Pearson con F1 sul test pulito")
ax.set_title("Inconsistenza del training e qualita' delle predizioni", loc="left", pad=48)
ax.legend(frameon=False, loc="upper left", ncols=2, bbox_to_anchor=(0, 1.14))
stile(ax)
fig.tight_layout()
fig.savefig("plot_blocco1_correlazioni.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Tabella Blocco 1 ---
tab1 = df1.groupby('Rumore_%').agg(
    IM=('IM', 'mean'), IH=('IH', 'mean'),
    F1_sporco=('F1_Score_test_sporco', 'mean'), F1_pulito=('F1_Score_test_pulito', 'mean'),
).reset_index()
base1 = tab1.loc[tab1['Rumore_%'] == 0, 'F1_pulito'].iloc[0]
tab1['Calo'] = base1 - tab1['F1_pulito']
tab1['IM'] = tab1['IM'].round(0).astype(int)
tab1['IH'] = tab1['IH'].round(0).astype(int)
tab1[['F1_sporco', 'F1_pulito', 'Calo']] = tab1[['F1_sporco', 'F1_pulito', 'Calo']].round(4)
tab1.to_csv('tabella_riassuntiva_blocco1.csv', index=False)
vista1 = tab1.copy()
for c in ['F1_sporco', 'F1_pulito', 'Calo']:
    vista1[c] = [f"{v:.4f}" for v in tab1[c]]
salva_tabella_immagine(
    vista1, "tabella_riassuntiva_blocco1.png",
    "Blocco 1 — inconsistenza e F1 (media dei 4 modelli) per livello di rumore",
    ["Rumore\n%", "IM", "IH", "F1\ntest sporco", "F1\ntest pulito", "Calo\n(test pulito)"],
    [0.7, 0.9, 0.9, 1.0, 1.0, 1.0],
)
print("Blocco 1: plot_blocco1_f1_test_pulito.png, plot_blocco1_sporco_vs_pulito.png, "
      "plot_blocco1_correlazioni.png, tabella_riassuntiva_blocco1.csv/.png")

# =================================================================
# BLOCCO 2 (se disponibile)
# =================================================================
if os.path.exists('blocco2_risultati_raw.csv'):
    df2 = pd.read_csv('blocco2_risultati_raw.csv')
    counts = sorted(df2['N_FD'].unique())
    colori = dict(zip(counts, PALETTE))

    # --- Grafico 4: F1 e IM al variare del numero di FD corrotte ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for n in counts:
        sub = df2[df2['N_FD'] == n]
        s = sub.groupby('Rumore_%')['F1_Score_test_pulito'].mean()
        linea(axes[0], s.index, s.values, colori[n], f'{n} FD corrotte')
        si = sub.drop_duplicates(['Rumore_%', 'Rep']).groupby('Rumore_%')['IM'].mean()
        linea(axes[1], si.index, si.values, colori[n], f'{n} FD corrotte', marker='s')
    axes[0].set_ylabel("F1 sul test pulito (media dei 4 modelli)")
    axes[0].set_title("Effetto sul modello", loc="left", fontsize=11)
    axes[1].set_ylabel("IM — conflitti a coppie")
    axes[1].set_title("Inconsistenza misurata", loc="left", fontsize=11)
    formato_migliaia(axes[1])
    for ax in axes:
        ax.set_xlabel("Rumore nel training (%)")
        ax.legend(frameon=False)
        stile(ax)
    fig.suptitle("Blocco 2 — cosa succede aumentando il numero di FD corrotte",
                 x=0.02, ha="left", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig("plot_blocco2_scaling_fd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Grafico 5: IM e F1 con il numero massimo di FD ---
    # Due pannelli sulla stessa ascissa, mai un doppio asse. Il titolo e la zona
    # ombreggiata dipendono dai dati: si segnala un'inversione solo se IM
    # raggiunge il massimo prima dell'ultimo livello di rumore.
    n_max = counts[-1]
    sub = df2[df2['N_FD'] == n_max]
    s_f1 = sub.groupby('Rumore_%')['F1_Score_test_pulito'].mean()
    s_im = sub.drop_duplicates(['Rumore_%', 'Rep']).groupby('Rumore_%')['IM'].mean()
    picco = int(s_im.idxmax())
    ultimo = int(s_im.index.max())
    inversione = picco < ultimo

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    if inversione:
        for ax in axes:
            ax.axvspan(picco, ultimo, color=INK_MUTED, alpha=0.08, linewidth=0)
    linea(axes[0], s_f1.index, s_f1.values, BLUE)
    axes[0].set_ylabel("F1 sul test pulito (media dei 4 modelli)")
    axes[0].set_title("Danno sul modello", loc="left", fontsize=11)
    linea(axes[1], s_im.index, s_im.values, ORANGE, marker='s')
    axes[1].set_ylabel("IM — conflitti a coppie")
    axes[1].set_title("Inconsistenza misurata", loc="left", fontsize=11)
    formato_migliaia(axes[1])
    for ax in axes:
        ax.set_xlabel("Rumore nel training (%)")
        stile(ax)
    titolo = (f"Con {n_max} FD corrotte, oltre il {picco}% di rumore IM cala mentre il danno cresce"
              if inversione else f"Con {n_max} FD corrotte, IM e danno crescono insieme con il rumore")
    fig.suptitle(titolo, x=0.02, ha="left", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig("plot_blocco2_im_e_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Tabella Blocco 2 ---
    tab2 = df2.pivot_table(index='Rumore_%', columns='N_FD',
                           values='F1_Score_test_pulito', aggfunc='mean').round(4)
    im2 = df2.drop_duplicates(['N_FD', 'Rumore_%', 'Rep']).pivot_table(
        index='Rumore_%', columns='N_FD', values='IM', aggfunc='mean').round(0)
    tabella = pd.DataFrame({'Rumore_%': tab2.index})
    for n in counts:
        tabella[f'F1_{n}FD'] = tab2[n].values
    for n in counts:
        tabella[f'IM_{n}FD'] = im2[n].astype(int).values
    tabella.to_csv('tabella_riassuntiva_blocco2.csv', index=False)
    vista2 = tabella.copy()
    for n in counts:
        vista2[f'F1_{n}FD'] = [f"{v:.4f}" for v in tabella[f'F1_{n}FD']]
        vista2[f'IM_{n}FD'] = [f"{v:,}".replace(",", ".") for v in tabella[f'IM_{n}FD']]
    salva_tabella_immagine(
        vista2, "tabella_riassuntiva_blocco2.png",
        "Blocco 2 — F1 sul test pulito e inconsistenza, per numero di FD corrotte",
        ["Rumore\n%"] + [f"F1\n{n} FD" for n in counts] + [f"IM\n{n} FD" for n in counts],
        [0.6] + [0.9] * len(counts) + [1.1] * len(counts),
    )
    print("Blocco 2: plot_blocco2_scaling_fd.png, plot_blocco2_im_e_f1.png, tabella_riassuntiva_blocco2.csv/.png")
else:
    print("blocco2_risultati_raw.csv non presente: sezione Blocco 2 saltata.")
