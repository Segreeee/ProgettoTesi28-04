"""
Genera i grafici esplicativi e le tabelle riassuntive per la relazione, in
forma simmetrica per i due capitoli sperimentali.

Blocco 1 (blocco1_risultati_raw.csv + CSV di blocco1_analisi.py): una FD
(rotta -> distanza) con le colonne ridondanti.
Blocco 2 (CSV di blocco2_analisi.py): 1, 2 e 4 FD corrotte.

Ogni blocco ha: F1 sul test pulito, test sporco contro test pulito e tabella
riassuntiva. Il Blocco 2 ha in piu' il confronto a parita' di righe sporche e
la saturazione di IM.

Palette e regole di stile seguono lo skill "dataviz" (palette categoriale
validata assegnata in ordine fisso, niente doppio asse, griglia recessiva,
nessuna banda calcolata fra modelli diversi).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
PALETTE = [BLUE, ORANGE, AQUA, YELLOW]

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


def salva_tabella_immagine(df, filename, titolo, col_labels, col_widths, larghezza=11):
    fig, ax = plt.subplots(figsize=(larghezza, 0.9 + 0.5 * len(df)))
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


df1 = pd.read_csv('blocco1_risultati_raw.csv')
modelli = sorted(df1['Modello'].unique())
COLORI_MODELLI = dict(zip(modelli, PALETTE))

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

tab1 = df1.groupby('Rumore_%').agg(
    IM=('IM_mean', 'mean'), IP=('IP_mean', 'mean'),
    IH_approx=('IH_approx_mean', 'mean'), IH_esatto=('IH_esatto_mean', 'mean'),
    F1_sporco=('F1_Score_test_sporco', 'mean'), F1_pulito=('F1_Score_test_pulito', 'mean'),
).reset_index()
base1 = tab1.loc[tab1['Rumore_%'] == 0, 'F1_pulito'].iloc[0]
tab1['Calo'] = base1 - tab1['F1_pulito']
for c in ['IM', 'IP', 'IH_approx', 'IH_esatto']:
    tab1[c] = tab1[c].round(0).astype(int)
tab1[['F1_sporco', 'F1_pulito', 'Calo']] = tab1[['F1_sporco', 'F1_pulito', 'Calo']].round(4)
tab1.to_csv('tabella_riassuntiva_blocco1.csv', index=False)
vista1 = tab1.copy()
for c in ['F1_sporco', 'F1_pulito', 'Calo']:
    vista1[c] = [f"{v:.4f}" for v in tab1[c]]
salva_tabella_immagine(
    vista1, "tabella_riassuntiva_blocco1.png",
    "Blocco 1 — indici osservati e F1 (media dei 4 modelli) per livello di rumore",
    ["Rumore\n%", "IM", "IP", "IH\n2-approx", "IH\nesatto", "F1\ntest sporco", "F1\ntest pulito",
     "Calo\n(test pulito)"],
    [0.6, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0],
)
print("Blocco 1: plot_blocco1_f1_test_pulito.png, plot_blocco1_sporco_vs_pulito.png, "
      "tabella_riassuntiva_blocco1.csv/.png")

if not os.path.exists('blocco2_scomposizione.csv'):
    print("CSV di blocco2_analisi.py non presenti: eseguire prima blocco2_analisi.py. Sezione Blocco 2 saltata.")
else:
    agg2 = pd.read_csv('blocco2_aggregato.csv')
    scomp2 = pd.read_csv('blocco2_scomposizione.csv')
    counts = sorted(scomp2['N_FD'].unique())
    colori = dict(zip(counts, PALETTE))
    INDICI = [('IM_mean', "IM — conflitti a coppie"), ('IP_mean', "IP — tuple coinvolte"),
              ('IH_approx_mean', "IH — tuple da correggere (2-approx)")]
    indici2 = {colonna: agg2.groupby(['N_FD', 'Rumore_%'])[colonna].mean() for colonna, _ in INDICI}

    # IH esatto: con 1 FD dalla formula chiusa (risultati del Blocco 2), con 2 e
    # 4 FD dall'ILP di ih_esatto_blocco2.py. Un livello e' disegnato solo se
    # tutte le sue istanze (repliche x fold) sono state risolte all'ottimo:
    # una media sulle sole istanze facili sarebbe distorta.
    def ih_esatto_serie(n):
        if n == 1:
            return agg2[agg2['N_FD'] == 1].groupby('Rumore_%')['IH_esatto_mean'].mean()
        if not os.path.exists('blocco2_ih_esatto.csv'):
            return None
        e = pd.read_csv('blocco2_ih_esatto.csv')
        e = e[e['N_FD'] == n]
        completi = e.groupby('Rumore_%')['Risolto'].all()
        serie = e[e['Rumore_%'].isin(completi[completi].index)].groupby('Rumore_%')['IH_esatto'].mean()
        return pd.concat([pd.Series({0: 0.0}), serie]) if len(serie) else None

    def linea_ih_esatto(ax, n, colore, etichetta):
        serie = ih_esatto_serie(n)
        if serie is not None:
            ax.plot(serie.index, serie.values, color=colore, linewidth=1.6, linestyle='--',
                    marker='o', markersize=4, markerfacecolor=SURFACE, label=etichetta, zorder=3)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for n in counts:
        s = scomp2[scomp2['N_FD'] == n]
        linea(axes[0], s['Rumore_%'], s['F1_test_pulito'], colori[n], f'{n} FD corrotte')
        for ax, (colonna, _) in zip(axes[1:], INDICI):
            serie = indici2[colonna].loc[n]
            linea(ax, serie.index, serie.values, colori[n], f'{n} FD corrotte', marker='s')
        linea_ih_esatto(axes[3], n, colori[n], f'{n} FD, esatto')
    axes[0].set_ylabel("F1 sul test pulito (media dei 4 modelli)")
    axes[0].set_title("Effetto sul modello", loc="left", fontsize=11)
    for ax, (_, etichetta) in zip(axes[1:], INDICI):
        ax.set_ylabel(etichetta)
        ax.set_title(etichetta.split(' — ')[0] + " osservato", loc="left", fontsize=11)
        formato_migliaia(ax)
    for ax in axes:
        ax.set_xlabel("Rumore nel training (%)")
        ax.legend(frameon=False)
        stile(ax)
    axes[3].set_ylabel("IH — tuple da correggere")
    axes[3].set_title("IH: 2-approssimato (continuo) ed esatto (tratteggiato)", loc="left", fontsize=11)
    axes[3].legend(frameon=False, fontsize=8, ncols=2, loc="upper left")
    fig.suptitle("Blocco 2 — effetto sul modello e indici osservati, per numero di FD corrotte",
                 x=0.01, ha="left", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig("plot_blocco2_scaling_fd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(counts), figsize=(12, 4.6), sharey=True)
    for ax, n in zip(axes, counts):
        s = scomp2[scomp2['N_FD'] == n]
        linea(ax, s['Rumore_%'], s['F1_test_sporco'], ORANGE, 'Test su dati sporchi', marker='s')
        linea(ax, s['Rumore_%'], s['F1_test_pulito'], BLUE, 'Test su dati puliti')
        ax.set_title(f"{n} FD corrotte", loc="left", fontsize=11)
        ax.set_xlabel("Rumore nel training (%)")
        stile(ax)
    axes[0].set_ylabel("F1 (media dei 4 modelli)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Blocco 2 — dove si misura cambia quanto danno si vede",
                 x=0.02, ha="left", fontsize=13, y=1.12)
    fig.tight_layout()
    fig.savefig("plot_blocco2_sporco_vs_pulito.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for n in counts:
        s = scomp2[scomp2['N_FD'] == n].sort_values('Quota_train_sporca')
        linea(ax, 100 * s['Quota_train_sporca'], s['F1_test_pulito'], colori[n], f'{n} FD corrotte')
    ax.set_xlabel("Righe di training effettivamente sporcate (%)")
    ax.set_ylabel("F1 sul test pulito (media dei 4 modelli)")
    ax.set_title("F1 sul test pulito e righe di training effettivamente sporcate", loc="left", pad=40)
    ax.legend(frameon=False, loc="upper left", ncols=3, bbox_to_anchor=(0, 1.12))
    stile(ax)
    fig.tight_layout()
    fig.savefig("plot_blocco2_quota_sporca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Le tre misure a confronto con il danno, alla configurazione piu' corrotta:
    # il rumore e' la variabile controllata, gli indici sono osservati.
    n_max = counts[-1]
    s_f1 = scomp2[scomp2['N_FD'] == n_max].set_index('Rumore_%')['F1_test_pulito']
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    linea(axes[0], s_f1.index, s_f1.values, BLUE)
    axes[0].set_ylabel("F1 sul test pulito (media dei 4 modelli)")
    axes[0].set_title("Danno sul modello", loc="left", fontsize=11)
    for ax, (colonna, etichetta) in zip(axes[1:], INDICI):
        serie = indici2[colonna].loc[n_max]
        linea(ax, serie.index, serie.values, ORANGE, marker='s',
              etichetta='2-approssimato' if colonna == 'IH_approx_mean' else None)
        ax.set_ylabel(etichetta)
        ax.set_title(etichetta.split(' — ')[0], loc="left", fontsize=11)
        formato_migliaia(ax)
    linea_ih_esatto(axes[3], n_max, BLUE, 'esatto (ILP, livelli risolti)')
    axes[3].set_ylabel("IH — tuple da correggere")
    axes[3].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.set_xlabel("Rumore nel training (%)")
        stile(ax)
    fig.suptitle(f"Con {n_max} FD corrotte: danno sul modello e andamento delle tre misure",
                 x=0.01, ha="left", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig("plot_blocco2_indici_e_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    piv = scomp2.pivot_table(index='Rumore_%', columns='N_FD', values=['F1_test_pulito', 'Quota_train_sporca'])
    tabella = pd.DataFrame({'Rumore_%': piv.index})
    for n in counts:
        tabella[f'F1_{n}FD'] = piv[('F1_test_pulito', n)].round(4).values
    for n in counts:
        tabella[f'Quota_sporca_{n}FD'] = piv[('Quota_train_sporca', n)].round(3).values
    for sigla, colonna in [('IM', 'IM_mean'), ('IP', 'IP_mean'), ('IH', 'IH_approx_mean')]:
        for n in counts:
            tabella[f'{sigla}_{n}FD'] = indici2[colonna].loc[n].round(0).astype(int).values
    tabella.to_csv('tabella_riassuntiva_blocco2.csv', index=False)
    vista2 = tabella.copy()
    for n in counts:
        vista2[f'F1_{n}FD'] = [f"{v:.4f}" for v in tabella[f'F1_{n}FD']]
        vista2[f'Quota_sporca_{n}FD'] = [f"{100 * v:.1f}%" for v in tabella[f'Quota_sporca_{n}FD']]
        for sigla in ['IM', 'IP', 'IH']:
            vista2[f'{sigla}_{n}FD'] = [f"{v:,}".replace(",", ".") for v in tabella[f'{sigla}_{n}FD']]
    salva_tabella_immagine(
        vista2, "tabella_riassuntiva_blocco2.png",
        "Blocco 2 — F1 sul test pulito, quota di righe di training sporche e indici osservati, per numero di FD",
        ["Rumore\n%"] + [f"F1\n{n} FD" for n in counts] + [f"Sporche\n{n} FD" for n in counts]
        + [f"{sigla}\n{n} FD" for sigla in ['IM', 'IP', 'IH'] for n in counts],
        [0.7] + [0.8] * len(counts) + [0.8] * len(counts) + [0.9] * (3 * len(counts)),
        larghezza=18,
    )
    print("Blocco 2: plot_blocco2_scaling_fd.png, plot_blocco2_sporco_vs_pulito.png, plot_blocco2_quota_sporca.png, "
          "plot_blocco2_indici_e_f1.png, tabella_riassuntiva_blocco2.csv/.png")
