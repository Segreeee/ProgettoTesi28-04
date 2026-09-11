"""
Blocco 2 — grafici e tabella riassuntiva (roadmap par. 9).

Stile copiato da ../genera_plot_finali.py (palette dataviz validata, un solo
hue per serie, niente doppio asse, griglia recessiva). Non lo importa: quel
file esegue codice all'importazione e questa cartella deve restare autonoma.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config as C

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CONFIGS = list(C.CONFIGURAZIONI)
COLORI = dict(zip(CONFIGS, [BLUE, ORANGE, AQUA, YELLOW]))       # ordine fisso, mai ciclato
ETICHETTE = {
    'C1': 'C1 — FD1 (rotta)',
    'C2': 'C2 — + FD2 (fascia distanza)',
    'C3': 'C3 — + FD3 (fascia oraria)',
    'C4': 'C4 — + FD4 (geografia di partenza)',
}
METRICA = 'F1_weighted'

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


def salva(fig, nome):
    percorso = os.path.join(C.QUI, nome)
    fig.savefig(percorso, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return nome


def salva_tabella(df, nome, titolo, intestazioni, larghezze):
    fig, ax = plt.subplots(figsize=(11, 0.9 + 0.5 * len(df)))
    ax.axis("off")
    ax.set_title(titolo, loc="left", fontsize=12, pad=16)
    tab = ax.table(cellText=df.values, colLabels=intestazioni, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 2.0)
    for ci, w in enumerate(larghezze):
        for ri in range(len(df) + 1):
            tab[ri, ci].set_width(w / sum(larghezze))
    for (r, c), cella in tab.get_celld().items():
        cella.set_edgecolor(GRID)
        if r == 0:
            cella.set_facecolor("#eef2f8")
            cella.set_text_props(color=INK, fontweight="bold", wrap=True)
        else:
            cella.set_facecolor(SURFACE if r % 2 else "#f6f6f4")
            cella.set_text_props(color=INK)
    fig.tight_layout()
    return salva(fig, nome)


def main():
    df = pd.read_csv(C.RISULTATI_RAW)
    prodotti = []

    # 1. PRINCIPALE: F1 vs rumore, una curva per configurazione.
    # Nessuna banda: la media e' su 4 modelli diversi, e la loro dispersione
    # non e' un'incertezza (la variabilita' fra fold e' nel grafico 2).
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for cfg in CONFIGS:
        s = df[df['Config_FD'] == cfg].groupby('Rumore_%')[METRICA].mean()
        linea(ax, s.index, s.values, COLORI[cfg], ETICHETTE[cfg])
    # C2, C3 e C4 sono quasi identiche e si coprono: lo si dichiara, altrimenti
    # una serie in legenda sembrerebbe mancare dal grafico.
    y30 = df[(df['Config_FD'] == 'C3') & (df['Rumore_%'] == 30)][METRICA].mean()
    ax.annotate("C2, C3 e C4 quasi sovrapposte", xy=(30, y30), xytext=(16, y30 + 0.035),
                fontsize=9, color=INK_SECONDARY,
                arrowprops=dict(arrowstyle='-', color=INK_MUTED, lw=0.8))
    ax.set_xlabel("Righe di training sporcate (%)")
    ax.set_ylabel("F1 sul test pulito (media dei 4 modelli)")
    ax.set_title("Training sporcato, test pulito: il degrado per configurazione di FD",
                 loc="left", pad=62)
    ax.legend(frameon=False, loc="upper left", ncols=2, bbox_to_anchor=(0, 1.22))
    stile(ax)
    fig.tight_layout()
    prodotti.append(salva(fig, "plot_f1_per_configurazione.png"))

    # 2. Un pannello per modello, banda = deviazione standard fra i fold.
    modelli = sorted(df['Modello'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    for ax, modello in zip(axes.flatten(), modelli):
        for cfg in CONFIGS:
            s = (df[(df['Modello'] == modello) & (df['Config_FD'] == cfg)]
                 .groupby('Rumore_%')[METRICA].agg(['mean', 'std']))
            ax.fill_between(s.index, s['mean'] - s['std'], s['mean'] + s['std'],
                            color=COLORI[cfg], alpha=0.12, linewidth=0)
            linea(ax, s.index, s['mean'].values, COLORI[cfg], ETICHETTE[cfg])
        # Le scale verticali sono diverse fra i pannelli: si riporta il calo reale,
        # altrimenti un calo di un punto sembrerebbe grande quanto uno di venti.
        dati = df[df['Modello'] == modello]
        base = dati[dati['Rumore_%'] == 0][METRICA].mean()
        cali = [base - dati[(dati['Config_FD'] == cfg) & (dati['Rumore_%'] == 40)][METRICA].mean()
                for cfg in CONFIGS]
        ax.text(0.02, 0.04, f"calo al 40%: {min(cali) * 100:.1f}–{max(cali) * 100:.1f} punti di F1",
                transform=ax.transAxes, fontsize=9, color=INK_SECONDARY)
        ax.set_title(modello, loc="left", fontsize=11)
        stile(ax)
    for ax in axes[1]:
        ax.set_xlabel("Righe di training sporcate (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("F1 sul test pulito (± std sui fold)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("F1 sul test pulito, per modello — attenzione: scala verticale diversa in ogni pannello",
                 x=0.02, ha="left", fontsize=13, y=1.07)
    fig.tight_layout()
    prodotti.append(salva(fig, "plot_f1_per_modello.png"))

    # 3. IM e IH vs rumore: pannelli affiancati, mai un doppio asse.
    misure = df.drop_duplicates(['Config_FD', 'Rumore_%', 'Fold']).groupby(['Config_FD', 'Rumore_%'])[['IM', 'IH']].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for cfg in CONFIGS:
        s = misure.loc[cfg]
        s_pos = s[s.index > 0]
        linea(axes[0], s_pos.index, s_pos['IM'].values, COLORI[cfg], ETICHETTE[cfg])
        linea(axes[1], s.index, s['IH'].values, COLORI[cfg], ETICHETTE[cfg], marker='s')
    axes[0].set_yscale('log')
    axes[0].set_ylabel("IM — coppie in conflitto (scala logaritmica)")
    axes[0].set_title("IM: cresce di ordini di grandezza", loc="left", fontsize=11)
    axes[1].set_ylabel("IH — righe da correggere")
    axes[1].set_title("IH: cresce piu' lentamente", loc="left", fontsize=11)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
    for ax in axes:
        ax.set_xlabel("Righe di training sporcate (%)")
        stile(ax)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.10))
    fig.suptitle("Inconsistenza misurata sul training sporcato", x=0.02, ha="left", fontsize=13, y=1.17)
    fig.tight_layout()
    prodotti.append(salva(fig, "plot_inconsistenza.png"))

    # 4. Correlazioni IM/IH con F1, a barre, per modello.
    corr = pd.read_csv(os.path.join(C.QUI, 'blocco2_correlazioni.csv'))
    corr = corr[corr['Ambito'] == 'tutte le configurazioni']
    x = range(len(modelli))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (metrica, colore) in enumerate([('IM', BLUE), ('IH', AQUA)]):
        valori = [corr[(corr['Modello'] == m) & (corr['Metrica'] == metrica)]['Spearman_rho'].iloc[0] for m in modelli]
        ax.bar([j + (i - 0.5) * 0.35 for j in x], valori, width=0.35, color=colore,
               label=f"{metrica} vs F1", zorder=3)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(modelli)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Correlazione di Spearman con F1 sul test pulito")
    ax.set_title("Inconsistenza del training e qualita' delle predizioni", loc="left", pad=48)
    ax.legend(frameon=False, loc="upper left", ncols=2, bbox_to_anchor=(0, 1.14))
    stile(ax)
    fig.tight_layout()
    prodotti.append(salva(fig, "plot_correlazioni.png"))

    # 5. Tabella riassuntiva.
    f1 = df.pivot_table(index='Rumore_%', columns='Config_FD', values=METRICA, aggfunc='mean').round(4)
    im = misure['IM'].unstack(0).round(0).astype(int)
    tab = pd.DataFrame({'Rumore %': f1.index})
    for cfg in CONFIGS:
        tab[f'F1 {cfg}'] = f1[cfg].values
    for cfg in CONFIGS:
        tab[f'IM {cfg}'] = im[cfg].values
    tab.to_csv(os.path.join(C.QUI, 'tabella_riassuntiva.csv'), index=False)   # valori numerici
    vista = tab.copy()                                                         # valori formattati
    for cfg in CONFIGS:
        vista[f'F1 {cfg}'] = [f"{v:.4f}" for v in tab[f'F1 {cfg}']]
        vista[f'IM {cfg}'] = [f"{v:,}".replace(",", ".") for v in tab[f'IM {cfg}']]
    prodotti.append(salva_tabella(
        vista, "tabella_riassuntiva.png",
        "F1 sul test pulito (media dei 4 modelli) e IM sul training sporcato, per configurazione",
        ["Rumore\n%"] + [f"F1\n{c}" for c in CONFIGS] + [f"IM\n{c}" for c in CONFIGS],
        [0.6] + [0.9] * len(CONFIGS) + [1.1] * len(CONFIGS)))

    print("Prodotti:", ", ".join(prodotti))


if __name__ == "__main__":
    main()
