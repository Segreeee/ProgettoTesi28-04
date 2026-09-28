"""
Palette e funzioni di stile copiate da genera_plot_finali.py del progetto
principale. Non si importa quel file perche' al caricamento rigenera i grafici
del progetto principale.
"""
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


def barre_correlazioni(ax, corr, modelli, titolo):
    """Spearman IM e IH contro F1 sul test pulito, una coppia di barre per modello."""
    x = range(len(modelli))
    for i, (metrica, colore, etichetta) in enumerate([('IM', BLUE, "IM (conflitti a coppie)"),
                                                      ('IH', AQUA, "IH (tuple minime da correggere)")]):
        valori = [corr[(corr['Modello'] == m) & (corr['Metrica_Inconsistenza'] == metrica)]['Spearman_rho'].iloc[0]
                  for m in modelli]
        ax.bar([j + (i - 0.5) * 0.35 for j in x], valori, width=0.35, color=colore, label=etichetta, zorder=3)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.replace(' ', '\n') for m in modelli])
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(titolo, loc="left", fontsize=11)
    stile(ax)


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
