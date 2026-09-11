"""
Genera i grafici esplicativi e le tabelle riassuntive per la relazione di tesi.

Blocco 1 (blocco1_risultati_raw.csv): disegno a due bracci sul target
DurataBucket, con doppia valutazione (test sporco / test pulito).
Blocco 2 (blocco2_risultati_raw.csv, opzionale): scaling 1 -> 5 -> 10 FD.

Palette e regole di stile seguono lo skill "dataviz" (palette categoriale
validata, un solo hue per serie, niente doppio asse, griglia recessiva).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------
# Palette (dataviz skill, modalita' light)
# ---------------------------------------------------------------
BLUE = "#2a78d6"      # slot 1 - Braccio A (incoerente)
ORANGE = "#eb6834"    # slot 2 - Braccio C (coerente)
AQUA = "#1baf7a"       # slot 3 - serie accessoria
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRID,
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
})

ARMS = [
    ('A', BLUE, 'o', 'Braccio A — incoerente (viola la FD)'),
    ('C', ORANGE, 's', 'Braccio C — coerente (preserva la FD)'),
]


def style_axes(ax):
    ax.grid(axis="y", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0)


def serie(df, arm, tt):
    """F1 medio sui 4 modelli (± std) per livello di rumore."""
    sub = df[df['Arm'] == arm]
    return (sub.groupby('Rumore_%')[f'F1_Score_{tt}']
               .agg(['mean', 'std']).reset_index().sort_values('Rumore_%'))


# =================================================================
# BLOCCO 1
# =================================================================
df1 = pd.read_csv('blocco1_risultati_raw.csv')

# --- Grafico 1 (PRINCIPALE): F1 vs rumore sul TEST PULITO ---
# NOTA: nei grafici aggregati non si disegna una banda di dispersione. La
# deviazione standard su queste serie e' calcolata su 4 modelli diversi messi
# insieme, quindi misura la differenza FRA modelli (la rete neurale sta molto
# sotto il decision tree), non l'incertezza della stima: presentarla come banda
# sarebbe fuorviante. La variabilita' vera, fra le 5 repliche di uno stesso
# modello, e' mostrata nel grafico per modello.
fig, ax = plt.subplots(figsize=(7.5, 5))
for arm, color, marker, label in ARMS:
    s = serie(df1, arm, 'test_pulito')
    ax.plot(s['Rumore_%'], s['mean'], color=color, linewidth=2, marker=marker,
            markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, label=label, zorder=3)
ax.set_xlabel("Rumore nel training (%)")
ax.set_ylabel("F1 sul test pulito (media dei 4 modelli)")
ax.set_title("Training sporcato, test su dati puliti:\nentrambi i tipi di errore degradano le predizioni",
             loc="left", pad=52)
ax.legend(frameon=False, loc="upper left", ncols=1, bbox_to_anchor=(0, 1.20))
style_axes(ax)
fig.tight_layout()
fig.savefig("plot_blocco1_f1_test_pulito.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Grafico 2: confronto metodologico test sporco vs test pulito ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, tt, titolo in zip(
        axes, ['test_sporco', 'test_pulito'],
        ["Test sui dati SPORCHI (metodo precedente)", "Test sui dati PULITI (metodo corretto)"]):
    for arm, color, marker, label in ARMS:
        s = serie(df1, arm, tt)
        ax.plot(s['Rumore_%'], s['mean'], color=color, linewidth=2, marker=marker,
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, label=label, zorder=3)
    ax.set_title(titolo, loc="left", fontsize=11)
    ax.set_xlabel("Rumore nel training (%)")
    style_axes(ax)
axes[0].set_ylabel("F1 medio (4 modelli)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.06))
fig.suptitle("Perche' il dataset di test cambia la conclusione", x=0.02, ha="left", fontsize=13, y=1.13)
fig.tight_layout()
fig.savefig("plot_blocco1_sporco_vs_pulito.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Grafico 3: dettaglio per modello, test pulito ---
modelli = sorted(df1['Modello'].unique())
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
axes = axes.flatten()
for ax, modello in zip(axes, modelli):
    for arm, color, marker, label in ARMS:
        sub = df1[(df1['Modello'] == modello) & (df1['Arm'] == arm)]
        s = sub.groupby('Rumore_%')['F1_Score_test_pulito'].agg(['mean', 'std']).reset_index()
        ax.fill_between(s['Rumore_%'], s['mean'] - s['std'], s['mean'] + s['std'],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(s['Rumore_%'], s['mean'], color=color, linewidth=2, marker=marker,
                markersize=5, markeredgecolor=SURFACE, markeredgewidth=1, label=label, zorder=3)
    ax.set_title(modello, loc="left", fontsize=11)
    style_axes(ax)
for ax in axes[2:]:
    ax.set_xlabel("Rumore nel training (%)")
for ax in [axes[0], axes[2]]:
    ax.set_ylabel("F1 sul test pulito")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.03))
fig.suptitle("Blocco 1 — F1 sul test pulito, per modello", x=0.02, ha="left", fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("plot_blocco1_f1_per_modello.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Grafico 4: correlazione IM/IH vs F1 (braccio A, test pulito) ---
corr = pd.read_csv('blocco1_correlazioni.csv')
corr = corr[corr['Valutazione'] == 'test_pulito']
x = range(len(modelli))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5.5))
im_vals = [corr[(corr['Modello'] == m) & (corr['Metrica_Inconsistenza'] == 'IM')]['Pearson_r'].iloc[0] for m in modelli]
ih_vals = [corr[(corr['Modello'] == m) & (corr['Metrica_Inconsistenza'] == 'IH')]['Pearson_r'].iloc[0] for m in modelli]
ax.bar([i - width / 2 for i in x], im_vals, width=width, color=BLUE, label="IM (conflitti a coppie)", zorder=3)
ax.bar([i + width / 2 for i in x], ih_vals, width=width, color=AQUA, label="IH (tuple minime da correggere)", zorder=3)
ax.axhline(0, color=AXIS, linewidth=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels(modelli)
ax.set_ylim(-1.05, 0.1)
ax.set_ylabel("Correlazione di Pearson con F1 (test pulito)")
ax.set_title("Piu' cresce l'inconsistenza nel training,\npiu' peggiorano le predizioni sui dati puliti",
             loc="left", pad=52)
ax.legend(frameon=False, loc="upper left", ncols=2, bbox_to_anchor=(0, 1.16))
style_axes(ax)
fig.tight_layout()
fig.savefig("plot_blocco1_correlazioni.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Grafici Blocco 1 salvati: plot_blocco1_f1_test_pulito.png, "
      "plot_blocco1_sporco_vs_pulito.png, plot_blocco1_f1_per_modello.png, "
      "plot_blocco1_correlazioni.png")


# --- Tabella riassuntiva Blocco 1 ---
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


tab = df1[df1['Arm'] == 'A'].groupby('Rumore_%').agg(IM=('IM', 'mean'), IH=('IH', 'mean')).reset_index()
for arm in ['A', 'C']:
    for tt in ['test_sporco', 'test_pulito']:
        m = df1[df1['Arm'] == arm].groupby('Rumore_%')[f'F1_Score_{tt}'].mean().reset_index()
        tab[f'{arm}_{tt}'] = m[f'F1_Score_{tt}'].round(4).values
tab['IM'] = tab['IM'].round(0).astype(int)
tab['IH'] = tab['IH'].round(0).astype(int)
tab['Delta_pulito'] = (tab['A_test_pulito'] - tab['C_test_pulito']).round(4)
tab = tab[['Rumore_%', 'IM', 'IH', 'A_test_sporco', 'C_test_sporco',
           'A_test_pulito', 'C_test_pulito', 'Delta_pulito']]
tab.to_csv('tabella_riassuntiva_blocco1.csv', index=False)
salva_tabella_immagine(
    tab, "tabella_riassuntiva_blocco1.png",
    "Blocco 1 — F1 per livello di rumore: test sporco vs test pulito, bracci A e C",
    ["Rumore\n%", "IM\n(braccio A)", "IH\n(braccio A)",
     "F1 A\ntest sporco", "F1 C\ntest sporco",
     "F1 A\ntest pulito", "F1 C\ntest pulito", "Delta A-C\ntest pulito"],
    col_widths=[0.6, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0],
)
print("Tabella Blocco 1 salvata: tabella_riassuntiva_blocco1.csv/.png")


# =================================================================
# BLOCCO 2 (se disponibile)
# =================================================================
if os.path.exists('blocco2_risultati_raw.csv'):
    df2 = pd.read_csv('blocco2_risultati_raw.csv')
    counts = sorted(df2['N_FD'].unique())
    colori = {counts[0]: BLUE, counts[1]: ORANGE, counts[2]: AQUA} if len(counts) >= 3 else {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    # F1 vs rumore, una serie per numero di FD
    for n in counts:
        s = (df2[df2['N_FD'] == n].groupby('Rumore_%')['F1_Score_test_pulito']
             .mean().reset_index())
        axes[0].plot(s['Rumore_%'], s['F1_Score_test_pulito'], color=colori.get(n, INK_MUTED),
                     linewidth=2, marker='o', markersize=6, markeredgecolor=SURFACE,
                     markeredgewidth=1, label=f'{n} FD corrotte', zorder=3)
    axes[0].set_xlabel("Rumore nel training (%)")
    axes[0].set_ylabel("F1 sul test pulito (media dei 4 modelli)")
    axes[0].set_title("Effetto sul modello", loc="left", fontsize=11)
    axes[0].legend(frameon=False)
    style_axes(axes[0])

    # IM vs rumore, una serie per numero di FD
    for n in counts:
        s = df2[df2['N_FD'] == n].groupby('Rumore_%')['IM'].mean().reset_index()
        axes[1].plot(s['Rumore_%'], s['IM'], color=colori.get(n, INK_MUTED), linewidth=2,
                     marker='o', markersize=6, markeredgecolor=SURFACE, markeredgewidth=1,
                     label=f'{n} FD corrotte', zorder=3)
    axes[1].set_xlabel("Rumore nel training (%)")
    axes[1].set_ylabel("IM — conflitti a coppie")
    axes[1].set_title("Inconsistenza misurata", loc="left", fontsize=11)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
    axes[1].legend(frameon=False)
    style_axes(axes[1])

    fig.suptitle("Blocco 2 — cosa succede aumentando il numero di FD corrotte",
                 x=0.02, ha="left", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig("plot_blocco2_scaling_fd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Grafico dedicato: l'inversione di segno di IM (10 FD) ---
    # Due pannelli affiancati sulla stessa scala delle ascisse: MAI un doppio
    # asse y. A destra IM, che cresce e poi cala; a sinistra F1, che peggiora
    # sempre. Nella zona ombreggiata (>=20%) le due curve vanno in direzioni
    # opposte: IM smette di tracciare il danno.
    n_max = counts[-1]
    s_f1 = (df2[df2['N_FD'] == n_max].groupby('Rumore_%')['F1_Score_test_pulito']
            .mean().reset_index())
    s_im = df2[df2['N_FD'] == n_max].groupby('Rumore_%')['IM'].mean().reset_index()
    picco = int(s_im.loc[s_im['IM'].idxmax(), 'Rumore_%'])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].axvspan(picco, 40, color=INK_MUTED, alpha=0.08, linewidth=0)
    axes[0].plot(s_f1['Rumore_%'], s_f1['F1_Score_test_pulito'], color=BLUE, linewidth=2,
                 marker='o', markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
    axes[0].set_ylabel("F1 sul test pulito (media dei 4 modelli)")
    axes[0].set_title("Il danno continua a crescere", loc="left", fontsize=11)

    axes[1].axvspan(picco, 40, color=INK_MUTED, alpha=0.08, linewidth=0)
    axes[1].plot(s_im['Rumore_%'], s_im['IM'], color=ORANGE, linewidth=2, marker='s',
                 markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
    axes[1].axvline(picco, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
    axes[1].annotate(f"picco al {picco}%,\npoi cala", xy=(picco, s_im['IM'].max()),
                     xytext=(picco + 3, s_im['IM'].max() * 0.72), fontsize=9, color=INK_SECONDARY)
    axes[1].set_ylabel("IM — conflitti a coppie")
    axes[1].set_title("Ma l'inconsistenza misurata cala", loc="left", fontsize=11)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))

    for ax in axes:
        ax.set_xlabel("Rumore nel training (%)")
        style_axes(ax)
    fig.suptitle(f"Con {n_max} FD corrotte, oltre il {picco}% di rumore IM smette di tracciare il danno",
                 x=0.02, ha="left", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig("plot_blocco2_im_inversione.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    tab2 = df2.pivot_table(index='Rumore_%', columns='N_FD',
                           values='F1_Score_test_pulito', aggfunc='mean').round(4).reset_index()
    im2 = df2.pivot_table(index='Rumore_%', columns='N_FD', values='IM', aggfunc='mean').round(0)
    for n in counts:
        tab2[f'IM_{n}'] = im2[n].astype(int).values
    tab2.columns = ['Rumore_%'] + [f'F1_{n}FD' for n in counts] + [f'IM_{n}FD' for n in counts]
    tab2.to_csv('tabella_riassuntiva_blocco2.csv', index=False)
    salva_tabella_immagine(
        tab2, "tabella_riassuntiva_blocco2.png",
        "Blocco 2 — F1 sul test pulito e inconsistenza, per numero di FD corrotte",
        ["Rumore\n%"] + [f"F1\n{n} FD" for n in counts] + [f"IM\n{n} FD" for n in counts],
        col_widths=[0.6] + [0.9] * len(counts) + [1.0] * len(counts),
    )
    print("Grafico e tabella Blocco 2 salvati.")
else:
    print("blocco2_risultati_raw.csv non ancora presente: sezione Blocco 2 saltata.")
