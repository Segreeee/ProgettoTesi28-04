"""
Controlli automatici del motore, da eseguire prima del run lungo.

1. IM e IP coincidono con la definizione (confronto di tutte le coppie) e con
   un grafo networkx costruito esplicitamente.
2. La copertura IH e' valida: togliendo quelle righe non resta alcun conflitto.
3. Garanzia training sporco / test pulito:
   - righe di training modificate = (righe * livello) // 100, e 0 a rumore 0%;
   - ogni cella sporcata ha un valore diverso dall'originale;
   - la funzione di sporcatura non modifica il suo input;
   - il test del fold resta identico al dataset pulito.
4. Configurazioni annidate: C(k+1) coincide con C(k) su tutte le colonne di C(k).

Esce con codice 1 se un controllo fallisce.
"""
import sys
import itertools
import warnings

import numpy as np
import pandas as pd
import networkx as nx

import config as C
import motore as M

warnings.filterwarnings('ignore')
ESITI = []


def controlla(descrizione, condizione, dettaglio=''):
    ESITI.append(bool(condizione))
    print(f"  [{'OK' if condizione else 'FALLITO'}] {descrizione}" + (f"  ({dettaglio})" if dettaglio else ''))


def forza_bruta(df, fds):
    """IM e IP dalla definizione: ogni coppia di righe confrontata esplicitamente."""
    d = df.reset_index(drop=True)
    coinvolte, archi = set(), 0
    for i, j in itertools.combinations(range(len(d)), 2):
        if any(all(d.at[i, c] == d.at[j, c] for c in lhs) and d.at[i, rhs] != d.at[j, rhs]
               for lhs, rhs in fds):
            archi += 1
            coinvolte.update((i, j))
    return archi, len(coinvolte)


def networkx_im_ip(df, fds):
    d = df.reset_index(drop=True)
    G = nx.Graph()
    G.add_nodes_from(range(len(d)))
    for lhs, rhs in fds:
        for _, g in d.groupby(list(lhs)):
            valori = g[rhs]
            if valori.nunique() <= 1:
                continue
            gruppi = [list(ix) for ix in valori.groupby(valori).groups.values()]
            for a, b in itertools.combinations(gruppi, 2):
                G.add_edges_from(itertools.product(a, b))
    return G.number_of_edges(), sum(1 for n in G.nodes if G.degree(n) > 0)


def main():
    campione = pd.read_csv(C.CAMPIONE_10K, low_memory=False)

    print("1. Misure di inconsistenza contro la definizione e contro networkx")
    for config in C.CONFIGURAZIONI:
        fds = C.fds_di(config)
        for livello in [10, 40]:
            sporco, _ = M.sporca_training(campione, C.colonne_da_sporcare(config), livello, seed=7)
            m = M.metriche_inconsistenza(sporco, fds)
            im_nx, ip_nx = networkx_im_ip(sporco, fds)
            controlla(f"{config} {livello}%: IM e IP uguali a networkx", (m['IM'], m['IP']) == (im_nx, ip_nx),
                      f"IM {m['IM']:,} / {im_nx:,}  IP {m['IP']:,} / {ip_nx:,}")
            piccolo = sporco.sample(n=350, random_state=1)
            mp = M.metriche_inconsistenza(piccolo, fds)
            controlla(f"{config} {livello}%: IM e IP uguali alla forza bruta (350 righe)",
                      (mp['IM'], mp['IP']) == forza_bruta(piccolo, fds))

    print("\n2. Validita' della copertura IH")
    for config in C.CONFIGURAZIONI:
        fds = C.fds_di(config)
        sporco, _ = M.sporca_training(campione, C.colonne_da_sporcare(config), 40, seed=7)
        cop = M.copertura_ih(sporco, fds)
        m = M.metriche_inconsistenza(sporco, fds)
        residue = M.conta_violazioni(sporco.loc[~cop], fds)
        controlla(f"{config}: togliendo la copertura restano 0 violazioni", residue == 0,
                  f"IH={int(cop.sum()):,}, IH_min={m['IH_min']:,}, violazioni residue={residue}")

    print("\n3. Garanzia training sporco / test pulito (dataset bilanciato, fold 0)")
    df, _ = M.carica_dataset_pulito()
    impronta = M.impronta_righe(df)
    train, test = M.dividi_fold(df, 0)
    controlla("training e test disgiunti e complementari",
              len(set(train.index) & set(test.index)) == 0 and len(train) + len(test) == len(df))
    for config in C.CONFIGURAZIONI:
        colonne = C.colonne_da_sporcare(config)
        for livello in C.LIVELLI_RUMORE:
            copia_prima = train.copy()
            sporco, posizioni = M.sporca_training(train, colonne, livello, C.seed_rumore(0, livello))
            attese = (len(train) * livello) // 100
            modificate = M.righe_modificate(sporco, impronta)
            ok = modificate == attese and train.equals(copia_prima)
            if livello > 0:
                diversi = all((sporco[c].to_numpy()[posizioni] != train[c].to_numpy()[posizioni]).all()
                              for c in colonne)
                intatte = np.setdiff1d(np.arange(len(train)), posizioni)
                fuori = sporco.iloc[intatte].equals(train.iloc[intatte])
                ok = ok and diversi and fuori
            if livello in (0, 40) or not ok:
                controlla(f"{config} {livello:>2}%: righe modificate = {attese:,}, input intatto, "
                          f"celle sporcate sempre diverse", ok, f"misurate {modificate:,}")
            elif not ok:
                controlla(f"{config} {livello}%", False)
    controlla("test identico al dataset pulito dopo tutte le sporcature",
              M.righe_modificate(test, impronta) == 0 and M.conta_violazioni(test, C.fds_di('C4')) == 0)

    print("\n4. Configurazioni annidate (stesse righe, stessi valori)")
    for livello in [5, 40]:
        seed = C.seed_rumore(0, livello)
        precedente = None
        for config in C.CONFIGURAZIONI:
            sporco, pos = M.sporca_training(train, C.colonne_da_sporcare(config), livello, seed)
            if precedente is not None:
                cfg_prec, sp_prec, pos_prec = precedente
                comuni = C.colonne_da_sporcare(cfg_prec)
                controlla(f"{livello}%: {config} coincide con {cfg_prec} sulle {len(comuni)} colonne comuni",
                          np.array_equal(pos, pos_prec) and sporco[comuni].equals(sp_prec[comuni]))
            precedente = (config, sporco, pos)

    print(f"\nEsito: {sum(ESITI)}/{len(ESITI)} controlli superati.")
    sys.exit(0 if all(ESITI) else 1)


if __name__ == "__main__":
    main()
