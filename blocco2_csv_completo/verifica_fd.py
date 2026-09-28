"""
Certificazione delle dipendenze funzionali (roadmap par. 3.3).

Prima di sporcare qualunque cosa, verifica che ogni FD usata — principale o
che giustifica una colonna ridondante — abbia 0 violazioni sui dati puliti,
sia sul CSV completo sia sul dataset bilanciato effettivamente usato. Cosi'
tutta l'inconsistenza misurata dopo e' quella iniettata, non preesistente.

Output: blocco2_verifica_fd.csv. Esce con errore se trova una violazione.
"""
import sys
import pandas as pd

import config as C
import motore as M


def main():
    righe = []
    fd_principali = [(nome, lhs, rhs, 'principale') for nome, (lhs, rhs) in C.FD.items()]
    fd_ridondanti = [(f'R{i}', lhs, rhs, 'ridondante') for i, (lhs, rhs) in enumerate(C.FD_RIDONDANTI, 1)]
    tutte = fd_principali + fd_ridondanti

    colonne = sorted({c for _, lhs, rhs, _ in tutte for c in list(lhs) + [rhs]})
    completo = pd.read_csv(C.CSV_ORIGINALE, usecols=colonne, low_memory=False)
    bilanciato, info = M.carica_dataset_pulito()
    if info:
        print(f"Dataset bilanciato creato: {info}")
    feature = set(bilanciato.columns) - set(C.COLONNE_NON_FEATURE)

    for nome, lhs, rhs, tipo in tutte:
        riga = {'FD': nome, 'Tipo': tipo, 'Dipendenza': f"{'+'.join(lhs)} -> {rhs}",
                'Significato': C.SIGNIFICATO_FD.get(nome, 'colonna ridondante del concetto aeroporto')}
        for etichetta, df in [('CSV_completo', completo), ('Dataset_bilanciato', bilanciato)]:
            riga[f'Righe_{etichetta}'] = len(df)
            riga[f'Gruppi_{etichetta}'] = int(df.groupby(list(lhs)).ngroups)
            riga[f'Violazioni_{etichetta}'] = M.conta_violazioni(df, [(lhs, rhs)])
            riga[f'NaN_{etichetta}'] = int(df[list(lhs) + [rhs]].isna().sum().sum())
        riga['Colonne_al_modello'] = all(c in feature for c in list(lhs) + [rhs])
        righe.append(riga)

    tabella = pd.DataFrame(righe)
    tabella.to_csv(C.VERIFICA_FD_CSV, index=False)
    print(tabella[['FD', 'Tipo', 'Dipendenza', 'Gruppi_CSV_completo', 'Violazioni_CSV_completo',
                   'Violazioni_Dataset_bilanciato', 'NaN_CSV_completo', 'Colonne_al_modello']]
          .to_string(index=False))
    print(f"\nSalvato: {C.VERIFICA_FD_CSV}")

    problemi = tabella[(tabella['Violazioni_CSV_completo'] > 0) | (tabella['Violazioni_Dataset_bilanciato'] > 0)
                       | (tabella['NaN_CSV_completo'] > 0) | (~tabella['Colonne_al_modello'])]
    if len(problemi):
        print("\nVERIFICA FALLITA:")
        print(problemi[['FD', 'Dipendenza']].to_string(index=False))
        sys.exit(1)
    print("\nVERIFICA SUPERATA: 0 violazioni, nessun valore mancante, tutte le colonne arrivano al modello.")


if __name__ == "__main__":
    main()
