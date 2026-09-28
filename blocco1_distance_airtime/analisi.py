"""
Analisi statistica del sottoprogetto, identica a blocco1_analisi.py del
progetto principale: ne riusa le funzioni, cambiando solo i file.

0. Controllo training sporcato / test pulito.
1. Aggregato per Modello x Rumore.
2. Scomposizione del danno (apprendimento contro input corrotto).
3. Correlazioni di Pearson e Spearman fra IM/IP/IH e F1.
4. Significativita' del degrado (t-test a un campione contro il baseline).
"""
import pandas as pd

import config as C
from blocco1_analisi import (controlla_training_e_test, build_aggregato, build_scomposizione,
                             build_correlazioni, build_test_degrado)

if __name__ == "__main__":
    df = pd.read_csv(C.percorso('risultati_raw.csv'))
    controlla_training_e_test(df)

    agg = build_aggregato(df)
    agg.to_csv(C.percorso('aggregato.csv'), index=False)
    print("1. Aggregato salvato in 'aggregato.csv'.")
    print(df.pivot_table(index='Rumore_%', columns='Modello', values='F1_Score_test_pulito',
                         aggfunc='mean').round(4).to_string())

    scomp = build_scomposizione(df)
    scomp.to_csv(C.percorso('scomposizione.csv'), index=False)
    print("\n2. Scomposizione del danno (media dei 4 modelli), salvata in 'scomposizione.csv':")
    print(scomp.to_string(index=False))

    corr = build_correlazioni(df)
    corr.to_csv(C.percorso('correlazioni.csv'), index=False)
    print("\n3. Correlazioni salvate in 'correlazioni.csv' (test pulito):")
    print(corr[corr['Valutazione'] == 'test_pulito'].to_string(index=False))

    test = build_test_degrado(df)
    test.to_csv(C.percorso('test_degrado.csv'), index=False)
    print("\n4. Significativita' del degrado (t-test a un campione), salvata in 'test_degrado.csv':")
    print(test.to_string(index=False))
    print(f"   Livelli con calo significativo: {int(test['Significativo'].sum())}/{len(test)}")
