"""
Test diagnostico minimo: DistanceGroup viene davvero corrotta?
Non esegue l'esperimento completo, solo l'iniezione di rumore su un singolo
caso, per isolare il problema dal checkpoint e da tutto il resto.

Uso: metti questo file nella stessa cartella di progettoTesi_v2.py e
blocco1_esperimento.py, poi lancialo con: python test_distancegroup.py
"""
from progettoTesi_v2 import inject_multiple_fd_noise
from blocco1_esperimento import carica_campione, FDS_LIST, REDUNDANT_COLS_MAP

print("Carico il campione pulito...")
df = carica_campione()

print(f"REDUNDANT_COLS_MAP contiene 'DistanceGroup'? "
      f"{'DistanceGroup' in REDUNDANT_COLS_MAP[(('Origin', 'Dest'), 'Distance')]}")

print("\nInietto rumore al 40% (stesso seed=100 usato nella replica 0 dell'esperimento vero)...")
df_noisy = inject_multiple_fd_noise(
    df, FDS_LIST, noise_level=0.40,
    corrupt_lhs=True, redundant_cols_map=REDUNDANT_COLS_MAP, seed=100,
)

n_diverse = (df_noisy['DistanceGroup'] != df['DistanceGroup']).sum()
attese = int(len(df) * 0.40)

print(f"\nRighe con DistanceGroup diversa dall'originale: {n_diverse}")
print(f"Righe attese (40% di {len(df)}):                  {attese}")

if n_diverse == 0:
    print("\n*** PROBLEMA REALE: DistanceGroup non viene mai toccata. ***")
    print("Non e' un problema di checkpoint: la funzione di iniezione non la corrompe.")
    print("Va controllato inject_multiple_fd_noise o REDUNDANT_COLS_MAP con piu' attenzione.")
elif abs(n_diverse - attese) <= attese * 0.02:
    print("\nOK: la corruzione funziona come previsto a livello di iniezione.")
    print("Se blocco1_risultati_raw.csv continua a dare numeri identici a prima,")
    print("il problema e' quasi certamente il checkpoint: cancella il CSV vecchio")
    print("PRIMA di rilanciare blocco1_esperimento.py.")
else:
    print("\nNumero anomalo, ne' zero ne' quello atteso: da indagare a parte.")
