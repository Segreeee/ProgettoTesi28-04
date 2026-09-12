import pandas as pd

# Numero di righe del campione. 30.000 e' il massimo per cui il grafo dei
# conflitti di networkx resta in memoria nel caso peggiore dell'esperimento
# (10 FD corrotte al 20% di rumore: ~3,3 milioni di archi, ~0,5 GB) anche con
# piu' processi in parallelo. Il numero di archi cresce circa con il quadrato
# delle righe: a 150.000 righe servirebbero ~88 milioni di archi (~13 GB).
N_RIGHE = 30000

file_originale = 'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_1.csv'
file_output = f'flight_sample_{N_RIGHE}.csv'

print(f"Sto caricando il dataset originale: {file_originale} ...")
print("Potrebbe volerci qualche secondo...")

try:
    df_completo = pd.read_csv(file_originale, low_memory=False)
    print(f"Caricamento completato! Il dataset contiene {df_completo.shape[0]} righe e {df_completo.shape[1]} colonne.")

    # Estrai N_RIGHE righe a caso (random_state=42 garantisce che estrarrai sempre le stesse se lo rilanci)
    df_campione = df_completo.sample(n=N_RIGHE, random_state=42)
    print(f"Campione estratto: {df_campione.shape[0]} righe.")

    # Salva il campione in un nuovo file CSV leggero
    df_campione.to_csv(file_output, index=False)
    print(f"Fatto! Campione salvato con successo come: {file_output}")

except FileNotFoundError:
    print(f"ERRORE: Non riesco a trovare il file '{file_originale}'.")
    print("Assicurati di aver scritto il nome corretto e che il file si trovi nella stessa cartella di questo script Python.")
except Exception as e:
    print(f"Si è verificato un errore durante la lettura: {e}")
