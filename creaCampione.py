import pandas as pd

N_RIGHE = 30000

file_originale = 'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_1.csv'
file_output = f'flight_sample_{N_RIGHE}.csv'

print(f"Sto caricando il dataset originale: {file_originale} ...")
print("Potrebbe volerci qualche secondo...")

try:
    df_completo = pd.read_csv(file_originale, low_memory=False)
    print(f"Caricamento completato! Il dataset contiene {df_completo.shape[0]} righe e {df_completo.shape[1]} colonne.")

    df_campione = df_completo.sample(n=N_RIGHE, random_state=42)
    print(f"Campione estratto: {df_campione.shape[0]} righe.")

    df_campione.to_csv(file_output, index=False)
    print(f"Fatto! Campione salvato con successo come: {file_output}")

except FileNotFoundError:
    print(f"ERRORE: Non riesco a trovare il file '{file_originale}'.")
    print("Assicurati di aver scritto il nome corretto e che il file si trovi nella stessa cartella di questo script Python.")
except Exception as e:
    print(f"Si è verificato un errore durante la lettura: {e}")
