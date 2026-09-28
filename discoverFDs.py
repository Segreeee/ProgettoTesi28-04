import pandas as pd
from itertools import combinations
import time

def robust_fd_discovery(df, max_lhs_size=2):
    """
    Esegue una ricerca ottimizzata di Dipendenze Funzionali (FD).
    max_lhs_size: numero massimo di colonne a sinistra della regola (LHS).
    """
    print(f"Inizio analisi su {df.shape[0]} righe e {df.shape[1]} colonne...")
    start_time = time.time()

    nunique = df.nunique()
    cols_to_keep = nunique[(nunique > 1) & (nunique < len(df))].index.tolist()
    df_opt = df[cols_to_keep]
    
    print(f"Colonne ridotte da {df.shape[1]} a {len(cols_to_keep)} (scartate costanti e ID univoci).")
    columns = df_opt.columns.tolist()
    
    valid_fds = []

    for size in range(1, max_lhs_size + 1):
        print(f"\n--- Cerco FD con {size} attributo/i a sinistra (LHS) ---")
        combinazioni_lhs = list(combinations(columns, size))
        tot_comb = len(combinazioni_lhs)
        
        for i, lhs in enumerate(combinazioni_lhs):
            if (i + 1) % 500 == 0 or i == 0:
                print(f"Progresso: {i + 1}/{tot_comb} combinazioni analizzate...")
                
            lhs = list(lhs)
            remaining_cols = [c for c in columns if c not in lhs]
            
            grouped = df_opt.groupby(lhs)
            
            for rhs in remaining_cols:
                if grouped[rhs].nunique().max() == 1:
                    valid_fds.append((lhs, rhs))

    print(f"\nRicerca completata in {time.time() - start_time:.2f} secondi!")
    print(f"Trovate {len(valid_fds)} Dipendenze Funzionali valide nel campione.")
    
    risultati = pd.DataFrame([
        {
            "LHS (Determinante)": " + ".join(fd[0]),
            "RHS (Dipendente)": fd[1],
            "Tipo": f"{len(fd[0])} -> 1"
        } for fd in valid_fds
    ])
    
    return risultati

if __name__ == "__main__":
    file_name = 'flight_sample_1000.csv' 
    
    try:
        df_sample = pd.read_csv(file_name)
        
        fds_trovate = robust_fd_discovery(df_sample, max_lhs_size=2)
        
        output_file = 'lista_FD_trovate.csv'
        fds_trovate.to_csv(output_file, index=False)
        print(f"\nRisultati salvati con successo in: {output_file}")
        
        print("\nAnteprima delle prime 20 FD trovate:")
        print(fds_trovate.head(20).to_string())
        
    except FileNotFoundError:
        print(f"ERRORE: Non trovo il file {file_name}. Assicurati di essere nella cartella giusta.")