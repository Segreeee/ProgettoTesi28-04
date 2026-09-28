"""
Esecuzione parallela con checkpoint, usata da blocco1_esperimento.py e da
blocco2_scaling_fd.py.

- Il numero di processi e' scelto in base alla RAM libera all'avvio.
- Ogni lavoro completato e' aggiunto subito al CSV dei risultati: se
  l'esecuzione si interrompe, alla ripartenza i lavori gia' fatti vengono
  saltati.
- Il parallelismo non cambia i risultati: ogni lavoro usa i propri seed, e
  nessun modello dipende dall'ordine di esecuzione.
"""
import os
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

RISERVA_RAM_GB = 1.0
MAX_PROCESSI = 4


def ram_disponibile_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        pass
    if os.name == 'nt':
        import ctypes

        class _Memoria(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        stato = _Memoria()
        stato.dwLength = ctypes.sizeof(_Memoria)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stato))
        return stato.ullAvailPhys / 1e9
    return 4.0


def numero_processi(ram_per_processo_gb, forzato=None):
    """Restituisce (processi, RAM libera in GB)."""
    ram = ram_disponibile_gb()
    if forzato:
        return forzato, ram
    return max(1, min(MAX_PROCESSI, int((ram - RISERVA_RAM_GB) / ram_per_processo_gb))), ram


class Registro:
    """Scrive ogni messaggio a schermo e, con data e ora, nel file di log."""

    def __init__(self, percorso):
        self.f = open(percorso, 'a', encoding='utf-8')

    def __call__(self, msg):
        riga = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(riga, flush=True)
        self.f.write(riga + '\n')
        self.f.flush()


def lavori_completati(file_risultati, colonne_chiave, righe_per_lavoro=4):
    """Chiavi dei lavori gia' presenti e completi (una riga per modello) nel CSV."""
    if not os.path.exists(file_risultati):
        return set()
    conteggi = pd.read_csv(file_risultati).groupby(colonne_chiave).size()
    return {(k if isinstance(k, tuple) else (k,)) for k, n in conteggi.items() if n == righe_per_lavoro}


def esegui_lavori(funzione, lavori, chiave_di, inizializzatore, file_risultati, colonne_chiave,
                  n_processi, log, etichetta='LAVORO'):
    """
    Esegue funzione(*argomenti) per ogni tupla di argomenti in `lavori`, in
    parallelo, saltando quelli gia' nel CSV. Ogni risultato (lista di righe)
    viene aggiunto al CSV appena disponibile. Al primo errore si ferma tutto.
    """
    fatti = lavori_completati(file_risultati, colonne_chiave)
    da_fare = [a for a in lavori if chiave_di(a) not in fatti]
    log(f"Lavori totali {len(lavori)}, gia' completati {len(lavori) - len(da_fare)}, da eseguire {len(da_fare)}")
    if not da_fare:
        return

    t0, completati = time.time(), 0
    with ProcessPoolExecutor(max_workers=n_processi, initializer=inizializzatore) as esecutore:
        futuri = {esecutore.submit(funzione, *argomenti): argomenti for argomenti in da_fare}
        for fut in as_completed(futuri):
            argomenti = futuri[fut]
            try:
                righe = fut.result()
            except Exception as e:
                log(f"ERRORE nel lavoro {chiave_di(argomenti)}: {e!r}")
                for altro in futuri:
                    altro.cancel()
                raise
            nuovo = not os.path.exists(file_risultati)
            pd.DataFrame(righe).to_csv(file_risultati, mode='a', header=nuovo, index=False)
            completati += 1
            eta = (time.time() - t0) / completati * (len(da_fare) - completati)
            im = f"IM={righe[0]['IM']:,}  " if 'IM' in righe[0] else ""
            log(f"{etichetta} {completati}/{len(da_fare)}  {chiave_di(argomenti)}  "
                f"{im}fine stimata tra {eta / 60:.0f} min")
