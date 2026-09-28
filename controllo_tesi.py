"""
Confronta i numeri citati nella tesi con quelli prodotti dagli esperimenti.

Estrae il testo dal PDF della tesi (senza dipendenze esterne: decomprime i
flussi e raccoglie le stringhe), individua i valori nelle forme usate nel testo
(0,9233 / 8,4 punti / 31,2) e segnala quelli che non corrispondono a nessun
valore presente nei CSV delle analisi, con il contesto in cui compaiono.

Non sostituisce la rilettura: serve a restringere il campo ai punti da
controllare a mano.

Uso:  python controllo_tesi.py "percorso/tesi.pdf" [--tolleranza 0.0002]
"""
import argparse
import glob
import re
import zlib

import numpy as np
import pandas as pd

# Numeri che compaiono in tesi ma non vengono dagli esperimenti: parametri del
# disegno, dimensioni, anni, soglie statistiche.
ATTESI_DI_CONTESTO = {0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0,
                      0.001, 0.01, 30000.0, 8090.0, 6472.0, 6470.0, 2025.0, 2026.0}


def testo_dal_pdf(percorso):
    dati = open(percorso, 'rb').read()
    pezzi = []
    for m in re.finditer(rb'stream\r?\n(.*?)endstream', dati, re.S):
        try:
            flusso = zlib.decompress(m.group(1))
        except Exception:
            continue
        if b'TJ' not in flusso and b'Tj' not in flusso:
            continue
        stringhe = [t.group(1) for t in re.finditer(rb'\(([^)]*)\)', flusso)]
        if stringhe:
            pezzi.append(b' '.join(stringhe).decode('latin-1'))
    testo = '\n'.join(pezzi)
    # pdflatex spezza le parole con spazi: si normalizzano gli spazi multipli.
    return re.sub(r'[ \t]+', ' ', testo)


def valori_attesi():
    """Tutti i numeri prodotti dalle analisi, con la loro provenienza.

    Include anche i valori derivati riportati in numeri_tesi.md (cali per
    modello, rapporti, medie), che nei CSV non compaiono come tali."""
    attesi = {}
    try:
        derivati = open('numeri_tesi.md', encoding='utf-8').read()
        for m in re.finditer(r'(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])', derivati):
            v = float(m.group(1))
            for valore in {round(v, 4), round(v * 100, 4), round(v, 1), round(v, 2)}:
                attesi.setdefault(valore, set()).add('numeri_tesi.md')
    except FileNotFoundError:
        pass
    for percorso in sorted(glob.glob('blocco*.csv') + glob.glob('rilevanza_fd.csv')
                           + glob.glob('verifica_ih.csv') + glob.glob('tabella_riassuntiva_*.csv')):
        try:
            df = pd.read_csv(percorso)
        except Exception:
            continue
        for colonna in df.columns:
            serie = pd.to_numeric(df[colonna], errors='coerce').dropna()
            for v in serie:
                for valore in {round(float(v), 4), round(float(v) * 100, 4), round(float(v), 1),
                               round(float(v), 2)}:
                    attesi.setdefault(valore, set()).add(f"{percorso}:{colonna}")
    return attesi


def numeri_nel_testo(testo):
    """Numeri con la virgola decimale, come sono scritti in tesi, con contesto."""
    trovati = []
    for m in re.finditer(r'(?<![\d,])(\d{1,3}(?:\.\d{3})*|\d+)\s?,\s?(\d{1,4})(?![\d,])', testo):
        intero = m.group(1).replace('.', '')
        try:
            valore = float(f"{intero}.{m.group(2)}")
        except ValueError:
            continue
        inizio, fine = max(0, m.start() - 90), min(len(testo), m.end() + 90)
        trovati.append((valore, re.sub(r'\s+', ' ', testo[inizio:fine]).strip()))
    return trovati


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf')
    parser.add_argument('--tolleranza', type=float, default=0.0002)
    parser.add_argument('--max-segnalazioni', type=int, default=80)
    args = parser.parse_args()

    testo = testo_dal_pdf(args.pdf)
    print(f"Testo estratto dal PDF: {len(testo):,} caratteri")
    attesi = valori_attesi()
    chiavi = np.array(sorted(attesi))
    print(f"Valori prodotti dalle analisi: {len(chiavi):,}")

    trovati = numeri_nel_testo(testo)
    print(f"Numeri decimali citati nella tesi: {len(trovati)}\n")

    sospetti = []
    for valore, contesto in trovati:
        if valore in ATTESI_DI_CONTESTO:
            continue
        vicino = chiavi[np.abs(chiavi - valore) <= max(args.tolleranza, abs(valore) * 1e-4)]
        if len(vicino) == 0:
            sospetti.append((valore, contesto))

    print(f"Numeri che NON corrispondono a nessun valore delle analisi: {len(sospetti)}")
    print("(vanno controllati a mano: possono essere numeri di un'altra fonte, "
          "arrotondamenti o valori da aggiornare)\n")
    for valore, contesto in sospetti[:args.max_segnalazioni]:
        print(f"  {valore:>12,.4f}  ...{contesto}...")
    if len(sospetti) > args.max_segnalazioni:
        print(f"  ... e altri {len(sospetti) - args.max_segnalazioni}")


if __name__ == "__main__":
    main()
