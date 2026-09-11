"""
Motore del Blocco 2: preparazione del dataset, sporcatura del training,
misure di inconsistenza senza grafo, addestramento e valutazione.

Scritto da zero per questa cartella: non importa nulla dal progetto principale.
"""
import os
import time
import zlib
import itertools

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config as C


# =================================================================
# 1. PREPARAZIONE DEL DATASET PULITO (una volta sola)
# =================================================================

def _colonne_da_caricare():
    """Colonne del CSV che superano la blacklist, piu' la fonte del target."""
    intestazione = pd.read_csv(C.CSV_ORIGINALE, nrows=0).columns
    escluse = set(C.BLACKLIST)
    tenute = []
    for c in intestazione:
        if c == C.TARGET_SORGENTE:
            tenute.append(c)
        elif c in escluse:
            continue
        elif 'ID' in c and c not in C.ID_AMMESSI:
            continue
        else:
            tenute.append(c)
    return tenute


def prepara_dataset_pulito():
    """
    Costruisce il dataset pulito e bilanciato su cui gira tutto l'esperimento:
      1. carica le colonne che superano la blacklist e crea il target;
      2. estrae il 45% delle righe, stratificato per classe di durata;
      3. bilancia le classi UNA VOLTA SOLA (roadmap par. 4.2);
      4. elimina le colonne interamente vuote (l'imputer le scarterebbe comunque);
      5. assegna i fold stratificati, identici per ogni livello e configurazione.
    Salva il risultato su file: tutti i processi e tutte le ripartenze usano
    esattamente le stesse righe e gli stessi fold.
    """
    df = pd.read_csv(C.CSV_ORIGINALE, usecols=_colonne_da_caricare(), low_memory=False)
    righe_csv = len(df)
    df[C.COL_RIGA_ORIGINALE] = np.arange(len(df))   # posizione nel CSV, per tracciabilita'

    fascia = pd.cut(df[C.TARGET_SORGENTE], bins=C.FASCE_TARGET, labels=C.ETICHETTE_TARGET)
    df = df.loc[fascia.notna()].copy()
    df[C.TARGET_COL] = fascia[fascia.notna()].astype(str)
    df = df.drop(columns=[C.TARGET_SORGENTE])

    campione = df.groupby(C.TARGET_COL).sample(frac=C.QUOTA_CAMPIONE,
                                               random_state=C.SEED_CAMPIONE)

    gruppi = [g for _, g in campione.groupby(C.TARGET_COL)]
    n_min = min(len(g) for g in gruppi)
    bilanciato = (pd.concat([g.sample(n=n_min, random_state=C.SEED_BILANCIAMENTO) for g in gruppi])
                  .sample(frac=1, random_state=C.SEED_BILANCIAMENTO)
                  .reset_index(drop=True))

    vuote = [c for c in bilanciato.columns if bilanciato[c].isna().all()]
    bilanciato = bilanciato.drop(columns=vuote)

    bilanciato[C.COL_FOLD] = -1
    skf = StratifiedKFold(n_splits=C.K_FOLD, shuffle=True, random_state=C.SEED_FOLD)
    for k, (_, idx_test) in enumerate(skf.split(bilanciato, bilanciato[C.TARGET_COL])):
        bilanciato.loc[idx_test, C.COL_FOLD] = k

    bilanciato.to_csv(C.DATASET_PULITO, index=False, compression='gzip')
    return {
        'righe_csv': righe_csv,
        'righe_campione': len(campione),
        'righe_per_classe': n_min,
        'righe_bilanciate': len(bilanciato),
        'colonne_vuote_rimosse': vuote,
    }


def carica_dataset_pulito():
    """
    Restituisce (dataset, info). Crea il file la prima volta; poi lo rilegge
    SEMPRE dal disco, anche subito dopo averlo creato, cosi' che i tipi delle
    colonne siano identici in ogni esecuzione e in ogni processo.
    """
    info = None
    if not os.path.exists(C.DATASET_PULITO):
        info = prepara_dataset_pulito()
    df = pd.read_csv(C.DATASET_PULITO, low_memory=False)
    return df, info


def dividi_fold(df, fold):
    """Training = tutti gli altri fold, test = il fold indicato. Nessuna copia:
    il test resta una vista del dataset pulito e non viene mai sporcato."""
    test = df.loc[df[C.COL_FOLD] == fold]
    train = df.loc[df[C.COL_FOLD] != fold]
    return train, test


# =================================================================
# 2. SPORCATURA DEL TRAINING
# =================================================================

def _seed_colonna(seed, colonna):
    """Seed stabile per (seed di riga, nome colonna): lo stesso in ogni configurazione."""
    return zlib.crc32(f'{seed}|{colonna}'.encode('utf-8')) & 0xFFFFFFFF


def sporca_training(train, colonne, livello, seed):
    """
    Restituisce (training sporcato, posizioni delle righe sporcate).

    Riceve SOLO il training e non lo modifica: lavora su una copia.
    - Sceglie una volta sola (livello% delle righe, arrotondato per difetto)
      le righe da sporcare; la scelta dipende solo dal seed, quindi e' la
      stessa per tutte le configurazioni a parita' di fold e livello.
    - Per ogni colonna sostituisce il valore con uno DIVERSO, estratto in modo
      uniforme fra i valori che la colonna assume nel training. Ogni colonna
      usa un generatore dedicato derivato da (seed, colonna): i valori sporchi
      di una colonna sono identici in tutte le configurazioni, per cui la
      configurazione C(k+1) coincide con C(k) piu' le sole colonne aggiunte.
    - Ogni colonna e' sporcata indipendentemente dalle altre: la riga diventa
      internamente incoerente (viola le FD), come nella corruzione del Blocco 1.
    A livello 0 restituisce una copia identica.
    """
    sporco = train.copy()
    if livello == 0 or not colonne:
        return sporco, np.array([], dtype=np.int64)

    n = len(train)
    k = (n * livello) // 100
    rng = np.random.RandomState(seed)
    posizioni = np.sort(rng.choice(n, size=k, replace=False))

    for col in colonne:
        rng_col = np.random.RandomState(_seed_colonna(seed, col))
        originali = train[col].to_numpy()
        dominio = pd.unique(originali)
        if len(dominio) < 2:
            raise ValueError(f"La colonna {col} ha un solo valore: impossibile sporcarla.")
        attuali = originali[posizioni]
        pos_attuali = pd.Index(dominio).get_indexer(attuali)
        estratti = rng_col.randint(0, len(dominio) - 1, size=k)
        # salta la posizione del valore attuale: il nuovo valore e' sempre diverso
        nuovi = dominio[np.where(estratti >= pos_attuali, estratti + 1, estratti)]
        valori = sporco[col].to_numpy(copy=True)
        valori[posizioni] = nuovi
        sporco[col] = valori

    return sporco, posizioni


# =================================================================
# 3. MISURE DI INCONSISTENZA SENZA GRAFO
# =================================================================
# Grafo dei conflitti: nodo = riga, arco = coppia di righe con stesso LHS e
# RHS diverso per almeno una FD. Su scala ampia il grafo arriva a milioni di
# archi e non e' costruibile: le tre misure si ottengono da conteggi per gruppo.

def _verifica_senza_nan(df, fds):
    for lhs, rhs in fds:
        for c in list(lhs) + [rhs]:
            if df[c].isna().any():
                raise ValueError(
                    f"La colonna {c} contiene valori mancanti: le misure per gruppo "
                    "non riprodurrebbero il confronto coppia per coppia sui NaN.")


def _coppie_uguali(df, colonne, cache):
    """Numero di coppie non ordinate di righe uguali su TUTTE le colonne indicate."""
    chiave = frozenset(colonne)
    if chiave not in cache:
        dim = df.groupby(sorted(chiave), sort=False).size().to_numpy(dtype=np.int64)
        cache[chiave] = int((dim * (dim - 1) // 2).sum())
    return cache[chiave]


def calcola_im(df, fds, cache=None):
    """
    IM esatto: numero di archi dell'unione dei grafi di conflitto delle FD.

    E_k = coppie con LHS_k uguale e RHS_k diverso. Per l'intersezione di un
    insieme T di FD si usa l'inclusione-esclusione sulle condizioni "RHS
    diverso": |∩_T E_k| = Σ_{U⊆T} (-1)^|U| · #coppie uguali su (LHS_T ∪ RHS_U).
    Poi |∪ E_k| = Σ_{T≠∅} (-1)^(|T|+1) · |∩_T E_k|.
    """
    cache = {} if cache is None else cache
    k = len(fds)
    im = 0
    for r in range(1, k + 1):
        for T in itertools.combinations(range(k), r):
            lhs_T = set(c for t in T for c in fds[t][0])
            intersezione = 0
            for u in range(len(T) + 1):
                for U in itertools.combinations(T, u):
                    colonne = lhs_T | {fds[t][1] for t in U}
                    intersezione += (-1) ** u * _coppie_uguali(df, colonne, cache)
            im += (-1) ** (r + 1) * intersezione
    return int(im)


def _righe_in_conflitto(df, lhs, rhs):
    """Una riga ha almeno un conflitto per la FD se e solo se il suo gruppo LHS
    contiene piu' di un valore del RHS."""
    return (df.groupby(lhs, sort=False)[rhs].transform('nunique') > 1).to_numpy()


def _copertura_fd(df, lhs, rhs):
    """
    Copertura minima del grafo di una singola FD. Dentro ogni gruppo LHS il grafo
    e' multipartito completo fra i valori del RHS: la copertura minima tiene il
    valore piu' frequente e marca tutte le altre righe (a parita', il primo
    incontrato). Restituisce la maschera delle righe da correggere.
    """
    per_valore = df.groupby(lhs + [rhs], sort=False)
    id_valore = per_valore.ngroup()
    n_valore = per_valore[rhs].transform('size')
    id_gruppo = df.groupby(lhs, sort=False).ngroup()
    tabella = pd.DataFrame({'gruppo': id_gruppo, 'valore': id_valore, 'n': n_valore})
    tenuto = (tabella.sort_values(['gruppo', 'n', 'valore'], ascending=[True, False, True])
                     .drop_duplicates('gruppo')
                     .set_index('gruppo')['valore'])
    return (id_valore != id_gruppo.map(tenuto)).to_numpy()


def copertura_ih(df, fds):
    """Unione delle coperture minime delle singole FD: e' una copertura valida del
    grafo complessivo (ogni arco appartiene a qualche FD, e li' e' coperto)."""
    maschera = np.zeros(len(df), dtype=bool)
    for lhs, rhs in fds:
        maschera |= _copertura_fd(df, list(lhs), rhs)
    return maschera


def metriche_inconsistenza(df, fds):
    """
    IM  : numero di coppie di righe in conflitto (esatto).
    IP  : numero di righe coinvolte in almeno un conflitto (esatto).
    IH  : numero di righe da correggere per sanare tutte le FD — unione delle
          coperture minime per FD; esatto con una sola FD, altrimenti limite
          superiore del minimo.
    IH_min : limite inferiore del minimo (la copertura piu' grande fra le FD).
    IM_per_FD : archi di ciascuna FD presa da sola.
    """
    _verifica_senza_nan(df, fds)
    colonne = sorted({c for lhs, rhs in fds for c in list(lhs) + [rhs]})
    sub = df[colonne].reset_index(drop=True)
    cache = {}

    im = calcola_im(sub, fds, cache)
    ip = np.zeros(len(sub), dtype=bool)
    ih = np.zeros(len(sub), dtype=bool)
    ih_min = 0
    im_per_fd = []
    for lhs, rhs in fds:
        lhs = list(lhs)
        ip |= _righe_in_conflitto(sub, lhs, rhs)
        cop = _copertura_fd(sub, lhs, rhs)
        ih |= cop
        ih_min = max(ih_min, int(cop.sum()))
        im_per_fd.append(_coppie_uguali(sub, set(lhs), cache)
                         - _coppie_uguali(sub, set(lhs) | {rhs}, cache))
    return {'IM': im, 'IP': int(ip.sum()), 'IH': int(ih.sum()),
            'IH_min': ih_min, 'IM_per_FD': im_per_fd}


def conta_violazioni(df, fds):
    """Numero di gruppi LHS che violano almeno una FD (0 = dati coerenti)."""
    return int(sum((df.groupby(list(lhs), sort=False)[rhs].nunique() > 1).sum()
                   for lhs, rhs in fds))


# =================================================================
# 4. CONTROLLO DELL'INTEGRITA' (garanzia training sporco / test pulito)
# =================================================================

def impronta_righe(df):
    """Impronta (hash) di ogni riga, indicizzata come il dataframe."""
    return pd.Series(pd.util.hash_pandas_object(df, index=True).to_numpy(), index=df.index)


def righe_modificate(df, impronta_pulita):
    """Numero di righe di df che differiscono dalla versione pulita originale,
    misurato confrontando le impronte (non dichiarato da chi ha sporcato)."""
    attuale = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return int((attuale != impronta_pulita.loc[df.index].to_numpy()).sum())


# =================================================================
# 5. ADDESTRAMENTO E VALUTAZIONE
# =================================================================

def _modelli(n_jobs_rf):
    """I 4 modelli RAW del Blocco 1: iperparametri di default, random_state=42,
    nessun class_weight, nessun tuning. n_jobs cambia solo la velocita' del
    Random Forest, non le sue predizioni (verificato)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=n_jobs_rf),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Neural Network": MLPClassifier(max_iter=1000, random_state=42),
    }


def allena_e_valuta(train_df, test_df, n_jobs_rf=1):
    """
    Riceve training e test GIA' separati (roadmap par. 5.2). Il preprocessore e'
    costruito qui e addestrato solo sul training. One-hot in formato sparso:
    predizioni identiche al formato denso, ma senza matrici da ~1 GB.
    """
    X_tr = train_df.drop(columns=C.COLONNE_NON_FEATURE)
    y_tr = train_df[C.TARGET_COL].to_numpy()
    X_te = test_df.drop(columns=C.COLONNE_NON_FEATURE)
    y_te = test_df[C.TARGET_COL].to_numpy()

    numeriche = X_tr.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoriche = X_tr.select_dtypes(include=['object', 'bool']).columns.tolist()
    preprocessore = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')),
                          ('scaler', StandardScaler())]), numeriche),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore',
                                                   sparse_output=True))]), categoriche),
    ])

    risultati = []
    for nome, modello in _modelli(n_jobs_rf).items():
        clf = Pipeline([('preprocessor', clone(preprocessore)), ('classifier', modello)])
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        durata = time.time() - t0
        iterazioni = getattr(clf.named_steps['classifier'], 'n_iter_', None)
        risultati.append({
            'Modello': nome,
            'Accuracy': accuracy_score(y_te, pred),
            'Precision': precision_score(y_te, pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_te, pred, average='weighted', zero_division=0),
            'F1_weighted': f1_score(y_te, pred, average='weighted', zero_division=0),
            'F1_macro': f1_score(y_te, pred, average='macro', zero_division=0),
            'Tempo_s': round(durata, 1),
            'Iterazioni': int(np.max(iterazioni)) if iterazioni is not None else np.nan,
        })
    return risultati
