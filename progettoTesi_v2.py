import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations, product
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

N_SPLITS = 5


def get_global_inconsistency_metrics(df, fds_list):
    """
    Costruisce un Grafo dei Conflitti Globale basato su una LISTA di Dipendenze Funzionali.
    fds_list: lista di tuple, es. [(['LHS1'], 'RHS1'), (['LHS2'], 'RHS2')]
    """
    G = nx.Graph()
    G.add_nodes_from(df.index)

    for lhs, rhs in fds_list:
        if df[rhs].isna().any():
            raise ValueError(
                f"La colonna RHS '{rhs}' contiene valori mancanti: il raggruppamento "
                "per valore non riprodurrebbe il confronto coppia-a-coppia sui NaN."
            )
        for _, group in df.groupby(lhs):
            vals = group[rhs]
            if vals.nunique() <= 1:
                continue
            buckets = [list(idx) for idx in vals.groupby(vals).groups.values()]
            for b1, b2 in combinations(buckets, 2):
                G.add_edges_from(product(b1, b2))

    IM = G.number_of_edges()
    IP = len([n for n in G.nodes() if G.degree(n) > 0])

    # min_weighted_vertex_cover e' 2-approssimato: IH_approx e' un LIMITE
    # SUPERIORE del minimo, che nel caso peggiore ne vale il doppio.
    vc_set = nx.algorithms.approximation.min_weighted_vertex_cover(G)
    IH_approx = len(vc_set)

    return IM, IP, IH_approx, G


def ih_esatto_una_fd(df, lhs, rhs):
    """
    Vertex cover minimo ESATTO del grafo dei conflitti di UNA sola FD.

    Con una sola FD il grafo e' l'unione disgiunta, sui gruppi del lato
    sinistro, di grafi multipartiti completi: le parti sono i valori distinti
    del lato destro dentro il gruppo. Il massimo insieme indipendente di un
    multipartito completo e' la parte piu' numerosa, quindi il vertex cover
    minimo di ogni gruppo vale (righe del gruppo - righe della classe piu'
    numerosa) e il minimo globale e' la loro somma.
    """
    totale = 0
    for _, gruppo in df.groupby(list(lhs)):
        conteggi = gruppo[rhs].value_counts()
        if len(conteggi) > 1:
            totale += int(len(gruppo) - conteggi.iloc[0])
    return totale


def indici_inconsistenza(df, fds_list):
    """
    IM, IP e IH sulle righe di `df`. IH e' riportato sia nella versione
    2-approssimata sia, quando le FD sono una sola, nel valore esatto.
    """
    IM, IP, IH_approx, _ = get_global_inconsistency_metrics(df, fds_list)
    IH_esatto = ih_esatto_una_fd(df, fds_list[0][0], fds_list[0][1]) if len(fds_list) == 1 else np.nan
    return {'IM': IM, 'IP': IP, 'IH_approx': IH_approx, 'IH_esatto': IH_esatto}


def indici_per_fold(df_sporco, fold_train_index, fds_list):
    """
    Indici di inconsistenza calcolati sulle righe di training di ciascun fold,
    cioe' esattamente le righe che i modelli vedono. Restituisce media e
    deviazione standard sui fold, piu' il numero di righe su cui sono calcolati.
    """
    per_fold = [indici_inconsistenza(df_sporco.loc[idx], fds_list) for idx in fold_train_index]
    tabella = pd.DataFrame(per_fold)
    risultato = {'Righe_indici': int(np.mean([len(idx) for idx in fold_train_index]))}
    for colonna in ['IM', 'IP', 'IH_approx', 'IH_esatto']:
        risultato[f'{colonna}_mean'] = float(tabella[colonna].mean())
        risultato[f'{colonna}_sd'] = float(tabella[colonna].std())
    return risultato

def inject_multiple_fd_noise(df, fds_list, noise_level, corrupt_lhs=True, redundant_cols_map=None, seed=42):
    """
    Applica il Data Poisoning alle colonne di una o più FD contemporaneamente.

    corrupt_lhs=True sporca, per le righe selezionate, TUTTE le colonne
    coinvolte nella FD (sia il LHS/determinante che il RHS/dipendente), non
    solo il RHS. Questo evita che il modello possa "aggirare" il danno
    ricostruendo l'informazione persa dal LHS rimasto pulito (il LHS di una
    FD è per definizione almeno tanto informativo quanto il RHS).

    redundant_cols_map: dict opzionale {(tuple(lhs), rhs): [colonne extra]}.
    Colonne ridondanti che codificano la stessa informazione della FD ma non
    ne fanno formalmente parte (es. OriginAirportID, che è in corrispondenza
    biunivoca con Origin) e che quindi rappresentano un'altra "via di fuga"
    per il modello se lasciate pulite. Vengono sporcate sulle stesse righe.
    """
    df_noisy = df.copy()
    if noise_level == 0.0:
        return df_noisy

    n_to_corrupt = int(len(df) * noise_level)
    redundant_cols_map = redundant_cols_map or {}
    rng = np.random.RandomState(seed)

    for lhs, rhs in fds_list:
        indices_to_corrupt = rng.choice(df.index, n_to_corrupt, replace=False)
        extra_cols = redundant_cols_map.get((tuple(lhs), rhs), [])
        columns_to_corrupt = (list(lhs) if corrupt_lhs else []) + [rhs] + list(extra_cols)

        for col in columns_to_corrupt:
            unique_values = df[col].unique()
            for idx in indices_to_corrupt:
                current_val = df_noisy.loc[idx, col]
                possible_vals = [v for v in unique_values if v != current_val]
                if possible_vals:
                    df_noisy.loc[idx, col] = rng.choice(possible_vals)

    return df_noisy


def _prepare_xy(df, target_col, extra_blacklist=None, seed_valutazione=42):
    """
    Applica blacklist anti-leakage, encoding del target e bilanciamento delle
    classi, restituendo (X, y) con l'INDICE ORIGINALE del dataframe preservato.

    Preservare l'indice è ciò che permette di valutare un modello addestrato
    sui dati sporchi usando le STESSE righe di test prese dal dataset pulito:
    poiché il target non viene mai corrotto, due dataframe che differiscono
    solo per le feature producono esattamente le stesse righe e lo stesso
    ordine (proprietà verificata e protetta dagli assert in ml_preparation).
    """
    df_ml = df.dropna(subset=[target_col]).copy()

    cols_to_drop = [
        'ArrDelay', 'ArrDelayMinutes', 'ArrDel15', 'ArrTime', 'ActualElapsedTime',
        'AirTime', 'TaxiIn', 'TaxiOut', 'WheelsOff', 'WheelsOn',
        'DepTime', 'DepDel15', 'ArrivalDelayGroups', 'DepartureDelayGroups',
        'FlightDate', 'Tail_Number', 'Flight_Number_Reporting_Airline',
        'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
        'Cancelled', 'CancellationCode', 'Diverted', 'FirstDepTime', 'TotalAddGTime', 'LongestAddGTime',
        'DepDelayMinutes', 'DelayGroups'
    ]
    if extra_blacklist:
        cols_to_drop.extend(extra_blacklist)

    extra_drops = [c for c in df_ml.columns if 'ID' in c and c not in ['OriginAirportID', 'DestAirportID']]
    cols_to_drop.extend(extra_drops)

    cols_to_drop = [c for c in cols_to_drop if c in df_ml.columns]
    if target_col in cols_to_drop:
        cols_to_drop.remove(target_col)

    df_ml = df_ml.drop(columns=cols_to_drop, errors='ignore')

    X = df_ml.drop(columns=[target_col])
    y = df_ml[target_col]

    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        y = (y > 15).astype(int)

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=y.index, name='TARGET')

    df_temp = X.copy()
    df_temp['TARGET_TEMP'] = y

    class_groups = [group for _, group in df_temp.groupby('TARGET_TEMP')]
    n_minimo = min(len(group) for group in class_groups)

    balanced_samples = [group.sample(n=n_minimo, random_state=seed_valutazione) for group in class_groups]
    df_balanced = pd.concat(balanced_samples).sample(frac=1, random_state=seed_valutazione)

    X_bal = df_balanced.drop(columns=['TARGET_TEMP'])
    y_bal = df_balanced['TARGET_TEMP']
    return X_bal, y_bal


def _metrics(y_true, preds):
    """Le 4 metriche standard, weighted (i dati sono bilanciati a monte)."""
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, preds, average='weighted', zero_division=0),
        "F1_Score": f1_score(y_true, preds, average='weighted', zero_division=0),
    }


def _righe_diverse(a, b):
    """Maschera delle righe di `a` che differiscono da `b` in almeno una colonna
    (stesso indice e stesse colonne; due NaN contano come uguali)."""
    diverse = np.zeros(len(a), dtype=bool)
    for c in a.columns:
        uguali = (a[c] == b[c]) | (a[c].isna() & b[c].isna())
        diverse |= ~uguali.to_numpy()
    return diverse


def fold_di_valutazione(df, target_col, extra_blacklist=None, seed_valutazione=42):
    """
    Righe di training di ciascun fold, con l'indice originale del dataframe.

    E' la stessa partizione usata da ml_preparation con lo stesso
    seed_valutazione: serve a calcolare gli indici di inconsistenza esattamente
    sulle righe che i modelli vedono in addestramento.
    """
    X, y = _prepare_xy(df, target_col, extra_blacklist, seed_valutazione)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed_valutazione)
    return [X.index[train_idx] for train_idx, _ in skf.split(X, y)]


def ml_preparation(df, target_col, extra_blacklist=None, df_eval=None, n_jobs_rf=None,
                   seed_valutazione=42) -> dict:
    """
    Addestra 4 modelli RAW in cross-validation stratificata a N_SPLITS fold e
    ne restituisce le metriche medie (e la deviazione standard sui fold).

    df       : dataset di TRAINING, eventualmente sporcato.
    seed_valutazione: governa il sottocampionamento di bilanciamento e la
               partizione in fold. Passando il seed della replica, ogni replica
               lavora su righe e fold propri: anche il baseline a rumore 0%
               acquista variabilita', condizione necessaria per confrontarlo
               con un test a due campioni.
    df_eval  : dataset opzionale da cui prendere il test set — tipicamente il
               dataset PULITO. Se fornito, ogni fold viene valutato DUE volte:
                 - 'test_sporco': righe di test prese da `df` (confronto);
                 - 'test_pulito': STESSE righe di test prese da `df_eval`.
               Il modello è sempre e solo addestrato sulle righe sporcate di
               `df`: cambia unicamente il dataset su cui si misura.
    n_jobs_rf: thread del Random Forest. Cambia solo la velocita': le
               predizioni sono identiche per qualunque valore (verificato).

    Nomi delle metriche restituite:
      Accuracy_test_sporco, Precision_test_sporco, Recall_test_sporco,
      F1_Score_test_sporco (+ *_std per Accuracy e F1) e, se df_eval è
      fornito, gli omologhi con suffisso _test_pulito, piu'
      Quota_train_sporca: quota media di righe di training che differiscono
      dal dataset pulito (0 a rumore 0%, cresce con il rumore).
    """
    X, y = _prepare_xy(df, target_col, extra_blacklist, seed_valutazione)

    X_eval = None
    righe_sporche = None
    if df_eval is not None:
        X_eval, y_eval = _prepare_xy(df_eval, target_col, extra_blacklist, seed_valutazione)
        if not X.index.equals(X_eval.index):
            raise ValueError(
                "Indici disallineati tra dataset di training e dataset di valutazione: "
                "il confronto test sporco / test pulito non sarebbe valido. "
                "Causa probabile: il target e' stato corrotto dall'iniezione di rumore."
            )
        if not np.array_equal(y.values, y_eval.values):
            raise ValueError(
                "Le etichette del dataset di training e di quello di valutazione differiscono: "
                "il target non deve mai essere corrotto."
            )
        X_eval = X_eval[X.columns]
        righe_sporche = _righe_diverse(X, X_eval)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=n_jobs_rf),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Neural Network": MLPClassifier(max_iter=1000, random_state=42)
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed_valutazione)
    fold_scores = {name: {'test_sporco': [], 'test_pulito': []} for name in models}
    quote_train_sporche = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        y_te = y.iloc[test_idx]
        if righe_sporche is not None:
            quote_train_sporche.append(float(righe_sporche[train_idx].mean()))

        for name, model in models.items():
            clf = Pipeline(steps=[('preprocessor', clone(preprocessor)),
                                  ('classifier', clone(model))])
            try:
                clf.fit(X_tr, y_tr)

                fold_scores[name]['test_sporco'].append(
                    _metrics(y_te, clf.predict(X.iloc[test_idx]))
                )
                if X_eval is not None:
                    fold_scores[name]['test_pulito'].append(
                        _metrics(y_te, clf.predict(X_eval.iloc[test_idx]))
                    )
            except Exception as e:
                print(f"Errore nel modello {name}: {e}")
                nan_metrics = {m: np.nan for m in ["Accuracy", "Precision", "Recall", "F1_Score"]}
                fold_scores[name]['test_sporco'].append(nan_metrics)
                if X_eval is not None:
                    fold_scores[name]['test_pulito'].append(nan_metrics)

    results = {}
    for name, per_test in fold_scores.items():
        row = {}
        for suffix, folds in per_test.items():
            if not folds:
                continue
            for m in ["Accuracy", "Precision", "Recall", "F1_Score"]:
                row[f"{m}_{suffix}"] = float(np.mean([f[m] for f in folds]))
            row[f"Accuracy_{suffix}_std"] = float(np.std([f["Accuracy"] for f in folds], ddof=1))
            row[f"F1_Score_{suffix}_std"] = float(np.std([f["F1_Score"] for f in folds], ddof=1))
        if quote_train_sporche:
            row["Quota_train_sporca"] = float(np.mean(quote_train_sporche))
        results[name] = row

    return results
