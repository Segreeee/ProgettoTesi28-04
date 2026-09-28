# Numeri verificati per la tesi

Generato da `numeri_tesi.py` a partire dai CSV delle analisi. Ogni numero citato nella tesi deve corrispondere a uno di questi valori.

Convenzioni: il rumore e' l'unica variabile controllata; IM, IP e IH sono osservati sulle righe di training di ciascun fold; IH e' riportato come 2-approssimato (limite superiore) e, con una sola FD, anche esatto.

## Blocco 1 — una FD (rotta -> distanza) con le colonne ridondanti
- Righe su cui sono calcolati gli indici (training di ogni fold): [np.int64(6472)]
- Baseline F1 sul test pulito: Decision Tree 0.9212, Logistic Regression 0.9231, Neural Network 0.9107, Random Forest 0.9306; media 0.9214

### F1 sul test pulito per modello e livello

| Rumore_% | Decision Tree | Logistic Regression | Neural Network | Random Forest | Media |
|---|---|---|---|---|---|
| 0 | 0.9212 | 0.9231 | 0.9107 | 0.9306 | 0.9214 |
| 5 | 0.9176 | 0.8685 | 0.9049 | 0.9287 | 0.9049 |
| 10 | 0.9133 | 0.8301 | 0.8978 | 0.9271 | 0.8921 |
| 20 | 0.9095 | 0.7763 | 0.887 | 0.9229 | 0.8739 |
| 30 | 0.9009 | 0.7295 | 0.8779 | 0.9157 | 0.856 |
| 40 | 0.894 | 0.6896 | 0.8644 | 0.9046 | 0.8382 |

### Calo di F1 rispetto al baseline (punti/100)

| Rumore_% | Decision Tree | Logistic Regression | Neural Network | Random Forest | Media |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0.0036 | 0.0546 | 0.0058 | 0.0019 | 0.0165 |
| 10 | 0.0079 | 0.093 | 0.0129 | 0.0035 | 0.0293 |
| 20 | 0.0117 | 0.1468 | 0.0237 | 0.0077 | 0.0475 |
| 30 | 0.0203 | 0.1936 | 0.0328 | 0.0149 | 0.0654 |
| 40 | 0.0272 | 0.2335 | 0.0463 | 0.026 | 0.0832 |

### Indici osservati (media sulle repliche)

| Rumore_% | IM_mean | IP_mean | IH_approx_mean | IH_esatto_mean |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 5 | 16.1 | 23.6 | 9.9 | 7.5 |
| 10 | 37.5 | 55 | 22.2 | 17.6 |
| 20 | 74.8 | 114.2 | 49.9 | 40.4 |
| 30 | 107.4 | 165.2 | 74 | 59.8 |
| 40 | 134.6 | 218.2 | 102.2 | 85.7 |

- IH approssimato / IH esatto negli esperimenti: da 1.19 a 1.32

### Scomposizione del danno (media dei 4 modelli)

| Rumore_% | F1_test_pulito | F1_test_sporco | Perdita_apprendimento | Perdita_input_corrotto | Perdita_totale_test_sporco | Quota_apprendimento_% |
|---|---|---|---|---|---|---|
| 0 | 0.9214 | 0.9214 | 0 | 0 | 0 |  |
| 5 | 0.9049 | 0.871 | 0.0165 | 0.0339 | 0.0504 | 32.7 |
| 10 | 0.8921 | 0.8272 | 0.0293 | 0.0649 | 0.0942 | 31.2 |
| 20 | 0.8739 | 0.7505 | 0.0475 | 0.1234 | 0.1709 | 27.8 |
| 30 | 0.856 | 0.6783 | 0.0654 | 0.1777 | 0.2431 | 26.9 |
| 40 | 0.8381 | 0.6115 | 0.0833 | 0.2266 | 0.3099 | 26.9 |

- Confronti con calo significativo (Welch, p < 0,05): 18 su 20

### Test di Welch per modello e livello

| Modello | Rumore_% | F1_baseline | F1_medio | Calo | p_value | Significativo |
|---|---|---|---|---|---|---|
| Decision Tree | 5 | 0.9212 | 0.9176 | 0.0036 | 0.2597 | 0 |
| Decision Tree | 10 | 0.9212 | 0.9133 | 0.0079 | 0.0157 | 1 |
| Decision Tree | 20 | 0.9212 | 0.9095 | 0.0118 | 0.0022 | 1 |
| Decision Tree | 30 | 0.9212 | 0.9009 | 0.0204 | 0 | 1 |
| Decision Tree | 40 | 0.9212 | 0.894 | 0.0272 | 0 | 1 |
| Logistic Regression | 5 | 0.9231 | 0.8685 | 0.0546 | 0 | 1 |
| Logistic Regression | 10 | 0.9231 | 0.8301 | 0.093 | 0 | 1 |
| Logistic Regression | 20 | 0.9231 | 0.7763 | 0.1467 | 0 | 1 |
| Logistic Regression | 30 | 0.9231 | 0.7295 | 0.1936 | 0 | 1 |
| Logistic Regression | 40 | 0.9231 | 0.6896 | 0.2335 | 0 | 1 |
| Neural Network | 5 | 0.9107 | 0.9049 | 0.0058 | 0.0004 | 1 |
| Neural Network | 10 | 0.9107 | 0.8978 | 0.0129 | 0 | 1 |
| Neural Network | 20 | 0.9107 | 0.887 | 0.0237 | 0 | 1 |
| Neural Network | 30 | 0.9107 | 0.8779 | 0.0328 | 0 | 1 |
| Neural Network | 40 | 0.9107 | 0.8644 | 0.0463 | 0 | 1 |
| Random Forest | 5 | 0.9306 | 0.9287 | 0.0019 | 0.2643 | 0 |
| Random Forest | 10 | 0.9306 | 0.9271 | 0.0035 | 0.0457 | 1 |
| Random Forest | 20 | 0.9306 | 0.9229 | 0.0077 | 0.0002 | 1 |
| Random Forest | 30 | 0.9306 | 0.9157 | 0.0149 | 0 | 1 |
| Random Forest | 40 | 0.9306 | 0.9046 | 0.026 | 0 | 1 |

## Blocco 2 — 1, 2 e 4 FD rilevanti

### F1 sul test pulito (media dei 4 modelli) per numero di FD

| Rumore_% | 1 | 2 | 4 |
|---|---|---|---|
| 0 | 0.9214 | 0.9214 | 0.9214 |
| 5 | 0.9015 | 0.8887 | 0.8852 |
| 10 | 0.8861 | 0.8697 | 0.8618 |
| 20 | 0.8684 | 0.8472 | 0.8272 |
| 30 | 0.8554 | 0.8299 | 0.7926 |
| 40 | 0.843 | 0.8099 | 0.7507 |

### Quota di righe di training effettivamente sporcate

| Rumore_% | 1 | 2 | 4 |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 5 | 0.0485 | 0.097 | 0.1852 |
| 10 | 0.0981 | 0.1886 | 0.342 |
| 20 | 0.196 | 0.3573 | 0.5892 |
| 30 | 0.2969 | 0.5076 | 0.7592 |
| 40 | 0.3971 | 0.641 | 0.8721 |

### IM osservato per numero di FD

| Rumore_% | 1 | 2 | 4 |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 5 | 1306 | 3603.8 | 92275.5 |
| 10 | 2596.2 | 6579.4 | 161630 |
| 20 | 4812.8 | 10760.7 | 251133 |
| 30 | 6957.2 | 13492 | 290361 |
| 40 | 8665.2 | 14741.6 | 289586 |

### IP osservato per numero di FD

| Rumore_% | 1 | 2 | 4 |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 5 | 1389.8 | 3093.1 | 6466.8 |
| 10 | 2537.8 | 4592.2 | 6471.8 |
| 20 | 4026.2 | 5779.9 | 6472 |
| 30 | 4995.1 | 6171.6 | 6472 |
| 40 | 5552.7 | 6311.6 | 6472 |

### IH_approx osservato per numero di FD

| Rumore_% | 1 | 2 | 4 |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 5 | 402.9 | 916.1 | 1868.4 |
| 10 | 802.9 | 1728.7 | 3244.1 |
| 20 | 1549.3 | 2996.6 | 4959.6 |
| 30 | 2291.7 | 3917.1 | 5750.4 |
| 40 | 2929.6 | 4486.9 | 6122.2 |

### F1 sul test pulito per configurazione e modello

| N_FD | Modello | 0 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|
| 1 | Decision Tree | 0.9212 | 0.9174 | 0.9119 | 0.9052 | 0.896 | 0.8824 |
| 1 | Logistic Regression | 0.9231 | 0.8587 | 0.809 | 0.7507 | 0.7123 | 0.6835 |
| 1 | Neural Network | 0.9107 | 0.9004 | 0.8966 | 0.8945 | 0.8925 | 0.8908 |
| 1 | Random Forest | 0.9306 | 0.9293 | 0.9268 | 0.9232 | 0.9207 | 0.9152 |
| 2 | Decision Tree | 0.9212 | 0.9134 | 0.9068 | 0.8907 | 0.8728 | 0.8417 |
| 2 | Logistic Regression | 0.9231 | 0.8177 | 0.7596 | 0.7006 | 0.6647 | 0.6353 |
| 2 | Neural Network | 0.9107 | 0.8974 | 0.8905 | 0.8831 | 0.8769 | 0.8695 |
| 2 | Random Forest | 0.9306 | 0.9264 | 0.9219 | 0.9143 | 0.905 | 0.8932 |
| 4 | Decision Tree | 0.9212 | 0.9119 | 0.9013 | 0.877 | 0.8521 | 0.8057 |
| 4 | Logistic Regression | 0.9231 | 0.8133 | 0.7493 | 0.685 | 0.6422 | 0.6035 |
| 4 | Neural Network | 0.9107 | 0.8915 | 0.8794 | 0.8495 | 0.8062 | 0.7654 |
| 4 | Random Forest | 0.9306 | 0.9243 | 0.9172 | 0.8972 | 0.8698 | 0.8281 |

### Scomposizione del danno per numero di FD

| N_FD | Rumore_% | Quota_train_sporca | F1_test_pulito | F1_test_sporco | Perdita_apprendimento | Perdita_input_corrotto | Perdita_totale_test_sporco | Quota_apprendimento_% |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.9214 | 0.9214 | 0 | 0 | 0 |  |
| 1 | 5 | 0.0485 | 0.9015 | 0.882 | 0.02 | 0.0195 | 0.0394 | 50.6 |
| 1 | 10 | 0.0981 | 0.8861 | 0.8536 | 0.0354 | 0.0324 | 0.0678 | 52.2 |
| 1 | 20 | 0.196 | 0.8684 | 0.8193 | 0.053 | 0.0491 | 0.1021 | 51.9 |
| 1 | 30 | 0.2969 | 0.8554 | 0.7935 | 0.066 | 0.0619 | 0.1279 | 51.6 |
| 1 | 40 | 0.3971 | 0.843 | 0.7733 | 0.0785 | 0.0696 | 0.1481 | 53 |
| 2 | 0 | 0 | 0.9214 | 0.9214 | 0 | 0 | 0 |  |
| 2 | 5 | 0.097 | 0.8887 | 0.8522 | 0.0327 | 0.0365 | 0.0692 | 47.2 |
| 2 | 10 | 0.1886 | 0.8697 | 0.8141 | 0.0517 | 0.0556 | 0.1073 | 48.2 |
| 2 | 20 | 0.3573 | 0.8472 | 0.7678 | 0.0742 | 0.0793 | 0.1536 | 48.3 |
| 2 | 30 | 0.5076 | 0.8299 | 0.741 | 0.0915 | 0.0889 | 0.1804 | 50.7 |
| 2 | 40 | 0.641 | 0.8099 | 0.7229 | 0.1115 | 0.087 | 0.1985 | 56.2 |
| 4 | 0 | 0 | 0.9214 | 0.9214 | 0 | 0 | 0 |  |
| 4 | 5 | 0.1852 | 0.8852 | 0.832 | 0.0362 | 0.0533 | 0.0895 | 40.4 |
| 4 | 10 | 0.342 | 0.8618 | 0.7681 | 0.0596 | 0.0937 | 0.1533 | 38.9 |
| 4 | 20 | 0.5892 | 0.8272 | 0.6606 | 0.0942 | 0.1666 | 0.2608 | 36.1 |
| 4 | 30 | 0.7592 | 0.7926 | 0.5676 | 0.1288 | 0.2249 | 0.3538 | 36.4 |
| 4 | 40 | 0.8721 | 0.7507 | 0.4899 | 0.1707 | 0.2608 | 0.4315 | 39.6 |

- Confronti con calo significativo (Welch): 58 su 60; per configurazione: 1 FD 18/20, 2 FD 20/20, 4 FD 20/20

### Confronto fra configurazioni a parita' di livello

| Confronto | significativi | confronti | delta_medio |
|---|---|---|---|
| 1 -> 2 FD | 19 | 20 | -0.0218 |
| 2 -> 4 FD | 18 | 20 | -0.0256 |

### Regressione F1 ~ quota + quota^2 + numero di FD

| Modello | Intervallo | N_punti | R2 | Coef_N_FD | t_N_FD | p_N_FD | Significativo |
|---|---|---|---|---|---|---|---|
| Decision Tree | tutti i punti | 80 | 0.9519 | 0.00457 | 5.812 | 0 | 1 |
| Decision Tree | quota <= 0.41 (comune alle configurazioni) | 55 | 0.9134 | 0.00289 | 6.101 | 0 | 1 |
| Logistic Regression | tutti i punti | 80 | 0.9544 | 0.01454 | 6.854 | 0 | 1 |
| Logistic Regression | quota <= 0.41 (comune alle configurazioni) | 55 | 0.9762 | 0.01877 | 12.703 | 0 | 1 |
| Neural Network | tutti i punti | 80 | 0.9453 | -0.0057 | -5.698 | 0 | 1 |
| Neural Network | quota <= 0.41 (comune alle configurazioni) | 55 | 0.823 | -0.00226 | -4.696 | 2e-05 | 1 |
| Random Forest | tutti i punti | 80 | 0.9547 | -0.00072 | -1.125 | 0.26426 | 0 |
| Random Forest | quota <= 0.41 (comune alle configurazioni) | 55 | 0.8554 | -7e-05 | -0.24 | 0.81091 | 0 |

### Punti a quota di righe sporche confrontabile

| Meno_FD | Rumore_meno_FD_% | Quota_meno_FD | F1_meno_FD | Piu_FD | Rumore_piu_FD_% | Quota_piu_FD | F1_piu_FD | Delta_quota | Delta_F1_piu_meno | Modelli_con_piu_FD_migliore |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 0.0981 | 0.8861 | 2 | 5 | 0.097 | 0.8887 | -0.0011 | 0.0027 | 3/4 |
| 1 | 20 | 0.196 | 0.8684 | 2 | 10 | 0.1886 | 0.8697 | -0.0074 | 0.0013 | 2/4 |
| 1 | 20 | 0.196 | 0.8684 | 4 | 5 | 0.1852 | 0.8852 | -0.0107 | 0.0168 | 3/4 |
| 2 | 10 | 0.1886 | 0.8697 | 4 | 5 | 0.1852 | 0.8852 | -0.0033 | 0.0156 | 4/4 |
| 2 | 20 | 0.3573 | 0.8472 | 4 | 10 | 0.342 | 0.8618 | -0.0153 | 0.0147 | 3/4 |

## Comportamento delle misure con 4 FD (replica 0, primo fold)

### Gruppi LHS

| FD_n | FD | 0 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|
| 1 | Distance -> DistanceGroup | 1206 | 1270 | 1303 | 1345 | 1371 | 1381 |
| 2 | Origin+Dest -> Distance | 2699 | 3364 | 3881 | 4732 | 5402 | 5797 |
| 3 | OriginAirportID -> Origin | 251 | 300 | 313 | 321 | 322 | 322 |
| 4 | DestAirportID -> Dest | 240 | 286 | 306 | 313 | 315 | 315 |

### Dimensione massima

| FD_n | FD | 0 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|
| 1 | Distance -> DistanceGroup | 70 | 64 | 62 | 55 | 35 | 33 |
| 2 | Origin+Dest -> Distance | 38 | 31 | 29 | 20 | 12 | 8 |
| 3 | OriginAirportID -> Origin | 276 | 265 | 259 | 229 | 206 | 177 |
| 4 | DestAirportID -> Dest | 330 | 313 | 304 | 261 | 227 | 215 |

### Coppie stesso LHS

| FD_n | FD | 0 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|
| 1 | Distance -> DistanceGroup | 32636 | 29234 | 26918 | 22709 | 18808 | 17013 |
| 2 | Origin+Dest -> Distance | 12040 | 9092 | 6959 | 3803 | 1767 | 924 |
| 3 | OriginAirportID -> Origin | 425490 | 391401 | 360208 | 294966 | 239680 | 196075 |
| 4 | DestAirportID -> Dest | 438759 | 402799 | 364306 | 305037 | 244263 | 201976 |

### Conflitti FD

| FD_n | FD | 0 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|
| 1 | Distance -> DistanceGroup | 0 | 2650 | 4978 | 8100 | 10106 | 11709 |
| 2 | Origin+Dest -> Distance | 0 | 1162 | 1674 | 1683 | 1180 | 745 |
| 3 | OriginAirportID -> Origin | 0 | 46334 | 78701 | 121262 | 136769 | 141922 |
| 4 | DestAirportID -> Dest | 0 | 45652 | 77807 | 117741 | 143526 | 146652 |

### Quota coppie in conflitto

| FD_n | FD | 0 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|
| 1 | Distance -> DistanceGroup | 0 | 0.0906 | 0.1849 | 0.3567 | 0.5373 | 0.6882 |
| 2 | Origin+Dest -> Distance | 0 | 0.1278 | 0.2406 | 0.4425 | 0.6678 | 0.8063 |
| 3 | OriginAirportID -> Origin | 0 | 0.1184 | 0.2185 | 0.4111 | 0.5706 | 0.7238 |
| 4 | DestAirportID -> Dest | 0 | 0.1133 | 0.2136 | 0.386 | 0.5876 | 0.7261 |

### Tuple coinvolte FD

| FD_n | FD | 0 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|
| 1 | Distance -> DistanceGroup | 0 | 2530 | 4180 | 5504 | 5983 | 6154 |
| 2 | Origin+Dest -> Distance | 0 | 1252 | 1727 | 1693 | 1419 | 1057 |
| 3 | OriginAirportID -> Origin | 0 | 6223 | 6385 | 6457 | 6469 | 6472 |
| 4 | DestAirportID -> Dest | 0 | 6213 | 6429 | 6465 | 6471 | 6471 |

### Misure complessive sulle stesse righe

| Rumore_% | IM_totale | IP_totale | IH_approx_totale |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 5 | 94760 | 6472 | 1921 |
| 10 | 161648 | 6472 | 3299 |
| 20 | 246745 | 6472 | 4932 |
| 30 | 289588 | 6472 | 5785 |
| 40 | 299391 | 6472 | 6119 |

## IH: approssimato contro esatto (verifica_ih.py)

- Rapporto approssimato / esatto: da 1.000 a 1.473 (garanzia teorica: al piu' 2)
- Scarto assoluto: da 0 a 443 tuple
- Con 1 FD la formula chiusa coincide con l'ottimo dell'ILP: True

### Confronto su istanze ridotte

| N_FD | Rumore_% | Righe | IM | IP | IH_approx | IH_esatto_ilp | Scarto_assoluto | Rapporto_approx_su_esatto |
|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 250 | 26 | 48 | 24 | 24 | 0 | 1 |
| 1 | 5 | 500 | 80 | 144 | 74 | 72 | 2 | 1.028 |
| 1 | 5 | 1000 | 301 | 480 | 257 | 254 | 3 | 1.012 |
| 1 | 5 | 2000 | 1193 | 1383 | 789 | 729 | 60 | 1.082 |
| 1 | 10 | 250 | 28 | 50 | 26 | 26 | 0 | 1 |
| 1 | 10 | 500 | 74 | 126 | 66 | 66 | 0 | 1 |
| 1 | 10 | 1000 | 288 | 462 | 249 | 241 | 8 | 1.033 |
| 1 | 10 | 2000 | 1250 | 1431 | 861 | 799 | 62 | 1.078 |
| 1 | 20 | 250 | 26 | 46 | 24 | 24 | 0 | 1 |
| 1 | 20 | 500 | 79 | 136 | 72 | 71 | 1 | 1.014 |
| 1 | 20 | 1000 | 302 | 484 | 261 | 255 | 6 | 1.024 |
| 1 | 20 | 2000 | 1260 | 1432 | 855 | 804 | 51 | 1.063 |
| 1 | 30 | 250 | 25 | 46 | 24 | 23 | 1 | 1.043 |
| 1 | 30 | 500 | 76 | 136 | 72 | 68 | 4 | 1.059 |
| 1 | 30 | 1000 | 287 | 467 | 251 | 243 | 8 | 1.033 |
| 1 | 30 | 2000 | 1254 | 1424 | 855 | 803 | 52 | 1.065 |
| 1 | 40 | 250 | 27 | 48 | 25 | 25 | 0 | 1 |
| 1 | 40 | 500 | 74 | 128 | 67 | 66 | 1 | 1.015 |
| 1 | 40 | 1000 | 286 | 461 | 251 | 240 | 11 | 1.046 |
| 1 | 40 | 2000 | 1252 | 1437 | 865 | 799 | 66 | 1.083 |
| 2 | 5 | 250 | 38 | 62 | 32 | 29 | 3 | 1.103 |
| 2 | 5 | 500 | 142 | 218 | 117 | 102 | 15 | 1.147 |
| 2 | 5 | 1000 | 517 | 650 | 374 | 324 | 50 | 1.154 |
| 2 | 5 | 2000 | 1997 | 1703 | 1049 | 879 | 170 | 1.193 |
| 2 | 10 | 250 | 36 | 60 | 34 | 29 | 5 | 1.172 |
| 2 | 10 | 500 | 135 | 201 | 113 | 98 | 15 | 1.153 |
| 2 | 10 | 1000 | 509 | 645 | 369 | 315 | 54 | 1.171 |
| 2 | 10 | 2000 | 1989 | 1700 | 1100 | 904 | 196 | 1.217 |
| 2 | 20 | 250 | 38 | 66 | 36 | 33 | 3 | 1.091 |
| 2 | 20 | 500 | 123 | 192 | 106 | 96 | 10 | 1.104 |
| 2 | 20 | 1000 | 470 | 603 | 344 | 301 | 43 | 1.143 |
| 2 | 20 | 2000 | 1877 | 1639 | 1056 | 879 | 177 | 1.201 |
| 2 | 30 | 250 | 31 | 54 | 29 | 25 | 4 | 1.16 |
| 2 | 30 | 500 | 111 | 169 | 93 | 82 | 11 | 1.134 |
| 2 | 30 | 1000 | 432 | 581 | 331 | 294 | 37 | 1.126 |
| 2 | 30 | 2000 | 1704 | 1592 | 1002 | 846 | 156 | 1.184 |
| 2 | 40 | 250 | 32 | 50 | 27 | 26 | 1 | 1.038 |
| 2 | 40 | 500 | 116 | 175 | 97 | 88 | 9 | 1.102 |
| 2 | 40 | 1000 | 412 | 555 | 318 | 280 | 38 | 1.136 |
| 2 | 40 | 2000 | 1624 | 1567 | 984 | 852 | 132 | 1.155 |
| 4 | 5 | 250 | 184 | 155 | 64 | 46 | 18 | 1.391 |
| 4 | 5 | 500 | 653 | 416 | 176 | 129 | 47 | 1.364 |
| 4 | 5 | 1000 | 2646 | 940 | 494 | 367 | 127 | 1.346 |
| 4 | 5 | 2000 | 10280 | 1977 | 1235 | 946 | 289 | 1.305 |
| 4 | 10 | 250 | 199 | 164 | 85 | 58 | 27 | 1.466 |
| 4 | 10 | 500 | 816 | 415 | 202 | 140 | 62 | 1.443 |
| 4 | 10 | 1000 | 3161 | 978 | 551 | 389 | 162 | 1.416 |
| 4 | 10 | 2000 | 15186 | 1996 | 1415 | 1021 | 394 | 1.386 |
| 4 | 20 | 250 | 314 | 213 | 111 | 79 | 32 | 1.405 |
| 4 | 20 | 500 | 1359 | 477 | 284 | 203 | 81 | 1.399 |
| 4 | 20 | 1000 | 5248 | 991 | 693 | 484 | 209 | 1.432 |
| 4 | 20 | 2000 | 21146 | 1997 | 1588 | 1149 | 439 | 1.382 |
| 4 | 30 | 250 | 368 | 216 | 136 | 93 | 43 | 1.462 |
| 4 | 30 | 500 | 1652 | 481 | 333 | 226 | 107 | 1.473 |
| 4 | 30 | 1000 | 6490 | 997 | 800 | 560 | 240 | 1.429 |
| 4 | 30 | 2000 | 26681 | 2000 | 1716 | 1273 | 443 | 1.348 |
| 4 | 40 | 250 | 376 | 211 | 144 | 105 | 39 | 1.371 |
| 4 | 40 | 500 | 1592 | 476 | 362 | 264 | 98 | 1.371 |
| 4 | 40 | 1000 | 6566 | 994 | 830 | 627 | 203 | 1.324 |

## IH esatto sulle istanze del Blocco 2 (fold di training, 6.472 righe)

Con 1 FD formula chiusa; con 2 e 4 FD programmazione lineare intera (`ih_esatto_blocco2.py`, limite di 150 s per istanza). Non risolti: 2 FD al 30% e 4 FD dal 20% (prova dei tempi); 2 FD al 40% risolto solo in parte.

### IH 2-approssimato contro esatto (medie su repliche e fold)

| N_FD | Rumore_% | Metodo | Istanze_risolte | IH_approx | IH_esatto | Rapporto |
|---|---|---|---|---|---|---|
| 1 | 5 | formula chiusa | 25/25 | 402.9 | 247.7 | 1.627 |
| 1 | 10 | formula chiusa | 25/25 | 802.9 | 501.9 | 1.6 |
| 1 | 20 | formula chiusa | 25/25 | 1549.3 | 999.2 | 1.55 |
| 1 | 30 | formula chiusa | 25/25 | 2291.7 | 1524.7 | 1.503 |
| 1 | 40 | formula chiusa | 25/25 | 2929.6 | 2017.6 | 1.452 |
| 2 | 5 | ILP | 25/25 | 916.1 | 543.2 | 1.687 |
| 2 | 10 | ILP | 25/25 | 1728.7 | 1051.7 | 1.644 |
| 2 | 20 | ILP | 25/25 | 2996.6 | 1946.3 | 1.54 |
| 2 | 40 | ILP | 8/25 | 4487.5 | 3309.1 | 1.356 |
| 4 | 5 | ILP | 25/25 | 1868.4 | 1096.4 | 1.704 |
| 4 | 10 | ILP | 25/25 | 3244.1 | 2032.9 | 1.596 |

## Rilevanza delle FD per l'obiettivo (analisi_rilevanza_fd.py)

### Calo di F1 al 40% sporcando una configurazione alla volta

| Configurazione | Colonne | Calo_medio | Modelli_significativi |
|---|---|---|---|
| Concetto rotta e distanza | 16 | 0.0832 | 4 |
| Concetto distanza | 2 | 0.0798 | 4 |
| FD Distance -> DistanceGroup | 2 | 0.0798 | 4 |
| FD Origin+Dest -> Distance | 3 | 0.0199 | 4 |
| Concetto aeroporto di origine | 7 | 0.0119 | 3 |
| Concetto aeroporto di destinazione | 7 | 0.0112 | 4 |
| FD Reporting_Airline -> IATA_CODE | 2 | 0.0028 | 0 |
| FD CRSDepTime -> DepTimeBlk | 2 | 0.0013 | 0 |
| FD DestAirportID -> DestState | 2 | 0.0008 | 0 |
| FD OriginAirportID -> OriginState | 2 | -0.0002 | 0 |
