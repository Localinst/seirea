Checklist (requisiti estratti)
Capire e pulire i dati delle partite (schema fornito).
Creare caratteristiche (feature engineering) utili per predire risultato (H/D/A) e probabilità.
Analisi esplorativa e statistiche descrittive per guidare feature design.
Grafici e visualizzazioni principali per EDA e report.
Costruire baseline e modelli ML (classici e tree-based).
Validazione temporale, metriche e calibrazione delle probabilità.
Explainability (feature importance, SHAP) e salvataggio artifact.
Deployment minimo (script che prende due squadre e restituisce probabilità).
Documentazione, test e quality gates (build/lint/test).
Piano d’azione (alto livello)
Data cleaning & consistency (team name mapping, date parsing, missing).
EDA numerica e visuale per colonna+relazioni.
Feature engineering: aggregate per squadra, rolling windows, head-to-head, ELO.
Modelli & pipeline: baseline, logistic multiclass, XGBoost/LightGBM, ensemble.
Time-aware validation, calibrazione, metriche e selezione modello.
Explainability + report + script di inferenza.
Tests, export modelli e README operativo.
Dettaglio operativo — Dati e pulizia
Verifiche iniziali:
parsing Date con formato %d/%m/%y.
uniformare nomi squadre (alias, spazi, caratteri accentati).
tipi numerici per HS, AS, HST, AST, HF, etc.
gestire righe duplicate e partite con valori mancanti.
Corrections / canonicalizzazione:
creare mappa team_id persistente.
rimuovere/flaggare partite non valide (es. valori negativi).
Assunzioni note:
non ci sono formazioni/odds nel dataset: caratteristiche tiro/contest non includono xG; ELO/rest-days necessitano calcolo interno o dati esterni.
Feature engineering (prioritarie)
Match-level features (dirette):
home/away indicators, shots, shots on target, corners, fouls, cards.
goal difference, shots diff, SoT diff.
Squadra-season aggregates (a partire da match history fino alla data):
season-to-date: pts, wins/draws/losses, goals for/against, xG proxy (se non disponibile), possession proxy se presente.
per-90 normalizzati quando utile.
Rolling recent form (last N matches for each team, N=3,5,10):
rolling points per match, rolling goal diff, rolling shot conversion.
Home/Away specific history:
home form vs away form, home advantage metrics (home win %).
Head-to-head features:
last K head-to-head results, average goals H/A in H2H.
Opponent-strength features:
opponent season rank, opponent rolling form, ELO rating (calcolato iterativamente).
Time features:
days since last match for each team, matchday in season, midweek indicator.
Contextual (if available or enrichable):
promoted/relegated flags, roster size, injuries (if later add).
Encoding:
categorical: target encoding for teams (with CV to avoid leakage) or use numeric aggregates instead of one-hot for many teams.
Baseline e modelli da provare
Baseline semplici:
Always-predict-home-win baseline.
Recent-form baseline: prefer team with higher rolling points.
Modelli ML:
Multinomial Logistic Regression (with regularization).
Random Forest / ExtraTrees.
Gradient boosted trees: XGBoost, LightGBM, CatBoost.
Calibrated probabilities (Isotonic/Platt) su modelli non calibrati.
Ensemble stacking (meta-learner su out-of-fold preds).
Target:
Multiclass classification (H, D, A) con probabilità.
Alternativa: regressione su goal diff e trasformare in probabilità tramite simulazione (Poisson) — utile se si vuole simulare scorelines.
Validazione e split
Split temporale obbligatorio:
train su t <= T, test su periodi successivi (es. train 2017–2022, val 2023, test 2024–2025).
usare rolling-origin evaluation (time-series cross-validation) per stima robusta.
Evitare data leakage:
tutte le aggregate e rolling devono essere calcolate usando solo match precedenti alla data target.
Metriche:
Log loss (cross-entropy) — prioritaria per probabilità.
Brier score (calibrazione).
Accuracy, macro F1, per-class precision/recall.
Confusion matrix e skill over baseline.
ROC AUC (one-vs-rest) come supplementare.
Statistiche da riportare:
improvement vs baseline (delta logloss, delta Brier).
calibration curve / reliability diagram.
Grafici e visualizzazioni (EDA + modello)
Dati grezzi:
Istogrammi: goals scored per match, goal diff.
Barplot: win rates per squadra (home vs away).
Boxplots: distribution di HS/HST/SoT per risultato.
Heatmap di correlazione tra features numeriche.
Feature-time:
serie temporali di punti per squadra (season progression).
heatmap stagionale (posizione/pts per giornata).
Rolling & form:
rolling mean di punti/goal diff per selezione di squadre.
Modello e valutazione:
Confusion matrix (test set).
Curva di calibrazione e reliability diagram.
Log-loss per stagione (trend).
Feature importance (bar chart) + SHAP summary plot.
SHAP dependence plot per top feature.
Precision/Recall per classe e curva ROC (one-vs-rest).
Report visuale:
dashboard PDF o notebook con tutte le figure in artifacts.
Explainability e diagnostica
Feature importance globali (per modello), SHAP summary e decision plots.
PDP (partial dependence) o ICE per top features.
Analisi errori:
esaminare partite peggiori (alto loss) e pattern di errore (es. underpredict draws).
Calibration:
applicare isotonic/platt o temperature scaling, selezionare tramite val set.
Engineering, pipeline e artefatti
Script / notebook:
data/prepare_data.py — parsing + cleaning + team mapping.
features/build_features.py — generate rolling aggregates, ELO, save features parquet/csv per date.
models/train.py — pipelines, CV, hyperparam tuning, save model(s) in artifacts.
models/evaluate.py — produce metriche e grafici in artifacts.
predict/predict_next.py — API CLI che prende HomeTeam, AwayTeam, Date e restituisce probabilità H/D/A.
Notebooks EDA: notebooks/eda.ipynb, notebooks/model_experiments.ipynb.
Persistenza:
salvare pipeline e modelli (joblib), scaler, encoder e artefatti di calibration.
Logging & reproducibility:
fissare seed, salvare config esperimenti (yaml/json) e risultati (CSV).
Tests:
unit test per funzione di feature build (pytest), test integrazione veloce per pipeline (train on small sample).
CI/Quality:
run linter (flake8), format (black), tests in CI se servito.
Folder suggeriti: data, notebooks/, models/, predict/, artifacts.
Edge cases e rischi
Team nuovi/insufficienti dati: usare encoding basato su league average o prior Bayesian smoothing.
Cambi stagionali (nomi/squadre): mantenere mappa storica.
Draws sottostimate: spesso problema, lavorare su class-weighting o ottimizzazione della log-loss.
Cambi nelle regole/format: verificare anomalie nelle stagioni.
Dati mancanti sistematici per colonne avanzate: considerare imputazione con mediana o indicatori di missing.
Metriche di successo e quality gates
Acceptance criteria:
modello supera baseline naive in log loss e accuracy su test out-of-time.
probabilità ben calibrate (Brier score inferiore e calibration curve vicino alla diagonale).
script di inferenza che ritorna probabilità in <200ms (locale).
Quality gates pre-rilascio:
tutti i test unitari/pass (pytest).
pipeline senza errori su dataset completo (smoke run).
grafici principali generati e salvati.
modello salvato con metadata (version, date, metrics).
Verifiche finali:
eseguire models/evaluate.py su test set e generare report.
Deliverables raccomandati
Script e notebook elencati sopra.
Cartella artifacts con modelli, scaler, calibration, figure principali.
README.md operativo: come ricreare pipeline e come usare predict_next.py.
Tests minimi e file requirements.txt con versioni (scikit-learn, xgboost, lightgbm, shap, pandas, matplotlib, seaborn).