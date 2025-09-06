
> python "C:\Users\ReadyToUse\Desktop\Data\serie a\tools\predict_with_bookies.py" --home Lazio --away Verona --date 2025-09-30
Colonne e significato (italiano)
---------------------------------
- season: stagione (es. "2010-2011").
- squad: nome della squadra.
- roster_size: numero di giocatori considerati (righe giocatore valide dopo pulizia).
- starters_count: numero di giocatori considerati titolari (default: Starts > 10). Modificabile nello script.
- age_mean: età media della rosa (anni).
- age_mean_starters: età media dei titolari (se non disponibili ritorna age_mean).
- pct_under23: frazione di giocatori con età < 23 (0-1).
- pct_gk / pct_df / pct_mf / pct_fw: frazione di portieri/difensori/centrocampisti/attaccanti nella rosa.
- goals_tot: somma dei gol (Gls) della rosa.
- assists_tot: somma degli assist (Ast) della rosa.
- xg_tot: somma dell'xG dei giocatori.
- xga_proxy: proxy per gol subiti (se esiste la colonna GA viene sommata, altrimenti NaN).
- team_xg_per90: xG normalizzato per 90' (xg_tot / (Min_sum / 90)).
- conversion_sh_to_g: goals_tot / Sh_total (se Sh_total > 0), altrimenti NaN.
- progressions_tot: somma delle metriche PrgC + PrgP + PrgR per tutti i giocatori (se disponibili).
- yellow_cards: somma cartellini gialli (CrdY).
- red_cards: somma cartellini rossi (CrdR).
- top5_mean_gpa: media G+A dei primi 5 giocatori per G+A.
- bottom5_mean_gpa: media G+A dei 5 giocatori con meno G+A.
- gap_top5_bottom5_gpa: differenza tra top5_mean_gpa e bottom5_mean_gpa.
- age_std: deviazione standard dell'età nella rosa.

Note
-----
- Valori NaN appaiono se le colonne non sono presenti nei file giocatore o se la denominazione differisce molto (lo script cerca per substring e case-insensitive ma non è infallibile).
- Lo script pulisce righe di riepilogo tipo "Squad Total" e "Opponent Total" prima delle aggregazioni.

Controlli rapidi (suggeriti)
----------------------------
- Conteggio squadre per stagione:

```python
import pandas as pd
df = pd.read_csv('data/team_aggregates.csv')
print(df.groupby('season').size())
```

- Squadre con più NaN:

```python
print(df.isna().sum(axis=1).sort_values(ascending=False).head(10))
```

Se vuoi, posso aggiornare il file `data/README.md` esistente per includere questa descrizione o aggiungere uno script di validazione automatico che esegua i controlli sopra.
