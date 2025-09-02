import pandas as pd
df = pd.read_html("https://fbref.com/en/comps/11/2024-2025/2024-2025-Serie-A-Stats", attrs={"id":"stats_squads_standard_for"})
print(df)