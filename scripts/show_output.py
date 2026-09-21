import pandas as pd
df = pd.read_csv('output.csv')
print(f"{'#':<4} {'Route':<24} {'Week':<12} {'Cost/tkm':<10} {'Flagged':<16} {'Note':<6}")
print("-" * 80)
for i, row in df.iterrows():
    note = str(row['matched_note_id']) if str(row['matched_note_id']) != 'nan' else ''
    print(f"{i+1:<4} {row['route']:<24} {row['week_of']:<12} {row['cost_per_tonne_km']:<10.2f} {row['flagged']:<16} {note}")
print()
print("Verdict summary:")
print(df['flagged'].value_counts().to_string())
