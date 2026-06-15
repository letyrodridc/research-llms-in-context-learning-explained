import pandas as pd
import glob

# Files to read
files = ['annotations_nico_final.csv', 'annotations_carmen_final.csv', 'annotations_leticia_final.csv']

# Read and concatenate
dfs = []
for file in files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not dfs:
    print("No data found.")
    exit()

combined_df = pd.concat(dfs, ignore_index=True)

# Define dimensions
dimensions = {
    'textual_groundedness': 'human_TG',
    'hallucination_free': 'human_HF',
    'concept_counting': 'human_CC',
    'comprehensibility': 'human_CP',
    'conciseness': 'human_Cn',
    'specificity': 'human_S',
    'discriminativeness': 'human_LD',
    'instruction_following': 'human_IF',
    'logical_coherence': 'human_LC'
}

# Calculate absolute distances
for judge_col, human_col in dimensions.items():
    dist_col = f'dist_{judge_col}'
    combined_df[dist_col] = (combined_df[judge_col] - combined_df[human_col]).abs()

# Save to CSV
combined_df.to_csv('comparison_results.csv', index=False)
print("Saved comparison results to comparison_results.csv")
