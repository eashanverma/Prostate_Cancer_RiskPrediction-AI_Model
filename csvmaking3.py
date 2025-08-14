import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

df = pd.read_csv("Train_Flattened.csv")

results = []
log_preview = []

def compute_gleason_label(mask_path):
    try:
        mask = np.array(Image.open(mask_path))

        # valid Gleason grades (3, 4, 5)
        valid_grades = [3, 4, 5]
        values, counts = np.unique(mask, return_counts=True)
        grade_counts = {
            int(v): int(c) for v, c in zip(values, counts) if v in valid_grades
        }

        if not grade_counts:
            return 0, 0, 0, 0  # No cancer (no valid grades)

        # Sort grades by frequency (area)
        sorted_grades = sorted(grade_counts.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_grades[0][0]
        secondary = sorted_grades[1][0] if len(sorted_grades) > 1 else primary
        gleason_score = primary + secondary

        # Assign risk label
        if gleason_score == 6:
            label = 1  # Low Risk (3+3)
        elif gleason_score == 7:
            label = 2  # Intermediate Risk
        elif gleason_score >= 8:
            label = 3  # High Risk
        else:
            label = 0  # fallback

        return primary, secondary, gleason_score, label

    except Exception as e:
        print(f"Error reading {mask_path}: {e}")
        return 0, 0, 0, 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Gleason Labels"):
    result = compute_gleason_label(row['mask_path'])
    results.append(result)

    if idx < 10:
        log_preview.append({
            "image": row['mask_path'].split('/')[-1],
            "primary": result[0],
            "secondary": result[1],
            "score": result[2],
            "label": result[3]
        })


df[['primary_grade', 'secondary_grade', 'gleason_score', 'risk_label']] = pd.DataFrame(results, index=df.index)

df.to_csv("Train_with_Gleason_and_Labels.csv", index=False)

print("\n🔍 Preview of First 10 Processed Samples:")
for entry in log_preview:
    print(f"  - {entry['image']} | Grades: {entry['primary']}+{entry['secondary']} = {entry['score']} → Label: {entry['label']}")

# Sanity check: show invalid Gleason scores (outside 0 or 6–10)
print("\n Invalid Gleason Scores (should NOT exist):")
invalid_scores = df[~df['gleason_score'].isin([0, 6, 7, 8, 9, 10])]
if not invalid_scores.empty:
    print(invalid_scores['gleason_score'].value_counts())
else:
    print(" None found. All Gleason scores are valid (0 or 6–10).")

print("\n Processing complete. Output saved to: Train_with_Gleason_and_Labels.csv")