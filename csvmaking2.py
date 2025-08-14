import pandas as pd

df = pd.read_excel('Train_with_map.xlsx')

flattened_rows = []

for _, row in df.iterrows():
    image_path = row['image_path']

    for expert_id in range(1, 7):
        map_col = f'map{expert_id}_path'
        mask_path = row.get(map_col)

        if pd.notna(mask_path):
            flattened_rows.append([image_path, mask_path, expert_id])

df_flat = pd.DataFrame(flattened_rows, columns=['image_path', 'mask_path', 'expert_id'])

df_flat.to_csv('Train_Flattened.csv', index=False)