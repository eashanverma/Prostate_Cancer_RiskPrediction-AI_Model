import os
import pandas as pd

base_dir = './'
img_dir = os.path.join(base_dir, 'Train Imgs')
map_dirs = [os.path.join(base_dir, f'Maps{i}_T') for i in range(1, 7)]

image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

data = []

for img_file in image_files:
    img_path = './' + os.path.join('Train Imgs', img_file).replace('\\', '/')
    maps = []

    for i, map_dir in enumerate(map_dirs, start=1):
        map_path = './' + os.path.join(f'Maps{i}_T', img_file.replace('.jpg', '_classimg_nonconvex.png')).replace('\\', '/')

        if os.path.exists(map_path):
            maps.append(map_path)
        else:
            maps.append(None)

    data.append([img_path] + maps)

columns = ['image_path'] + [f'map{i}_path' for i in range(1, 7)]
df = pd.DataFrame(data, columns=columns)

df.to_excel('Train_with_map.xlsx', index=False)