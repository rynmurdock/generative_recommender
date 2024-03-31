
# by user row number in each_to_each
USER = 100
N_IMAGES_POOL = 1000

import pandas as pd
from PIL import Image
import random

srs = pd.read_hdf('flick_each_to_each_filtered.hdf', key='huh', index_col=0).iloc[USER]
print(srs)

new_df = pd.read_hdf('replace_flicker_each_to_each.hdf', key='huh', index_col=0)

interactions = srs.tolist()
interactions = [0] * len(srs.tolist())
image_ids = srs.index.tolist()

for _ in range(20):
    ind = random.randint(0, N_IMAGES_POOL)
    if interactions[ind] != 0:
        print('Already scored')
        continue
    Image.open(f'data/FLICKR-AES-001/40K/{image_ids[ind]}').show()
    score = input('1 to 5: ')
    score = 0. if score not in [str(i) for i in list(range(1, 10))] else float(score)
    interactions[ind] = score


new_df.iloc[USER] = interactions
new_df.to_hdf('replace_flicker_each_to_each.hdf', key='huh')
print(new_df.iloc[USER])
