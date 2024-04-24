import pandas as pd
import tqdm

ratings = pd.read_csv('data/FLICKR-AES_image_labeled_by_each_worker.csv', delimiter=',',)
ratings.columns = ratings.columns.str.strip()

each_to_each = pd.DataFrame(index=pd.unique(ratings.worker), columns=pd.unique(ratings.imagePair)).fillna(0)

for ind, row in tqdm.tqdm(ratings.iterrows(), total=len(ratings)):
    each_to_each.loc[row['worker'], row['imagePair']] = row['score']

each_to_each.to_hdf('flicker_each_to_each.hdf', key='huh')

