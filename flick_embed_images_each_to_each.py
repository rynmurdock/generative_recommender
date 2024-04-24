


IM_PATH = 'data/FLICKR-AES-001/40K/'
for_model = 'kandinsky'

import pandas as pd
import torch
import tqdm
from PIL import Image

device = 'cuda'
if for_model == 'kandinsky':
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
    model = torch.compile(model.to(device))
else:
    import clip
    model, preprocess = clip.load(for_model, jit=False)
    _ = model.eval()


each_df = pd.read_hdf('flicker_each_to_each.hdf', index_col=0, key='huh')
all_image_features = []

images = each_df.columns.tolist()


bsz = 32
excepted = 0
done = 0
# now embed images with CLIP
for idx, imagen in enumerate(tqdm.tqdm(range(0, len(images), bsz))):
    #if idx >= 2048:
    #    column_numbers = [x for x in range(0, done)]  # list of columns' integer indices
    #    each_df = each_df.iloc[:, column_numbers]
    #    break
    try:
        image = torch.cat([preprocess(Image.open(IM_PATH+f'{images[idx]}')).unsqueeze(0).to(device) for idx in range(imagen, min(imagen+bsz, len(images)-1))])
    except Exception as e:
        print(e)
        [each_df.drop(columns=images[i], inplace=True) for i in range(imagen, min(imagen+bsz, len(images)-1))]
        excepted += 1
        continue

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        all_image_features.append(image_features)
        done += 1

all_image_features = torch.cat(all_image_features)
torch.save(all_image_features, 'image_features.pt')
each_df.to_hdf('flick_each_to_each_filtered', key='huh')
print(f'excepted: {excepted} done: {done}')

