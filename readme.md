1. First download the FLICKER-AES dataset to data/FLICKR-AES-001 with FLICKR-AES_image_labeled_by_each_worker.csv in data/ as well.
2. Build the user-to-image interaction matrix with flick_data_to_matrix.py
3. Create CLIP image embeddings for each image with flick_embed_images_each_to_each.py
4. Insert ratings as user number 100 using flick_score_images.py
5. Optimize user embeddings from proxy_item_learn.py
6. Visualize user embeddings with Kandinsky or IP Adapter by adding them to a CLIP embedding.

