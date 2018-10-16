from sklearn.decomposition import TruncatedSVD

import commons

# TODO use TF here
# This script performs
# 1 embedding dimensionality reduction from 512 to 64

df = commons.read_ds('embedded')
embeds = commons.expand_nparray_of_lists(df['embedding'].values)

svd = TruncatedSVD(n_components=64, n_iter=15)
reduced_embeds = svd.fit_transform(embeds).tolist()

print('svd done', df.shape)

df['reduced_embedding'] = reduced_embeds

df.to_csv("data/reduced.csv")