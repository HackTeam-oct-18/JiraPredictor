from sklearn.decomposition import TruncatedSVD

import commons

# TODO use TF here

df = commons.read_ds('all')
embeds = commons.expand_nparray_of_lists(df['embedding'].values)

svd = TruncatedSVD(n_components=64, n_iter=50)
reduced_embeds = svd.fit_transform(embeds).tolist()

print('svd done')

df['reduced_embedding'] = reduced_embeds

df.to_csv("data/reduced.csv")