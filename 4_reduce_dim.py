from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib

import commons

# TODO use TF here
# This script performs
# 1 linear dimensionality reduction from 512 to 64 embedding
# 2 saving model and reduced embedding to fs

all_df = commons.read_ds('embedded')
train_df = all_df[int(all_df.shape[0] * commons.test_train_ration + 0.5):]
train_embeds = commons.expand_nparray_of_lists(train_df['embedding'].values)
all_embeds = commons.expand_nparray_of_lists(all_df['embedding'].values)

print('Fitting TruncatedSVD with train data')
svd = TruncatedSVD(n_components=128, n_iter=50, random_state=42)
svd.fit(train_embeds)

print('Reducing dimensionality for all data set')
reduced_embeds = svd.transform(all_embeds).tolist()

print('Got SVD model that linearly reduces %d components to %d ones and saves %.2f%% of source variance' %
      (train_embeds.shape[1], svd.n_components, svd.explained_variance_ratio_.sum() * 100.))

print('Saving TruncatedSVD model')
joblib.dump(svd, 'models_cache/truncated-svd64.dec', 1)

print('Saving reduced data')
all_df['reduced_embedding'] = reduced_embeds
all_df.to_csv("data/reduced.csv")
