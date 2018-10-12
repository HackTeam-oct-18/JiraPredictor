import time

from ggplot import *
from sklearn.manifold import TSNE

import commons

df = commons.read_ds('all')
data = commons.expand_nparray_of_lists(df['embedding'].values)

time_start = time.time()
tsne = TSNE(n_components=64, verbose=1, perplexity=30.0, n_iter=250)
tsne_results = tsne.fit_transform(data)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

df['x-tsne'] = tsne_results[:, 0]
df['y-tsne'] = tsne_results[:, 1]

###
# df.to_csv("data/tsne.csv")
###

df = commons.read_ds('tsne')

new_df = df[['x-tsne', 'y-tsne', 'time']]

chart = ggplot(new_df, aes(x='x-tsne', y='y-tsne', color='time')) \
        + geom_point(size=50, alpha=0.1) \
        + ggtitle("tSNE dimensions colored by classes")
# print(chart)
chart
chart.show()
