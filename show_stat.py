import pandas as pd

# This script performs
# 1 showing baseline numbers for given data set

df = pd.read_csv('data/all.csv')
time = df['time']
estimate = df['estimate']

human_me = time - estimate
human_mpe = human_me / time
print('Human MAE \t\t%.2fh' % human_me.abs().mean())
print('Human MSE \t\t%.2fhh' % (human_me * human_me).mean())
print('Human MAPE \t\t%.2f%%' % (human_mpe * 100.).abs().mean())
print('Human MSPE \t\t%.2f%%' % (human_mpe * human_mpe * 100.).mean())

median = time.median()
median_me = time - median
median_mpe = median_me / time
print('\nMedian time value is %.3fh' % median)
print('Median MAE  \t%.2fh' % median_me.abs().mean())
print('Median MSE  \t%.2fhh' % (median_me * median_me).abs().mean())
print('Median MAPE \t%.2f%%' % (median_mpe * 100.).abs().mean())
print('Median MSPE \t%.2f%%' % (median_mpe * median_mpe * 100.).abs().mean())

mean = time.mean()
mean_me = time - mean
mean_mpe = mean_me / time
print('\nMean time value is %.3fh' % mean)
print('Mean MAE \t\t%.2fh' % mean_me.abs().mean())
print('Mean MSE \t\t%.2fhh' % (mean_me * mean_me).abs().mean())
print('Mean MAPE \t\t%.2f%%' % (mean_me * 100.).abs().mean())
print('Mean MSPE \t\t%.2f%%' % (mean_mpe * mean_mpe * 100.).abs().mean())

