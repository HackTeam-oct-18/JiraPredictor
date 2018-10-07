import pandas as pd

df = pd.read_csv('data/all.csv')
time = df['time']
estimate = df['estimate']

print('Human MAE', (time - estimate).abs().mean())
print('Human MAPE', ((time - estimate) * 100. / time).abs().mean())
print('Dummy model MAE', (time - time.median()).abs().mean())
print('Dummy model MAPE', ((time - time.median()) * 100. / time).abs().mean())
