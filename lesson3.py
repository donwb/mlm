from pandas import read_csv
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(url, names=names)
print(data.shape)

print(data.head(n=5))
print(data.tail(3))

print(data.columns)

print(data.describe())
