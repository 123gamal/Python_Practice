import pandas as pd
from scipy import stats
df = pd.read_csv('C:\\Users\\gamal\\OneDrive\\Desktop\\file\\file\\iris_dataset.csv', header=None)
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'Label']
df = df.dropna().reset_index(drop=True)
num = df.iloc[:, 0:4]
zScore = stats.zscore(num)
zScoredf = pd.DataFrame(zScore, columns=num.columns)
zScoredf = zScoredf.abs()
threshold = 3

outlaires = (zScoredf > threshold).any(axis=1)
print(df[outlaires])
dfcleaned = df[~outlaires]
mean_values = df['sepal length'].mean()
df.loc[zScoredf['sepal length'] > threshold, 'sepal length'] = mean_values
mean_values = df['sepal width'].mean()
df.loc[zScoredf['sepal width'] > threshold, 'sepal width'] = mean_values
mean_values = df['petal length'].mean()
df.loc[zScoredf['petal length'] > threshold, 'petal length'] = mean_values
mean_values = df['petal width'].mean()
df.loc[zScoredf['petal width'] > threshold, 'petal width'] = mean_values


print(dfcleaned)
print(df)
print(df[outlaires])

df.to_csv('C:\\Users\\gamal\\OneDrive\\Desktop\\file\\file\\cleanDup_irisThree.csv', index=False)
