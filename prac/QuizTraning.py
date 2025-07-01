import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as pip
import seaborn as sns

df = pd.read_csv('C:\\Users\\gamal\\OneDrive\\Desktop\\iris - dummy.csv', header=None)
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'Label']

df = df.dropna().reset_index(drop=True)
target = df['Label']
features = df.iloc[:, :-1]
label_En = LabelEncoder().fit_transform(target)
target = pd.DataFrame(label_En, columns=['Label'])

features_En = MinMaxScaler().fit_transform(features)
features = pd.DataFrame(features_En, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])

#correlation = features.corr()
#sns.heatmap(correlation, annot=True, cmap="coolwarm", cbar=True, square=True)
#plt.title('correlation heatmap')
#plt.show()

df = pd.concat([features, target], axis=1)
df = df.drop(columns=['petal width'])


features2 = df.iloc[:, :-1]
correlation = features2.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", cbar=True, square=True)
plt.title('correlation heatmap')
plt.show()

print(df)
