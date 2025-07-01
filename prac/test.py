from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter

pd.set_option('future.no_silent_downcasting', True)
df = pd.read_csv('C:\\Users\\gamal\\OneDrive\\Desktop\\heart_disease_uci.csv')

print(df.shape)
print(df.isna().sum())
print(df.isna().sum().sum())
print(df.isna().any())
print(df[df.isna().any(axis=1)])
print(df.info())

categorical_columns = ['fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']
numeric_columns = ['trestbps', 'chol', 'thalch', 'oldpeak']

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode().iloc[0])

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

print(df.shape)
print(df.isna().sum())
print(df.isna().sum().sum())
print(df.isna().any())
print(df[df.isna().any(axis=1)])

print(df.shape)
duplicates = df.duplicated()
print(duplicates.sum())

numeric_columns = df.select_dtypes(include=['float64', 'int64'])

threshold = 3
z_scores = stats.zscore(numeric_columns)
z_scores_df = pd.DataFrame(z_scores, columns=numeric_columns.columns)
outliers = (z_scores_df.abs() > threshold).any(axis=1)
print(outliers.sum())
print(df[outliers])

df = df[~outliers]
print(df.shape)

label_columns = ['gender', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
label_encoder = LabelEncoder()

for col in label_columns:
    df[col] = label_encoder.fit_transform(df[col])

scaler = MinMaxScaler()

numeric_cols_to_scale = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'num']
for col in numeric_cols_to_scale:
    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

print("Before PCA, checking for NaN values...")
print(df.isna().sum())
print("Total NaN values:", df.isna().sum().sum())

df = df.fillna(df.mean())

print("After filling, checking for NaN values...")
print(df.isna().sum())
print("Total NaN values:", df.isna().sum().sum())

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
print("Numeric columns for PCA:", numeric_columns)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df[numeric_columns])
df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
print("the PCA DataFrame")
print(df_pca)

target_col = 'num'


df['num'] = df['num'].astype('category')


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['num'])

print("Class distribution before handling imbalance:", Counter(y))


X = df.drop(columns=['id', 'num'])

ros = RandomOverSampler(random_state=42)
X_resampled_ros, y_resampled_ros = ros.fit_resample(X, y)
print("Class distribution after RandomOverSampler:", Counter(y_resampled_ros))


smote = SMOTE(random_state=42)
X_resampled_smote, y_resampled_smote = smote.fit_resample(X, y)
print("Class distribution after SMOTE:", Counter(y_resampled_smote))


numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap")
plt.show()

df_cleaned = df.drop(columns=['id', 'dataset'])

corr_matrix = df_cleaned.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title("Correlation Heatmap After Dropping 'id' and 'dataset'")
plt.show()

