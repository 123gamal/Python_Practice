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

# 1. Data Cleaning -----------------------------------------------------------------------------------------------------

# Check for missing values ---------------------------------------------------------------------------------------------

print(df.shape)
print(df.isna().sum())
print(df.isna().sum().sum())
print(df.isna().any())
print(df[df.isna().any(axis=1)])
print(df.info())

# because the missing value is more than the half or rows and that is too much -----------------------------------------
# in this  case the fill is better than the drop -----------------------------------------------------------------------

#there is no nulls -----------------------------------------------------------------------------------------------------

id_col = df["id"]
age_col = df["age"]
gender_col = df["gender"]
dataset_col = df["dataset"]
cp_col = df["cp"]
num_col = df["num"]

# fill the non-numeric column ------------------------------------------------------------------------------------------
# we can't fill the non-numeric with anything but the mode -------------------------------------------------------------

fbs_col = df["fbs"]
fbs_col = fbs_col.fillna(fbs_col.mode().iloc[0])


restecg_col = df["restecg"]
restecg_col = restecg_col.fillna(restecg_col.mode().iloc[0])

exang_col = df["exang"]
exang_col = exang_col.fillna(exang_col.mode().iloc[0])


slope_col = df["slope"]
slope_col = slope_col.fillna(slope_col.mode().iloc[0])

thal_col = df["thal"]
thal_col = thal_col.fillna(thal_col.mode().iloc[0])

# fill the numeric column ----------------------------------------------------------------------------------------------

trestbps_col = df["trestbps"]
trestbps_col = trestbps_col.fillna(trestbps_col.mode().iloc[0])

chol_col = df["chol"]
chol_col = chol_col.fillna(int(chol_col.mean()))

thalch_col = df["thalch"]
thalch_col = thalch_col.fillna(int(thalch_col.mean()))

oldpeak_col = df["oldpeak"]
oldpeak_col = oldpeak_col.fillna(oldpeak_col.mean())

ca_col = df["ca"]
ca_col = ca_col.fillna(ca_col.mode().iloc[0])

# reassemble the data set ----------------------------------------------------------------------------------------------

df = pd.concat([id_col,age_col,gender_col,dataset_col,cp_col,trestbps_col,chol_col,fbs_col,restecg_col,thalch_col,
                exang_col,oldpeak_col,slope_col,ca_col,thal_col,num_col] , axis=1)

# Check for missing values ---------------------------------------------------------------------------------------------

print(df.shape)
print(df.isna().sum())
print(df.isna().sum().sum())
print(df.isna().any())
print(df[df.isna().any(axis=1)])

# Handle duplicates-----------------------------------------------------------------------------------------------------
# check for duplicates -------------------------------------------------------------------------------------------------

print(df.shape)
duplicates = df.duplicated()
print(duplicates.sum())

# there is no duplicates -----------------------------------------------------------------------------------------------
# Handle outliers-------------------------------------------------------------------------------------------------------

# numeric columns ------------------------------------------------------------------------------------------------------

numeric_columns = df.select_dtypes(include=['float64', 'int64'])

# Check outliers using Z-score -----------------------------------------------------------------------------------------

threshold = 3
z_scores = stats.zscore(numeric_columns)
z_scores_df = pd.DataFrame(z_scores, columns=numeric_columns.columns)
outliers = (z_scores_df.abs() > threshold).any(axis=1)
print(outliers.sum())
print(df[outliers])

# in this case the number of the outlier is small so it will be easier to drop it than to fill it ----------------------

df = df[~outliers]
print(df.shape)

#-----------------------------------------------------------------------------------------------------------------------

# 2. Data Preprocessing ------------------------------------------------------------------------------------------------

# Label encoding categorical variables ---------------------------------------------------------------------------------

gender_col = df["gender"]
gender_col2 = LabelEncoder().fit_transform(gender_col)
gender_col = pd.DataFrame(gender_col2, columns=["gender"])

dataset_col = df["dataset"]
dataset_col2 = LabelEncoder().fit_transform(dataset_col)
dataset_col = pd.DataFrame(dataset_col2, columns=["dataset"])

cp_col = df["cp"]
cp_col2 = LabelEncoder().fit_transform(cp_col)
cp_col = pd.DataFrame(cp_col2, columns=["cp"])

fbs_col = df["fbs"]
fbs_col2 = LabelEncoder().fit_transform(fbs_col)
fbs_col = pd.DataFrame(fbs_col2, columns=["fbs"])

restecg_col = df["restecg"]
restecg_col2 = LabelEncoder().fit_transform(restecg_col)
restecg_col = pd.DataFrame(restecg_col2, columns=["restecg"])

exang_col = df["exang"]
exang_col2 = LabelEncoder().fit_transform(exang_col)
exang_col = pd.DataFrame(exang_col2, columns=["exang"])

slope_col = df["slope"]
slope_col2 = LabelEncoder().fit_transform(slope_col)
slope_col = pd.DataFrame(slope_col2, columns=["slope"])

thal_col = df["thal"]
thal_col2 = LabelEncoder().fit_transform(thal_col)
thal_col = pd.DataFrame(thal_col2, columns=["thal"])

# Scale numeric data using MinMaxScaler --------------------------------------------------------------------------------
age_col = df["age"]
age_col2 = MinMaxScaler().fit_transform(age_col.values.reshape(-1, 1))
age_col = pd.DataFrame(age_col2, columns=["age"])

trestbps_col = df["trestbps"]
trestbps_col2 = MinMaxScaler().fit_transform(trestbps_col.values.reshape(-1, 1))
trestbps_col = pd.DataFrame(trestbps_col2, columns=["trestbps"])

chol_col = df["chol"]
chol_col2 = MinMaxScaler().fit_transform(chol_col.values.reshape(-1, 1))
chol_col = pd.DataFrame(chol_col2, columns=["chol"])

thalch_col = df["thalch"]
thalch_col2 = MinMaxScaler().fit_transform(thalch_col.values.reshape(-1, 1))
thalch_col = pd.DataFrame(thalch_col2, columns=["thalch"])

oldpeak_col = df["oldpeak"]
oldpeak_col2 = MinMaxScaler().fit_transform(oldpeak_col.values.reshape(-1, 1))
oldpeak_col = pd.DataFrame(oldpeak_col2, columns=["oldpeak"])

ca_col = df["ca"]
ca_col2 = MinMaxScaler().fit_transform(ca_col.values.reshape(-1, 1))
ca_col = pd.DataFrame(ca_col2, columns=["ca"])

num_col = df["num"]
num_col2 = MinMaxScaler().fit_transform(num_col.values.reshape(-1, 1))
num_col = pd.DataFrame(num_col2, columns=["num"])

# reassemble the data set ----------------------------------------------------------------------------------------------

df = pd.concat([id_col,age_col,gender_col,dataset_col,cp_col,trestbps_col,chol_col,fbs_col,restecg_col,thalch_col,
                exang_col,oldpeak_col,slope_col,ca_col,thal_col,num_col] , axis=1)

print(df.shape)
print(df)
# PCA -------------------------------------------------------------------------------
# Check for remaining NaN values
print("Before PCA, checking for NaN values...")
print(df.isna().sum())
print("Total NaN values:", df.isna().sum().sum())

# Handle remaining NaNs (e.g., fill them with the mean for numeric columns)
df = df.fillna(df.mean())

# Recheck to ensure there are no NaNs
print("After filling, checking for NaN values...")
print(df.isna().sum())
print("Total NaN values:", df.isna().sum().sum())

# Ensure numeric_columns contains only numeric data types
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
print("Numeric columns for PCA:", numeric_columns)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df[numeric_columns])
df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

# Handle class imbalance -----------------------------------------------------------------------------------------------
# Check if the dataset is imbalanced
target_col = 'num'  # Target column indicating class labels

# Apply RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled_ros, y_resampled_ros = ros.fit_resample(X, y)
print("Class distribution after RandomOverSampler:", Counter(y_resampled_ros))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled_smote, y_resampled_smote = smote.fit_resample(X, y)
print("Class distribution after SMOTE:", Counter(y_resampled_smote))

# 4. Feature Selection--------------------------------------------------------------------------------------------------
# Plot correlation heatmap----------------------------------------------------------------------------------------------
correlation_matrix = df.iloc[:, 1:-1].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap")
plt.show()
