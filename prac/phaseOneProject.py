from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer


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
    #df[col] = scaler.fit_transform(df[[col]])


print("Before PCA, checking for NaN values...")
print(df.isna().sum())
print("Total NaN values:", df.isna().sum().sum())

df = df.fillna(df.mean())

print("After filling, checking for NaN values...")
print(df.isna().sum())
print("Total NaN values:", df.isna().sum().sum())

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
print("Numeric columns for PCA:", numeric_columns)


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

print('-------------------------------------------')
print(df_cleaned)



#-------------------------------------------------------------------------------------------------------

target_col = 'num'
df['num'] = df['num'].astype('category')

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['num'])

X = df.drop(columns=['id', 'num'])

X_train, X_val, y_train, y_val = train_test_split(X_resampled_smote, y_resampled_smote, test_size=0.2, random_state=42)

#SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_val)
print("\nSVM Classification Report:")
print(classification_report(y_val, svm_predictions))

#Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_val)
print("\nRandom Forest Classification Report:")
print(classification_report(y_val, rf_predictions))

#Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_val)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_val, lr_predictions))

#Unsupervised Learning (K-Means)
#k_values = range(1, 11)
k_values = range(1, 21)
inertia = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Apply KMeans for each k
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)

#Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()


#---------------------------------------------------------------------------------------
# Running the model without PCA
print("=== Model Performance WITHOUT PCA ===")
X_train, X_val, y_train, y_val = train_test_split(X_resampled_smote, y_resampled_smote, test_size=0.2, random_state=42)

# SVM Model
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_val)
print("\nSVM WITHOUT PCA:")
print(classification_report(y_val, svm_predictions))

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_val)
print("\nRandom Forest WITHOUT PCA:")
print(classification_report(y_val, rf_predictions))

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_val)
print("\nLogistic Regression WITHOUT PCA:")
print(classification_report(y_val, lr_predictions))

# Applying PCA
print("\n=== Model Performance WITH PCA ===")
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)


#---------- #
# # SVM with PCA
svm_model.fit(X_train_pca, y_train)
svm_predictions_pca = svm_model.predict(X_val_pca)
# print("\nSVM WITH PCA:")
# print(classification_report(y_val, svm_predictions_pca))

# Random Forest with PCA
#---------- #
rf_model.fit(X_train_pca, y_train)
rf_predictions_pca = rf_model.predict(X_val_pca)
# print("\nRandom Forest WITH PCA:")
# print(classification_report(y_val, rf_predictions_pca))

# Logistic Regression with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_cleaned.drop(columns=['num']))

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_pca_resampled, y_pca_resampled = smote.fit_resample(X_pca, y)

# Train-test split
X_train_pca, X_val_pca, y_train_pca, y_val_pca = train_test_split(
    X_pca_resampled, y_pca_resampled, test_size=0.2, random_state=42
)

# Logistic Regression with PCA
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_pca, y_train_pca)
lr_predictions_pca = lr_model.predict(X_val_pca)


#---------- #
# print("\nLogistic Regression WITH PCA:")
# print(classification_report(y_val_pca, lr_predictions_pca, zero_division=0))









# ---------------------Selected Featres-------------------------
selected_features = ['age', 'chol', 'thalch', 'oldpeak', 'ca']
X_selected = df_cleaned[selected_features]
smote = SMOTE(random_state=42)
X_selected_resampled, y_selected_resampled = smote.fit_resample(X_selected, y)

# Train-test split
X_train_sel, X_val_sel, y_train_sel, y_val_sel = train_test_split(
    X_selected_resampled, y_selected_resampled, test_size=0.2, random_state=42
)

# Logistic Regression with Selected Features
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_sel, y_train_sel)
lr_predictions_sel = lr_model.predict(X_val_sel)

# Classification report
#---------- #
# print("\nLogistic Regression WITH Selected Features:")
# print(classification_report(y_val_sel, lr_predictions_sel, zero_division=0))


# Correlation matrix for the selected features
selected_features_corr = X_selected.corr()

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(selected_features_corr, annot=True, cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap for Selected Features")
plt.show()




# Ensure proper resampling of selected features
selected_features = ['age', 'chol', 'thalch', 'oldpeak', 'ca']
X_selected = df_cleaned[selected_features]

# Apply SMOTE to handle class imbalance for selected features
smote = SMOTE(random_state=42)
X_selected_resampled, y_selected_resampled = smote.fit_resample(X_selected, y)

# Train-test split for selected features
X_train_sel, X_val_sel, y_train_sel, y_val_sel = train_test_split(
    X_selected_resampled, y_selected_resampled, test_size=0.2, random_state=42
)

# Random Forest with Selected Features
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_sel, y_train_sel)


rf_predictions_sel = rf_model.predict(X_val_sel)
print("\nRandom Forest WITH Selected Features:")
print(classification_report(y_val_sel, rf_predictions_sel))

# Correlation matrix for the selected features
selected_features_corr = X_selected.corr()

# Plot the correlation heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(selected_features_corr, annot=True, cmap="coolwarm", cbar=True, square=True)
# plt.title("Correlation Heatmap for Selected Features")
# plt.show()

#-------------------------with all , without----------------------------------------
target_col = 'num'


print("Columns in df_cleaned:", df_cleaned.columns)


# Drop 'id' and 'dataset' only if they exist
X_all_features = df_cleaned.drop(columns=[target_col, 'id', 'dataset'], errors='ignore')


# Drop the selected features in addition to the target column
selected_features = ['age', 'chol', 'thalch', 'oldpeak', 'ca']  # Adjust based on your selection
X_without_selected = df_cleaned.drop(columns=selected_features + [target_col], errors='ignore')


smote = SMOTE(random_state=42)

# All features
X_all_resampled, y_all_resampled = smote.fit_resample(X_all_features, y)

# Without selected features
X_without_resampled, y_without_resampled = smote.fit_resample(X_without_selected, y)

# Train-test split for "all features"
X_train_all, X_val_all, y_train_all, y_val_all = train_test_split(
    X_all_resampled, y_all_resampled, test_size=0.2, random_state=42
)

# Train-test split for "without selected features"
X_train_without, X_val_without, y_train_without, y_val_without = train_test_split(
    X_without_resampled, y_without_resampled, test_size=0.2, random_state=42
)

# Logistic Regression for all features
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_all, y_train_all)
lr_predictions_all = lr_model.predict(X_val_all)

print("\nLogistic Regression WITH ALL Features:")
print(classification_report(y_val_all, lr_predictions_all, zero_division=0))

# Logistic Regression for without selected features
lr_model.fit(X_train_without, y_train_without)
lr_predictions_without = lr_model.predict(X_val_without)

#---------- #
# print("\nLogistic Regression WITHOUT Selected Features:")
# print(classification_report(y_val_without, lr_predictions_without, zero_division=0))


#--------------------------------------------------------------------------------------------------------------------
ros = RandomOverSampler(random_state=42)
X_resampled_ros, y_resampled_ros = ros.fit_resample(X, y)
print("Class distribution after RandomOverSampler:", Counter(y_resampled_ros))

# Handling imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled_smote, y_resampled_smote = smote.fit_resample(X, y)
print("Class distribution after SMOTE:", Counter(y_resampled_smote))


# Handling imbalance without oversampling (original data)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model on Original Data
print("=== Model Performance WITHOUT Resampling ===")

# Model performance WITHOUT Resampling
print("=== Model Performance WITHOUT Resampling ===")
# SVM Model
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_val)
# print("\nSVM WITHOUT Resampling:")
# print(classification_report(y_val, svm_predictions, zero_division=1))

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_val)
# print("\nRandom Forest WITHOUT Resampling:")
# print(classification_report(y_val, rf_predictions, zero_division=1))

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_val)
print("\nLogistic Regression WITHOUT Resampling:")
print(classification_report(y_val, lr_predictions, zero_division=1))

# Running the model on the data resampled using SMOTE
X_train, X_val, y_train, y_val = train_test_split(X_resampled_smote, y_resampled_smote, test_size=0.2, random_state=42)

# SVM Model on SMOTE
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions_smote = svm_model.predict(X_val)
#---------- #
# print("\nSVM WITH SMOTE:")
# print(classification_report(y_val, svm_predictions_smote, zero_division=1))

# Random Forest on SMOTE
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions_smote = rf_model.predict(X_val)
#---------- #
# print("\nRandom Forest WITH SMOTE:")
# print(classification_report(y_val, rf_predictions_smote, zero_division=1))

# Logistic Regression on SMOTE
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions_smote = lr_model.predict(X_val)
print("\nLogistic Regression WITH SMOTE:")
print(classification_report(y_val, lr_predictions_smote, zero_division=1))

# Running the model on the data resampled using RandomOverSampler
X_train, X_val, y_train, y_val = train_test_split(X_resampled_ros, y_resampled_ros, test_size=0.2, random_state=42)

# SVM Model on RandomOverSampler Data
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions_ros = svm_model.predict(X_val)
print("\nSVM WITH RandomOverSampler:")
print(classification_report(y_val, svm_predictions_ros, zero_division=1))

# Random Forest on RandomOverSampler Data
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions_ros = rf_model.predict(X_val)
print("\nRandom Forest WITH RandomOverSampler:")
print(classification_report(y_val, rf_predictions_ros, zero_division=1))

# Logistic Regression on RandomOverSampler Data
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions_ros = lr_model.predict(X_val)
#---------- #
# print("\nLogistic Regression WITH RandomOverSampler:")
# print(classification_report(y_val, lr_predictions_ros, zero_division=1))
