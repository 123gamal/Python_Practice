import pandas as pd

# Load dataset and set column headers
df = pd.read_csv('C:\\Users\\gamal\\OneDrive\\Desktop\\file\\file\\iris_dataset.csv', header=None)
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'Label']

# Display basic information
print(df.shape)
print(df.info())
count = df['Label'].value_counts()
print(count)

######################################## Missing Values ######################################
pd.set_option('display.float_format', '{:.2f}'.format)

# Check for missing values
print("Missing values per row and column:")
print(df.isna())
print("Any missing values in each column?")
print(df.isna().any())
print("Sum of missing values in each column:")
print(df.isna().sum())
print("Total number of missing values:")
print(df.isna().sum().sum())
print("Rows with any missing values:")
print(df[df.isna().any(axis=1)])
#
# # Drop rows with missing values and reset index
# df = df.dropna().reset_index(drop=True)
# print("DataFrame after dropping missing values:")
# print(df)

# Imputation using mean, median, and mode for numerical columns
# labels = df['Label']
# numeric = df.iloc[:, :4].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, coerce errors
#
# # Fill NaN values with mean, median, or mode as needed
# numeric = numeric.fillna(numeric.mean())  # Could also use median() or mode().iloc[0]
# print("Numeric data after filling NaN values:")
# print(numeric)
#
# # Concatenate numeric columns and label column
# df = pd.concat([numeric, labels], axis=1)
# print("Final DataFrame after filling missing values:")
# print(df.to_string())  # Print full DataFrame
#
# ######################################## Duplicates ######################################
#
# # Check for duplicate rows and print them
# duplicates = df.duplicated()
# print("Duplicate rows:")
# print(df[duplicates])
#
# # Count and drop duplicates, keeping the last occurrence by default
# num_duplicates = df.duplicated().sum()
# print(f'Number of duplicate rows: {num_duplicates}')
# df = df.drop_duplicates().reset_index(drop=True)
# print("DataFrame after dropping duplicates:")
# print(df)
#
# # Custom drop based on a specific column subset (e.g., 'petal width')
# df_custom_cleaned = df.drop_duplicates(subset=['petal width'])
# print("DataFrame after custom duplicate drop on 'petal width':")
# print(df_custom_cleaned)
#
# ######################################## Example Duplicate Check ######################################
#
# # Example DataFrame for demonstration of duplicate handling
# df_example = pd.DataFrame({
#     'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
#     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
#     'rating': [4, 4, 3.5, 15, 5]
# })
#
# # Find duplicates based on 'brand', keeping the last occurrence
# dup_example = df_example.duplicated(subset=['brand'], keep='last')
# print("Duplicate status for 'brand' column:")
# print(dup_example)
#
# # Drop duplicates based on 'brand' column, keeping the last occurrence
# d = df_example.drop_duplicates(subset=['brand'], keep='last')
# print("Example DataFrame after dropping duplicates on 'brand':")
# print(d)
