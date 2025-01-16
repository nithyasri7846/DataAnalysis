# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Replace 'your_dataset.csv' with the name of your dataset file
df = pd.read_csv('train.csv')

# Check the column names
print("\nColumn Names:")
print(df.columns)

# Data distribution: Histograms
print("\nVisualizing Data Distributions with Histograms:")
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Visualizing data distribution of a specific column
print("\nDistribution of a Specific Column:")
column_name = input("Enter column name for distribution plot: ")
if column_name in df.columns:
    sns.histplot(df[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")
    plt.show()
else:
    print(f"Column '{column_name}' not found in the dataset.")

# Scatter plot for two columns
print("\nScatter Plot:")
x_col = input("Enter column name for X-axis: ")
y_col = input("Enter column name for Y-axis: ")
if x_col in df.columns and y_col in df.columns:
    df.plot.scatter(x=x_col, y=y_col, figsize=(8, 6))
    plt.title(f"Scatter Plot: {x_col} vs {y_col}")
    plt.show()
else:
    print(f"One or both columns not found in the dataset.")

# Pairplot for relationships between numerical columns
print("\nPairplot for Numerical Relationships:")
sns.pairplot(df.select_dtypes(include=np.number))
plt.show()

numerical_df = df.select_dtypes(include=np.number)

# Check if there are numerical columns in the dataset
if numerical_df.empty:
    print("No numerical columns found in the dataset. Correlation matrix cannot be computed.")
else:
    # Correlation matrix
    print("\nCorrelation Matrix:")
    correlation_matrix = numerical_df.corr()
    print(correlation_matrix)

    # Heatmap for correlations
    print("\nHeatmap of Correlations:")
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.show()

# Outlier detection using boxplot
print("\nBoxplot for Outlier Detection:")
boxplot_column = input("Enter column name for boxplot: ")
if boxplot_column in df.columns:
    sns.boxplot(x=df[boxplot_column])
    plt.title(f"Boxplot of {boxplot_column}")
    plt.show()
else:
    print(f"Column '{boxplot_column}' not found in the dataset.")

# Outlier analysis using IQR
print("\nOutlier Analysis Using IQR:")
iqr_column = input("Enter column name for IQR outlier detection: ")
if iqr_column in df.columns:
    Q1 = df[iqr_column].quantile(0.25)
    Q3 = df[iqr_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[iqr_column] < lower_bound) | (df[iqr_column] > upper_bound)]
    print(f"Outliers in {iqr_column}:")
    print(outliers)
else:
    print(f"Column '{iqr_column}' not found in the dataset.")

# Save the cleaned dataset (optional)
save_cleaned = input("\nDo you want to save the cleaned dataset? (yes/no): ")
if save_cleaned.lower() == 'yes':
    df.to_csv('cleaned_dataset.csv', index=False)
    print("Cleaned dataset saved as 'cleaned_dataset.csv'.")
else:
    print("Cleaned dataset not saved.")

print("\nEDA Complete!")
