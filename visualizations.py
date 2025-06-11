# visualizations.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_bar_chart(data, x_col, y_col):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=x_col, hue=y_col)
    plt.title(f'{y_col} Distribution by {x_col}')
    plt.xlabel(x_col)
    plt.ylabel('Count')
    plt.legend(title=y_col)
    plt.tight_layout()
    return plt

def plot_line_chart(data, x_col, y_col):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x=x_col, y=y_col)
    plt.title(f'Trend of {y_col} Over {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    return plt

def plot_pie_chart(data, column):
    plt.figure(figsize=(8, 8))
    data[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'Distribution of {column}')
    plt.ylabel('')
    plt.tight_layout()
    return plt

def plot_heatmap(data):
    plt.figure(figsize=(10, 8))
    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    # Check if there are any numeric columns
    if numeric_data.empty:
        plt.text(0.5, 0.5, 'No numeric data available for correlation.', 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=15, color='red')
        plt.axis('off')  # Hide axes
        return plt

    correlation = numeric_data.corr()
    
    # Check if the correlation DataFrame is empty
    if correlation.empty:
        plt.text(0.5, 0.5, 'Correlation matrix is empty.', 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=15, color='red')
        plt.axis('off')  # Hide axes
        return plt

    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    return plt

def plot_scatter(data, x_col, y_col):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_col, y=y_col)
    plt.title(f'Scatter Plot of {y_col} vs {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    return plt