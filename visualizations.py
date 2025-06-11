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
    correlation = data.corr()
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