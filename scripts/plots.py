import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the path to the logs directory one level up
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

# Create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define file paths
log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

# Create handlers
info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Create a logger and set its level
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Capture all info and above
logger.addHandler(info_handler)
logger.addHandler(error_handler)

# Optional: Add a console handler if you want to see logs in the console
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)  # Or ERROR if you want only error messages in the console
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

def univariate_analysis(df, column):
    """Function to perform univariate analysis on a single column."""
    logger.info(f'Starting univariate analysis for column: {column}')
    plt.figure(figsize=(8, 6))
    
    try:
        if df[column].dtype == 'object':
            sns.countplot(x=column, data=df)
            plt.title(f'Count Plot of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
        else:
            sns.histplot(df[column], kde=True, bins=30)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
        
        plt.show()
        logger.info(f'Completed univariate analysis for column: {column}')
    except Exception as e:
        logger.error(f'Error during univariate analysis for column {column}: {e}')

def bivariate_analysis(df, column1, column2):
    """Function to perform bivariate analysis on two columns."""
    logger.info(f'Starting bivariate analysis for columns: {column1} and {column2}')
    plt.figure(figsize=(8, 6))
    
    try:
        if df[column1].dtype != 'object' and df[column2].dtype != 'object':
            sns.scatterplot(x=column1, y=column2, data=df)
            plt.title(f'Scatter Plot of {column1} vs {column2}')
            plt.xlabel(column1)
            plt.ylabel(column2)
        elif df[column1].dtype == 'object' or df[column1].dtype == 'int64':
            sns.boxplot(x=column1, y=column2, data=df)
            plt.title(f'Box Plot of {column2} by {column1}')
            plt.xlabel(column1)
            plt.ylabel(column2)

        plt.show()
        logger.info(f'Completed bivariate analysis for columns: {column1} and {column2}')
    except Exception as e:
        logger.error(f'Error during bivariate analysis for columns {column1} and {column2}: {e}')

def multivariate_analysis(df, columns):
    """Function to perform multivariate analysis on multiple columns."""
    logger.info(f'Starting multivariate analysis for columns: {columns}')
    
    try:
        if len(columns) > 2:
            sns.pairplot(df[columns])
            plt.show()
        else:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[columns].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Heatmap')
            plt.show()

        logger.info(f'Completed multivariate analysis for columns: {columns}')
    except Exception as e:
        logger.error(f'Error during multivariate analysis for columns {columns}: {e}')

def plot_correlation(df, col1, col2):
    """Plots the correlation between two numeric columns in a DataFrame."""
    logger.info(f'Starting correlation analysis between {col1} and {col2}')
    
    try:
        corr_coefficient = df[col1].corr(df[col2])
        logger.info(f'Correlation coefficient between {col1} and {col2}: {corr_coefficient:.4f}')
        
        plt.figure(figsize=(8, 6))
        sns.regplot(x=col1, y=col2, data=df, scatter_kws={'s':10}, line_kws={'color':'red'})
        
        plt.title(f"Correlation between {col1} and {col2}\n(correlation coefficient = {corr_coefficient:.4f})")
        plt.xlabel(col1)
        plt.ylabel(col2)
        
        plt.show()
        logger.info(f'Completed correlation analysis between {col1} and {col2}')
    except Exception as e:
        logger.error(f'Error during correlation analysis between {col1} and {col2}: {e}')

def visualize_promo_effects(df):
    """Analyzes promotional effects on sales."""
    logger.info('Starting analysis of promotional effects on sales')

    try:
        plt.figure(figsize=(10, 5))
        sales_by_promo = df.groupby('Promo')['Sales'].mean().reset_index()
        sns.barplot(x='Promo', y='Sales', data=sales_by_promo, palette='Blues')
        plt.title('Average Sales by Promo Status')
        plt.xlabel('Promo Status (0: No, 1: Yes)')
        plt.ylabel('Average Sales')
        plt.xticks(ticks=[0, 1], labels=['Without Promotion', 'With Promotion'])
        plt.show()

        plt.figure(figsize=(10, 5))
        customers_by_promo = df.groupby('Promo')['Customers'].mean().reset_index()
        sns.barplot(x='Promo', y='Customers', data=customers_by_promo, palette='Greens')
        plt.title('Average Number of Customers by Promo Status')
        plt.xlabel('Promo Status (0: No, 1: Yes)')
        plt.ylabel('Average Number of Customers')
        plt.xticks(ticks=[0, 1], labels=['Without Promotion', 'With Promotion'])
        plt.show()

        df['Sales_per_Customer'] = df['Sales'] / df['Customers']
        plt.figure(figsize=(10, 5))
        sales_per_customer_by_promo = df.groupby('Promo')['Sales_per_Customer'].mean().reset_index()
        sns.barplot(x='Promo', y='Sales_per_Customer', data=sales_per_customer_by_promo, palette='Reds')
        plt.title('Average Sales per Customer by Promo Status')
        plt.xlabel('Promo Status (0: No, 1: Yes)')
        plt.ylabel('Average Sales per Customer')
        plt.xticks(ticks=[0, 1], labels=['Without Promotion', 'With Promotion'])
        plt.show()

        logger.info('Completed analysis of promotional effects on sales')
    except Exception as e:
        logger.error(f'Error during promotional effects analysis: {e}')

def analyze_customer_behavior(df):
    """Analyzes customer behavior during store opening and closing times."""
    logger.info('Analyzing customer behavior based on store status (Open/Closed)')

    try:
        opening_summary = df.groupby('Open').agg({
            'Sales': 'sum',
            'Customers': 'sum'
        }).reset_index()

        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.bar(opening_summary['Open'].astype(str), opening_summary['Sales'], color=['red', 'green'])
        plt.title('Total Sales by Store Status (Open/Closed)')
        plt.xlabel('Store Status (0 = Closed, 1 = Open)')
        plt.ylabel('Total Sales')
        plt.xticks([0, 1], ['Closed', 'Open'])
        plt.grid(axis='y')

        plt.subplot(2, 1, 2)
        plt.bar(opening_summary['Open'].astype(str), opening_summary['Customers'], color=['red', 'green'])
        plt.title('Total Customers by Store Status (Open/Closed)')
        plt.xlabel('Store Status (0 = Closed, 1 = Open)')
        plt.ylabel('Total Customers')
        plt.xticks([0, 1], ['Closed', 'Open'])
        plt.grid(axis='y')

        plt.tight_layout()
        plt.show()

        logger.info('Completed analysis of customer behavior')
        return opening_summary
    except Exception as e:
        logger.error(f'Error during customer behavior analysis: {e}')




def analyze_competition_distance_effect_on_sales(df):
    """
    Analyzes the effect of distance to the next competitor on sales.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Sales' and 'CompetitionDistance' columns.

    Returns:
    pd.DataFrame: Summary of sales for different distance ranges to the next competitor.
    float: Correlation coefficient between CompetitionDistance and Sales.
    """
    logger.info('Starting analysis of competition distance effect on sales')
    
    # Define bins for competition distance
    bins = [0, 250, 500, 750, 1000, 1500, 2000, df['CompetitionDistance'].max()]
    labels = ['0-250', '251-500', '501-750', '751-1000', '1001-1500', '1501-2000', '2000+']

    # Create a new column for distance categories
    df['DistanceCategory'] = pd.cut(df['CompetitionDistance'], bins=bins, labels=labels, right=False)

    # Group by the new DistanceCategory and aggregate sales
    distance_summary = df.groupby('DistanceCategory').agg({
        'Sales': 'sum'
    }).reset_index()

    logger.info('Sales aggregated by distance category')

    # Plotting sales by distance category
    plt.figure(figsize=(10, 6))
    plt.bar(distance_summary['DistanceCategory'], distance_summary['Sales'], color='lightgreen')
    plt.title('Total Sales by Distance to Next Competitor')
    plt.xlabel('Distance to Next Competitor (meters)')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()





def analyze_assortment_effect_on_sales(df):
    """Analyzes the effect of assortment type on sales."""
    logger.info('Analyzing the effect of assortment type on sales')

    try:
        assortment_summary = df.groupby('Assortment').agg({
            'Sales': 'sum'
        }).reset_index()

        assortment_summary = assortment_summary.sort_values(by='Sales', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.bar(assortment_summary['Assortment'], assortment_summary['Sales'], color='skyblue')
        plt.title('Total Sales by Assortment Type')
        plt.xlabel('Assortment Type')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.show()

        logger.info('Completed analysis of assortment effect on sales')
        return assortment_summary
    except Exception as e:
        logger.error(f'Error during assortment effect analysis: {e}')