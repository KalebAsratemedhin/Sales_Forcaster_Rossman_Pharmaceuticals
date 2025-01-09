import pandas as pd
import os
import logging

# Logging configuration
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(info_handler)
logger.addHandler(error_handler)

def data_loader(path):
    """Load data from a CSV file."""
    logger.info(f'Loading data from {path}')
    data = pd.read_csv(path)
    logger.info('Data loaded successfully')
    return data

def column_summary(df):
    """Generate a summary of columns in the DataFrame."""
    logger.info('Generating column summary')
    summary_data = []
    
    for col_name in df.columns:
        col_dtype = df[col_name].dtype
        num_of_nulls = df[col_name].isnull().sum()
        num_of_non_nulls = df[col_name].notnull().sum()
        num_of_distinct_values = df[col_name].nunique()
        
        if num_of_distinct_values <= 10:
            distinct_values_counts = df[col_name].value_counts().to_dict()
        else:
            top_10_values_counts = df[col_name].value_counts().head(10).to_dict()
            distinct_values_counts = {k: v for k, v in sorted(top_10_values_counts.items(), key=lambda item: item[1], reverse=True)}

        summary_data.append({
            'col_name': col_name,
            'col_dtype': col_dtype,
            'num_of_nulls': num_of_nulls,
            'num_of_non_nulls': num_of_non_nulls,
            'num_of_distinct_values': num_of_distinct_values,
            'distinct_values_counts': distinct_values_counts
        })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info('Column summary generated successfully')
    return summary_df

def impute_missing_values(df: pd.DataFrame, column: str, method: str = 'mean') -> pd.DataFrame:
    """Impute missing values in the specified column of a DataFrame."""
    logger.info(f'Imputing missing values in column: {column} using method: {method}')
    
    if method not in ['mean', 'median', 'mode']:
        logger.error("Invalid imputation method provided.")
        raise ValueError("Method must be 'mean', 'mode' or 'median'")
    
    if method == 'mean':
        value = df[column].mean()
    elif method == 'median':
        value = df[column].median()
    elif method == 'mode':
        value = df[column].mode()[0]
    
    df[column].fillna(value, inplace=True)
    logger.info(f'Missing values imputed in column: {column}')
    
    return df

def impute_with_historical_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear'."""
    logger.info('Imputing missing values with historical averages')
    
    mode_month = df['CompetitionOpenSinceMonth'].mode()[0]
    mode_year = df['CompetitionOpenSinceYear'].mode()[0]
    
    df['CompetitionOpenSinceMonth'].fillna(mode_month, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(mode_year, inplace=True)
    
    logger.info('Missing values imputed with historical averages')
    return df

def handle_no_promo_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing promo data."""
    logger.info('Handling missing promo data')
    
    df['Promo2SinceWeek'].fillna(0, inplace=True)  
    df['Promo2SinceYear'].fillna(0, inplace=True)  
    df['PromoInterval'].fillna('No Promo', inplace=True)

    logger.info('Missing promo data handled successfully')
    return df

def save_dataframe_to_csv(df: pd.DataFrame, filename: str, directory: str = '../../data') -> None:
    """Save a DataFrame to a CSV file."""
    logger.info(f'Saving DataFrame to {directory}/{filename}')
    
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    
    df.to_csv(file_path, index=False)
    logger.info(f'DataFrame saved to {file_path}')