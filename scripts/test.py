import pandas as pd 

def add(filepath: str)-> pd.DataFrame:
    """
    Loads a dataset from a CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the loaded data.

    Example:
    --------
    >>> df = load_dataset('raw_analyst_ratings.csv')
    >>> df.head()
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Successfully loaded dataset: {filepath}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
    except pd.errors.ParserError:
        print(f"❌ Error parsing the file: {filepath}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")