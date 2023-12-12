import pandas as pd
import os

def check_folder_exists(folder_path):
    """
    Check if a folder exists.

    Parameters:
    - folder_path (str): Path of the folder to check.

    Returns:
    - bool: True if the folder exists, False otherwise.
    """
    return os.path.exists(folder_path)

def create_folder(folder_path):
    """
    Create a folder if it does not exist.

    Parameters:
    - folder_path (str): Path of the folder to create.
    """
    if not check_folder_exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def store_csv(data,columns,path):
    # Create a 2D dataframe
    df = pd.DataFrame(data, columns=columns)

    # Save the dataframe to a CSV file
    df.to_csv(path, index=False)

    print(f"\nDataFrame saved to {path}")
