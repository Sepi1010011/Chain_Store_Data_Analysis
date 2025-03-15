import pandas as pd
import numpy as np


class DatasetLoader:
    def __init__(self, file_path: str):
        self.file_path :str = file_path
        self.df = self.load_dataset()

    def load_dataset(self) -> pd.DataFrame:
        """
        Loads dataset based on file type.
        Supports CSV, Excel, JSON, and Parquet formats.
        """
        try:
            if self.file_path.name.endswith(".csv"):
                df = pd.read_csv(self.file_path)
                
            elif self.file_path.name.endswith(".xlsx") or self.file_path.name.endswith(".xls"):
                df = pd.read_excel(self.file_path)
            
            elif self.file_path.name.endswith(".json"):
                df = pd.read_json(self.file_path)
            
            elif self.file_path.name.endswith(".parquet"):
                df = pd.read_parquet(self.file_path)
            
            else:
                raise ValueError("Unsupported file format. Supported formats: CSV, Excel, JSON, Parquet")
            
            return df
        
        except FileNotFoundError:
            print("Error: File not found. Please check the file path.")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
        
        return pd.DataFrame()
    
    def dataset_info(self) -> dict:
        """
        Prints basic information about the dataset.
        """
        if self.df.empty:
            print("No data loaded.")
            return
        
        data_info = {}
        
        data_info["Shape"] = self.df.shape
        data_info["Columns"] = self.df.columns.tolist()
        data_info["Data Types"] = self.df.dtypes
        data_info["Missing Values"] = self.df.isnull().sum()

        return data_info
        
    def dataset_adv_info(self) -> dict:
        """
        Prints advanced information about the dataset.
        """
        if self.df.empty:
            print("No data loaded.")
            return 
    
        data_adv_info = {}
    
        data_adv_info["Head"] = self.df.head()
        data_adv_info["Description"] = self.df.describe()
        
        return data_adv_info
        
    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the loaded DataFrame.
        """
        return self.df




