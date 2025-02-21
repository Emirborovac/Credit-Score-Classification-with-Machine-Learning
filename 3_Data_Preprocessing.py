#!/usr/bin/env python3
"""
Credit Score Data Preprocessing Script
------------------------------------
This script preprocesses the credit score dataset by:
1. Converting categorical variables to numerical
2. Creating train/test split files
3. Saving preprocessed split datasets separately
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CreditDataPreprocessor:
    """Handles preprocessing of credit score data and dataset splitting."""
    
    def __init__(self, input_path: str, output_dir: str):
        """
        Initialize the preprocessor.
        
        Args:
            input_path (str): Path to the raw data CSV file
            output_dir (str): Directory to save processed files
        """
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.data = None
        self.features = ["Annual_Income", "Monthly_Inhand_Salary", 
                        "Num_Bank_Accounts", "Num_Credit_Card", 
                        "Interest_Rate", "Num_of_Loan", 
                        "Delay_from_due_date", "Num_of_Delayed_Payment", 
                        "Outstanding_Debt", "Credit_History_Age", 
                        "Monthly_Balance"]
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> None:
        """Load the raw credit score dataset."""
        try:
            logger.info(f"Loading data from {self.input_path}")
            self.data = pd.read_csv(self.input_path)
            logger.info(f"Successfully loaded {len(self.data)} records")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self) -> None:
        """Apply preprocessing steps to the data."""
        try:
            logger.info("Starting data preprocessing")
            
            # Convert Credit_Mix to numerical values
            self.data["Credit_Mix"] = self.data["Credit_Mix"].map({
                "Standard": 1,
                "Good": 2,
                "Bad": 0
            })
            
            logger.info("Preprocessing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise
    
    def create_train_test_split(self) -> None:
        """Create and save train/test split datasets."""
        try:
            logger.info("Creating train/test split from data")
            
            # First split the raw features and target
            X = self.data[self.features]
            y = self.data["Credit_Score"]
            
            # Create train/test split BEFORE saving any data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42
            )
            
            # Save training data
            train_features_path = self.output_dir / 'X_train.npy'
            train_target_path = self.output_dir / 'y_train.npy'
            np.save(train_features_path, X_train.to_numpy(), allow_pickle=True)
            np.save(train_target_path, y_train.to_numpy(), allow_pickle=True)
            logger.info(f"Saved training data to {train_features_path} and {train_target_path}")
            
            # Save test data separately
            test_features_path = self.output_dir / 'X_test.npy'
            test_target_path = self.output_dir / 'y_test.npy'
            np.save(test_features_path, X_test.to_numpy(), allow_pickle=True)
            np.save(test_target_path, y_test.to_numpy(), allow_pickle=True)
            logger.info(f"Saved test data to {test_features_path} and {test_target_path}")
            
            # Save as CSV for human readability
            train_data = X_train.copy()
            train_data['Credit_Score'] = y_train
            test_data = X_test.copy()
            test_data['Credit_Score'] = y_test
            
            train_data.to_csv(self.output_dir / 'train_split.csv', index=False)
            test_data.to_csv(self.output_dir / 'test_split.csv', index=False)
            
            logger.info(f"Training set size: {len(train_data)} records")
            logger.info(f"Test set size: {len(test_data)} records")
            
        except Exception as e:
            logger.error(f"Error creating train/test split: {str(e)}")
            raise

def main():
    """Main function to run the preprocessing pipeline."""
    try:
        # Initialize preprocessor
        preprocessor = CreditDataPreprocessor(
            input_path="./Credit-Score-Data/Credit Score Data/train.csv",
            output_dir="./Credit-Score-Data/Credit Score Data/processed"
        )
        
        # Run preprocessing pipeline
        preprocessor.load_data()
        preprocessor.preprocess_data()
        preprocessor.create_train_test_split()
        
        logger.info("Preprocessing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()