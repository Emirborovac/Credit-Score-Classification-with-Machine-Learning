#!/usr/bin/env python3
"""
Credit Score Model Evaluation Script
----------------------------------
This script evaluates the trained model on the held-out test set
and provides detailed performance metrics.
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model():
    """Load the trained model and test data, then evaluate performance."""
    try:
        # Define paths
        model_path = Path("./Credit-Score-Data/Credit Score Data/models/random_forest_model.pkl")
        processed_dir = Path("./Credit-Score-Data/Credit Score Data/processed")
        
        # Load the model
        logger.info("Loading the trained model...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        # Load test data specifically
        logger.info("Loading test data...")
        X_test = np.load(processed_dir / "X_test.npy", allow_pickle=True)
        y_test = np.load(processed_dir / "y_test.npy", allow_pickle=True)
        
        # Make predictions
        logger.info("Making predictions on test data...")
        predictions = model.predict(X_test)
        
        # Calculate and display metrics
        print("\nModel Performance on Test Set:")
        print("=============================")
        
        # Classification Report
        print("\nClassification Report:")
        print("---------------------")
        class_report = classification_report(y_test, predictions)
        print(class_report)
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        print("----------------")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Class-wise accuracy
        print("\nClass-wise Accuracy:")
        print("-------------------")
        classes = np.unique(y_test)
        for i, class_name in enumerate(classes):
            class_correct = cm[i, i]
            class_total = cm[i].sum()
            class_accuracy = class_correct/class_total
            print(f"{class_name}: {class_correct}/{class_total} = {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
        
        # Overall accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Model parameters and feature importance
        print("\nModel Parameters and Feature Importance:")
        print("--------------------------------------")
        print(f"Number of trees: {model.n_estimators}")
        print(f"Max depth: {model.max_depth}")
        
        features = ["Annual_Income", "Monthly_Inhand_Salary", 
                   "Num_Bank_Accounts", "Num_Credit_Card", 
                   "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment", 
                   "Outstanding_Debt", "Credit_History_Age", 
                   "Monthly_Balance"]
        
        # Feature importance visualization
        importance_dict = {feat: imp for feat, imp in zip(features, model.feature_importances_)}
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        plt.figure(figsize=(12, 6))
        plt.bar(sorted_importance.keys(), sorted_importance.values())
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
        
        # Print feature importance
        print("\nFeature Importance Ranking:")
        for feat, imp in sorted_importance.items():
            print(f"- {feat}: {imp:.4f}")
        
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_model()