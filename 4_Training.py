#!/usr/bin/env python3
"""
Credit Score Model Training Script
--------------------------------
This script trains a Random Forest model on the preprocessed training data
and saves the model for later use.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
processed_dir = Path("./Credit-Score-Data/Credit Score Data/processed")
model_dir = Path("./Credit-Score-Data/Credit Score Data/models")
model_dir.mkdir(exist_ok=True)

try:
    # Load only the training data
    logger.info("Loading training data...")
    X_train = np.load(processed_dir / "X_train.npy", allow_pickle=True)
    y_train = np.load(processed_dir / "y_train.npy", allow_pickle=True)
    
    logger.info(f"Loaded training data with {len(X_train)} samples")

    # Initialize and train Random Forest model
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Perform cross-validation
    logger.info("Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train final model on full training set
    logger.info("Training final model on full training set...")
    model.fit(X_train, y_train)

    # Save the model
    model_path = model_dir / "random_forest_model.pkl"
    logger.info(f"Saving model to {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info("Model training completed successfully!")

except FileNotFoundError as e:
    logger.error(f"Required data file not found: {str(e)}")
    raise
except Exception as e:
    logger.error(f"Error during model training: {str(e)}")
    raise