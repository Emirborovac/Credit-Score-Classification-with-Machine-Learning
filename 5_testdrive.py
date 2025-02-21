#!/usr/bin/env python3
import numpy as np
import pickle
from pathlib import Path

def load_model():
    """Load the trained model from pickle file."""
    model_path = Path("./Credit-Score-Data/Credit Score Data/models/random_forest_model.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_credit_score(model):
    """Get user input and make credit score prediction."""
    print("\nCredit Score Prediction System")
    print("-----------------------------")
    
    try:
        # Get user inputs
        inputs = {
            "Annual_Income": float(input("Annual Income: ")),
            "Monthly_Inhand_Salary": float(input("Monthly Inhand Salary: ")),
            "Num_Bank_Accounts": float(input("Number of Bank Accounts: ")),
            "Num_Credit_Card": float(input("Number of Credit Cards: ")),
            "Interest_Rate": float(input("Interest Rate: ")),
            "Num_of_Loan": float(input("Number of Loans: ")),
            "Delay_from_due_date": float(input("Average number of days delayed: ")),
            "Num_of_Delayed_Payment": float(input("Number of Delayed Payments: ")),
            "Outstanding_Debt": float(input("Outstanding Debt: ")),
            "Credit_History_Age": float(input("Credit History Age: ")),
            "Monthly_Balance": float(input("Monthly Balance: "))
        }
        
        # Create feature array
        features = np.array([[
            inputs["Annual_Income"],
            inputs["Monthly_Inhand_Salary"],
            inputs["Num_Bank_Accounts"],
            inputs["Num_Credit_Card"],
            inputs["Interest_Rate"],
            inputs["Num_of_Loan"],
            inputs["Delay_from_due_date"],
            inputs["Num_of_Delayed_Payment"],
            inputs["Outstanding_Debt"],
            inputs["Credit_History_Age"],
            inputs["Monthly_Balance"]
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        print("\nPrediction Results:")
        print("------------------")
        print(f"Predicted Credit Score: {prediction}")
        
        return prediction
        
    except ValueError as e:
        print("\nError: Please enter valid numerical values for all fields")
        return None
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return None

def main():
    """Main function to run predictions."""
    try:
        # Load the model
        model = load_model()
        
        while True:
            # Make prediction
            predict_credit_score(model)
            
            # Ask if user wants to make another prediction
            again = input("\nWould you like to make another prediction? (y/n): ").lower()
            if again != 'y':
                break
                
        print("\nThank you for using the Credit Score Prediction System!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()