# =============================================================================
# STEP 1: Initial Data Loading
# =============================================================================

# Import required libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

#Read the dataset
data = pd.read_csv("./Credit-Score-Data/Credit Score Data/train.csv")
#print(data.head())

# =============================================================================
# STEP 2: Data Inspection
# =============================================================================

# Check data info/structure
#print(data.info())

# Check for null values
#print(data.isnull().sum())

# Check target variable distribution
#print(data["Credit_Score"].value_counts())

# =============================================================================
# STEP 3: Feature Exploration
# =============================================================================

# Explore occupation impact
#fig = px.box(data, 
#             x="Occupation",  
#             color="Credit_Score", 
#             title="Credit Scores Based on Occupation", 
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.show()

# Explore annual income impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Annual_Income", 
#             color="Credit_Score",
#             title="Credit Scores Based on Annual Income", 
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore monthly salary impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Monthly_Inhand_Salary", 
#             color="Credit_Score",
#             title="Credit Scores Based on Monthly Inhand Salary", 
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore number of bank accounts impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Num_Bank_Accounts", 
#             color="Credit_Score",
#             title="Credit Scores Based on Number of Bank Accounts", 
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore number of credit cards impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Num_Credit_Card", 
#             color="Credit_Score",
#             title="Credit Scores Based on Number of Credit cards", 
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore interest rate impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Interest_Rate", 
#             color="Credit_Score",
#             title="Credit Scores Based on the Average Interest rates", 
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore number of loans impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Num_of_Loan", 
#             color="Credit_Score", 
#             title="Credit Scores Based on Number of Loans Taken by the Person",
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore delay from due date impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Delay_from_due_date", 
#             color="Credit_Score",
#             title="Credit Scores Based on Average Number of Days Delayed for Credit card Payments", 
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore number of delayed payments impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Num_of_Delayed_Payment", 
#             color="Credit_Score", 
#             title="Credit Scores Based on Number of Delayed Payments",
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore outstanding debt impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Outstanding_Debt", 
#             color="Credit_Score", 
#             title="Credit Scores Based on Outstanding Debt",
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore credit utilization ratio impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Credit_Utilization_Ratio", 
#             color="Credit_Score",
#             title="Credit Scores Based on Credit Utilization Ratio", 
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore credit history age impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Credit_History_Age", 
#             color="Credit_Score", 
#             title="Credit Scores Based on Credit History Age",
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore EMI per month impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Total_EMI_per_month", 
#             color="Credit_Score", 
#             title="Credit Scores Based on Total Number of EMIs per Month",
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore amount invested monthly impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Amount_invested_monthly", 
#             color="Credit_Score", 
#             title="Credit Scores Based on Amount Invested Monthly",
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# Explore monthly balance impact
#fig = px.box(data, 
#             x="Credit_Score", 
#             y="Monthly_Balance", 
#             color="Credit_Score", 
#             title="Credit Scores Based on Monthly Balance Left",
#             color_discrete_map={'Poor':'red',
#                                'Standard':'yellow',
#                                'Good':'green'})
#fig.update_traces(quartilemethod="exclusive")
#fig.show()

# =============================================================================
# STEP 4: Data Preprocessing
# =============================================================================

# Convert Credit_Mix to numerical values
data["Credit_Mix"] = data["Credit_Mix"].map({"Standard": 1, 
                                            "Good": 2, 
                                            "Bad": 0})

# =============================================================================
# STEP 5: Model Training Setup
# =============================================================================

# Split data into features and target
x = np.array(data[["Annual_Income", "Monthly_Inhand_Salary", 
                   "Num_Bank_Accounts", "Num_Credit_Card", 
                   "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment", 
                   "Outstanding_Debt", 
                   "Credit_History_Age", "Monthly_Balance"]])
y = np.array(data["Credit_Score"])

# Split into training and test sets
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                               test_size=0.33, 
                                               random_state=42)

# =============================================================================
# STEP 6: Model Training
# =============================================================================

# Train Random Forest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

# =============================================================================
# STEP 7: Model Implementation/Prediction
# =============================================================================

# Function to make predictions
def predict_credit_score():
    print("Credit Score Prediction : ")
    a = float(input("Annual Income: "))
    b = float(input("Monthly Inhand Salary: "))
    c = float(input("Number of Bank Accounts: "))
    d = float(input("Number of Credit cards: "))
    e = float(input("Interest rate: "))
    f = float(input("Number of Loans: "))
    g = float(input("Average number of days delayed by the person: "))
    h = float(input("Number of delayed payments: "))
    j = float(input("Outstanding Debt: "))
    k = float(input("Credit History Age: "))
    l = float(input("Monthly Balance: "))
    
    features = np.array([[a, b, c, d, e, f, g, h, j, k, l]])
    prediction = model.predict(features)
    print("Predicted Credit Score = ", prediction)
    return prediction

# Make a prediction
predict_credit_score()

# Optional: Add model evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
predictions = model.predict(xtest)
print("\nModel Performance:")
print(classification_report(ytest, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(ytest, predictions))
print(f"\nOverall Accuracy: {accuracy_score(ytest, predictions):.4f} ({accuracy_score(ytest, predictions)*100:.2f}%)")