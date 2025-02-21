#!/usr/bin/env python3
"""
Credit Score Analysis Dashboard
------------------------------
An interactive dashboard for analyzing credit score relationships with various financial metrics.
This tool provides visual insights into how different financial variables correlate with credit scores.

Author: [Your Name]
Date: February 20, 2025
"""

import pandas as pd
import seaborn as sns
import panel as pn
import matplotlib.pyplot as plt
from typing import Dict, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CreditScoreAnalyzer:
    """A class to analyze and visualize credit score relationships with financial metrics."""
    
    def __init__(self, data_path: str):
        """
        Initialize the Credit Score Analyzer.
        
        Args:
            data_path (str): Path to the credit score dataset CSV file
        """
        self.data_path = data_path
        self.data = None
        self.y_vars = {
            "Annual_Income": "Annual Income",
            "Monthly_Inhand_Salary": "Monthly Inhand Salary",
            "Num_Bank_Accounts": "Number of Bank Accounts",
            "Num_Credit_Card": "Number of Credit Cards",
            "Interest_Rate": "Interest Rate",
            "Num_of_Loan": "Number of Loans",
            "Delay_from_due_date": "Delay from Due Date",
            "Num_of_Delayed_Payment": "Number of Delayed Payments",
            "Outstanding_Debt": "Outstanding Debt",
            "Credit_Utilization_Ratio": "Credit Utilization Ratio",
            "Credit_History_Age": "Credit History Age",
            "Total_EMI_per_month": "Total EMI per Month",
            "Amount_invested_monthly": "Amount Invested Monthly",
            "Monthly_Balance": "Monthly Balance"
        }
        self.colors = {
            'Poor': '#6a0dad',      # Deep Purple
            'Standard': '#1e90ff',   # Cosmic Blue
            'Good': '#00fa9a'        # Mint Green
        }
        self._load_data()
        self._setup_style()
    
    def _load_data(self) -> None:
        """Load and validate the credit score dataset."""
        try:
            if not Path(self.data_path).exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            self.data = pd.read_csv(self.data_path)
            required_columns = ['Credit_Score'] + list(self.y_vars.keys())
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
                
            logger.info(f"Successfully loaded dataset with {len(self.data)} records")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _setup_style(self) -> None:
        """Configure the visualization style settings."""
        plt.style.use('seaborn-v0_8-darkgrid')  # Using a valid matplotlib style
        sns.set_theme(style="darkgrid")  # Set seaborn theme
        sns.set_palette(list(self.colors.values()))
        plt.rcParams.update({
            'figure.figsize': (12, 6),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def create_boxplot(self, variable: str) -> plt.Figure:
        """
        Create a boxplot for the specified variable against credit scores.
        
        Args:
            variable (str): The financial metric to plot
            
        Returns:
            plt.Figure: The generated matplotlib figure
        """
        if variable not in self.y_vars:
            raise ValueError(f"Invalid variable: {variable}")
        
        plt.figure()
        ax = sns.boxplot(
            data=self.data,
            x='Credit_Score',
            y=variable,
            palette=self.colors,
            showfliers=True
        )
        
        plt.title(f"{self.y_vars[variable]} Distribution by Credit Score")
        plt.xlabel("Credit Score Category")
        plt.ylabel(self.y_vars[variable])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def create_dashboard(self) -> pn.Column:
        """
        Create and return the interactive dashboard.
        
        Returns:
            pn.Column: The panel dashboard object
        """
        pn.extension()
        
        # Create variable selector widget
        variable_selector = pn.widgets.Select(
            name='Select Financial Metric',
            options=list(self.y_vars.keys()),
            value=list(self.y_vars.keys())[0]
        )
        
        # Create plot update callback
        @pn.depends(variable_selector)
        def update_plot(variable: str) -> pn.pane.Matplotlib:
            return pn.pane.Matplotlib(self.create_boxplot(variable))
        
        # Compose dashboard layout
        dashboard = pn.Column(
            pn.pane.Markdown("# Credit Score Analysis Dashboard", styles={'text-align': 'center'}),
            pn.pane.Markdown("Analyze the relationship between credit scores and various financial metrics."),
            pn.Row(
                pn.Column(variable_selector, width=300),
                update_plot
            )
        )
        
        return dashboard

def main():
    """Main function to run the Credit Score Analysis Dashboard."""
    try:
        # Initialize analyzer
        analyzer = CreditScoreAnalyzer("./Credit-Score-Data/Credit Score Data/train.csv")
        
        # Create and display dashboard
        dashboard = analyzer.create_dashboard()
        dashboard.show()
        
        logger.info("Dashboard successfully launched")
        
    except Exception as e:
        logger.error(f"Failed to launch dashboard: {str(e)}")
        raise

if __name__ == "__main__":
    main()