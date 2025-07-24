# covid_recovery_analysis.py
# #num5: COVID Recovery Analysis for H4 Testing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class CovidRecoveryAnalysis:
    def __init__(self, data_path, airline_classification, shock_data):
        self.data_path = Path(data_path)
        self.airline_classification = airline_classification
        self.shock_data = shock_data
        self.results = {}
        
    def load_recovery_data(self):
        """Load data for COVID recovery analysis"""
        print("Loading COVID recovery data...")
        
        # Load T-100 data for traffic analysis
        self.recovery_data = []
        years = range(2018, 2025)  # Include pre-COVID baseline
        
        for year in years:
            file_path = self.data_path / 't_100' / f't_100_{year}.parquet'
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['Year'] = year
                self.recovery_data.append(df)
                print(f"Loaded T100 {year}: {len(df):,} rows")
                
        self.recovery_data = pd.concat(self.recovery_data, ignore_index=True)
        
        # Also load O&D data for market share analysis
        self.od_recovery_data = []
        for year in years:
            file_path = self.data_path / 'od' / f'od_{year}.parquet'
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['Year'] = year
                self.od_recovery_data.append(df)
                
        if self.od_recovery_data:
            self.od_recovery_data = pd.concat(self.od_recovery_data, ignore_index=True)
            
    def classify_carriers(self):
        """Add carrier type classification"""
        carrier_to_type = {}
        for carrier_type, carriers in self.airline_classification.items():
            for carrier in carriers:
                carrier_to_type[carrier] = carrier_type
                
        self.recovery_data['Carrier_Type'] = self.recovery_data['Mkt Al'].map(carrier_to_type)
        self.recovery_data = self.recovery_data.dropna(subset=['Carrier_Type'])
        
        if hasattr(self, 'od_recovery_data'):
            self.od_recovery_data['Carrier_Type'] = self.od_recovery_data['Mkt'].map(carrier_to_type)
            self.od_recovery_data = self.od_recovery_data.dropna(subset=['Carrier_Type'])
        
    def calculate_monthly_traffic_by_carrier_type(self):
        """Calculate monthly traffic aggregated by carrier type"""
        print("Calculating monthly traffic by carrier type...")
        
        # Aggregate T-100 data by carrier type and month
        monthly_data = self.recovery_data.groupby(['Year', 'Month', 'Carrier_Type']).agg({
            'Onboards': 'sum',
            'ASMs': 'sum',
            'RPMs': 'sum'
        }).reset_index()
        
        # Create date column
        monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
        
        # Calculate load factor
        monthly_data['Load_Factor'] = (monthly_data['RPMs'] / monthly_data['ASMs'] * 100).fillna(0)
        
        self.monthly_traffic = monthly_data
        return monthly_data
        
    def establish_baseline_performance(self):
        """Establish 2019 baseline for recovery analysis"""
        print("Establishing 2019 baseline...")
        
        baseline_2019 = self.monthly_traffic[self.monthly_traffic['Year'] == 2019].groupby(['Month', 'Carrier_Type']).agg({
            'Onboards': 'mean',
            'ASMs': 'mean',
            'RPMs': 'mean',
            'Load_Factor': 'mean'
        }).reset_index()
        
        self.baseline_2019 = baseline_2019
        return baseline_2019
        
    def calculate_recovery_metrics(self):
        """Calculate recovery metrics relative to 2019 baseline"""
        print("Calculating recovery metrics...")
        
        recovery_metrics = []
        
        for carrier_type in self.airline_classification.keys():
            carrier_data = self.monthly_traffic[self.monthly_traffic['Carrier_Type'] == carrier_type]
            baseline_data = self.baseline_2019[self.baseline_2019['Carrier_Type'] == carrier_type]
            
            if len(baseline_data) == 0:
                continue
                
            # Calculate monthly recovery percentages
            for _, row in carrier_data.iterrows():
                if row['Year'] < 2020:  # Skip pre-COVID data
                    continue
                    
                # Find corresponding baseline month
                baseline_month = baseline_data[baseline_data['Month'] == row['Month']]
                
                if len(baseline_month) == 0:
                    continue
                    
                baseline_pax = baseline_month['Onboards'].iloc[0]
                baseline_asm = baseline_month['ASMs'].iloc[0]
                baseline_rpm = baseline_month['RPMs'].iloc[0]
                
                if baseline_pax > 0:
                    recovery_pct = (row['Onboards'] / baseline_pax) * 100
                else:
                    recovery_pct = 0
                    
                asm_recovery_pct = (row['ASMs'] / baseline_asm) * 100 if baseline_asm > 0 else 0
                rpm_recovery_pct = (row['RPMs'] / baseline_rpm) * 100 if baseline_rpm > 0 else 0
                
                recovery_metrics.append({
                    'Carrier_Type': carrier_type,