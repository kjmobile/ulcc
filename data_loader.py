# data_loader.py
# #num7: Data Loading and Preprocessing Module

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, data_path="data"):
        self.data_path = Path(data_path)
        self.airline_classification = {
            'ULCC': ['NK', 'F9', 'G4'],
            'LCC': ['WN', 'FL', 'SY'], 
            'Hybrid': ['AS', 'B6', 'HA', 'VX'],
            'Legacy': ['AA', 'DL', 'UA', 'US']
        }
        
    def load_od_data(self, years=None):
        """Load O&D data for specified years"""
        if years is None:
            years = range(2014, 2025)
            
        od_data = []
        for year in years:
            file_path = self.data_path / 'od' / f'od_{year}.parquet'
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['Year'] = year
                od_data.append(df)
                print(f"Loaded OD {year}: {len(df):,} rows")
            else:
                print(f"Warning: OD data for {year} not found")
                
        if od_data:
            combined_data = pd.concat(od_data, ignore_index=True)
            return self._add_carrier_classification(combined_data, 'Mkt')
        else:
            return pd.DataFrame()
            
    def load_t100_data(self, years=None):
        """Load T-100 data for specified years"""
        if years is None:
            years = range(2014, 2025)
            
        t100_data = []
        for year in years:
            file_path = self.data_path / 't_100' / f't_100_{year}.parquet'
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['Year'] = year
                t100_data.append(df)
                print(f"Loaded T100 {year}: {len(df):,} rows")
            else:
                print(f"Warning: T100 data for {year} not found")
                
        if t100_data:
            combined_data = pd.concat(t100_data, ignore_index=True)
            return self._add_carrier_classification(combined_data, 'Mkt Al')
        else:
            return pd.DataFrame()
            
    def load_shock_data(self):
        """Load economic shock data"""
        file_path = self.data_path / 'analysis' / 'shock_2014_2024.parquet'
        if file_path.exists():
            shock_data = pd.read_parquet(file_path)
            print(f"Loaded shock data: {len(shock_data)} rows")
            return shock_data
        else:
            print("Warning: Shock data not found")
            return pd.DataFrame()
            
    def _add_carrier_classification(self, data, carrier_column):
        """Add carrier type classification to data"""
        carrier_to_type = {}
        for carrier_type, carriers in self.airline_classification.items():
            for carrier in carriers:
                carrier_to_type[carrier] = carrier_type
                
        data['Carrier_Type'] = data[carrier_column].map(carrier_to_type)
        
        # Log classification results
        classified = data.dropna(subset=['Carrier_Type'])
        print(f"Classified carriers: {len(classified):,} / {len(data):,} rows ({len(classified)/len(data)*100:.1f}%)")
        
        return data
        
    def get_data_summary(self):
        """Get summary of available data"""
        summary = {
            'od_files': [],
            't100_files': [],
            'analysis_files': []
        }
        
        # Check O&D files
        od_dir = self.data_path / 'od'
        if od_dir.exists():
            for year in range(2014, 2025):
                file_path = od_dir / f'od_{year}.parquet'
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    summary['od_files'].append(f'od_{year}.parquet ({size_mb:.1f} MB)')
                    
        # Check T-100 files  
        t100_dir = self.data_path / 't_100'
        if t100_dir.exists():
            for year in range(2014, 2025):
                file_path = t100_dir / f't_100_{year}.parquet'
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    summary['t100_files'].append(f't_100_{year}.parquet ({size_mb:.1f} MB)')
                    
        # Check analysis files
        analysis_dir = self.data_path / 'analysis'
        if analysis_dir.exists():
            for file_path in analysis_dir.glob('*.parquet'):
                size_mb = file_path.stat().st_size / (1024 * 1024)
                summary['analysis_files'].append(f'{file_path.name} ({size_mb:.1f} MB)')
                
        return summary
        
    def validate_data_integrity(self, data, data_type='od'):
        """Validate data integrity"""
        issues = []
        
        if len(data) == 0:
            issues.append("Dataset is empty")
            return issues
            
        # Check for required columns
        if data_type == 'od':
            required_cols = ['Org', 'Dst', 'Year', 'Month', 'Passengers']
        elif data_type == 't100':
            required_cols = ['Orig', 'Dest', 'Year', 'Month', 'Onboards']
        else:
            required_cols = ['Year', 'Month']
            
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            
        # Check for data quality issues
        if data_type in ['od', 't100']:
            passenger_col = 'Passengers' if data_type == 'od' else 'Onboards'
            if passenger_col in data.columns:
                negative_pax = (data[passenger_col] < 0).sum()
                if negative_pax > 0:
                    issues.append(f"Negative passenger values: {negative_pax}")
                    
                zero_pax = (data[passenger_col] == 0).sum()
                if zero_pax > len(data) * 0.1:  # More than 10% zeros
                    issues.append(f"High proportion of zero passenger values: {zero_pax}")
                    
        # Check date ranges
        if 'Year' in data.columns:
            year_range = f"{data['Year'].min()}-{data['Year'].max()}"
            if data['Year'].min() < 2014 or data['Year'].max() > 2024:
                issues.append(f"Unexpected year range: {year_range}")
                
        # Check for duplicates
        if data_type == 'od':
            dup_cols = ['Org', 'Dst', 'Year', 'Month', 'Mkt']
        elif data_type == 't100':
            dup_cols = ['Orig', 'Dest', 'Year', 'Month', 'Mkt Al']
        else:
            dup_cols = ['Year', 'Month']
            
        if all(col in data.columns for col in dup_cols):
            duplicates = data.duplicated(subset=dup_cols).sum()
            if duplicates > 0:
                issues.append(f"Duplicate records: {duplicates}")
                
        return issues
