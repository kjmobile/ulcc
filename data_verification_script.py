# data_verification.py
# Verify actual data structure and column names to fix basecode.py

import pandas as pd
import numpy as np
from pathlib import Path
import os

def check_csv_classification():
    """Check the actual structure of airline classification CSV"""
    print("="*60)
    print("CHECKING AIRLINE CLASSIFICATION CSV")
    print("="*60)
    
    csv_files = [
        'airline_classification_4way.csv',
        'airline_classification.csv',
        'classification.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"\n✅ Found: {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print("\nFirst 5 rows:")
                print(df.head())
                print("\nUnique values in each column:")
                for col in df.columns:
                    unique_vals = df[col].unique()
                    print(f"  {col}: {len(unique_vals)} unique values")
                    if len(unique_vals) <= 20:
                        print(f"    Values: {list(unique_vals)}")
                    else:
                        print(f"    Sample: {list(unique_vals[:10])}")
                return df, csv_file
            except Exception as e:
                print(f"❌ Error reading {csv_file}: {e}")
        else:
            print(f"❌ Not found: {csv_file}")
    
    print("\n⚠️  No classification CSV found!")
    return None, None

def check_od_data_structure():
    """Check actual OD data structure"""
    print("\n" + "="*60)
    print("CHECKING OD DATA STRUCTURE")
    print("="*60)
    
    # Try to find any OD file
    od_files = []
    if os.path.exists('data/od/'):
        od_files = [f for f in os.listdir('data/od/') if f.endswith('.parquet')]
        od_files = sorted(od_files)
    
    if not od_files:
        print("❌ No OD files found in data/od/")
        return None
    
    # Check the most recent file
    latest_file = od_files[-1]
    print(f"\n✅ Checking: data/od/{latest_file}")
    
    try:
        df = pd.read_parquet(f'data/od/{latest_file}')
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Check carrier columns specifically
        carrier_cols = [col for col in df.columns if col.lower() in ['opr', 'mkt', 'carrier', 'airline']]
        print(f"\nCarrier-related columns: {carrier_cols}")
        
        for col in carrier_cols:
            unique_carriers = df[col].unique()
            print(f"\n{col} column:")
            print(f"  Unique values: {len(unique_carriers)}")
            print(f"  Sample carriers: {list(unique_carriers[:20])}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading OD data: {e}")
        return None

def check_t100_data_structure():
    """Check actual T-100 data structure"""
    print("\n" + "="*60)
    print("CHECKING T-100 DATA STRUCTURE") 
    print("="*60)
    
    # Try to find any T-100 file
    t100_files = []
    if os.path.exists('data/t_100/'):
        t100_files = [f for f in os.listdir('data/t_100/') if f.endswith('.parquet')]
        t100_files = sorted(t100_files)
    
    if not t100_files:
        print("❌ No T-100 files found in data/t_100/")
        return None
    
    # Check the most recent file
    latest_file = t100_files[-1]
    print(f"\n✅ Checking: data/t_100/{latest_file}")
    
    try:
        df = pd.read_parquet(f'data/t_100/{latest_file}')
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Check carrier columns specifically
        carrier_cols = [col for col in df.columns if any(x in col.lower() for x in ['mkt', 'carrier', 'airline', 'orig', 'dest'])]
        print(f"\nRelevant columns: {carrier_cols}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading T-100 data: {e}")
        return None

def test_current_basecode():
    """Test if current basecode.py works and what it produces"""
    print("\n" + "="*60)
    print("TESTING CURRENT BASECODE.PY")
    print("="*60)
    
    try:
        # Try to import and run current basecode
        from basecode import prepare_base_data, load_airline_classification
        
        print("✅ basecode.py imports successfully")
        
        # Test classification loading
        print("\nTesting load_airline_classification()...")
        classification_map = load_airline_classification()
        if classification_map:
            print(f"✅ Classification loaded: {len(classification_map)} airlines")
            print("Sample mappings:")
            for i, (airline, bm) in enumerate(list(classification_map.items())[:10]):
                print(f"  {airline} -> {bm}")
            
            # Check business model distribution
            bm_counts = {}
            for bm in classification_map.values():
                bm_counts[bm] = bm_counts.get(bm, 0) + 1
            print(f"Business model distribution: {bm_counts}")
        else:
            print("❌ Classification loading failed")
        
        # Test base data preparation (minimal)
        print("\nTesting prepare_base_data()...")
        try:
            base_data = prepare_base_data(include_route_presence=False)
            if base_data and base_data.get('combined_od') is not None:
                print("✅ Base data preparation successful")
                
                combined_od = base_data['combined_od']
                print(f"Combined OD shape: {combined_od.shape}")
                
                if 'Business_Model' in combined_od.columns:
                    print("Business model distribution in combined_od:")
                    print(combined_od['Business_Model'].value_counts())
                else:
                    print("❌ No Business_Model column in combined_od")
                
                # Check what carrier columns exist
                carrier_cols = [col for col in combined_od.columns if col.lower() in ['opr', 'mkt', 'carrier', 'airline']]
                print(f"Available carrier columns: {carrier_cols}")
                
            else:
                print("❌ Base data preparation failed or returned None")
        except Exception as e:
            print(f"❌ Error in prepare_base_data(): {e}")
            
    except ImportError as e:
        print(f"❌ Cannot import basecode.py: {e}")
    except Exception as e:
        print(f"❌ Error testing basecode.py: {e}")

def recommend_fixes(csv_df, csv_filename, od_df, t100_df):
    """Based on actual data, recommend fixes for basecode.py"""
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR BASECODE.PY FIXES")
    print("="*60)
    
    fixes = []
    
    # CSV file recommendations
    if csv_df is not None:
        print(f"\n1. AIRLINE CLASSIFICATION CSV ({csv_filename}):")
        cols = list(csv_df.columns)
        print(f"   Actual columns: {cols}")
        
        # Guess the right columns
        carrier_col = None
        bm_col = None
        
        for col in cols:
            if col.lower() in ['carrier', 'airline', 'code']:
                carrier_col = col
            elif 'business' in col.lower() or 'model' in col.lower() or 'type' in col.lower():
                bm_col = col
        
        if carrier_col and bm_col:
            print(f"   ✅ Use: carrier_col='{carrier_col}', bm_col='{bm_col}'")
            fixes.append(f"Fix load_airline_classification(): use '{carrier_col}' and '{bm_col}'")
        else:
            print("   ❌ Cannot identify correct columns automatically")
    
    # OD data recommendations  
    if od_df is not None:
        print(f"\n2. OD DATA MAPPING:")
        cols = list(od_df.columns)
        print(f"   Available columns: {cols}")
        
        # Check for operating vs marketing carrier
        if 'Opr' in cols and 'Mkt' in cols:
            print("   ✅ Both 'Opr' and 'Mkt' available")
            print("   Recommendation: Use 'Opr' for business model classification")
            fixes.append("Change combined_od mapping from 'Mkt' to 'Opr'")
        elif 'Opr' in cols:
            print("   ✅ Use 'Opr' column")
            fixes.append("Use 'Opr' for business model classification")
        elif 'Mkt' in cols:
            print("   ⚠️  Only 'Mkt' available - use with caution")
        else:
            print("   ❌ No clear carrier column found")
    
    # T-100 data recommendations
    if t100_df is not None:
        print(f"\n3. T-100 DATA MAPPING:")
        cols = list(t100_df.columns)
        airline_cols = [col for col in cols if 'mkt' in col.lower() or 'airline' in col.lower()]
        print(f"   Airline columns: {airline_cols}")
        if airline_cols:
            recommended_col = airline_cols[0]
            print(f"   ✅ Recommended: '{recommended_col}'")
            fixes.append(f"Use '{recommended_col}' for T-100 business model classification")
    
    # Route definition recommendations
    if od_df is not None:
        print(f"\n4. ROUTE DEFINITION:")
        airport_cols = [col for col in od_df.columns if col.lower() in ['org', 'dst', 'orig', 'dest']]
        print(f"   Airport columns: {airport_cols}")
        if len(airport_cols) >= 2:
            print(f"   ✅ Use pure route: '{airport_cols[0]}-{airport_cols[1]}'")
            fixes.append(f"Use pure route definition: {airport_cols[0]}-{airport_cols[1]}")
        
    print(f"\n📋 SUMMARY OF REQUIRED FIXES:")
    for i, fix in enumerate(fixes, 1):
        print(f"   {i}. {fix}")
    
    return fixes

def run_complete_verification():
    """Run complete data verification"""
    print("ULCC DATA STRUCTURE VERIFICATION")
    print("="*60)
    print("This script will check actual data structure and recommend fixes")
    print("="*60)
    
    # Check all data sources
    csv_df, csv_filename = check_csv_classification()
    od_df = check_od_data_structure() 
    t100_df = check_t100_data_structure()
    
    # Test current basecode
    test_current_basecode()
    
    # Provide recommendations
    fixes = recommend_fixes(csv_df, csv_filename, od_df, t100_df)
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Review the recommendations above")
    print("2. Apply fixes to basecode.py")  
    print("3. Re-test with updated basecode.py")
    print("4. Run H1 analysis with corrected data structure")
    
    return {
        'csv_data': csv_df,
        'csv_filename': csv_filename,
        'od_data': od_df,
        't100_data': t100_df,
        'recommended_fixes': fixes
    }

if __name__ == "__main__":
    verification_results = run_complete_verification()