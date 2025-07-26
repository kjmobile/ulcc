import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def setup_manuscript_style():
    """Setup matplotlib for publication-quality figures"""
    plt.style.use('default')
    
    manuscript_settings = {
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3
    }
    
    plt.rcParams.update(manuscript_settings)
    
    return {
        'ULCC': '#d62728',
        'LCC': '#ff7f0e',
        'Hybrid': '#1f77b4',
        'Legacy': '#2ca02c'
    }

CARRIER_COLORS = setup_manuscript_style()

def load_airline_classification():
    """Load airline classification from CSV file"""
    try:
        classification_df = pd.read_csv('airline_classification_4way.csv')
        classification_map = dict(zip(
            classification_df['Airline'],
            classification_df['Carrier_Type']
        ))
        print(f"Loaded {len(classification_map)} airlines")
        return classification_map
    except Exception as e:
        print(f"Error loading classification: {e}")
        return {
            'AA': 'Legacy', 'DL': 'Legacy', 'UA': 'Legacy', 'US': 'Legacy',
            'NK': 'ULCC', 'F9': 'ULCC', 'G4': 'ULCC',
            'WN': 'LCC', 'FL': 'LCC', 'SY': 'LCC',
            'AS': 'Hybrid', 'B6': 'Hybrid', 'HA': 'Hybrid', 'VX': 'Hybrid'
        }

def load_all_data():
    """Load all datasets"""
    classification_map = load_airline_classification()
    
    od_years = {}
    for year in range(2014, 2025):
        try:
            od_years[year] = pd.read_parquet(f'data/od/od_{year}.parquet')
        except:
            continue
    
    t100_years = {}
    for year in range(2014, 2025):
        try:
            t100_years[year] = pd.read_parquet(f'data/t_100/t_100_{year}.parquet')
        except:
            continue
    
    shock_data = pd.read_parquet('data/analysis/shock_2014_2024.parquet')
    
    print(f"Loaded OD: {len(od_years)} years, T100: {len(t100_years)} years")
    return od_years, t100_years, shock_data, classification_map

def prepare_combined_data(od_years, t100_years, classification_map):
    """Combine data with critical fixes applied"""
    
    # Combine OD data
    all_od_data = []
    for year, df in od_years.items():
        df_copy = df.copy()
        df_copy['Year'] = year
        all_od_data.append(df_copy)
    
    combined_od = pd.concat(all_od_data, ignore_index=True)
    combined_od['Business_Model'] = combined_od['Opr'].map(classification_map)
    combined_od = combined_od.dropna(subset=['Business_Model'])
    
    valid_types = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    combined_od = combined_od[combined_od['Business_Model'].isin(valid_types)]
    
    # Combine T100 data
    all_t100_data = []
    for year, df in t100_years.items():
        df_copy = df.copy()
        df_copy['Year'] = year
        df_copy.rename(columns={'Orig': 'Org', 'Dest': 'Dst'}, inplace=True)
        all_t100_data.append(df_copy)
    
    combined_t100 = pd.concat(all_t100_data, ignore_index=True)
    combined_t100['Business_Model'] = combined_t100['Mkt Al'].map(classification_map)
    combined_t100 = combined_t100.dropna(subset=['Business_Model'])
    combined_t100 = combined_t100[combined_t100['Business_Model'].isin(valid_types)]
    
    print(f"Combined OD: {len(combined_od):,} rows, T100: {len(combined_t100):,} rows")
    return combined_od, combined_t100

def create_route_datasets(combined_od, combined_t100):
    """Create route-level datasets"""
    combined_od['Route'] = combined_od['Org'] + '-' + combined_od['Dst']
    
    route_carrier_year = combined_od.groupby(['Route', 'Opr', 'Year', 'Business_Model']).agg({
        'Passengers': 'sum'
    }).reset_index()
    
    route_presence = route_carrier_year.pivot_table(
        index=['Route', 'Opr', 'Business_Model'], 
        columns='Year', 
        values='Passengers', 
        fill_value=0
    )
    route_presence = (route_presence > 0).astype(int)
    
    route_level_od = combined_od.groupby(['Org', 'Dst', 'Year', 'Business_Model']).agg({
        'Passengers': 'sum'
    }).reset_index()
    
    route_level_t100 = combined_t100.groupby(['Org', 'Dst', 'Year']).agg({
        'Load Factor': 'mean',
        'Onboards': 'sum',
        'ASMs': 'sum',
        'RPMs': 'sum'
    }).reset_index()
    
    return route_presence, route_level_od, route_level_t100

def prepare_base_data(include_route_presence=True):
    """Main data preparation function with critical fixes"""
    
    od_years, t100_years, shock_data, classification_map = load_all_data()
    
    if not od_years or not classification_map:
        print("Error: Cannot proceed without data")
        return None
    
    combined_od, combined_t100 = prepare_combined_data(od_years, t100_years, classification_map)
    
    if include_route_presence:
        route_presence, route_level_od, route_level_t100 = create_route_datasets(combined_od, combined_t100)
    else:
        route_presence = route_level_od = route_level_t100 = None
    
    base_data = {
        'od_years': od_years,
        't100_years': t100_years,
        'shock_data': shock_data,
        'classification_map': classification_map,
        'combined_od': combined_od,
        'combined_t100': combined_t100,
        'route_presence': route_presence,
        'route_level_od': route_level_od,
        'route_level_t100': route_level_t100,
        'colors': CARRIER_COLORS
    }
    
    print(f"Base data ready: {len(combined_od):,} OD records, {len(combined_t100):,} T100 records")
    return base_data

if __name__ == "__main__":
    base_data = prepare_base_data()