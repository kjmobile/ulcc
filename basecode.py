#num1: Import and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Manuscript-quality figure settings
def setup_manuscript_style():
    """Setup matplotlib for publication-quality figures"""
    
    plt.style.use('default')  # Start with clean slate
    
    # Academic paper style settings
    manuscript_settings = {
        # Font and text
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        
        # Figure layout
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Axes and grid
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Lines and markers
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8
    }
    
    plt.rcParams.update(manuscript_settings)
    
    # Academic color palette (consistent across all figures)
    academic_colors = {
        'ULCC': '#d62728',    # Red
        'LCC': '#ff7f0e',     # Orange  
        'Hybrid': '#1f77b4',  # Blue
        'Legacy': '#2ca02c'   # Green
    }
    
    return academic_colors

# Set up manuscript style by default
CARRIER_COLORS = setup_manuscript_style()

#num2: Load airline classification
def load_airline_classification():
    """Load airline classification from CSV file"""
    try:
        classification_df = pd.read_csv('airline_classification_4way.csv')
        
        classification_map = {}
        for _, row in classification_df.iterrows():
            airline_code = row['Airline']
            carrier_type = row['Carrier_Type']
            classification_map[airline_code] = carrier_type
        
        print(f"Loaded classification for {len(classification_map)} airlines")
        print(f"Carrier types: {set(classification_map.values())}")
        
        return classification_map
    except Exception as e:
        print(f"Error loading classification file: {e}")
        return {
            'AA': 'Legacy', 'DL': 'Legacy', 'UA': 'Legacy', 'US': 'Legacy',
            'NK': 'ULCC', 'F9': 'ULCC', 'G4': 'ULCC',
            'WN': 'LCC', 'FL': 'LCC', 'SY': 'LCC',
            'AS': 'Hybrid', 'B6': 'Hybrid', 'HA': 'Hybrid', 'VX': 'Hybrid'
        }

#num3: Load all datasets
def load_all_data():
    """Load all necessary datasets"""
    try:
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
        
        print(f"Loaded OD data: {len(od_years)} years")
        print(f"Loaded T100 data: {len(t100_years)} years")
        print(f"Shock data shape: {shock_data.shape}")
        
        return od_years, t100_years, shock_data, classification_map
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}, {}, None, {}

#num4: Prepare combined dataset for all analyses
def prepare_combined_data(od_years, t100_years, classification_map):
    """Prepare and combine all data for analysis"""
    
    print("Preparing combined dataset...")
    
    # Combine all OD data
    all_od_data = []
    for year, df in od_years.items():
        df_copy = df.copy()
        df_copy['Year'] = year
        all_od_data.append(df_copy)
    
    combined_od = pd.concat(all_od_data, ignore_index=True)
    combined_od['Business_Model'] = combined_od['Mkt'].map(classification_map)
    combined_od = combined_od.dropna(subset=['Business_Model'])
    
    # Filter to main carrier types
    valid_types = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    combined_od = combined_od[combined_od['Business_Model'].isin(valid_types)]
    
    print(f"Combined OD data: {len(combined_od):,} rows")
    print(f"Carrier distribution:")
    print(combined_od['Business_Model'].value_counts())
    
    # Combine all T100 data
    all_t100_data = []
    for year, df in t100_years.items():
        df_copy = df.copy()
        df_copy['Year'] = year
        df_copy.rename(columns={'Orig': 'Org', 'Dest': 'Dst'}, inplace=True)
        all_t100_data.append(df_copy)
    
    combined_t100 = pd.concat(all_t100_data, ignore_index=True)
    
    print(f"Combined T100 data: {len(combined_t100):,} rows")
    
    return combined_od, combined_t100

#num5: Create route-level datasets
def create_route_datasets(combined_od, combined_t100):
    """Create route-level analysis datasets"""
    
    print("Creating route-level datasets...")
    
    # Route presence by year and model for market behavior analysis
    combined_od['Route_Carrier'] = (combined_od['Org'] + '_' + 
                                   combined_od['Dst'] + '_' + 
                                   combined_od['Mkt'])
    
    route_year_model = combined_od.groupby(['Route_Carrier', 'Year', 'Business_Model']).size().reset_index(name='count')
    
    route_presence = route_year_model.pivot_table(
        index=['Route_Carrier', 'Business_Model'], 
        columns='Year', 
        values='count', 
        fill_value=0
    )
    route_presence = (route_presence > 0).astype(int)
    
    # Route-level aggregations for competitive analysis
    route_level_od = combined_od.groupby(['Org', 'Dst', 'Year', 'Business_Model']).agg({
        'Passengers': 'sum'
    }).reset_index()
    
    route_level_t100 = combined_t100.groupby(['Org', 'Dst', 'Year']).agg({
        'Load Factor': 'mean',
        'Onboards': 'sum',
        'ASMs': 'sum',
        'RPMs': 'sum'
    }).reset_index()
    
    print("Route datasets created")
    
    return route_presence, route_level_od, route_level_t100

#num6: Main data preparation function
def prepare_base_data():
    """Main function to prepare all base data"""
    
    print("ULCC STRATEGIC VOLATILITY ANALYSIS - DATA PREPARATION")
    print("=" * 60)
    
    # Load all data
    od_years, t100_years, shock_data, classification_map = load_all_data()
    
    if not od_years or not classification_map:
        print("Error: Cannot proceed without data")
        return None
    
    # Prepare combined datasets
    combined_od, combined_t100 = prepare_combined_data(od_years, t100_years, classification_map)
    
    # Create route-level datasets
    route_presence, route_level_od, route_level_t100 = create_route_datasets(combined_od, combined_t100)
    
    base_data = {
        'od_years': od_years,
        't100_years': t100_years,
        'shock_data': shock_data,
        'classification_map': classification_map,
        'combined_od': combined_od,
        'combined_t100': combined_t100,
        'route_presence': route_presence,
        'route_level_od': route_level_od,
        'route_level_t100': route_level_t100
    }
    
    print("=" * 60)
    print("BASE DATA PREPARATION COMPLETE!")
    print("=" * 60)
    
    return base_data

if __name__ == "__main__":
    base_data = prepare_base_data()