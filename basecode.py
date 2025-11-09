# #018 - basecode.py Simplified (No try/except, Core functions only)

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

def load_data():
    """Load all required datasets"""
    
    # Load classification
    classification_df = pd.read_csv('airline_classification_4way.csv')
    classification_map = dict(zip(classification_df['Airline'], classification_df['Carrier_Type']))
    
    # Load OD data by year
    od_years = {}
    for year in range(2014, 2025):
        od_years[year] = pd.read_parquet(f'data/od/od_{year}.parquet')
    
    # Load T-100 data by year  
    t100_years = {}
    for year in range(2014, 2025):
        t100_years[year] = pd.read_parquet(f'data/t_100/t_100_{year}.parquet')
    
    # Load shock data
    shock_data = pd.read_parquet('data/analysis/shock_2014_2024.parquet')
    
    print(f"Loaded: {len(classification_map)} airlines, {len(od_years)} OD years, {len(t100_years)} T100 years")
    
    return od_years, t100_years, shock_data, classification_map

def analyze_market_share_by_period(base_data):
    """Calculate market share using Marketing Carrier (Mkt) approach"""
    
    od_years = base_data['od_years']
    classification_map = base_data['classification_map']
    
    periods = {
        'Early (2014-2016)': range(2014, 2017),
        'Pre-COVID (2017-2019)': range(2017, 2020),
        'Recovery (2022-2023)': range(2022, 2024),
        'Current (2024)': [2024]
    }
    
    period_results = {}
    
    for period_name, years in periods.items():
        period_passengers = {}
        
        for year in years:
            year_data = od_years[year].copy()
            
            # Use Marketing Carrier (Mkt) for strategic analysis
            year_data = year_data[year_data['Mkt'].isin(classification_map.keys())].copy()
            year_data['Business_Model'] = year_data['Mkt'].map(classification_map)
            
            year_totals = year_data.groupby('Business_Model')['Passengers'].sum()
            
            for bm, passengers in year_totals.items():
                period_passengers[bm] = period_passengers.get(bm, 0) + passengers
        
        total_passengers = sum(period_passengers.values())
        market_shares = {bm: (passengers / total_passengers * 100) 
                        for bm, passengers in period_passengers.items()}
        period_results[period_name] = pd.Series(market_shares).round(1)
    
    return period_results

def get_carrier_info(classification_map):
    """Get carrier information for each business model"""
    
    # Count carriers by business model
    bm_counts = {}
    for airline, bm in classification_map.items():
        bm_counts[bm] = bm_counts.get(bm, 0) + 1
    
    carrier_info = {
        'Legacy': {
            'characteristics': 'Hub-and-spoke networks, full-service offerings, frequent flyer programs, multi-cabin classes',
            'carriers': 'American (AA), Delta (DL), United (UA), US Airways (US)'
        },
        'ULCC': {
            'characteristics': 'Unbundled pricing, high-density seating, minimal amenities, ancillary fee dependent',
            'carriers': 'Spirit (NK), Frontier (F9), Allegiant (G4)'
        },
        'LCC': {
            'characteristics': 'Simplified service model, some complementary offerings, moderate cost discipline',
            'carriers': 'Southwest (WN), AirTran (FL), Sun Country (SY)'
        },
        'Hybrid': {
            'characteristics': 'Combined models, selective premium offerings, network evolution',
            'carriers': 'Alaska (AS), JetBlue (B6), Hawaiian (HA), Virgin America (VX)'
        }
    }
    
    # Add Other if exists
    if 'Other' in bm_counts:
        carrier_info['Other'] = {
            'characteristics': 'Regional carriers and specialized operators not fitting primary classifications',
            'carriers': f'Various regional and specialized carriers ({bm_counts["Other"]} carriers)'
        }
    
    return carrier_info

def create_table_3_1(base_data):
    """Create Table 3.1: Airline Classification and Market Share by Period"""
    
    print("\nTable 3.1: Airline Classification and Market Share by Period")
    print("=" * 80)
    
    period_results = analyze_market_share_by_period(base_data)
    carrier_info = get_carrier_info(base_data['classification_map'])
    
    # Create table
    table_data = []
    bm_order = ['Legacy', 'ULCC', 'LCC', 'Hybrid', 'Other']
    
    for bm in bm_order:
        if bm in carrier_info:
            row = {
                'BM': bm,
                'Key_Characteristics': carrier_info[bm]['characteristics'],
                'Airlines_Code': carrier_info[bm]['carriers']
            }
            
            for period_name in period_results.keys():
                share = period_results[period_name].get(bm, 0)
                row[period_name] = f"{share:.1f}%"
            
            table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    
    # Display table
    print(f"\n{'BM':<8} {'Key Characteristics':<40} {'Airlines':<15} ", end="")
    for period in period_results.keys():
        print(f"{period:<25}", end="")
    print()
    print("-" * 140)
    
    for _, row in table_df.iterrows():
        bm = row['BM']
        characteristics = row['Key_Characteristics'][:37] + "..." if len(row['Key_Characteristics']) > 40 else row['Key_Characteristics']
        airlines = row['Airlines_Code'][:12] + "..." if len(row['Airlines_Code']) > 15 else row['Airlines_Code']
        
        print(f"{bm:<8} {characteristics:<40} {airlines:<15} ", end="")
        
        for period in period_results.keys():
            print(f"{row[period]:<25}", end="")
        print()
    
    print("-" * 140)
    print("Note: Market shares based on Marketing Carrier (Mkt) for strategic competition analysis.")
    print("Regional airlines operating under major carrier brands are attributed to their marketing partners.")
    
    # Save table
    from pathlib import Path
    Path('paper_1_outputs').mkdir(exist_ok=True)
    table_df.to_csv('paper_1_outputs/Table_3_1_Market_Share_Analysis.csv', index=False)

    print(f"\nTable saved: paper_1_outputs/Table_3_1_Market_Share_Analysis.csv")
    
    return table_df

def prepare_combined_data(od_years, t100_years, classification_map):
    """Combine data for analysis (Marketing Carrier basis)"""
    
    # Combine OD data
    all_od_data = []
    for year, df in od_years.items():
        df_copy = df.copy()
        df_copy['Year'] = year
        all_od_data.append(df_copy)
    
    combined_od = pd.concat(all_od_data, ignore_index=True)
    
    # Use Marketing Carrier (Mkt) for strategic analysis
    combined_od = combined_od[combined_od['Mkt'].isin(classification_map.keys())].copy()
    combined_od['Business_Model'] = combined_od['Mkt'].map(classification_map)
    
    # Combine T100 data
    all_t100_data = []
    for year, df in t100_years.items():
        df_copy = df.copy()
        df_copy['Year'] = year
        df_copy.rename(columns={'Orig': 'Org', 'Dest': 'Dst'}, inplace=True)
        all_t100_data.append(df_copy)
    
    combined_t100 = pd.concat(all_t100_data, ignore_index=True)
    
    # Use Marketing Carrier for T100 as well
    combined_t100 = combined_t100[combined_t100['Mkt Al'].isin(classification_map.keys())].copy()
    combined_t100['Business_Model'] = combined_t100['Mkt Al'].map(classification_map)
    
    print(f"Combined OD: {len(combined_od):,} rows, T100: {len(combined_t100):,} rows")
    return combined_od, combined_t100

def create_route_datasets(combined_od, combined_t100):
    """Create route-level datasets for analysis"""
    
    # Create route identifier
    combined_od['Route'] = combined_od['Org'] + '-' + combined_od['Dst']
    
    print("Creating route presence matrix...")
    
    # Aggregate to route-carrier-year level
    route_carrier_year = combined_od.groupby(['Route', 'Mkt', 'Year', 'Business_Model']).agg({
        'Passengers': 'sum'
    }).reset_index()
    
    # Apply traffic threshold
    route_carrier_year = route_carrier_year[route_carrier_year['Passengers'] >= 100]
    
    print(f"Route-carrier-year data: {len(route_carrier_year):,} rows")
    
    # Create pivot table for route presence
    route_presence = route_carrier_year.pivot_table(
        index=['Route', 'Mkt', 'Business_Model'], 
        columns='Year', 
        values='Passengers', 
        fill_value=0
    )
    route_presence = (route_presence > 0).astype(int)
    
    print(f"Route presence matrix: {route_presence.shape}")
    
    # Route-level aggregations
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

def prepare_airline_routes_by_year(od_years, classification_map):
    """Prepare airline routes by year for H1 analysis"""
    
    print("Processing airline routes by year...")
    airline_routes_by_year = {}
    
    for year in sorted(od_years.keys()):
        print(f"Processing {year}...", end='')
        year_data = od_years[year].copy()
        
        # Use Marketing Carrier consistently
        year_data = year_data[year_data['Mkt'].isin(classification_map.keys())].copy()
        year_data['Business_Model'] = year_data['Mkt'].map(classification_map)
        
        # Store route sets by airline
        year_routes = {}
        for airline, group in year_data.groupby('Mkt'):
            bm = group['Business_Model'].iloc[0]
            routes = set((group['Org'] + '-' + group['Dst']).unique())
            year_routes[airline] = {'routes': routes, 'business_model': bm}
        
        airline_routes_by_year[year] = year_routes
        print(f" done")
    
    return airline_routes_by_year

def prepare_base_data():
    """Main data preparation function - Hub for all analyses"""
    
    print("ULCC STRATEGIC VOLATILITY ANALYSIS - DATA PREPARATION")
    print("=" * 60)
    
    od_years, t100_years, shock_data, classification_map = load_data()
    
    print(f"Loaded classification for {len(classification_map)} airlines")
    print(f"Loaded OD data: {len(od_years)} years")
    print(f"Loaded T100 data: {len(t100_years)} years")
    
    # Prepare combined datasets for analysis
    combined_od, combined_t100 = prepare_combined_data(od_years, t100_years, classification_map)
    
    # Create route datasets for H2, H3 analysis
    route_presence, route_level_od, route_level_t100 = create_route_datasets(combined_od, combined_t100)
    
    # Prepare airline routes by year for H1 analysis
    airline_routes_by_year = prepare_airline_routes_by_year(od_years, classification_map)
    
    base_data = {
        # Raw data
        'od_years': od_years,
        't100_years': t100_years, 
        'shock_data': shock_data,
        'classification_map': classification_map,
        
        # Combined datasets for analysis
        'combined_od': combined_od,
        'combined_t100': combined_t100,
        
        # Route datasets
        'route_presence': route_presence,
        'route_level_od': route_level_od,
        'route_level_t100': route_level_t100,
        
        # Airline routes by year for H1
        'airline_routes_by_year': airline_routes_by_year,
        
        # Visualization
        'colors': CARRIER_COLORS
    }
    
    print("Base data preparation complete!")
    print(f"Combined OD: {len(combined_od):,} rows")
    print(f"Route presence: {route_presence.shape}")
    print(f"Airline routes by year: {len(airline_routes_by_year)} years")
    
    # Generate Table 3.1
    create_table_3_1(base_data)
    
    return base_data

if __name__ == "__main__":
    base_data = prepare_base_data()