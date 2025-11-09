# h2b_supplementary.py - Panel C를 Spirit=100 기준으로 수정

# #num1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import seaborn as sns
import os
import networkx as nx  # Added for proper modularity calculation
warnings.filterwarnings('ignore')

# Define carrier colors from basecode (combined_analysis.py)
CARRIER_COLORS = {
   'NK': '#FF6B6B',  # Spirit - Red/Coral
   'F9': '#4ECDC4',  # Frontier - Teal  
   'G4': '#45B7D1',  # Allegiant - Sky Blue
}

# Standard alpha from basecode
ALPHA_LEVEL = 0.8

# Data paths
DATA_PATH = 'data'
FARE_PATH = os.path.join(DATA_PATH, 'fare', 'fare_all.parquet')
T100_PATH = os.path.join(DATA_PATH, 't_100')
OUTPUT_PATH = 'paper_1_outputs'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_PATH):
   os.makedirs(OUTPUT_PATH)

# #num2: Calculate operational metrics
def calculate_operational_metrics(df_fare, df_t100):
   """
   Calculate Load Factor, Yield, Stage Length for each carrier
   """
   results = {}
   
   for carrier in ['NK', 'F9', 'G4']:
       # Filter data
       carrier_fare = df_fare[df_fare['Mkt'] == carrier]
       carrier_t100 = df_t100[df_t100['Mkt Al'] == carrier]
       
       if len(carrier_fare) > 0 and len(carrier_t100) > 0:
           # Load Factor from T100
           load_factor = carrier_t100['Load Factor'].mean()
           
           # Yield from Fare data (cents per passenger mile)
           total_revenue = (carrier_fare['Avg_Fare'] * carrier_fare['Passengers']).sum()
           total_passenger_miles = (carrier_fare['Miles'] * carrier_fare['Passengers']).sum()
           yield_per_mile = (total_revenue / total_passenger_miles * 100) if total_passenger_miles > 0 else 0
           
           # Stage Length (weighted average by passengers)
           stage_length = np.average(carrier_fare['Miles'], weights=carrier_fare['Passengers'])
           
           results[carrier] = {
               'load_factor': load_factor,
               'yield': yield_per_mile,
               'stage_length': stage_length
           }
   
   return results

# #num3: Calculate route-level HHI
def calculate_route_hhi_comparison_fast(df_fare):
   """
   Calculate weighted average HHI for routes operated by each carrier
   """
   print("\nCalculating Route-Level Competition (HHI)...")
   
   # Aggregate all data
   df_agg = df_fare.groupby(['Org', 'Dst', 'Mkt'])['Passengers'].sum().reset_index()
   df_agg['Route'] = df_agg['Org'] + '-' + df_agg['Dst']
   
   # Calculate total passengers per route
   route_totals = df_agg.groupby('Route')['Passengers'].sum().reset_index()
   route_totals.columns = ['Route', 'Total_Pax']
   
   # Merge to get market shares
   route_shares = df_agg.merge(route_totals, on='Route')
   route_shares['Market_Share'] = (route_shares['Passengers'] / route_shares['Total_Pax']) * 100
   
   # Calculate HHI for each route
   route_hhi = route_shares.groupby('Route').apply(
       lambda x: (x['Market_Share'] ** 2).sum()
   ).to_dict()
   
   route_shares['HHI'] = route_shares['Route'].map(route_hhi)
   
   # Calculate metrics for each ULCC
   results = {}
   for carrier in ['NK', 'F9', 'G4']:
       carrier_data = route_shares[route_shares['Mkt'] == carrier].copy()
       
       if len(carrier_data) > 0:
           total_pax = carrier_data['Passengers'].sum()
           
           results[carrier] = {
               'weighted_avg_hhi': np.average(carrier_data['HHI'], 
                                            weights=carrier_data['Passengers']),
               'num_routes': len(carrier_data),
               'monopoly_routes': (carrier_data['HHI'] >= 8000).sum(),
               'high_competition_routes': (carrier_data['HHI'] < 2500).sum()
           }
   
   return results

# #num4: Create integrated comparison table with Modularity Score
def create_integrated_comparison_table(ops_metrics, hhi_metrics, avg_volatilities, network_metrics, df_fare_full):
   """
   Create integrated comparison table with Modularity Score added
   """
   carriers = ['NK', 'F9', 'G4']
   carrier_names = {'NK': 'Spirit', 'F9': 'Frontier', 'G4': 'Allegiant'}
   
   # Create data structure
   table_data = {
       'Metric': [
           'Load Factor (%)',
           'Stage Length (miles)',
           'Yield (¢/mile)',
           'Weighted Avg HHI',
           'Number of Routes',
           'Network Modularity',  # Changed name
           'Route Adaptability (%)',
           'Average Fare ($)',
           'Total Passengers (M)',
           '--- Relative to Spirit ---',
           'Load Factor Diff (%)',
           'Stage Length Diff (miles)',
           'Yield Diff (¢/mile)',
           'Modularity Diff',  # Changed name
           'Adaptability Diff (%)'
       ]
   }
   
   # Add data for each carrier
   for i, carrier in enumerate(carriers):
       carrier_col = []
       
       # Basic metrics
       carrier_col.append(f"{ops_metrics[carrier]['load_factor']:.1f}")
       carrier_col.append(f"{ops_metrics[carrier]['stage_length']:.0f}")
       carrier_col.append(f"{ops_metrics[carrier]['yield']:.2f}")
       carrier_col.append(f"{hhi_metrics[carrier]['weighted_avg_hhi']:.0f}" if carrier in hhi_metrics else "N/A")
       carrier_col.append(f"{hhi_metrics[carrier]['num_routes']}" if carrier in hhi_metrics else "N/A")
       carrier_col.append(f"{network_metrics[carrier]['modularity']:.3f}" if carrier in network_metrics else "N/A")  # Changed to use proper modularity
       carrier_col.append(f"{avg_volatilities[i]:.1f}" if i < len(avg_volatilities) else "N/A")
       carrier_col.append(f"{df_fare_full[df_fare_full['Mkt']==carrier]['Avg_Fare'].mean():.0f}")
       carrier_col.append(f"{df_fare_full[df_fare_full['Mkt']==carrier]['Passengers'].sum()/1e6:.1f}")
       
       # Separator
       carrier_col.append("---")
       
       # Relative to Spirit
       if carrier == 'NK':
           carrier_col.extend(['Baseline', 'Baseline', 'Baseline', 'Baseline', 'Baseline'])
       else:
           carrier_col.append(f"{ops_metrics[carrier]['load_factor'] - ops_metrics['NK']['load_factor']:+.1f}")
           carrier_col.append(f"{ops_metrics[carrier]['stage_length'] - ops_metrics['NK']['stage_length']:+.0f}")
           carrier_col.append(f"{ops_metrics[carrier]['yield'] - ops_metrics['NK']['yield']:+.2f}")
           carrier_col.append(f"{network_metrics[carrier]['modularity'] - network_metrics['NK']['modularity']:+.3f}")  # Changed to use proper modularity
           carrier_col.append(f"{avg_volatilities[i] - avg_volatilities[0]:+.1f}")
       
       table_data[carrier_names[carrier]] = carrier_col
   
   return pd.DataFrame(table_data)

# #num5: Create four-panel visualization with Network Modularity (Spirit=100)
def create_four_panel_analysis(df_fare_full, df_t100):
   """
   Create 1x4 panel figure with Network Modularity as Panel C (normalized to Spirit=100)
   """
   print("\nCreating Figure 4.5: Four-Panel Analysis...")
   
   # Filter for ULCC for operational metrics
   ulcc_carriers = ['NK', 'F9', 'G4']
   df_fare_ulcc = df_fare_full[df_fare_full['Mkt'].isin(ulcc_carriers)]
   
   # Calculate metrics
   ops_metrics = calculate_operational_metrics(df_fare_ulcc, df_t100)
   hhi_metrics = calculate_route_hhi_comparison_fast(df_fare_full)
   
   # Create 1x4 figure
   fig, axes = plt.subplots(1, 4, figsize=(20, 5))
   
   carriers = ['NK', 'F9', 'G4']
   carrier_names = {'NK': 'Spirit', 'F9': 'Frontier', 'G4': 'Allegiant'}
   x_pos = np.arange(len(carriers))
   
   # ====================================
   # Panel A: Route Competition (HHI)
   # ====================================
   ax1 = axes[0]
   
   years = sorted(df_fare_full['Year'].unique())
   yearly_hhi = {carrier: [] for carrier in carriers}
   
   min_hhi_val = 10000
   max_hhi_val = 0
   
   for year in years:
       year_data = df_fare_full[df_fare_full['Year'] == year].copy()
       year_agg = year_data.groupby(['Org', 'Dst', 'Mkt'])['Passengers'].sum().reset_index()
       year_agg['Route'] = year_agg['Org'] + '-' + year_agg['Dst']
       
       route_totals = year_agg.groupby('Route')['Passengers'].sum()
       year_agg['Total_Pax'] = year_agg['Route'].map(route_totals)
       year_agg['Market_Share'] = (year_agg['Passengers'] / year_agg['Total_Pax']) * 100
       
       route_hhi = year_agg.groupby('Route').apply(
           lambda x: (x['Market_Share'] ** 2).sum()
       ).to_dict()
       
       year_agg['HHI'] = year_agg['Route'].map(route_hhi)
       
       for carrier in carriers:
           carrier_data = year_agg[year_agg['Mkt'] == carrier]
           if len(carrier_data) > 0:
               total_pax = carrier_data['Passengers'].sum()
               if total_pax > 0:
                   weighted_hhi = np.average(carrier_data['HHI'], weights=carrier_data['Passengers'])
                   yearly_hhi[carrier].append(weighted_hhi)
                   min_hhi_val = min(min_hhi_val, weighted_hhi)
                   max_hhi_val = max(max_hhi_val, weighted_hhi)
               else:
                   yearly_hhi[carrier].append(np.nan)
           else:
               yearly_hhi[carrier].append(np.nan)
   
   y_min = max(1000, min_hhi_val - 500)
   y_max = min(10000, max_hhi_val + 500)
   
   for carrier in carriers:
       plot_years = []
       plot_hhi = []
       for i, year in enumerate(years):
           if i < len(yearly_hhi[carrier]) and not np.isnan(yearly_hhi[carrier][i]):
               plot_years.append(year)
               plot_hhi.append(yearly_hhi[carrier][i])
       
       if len(plot_years) > 0:
           ax1.plot(plot_years, plot_hhi, 
                   marker='o', label=carrier_names[carrier], 
                   color=CARRIER_COLORS[carrier], linewidth=2, markersize=6,
                   alpha=ALPHA_LEVEL)
   
   ax1.axvspan(2020, 2021, alpha=0.2, color='gray')
   ax1.text(2020.5, y_min + (y_max - y_min) * 0.8, 'COVID', 
            ha='center', fontsize=9, alpha=0.7)
   
   ax1.axhline(y=2500, color='green', linestyle=':', alpha=0.3, linewidth=1)
   ax1.axhline(y=5000, color='orange', linestyle=':', alpha=0.3, linewidth=1)
   if max_hhi_val > 7500:
       ax1.axhline(y=8000, color='red', linestyle=':', alpha=0.3, linewidth=1)
   
   ax1.set_xlabel('Year')
   ax1.set_ylabel('Weighted Average HHI')
   ax1.set_title('Panel A: Route Competition (HHI)', fontweight='bold')
   ax1.legend(loc='best', frameon=False, fontsize=8)
   ax1.grid(True, alpha=0.3)
   ax1.set_ylim(y_min, y_max)
   
   # ====================================
   # Panel B: Operational Metrics
   # ====================================
   ax2 = axes[1]
   width = 0.25
   
   spirit_lf = ops_metrics['NK']['load_factor']
   spirit_stage = ops_metrics['NK']['stage_length']
   spirit_yield = ops_metrics['NK']['yield']
   
   load_factors_norm = [(ops_metrics[c]['load_factor']/spirit_lf)*100 for c in carriers]
   stage_lengths_norm = [(ops_metrics[c]['stage_length']/spirit_stage)*100 for c in carriers]
   yields_norm = [(ops_metrics[c]['yield']/spirit_yield)*100 for c in carriers]
   
   bars1 = ax2.bar(x_pos - width, load_factors_norm, width, 
                   label='Load Factor', 
                   color='#4169E1', alpha=ALPHA_LEVEL, edgecolor='black', linewidth=1)
   bars2 = ax2.bar(x_pos, stage_lengths_norm, width, 
                   label='Stage Length', 
                   color='#8B4513', alpha=ALPHA_LEVEL, edgecolor='black', linewidth=1)
   bars3 = ax2.bar(x_pos + width, yields_norm, width, 
                   label='Yield', 
                   color='#DC143C', alpha=ALPHA_LEVEL, edgecolor='black', linewidth=1)
   
   for bars, norm_values in [(bars1, load_factors_norm), 
                              (bars2, stage_lengths_norm), 
                              (bars3, yields_norm)]:
       for bar, norm_val in zip(bars, norm_values):
           ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                   f'{norm_val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
   
   ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
   ax2.text(2.5, 101, 'Spirit baseline', fontsize=8, alpha=0.5, ha='right')
   
   ax2.set_xlabel('Carrier')
   ax2.set_ylabel('Relative Performance (Spirit = 100)')
   ax2.set_title('Panel B: Operational Metrics', fontweight='bold')
   ax2.set_xticks(x_pos)
   ax2.set_xticklabels([carrier_names[c] for c in carriers])
   ax2.legend(loc='upper left', frameon=False, fontsize=8)
   ax2.grid(True, alpha=0.3, axis='y')
   ax2.set_ylim(70, 135)
   
   # ====================================
   # Panel C: Network Modularity (FIXED - Using proper Louvain method)
   # ====================================
   ax3 = axes[2]
   
   print("\nCalculating Network Modularity...")
   
   # Calculate network metrics for each carrier
   network_metrics = {}
   
   for carrier in carriers:
       carrier_data = df_fare_ulcc[df_fare_ulcc['Mkt'] == carrier]
       
       if len(carrier_data) > 0:
           # Build network graph for modularity calculation
           G = nx.Graph()
           for _, row in carrier_data.iterrows():
               if G.has_edge(row['Org'], row['Dst']):
                   G[row['Org']][row['Dst']]['weight'] += row['Passengers']
               else:
                   G.add_edge(row['Org'], row['Dst'], weight=row['Passengers'])
           
           # Calculate proper modularity using Louvain method
           try:
               communities = nx.community.louvain_communities(G, weight='weight', seed=42)
               modularity = nx.community.modularity(G, communities, weight='weight')
           except:
               modularity = 0
           
           # Calculate hub concentration (using top 3 airports) - keep for reference
           airports = set(carrier_data['Org'].unique()) | set(carrier_data['Dst'].unique())
           n_airports = len(airports)
           routes = carrier_data[['Org', 'Dst']].drop_duplicates()
           n_routes = len(routes)
           
           airport_traffic = {}
           for airport in airports:
               orig_pax = carrier_data[carrier_data['Org'] == airport]['Passengers'].sum()
               dest_pax = carrier_data[carrier_data['Dst'] == airport]['Passengers'].sum()
               airport_traffic[airport] = orig_pax + dest_pax
           
           sorted_airports = sorted(airport_traffic.items(), key=lambda x: x[1], reverse=True)
           top3_traffic = sum([traffic for _, traffic in sorted_airports[:3]])
           total_traffic = sum(airport_traffic.values())
           hub_concentration = (top3_traffic / total_traffic * 100) if total_traffic > 0 else 0
           
           # Calculate network density
           possible_routes = n_airports * (n_airports - 1)  # Directed
           network_density = (n_routes / possible_routes * 100) if possible_routes > 0 else 0
           
           network_metrics[carrier] = {
               'hub_concentration': hub_concentration,
               'network_density': network_density,
               'modularity': modularity,  # Using proper modularity
               'n_airports': n_airports,
               'n_routes': n_routes
           }
   
   # NORMALIZE to Spirit=100
   spirit_hub = network_metrics['NK']['hub_concentration']
   spirit_density = network_metrics['NK']['network_density']
   spirit_modularity = network_metrics['NK']['modularity']
   
   width = 0.25
   
   # Normalized values (Spirit = 100)
   hub_conc_norm = [(network_metrics[c]['hub_concentration']/spirit_hub)*100 for c in carriers]
   net_density_norm = [(network_metrics[c]['network_density']/spirit_density)*100 for c in carriers]
   modularity_norm = [(network_metrics[c]['modularity']/spirit_modularity)*100 if spirit_modularity > 0 else 100 for c in carriers]
   
   bars1 = ax3.bar(x_pos - width, hub_conc_norm, width, 
                   label='Hub Concentration', 
                   color='#FF6B6B', alpha=ALPHA_LEVEL, edgecolor='black', linewidth=1)
   bars2 = ax3.bar(x_pos, net_density_norm, width, 
                   label='Network Density', 
                   color='#4ECDC4', alpha=ALPHA_LEVEL, edgecolor='black', linewidth=1)
   bars3 = ax3.bar(x_pos + width, modularity_norm, width, 
                   label='Network Modularity', 
                   color='#45B7D1', alpha=ALPHA_LEVEL, edgecolor='black', linewidth=1)
   
   # Add value labels
   for bars, values in [(bars1, hub_conc_norm), (bars2, net_density_norm), (bars3, modularity_norm)]:
       for bar, val in zip(bars, values):
           ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
   
   # Add baseline line
   ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
   ax3.text(2.5, 101, 'Spirit baseline', fontsize=8, alpha=0.5, ha='right')
   
   ax3.set_xlabel('Carrier')
   ax3.set_ylabel('Relative Score (Spirit = 100)')
   ax3.set_title('Panel C: Network Structure', fontweight='bold')
   ax3.set_xticks(x_pos)
   ax3.set_xticklabels([carrier_names[c] for c in carriers])
   ax3.legend(loc='upper left', frameon=False, fontsize=8)
   ax3.grid(True, alpha=0.3, axis='y')
   ax3.set_ylim(30, 165)  # Adjusted for normalized values - increased upper limit
   
   # ====================================
   # Panel D: Route Adaptability
   # ====================================
   ax4 = axes[3]
   
   print("\nCalculating Average Route Adaptability...")
   
   od_years = {}
   for year in range(2014, 2025):
       year_fare = df_fare_full[df_fare_full['Year'] == year]
       if len(year_fare) > 0:
           od_years[year] = year_fare
   
   all_volatilities = {carrier: [] for carrier in carriers}
   
   for year in range(2015, 2025):
       if year in od_years and year-1 in od_years:
           for carrier in carriers:
               curr_year = od_years[year][od_years[year]['Mkt'] == carrier]
               prev_year = od_years[year-1][od_years[year-1]['Mkt'] == carrier]
               
               if len(curr_year) > 0 and len(prev_year) > 0:
                   curr_routes = set(curr_year['Org'] + '-' + curr_year['Dst'])
                   prev_routes = set(prev_year['Org'] + '-' + prev_year['Dst'])
                   
                   routes_added = len(curr_routes - prev_routes)
                   routes_dropped = len(prev_routes - curr_routes)
                   total_changes = routes_added + routes_dropped
                   base_routes = len(prev_routes)
                   
                   if base_routes > 0:
                       volatility = (total_changes / base_routes) * 100
                       all_volatilities[carrier].append(volatility)
   
   avg_volatilities = []
   std_volatilities = []
   for carrier in carriers:
       if all_volatilities[carrier]:
           avg_vol = np.mean(all_volatilities[carrier])
           std_vol = np.std(all_volatilities[carrier])
           avg_volatilities.append(avg_vol)
           std_volatilities.append(std_vol)
       else:
           avg_volatilities.append(0)
           std_volatilities.append(0)
   
   bars = ax4.bar(x_pos, avg_volatilities, 
                  color=[CARRIER_COLORS[c] for c in carriers],
                  alpha=ALPHA_LEVEL, edgecolor='black', linewidth=1,
                  yerr=std_volatilities, capsize=5)
   
   for bar, val, std in zip(bars, avg_volatilities, std_volatilities):
       ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
   
   ax4.set_xlabel('Carrier')
   ax4.set_ylabel('Average Route Adaptability (%)')
   ax4.set_title('Panel D: Route Adaptability', fontweight='bold')
   ax4.set_xticks(x_pos)
   ax4.set_xticklabels([carrier_names[c] for c in carriers])
   ax4.grid(True, alpha=0.3, axis='y')
   ax4.set_ylim(0, 40)
   
   plt.tight_layout()
   
   # Save figure as Figure 4.5
   try:
       output_file = os.path.join(OUTPUT_PATH, 'Figure_4.5_H2b_Analysis.png')
       plt.savefig(output_file, dpi=300, bbox_inches='tight')
       print(f"\nFigure 4.5 saved to: {output_file}")
   except Exception as e:
       print(f"Warning: Could not save Figure 4.5 - {e}")
   
   plt.show()
   
   # Return network_metrics along with other metrics
   return ops_metrics, hhi_metrics, avg_volatilities, network_metrics

# #num6: Main function (동일)
def run_h2_supplementary_analysis(base_data=None):
   """
   Main function to run H2b supplementary analysis
   """
   print("="*80)
   print("H2B SUPPLEMENTARY ANALYSIS")
   print("="*80)
   
   try:
       # Load data
       print("\nLoading data...")
       df_fare_full = pd.read_parquet(FARE_PATH)
       
       # Load all T100 data
       t100_frames = []
       for year in range(2014, 2025):
           file_path = os.path.join(T100_PATH, f't_100_{year}.parquet')
           if os.path.exists(file_path):
               t100_frames.append(pd.read_parquet(file_path))
       df_t100 = pd.concat(t100_frames, ignore_index=True)
       
       print(f"Loaded {len(df_fare_full):,} total fare records")
       print(f"Loaded {len(df_t100):,} T100 records")
       
       # Filter T100 for ULCCs only
       ulcc_carriers = ['NK', 'F9', 'G4']
       df_t100_ulcc = df_t100[df_t100['Mkt Al'].isin(ulcc_carriers)]
       
       print(f"Filtered to {len(df_t100_ulcc):,} ULCC T100 records")
       
       # Run analysis - receives 4 return values
       ops_metrics, hhi_metrics, avg_volatilities, network_metrics = create_four_panel_analysis(df_fare_full, df_t100_ulcc)
       
       # Table and other outputs remain the same...
       print("\n" + "="*80)
       print("TABLE 4.5: INTEGRATED COMPARISON")
       print("="*80)
       
       integrated_table = create_integrated_comparison_table(ops_metrics, hhi_metrics, avg_volatilities, network_metrics, df_fare_full)
       print("\n" + integrated_table.to_string(index=False))
       
       # Save table
       try:
           table_file = os.path.join(OUTPUT_PATH, 'Table_4.5_H2b_Comparison.csv')
           integrated_table.to_csv(table_file, index=False)
           print(f"\nTable 4.5 saved to: {table_file}")
       except Exception as e:
           print(f"Warning: Could not save Table 4.5 - {e}")
       
       print("\n" + "="*80)
       print("Analysis completed!")
       print("="*80)
       
       return {
           'ops_metrics': ops_metrics,
           'hhi_metrics': hhi_metrics,
           'avg_volatilities': avg_volatilities,
           'network_metrics': network_metrics
       }
       
   except Exception as e:
       print(f"\nError: {e}")
       import traceback
       traceback.print_exc()
       return None

# Execute when run as script
if __name__ == "__main__":
   results = run_h2_supplementary_analysis()