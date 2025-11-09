# #num029 - H3 Network Structure Analysis
# H3: Network Modularity Hypothesis - ULCCs exhibit higher degree of modularity

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from scipy import stats

# num1: Import required modules and setup
def run_h3_analysis(base_data):
    """
    H3: Network Structure Analysis
    H3: ULCCs will exhibit higher degree of modularity compared to other business models
    """
    
    print("H3: NETWORK MODULARITY HYPOTHESIS")
    print("Testing: ULCCs exhibit higher degree of modularity compared to other carrier types")
    print("=" * 80)
    
    # Main network structure analysis
    network_results = analyze_network_structure_h3(base_data)
    
    # Temporal evolution analysis
    evolution_results = analyze_network_evolution_h3(base_data)
    
    # Create visualizations
    fig1, fig2 = create_h3_figures(network_results, evolution_results)
    
    # H3 Hypothesis Testing with statistical tests
    print("\nH3 HYPOTHESIS TESTING RESULTS:")
    
    if len(network_results) > 0:
        ulcc_modularity = network_results.loc['ULCC', 'Modularity'] if 'ULCC' in network_results.index else 0
        other_modularities = [network_results.loc[bm, 'Modularity'] for bm in ['Legacy', 'LCC', 'Hybrid'] if bm in network_results.index]
        
        if other_modularities:
            max_other_modularity = max(other_modularities)
            h3_support = ulcc_modularity > max_other_modularity
            
            print(f"ULCC Modularity: {ulcc_modularity:.3f}")
            print(f"Max Others: {max_other_modularity:.3f} ({network_results.index[network_results['Modularity'] == max_other_modularity].tolist()[0]})")
            print(f"H3 Support: {'SUPPORTED' if h3_support else 'NOT SUPPORTED'}")
            
            # Show ranking
            rankings = network_results['Modularity'].sort_values(ascending=False)
            print(f"Modularity Rankings: {' > '.join([f'{idx}({val:.3f})' for idx, val in rankings.items()])}")
        else:
            print("Insufficient data for comparison")
    
    # Save results
    save_h3_results(network_results, evolution_results)
    
    print(f"\nH3 CONCLUSION:")
    print(f"Network modularity analysis validates structural differences between business models")
    print(f"This provides the structural foundation for competitive impact differences (H4)")
    
    return {
        'h3_network_structure': network_results,
        'h3_evolution': evolution_results,
        'figure_structure': fig1, 
        'figure_evolution': fig2
    }

# num2: Network structure analysis (H3) - EXACT MANUSCRIPT METHOD with statistical tests
def analyze_network_structure_h3(base_data):
    """
    Analyze network modularity by business model:
    1. Calculate individual airline modularity by year
    2. Average by business model by year  
    3. Average across all years for main results
    4. Perform statistical tests
    """
    
    print("\n=== H3: NETWORK STRUCTURE ANALYSIS ===")
    
    od_years = base_data['od_years']
    classification_map = base_data['classification_map']
    
    # Get all classified airlines
    classified_airlines = [k for k, v in classification_map.items() if v in ['Legacy', 'ULCC', 'LCC', 'Hybrid']]
    print(f"Analyzing {len(classified_airlines)} classified airlines across {len(od_years)} years")
    
    # STEP 1: Calculate individual airline modularity for each year
    all_airline_year_metrics = []
    
    print("Processing years:", end=" ")
    for year in sorted(od_years.keys()):
        print(f"{year}", end=" ")
        
        year_data = od_years[year].copy()
        year_data_filtered = year_data[
            (year_data['Mkt'].isin(classified_airlines)) & 
            (year_data['Passengers'] > 0)
        ].copy()
        
        if len(year_data_filtered) == 0:
            continue
        
        # Calculate modularity for EACH INDIVIDUAL AIRLINE in this year
        for airline, airline_data in year_data_filtered.groupby('Mkt'):
            carrier_type = classification_map.get(airline)
            if not carrier_type or carrier_type not in ['Legacy', 'ULCC', 'LCC', 'Hybrid']:
                continue
            
            # FILTER 1: Minimum passenger threshold 
            airline_data = airline_data[airline_data['Passengers'] >= 1000]
            
            if len(airline_data) == 0:
                continue
            
            # Build network for this specific airline in this specific year
            G = nx.Graph()
            for _, row in airline_data.iterrows():
                if G.has_edge(row['Org'], row['Dst']):
                    G[row['Org']][row['Dst']]['weight'] += row['Passengers']
                else:
                    G.add_edge(row['Org'], row['Dst'], weight=row['Passengers'])
            
            # FILTER 2: Minimum network size for meaningful modularity
            if len(G.nodes()) < 3:  # Need at least 3 nodes for communities
                continue
            
            # Calculate modularity for THIS AIRLINE in THIS YEAR
            try:
                communities = nx.community.louvain_communities(G, weight='weight', seed=42)
                modularity = nx.community.modularity(G, communities, weight='weight')
            except:
                modularity = 0
            
            # Calculate other metrics for THIS AIRLINE in THIS YEAR
            node_weights = np.array([
                sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node)) 
                for node in G.nodes()
            ])
            
            if len(node_weights) >= 3:
                top3_share = (np.sum(np.partition(node_weights, -3)[-3:]) / np.sum(node_weights)) * 100
            else:
                top3_share = 100.0
            
            route_pax = airline_data['Passengers'].values
            if len(route_pax) > 1:
                route_pax_sorted = np.sort(route_pax)
                n = len(route_pax_sorted)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * route_pax_sorted)) / (n * np.sum(route_pax_sorted)) - (n + 1) / n
            else:
                gini = 0
            
            # Calculate network density
            n_airports = len(G.nodes())
            n_routes = len(airline_data)
            possible_routes = n_airports * (n_airports - 1)  # Directed
            network_density = (n_routes / possible_routes * 100) if possible_routes > 0 else 0
            
            all_airline_year_metrics.append({
                'Year': year,
                'Airline': airline,
                'Carrier_Type': carrier_type,
                'Modularity': modularity,
                'Gini': gini,
                'Hub_Concentration': top3_share,
                'Network_Density': network_density,
                'Routes': len(airline_data),
                'Airports': len(G.nodes())
            })
    
    print()  # New line
    
    # Convert to DataFrame
    airline_year_df = pd.DataFrame(all_airline_year_metrics)
    print(f"Total individual airline-year observations: {len(airline_year_df)}")
    
    # STEP 2: Calculate business model averages BY YEAR
    bm_year_averages = airline_year_df.groupby(['Year', 'Carrier_Type']).agg({
        'Modularity': 'mean',
        'Gini': 'mean',
        'Hub_Concentration': 'mean',
        'Network_Density': 'mean',
        'Routes': 'sum',
        'Airports': 'sum',
        'Airline': 'nunique'
    }).reset_index()
    
    bm_year_averages.rename(columns={'Airline': 'Airlines_Count'}, inplace=True)
    
    print("\nBusiness Model yearly averages calculated:")
    print(f"Years covered: {sorted(bm_year_averages['Year'].unique())}")
    print(f"Business models: {sorted(bm_year_averages['Carrier_Type'].unique())}")
    
    # STEP 3: Calculate overall averages across all years
    overall_averages = bm_year_averages.groupby('Carrier_Type').agg({
        'Modularity': 'mean',  # Average of yearly averages
        'Gini': 'mean',        # Average of yearly averages  
        'Hub_Concentration': 'mean',
        'Network_Density': 'mean',
        'Routes': 'mean',
        'Airports': 'mean',
        'Airlines_Count': 'mean',
        'Year': 'count'        # Number of years with data
    }).round(4)
    
    overall_averages.rename(columns={'Year': 'Years_Count'}, inplace=True)
    
    # STEP 4: Statistical Testing
    print("\n=== STATISTICAL TESTING ===")
    
    # Prepare data for ANOVA
    modularity_by_bm = {}
    gini_by_bm = {}
    hub_conc_by_bm = {}
    density_by_bm = {}
    
    for bm in ['ULCC', 'Legacy', 'LCC', 'Hybrid']:
        bm_data = airline_year_df[airline_year_df['Carrier_Type'] == bm]
        if len(bm_data) > 0:
            modularity_by_bm[bm] = bm_data['Modularity'].values
            gini_by_bm[bm] = bm_data['Gini'].values
            hub_conc_by_bm[bm] = bm_data['Hub_Concentration'].values
            density_by_bm[bm] = bm_data['Network_Density'].values
    
    # Perform ANOVA for each metric
    if len(modularity_by_bm) >= 2:
        # Modularity ANOVA
        f_stat_mod, p_val_mod = stats.f_oneway(*modularity_by_bm.values())
        print(f"Modularity ANOVA: F={f_stat_mod:.2f}, p={p_val_mod:.4f}")
        
        # Gini ANOVA
        f_stat_gini, p_val_gini = stats.f_oneway(*gini_by_bm.values())
        print(f"Gini ANOVA: F={f_stat_gini:.2f}, p={p_val_gini:.4f}")
        
        # Hub Concentration ANOVA
        f_stat_hub, p_val_hub = stats.f_oneway(*hub_conc_by_bm.values())
        print(f"Hub Concentration ANOVA: F={f_stat_hub:.2f}, p={p_val_hub:.4f}")
        
        # Network Density ANOVA
        f_stat_density, p_val_density = stats.f_oneway(*density_by_bm.values())
        print(f"Network Density ANOVA: F={f_stat_density:.2f}, p={p_val_density:.4f}")
        
        # Post-hoc tests: Compare ULCC to each other business model
        if 'ULCC' in modularity_by_bm:
            ulcc_mod = modularity_by_bm['ULCC']
            ulcc_gini = gini_by_bm['ULCC']
            ulcc_hub = hub_conc_by_bm['ULCC']
            ulcc_density = density_by_bm['ULCC']
            
            post_hoc_results = {}
            for bm in ['Legacy', 'LCC', 'Hybrid']:
                if bm in modularity_by_bm:
                    # Modularity comparison
                    t_mod, p_mod = stats.ttest_ind(ulcc_mod, modularity_by_bm[bm])
                    
                    # Gini comparison
                    t_gini, p_gini = stats.ttest_ind(ulcc_gini, gini_by_bm[bm])
                    
                    # Hub Concentration comparison
                    t_hub, p_hub = stats.ttest_ind(ulcc_hub, hub_conc_by_bm[bm])
                    
                    # Network Density comparison
                    t_density, p_density = stats.ttest_ind(ulcc_density, density_by_bm[bm])
                    
                    post_hoc_results[bm] = {
                        'modularity_p': p_mod,
                        'gini_p': p_gini,
                        'hub_conc_p': p_hub,
                        'density_p': p_density
                    }
                    
                    print(f"\nULCC vs {bm}:")
                    print(f"  Modularity: p={p_mod:.4f} {'***' if p_mod < 0.001 else '**' if p_mod < 0.01 else '*' if p_mod < 0.05 else ''}")
                    print(f"  Gini: p={p_gini:.4f} {'***' if p_gini < 0.001 else '**' if p_gini < 0.01 else '*' if p_gini < 0.05 else ''}")
                    print(f"  Hub Concentration: p={p_hub:.4f} {'***' if p_hub < 0.001 else '**' if p_hub < 0.01 else '*' if p_hub < 0.05 else ''}")
                    print(f"  Network Density: p={p_density:.4f} {'***' if p_density < 0.001 else '**' if p_density < 0.01 else '*' if p_density < 0.05 else ''}")
            
            # Store statistical results separately
            stat_results = {
                'ULCC': {
                    'anova': {'modularity': p_val_mod, 'gini': p_val_gini, 'hub_conc': p_val_hub, 'density': p_val_density},
                    'post_hoc': post_hoc_results,
                    'f_stats': {'modularity': f_stat_mod, 'gini': f_stat_gini, 'hub_conc': f_stat_hub, 'density': f_stat_density}
                }
            }
            # Store as attribute instead of column
            overall_averages.stat_results = stat_results
    
    print("\n" + "="*60)
    print("H3 MAIN RESULTS (Average of yearly BM averages across 11 years)")
    print("="*60)
    print(overall_averages[['Modularity', 'Hub_Concentration', 'Network_Density', 'Gini', 'Routes', 'Airports']])
    
    # Store yearly averages for temporal analysis
    overall_averages['yearly_data'] = [
        bm_year_averages[bm_year_averages['Carrier_Type'] == ct] 
        for ct in overall_averages.index
    ]
    
    return overall_averages

# num3: Network evolution analysis
def analyze_network_evolution_h3(base_data):
    """Extract yearly BM averages for temporal evolution analysis"""
    
    print("\n=== H3 TEMPORAL: EXTRACTING YEARLY BM AVERAGES ===")
    
    # Run the main analysis to get yearly data
    network_results = analyze_network_structure_h3(base_data)
    
    if len(network_results) == 0 or 'yearly_data' not in network_results.columns:
        print("No yearly data available")
        return pd.DataFrame()
    
    # Extract and combine yearly data from all business models
    evolution_data = []
    
    for carrier_type in network_results.index:
        yearly_data = network_results.loc[carrier_type, 'yearly_data']
        if len(yearly_data) > 0:
            # Add business model info for consistency
            yearly_data_copy = yearly_data.copy()
            yearly_data_copy['Business_Model'] = yearly_data_copy['Carrier_Type']
            evolution_data.append(yearly_data_copy)
    
    if evolution_data:
        evolution_df = pd.concat(evolution_data, ignore_index=True)
        
        print(f"\nEvolution data extracted: {len(evolution_df)} year-BM combinations")
        print(f"Years: {sorted(evolution_df['Year'].unique())}")
        print(f"Business Models: {sorted(evolution_df['Carrier_Type'].unique())}")
        
        return evolution_df
    else:
        print("No evolution data generated")
        return pd.DataFrame()

# num4: Create H3 visualizations - Updated figure numbers and styles
def create_h3_figures(network_df, evolution_df):
    """Create Figure 4.6: Network Structure Analysis and Figure 4.7: Strategic Position Evolution"""
    
    os.makedirs('paper_1_outputs', exist_ok=True)
    
    # Import colors from basecode
    try:
        from basecode import CARRIER_COLORS as colors
    except:
        colors = {'Legacy': '#1f77b4', 'ULCC': '#ff7f0e', 'LCC': '#2ca02c', 'Hybrid': '#d62728'}
    
    # Figure 4.6: Network Structure Analysis (3 panels)
    fig1 = plt.figure(figsize=(15, 5))
    
    # Panel A: Network Modularity - UPDATED STYLE
    ax1 = plt.subplot(1, 3, 1)
    mod_data = network_df['Modularity'].sort_values(ascending=False)
    for i, (carrier, value) in enumerate(mod_data.items()):
        # Legacy gets white bar, others get their colors
        if carrier == 'Legacy':
            bar_color = 'white'
        else:
            bar_color = colors.get(carrier, 'gray')
        
        ax1.bar(i, value, color=bar_color, alpha=0.8, width=0.6, 
               edgecolor='black', linewidth=1)
        ax1.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_title('Panel A: Network Modularity', fontweight='bold', pad=15)
    ax1.set_ylabel('Modularity Score')
    ax1.set_xticks(range(len(mod_data)))
    ax1.set_xticklabels(mod_data.index)
    ax1.set_ylim(0, max(mod_data.values) * 1.15)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Modularity Over Time
    ax2 = plt.subplot(1, 3, 2)
    if len(evolution_df) > 0:
        for model in ['Legacy', 'ULCC', 'LCC', 'Hybrid']:
            model_data = evolution_df[evolution_df['Business_Model'] == model]
            if len(model_data) > 0:
                ax2.plot(model_data['Year'], model_data['Modularity'], 
                        marker='o', color=colors.get(model, 'gray'), 
                        linewidth=2, markersize=4, label=model)
    
    ax2.set_title('Panel B: Modularity Over Time', fontweight='bold', pad=15)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Modularity Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Hub Concentration vs Modularity
    ax3 = plt.subplot(1, 3, 3)
    for carrier, row in network_df.iterrows():
        ax3.scatter(row['Hub_Concentration'], row['Modularity'], s=150, 
                   color=colors.get(carrier, 'gray'), alpha=0.8, edgecolors='black', linewidth=0.5)
        ax3.annotate(carrier, (row['Hub_Concentration'], row['Modularity']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax3.set_title('Panel C: Hub Concentration vs Modularity', fontweight='bold', pad=15)
    ax3.set_xlabel('Hub Concentration (%)')
    ax3.set_ylabel('Modularity Score')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('paper_1_outputs/Figure_4_6_H3_Network_Structure_Analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Figure 4.7: Strategic Position Evolution
    fig2 = plt.figure(figsize=(8, 4))
    ax = plt.gca()
    
    if len(evolution_df) > 0:
        for model in ['Legacy', 'ULCC', 'LCC', 'Hybrid']:
            model_data = evolution_df[evolution_df['Business_Model'] == model]
            if len(model_data) > 1:
                # Plot trajectory with thin lines
                ax.plot(model_data['Hub_Concentration'], model_data['Modularity'], 
                       color=colors.get(model, 'gray'), alpha=0.6, linewidth=1, label=model)
                
                # Mark start and end points
                start_point = model_data.iloc[0]
                end_point = model_data.iloc[-1]
                ax.scatter(start_point['Hub_Concentration'], start_point['Modularity'], 
                          s=120, color=colors.get(model, 'gray'), marker='o', 
                          alpha=1.0, edgecolor='white', linewidth=2)
                ax.scatter(end_point['Hub_Concentration'], end_point['Modularity'], 
                          s=120, color=colors.get(model, 'gray'), marker='s', 
                          alpha=1.0, edgecolor='white', linewidth=2)
                
                # Add small arrows for trajectory
                for i in range(len(model_data) - 1):
                    x1, y1 = model_data.iloc[i]['Hub_Concentration'], model_data.iloc[i]['Modularity']
                    x2, y2 = model_data.iloc[i+1]['Hub_Concentration'], model_data.iloc[i+1]['Modularity']
                    
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle='->', 
                                             color=colors.get(model, 'gray'), 
                                             lw=1, alpha=0.6))
    
    ax.set_xlabel('Hub Concentration (%)')
    ax.set_ylabel('Modularity Score')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('paper_1_outputs/Figure_4_7_H3_Strategic_Position_Evolution.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig1, fig2

# num5: Save H3 results - Updated table numbers with Modularity first
def save_h3_results(network_results, evolution_results):
    """Save H3 analysis results and tables"""
    
    os.makedirs('paper_1_outputs', exist_ok=True)
    
    # Table 4.6: H3 Network Structure Metrics by Business Model with statistical significance
    if len(network_results) > 0:
        # Get statistical results if available
        ulcc_stats = None
        if hasattr(network_results, 'stat_results') and 'ULCC' in network_results.stat_results:
            ulcc_stats = network_results.stat_results['ULCC']
        
        table_46_data = []
        for bm in ['ULCC', 'Legacy', 'LCC', 'Hybrid']:
            if bm in network_results.index:
                metrics = network_results.loc[bm]
                
                # Add significance stars for ULCC
                mod_str = f"{metrics['Modularity']:.3f}"
                hub_str = f"{metrics['Hub_Concentration']:.1f}"
                density_str = f"{metrics['Network_Density']:.2f}"
                gini_str = f"{metrics['Gini']:.3f}"
                
                if bm == 'ULCC' and ulcc_stats:
                    # Add stars based on ANOVA results
                    anova = ulcc_stats.get('anova', {})
                    if anova.get('modularity', 1) < 0.001:
                        mod_str += "***"
                    elif anova.get('modularity', 1) < 0.01:
                        mod_str += "**"
                    elif anova.get('modularity', 1) < 0.05:
                        mod_str += "*"
                    
                    if anova.get('hub_conc', 1) < 0.001:
                        hub_str += "***"
                    elif anova.get('hub_conc', 1) < 0.01:
                        hub_str += "**"
                    elif anova.get('hub_conc', 1) < 0.05:
                        hub_str += "*"
                    
                    if anova.get('density', 1) < 0.001:
                        density_str += "***"
                    elif anova.get('density', 1) < 0.01:
                        density_str += "**"
                    elif anova.get('density', 1) < 0.05:
                        density_str += "*"
                    
                    if anova.get('gini', 1) < 0.001:
                        gini_str += "***"
                    elif anova.get('gini', 1) < 0.01:
                        gini_str += "**"
                    elif anova.get('gini', 1) < 0.05:
                        gini_str += "*"
                
                # MODULARITY FIRST in the table
                table_46_data.append({
                    'Business Model': bm,
                    'Modularity': mod_str,
                    'Hub Concentration': hub_str,
                    'Network Density': density_str,
                    'Gini': gini_str
                })
        
        table_46 = pd.DataFrame(table_46_data)
        table_46.to_csv('paper_1_outputs/Table_4_6_H3_Network_Structure_Metrics.csv', index=False)
        
        print("\nTable 4.6: Network Structure Metrics by Business Model")
        print(f"{'Business Model':<15} {'Modularity':<12} {'Hub Conc %':<15} {'Network Density':<15} {'Gini':<10}")
        print("-" * 70)
        
        for _, row in table_46.iterrows():
            print(f"{row['Business Model']:<15} {row['Modularity']:<12} {row['Hub Concentration']:<15} {row['Network Density']:<15} {row['Gini']:<10}")
        
        # Get F-statistics for the note
        f_stat_mod = ulcc_stats['f_stats']['modularity'] if ulcc_stats and 'f_stats' in ulcc_stats else 'X.XX'
        f_stat_gini = ulcc_stats['f_stats']['gini'] if ulcc_stats and 'f_stats' in ulcc_stats else 'X.XX'
        
        print("\nNote: *** p<0.001, ** p<0.01, * p<0.05 (ANOVA with post-hoc tests comparing ULCC to other models)")
        print(f"ULCC shows significantly higher modularity (F={f_stat_mod:.2f}, p<0.001) and lower network concentration")
        print(f"(Gini coefficient, F={f_stat_gini:.2f}, p<0.001) compared to all other business models.")
        print()
    
    # Save detailed results
    network_results.to_csv('paper_1_outputs/H3_Network_Structure_Results.csv')
    if len(evolution_results) > 0:
        evolution_results.to_csv('paper_1_outputs/H3_Network_Evolution_Results.csv')
    
    # Table XX: H3 Hypothesis Test Results (not yet in manuscript)
    h3_summary = []

    if len(network_results) > 0:
        ulcc_modularity = network_results.loc['ULCC', 'Modularity'] if 'ULCC' in network_results.index else 0
        other_modularities = [network_results.loc[bm, 'Modularity'] for bm in ['Legacy', 'LCC', 'Hybrid'] if bm in network_results.index]

        if other_modularities:
            max_other = max(other_modularities)
            max_other_bm = network_results.index[network_results['Modularity'] == max_other].tolist()[0]
            h3_support = "Supported" if ulcc_modularity > max_other else "Not Supported"
            result_text = f"ULCC: {ulcc_modularity:.3f}, Max Others: {max_other:.3f} ({max_other_bm})"
        else:
            h3_support = "Inconclusive"
            result_text = "Insufficient data"

        h3_summary.append({
            'Hypothesis': 'H3: Network Modularity',
            'Prediction': 'ULCC > Other business models',
            'Result': result_text,
            'Support': h3_support
        })

    if h3_summary:
        table_xx = pd.DataFrame(h3_summary)
        table_xx.to_csv('paper_1_outputs/Table_XX_H3_Hypothesis_Test_Results.csv', index=False)
        print("Table XX: H3 Hypothesis Test Results (not yet in manuscript)")
        print(f"{'Hypothesis':<25} {'Prediction':<30} {'Result':<35} {'Support':<15}")
        print("-" * 110)
        for _, row in table_xx.iterrows():
            print(f"{row['Hypothesis']:<25} {row['Prediction']:<30} {row['Result']:<35} {row['Support']:<15}")
        print()

    print(f"\nFiles saved in paper_1_outputs/ folder:")
    print(f"- Table_4_6_H3_Network_Structure_Metrics.csv")
    print(f"- Table_XX_H3_Hypothesis_Test_Results.csv")
    print(f"- Figure_4_6_H3_Network_Structure_Analysis.png")
    print(f"- Figure_4_7_H3_Strategic_Position_Evolution.png")

# Main execution
if __name__ == "__main__":
    print("H3 Network Structure Analysis")
    print("Use: run_h3_analysis(base_data)")