#num1: Import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

#num2: Network structure analysis (H2)
def analyze_network_structure_h2(base_data):
    """Analyze network modularity by business model"""
    
    print("\n=== H2: NETWORK STRUCTURE ANALYSIS ===")
    
    od_years = base_data['od_years']
    classification_map = base_data['classification_map']
    
    network_results = {}
    
    # Use recent years data
    recent_years = [2022, 2023, 2024]
    all_recent_data = []
    
    for year in recent_years:
        if year in od_years:
            df = od_years[year].copy()
            df['Business_Model'] = df['Mkt'].map(classification_map)
            all_recent_data.append(df)
    
    if not all_recent_data:
        print("No recent data available")
        return pd.DataFrame()
    
    combined_recent = pd.concat(all_recent_data, ignore_index=True)
    combined_recent = combined_recent.dropna(subset=['Business_Model'])
    
    # Filter to main carrier types
    valid_types = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    combined_recent = combined_recent[combined_recent['Business_Model'].isin(valid_types)]
    
    route_model_data = combined_recent.groupby(['Org', 'Dst', 'Business_Model']).agg({
        'Passengers': 'sum'
    }).reset_index()
    
    route_model_data = route_model_data[route_model_data['Passengers'] >= 1000]
    
    for model in valid_types:
        model_routes = route_model_data[route_model_data['Business_Model'] == model]
        
        if len(model_routes) < 3:
            continue
        
        # Create network graph
        G = nx.from_pandas_edgelist(model_routes, 
                                   source='Org', 
                                   target='Dst', 
                                   edge_attr='Passengers',
                                   create_using=nx.Graph())
        
        # Calculate modularity
        try:
            communities = nx.community.louvain_communities(G, weight='Passengers')
            modularity = nx.community.modularity(G, communities, weight='Passengers')
        except:
            modularity = 0
        
        # Hub concentration calculation
        node_traffic = {}
        for node in G.nodes():
            node_edges = [(node, neighbor, G[node][neighbor]['Passengers']) 
                         for neighbor in G.neighbors(node)]
            total_traffic = sum([edge[2] for edge in node_edges])
            node_traffic[node] = total_traffic
        
        if node_traffic:
            traffic_values = np.array(list(node_traffic.values()))
            total_traffic = traffic_values.sum()
            top3_traffic = np.partition(traffic_values, -3)[-3:].sum()
            top3_share = (top3_traffic / total_traffic * 100) if total_traffic > 0 else 0
        else:
            top3_share = 0
        
        # Gini coefficient calculation
        route_volumes = model_routes['Passengers'].values
        if len(route_volumes) > 1:
            sorted_volumes = np.sort(route_volumes)
            n = len(sorted_volumes)
            cumsum = np.cumsum(sorted_volumes)
            gini = (n + 1 - 2 * np.sum((n + 1 - np.arange(1, n + 1)) * sorted_volumes) / np.sum(sorted_volumes)) / n
        else:
            gini = 0
        
        network_results[model] = {
            'Modularity': modularity,
            'Gini': gini,
            'Top3Hub%': top3_share,
            'Routes': len(model_routes),
            'Airports': len(G.nodes()),
            'Density': nx.density(G)
        }
    
    network_df = pd.DataFrame(network_results).T
    print("\nNetwork Structure Results:")
    print(network_df.round(3))
    
    return network_df

#num3: Temporal network evolution analysis
def analyze_network_evolution_h2(base_data):
    """Analyze network structure evolution over time"""
    
    print("\n=== H2 TEMPORAL: NETWORK EVOLUTION ANALYSIS ===")
    
    od_years = base_data['od_years']
    classification_map = base_data['classification_map']
    
    evolution_data = []
    
    for year in range(2014, 2025):
        if year not in od_years:
            continue
            
        year_data = od_years[year].copy()
        year_data['Business_Model'] = year_data['Mkt'].map(classification_map)
        year_data = year_data.dropna(subset=['Business_Model'])
        
        valid_types = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
        year_data = year_data[year_data['Business_Model'].isin(valid_types)]
        
        route_model_data = year_data.groupby(['Org', 'Dst', 'Business_Model']).agg({
            'Passengers': 'sum'
        }).reset_index()
        
        route_model_data = route_model_data[route_model_data['Passengers'] >= 1000]
        
        for model in valid_types:
            model_routes = route_model_data[route_model_data['Business_Model'] == model]
            
            if len(model_routes) < 3:
                continue
            
            G = nx.from_pandas_edgelist(model_routes, 
                                       source='Org', 
                                       target='Dst', 
                                       edge_attr='Passengers',
                                       create_using=nx.Graph())
            
            try:
                communities = nx.community.louvain_communities(G, weight='Passengers')
                modularity = nx.community.modularity(G, communities, weight='Passengers')
            except:
                modularity = 0
            
            # Hub concentration
            node_traffic = {}
            for node in G.nodes():
                node_edges = [(node, neighbor, G[node][neighbor]['Passengers']) 
                             for neighbor in G.neighbors(node)]
                total_traffic = sum([edge[2] for edge in node_edges])
                node_traffic[node] = total_traffic
            
            if node_traffic:
                traffic_values = np.array(list(node_traffic.values()))
                total_traffic = traffic_values.sum()
                top3_traffic = np.partition(traffic_values, -3)[-3:].sum()
                top3_share = (top3_traffic / total_traffic * 100) if total_traffic > 0 else 0
            else:
                top3_share = 0
            
            evolution_data.append({
                'Year': year,
                'Business_Model': model,
                'Modularity': modularity,
                'Top3Hub%': top3_share,
                'Routes': len(model_routes)
            })
    
    evolution_df = pd.DataFrame(evolution_data)
    print(f"\nNetwork evolution data: {len(evolution_df)} observations")
    
    return evolution_df

#num4: Create H2 visualization
def create_h2_figure(network_df, evolution_df):
    """Create Figure 4.3: Network Structure Analysis"""
    
    os.makedirs('figures', exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Import colors from basecode
    from basecode import CARRIER_COLORS as colors
    
    # Main network structure analysis (top row)
    ax1 = plt.subplot(2, 3, 1)
    mod_data = network_df['Modularity'].sort_values(ascending=False)
    for i, (carrier, value) in enumerate(mod_data.items()):
        ax1.bar(i, value, color=colors[carrier], alpha=0.8, width=0.6, 
               edgecolor='black', linewidth=0.5)
        ax1.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_title('Panel A: Network Modularity', fontweight='bold', pad=15)
    ax1.set_ylabel('Modularity Score')
    ax1.set_xticks(range(len(mod_data)))
    ax1.set_xticklabels(mod_data.index)
    ax1.set_ylim(0, max(mod_data.values) * 1.15)
    
    ax2 = plt.subplot(2, 3, 2)
    for carrier, row in network_df.iterrows():
        ax2.scatter(row['Top3Hub%'], row['Modularity'], s=150, 
                   color=colors[carrier], alpha=0.8, edgecolors='black', linewidth=0.5)
        ax2.annotate(carrier, (row['Top3Hub%'], row['Modularity']), 
                    xytext=(3, 3), textcoords='offset points', fontsize=9)
    
    ax2.set_title('Panel B: Hub Concentration vs Modularity', fontweight='bold', pad=15)
    ax2.set_xlabel('Top 3 Hub Concentration (%)')
    ax2.set_ylabel('Modularity Score')
    
    ax3 = plt.subplot(2, 3, 3)
    gini_data = network_df['Gini'].sort_values(ascending=False)
    for i, (carrier, value) in enumerate(gini_data.items()):
        ax3.bar(i, value, color=colors[carrier], alpha=0.8, width=0.6, 
               edgecolor='black', linewidth=0.5)
        ax3.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax3.set_title('Panel C: Network Inequality', fontweight='bold', pad=15)
    ax3.set_ylabel('Gini Coefficient')
    ax3.set_xticks(range(len(gini_data)))
    ax3.set_xticklabels(gini_data.index)
    ax3.set_ylim(0, max(gini_data.values) * 1.15)
    
    # Network evolution analysis (bottom row)
    if len(evolution_df) > 0:
        ax4 = plt.subplot(2, 3, 4)
        for model in ['Legacy', 'ULCC', 'LCC', 'Hybrid']:
            model_data = evolution_df[evolution_df['Business_Model'] == model]
            if len(model_data) > 0:
                ax4.plot(model_data['Year'], model_data['Modularity'], 
                        marker='o', label=model, color=colors[model], linewidth=2, markersize=4)
        
        ax4.set_title('Panel D: Modularity Evolution', fontweight='bold', pad=15)
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Modularity Score')
        ax4.legend(frameon=False)
        
        ax5 = plt.subplot(2, 3, 5)
        for model in ['Legacy', 'ULCC', 'LCC', 'Hybrid']:
            model_data = evolution_df[evolution_df['Business_Model'] == model]
            if len(model_data) > 0:
                ax5.plot(model_data['Year'], model_data['Top3Hub%'], 
                        marker='s', label=model, color=colors[model], linewidth=2, markersize=4)
        
        ax5.set_title('Panel E: Hub Concentration Evolution', fontweight='bold', pad=15)
        ax5.set_xlabel('Year')
        ax5.set_ylabel('Top 3 Hub Concentration (%)')
        ax5.legend(frameon=False)
        
        # Strategic trajectory plot
        ax6 = plt.subplot(2, 3, 6)
        for model in ['Legacy', 'ULCC', 'LCC', 'Hybrid']:
            model_data = evolution_df[evolution_df['Business_Model'] == model]
            if len(model_data) > 1:
                ax6.plot(model_data['Top3Hub%'], model_data['Modularity'], 
                        marker='o', label=model, color=colors[model], alpha=0.7, linewidth=1.5)
                
                # Mark start and end points
                start_point = model_data.iloc[0]
                end_point = model_data.iloc[-1]
                ax6.scatter(start_point['Top3Hub%'], start_point['Modularity'], 
                           s=100, color=colors[model], marker='o', alpha=1.0, edgecolor='white')
                ax6.scatter(end_point['Top3Hub%'], end_point['Modularity'], 
                           s=100, color=colors[model], marker='s', alpha=1.0, edgecolor='white')
        
        ax6.set_title('Panel F: Strategic Position Evolution', fontweight='bold', pad=15)
        ax6.set_xlabel('Top 3 Hub Concentration (%)')
        ax6.set_ylabel('Modularity Score')
        ax6.legend(frameon=False)
    
    #plt.suptitle('Figure 4.3: Network Structure Analysis', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save in both formats
    plt.savefig('figures/Figure_4_3_Network_Structure.png', dpi=300, bbox_inches='tight', facecolor='white')
    #plt.savefig('figures/Figure_4_3_Network_Structure.eps', format='eps', bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

#num5: Main H2 analysis function
def run_h2_analysis(base_data):
    """Run complete H2 analysis"""
    
    print("RUNNING H2: NETWORK STRUCTURE ANALYSIS")
    print("=" * 50)
    
    # Main network structure analysis
    network_results = analyze_network_structure_h2(base_data)
    
    # Temporal evolution analysis
    evolution_results = analyze_network_evolution_h2(base_data)
    
    # Create visualization
    fig = create_h2_figure(network_results, evolution_results)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    network_results.to_csv('results/H2_Network_Structure_Results.csv')
    evolution_results.to_csv('results/H2_Network_Evolution_Results.csv')
    
    print("\nH2 Analysis Complete!")
    print("Results saved in 'results/' directory")
    print("Figures saved in 'figures/' directory")
    
    return {
        'network_results': network_results,
        'evolution_results': evolution_results,
        'figure': fig
    }

if __name__ == "__main__":
    from basecode import prepare_base_data
    base_data = prepare_base_data()
    if base_data:
        h2_results = run_h2_analysis(base_data)