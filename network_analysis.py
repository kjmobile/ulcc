# network_analysis.py
# #num2: Network Structure Analysis for H2 Testing

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from community import community_louvain

class NetworkAnalysis:
    def __init__(self, data_path, airline_classification):
        self.data_path = Path(data_path)
        self.airline_classification = airline_classification
        self.results = {}
        
    def load_network_data(self):
        """Load O&D data for network construction"""
        print("Loading O&D data for network analysis...")
        
        # Load recent years for network structure
        years = [2022, 2023, 2024]
        self.network_data = []
        
        for year in years:
            file_path = self.data_path / 'od' / f'od_{year}.parquet'
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['Year'] = year
                self.network_data.append(df)
                print(f"Loaded {year}: {len(df):,} rows")
        
        self.network_data = pd.concat(self.network_data, ignore_index=True)
        print(f"Total network data: {len(self.network_data):,} rows")
        
    def classify_carriers(self):
        """Add carrier type classification to data"""
        # Create reverse mapping
        carrier_to_type = {}
        for carrier_type, carriers in self.airline_classification.items():
            for carrier in carriers:
                carrier_to_type[carrier] = carrier_type
                
        self.network_data['Carrier_Type'] = self.network_data['Mkt'].map(carrier_to_type)
        
        # Keep only classified carriers
        self.network_data = self.network_data.dropna(subset=['Carrier_Type'])
        print(f"Data with carrier types: {len(self.network_data):,} rows")
        
    def calculate_network_metrics_by_carrier(self):
        """Calculate network metrics for each carrier"""
        print("Calculating network metrics by carrier...")
        
        carrier_metrics = []
        
        for carrier_type in self.airline_classification.keys():
            # Get carriers of this type
            carriers = self.airline_classification[carrier_type]
            
            # Aggregate data for this carrier type
            type_data = self.network_data[self.network_data['Carrier_Type'] == carrier_type]
            
            if len(type_data) == 0:
                continue
                
            # Create aggregated network
            route_traffic = type_data.groupby(['Org', 'Dst'])['Passengers'].sum().reset_index()
            
            # Create networkx graph
            G = nx.from_pandas_edgelist(route_traffic, 
                                      source='Org', 
                                      target='Dst', 
                                      edge_attr='Passengers',
                                      create_using=nx.Graph())
            
            # Calculate metrics
            metrics = self._calculate_single_network_metrics(G, carrier_type)
            carrier_metrics.append(metrics)
            
        self.carrier_metrics = pd.DataFrame(carrier_metrics)
        return self.carrier_metrics
        
    def _calculate_single_network_metrics(self, G, carrier_type):
        """Calculate metrics for a single network"""
        if len(G.nodes()) == 0:
            return {
                'Type': carrier_type,
                'Modularity': 0,
                'Density': 0,
                'Hub_Concentration_Gini': 0,
                'Top3_Hub_Share': 0,
                'Avg_Clustering': 0,
                'Avg_Path_Length': 0
            }
            
        # Modularity
        try:
            partition = community_louvain.best_partition(G)
            modularity = community_louvain.modularity(partition, G)
        except:
            modularity = 0
            
        # Density
        density = nx.density(G)
        
        # Hub concentration metrics
        degrees = dict(G.degree(weight='Passengers'))
        if len(degrees) > 0:
            degree_values = list(degrees.values())
            total_traffic = sum(degree_values)
            
            # Gini coefficient
            gini = self._calculate_gini(degree_values)
            
            # Top 3 hub share
            top3_traffic = sum(sorted(degree_values, reverse=True)[:3])
            top3_share = top3_traffic / total_traffic if total_traffic > 0 else 0
        else:
            gini = 0
            top3_share = 0
            
        # Clustering
        try:
            avg_clustering = nx.average_clustering(G)
        except:
            avg_clustering = 0
            
        # Average path length (for largest component)
        try:
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph)
        except:
            avg_path_length = 0
            
        return {
            'Type': carrier_type,
            'Modularity': modularity,
            'Density': density,
            'Hub_Concentration_Gini': gini,
            'Top3_Hub_Share': top3_share,
            'Avg_Clustering': avg_clustering,
            'Avg_Path_Length': avg_path_length
        }
        
    def _calculate_gini(self, values):
        """Calculate Gini coefficient"""
        if len(values) == 0:
            return 0
            
        values = np.array(values)
        values = np.sort(values)
        n = len(values)
        
        if np.sum(values) == 0:
            return 0
            
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
    def create_network_visualization(self):
        """Create network structure visualization"""
        print("Creating network structure visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel 1: Modularity comparison
        ax1.bar(self.carrier_metrics['Type'], self.carrier_metrics['Modularity'], 
                color=['red', 'blue', 'green', 'orange'])
        ax1.set_title('H2: Network Modularity', fontweight='bold')
        ax1.set_ylabel('Modularity Score')
        ax1.set_ylim(0, 0.6)
        
        # Add values on bars
        for i, v in enumerate(self.carrier_metrics['Modularity']):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
            
        # Panel 2: Hub concentration vs modularity scatter
        ax2.scatter(self.carrier_metrics['Top3_Hub_Share'], 
                   self.carrier_metrics['Modularity'],
                   s=100, c=['red', 'blue', 'green', 'orange'])
        
        for i, row in self.carrier_metrics.iterrows():
            ax2.annotate(row['Type'], 
                        (row['Top3_Hub_Share'], row['Modularity']),
                        xytext=(5, 5), textcoords='offset points')
                        
        ax2.set_xlabel('Top 3 Hub Concentration (%)')
        ax2.set_ylabel('Modularity Score')
        ax2.set_title('Hub Concentration vs Modularity')
        
        # Panel 3: Network efficiency
        efficiency_score = (self.carrier_metrics['Modularity'] * 0.4 + 
                          (1 - self.carrier_metrics['Density']) * 0.3 +
                          (1 - self.carrier_metrics['Hub_Concentration_Gini']) * 0.3)
        
        ax3.bar(self.carrier_metrics['Type'], efficiency_score,
                color=['red', 'blue', 'green', 'orange'])
        ax3.set_title('Composite Network Efficiency')
        ax3.set_ylabel('Efficiency Score')
        
        # Panel 4: Network architecture radar
        from math import pi
        categories = ['Hub_Independence', 'Clustering', 'Modularity', 'Efficiency']
        
        # Normalize metrics for radar chart
        metrics_normalized = self.carrier_metrics.copy()
        metrics_normalized['Hub_Independence'] = 1 - metrics_normalized['Top3_Hub_Share']
        metrics_normalized['Clustering'] = metrics_normalized['Avg_Clustering']
        metrics_normalized['Efficiency'] = 1 - metrics_normalized['Density']
        
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        
        ax4 = plt.subplot(224, projection='polar')
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, (_, row) in enumerate(self.carrier_metrics.iterrows()):
            values = [
                1 - row['Top3_Hub_Share'],
                row['Avg_Clustering'],
                row['Modularity'],
                1 - row['Density']
            ]
            values += values[:1]
            
            ax4.plot(angles, values, 'o-', linewidth=2, 
                    label=row['Type'], color=colors[i])
            ax4.fill(angles, values, alpha=0.25, color=colors[i])
            
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('Network Architecture Profiles')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))
        
        plt.tight_layout()
        plt.savefig('report/figure_4_2_h2_network_structure.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def test_h2_hypothesis(self):
        """Test H2: Network Modularity Hypothesis"""
        print("\nTesting H2: Network Modularity Hypothesis")
        print("Expected order: ULCC > LCC > Hybrid > Legacy")
        
        # Sort by modularity
        sorted_metrics = self.carrier_metrics.sort_values('Modularity', ascending=False)
        actual_order = sorted_metrics['Type'].tolist()
        
        print(f"Actual modularity order: {' > '.join(actual_order)}")
        
        # Check if ULCC has highest modularity
        ulcc_modularity = self.carrier_metrics[self.carrier_metrics['Type'] == 'ULCC']['Modularity'].iloc[0]
        max_modularity = self.carrier_metrics['Modularity'].max()
        
        h2_supported = (ulcc_modularity == max_modularity)
        
        print(f"ULCC modularity: {ulcc_modularity:.3f}")
        print(f"Maximum modularity: {max_modularity:.3f}")
        print(f"H2 Hypothesis: {'SUPPORTED' if h2_supported else 'NOT SUPPORTED'}")
        
        return {
            'hypothesis': 'H2: Network Modularity',
            'expected': 'ULCC > LCC > Hybrid > Legacy',
            'actual': ' > '.join(actual_order),
            'supported': h2_supported,
            'ulcc_modularity': ulcc_modularity,
            'metrics': self.carrier_metrics
        }
        
    def save_results(self):
        """Save network analysis results"""
        # Save metrics table
        self.carrier_metrics.to_csv('report/table_4_2_network_structure.csv', index=False)
        
        # Create summary table for manuscript
        table_data = []
        for _, row in self.carrier_metrics.iterrows():
            table_data.append({
                'Model': row['Type'],
                'Modularity': f"{row['Modularity']:.3f}",
                'Gini': f"{row['Hub_Concentration_Gini']:.3f}",
                'Top3Hub%': f"{row['Top3_Hub_Share']*100:.1f}"
            })
            
        summary_df = pd.DataFrame(table_data)
        print("\nTable 4.2: Network Structure Metrics by Business Model")
        print("-" * 67)
        print(summary_df.to_string(index=False))
        print("-" * 67)
        
        return summary_df
        
    def run_analysis(self):
        """Run complete network analysis"""
        self.load_network_data()
        self.classify_carriers()
        self.calculate_network_metrics_by_carrier()
        
        # Test hypothesis
        h2_results = self.test_h2_hypothesis()
        
        # Create visualizations
        self.create_network_visualization()
        
        # Save results
        summary_table = self.save_results()
        
        self.results = {
            'hypothesis_test': h2_results,
            'metrics': self.carrier_metrics,
            'summary_table': summary_table
        }
        
        return self.results
