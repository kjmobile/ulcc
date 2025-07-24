# market_behavior_analysis.py
# #num3: Market Behavior Analysis for H1 Testing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class MarketBehaviorAnalysis:
    def __init__(self, data_path, airline_classification):
        self.data_path = Path(data_path)
        self.airline_classification = airline_classification
        self.results = {}
        
    def load_market_data(self):
        """Load O&D data for market behavior analysis"""
        print("Loading O&D data for market behavior analysis...")
        
        self.market_data = []
        years = range(2014, 2025)
        
        for year in years:
            file_path = self.data_path / 'od' / f'od_{year}.parquet'
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['Year'] = year
                self.market_data.append(df)
                print(f"Loaded {year}: {len(df):,} rows")
                
        self.market_data = pd.concat(self.market_data, ignore_index=True)
        print(f"Total market data: {len(self.market_data):,} rows")
        
    def classify_carriers(self):
        """Add carrier type classification"""
        carrier_to_type = {}
        for carrier_type, carriers in self.airline_classification.items():
            for carrier in carriers:
                carrier_to_type[carrier] = carrier_type
                
        self.market_data['Carrier_Type'] = self.market_data['Mkt'].map(carrier_to_type)
        self.market_data = self.market_data.dropna(subset=['Carrier_Type'])
        print(f"Classified data: {len(self.market_data):,} rows")
        
    def calculate_route_dynamics(self):
        """Calculate route entry/exit patterns by carrier type and year"""
        print("Calculating route dynamics...")
        
        # Create route presence matrix
        route_presence = self.market_data.groupby(['Carrier_Type', 'Mkt', 'Org', 'Dst', 'Year']).size().reset_index()
        route_presence['Route'] = route_presence['Org'] + '-' + route_presence['Dst']
        route_presence['Present'] = 1
        
        # Pivot to get presence by year
        presence_matrix = route_presence.pivot_table(
            index=['Carrier_Type', 'Mkt', 'Route'],
            columns='Year',
            values='Present',
            fill_value=0
        )
        
        # Calculate entry/exit events
        dynamics_results = []
        
        for carrier_type in self.airline_classification.keys():
            type_data = presence_matrix[presence_matrix.index.get_level_values(0) == carrier_type]
            
            if len(type_data) == 0:
                continue
                
            yearly_dynamics = self._calculate_yearly_dynamics(type_data, carrier_type)
            dynamics_results.extend(yearly_dynamics)
            
        self.route_dynamics = pd.DataFrame(dynamics_results)
        return self.route_dynamics
        
    def _calculate_yearly_dynamics(self, presence_data, carrier_type):
        """Calculate yearly entry/exit rates for a carrier type"""
        results = []
        years = presence_data.columns
        
        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            
            if prev_year not in presence_data.columns or curr_year not in presence_data.columns:
                continue
                
            # Routes in previous year
            prev_routes = set(presence_data[presence_data[prev_year] == 1].index)
            
            # Routes in current year
            curr_routes = set(presence_data[presence_data[curr_year] == 1].index)
            
            # Calculate metrics
            if len(prev_routes) > 0:
                # Entry: routes in current but not previous
                new_routes = curr_routes - prev_routes
                entry_rate = len(new_routes) / len(prev_routes)
                
                # Exit: routes in previous but not current
                exited_routes = prev_routes - curr_routes
                exit_rate = len(exited_routes) / len(prev_routes)
                
                # Retention: routes in both years
                continuing_routes = curr_routes & prev_routes
                retention_rate = len(continuing_routes) / len(prev_routes)
                
                # Churn: total change
                churn_rate = entry_rate + exit_rate
                
                # Net growth
                net_growth = entry_rate - exit_rate
                
                results.append({
                    'Carrier_Type': carrier_type,
                    'Year': curr_year,
                    'Entry_Rate': entry_rate,
                    'Exit_Rate': exit_rate,
                    'Retention_Rate': retention_rate,
                    'Route_Churn': churn_rate,
                    'Net_Growth': net_growth,
                    'Total_Routes_Prev': len(prev_routes),
                    'Total_Routes_Curr': len(curr_routes),
                    'New_Routes': len(new_routes),
                    'Exited_Routes': len(exited_routes)
                })
                
        return results
        
    def analyze_covid_impact(self):
        """Analyze market behavior during COVID period"""
        print("Analyzing COVID impact on market behavior...")
        
        # Define periods
        pre_covid = self.route_dynamics[self.route_dynamics['Year'].isin([2017, 2018, 2019])]
        covid_period = self.route_dynamics[self.route_dynamics['Year'].isin([2020, 2021])]
        post_covid = self.route_dynamics[self.route_dynamics['Year'].isin([2022, 2023, 2024])]
        
        # Calculate period averages
        periods = {
            'Pre-COVID': pre_covid,
            'COVID': covid_period,
            'Post-COVID': post_covid
        }
        
        covid_analysis = {}
        for period_name, period_data in periods.items():
            if len(period_data) > 0:
                covid_analysis[period_name] = period_data.groupby('Carrier_Type')[
                    ['Entry_Rate', 'Exit_Rate', 'Route_Churn', 'Net_Growth']
                ].mean()
                
        self.covid_analysis = covid_analysis
        return covid_analysis
        
    def create_market_behavior_visualization(self):
        """Create comprehensive market behavior visualization"""
        print("Creating market behavior visualization...")
        
        # Calculate average metrics by carrier type
        avg_metrics = self.route_dynamics.groupby('Carrier_Type')[
            ['Entry_Rate', 'Exit_Rate', 'Route_Churn', 'Net_Growth', 'Retention_Rate']
        ].mean()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel 1: Market Dynamism Index (Route Churn)
        colors = ['red', 'orange', 'green', 'blue']
        churn_data = avg_metrics['Route_Churn'].sort_values(ascending=False)
        
        bars1 = ax1.bar(range(len(churn_data)), churn_data.values, color=colors)
        ax1.set_xticks(range(len(churn_data)))
        ax1.set_xticklabels(churn_data.index, rotation=45)
        ax1.set_ylabel('Route Churn Rate (%)')
        ax1.set_title('Market Dynamism Index', fontweight='bold')
        ax1.set_ylim(0, 0.3)
        
        # Add value labels
        for i, v in enumerate(churn_data.values):
            ax1.text(i, v + 0.005, f'{v:.1%}', ha='center', fontweight='bold')
            
        # Panel 2: Entry vs Exit Dynamics
        ax2.scatter(avg_metrics['Entry_Rate'], avg_metrics['Exit_Rate'], 
                   s=200, c=colors, alpha=0.7)
        
        for carrier_type in avg_metrics.index:
            ax2.annotate(carrier_type, 
                        (avg_metrics.loc[carrier_type, 'Entry_Rate'],
                         avg_metrics.loc[carrier_type, 'Exit_Rate']),
                        xytext=(5, 5), textcoords='offset points')
                        
        ax2.set_xlabel('Market Entry Rate (%)')
        ax2.set_ylabel('Market Exit Rate (%)')
        ax2.set_title('Entry vs Exit Dynamics')
        ax2.plot([0, 0.3], [0, 0.3], 'k--', alpha=0.5, label='Equal Entry/Exit')
        ax2.legend()
        
        # Panel 3: Net Market Growth
        net_growth = avg_metrics['Net_Growth'].sort_values(ascending=False)
        colors_sorted = [colors[list(avg_metrics.index).index(carrier)] for carrier in net_growth.index]
        
        bars3 = ax3.bar(range(len(net_growth)), net_growth.values, color=colors_sorted)
        ax3.set_xticks(range(len(net_growth)))
        ax3.set_xticklabels(net_growth.index, rotation=45)
        ax3.set_ylabel('Net Growth Rate (%)')
        ax3.set_title('Net Market Growth')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(net_growth.values):
            ax3.text(i, v + 0.005 if v >= 0 else v - 0.01, f'{v:+.1%}', 
                    ha='center', fontweight='bold')
            
        # Panel 4: Time series of market churn by carrier type
        time_series = self.route_dynamics.groupby(['Year', 'Carrier_Type'])['Route_Churn'].mean().unstack()
        
        for i, carrier_type in enumerate(time_series.columns):
            ax4.plot(time_series.index, time_series[carrier_type], 
                    color=colors[i], marker='o', label=carrier_type, linewidth=2)
            
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Route Churn Rate')
        ax4.set_title('Route Churn Rate by Airline Type')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Mark COVID period
        ax4.axvspan(2020, 2021, alpha=0.2, color='red', label='COVID-19')
        
        plt.tight_layout()
        plt.savefig('report/figure_4_1_h1_market_behavior.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def test_h1_hypothesis(self):
        """Test H1: Market Entry and Exit Hypothesis"""
        print("\nTesting H1: Market Entry and Exit Hypothesis")
        print("Expected order: ULCC > LCC > Hybrid > Legacy (market dynamism)")
        
        avg_metrics = self.route_dynamics.groupby('Carrier_Type')[
            ['Entry_Rate', 'Exit_Rate', 'Route_Churn']
        ].mean()
        
        # Sort by route churn (total dynamism)
        churn_ranking = avg_metrics.sort_values('Route_Churn', ascending=False)
        actual_order = churn_ranking.index.tolist()
        
        print(f"Actual dynamism order: {' > '.join(actual_order)}")
        
        # Check specific metrics
        ulcc_rank = actual_order.index('ULCC') + 1 if 'ULCC' in actual_order else None
        
        print("\nDetailed Results:")
        for carrier_type in ['ULCC', 'LCC', 'Hybrid', 'Legacy']:
            if carrier_type in avg_metrics.index:
                metrics = avg_metrics.loc[carrier_type]
                rank = actual_order.index(carrier_type) + 1
                print(f"{carrier_type}:")
                print(f"  Entry: {metrics['Entry_Rate']:.1%}")
                print(f"  Exit: {metrics['Exit_Rate']:.1%}")
                print(f"  Churn: {metrics['Route_Churn']:.1%}")
                print(f"  Rank: {rank}")
                
        # Hypothesis evaluation
        expected_order = ['ULCC', 'LCC', 'Hybrid', 'Legacy']
        ulcc_top2 = ulcc_rank <= 2 if ulcc_rank else False
        
        return {
            'hypothesis': 'H1: Market Entry and Exit',
            'expected': 'ULCC > LCC > Hybrid > Legacy',
            'actual': ' > '.join(actual_order),
            'ulcc_rank': ulcc_rank,
            'supported': ulcc_top2,
            'metrics': avg_metrics
        }
        
    def save_results(self):
        """Save market behavior analysis results"""
        # Save detailed dynamics
        self.route_dynamics.to_csv('report/airline_market_behavior_detailed.csv', index=False)
        
        # Save summary metrics
        avg_metrics = self.route_dynamics.groupby('Carrier_Type')[
            ['Entry_Rate', 'Exit_Rate', 'Route_Churn', 'Net_Growth', 'Retention_Rate']
        ].mean()
        
        # Create manuscript table
        table_data = []
        for carrier_type in ['Hybrid', 'LCC', 'Legacy', 'ULCC']:
            if carrier_type in avg_metrics.index:
                metrics = avg_metrics.loc[carrier_type]
                table_data.append({
                    'Business Model': carrier_type,
                    'Entry%': f"{metrics['Entry_Rate']:.1%}",
                    'Exit%': f"{metrics['Exit_Rate']:.1%}",
                    'Churn%': f"{metrics['Route_Churn']:.1%}",
                    'Net%': f"{metrics['Net_Growth']:.1%}",
                    'Persist%': f"{metrics['Retention_Rate']:.1%}"
                })
                
        summary_df = pd.DataFrame(table_data)
        summary_df.to_csv('report/table_4_1_market_dynamics.csv', index=False)
        
        print("\nTable 4.1: Market Behavior Patterns by Business Model")
        print("-" * 75)
        print(summary_df.to_string(index=False))
        print("-" * 75)
        
        return summary_df
        
    def run_analysis(self):
        """Run complete market behavior analysis"""
        self.load_market_data()
        self.classify_carriers()
        self.calculate_route_dynamics()
        
        # Analyze COVID impact
        covid_analysis = self.analyze_covid_impact()
        
        # Test hypothesis
        h1_results = self.test_h1_hypothesis()
        
        # Create visualizations
        self.create_market_behavior_visualization()
        
        # Save results
        summary_table = self.save_results()
        
        self.results = {
            'hypothesis_test': h1_results,
            'route_dynamics': self.route_dynamics,
            'covid_analysis': covid_analysis,
            'summary_table': summary_table
        }
        
        return self.results
