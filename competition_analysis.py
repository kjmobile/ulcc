# competition_analysis.py
# #num4: Competition Impact Analysis for H3 Testing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CompetitionAnalysis:
    def __init__(self, data_path, airline_classification):
        self.data_path = Path(data_path)
        self.airline_classification = airline_classification
        self.results = {}
        
    def load_competition_data(self):
        """Load data for competition analysis"""
        print("Loading competition data...")
        
        # Load O&D data for competition analysis
        self.od_data = []
        years = [2019, 2022, 2023, 2024]  # Focus on recent years
        
        for year in years:
            file_path = self.data_path / 'od' / f'od_{year}.parquet'
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['Year'] = year
                self.od_data.append(df)
                print(f"Loaded OD {year}: {len(df):,} rows")
                
        self.od_data = pd.concat(self.od_data, ignore_index=True)
        
        # Load T-100 data for load factor analysis
        self.t100_data = []
        for year in years:
            file_path = self.data_path / 't_100' / f't_100_{year}.parquet'
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['Year'] = year
                self.t100_data.append(df)
                print(f"Loaded T100 {year}: {len(df):,} rows")
                
        if self.t100_data:
            self.t100_data = pd.concat(self.t100_data, ignore_index=True)
        
    def classify_carriers(self):
        """Add carrier type classification"""
        carrier_to_type = {}
        for carrier_type, carriers in self.airline_classification.items():
            for carrier in carriers:
                carrier_to_type[carrier] = carrier_type
                
        self.od_data['Carrier_Type'] = self.od_data['Mkt'].map(carrier_to_type)
        self.od_data = self.od_data.dropna(subset=['Carrier_Type'])
        
        if hasattr(self, 't100_data') and len(self.t100_data) > 0:
            self.t100_data['Carrier_Type'] = self.t100_data['Mkt Al'].map(carrier_to_type)
            self.t100_data = self.t100_data.dropna(subset=['Carrier_Type'])
        
    def calculate_route_competition_metrics(self):
        """Calculate competition metrics for each route"""
        print("Calculating route-level competition metrics...")
        
        # Aggregate by route, year, month
        route_data = self.od_data.groupby(['Year', 'Month', 'Org', 'Dst', 'Carrier_Type'])['Passengers'].sum().reset_index()
        route_data['Route'] = route_data['Org'] + '-' + route_data['Dst']
        
        # Calculate market shares and concentration
        competition_metrics = []
        
        for (year, month, route), group in route_data.groupby(['Year', 'Month', 'Route']):
            total_pax = group['Passengers'].sum()
            
            if total_pax <= 0:
                continue
                
            # Calculate market shares by carrier type
            shares = group.groupby('Carrier_Type')['Passengers'].sum() / total_pax
            
            # HHI calculation
            carrier_shares = group.groupby(['Carrier_Type', 'Org', 'Dst'])['Passengers'].sum() / total_pax
            hhi = (carrier_shares ** 2).sum()
            
            # Presence indicators
            present_types = set(group['Carrier_Type'].unique())
            
            metrics = {
                'Year': year,
                'Month': month,
                'Route': route,
                'Org': group['Org'].iloc[0],
                'Dst': group['Dst'].iloc[0],
                'Total_Passengers': total_pax,
                'HHI': hhi,
                'Num_Carriers': len(group),
                'Has_ULCC': 'ULCC' in present_types,
                'Has_LCC': 'LCC' in present_types,
                'Has_Hybrid': 'Hybrid' in present_types,
                'Has_Legacy': 'Legacy' in present_types,
                'ULCC_Share': shares.get('ULCC', 0),
                'LCC_Share': shares.get('LCC', 0),
                'Hybrid_Share': shares.get('Hybrid', 0),
                'Legacy_Share': shares.get('Legacy', 0)
            }
            
            competition_metrics.append(metrics)
            
        self.competition_data = pd.DataFrame(competition_metrics)
        print(f"Competition metrics calculated for {len(self.competition_data):,} route-months")
        
        return self.competition_data
        
    def analyze_h3a_market_concentration(self):
        """H3a: Analyze ULCC impact on market concentration"""
        print("\nAnalyzing H3a: ULCC Impact on Market Concentration")
        
        # Compare HHI with and without ULCC presence
        ulcc_routes = self.competition_data[self.competition_data['Has_ULCC'] == True]
        no_ulcc_routes = self.competition_data[self.competition_data['Has_ULCC'] == False]
        
        hhi_with_ulcc = ulcc_routes['HHI'].mean()
        hhi_without_ulcc = no_ulcc_routes['HHI'].mean()
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(ulcc_routes['HHI'], no_ulcc_routes['HHI'])
        
        # Effect size
        effect_size = hhi_without_ulcc - hhi_with_ulcc
        percentage_reduction = (effect_size / hhi_without_ulcc) * 100
        
        print(f"Routes without ULCC: HHI = {hhi_without_ulcc:.3f}")
        print(f"Routes with ULCC: HHI = {hhi_with_ulcc:.3f}")
        print(f"HHI reduction: {effect_size:.3f} ({percentage_reduction:.1f}%)")
        print(f"T-test: t = {t_stat:.3f}, p = {p_value:.4f}")
        
        h3a_supported = (effect_size > 0 and p_value < 0.05)
        
        return {
            'hhi_without_ulcc': hhi_without_ulcc,
            'hhi_with_ulcc': hhi_with_ulcc,
            'effect_size': effect_size,
            'percentage_reduction': percentage_reduction,
            't_stat': t_stat,
            'p_value': p_value,
            'supported': h3a_supported
        }
        
    def analyze_h3b_load_factor_impact(self):
        """H3b: Analyze ULCC impact on incumbent load factors"""
        print("\nAnalyzing H3b: ULCC Impact on Incumbent Load Factors")
        
        if not hasattr(self, 't100_data') or len(self.t100_data) == 0:
            print("T-100 data not available for load factor analysis")
            return {'supported': False, 'reason': 'No T-100 data available'}
            
        # Calculate route-level load factors
        t100_routes = self.t100_data.groupby(['Year', 'Month', 'Orig', 'Dest', 'Carrier_Type']).agg({
            'Load Factor': 'mean',
            'Onboards': 'sum',
            'ASMs': 'sum'
        }).reset_index()
        
        t100_routes['Route'] = t100_routes['Orig'] + '-' + t100_routes['Dest']
        
        # Merge with competition data
        merged_data = t100_routes.merge(
            self.competition_data[['Year', 'Month', 'Route', 'Has_ULCC', 'ULCC_Share']],
            on=['Year', 'Month', 'Route'],
            how='inner'
        )
        
        # Focus on incumbent carriers (Legacy and Hybrid)
        incumbent_data = merged_data[merged_data['Carrier_Type'].isin(['Legacy', 'Hybrid'])]
        
        if len(incumbent_data) == 0:
            print("No incumbent data available for analysis")
            return {'supported': False, 'reason': 'No incumbent data'}
            
        # Compare load factors with and without ULCC
        with_ulcc = incumbent_data[incumbent_data['Has_ULCC'] == True]['Load Factor']
        without_ulcc = incumbent_data[incumbent_data['Has_ULCC'] == False]['Load Factor']
        
        if len(with_ulcc) == 0 or len(without_ulcc) == 0:
            print("Insufficient data for load factor comparison")
            return {'supported': False, 'reason': 'Insufficient data'}
            
        lf_with_ulcc = with_ulcc.mean()
        lf_without_ulcc = without_ulcc.mean()
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(with_ulcc, without_ulcc)
        
        effect_size = lf_with_ulcc - lf_without_ulcc
        
        print(f"Incumbent LF without ULCC: {lf_without_ulcc:.1f}%")
        print(f"Incumbent LF with ULCC: {lf_with_ulcc:.1f}%")
        print(f"Effect: {effect_size:+.1f}pp")
        print(f"T-test: t = {t_stat:.3f}, p = {p_value:.4f}")
        
        # H3b expects negative effect (lower LF with ULCC)
        h3b_supported = (effect_size < 0 and p_value < 0.05)
        
        # Check for Load Factor Paradox
        paradox = (effect_size > 0 and p_value < 0.05)
        
        return {
            'lf_without_ulcc': lf_without_ulcc,
            'lf_with_ulcc': lf_with_ulcc,
            'effect_size': effect_size,
            't_stat': t_stat,
            'p_value': p_value,
            'supported': h3b_supported,
            'paradox_detected': paradox
        }
        
    def create_competition_visualization(self):
        """Create competition impact visualization"""
        print("Creating competition impact visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel 1: HHI comparison (H3a)
        ulcc_hhi = self.competition_data[self.competition_data['Has_ULCC'] == True]['HHI']
        no_ulcc_hhi = self.competition_data[self.competition_data['Has_ULCC'] == False]['HHI']
        
        ax1.hist(no_ulcc_hhi, bins=30, alpha=0.7, label='No ULCC', color='blue', density=True)
        ax1.hist(ulcc_hhi, bins=30, alpha=0.7, label='ULCC Present', color='red', density=True)
        ax1.set_xlabel('HHI (Market Concentration)')
        ax1.set_ylabel('Density')
        ax1.set_title('H3a: Market Concentration')
        ax1.legend()
        ax1.axvline(no_ulcc_hhi.mean(), color='blue', linestyle='--', alpha=0.8, 
                   label=f'No ULCC avg: {no_ulcc_hhi.mean():.3f}')
        ax1.axvline(ulcc_hhi.mean(), color='red', linestyle='--', alpha=0.8,
                   label=f'ULCC avg: {ulcc_hhi.mean():.3f}')
        
        # Panel 2: Market share distribution
        shares = ['ULCC_Share', 'LCC_Share', 'Hybrid_Share', 'Legacy_Share']
        colors = ['red', 'orange', 'purple', 'blue']
        avg_shares = [self.competition_data[share].mean() for share in shares]
        
        ax2.pie(avg_shares, labels=['ULCC', 'LCC', 'Hybrid', 'Legacy'], colors=colors, autopct='%1.1f%%')
        ax2.set_title('Average Market Share Distribution')
        
        # Panel 3: Route count by carrier presence
        presence_counts = {
            'ULCC Only': len(self.competition_data[
                (self.competition_data['Has_ULCC']) & 
                (~self.competition_data['Has_LCC']) & 
                (~self.competition_data['Has_Hybrid']) & 
                (~self.competition_data['Has_Legacy'])
            ]),
            'LCC Only': len(self.competition_data[
                (~self.competition_data['Has_ULCC']) & 
                (self.competition_data['Has_LCC']) & 
                (~self.competition_data['Has_Hybrid']) & 
                (~self.competition_data['Has_Legacy'])
            ]),
            'Legacy Only': len(self.competition_data[
                (~self.competition_data['Has_ULCC']) & 
                (~self.competition_data['Has_LCC']) & 
                (~self.competition_data['Has_Hybrid']) & 
                (self.competition_data['Has_Legacy'])
            ]),
            'Mixed Competition': len(self.competition_data) - sum([
                len(self.competition_data[
                    (self.competition_data['Has_ULCC']) & 
                    (~self.competition_data['Has_LCC']) & 
                    (~self.competition_data['Has_Hybrid']) & 
                    (~self.competition_data['Has_Legacy'])
                ]),
                len(self.competition_data[
                    (~self.competition_data['Has_ULCC']) & 
                    (self.competition_data['Has_LCC']) & 
                    (~self.competition_data['Has_Hybrid']) & 
                    (~self.competition_data['Has_Legacy'])
                ]),
                len(self.competition_data[
                    (~self.competition_data['Has_ULCC']) & 
                    (~self.competition_data['Has_LCC']) & 
                    (~self.competition_data['Has_Hybrid']) & 
                    (self.competition_data['Has_Legacy'])
                ])
            ])
        }
        
        ax3.bar(presence_counts.keys(), presence_counts.values(), 
               color=['red', 'orange', 'blue', 'gray'])
        ax3.set_ylabel('Number of Route-Months')
        ax3.set_title('Route Competition Patterns')
        ax3.tick_params(axis='x', rotation=45)
        
        # Panel 4: ULCC penetration by market size
        # Create market size bins
        self.competition_data['Market_Size_Bin'] = pd.qcut(
            self.competition_data['Total_Passengers'], 
            q=5, 
            labels=['Smallest', 'Small', 'Medium', 'Large', 'Largest']
        )
        
        ulcc_penetration = self.competition_data.groupby('Market_Size_Bin')['Has_ULCC'].mean()
        
        ax4.bar(range(len(ulcc_penetration)), ulcc_penetration.values, color='red', alpha=0.7)
        ax4.set_xticks(range(len(ulcc_penetration)))
        ax4.set_xticklabels(ulcc_penetration.index)
        ax4.set_ylabel('ULCC Presence Rate')
        ax4.set_title('ULCC Penetration by Market Size')
        
        # Add percentage labels
        for i, v in enumerate(ulcc_penetration.values):
            ax4.text(i, v + 0.01, f'{v:.1%}', ha='center')
            
        plt.tight_layout()
        plt.savefig('report/figure_4_3_h3_competition_impact.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def test_h3_hypotheses(self):
        """Test all H3 hypotheses"""
        print("\nTesting H3: Competition Impact Hypotheses")
        
        # Test H3a
        h3a_results = self.analyze_h3a_market_concentration()
        
        # Test H3b
        h3b_results = self.analyze_h3b_load_factor_impact()
        
        print(f"\nH3a (Market Concentration): {'SUPPORTED' if h3a_results['supported'] else 'NOT SUPPORTED'}")
        print(f"H3b (Load Factor Impact): {'SUPPORTED' if h3b_results['supported'] else 'NOT SUPPORTED'}")
        
        if h3b_results.get('paradox_detected', False):
            print("⚠️  LOAD FACTOR PARADOX DETECTED: ULCC competition increases incumbent load factors")
            
        return {
            'H3a': h3a_results,
            'H3b': h3b_results
        }
        
    def save_results(self):
        """Save competition analysis results"""
        # Save competition data
        self.competition_data.to_csv('report/route_competition_analysis.csv', index=False)
        
        # Create summary statistics
        summary_stats = {
            'Total_Route_Months': len(self.competition_data),
            'Routes_with_ULCC': self.competition_data['Has_ULCC'].sum(),
            'ULCC_Penetration_Rate': self.competition_data['Has_ULCC'].mean(),
            'Avg_HHI_Overall': self.competition_data['HHI'].mean(),
            'Avg_HHI_with_ULCC': self.competition_data[self.competition_data['Has_ULCC']]['HHI'].mean(),
            'Avg_HHI_without_ULCC': self.competition_data[~self.competition_data['Has_ULCC']]['HHI'].mean()
        }
        
        print("\nCompetition Analysis Summary:")
        for key, value in summary_stats.items():
            if 'Rate' in key or 'HHI' in key:
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value:,}")
                
        return summary_stats
        
    def run_analysis(self):
        """Run complete competition analysis"""
        self.load_competition_data()
        self.classify_carriers()
        self.calculate_route_competition_metrics()
        
        # Test hypotheses
        h3_results = self.test_h3_hypotheses()
        
        # Create visualizations
        self.create_competition_visualization()
        
        # Save results
        summary_stats = self.save_results()
        
        self.results = {
            'hypothesis_tests': h3_results,
            'competition_data': self.competition_data,
            'summary_stats': summary_stats
        }
        
        return self.results
