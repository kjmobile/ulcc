# covid_recovery_analysis.py
# #num5: COVID Recovery Analysis for H4 Testing - Complete

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class CovidRecoveryAnalysis:
    def __init__(self, data_path, airline_classification, shock_data):
        self.data_path = Path(data_path)
        self.airline_classification = airline_classification
        self.shock_data = shock_data
        self.results = {}
        
    def load_recovery_data(self):
        """Load data for COVID recovery analysis"""
        print("Loading COVID recovery data...")
        
        # Load T-100 data for traffic analysis
        self.recovery_data = []
        years = range(2018, 2025)
        
        for year in years:
            file_path = self.data_path / 't_100' / f't_100_{year}.parquet'
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df['Year'] = year
                self.recovery_data.append(df)
                print(f"Loaded T100 {year}: {len(df):,} rows")
                
        self.recovery_data = pd.concat(self.recovery_data, ignore_index=True)
        
    def classify_carriers(self):
        """Add carrier type classification"""
        carrier_to_type = {}
        for carrier_type, carriers in self.airline_classification.items():
            for carrier in carriers:
                carrier_to_type[carrier] = carrier_type
                
        self.recovery_data['Carrier_Type'] = self.recovery_data['Mkt Al'].map(carrier_to_type)
        self.recovery_data = self.recovery_data.dropna(subset=['Carrier_Type'])
        
    def calculate_monthly_traffic_by_carrier_type(self):
        """Calculate monthly traffic aggregated by carrier type"""
        print("Calculating monthly traffic by carrier type...")
        
        monthly_data = self.recovery_data.groupby(['Year', 'Month', 'Carrier_Type']).agg({
            'Onboards': 'sum',
            'ASMs': 'sum',
            'RPMs': 'sum'
        }).reset_index()
        
        monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
        monthly_data['Load_Factor'] = (monthly_data['RPMs'] / monthly_data['ASMs'] * 100).fillna(0)
        
        self.monthly_traffic = monthly_data
        return monthly_data
        
    def establish_baseline_performance(self):
        """Establish 2019 baseline for recovery analysis"""
        print("Establishing 2019 baseline...")
        
        baseline_2019 = self.monthly_traffic[self.monthly_traffic['Year'] == 2019].groupby(['Month', 'Carrier_Type']).agg({
            'Onboards': 'mean',
            'ASMs': 'mean',
            'RPMs': 'mean'
        }).reset_index()
        
        self.baseline_2019 = baseline_2019
        return baseline_2019
        
    def calculate_recovery_metrics(self):
        """Calculate recovery metrics relative to 2019 baseline"""
        print("Calculating recovery metrics...")
        
        recovery_data = []
        
        for carrier_type in self.airline_classification.keys():
            carrier_monthly = self.monthly_traffic[self.monthly_traffic['Carrier_Type'] == carrier_type]
            baseline = self.baseline_2019[self.baseline_2019['Carrier_Type'] == carrier_type]
            
            if len(baseline) == 0:
                continue
                
            for _, row in carrier_monthly.iterrows():
                if row['Year'] < 2020:
                    continue
                    
                baseline_month = baseline[baseline['Month'] == row['Month']]
                if len(baseline_month) == 0:
                    continue
                    
                baseline_pax = baseline_month['Onboards'].iloc[0]
                recovery_pct = (row['Onboards'] / baseline_pax) * 100 if baseline_pax > 0 else 0
                
                recovery_data.append({
                    'Carrier_Type': carrier_type,
                    'Year': row['Year'],
                    'Month': row['Month'],
                    'Date': row['Date'],
                    'Recovery_Pct': recovery_pct,
                    'Actual_Pax': row['Onboards'],
                    'Baseline_Pax': baseline_pax
                })
                
        self.recovery_metrics = pd.DataFrame(recovery_data)
        return self.recovery_metrics
        
    def analyze_trough_performance(self):
        """Analyze trough performance during COVID"""
        print("Analyzing trough performance...")
        
        # Find minimum recovery percentage for each carrier type
        trough_analysis = {}
        
        for carrier_type in self.airline_classification.keys():
            carrier_data = self.recovery_metrics[self.recovery_metrics['Carrier_Type'] == carrier_type]
            
            if len(carrier_data) == 0:
                continue
                
            # Find trough (minimum recovery)
            min_recovery = carrier_data['Recovery_Pct'].min()
            trough_date = carrier_data[carrier_data['Recovery_Pct'] == min_recovery]['Date'].iloc[0]
            
            trough_analysis[carrier_type] = {
                'Trough_Recovery_Pct': min_recovery,
                'Trough_Date': trough_date,
                'Months_from_COVID_Start': (trough_date - pd.Timestamp('2020-03-01')).days / 30.44
            }
            
        self.trough_analysis = trough_analysis
        return trough_analysis
        
    def calculate_recovery_speed(self):
        """Calculate recovery speed (months to reach 90% of 2019 levels)"""
        print("Calculating recovery speed...")
        
        covid_start = pd.Timestamp('2020-03-01')
        recovery_speeds = {}
        
        for carrier_type in self.airline_classification.keys():
            carrier_data = self.recovery_metrics[
                (self.recovery_metrics['Carrier_Type'] == carrier_type) &
                (self.recovery_metrics['Date'] >= covid_start)
            ].sort_values('Date')
            
            if len(carrier_data) == 0:
                continue
                
            # Find first month with sustained 90% recovery
            recovery_90_months = []
            for i, (_, row) in enumerate(carrier_data.iterrows()):
                if row['Recovery_Pct'] >= 90:
                    # Check if next 2 months also above 90%
                    future_months = carrier_data.iloc[i:i+3]
                    if len(future_months) >= 3 and all(future_months['Recovery_Pct'] >= 90):
                        recovery_date = row['Date']
                        months_to_recovery = (recovery_date - covid_start).days / 30.44
                        recovery_90_months.append(months_to_recovery)
                        break
                        
            if recovery_90_months:
                recovery_speeds[carrier_type] = {
                    'Months_to_90pct': recovery_90_months[0],
                    'Recovery_Date': recovery_date
                }
            else:
                recovery_speeds[carrier_type] = {
                    'Months_to_90pct': None,
                    'Recovery_Date': None
                }
                
        self.recovery_speeds = recovery_speeds
        return recovery_speeds
        
    def analyze_market_share_evolution(self):
        """Analyze market share changes during COVID"""
        print("Analyzing market share evolution...")
        
        if not hasattr(self, 'od_recovery_data'):
            print("O&D data not available for market share analysis")
            return {}
            
        # Calculate annual market shares
        annual_shares = self.od_recovery_data.groupby(['Year', 'Carrier_Type'])['Passengers'].sum().unstack(fill_value=0)
        annual_shares = annual_shares.div(annual_shares.sum(axis=1), axis=0)
        
        # Compare 2019 vs 2023
        if 2019 in annual_shares.index and 2023 in annual_shares.index:
            share_changes = {}
            for carrier_type in self.airline_classification.keys():
                if carrier_type in annual_shares.columns:
                    share_2019 = annual_shares.loc[2019, carrier_type]
                    share_2023 = annual_shares.loc[2023, carrier_type]
                    change = share_2023 - share_2019
                    
                    share_changes[carrier_type] = {
                        'Share_2019': share_2019,
                        'Share_2023': share_2023,
                        'Change_pp': change * 100,
                        'Change_pct': (change / share_2019) * 100 if share_2019 > 0 else 0
                    }
                    
            self.market_share_evolution = share_changes
            return share_changes
        else:
            return {}
            
    def create_recovery_visualization(self):
        """Create COVID recovery visualization"""
        print("Creating COVID recovery visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Panel 1: Recovery trajectories
        colors = {'ULCC': 'red', 'LCC': 'orange', 'Hybrid': 'purple', 'Legacy': 'blue'}
        
        for carrier_type in self.airline_classification.keys():
            carrier_data = self.recovery_metrics[self.recovery_metrics['Carrier_Type'] == carrier_type]
            if len(carrier_data) > 0:
                carrier_data = carrier_data.sort_values('Date')
                ax1.plot(carrier_data['Date'], carrier_data['Recovery_Pct'], 
                        color=colors.get(carrier_type, 'gray'), 
                        linewidth=2, label=carrier_type, marker='o', markersize=3)
                
        ax1.axhline(y=90, color='black', linestyle='--', alpha=0.5, label='90% Recovery Target')
        ax1.axhline(y=100, color='black', linestyle='-', alpha=0.3, label='Pre-COVID Level')
        ax1.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-01'), 
                   alpha=0.2, color='red', label='COVID Period')
        ax1.set_ylabel('Traffic Recovery (% of 2019)')
        ax1.set_title('H4: COVID-19 Traffic Recovery Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 120)
        
        # Panel 2: Market share evolution
        if hasattr(self, 'market_share_evolution'):
            carriers = list(self.market_share_evolution.keys())
            shares_2019 = [self.market_share_evolution[c]['Share_2019'] * 100 for c in carriers]
            shares_2023 = [self.market_share_evolution[c]['Share_2023'] * 100 for c in carriers]
            
            x = np.arange(len(carriers))
            width = 0.35
            
            ax2.bar(x - width/2, shares_2019, width, label='2019', alpha=0.7, color='lightblue')
            ax2.bar(x + width/2, shares_2023, width, label='2023', alpha=0.7, color='darkblue')
            
            ax2.set_ylabel('Market Share (%)')
            ax2.set_title('Market Share Evolution (2019 vs 2023)')
            ax2.set_xticks(x)
            ax2.set_xticklabels(carriers)
            ax2.legend()
            
            # Add change labels
            for i, carrier in enumerate(carriers):
                change = self.market_share_evolution[carrier]['Change_pp']
                ax2.text(i, max(shares_2019[i], shares_2023[i]) + 1, 
                        f'{change:+.1f}pp', ha='center', fontweight='bold',
                        color='green' if change > 0 else 'red')
        
        # Panel 3: Recovery speed comparison
        if hasattr(self, 'recovery_speeds'):
            speed_data = {k: v['Months_to_90pct'] for k, v in self.recovery_speeds.items() 
                         if v['Months_to_90pct'] is not None}
            
            if speed_data:
                carriers = list(speed_data.keys())
                months = list(speed_data.values())
                colors_list = [colors.get(c, 'gray') for c in carriers]
                
                bars = ax3.bar(carriers, months, color=colors_list, alpha=0.7)
                ax3.set_ylabel('Months to 90% Recovery')
                ax3.set_title('Recovery Speed Comparison')
                
                # Add value labels
                for bar, month in zip(bars, months):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{month:.0f}mo', ha='center', fontweight='bold')
        
        # Panel 4: Trough performance
        if hasattr(self, 'trough_analysis'):
            trough_data = {k: v['Trough_Recovery_Pct'] for k, v in self.trough_analysis.items()}
            
            if trough_data:
                carriers = list(trough_data.keys())
                trough_pcts = list(trough_data.values())
                colors_list = [colors.get(c, 'gray') for c in carriers]
                
                bars = ax4.bar(carriers, trough_pcts, color=colors_list, alpha=0.7)
                ax4.set_ylabel('Trough Performance (% of 2019)')
                ax4.set_title('COVID Trough Depth by Carrier Type')
                
                # Add value labels
                for bar, pct in zip(bars, trough_pcts):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                            f'{pct:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('report/figure_4_4_h4_covid_recovery.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def test_h4_hypothesis(self):
        """Test H4: COVID Recovery Hypothesis"""
        print("\nTesting H4: COVID Recovery Speed Hypothesis")
        print("Expected: ULCC fastest recovery")
        
        if not hasattr(self, 'recovery_speeds'):
            return {'supported': False, 'reason': 'No recovery speed data'}
            
        # Get recovery times
        recovery_times = {k: v['Months_to_90pct'] for k, v in self.recovery_speeds.items() 
                         if v['Months_to_90pct'] is not None}
        
        if len(recovery_times) == 0:
            return {'supported': False, 'reason': 'No recovery time data'}
            
        # Sort by recovery speed (fastest first)
        sorted_recovery = sorted(recovery_times.items(), key=lambda x: x[1])
        fastest_carrier = sorted_recovery[0][0] if sorted_recovery else None
        
        print(f"Recovery speed ranking (fastest to slowest):")
        for i, (carrier, months) in enumerate(sorted_recovery):
            print(f"{i+1}. {carrier}: {months:.0f} months")
            
        h4_supported = (fastest_carrier == 'ULCC')
        
        return {
            'hypothesis': 'H4: COVID Recovery Speed',
            'expected': 'ULCC fastest recovery',
            'actual': f"{fastest_carrier} fastest" if fastest_carrier else "No data",
            'supported': h4_supported,
            'recovery_times': recovery_times,
            'ranking': [carrier for carrier, _ in sorted_recovery]
        }
        
    def save_results(self):
        """Save COVID recovery analysis results"""
        # Save recovery metrics
        if hasattr(self, 'recovery_metrics'):
            self.recovery_metrics.to_csv('report/covid_recovery_metrics.csv', index=False)
            
        # Create summary tables
        recovery_summary = []
        
        if hasattr(self, 'recovery_speeds') and hasattr(self, 'trough_analysis'):
            for carrier_type in self.airline_classification.keys():
                speed_data = self.recovery_speeds.get(carrier_type, {})
                trough_data = self.trough_analysis.get(carrier_type, {})
                
                recovery_summary.append({
                    'Model': carrier_type,
                    '90%Recovery': f"{speed_data.get('Months_to_90pct', 'N/A'):.0f}" if speed_data.get('Months_to_90pct') else 'N/A',
                    'Trough%': f"{trough_data.get('Trough_Recovery_Pct', 0):.1f}"
                })
                
        summary_df = pd.DataFrame(recovery_summary)
        summary_df.to_csv('report/table_4_5_recovery_analysis.csv', index=False)
        
        print("\nTable 4.5: COVID-19 Recovery Speed Analysis")
        print("-" * 50)
        print(summary_df.to_string(index=False))
        print("-" * 50)
        
        return summary_df
        
    def run_analysis(self):
        """Run complete COVID recovery analysis"""
        self.load_recovery_data()
        self.classify_carriers()
        self.calculate_monthly_traffic_by_carrier_type()
        self.establish_baseline_performance()
        self.calculate_recovery_metrics()
        
        # Analyze recovery patterns
        self.analyze_trough_performance()
        self.calculate_recovery_speed()
        self.analyze_market_share_evolution()
        
        # Test hypothesis
        h4_results = self.test_h4_hypothesis()
        
        # Create visualizations
        self.create_recovery_visualization()
        
        # Save results
        summary_table = self.save_results()
        
        self.results = {
            'hypothesis_test': h4_results,
            'recovery_metrics': getattr(self, 'recovery_metrics', pd.DataFrame()),
            'recovery_speeds': getattr(self, 'recovery_speeds', {}),
            'trough_analysis': getattr(self, 'trough_analysis', {}),
            'market_share_evolution': getattr(self, 'market_share_evolution', {}),
            'summary_table': summary_table
        }
        
        return self.results
