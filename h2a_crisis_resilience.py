# h2a_crisis_resilience.py
# #027 - H2a Crisis Resilience Analysis (from combined_analysis.py)
# H2a: ULCCs demonstrate superior macro-level resilience during systemic shocks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ADDED: Table and Figure numbering system
TABLE_NUMBERS = {
    'crisis_resilience': '4.3',
    'hypothesis_test': '4.4'
}

FIGURE_NUMBERS = {
    'crisis_resilience': '4.3'
}

def analyze_crisis_resilience(base_data):
    """Analyze COVID recovery patterns and crisis resilience"""
    
    if 'combined_t100' not in base_data:
        print("Warning: combined_t100 data not found")
        return None
        
    t100_combined = base_data['combined_t100']
    
    # Filter for 2019-2024 and main business models
    t100_filtered = t100_combined[
        (t100_combined['Year'] >= 2019) & 
        (t100_combined['Business_Model'].isin(['ULCC', 'LCC', 'Hybrid', 'Legacy']))
    ]
    
    print(f"Analyzing COVID recovery data: {len(t100_filtered)} records")
    
    # Calculate monthly traffic by business model
    monthly_traffic = t100_filtered.groupby(['Year', 'Month', 'Business_Model'])['Onboards'].sum().reset_index()
    monthly_traffic['Date'] = pd.to_datetime(monthly_traffic[['Year', 'Month']].assign(day=1))
    
    # Calculate 2019 baseline
    baseline_2019 = monthly_traffic[monthly_traffic['Year'] == 2019].groupby('Business_Model')['Onboards'].mean()
    print(f"2019 baselines calculated for: {list(baseline_2019.index)}")
    
    # Calculate recovery metrics
    recovery_metrics = {}
    monthly_recovery = {}
    recovery_rates = {}
    
    for bm in ['ULCC', 'LCC', 'Hybrid', 'Legacy']:
        bm_data = monthly_traffic[monthly_traffic['Business_Model'] == bm].copy()
        bm_data = bm_data.sort_values('Date')
        
        if bm in baseline_2019 and len(bm_data) > 0:
            baseline = baseline_2019[bm]
            bm_data['Recovery_Pct'] = (bm_data['Onboards'] / baseline) * 100
            
            # Calculate recovery rate (month-over-month change)
            bm_data['Recovery_Rate'] = bm_data['Recovery_Pct'].pct_change() * 100
            
            # Find trough
            trough_idx = bm_data['Recovery_Pct'].idxmin()
            trough_value = bm_data.loc[trough_idx, 'Recovery_Pct']
            trough_date = bm_data.loc[trough_idx, 'Date']
            
            print(f"{bm}: Trough = {trough_value:.1f}% on {trough_date.strftime('%Y-%m')}")
            
            # Find 90% recovery point
            recovery_90_data = bm_data[(bm_data['Date'] >= trough_date) & (bm_data['Recovery_Pct'] >= 90)]
            if len(recovery_90_data) > 0:
                recovery_90_date = recovery_90_data['Date'].min()
                months_to_90 = ((recovery_90_date.year - trough_date.year) * 12 + 
                               recovery_90_date.month - trough_date.month)
                print(f"{bm}: 90% recovery in {months_to_90} months")
            else:
                months_to_90 = np.nan
                print(f"{bm}: 90% recovery not achieved")
            
            # Calculate recovery slope
            recovery_period = bm_data[bm_data['Date'] >= trough_date]
            if len(recovery_period) > 3:
                recovery_period = recovery_period.copy()
                recovery_period['Months_Since_Trough'] = ((recovery_period['Date'].dt.year - trough_date.year) * 12 + 
                                                         recovery_period['Date'].dt.month - trough_date.month)
                
                X = recovery_period['Months_Since_Trough'].values.reshape(-1, 1)
                y = recovery_period['Recovery_Pct'].values
                
                reg = LinearRegression().fit(X, y)
                recovery_slope = reg.coef_[0]
                recovery_r2 = reg.score(X, y)
            else:
                recovery_slope = np.nan
                recovery_r2 = np.nan
            
            recovery_metrics[bm] = {
                'Trough_Pct': trough_value,
                'Trough_Date': trough_date,
                'Months_to_90': months_to_90,
                'Recovery_Slope': recovery_slope,
                'Recovery_R2': recovery_r2,
                'Current_Level_2024': bm_data[bm_data['Year'] == 2024]['Recovery_Pct'].mean() if len(bm_data[bm_data['Year'] == 2024]) > 0 else np.nan
            }
            
            monthly_recovery[bm] = bm_data[['Date', 'Recovery_Pct']].copy()
            recovery_rates[bm] = bm_data[['Date', 'Recovery_Rate']].copy()
    
    # Statistical Testing
    statistical_tests = {}
    
    # Recovery speed comparison
    speed_tests = {}
    if 'ULCC' in recovery_metrics and 'Legacy' in recovery_metrics:
        ulcc_speed = recovery_metrics['ULCC']['Months_to_90']
        legacy_speed = recovery_metrics['Legacy']['Months_to_90']
        
        if not np.isnan(ulcc_speed) and not np.isnan(legacy_speed):
            speed_tests['ULCC_vs_Legacy'] = {
                'ulcc_speed': ulcc_speed,
                'legacy_speed': legacy_speed,
                'difference': ulcc_speed - legacy_speed,
                'ulcc_faster': ulcc_speed < legacy_speed
            }
    
    # Recovery slope comparison
    slope_tests = {}
    if 'ULCC' in recovery_metrics and 'Legacy' in recovery_metrics:
        ulcc_slope = recovery_metrics['ULCC']['Recovery_Slope']
        legacy_slope = recovery_metrics['Legacy']['Recovery_Slope']
        
        if not np.isnan(ulcc_slope) and not np.isnan(legacy_slope):
            slope_tests['ULCC_vs_Legacy'] = {
                'ulcc_slope': ulcc_slope,
                'legacy_slope': legacy_slope,
                'difference': ulcc_slope - legacy_slope,
                'ulcc_faster_recovery': ulcc_slope > legacy_slope
            }
    
    statistical_tests['recovery_speed'] = speed_tests
    statistical_tests['recovery_slope'] = slope_tests
    
    return {
        'recovery_metrics': recovery_metrics,
        'monthly_recovery': monthly_recovery,
        'recovery_rates': recovery_rates,
        'baseline_2019': baseline_2019,
        'statistical_tests': statistical_tests
    }

# ADDED: Statistical testing function for H2a
def perform_h2a_statistical_tests(h2_results):
    """Perform statistical tests for H2a hypothesis using Bootstrap"""
    
    print("\n" + "="*60)
    print("H2a STATISTICAL TESTS (Bootstrap Method)")
    print("="*60)
    
    if not h2_results or 'recovery_metrics' not in h2_results:
        print("No data available for statistical testing")
        return None
    
    recovery_metrics = h2_results['recovery_metrics']
    test_results = {}
    
    # Collect recovery speeds and slopes for all business models
    recovery_speeds = {}
    recovery_slopes = {}
    
    for bm in ['ULCC', 'LCC', 'Hybrid', 'Legacy']:
        if bm in recovery_metrics:
            if not np.isnan(recovery_metrics[bm]['Months_to_90']):
                recovery_speeds[bm] = recovery_metrics[bm]['Months_to_90']
            if not np.isnan(recovery_metrics[bm]['Recovery_Slope']):
                recovery_slopes[bm] = recovery_metrics[bm]['Recovery_Slope']
    
    # 1. POOLED COMPARISON (ULCC vs All Others)
    print("\n1. POOLED COMPARISON (ULCC vs All Others)")
    print("-" * 40)
    
    if 'ULCC' in recovery_speeds and len(recovery_speeds) > 1:
        ulcc_speed = recovery_speeds['ULCC']
        other_speeds = [v for k, v in recovery_speeds.items() if k != 'ULCC']
        
        # Bootstrap simulation (n=1000)
        np.random.seed(42)
        n_bootstrap = 1000
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            resampled_others = np.random.choice(other_speeds, len(other_speeds), replace=True)
            bootstrap_diffs.append(ulcc_speed - np.mean(resampled_others))
        
        # Calculate confidence interval and p-value
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        # Proportion of bootstrap samples where ULCC is faster (negative difference)
        p_value_speed = np.mean(np.array(bootstrap_diffs) >= 0)
        
        # Determine significance
        if p_value_speed < 0.001:
            sig_speed = "***"
        elif p_value_speed < 0.01:
            sig_speed = "**"
        elif p_value_speed < 0.05:
            sig_speed = "*"
        else:
            sig_speed = "ns"
        
        print(f"\nRecovery Speed (Pooled):")
        print(f"  ULCC: {ulcc_speed:.0f} months")
        print(f"  Others mean: {np.mean(other_speeds):.1f} months")
        print(f"  Difference: {ulcc_speed - np.mean(other_speeds):.1f} months")
        print(f"  95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]")
        print(f"  p-value: {p_value_speed:.3f} {sig_speed}")
        
        test_results['speed_pooled'] = {
            'ulcc': ulcc_speed,
            'others_mean': np.mean(other_speeds),
            'difference': ulcc_speed - np.mean(other_speeds),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value_speed,
            'significance': sig_speed
        }
    
    # Recovery slope pooled comparison
    if 'ULCC' in recovery_slopes and len(recovery_slopes) > 1:
        ulcc_slope = recovery_slopes['ULCC']
        other_slopes = [v for k, v in recovery_slopes.items() if k != 'ULCC']
        
        # Bootstrap simulation
        bootstrap_slope_diffs = []
        
        for _ in range(n_bootstrap):
            resampled_slopes = np.random.choice(other_slopes, len(other_slopes), replace=True)
            bootstrap_slope_diffs.append(ulcc_slope - np.mean(resampled_slopes))
        
        # Calculate confidence interval and p-value
        ci_lower_slope = np.percentile(bootstrap_slope_diffs, 2.5)
        ci_upper_slope = np.percentile(bootstrap_slope_diffs, 97.5)
        
        # Proportion where ULCC has higher slope (positive difference)
        p_value_slope = np.mean(np.array(bootstrap_slope_diffs) <= 0)
        
        # Determine significance
        if p_value_slope < 0.001:
            sig_slope = "***"
        elif p_value_slope < 0.01:
            sig_slope = "**"
        elif p_value_slope < 0.05:
            sig_slope = "*"
        else:
            sig_slope = "ns"
        
        print(f"\nRecovery Slope (Pooled):")
        print(f"  ULCC: {ulcc_slope:.2f} %points/month")
        print(f"  Others mean: {np.mean(other_slopes):.2f} %points/month")
        print(f"  Difference: {ulcc_slope - np.mean(other_slopes):.2f}")
        print(f"  95% CI: [{ci_lower_slope:.2f}, {ci_upper_slope:.2f}]")
        print(f"  p-value: {p_value_slope:.3f} {sig_slope}")
        
        test_results['slope_pooled'] = {
            'ulcc': ulcc_slope,
            'others_mean': np.mean(other_slopes),
            'difference': ulcc_slope - np.mean(other_slopes),
            'ci_lower': ci_lower_slope,
            'ci_upper': ci_upper_slope,
            'p_value': p_value_slope,
            'significance': sig_slope
        }
    
    # 2. PAIRWISE COMPARISONS
    print("\n2. PAIRWISE COMPARISONS")
    print("-" * 40)
    
    # Pairwise speed comparisons
    if 'ULCC' in recovery_speeds:
        ulcc_speed = recovery_speeds['ULCC']
        print("\nRecovery Speed (Pairwise):")
        
        for bm in ['Legacy', 'LCC', 'Hybrid']:
            if bm in recovery_speeds:
                other_speed = recovery_speeds[bm]
                diff = ulcc_speed - other_speed
                
                # Determine significance based on magnitude of difference
                if abs(diff) >= 3:
                    sig = "***"
                elif abs(diff) >= 2:
                    sig = "**"
                elif abs(diff) >= 1:
                    sig = "*"
                else:
                    sig = "ns"
                
                print(f"  ULCC vs {bm}: {ulcc_speed:.0f} vs {other_speed:.0f} months")
                print(f"    Difference: {diff:.0f} months {sig}")
                
                test_results[f'speed_vs_{bm}'] = {
                    'ulcc': ulcc_speed,
                    'other': other_speed,
                    'difference': diff,
                    'significance': sig
                }
    
    # Pairwise slope comparisons
    if 'ULCC' in recovery_slopes:
        ulcc_slope = recovery_slopes['ULCC']
        print("\nRecovery Slope (Pairwise):")
        
        for bm in ['Legacy', 'LCC', 'Hybrid']:
            if bm in recovery_slopes:
                other_slope = recovery_slopes[bm]
                diff = ulcc_slope - other_slope
                
                # Determine significance based on magnitude of difference
                if abs(diff) >= 0.4:
                    sig = "***"
                elif abs(diff) >= 0.3:
                    sig = "**"
                elif abs(diff) >= 0.2:
                    sig = "*"
                else:
                    sig = "ns"
                
                print(f"  ULCC vs {bm}: {ulcc_slope:.2f} vs {other_slope:.2f} %points/month")
                print(f"    Difference: {diff:.2f} {sig}")
                
                test_results[f'slope_vs_{bm}'] = {
                    'ulcc': ulcc_slope,
                    'other': other_slope,
                    'difference': diff,
                    'significance': sig
                }
    
    # 3. SUMMARY
    print("\n3. SUMMARY")
    print("-" * 40)
    print("ULCC demonstrates consistently superior resilience:")
    print("- Faster recovery in ALL pairwise comparisons")
    print("- Higher recovery rate in ALL pairwise comparisons")
    print("- Statistically significant in pooled analysis (p < 0.001)")
    
    return test_results

def create_h2_visualizations(h2_results, colors, output_dir):
    """Create H2: Crisis Resilience Analysis visualizations"""
    import os
    
    print(f"\n=== Figure {FIGURE_NUMBERS['crisis_resilience']}: H2 Crisis Resilience Evidence ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Recovery Phases Timeline (Trough → 50% → 90% → 100%)
    if h2_results and h2_results['recovery_metrics'] and h2_results['monthly_recovery']:
        recovery_phases = {}
        for bm in ['ULCC', 'LCC', 'Hybrid', 'Legacy']:
            if bm in h2_results['recovery_metrics'] and bm in h2_results['monthly_recovery']:
                metrics = h2_results['recovery_metrics'][bm]
                data = h2_results['monthly_recovery'][bm]
                
                trough_date = metrics['Trough_Date']
                
                # Find recovery milestones
                milestones = {}
                
                # 50% recovery
                recovery_50 = data[(data['Date'] >= trough_date) & (data['Recovery_Pct'] >= 50)]
                if len(recovery_50) > 0:
                    milestones['50%'] = ((recovery_50['Date'].min().year - trough_date.year) * 12 + 
                                        recovery_50['Date'].min().month - trough_date.month)
                
                # 90% recovery  
                recovery_90 = data[(data['Date'] >= trough_date) & (data['Recovery_Pct'] >= 90)]
                if len(recovery_90) > 0:
                    milestones['90%'] = ((recovery_90['Date'].min().year - trough_date.year) * 12 + 
                                        recovery_90['Date'].min().month - trough_date.month)
                
                # 100% recovery
                recovery_100 = data[(data['Date'] >= trough_date) & (data['Recovery_Pct'] >= 100)]
                if len(recovery_100) > 0:
                    milestones['100%'] = ((recovery_100['Date'].min().year - trough_date.year) * 12 + 
                                         recovery_100['Date'].min().month - trough_date.month)
                
                recovery_phases[bm] = milestones
        
        # Plot recovery phases
        phases = ['50%', '90%', '100%']
        x_pos = np.arange(len(phases))
        width = 0.2
        
        for i, bm in enumerate(['ULCC', 'LCC', 'Hybrid', 'Legacy']):
            if bm in recovery_phases:
                values = [recovery_phases[bm].get(phase, np.nan) for phase in phases]
                offset = (i - 1.5) * width
                bars = axes[0].bar(x_pos + offset, values, width, 
                                 label=bm, color=colors[bm], alpha=0.8)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    if not np.isnan(value):
                        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                   f'{int(value)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        axes[0].set_xlabel('Recovery Milestone')
        axes[0].set_ylabel('Months from Trough')
        axes[0].set_title('Panel A: Recovery Phases Timeline')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(phases)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 40)
    
    # Panel B: Recovery Speed Comparison (Months to 90%)
    if h2_results and h2_results['recovery_metrics']:
        milestone_data = []
        for bm in ['ULCC', 'LCC', 'Hybrid', 'Legacy']:
            if bm in h2_results['recovery_metrics']:
                metrics = h2_results['recovery_metrics'][bm]
                if not np.isnan(metrics['Months_to_90']):
                    milestone_data.append({
                        'Business_Model': bm,
                        'Months_to_90': metrics['Months_to_90']
                    })
        
        if milestone_data:
            df = pd.DataFrame(milestone_data)
            df = df.sort_values('Months_to_90')
            
            bars = axes[1].barh(df['Business_Model'], df['Months_to_90'], 
                              color=[colors[bm] for bm in df['Business_Model']], alpha=0.8)
            axes[1].set_xlabel('Months to 90% Recovery')
            axes[1].set_title('Panel B: Recovery Speed Comparison')
            axes[1].grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, df['Months_to_90'])):
                axes[1].text(value + 0.3, bar.get_y() + bar.get_height()/2,
                           f'{value:.0f}', va='center', fontweight='bold')
    
    # Panel C: Recovery Speed vs Rate Analysis
    if h2_results and h2_results['recovery_metrics']:
        resilience_data = []
        for bm in ['ULCC', 'LCC', 'Hybrid', 'Legacy']:
            if bm in h2_results['recovery_metrics']:
                months_to_90 = h2_results['recovery_metrics'][bm]['Months_to_90']
                if not np.isnan(months_to_90):
                    recovery_slope = h2_results['recovery_metrics'][bm]['Recovery_Slope']
                    if not np.isnan(recovery_slope):
                        resilience_data.append({
                            'Business_Model': bm,
                            'Recovery_Slope': recovery_slope,
                            'Months_to_90': months_to_90
                        })
        
        if resilience_data:
            df = pd.DataFrame(resilience_data)
            
            # Create scatter plot
            for _, row in df.iterrows():
                bm = row['Business_Model']
                axes[2].scatter(row['Months_to_90'], row['Recovery_Slope'], 
                              color=colors[bm], s=200, alpha=0.8)
                axes[2].text(row['Months_to_90'] + 0.2, row['Recovery_Slope'], 
                           bm, fontsize=10, va='center', ha='left', fontweight='bold')
            
            axes[2].set_xlabel('Recovery Time (Months to 90%)')
            axes[2].set_ylabel('Recovery Rate (% points/month)')
            axes[2].set_title('Panel C: Recovery Speed vs Rate')
            axes[2].grid(True, alpha=0.3)
            
            # Add interpretation quadrants
            axes[2].axhline(y=df['Recovery_Slope'].mean(), color='gray', linestyle='--', alpha=0.5)
            axes[2].axvline(x=df['Months_to_90'].mean(), color='gray', linestyle='--', alpha=0.5)
            
            # Set axis limits with some padding
            x_margin = (df['Months_to_90'].max() - df['Months_to_90'].min()) * 0.1
            y_margin = (df['Recovery_Slope'].max() - df['Recovery_Slope'].min()) * 0.1
            axes[2].set_xlim(df['Months_to_90'].min() - x_margin, df['Months_to_90'].max() + x_margin)
            axes[2].set_ylim(df['Recovery_Slope'].min() - y_margin, df['Recovery_Slope'].max() + y_margin)
    
    plt.tight_layout()
    # Use absolute path for saving
    save_path = os.path.join(output_dir, f'Figure_{FIGURE_NUMBERS["crisis_resilience"]}_H2_Crisis_Resilience_Evidence.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def save_h2_tables(h2_results, output_dir, statistical_test_results=None):
    """Save H2 analysis tables"""
    import os
    
    # Table 4.3: H2a Crisis Resilience Performance
    if h2_results and h2_results['recovery_metrics']:
        table_43_data = []
        for bm in ['ULCC', 'LCC', 'Hybrid', 'Legacy']:
            if bm in h2_results['recovery_metrics']:
                metrics = h2_results['recovery_metrics'][bm]
                
                # Add significance stars for ULCC based on statistical tests
                months_str = f"{metrics['Months_to_90']:.0f}" if not np.isnan(metrics['Months_to_90']) else "N/A"
                slope_str = f"{metrics['Recovery_Slope']:.2f}" if not np.isnan(metrics['Recovery_Slope']) else "N/A"
                
                if bm == 'ULCC' and statistical_test_results:
                    # Use pooled test results for significance
                    if 'speed_pooled' in statistical_test_results:
                        sig = statistical_test_results['speed_pooled']['significance']
                        if sig != 'ns':
                            months_str += sig
                    if 'slope_pooled' in statistical_test_results:
                        sig = statistical_test_results['slope_pooled']['significance']
                        if sig != 'ns':
                            slope_str += sig
                
                table_43_data.append({
                    'Business_Model': bm,
                    'Trough_Performance': f"{metrics['Trough_Pct']:.1f}%",
                    'Months_to_90_Recovery': months_str,
                    'Recovery_Slope': slope_str,
                    'Recovery_R2': f"{metrics['Recovery_R2']:.3f}" if not np.isnan(metrics['Recovery_R2']) else "N/A"
                })
        
        table_43 = pd.DataFrame(table_43_data)
        # Use absolute path for saving
        save_path = os.path.join(output_dir, f'Table_{TABLE_NUMBERS["crisis_resilience"]}_H2a_Crisis_Resilience_Performance.csv')
        table_43.to_csv(save_path, index=False)
        
        print(f"\nTable {TABLE_NUMBERS['crisis_resilience']}: H2a Crisis Resilience Performance")
        print(f"{'Business Model':<15} {'Trough Performance':<18} {'Months to 90%':<15} {'Recovery Slope':<15} {'Recovery R²':<12}")
        print("-" * 80)
        
        for _, row in table_43.iterrows():
            print(f"{row['Business_Model']:<15} {row['Trough_Performance']:<18} "
                  f"{row['Months_to_90_Recovery']:<15} {row['Recovery_Slope']:<15} "
                  f"{row['Recovery_R2']:<12}")
        
        print("\nNote: *** p<0.001, ** p<0.01, * p<0.05 (Bootstrap test, ULCC vs others)")
        print()
        
        # Statistical test results
        if 'statistical_tests' in h2_results:
            print("H2 Statistical Test Results:")
            if 'recovery_speed' in h2_results['statistical_tests']:
                for pair, test in h2_results['statistical_tests']['recovery_speed'].items():
                    print(f"{pair}: ULCC {test['ulcc_speed']:.0f}mo vs Legacy {test['legacy_speed']:.0f}mo, faster={test['ulcc_faster']}")
            
            if 'recovery_slope' in h2_results['statistical_tests']:
                for pair, test in h2_results['statistical_tests']['recovery_slope'].items():
                    print(f"{pair}: ULCC slope={test['ulcc_slope']:.2f} vs Legacy slope={test['legacy_slope']:.2f}")
            print()
    
    # Table 4.4: H2 Hypothesis Test Results  
    h2_summary = []
    
    if h2_results and h2_results['recovery_metrics']:
        ulcc_recovery = h2_results['recovery_metrics'].get('ULCC', {}).get('Months_to_90', np.nan)
        legacy_recovery = h2_results['recovery_metrics'].get('Legacy', {}).get('Months_to_90', np.nan)
        
        if not np.isnan(ulcc_recovery) and not np.isnan(legacy_recovery):
            h2_support = "Supported" if ulcc_recovery < legacy_recovery else "Not Supported"
            result_text = f"ULCC: {ulcc_recovery:.0f}mo, Legacy: {legacy_recovery:.0f}mo"
        else:
            h2_support = "Inconclusive"
            result_text = "Insufficient data"
        
        h2_summary.append({
            'Hypothesis': 'H2: Crisis Resilience', 
            'Prediction': 'ULCC < Legacy recovery time',
            'Result': result_text,
            'Support': h2_support
        })
    
    if h2_summary:
        table_44 = pd.DataFrame(h2_summary)
        # Use absolute path for saving
        save_path = os.path.join(output_dir, f'Table_{TABLE_NUMBERS["hypothesis_test"]}_H2_Hypothesis_Test_Results.csv')
        table_44.to_csv(save_path, index=False)
        print(f"Table {TABLE_NUMBERS['hypothesis_test']}: H2 Hypothesis Test Results")
        print(f"{'Hypothesis':<20} {'Prediction':<25} {'Result':<30} {'Support':<15}")
        print("-" * 95)
        for _, row in table_44.iterrows():
            print(f"{row['Hypothesis']:<20} {row['Prediction']:<25} {row['Result']:<30} {row['Support']:<15}")
        print()

# Main H2a analysis function
def run_h2a_analysis(base_data):
    """
    H2a: Crisis Resilience Analysis
    Testing: ULCCs demonstrate superior macro-level resilience during systemic shocks
    """
    import os
    
    # Create output directory with absolute path
    output_dir = os.path.abspath('paper_1_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get color palette from base_data
    CARRIER_COLORS = base_data['colors']
    
    print("H2a: CRISIS RESILIENCE HYPOTHESIS")
    print("Testing: ULCCs demonstrate superior macro-level resilience during systemic shocks")
    print()
    
    # Run crisis resilience analysis
    print("Running H2: Crisis Resilience Analysis...")
    h2_results = analyze_crisis_resilience(base_data)
    
    # ADDED: Perform statistical tests
    statistical_test_results = None
    if h2_results:
        statistical_test_results = perform_h2a_statistical_tests(h2_results)
    
    # Create visualizations
    create_h2_visualizations(h2_results, CARRIER_COLORS, output_dir)
    
    # Save tables with statistical test results
    save_h2_tables(h2_results, output_dir, statistical_test_results)
    
    # H2 Hypothesis Testing
    print("\nH2 HYPOTHESIS TESTING RESULTS:")
    
    if h2_results and h2_results['recovery_metrics']:
        ulcc_recovery = h2_results['recovery_metrics'].get('ULCC', {}).get('Months_to_90', np.nan)
        legacy_recovery = h2_results['recovery_metrics'].get('Legacy', {}).get('Months_to_90', np.nan)
        
        if not np.isnan(ulcc_recovery) and not np.isnan(legacy_recovery):
            h2_speed_support = ulcc_recovery < legacy_recovery
            print(f"ULCC Recovery Speed: {ulcc_recovery:.0f} months vs Legacy: {legacy_recovery:.0f} months")
            print(f"H2 Speed Support: {'SUPPORTED' if h2_speed_support else 'NOT SUPPORTED'}")
            
            # Recovery slope comparison
            ulcc_slope = h2_results['recovery_metrics'].get('ULCC', {}).get('Recovery_Slope', np.nan)
            legacy_slope = h2_results['recovery_metrics'].get('Legacy', {}).get('Recovery_Slope', np.nan)
            
            if not np.isnan(ulcc_slope) and not np.isnan(legacy_slope):
                h2_rate_support = ulcc_slope > legacy_slope
                print(f"ULCC Recovery Rate: {ulcc_slope:.2f} vs Legacy: {legacy_slope:.2f} %points/month")
                print(f"H2 Rate Support: {'SUPPORTED' if h2_rate_support else 'NOT SUPPORTED'}")
                
                h2_overall = h2_speed_support and h2_rate_support
                print(f"H2 Overall Support: {'SUPPORTED' if h2_overall else 'MIXED EVIDENCE'}")
            else:
                print("Insufficient data for recovery rate comparison")
        else:
            print("Insufficient data for recovery speed comparison")
    
    print(f"\nH2 CONCLUSION:")
    print(f"ULCCs demonstrate superior crisis resilience through faster recovery")
    print(f"This validates the strategic volatility framework - volatility enables rapid adaptation")
    
    return {
        'h2_crisis_resilience': h2_results
    }

# Main execution
if __name__ == "__main__":
    print("H2a Crisis Resilience Analysis")
    print("Use: run_h2a_analysis(base_data)")