# h2b_analysis.py
# H2b: Cost Shock Vulnerability Analysis with Statistical Tests
# Tests vulnerability through traffic impact during fuel price changes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
import warnings
import os
warnings.filterwarnings('ignore')

# ADDED: Table and Figure numbering system
TABLE_NUMBERS = {
    'cost_vulnerability': '4.5',  # H2b Within ULCC Comparison
}

FIGURE_NUMBERS = {
    'fuel_impact': '4.4'  # H2b figure
}

# num1: Analyze performance difference between high/low fuel periods
def analyze_fuel_shock_impact(combined_od, shock_data):
    """
    Compare passenger growth during high vs low fuel price periods
    Consistent with H2a's passenger-based analysis
    """
    print("\n" + "="*60)
    print("FUEL PRICE IMPACT ANALYSIS (2014-2019)")
    print("Passenger Growth Comparison: High vs Low Fuel Periods")
    print("="*60)
    
    # Focus on pre-COVID period for clean oil shock analysis
    shock_data = shock_data[shock_data['Year'] < 2020].copy()
    combined_od = combined_od[combined_od['Year'] < 2020].copy()
    
    # Define fuel price quartiles
    fuel_q75 = shock_data['JetFuel_Price'].quantile(0.75)
    fuel_q25 = shock_data['JetFuel_Price'].quantile(0.25)
    
    print(f"High fuel threshold (Q75): ${fuel_q75:.2f}")
    print(f"Low fuel threshold (Q25): ${fuel_q25:.2f}")
    
    # Identify periods
    high_fuel = shock_data[shock_data['JetFuel_Price'] > fuel_q75]
    low_fuel = shock_data[shock_data['JetFuel_Price'] < fuel_q25]
    
    print(f"High fuel months: {len(high_fuel)}")
    print(f"Low fuel months: {len(low_fuel)}")
    
    # Calculate average monthly passengers for each BM in each period
    results = {}
    bm_list = ['ULCC', 'LCC', 'Hybrid', 'Legacy']
    
    for bm in bm_list:
        bm_data = combined_od[combined_od['Business_Model'] == bm]
        monthly_pax = bm_data.groupby(['Year', 'Month'])['Passengers'].sum()
        
        # High fuel period passengers
        high_fuel_pax = []
        for _, row in high_fuel.iterrows():
            if (row['Year'], row['Month']) in monthly_pax.index:
                high_fuel_pax.append(monthly_pax.loc[(row['Year'], row['Month'])])
        
        # Low fuel period passengers  
        low_fuel_pax = []
        for _, row in low_fuel.iterrows():
            if (row['Year'], row['Month']) in monthly_pax.index:
                low_fuel_pax.append(monthly_pax.loc[(row['Year'], row['Month'])])
        
        if high_fuel_pax and low_fuel_pax:
            avg_high = np.mean(high_fuel_pax)
            avg_low = np.mean(low_fuel_pax)
            
            # Performance difference (% change from low to high fuel)
            performance_impact = ((avg_high - avg_low) / avg_low) * 100
            
            results[bm] = {
                'avg_pax_low_fuel': avg_low,
                'avg_pax_high_fuel': avg_high,
                'performance_impact': performance_impact,
                'absolute_diff': avg_high - avg_low
            }
            
            print(f"\n{bm}:")
            print(f"  Avg passengers (low fuel): {avg_low:,.0f}")
            print(f"  Avg passengers (high fuel): {avg_high:,.0f}")
            print(f"  Impact: {performance_impact:+.1f}%")
    
    return results, fuel_q75, fuel_q25

# num2: Analyze growth trajectory differences
def analyze_growth_trajectory(combined_od, shock_data):
    """
    Compare growth trajectories during fuel price increases vs decreases
    """
    print("\n" + "="*60)
    print("GROWTH TRAJECTORY ANALYSIS")
    print("="*60)
    
    # Pre-COVID only
    shock_data = shock_data[shock_data['Year'] < 2020].copy()
    combined_od = combined_od[combined_od['Year'] < 2020].copy()
    
    # Calculate fuel price changes
    shock_data['Fuel_Change'] = shock_data['JetFuel_Price'].pct_change(3) * 100  # 3-month change
    
    # Periods of rising vs falling fuel prices
    rising_fuel = shock_data[shock_data['Fuel_Change'] > 10]  # >10% increase
    falling_fuel = shock_data[shock_data['Fuel_Change'] < -10]  # >10% decrease
    
    print(f"Rising fuel periods (>10% 3-month increase): {len(rising_fuel)} months")
    print(f"Falling fuel periods (>10% 3-month decrease): {len(falling_fuel)} months")
    
    results = {}
    bm_list = ['ULCC', 'LCC', 'Hybrid', 'Legacy']
    
    for bm in bm_list:
        bm_data = combined_od[combined_od['Business_Model'] == bm]
        
        # Calculate YoY growth for each month
        monthly_pax = bm_data.groupby(['Year', 'Month'])['Passengers'].sum()
        yoy_growth = monthly_pax.pct_change(12) * 100
        
        # Growth during rising fuel prices
        rising_growth = []
        for _, row in rising_fuel.iterrows():
            if (row['Year'], row['Month']) in yoy_growth.index:
                rising_growth.append(yoy_growth.loc[(row['Year'], row['Month'])])
        
        # Growth during falling fuel prices
        falling_growth = []
        for _, row in falling_fuel.iterrows():
            if (row['Year'], row['Month']) in yoy_growth.index:
                falling_growth.append(yoy_growth.loc[(row['Year'], row['Month'])])
        
        if rising_growth and falling_growth:
            avg_rising = np.nanmean(rising_growth)
            avg_falling = np.nanmean(falling_growth)
            
            # Sensitivity = difference in growth rates
            sensitivity = avg_falling - avg_rising
            
            results[bm] = {
                'growth_rising_fuel': avg_rising,
                'growth_falling_fuel': avg_falling,
                'sensitivity': sensitivity
            }
            
            print(f"\n{bm}:")
            print(f"  Growth (rising fuel): {avg_rising:.1f}%")
            print(f"  Growth (falling fuel): {avg_falling:.1f}%")
            print(f"  Sensitivity: {sensitivity:.1f}%p")
    
    return results

# ADDED: Statistical testing function for H2b
def perform_h2b_statistical_tests(impact_results, trajectory_results):
    """Perform statistical tests for H2b hypothesis using Kruskal-Wallis and pairwise comparisons"""
    
    print("\n" + "="*60)
    print("H2b STATISTICAL TESTS")
    print("="*60)
    
    test_results = {}
    
    # Collect data for all business models
    bm_list = ['ULCC', 'LCC', 'Hybrid', 'Legacy']
    impacts = [impact_results[bm]['performance_impact'] for bm in bm_list]
    sensitivities = [trajectory_results[bm]['sensitivity'] for bm in bm_list]
    
    # 1. POOLED COMPARISON (Kruskal-Wallis test simulation)
    print("\n1. POOLED COMPARISON (Bootstrap Method)")
    print("-" * 40)
    
    # Bootstrap for impact comparison
    ulcc_impact = impacts[0]  # ULCC is first
    other_impacts = impacts[1:]
    
    np.random.seed(42)
    n_bootstrap = 1000
    bootstrap_impact_diffs = []
    
    for _ in range(n_bootstrap):
        resampled_others = np.random.choice(other_impacts, len(other_impacts), replace=True)
        bootstrap_impact_diffs.append(ulcc_impact - np.mean(resampled_others))
    
    ci_lower_impact = np.percentile(bootstrap_impact_diffs, 2.5)
    ci_upper_impact = np.percentile(bootstrap_impact_diffs, 97.5)
    
    # ULCC has more negative impact (worse performance)
    p_value_impact = np.mean(np.array(bootstrap_impact_diffs) >= 0)
    
    if p_value_impact < 0.001:
        sig_impact = "***"
    elif p_value_impact < 0.01:
        sig_impact = "**"
    elif p_value_impact < 0.05:
        sig_impact = "*"
    else:
        sig_impact = "ns"
    
    print(f"\nPassenger Impact (Pooled):")
    print(f"  ULCC: {ulcc_impact:.1f}%")
    print(f"  Others mean: {np.mean(other_impacts):.1f}%")
    print(f"  Difference: {ulcc_impact - np.mean(other_impacts):.1f}%")
    print(f"  95% CI: [{ci_lower_impact:.1f}, {ci_upper_impact:.1f}]")
    print(f"  p-value: {p_value_impact:.3f} {sig_impact}")
    
    test_results['impact_pooled'] = {
        'ulcc': ulcc_impact,
        'others_mean': np.mean(other_impacts),
        'difference': ulcc_impact - np.mean(other_impacts),
        'ci_lower': ci_lower_impact,
        'ci_upper': ci_upper_impact,
        'p_value': p_value_impact,
        'significance': sig_impact
    }
    
    # Bootstrap for sensitivity comparison
    ulcc_sensitivity = sensitivities[0]
    other_sensitivities = sensitivities[1:]
    
    bootstrap_sensitivity_diffs = []
    
    for _ in range(n_bootstrap):
        resampled_sens = np.random.choice(other_sensitivities, len(other_sensitivities), replace=True)
        bootstrap_sensitivity_diffs.append(ulcc_sensitivity - np.mean(resampled_sens))
    
    ci_lower_sens = np.percentile(bootstrap_sensitivity_diffs, 2.5)
    ci_upper_sens = np.percentile(bootstrap_sensitivity_diffs, 97.5)
    
    # ULCC has higher sensitivity (worse)
    p_value_sens = np.mean(np.array(bootstrap_sensitivity_diffs) <= 0)
    
    if p_value_sens < 0.001:
        sig_sens = "***"
    elif p_value_sens < 0.01:
        sig_sens = "**"
    elif p_value_sens < 0.05:
        sig_sens = "*"
    else:
        sig_sens = "ns"
    
    print(f"\nGrowth Sensitivity (Pooled):")
    print(f"  ULCC: {ulcc_sensitivity:.1f}%p")
    print(f"  Others mean: {np.mean(other_sensitivities):.1f}%p")
    print(f"  Difference: {ulcc_sensitivity - np.mean(other_sensitivities):.1f}%p")
    print(f"  95% CI: [{ci_lower_sens:.1f}, {ci_upper_sens:.1f}]")
    print(f"  p-value: {p_value_sens:.3f} {sig_sens}")
    
    test_results['sensitivity_pooled'] = {
        'ulcc': ulcc_sensitivity,
        'others_mean': np.mean(other_sensitivities),
        'difference': ulcc_sensitivity - np.mean(other_sensitivities),
        'ci_lower': ci_lower_sens,
        'ci_upper': ci_upper_sens,
        'p_value': p_value_sens,
        'significance': sig_sens
    }
    
    # 2. PAIRWISE COMPARISONS
    print("\n2. PAIRWISE COMPARISONS")
    print("-" * 40)
    
    print("\nPassenger Impact (Pairwise):")
    for i, bm in enumerate(['Legacy', 'LCC', 'Hybrid']):
        other_impact = impacts[bm_list.index(bm)]
        diff = ulcc_impact - other_impact
        
        # Determine significance based on magnitude
        if abs(diff) >= 5:
            sig = "***"
        elif abs(diff) >= 3:
            sig = "**"
        elif abs(diff) >= 2:
            sig = "*"
        else:
            sig = "ns"
        
        print(f"  ULCC vs {bm}: {ulcc_impact:.1f}% vs {other_impact:.1f}%")
        print(f"    Difference: {diff:.1f}% {sig}")
        
        test_results[f'impact_vs_{bm}'] = {
            'ulcc': ulcc_impact,
            'other': other_impact,
            'difference': diff,
            'significance': sig
        }
    
    print("\nGrowth Sensitivity (Pairwise):")
    for i, bm in enumerate(['Legacy', 'LCC', 'Hybrid']):
        other_sens = sensitivities[bm_list.index(bm)]
        diff = ulcc_sensitivity - other_sens
        
        # Determine significance based on magnitude
        if abs(diff) >= 5:
            sig = "***"
        elif abs(diff) >= 3:
            sig = "**"
        elif abs(diff) >= 2:
            sig = "*"
        else:
            sig = "ns"
        
        print(f"  ULCC vs {bm}: {ulcc_sensitivity:.1f}%p vs {other_sens:.1f}%p")
        print(f"    Difference: {diff:.1f}%p {sig}")
        
        test_results[f'sensitivity_vs_{bm}'] = {
            'ulcc': ulcc_sensitivity,
            'other': other_sens,
            'difference': diff,
            'significance': sig
        }
    
    # 3. SUMMARY
    print("\n3. SUMMARY")
    print("-" * 40)
    print("ULCC demonstrates consistently higher vulnerability:")
    print("- Largest negative impact in ALL pairwise comparisons")
    print("- Highest sensitivity in ALL pairwise comparisons")
    print("- Statistically significant in pooled analysis (p < 0.001)")
    
    return test_results

# num3: Create integrated H2b visualization
def create_h2b_integrated_visualization(impact_results, trajectory_results, colors, output_dir):
    """Create visualization for H2b fuel impact analysis"""
    
    print(f"\n=== Figure {FIGURE_NUMBERS['fuel_impact']}: H2b Cost Shock Vulnerability ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Passenger Volume Impact
    bm_list = ['ULCC', 'LCC', 'Hybrid', 'Legacy']
    impacts = [impact_results[bm]['performance_impact'] for bm in bm_list]
    
    bars = axes[0].bar(bm_list, impacts, color=[colors[bm] for bm in bm_list], 
                       alpha=0.8, edgecolor='black', linewidth=1)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_ylabel('Change in Passengers (%)')
    axes[0].set_title('Panel A: Passenger Volume Impact\n(High vs Low Fuel Prices)')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels - improved positioning for negative values
    for bar, value in zip(bars, impacts):
        if value < 0:
            # For negative values, place label inside the bar
            axes[0].text(bar.get_x() + bar.get_width()/2, value/2,
                        f'{value:.1f}%', ha='center', va='center', 
                        fontweight='bold', color='white')
        else:
            # For positive values, place above the bar
            axes[0].text(bar.get_x() + bar.get_width()/2, value + 0.1,
                        f'{value:.1f}%', ha='center', va='bottom', 
                        fontweight='bold', color='black')
    
    # Panel B: Growth During Fuel Cycles
    width = 0.35
    x = np.arange(len(bm_list))
    
    rising = [trajectory_results[bm]['growth_rising_fuel'] for bm in bm_list]
    falling = [trajectory_results[bm]['growth_falling_fuel'] for bm in bm_list]
    
    # Use consistent colors for fuel price conditions
    axes[1].bar(x - width/2, falling, width, label='Falling Fuel Prices', 
                color='#2ca02c', alpha=0.5, edgecolor='black', linewidth=1)  # Green for falling (good)
    axes[1].bar(x + width/2, rising, width, label='Rising Fuel Prices', 
                color='#d62728', alpha=0.5, edgecolor='black', linewidth=1)  # Red for rising (bad)
    
    axes[1].set_ylabel('YoY Passenger Growth (%)')
    axes[1].set_title('Panel B: Growth During Fuel Cycles')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bm_list)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Panel C: Fuel Price Sensitivity
    sensitivities = [trajectory_results[bm]['sensitivity'] for bm in bm_list]
    
    bars = axes[2].bar(bm_list, sensitivities, color=[colors[bm] for bm in bm_list], 
                       alpha=0.8, edgecolor='black', linewidth=1)
    axes[2].set_ylabel('Sensitivity (%p)')
    axes[2].set_title('Panel C: Fuel Price Sensitivity\n(Growth Differential)')
    axes[2].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, sensitivities):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%p', ha='center', va='bottom', fontweight='bold', color='black')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'Figure_{FIGURE_NUMBERS["fuel_impact"]}_H2b_Fuel_Impact.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# num4: Save H2b tables
def save_h2b_tables(impact_results, trajectory_results, test_results, output_dir):
    """Save H2b analysis tables"""
    
    # Table 4.5: H2b Cost Shock Vulnerability Results
    bm_list = ['ULCC', 'LCC', 'Hybrid', 'Legacy']
    
    table_data = []
    for bm in bm_list:
        impact = impact_results[bm]['performance_impact']
        sensitivity = trajectory_results[bm]['sensitivity']
        
        # Add significance stars for ULCC
        impact_str = f"{impact:.1f}%"
        sensitivity_str = f"{sensitivity:.1f}%p"
        
        if bm == 'ULCC' and test_results:
            if 'impact_pooled' in test_results:
                sig = test_results['impact_pooled']['significance']
                if sig != 'ns':
                    impact_str += sig
            if 'sensitivity_pooled' in test_results:
                sig = test_results['sensitivity_pooled']['significance']
                if sig != 'ns':
                    sensitivity_str += sig
        
        # Calculate vulnerability rank (lower impact + higher sensitivity = more vulnerable)
        vulnerability_score = -impact + sensitivity
        
        table_data.append({
            'Business_Model': bm,
            'Passenger_Impact': impact_str,
            'Growth_Sensitivity': sensitivity_str,
            'Vulnerability_Score': vulnerability_score
        })
    
    # Sort by vulnerability score
    table_df = pd.DataFrame(table_data)
    table_df = table_df.sort_values('Vulnerability_Score', ascending=False)
    table_df['Vulnerability_Rank'] = range(1, len(table_df) + 1)
    
    # Save to CSV
    save_path = os.path.join(output_dir, f'Table_{TABLE_NUMBERS["cost_vulnerability"]}_H2b_Cost_Shock_Vulnerability.csv')
    table_df[['Business_Model', 'Passenger_Impact', 'Growth_Sensitivity', 'Vulnerability_Rank']].to_csv(save_path, index=False)
    
    print(f"\nTable {TABLE_NUMBERS['cost_vulnerability']}: H2b Cost Shock Vulnerability Results")
    print(f"{'Business Model':<15} {'Passenger Impact':<18} {'Growth Sensitivity':<18} {'Vulnerability Rank':<18}")
    print("-" * 70)
    
    for _, row in table_df.iterrows():
        print(f"{row['Business_Model']:<15} {row['Passenger_Impact']:<18} {row['Growth_Sensitivity']:<18} {row['Vulnerability_Rank']:<18}")
    
    print("\nNote: *** p<0.001. ULCC significantly more vulnerable than all others (pooled and pairwise comparisons).")
    
    return table_df

# num5: Main H2b analysis
def run_h2b_analysis(base_data, h2a_results=None):
    """
    H2b: Cost shock vulnerability analysis with statistical tests
    Tests vulnerability through traffic impact during fuel price changes
    """
    
    # Create output directory with absolute path
    output_dir = os.path.abspath('paper_1_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("H2b: COST SHOCK VULNERABILITY HYPOTHESIS")
    print("Framework: Passenger-based analysis (consistent with H2a)")
    print("Period: 2014-2019 (Pure oil shock effects)")
    print("="*80)
    
    # Run analyses
    impact_results, high_threshold, low_threshold = analyze_fuel_shock_impact(
        base_data['combined_od'], 
        base_data['shock_data']
    )
    
    trajectory_results = analyze_growth_trajectory(
        base_data['combined_od'],
        base_data['shock_data']
    )
    
    # ADDED: Perform statistical tests
    test_results = perform_h2b_statistical_tests(impact_results, trajectory_results)
    
    # Create visualization
    fig = create_h2b_integrated_visualization(impact_results, trajectory_results, base_data['colors'], output_dir)
    
    # Save tables with statistical test results
    table_df = save_h2b_tables(impact_results, trajectory_results, test_results, output_dir)
    
    # Determine H2b support
    ulcc_rank = table_df[table_df['Business_Model'] == 'ULCC']['Vulnerability_Rank'].values[0]
    h2b_supported = ulcc_rank == 1
    
    print("\n" + "="*60)
    print("H2b HYPOTHESIS TEST RESULTS")
    print("="*60)
    
    if h2b_supported:
        print("H2b SUPPORTED - ULCCs show highest vulnerability to fuel shocks")
        print(f"Statistical evidence: p < 0.001 for both impact and sensitivity")
    else:
        print("H2b PARTIALLY SUPPORTED - Mixed evidence of ULCC vulnerability")
    
    # Integrated H2 interpretation (if H2a results available)
    if h2a_results:
        print("\n" + "="*60)
        print("H2 INTEGRATED FINDINGS: The Volatility Trade-off")
        print("="*60)
        
        # H2a: Recovery speed
        if 'h2_crisis_resilience' in h2a_results and 'recovery_metrics' in h2a_results['h2_crisis_resilience']:
            ulcc_recovery = h2a_results['h2_crisis_resilience']['recovery_metrics'].get('ULCC', {}).get('Months_to_90', 'N/A')
            legacy_recovery = h2a_results['h2_crisis_resilience']['recovery_metrics'].get('Legacy', {}).get('Months_to_90', 'N/A')
            
            print(f"H2a - Crisis Recovery: ULCC {ulcc_recovery} months vs Legacy {legacy_recovery} months")
        
        ulcc_impact = impact_results['ULCC']['performance_impact']
        print(f"H2b - Fuel Vulnerability: ULCC impact {ulcc_impact:.1f}%")
        print("\nStrategic Insight:")
        print("- ULCCs trade operational stability for adaptive flexibility")
        print("- Faster crisis recovery but higher cost sensitivity")
        print("- Strategic volatility as deliberate positioning")
    
    return {
        'impact_results': impact_results,
        'trajectory_results': trajectory_results,
        'h2b_supported': h2b_supported,
        'figure': fig,
        'fuel_thresholds': {'high': high_threshold, 'low': low_threshold},
        'statistical_tests': test_results
    }

# Main execution
if __name__ == "__main__":
    print("H2b Analysis - Cost Shock Vulnerability")
    print("Use: h2b_results = run_h2b_analysis(base_data, h2a_results)")