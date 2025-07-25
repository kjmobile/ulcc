#num1: Import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

#num2: Competitive impact analysis (H3)
def analyze_competitive_impact_h3(base_data):
    """Analyze ULCC competitive effects on incumbents"""
    
    print("\n=== H3: COMPETITIVE IMPACT ANALYSIS ===")
    
    od_years = base_data['od_years']
    t100_years = base_data['t100_years']
    classification_map = base_data['classification_map']
    
    # Focus on recent years with good data coverage
    analysis_years = [2022, 2023, 2024]
    
    all_route_data = []
    
    for year in analysis_years:
        if year not in od_years or year not in t100_years:
            continue
            
        # Process OD data
        od_data = od_years[year].copy()
        od_data['Business_Model'] = od_data['Mkt'].map(classification_map)
        od_data = od_data.dropna(subset=['Business_Model'])
        
        valid_types = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
        od_data = od_data[od_data['Business_Model'].isin(valid_types)]
        
        # Process T100 data
        t100_data = t100_years[year].copy()
        t100_data.rename(columns={'Orig': 'Org', 'Dest': 'Dst'}, inplace=True)
        
        # Route-level aggregations
        route_od = od_data.groupby(['Org', 'Dst']).agg({
            'Passengers': 'sum'
        }).reset_index()
        
        route_t100 = t100_data.groupby(['Org', 'Dst']).agg({
            'Load Factor': 'mean',
            'Onboards': 'sum'
        }).reset_index()
        
        # Market shares calculation
        route_shares = od_data.groupby(['Org', 'Dst', 'Business_Model']).agg({
            'Passengers': 'sum'
        }).reset_index()
        
        route_totals = route_shares.groupby(['Org', 'Dst'])['Passengers'].sum().reset_index()
        route_totals.rename(columns={'Passengers': 'Total_Passengers'}, inplace=True)
        
        route_shares = route_shares.merge(route_totals, on=['Org', 'Dst'])
        route_shares['Market_Share'] = route_shares['Passengers'] / route_shares['Total_Passengers']
        
        # ULCC presence calculation
        ulcc_routes = route_shares[route_shares['Business_Model'] == 'ULCC']
        ulcc_by_route = ulcc_routes.groupby(['Org', 'Dst']).agg({
            'Market_Share': 'sum'
        }).reset_index()
        ulcc_by_route.rename(columns={'Market_Share': 'ULCC_Share'}, inplace=True)
        
        # HHI calculation
        hhi_data = route_shares.groupby(['Org', 'Dst']).apply(
            lambda x: (x['Market_Share'] ** 2).sum(), include_groups=False
        ).reset_index()
        hhi_data.rename(columns={0: 'HHI'}, inplace=True)
        
        # Combine datasets
        route_analysis = route_od.merge(route_t100, on=['Org', 'Dst'], how='inner')
        route_analysis = route_analysis.merge(ulcc_by_route, on=['Org', 'Dst'], how='left')
        route_analysis = route_analysis.merge(hhi_data, on=['Org', 'Dst'], how='left')
        
        # Feature engineering
        route_analysis['ULCC_Share'] = route_analysis['ULCC_Share'].fillna(0)
        route_analysis['Has_ULCC'] = route_analysis['ULCC_Share'] > 0
        route_analysis['Year'] = year
        
        all_route_data.append(route_analysis)
    
    if not all_route_data:
        print("No data available for competitive analysis")
        return None, None
    
    # Combine and filter data
    combined_analysis = pd.concat(all_route_data, ignore_index=True)
    
    valid_mask = (
        (combined_analysis['Load Factor'] > 0) & 
        (combined_analysis['Load Factor'] <= 100) &
        (combined_analysis['Passengers'] >= 1000) &
        combined_analysis['HHI'].notna()
    )
    combined_analysis = combined_analysis[valid_mask]
    
    print(f"\nTotal routes analyzed: {len(combined_analysis)}")
    ulcc_routes_count = combined_analysis['Has_ULCC'].sum()
    print(f"Routes with ULCC: {ulcc_routes_count}")
    print(f"Routes without ULCC: {len(combined_analysis) - ulcc_routes_count}")
    
    # H3a: Market concentration
    hhi_comparison = combined_analysis.groupby('Has_ULCC')['HHI'].mean()
    hhi_with_ulcc = hhi_comparison.get(True, 0)
    hhi_without_ulcc = hhi_comparison.get(False, 0)
    
    print(f"\nH3a - Market Concentration:")
    print(f"Average HHI with ULCC: {hhi_with_ulcc:.3f}")
    print(f"Average HHI without ULCC: {hhi_without_ulcc:.3f}")
    
    # H3b: Load factor impact
    lf_comparison = combined_analysis.groupby('Has_ULCC')['Load Factor'].mean()
    lf_with_ulcc = lf_comparison.get(True, 0)
    lf_without_ulcc = lf_comparison.get(False, 0)
    
    print(f"\nH3b - Load Factor Impact:")
    print(f"Average Load Factor with ULCC: {lf_with_ulcc:.1f}%")
    print(f"Average Load Factor without ULCC: {lf_without_ulcc:.1f}%")
    
    # Correlation analysis
    ulcc_routes_data = combined_analysis[combined_analysis['ULCC_Share'] > 0]
    
    correlations = {}
    if len(ulcc_routes_data) > 10:
        corr_lf, p_lf = pearsonr(ulcc_routes_data['ULCC_Share'], ulcc_routes_data['Load Factor'])
        corr_hhi, p_hhi = pearsonr(ulcc_routes_data['ULCC_Share'], ulcc_routes_data['HHI'])
        
        correlations = {
            'ULCC_LF_corr': corr_lf,
            'ULCC_LF_p': p_lf,
            'ULCC_HHI_corr': corr_hhi,
            'ULCC_HHI_p': p_hhi
        }
        
        print(f"\nCorrelation between ULCC share and Load Factor: {corr_lf:.3f} (p={p_lf:.3f})")
        print(f"Correlation between ULCC share and HHI: {corr_hhi:.3f} (p={p_hhi:.3f})")
    
    competitive_results = {
        'HHI_with_ULCC': hhi_with_ulcc,
        'HHI_without_ULCC': hhi_without_ulcc,
        'LF_with_ULCC': lf_with_ulcc,
        'LF_without_ULCC': lf_without_ulcc,
        'Routes_analyzed': len(combined_analysis),
        'Routes_with_ULCC': ulcc_routes_count,
        'correlations': correlations
    }
    
    return competitive_results, combined_analysis

#num3: Create H3 visualization
def create_h3_figure(competitive_results, combined_analysis):
    """Create competitive impact visualization"""
    
    os.makedirs('figures', exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('H3: Competitive Impact Analysis', fontsize=16, fontweight='bold')
    
    # Import colors from basecode
    from basecode import CARRIER_COLORS
    ulcc_color = CARRIER_COLORS['ULCC']
    
    # H3a: HHI comparison
    hhi_data = [competitive_results['HHI_without_ULCC'], competitive_results['HHI_with_ULCC']]
    bars1 = axes[0,0].bar(['No ULCC', 'With ULCC'], hhi_data, 
                         color=['lightblue', ulcc_color], alpha=0.8, 
                         edgecolor='black', linewidth=0.5, width=0.6)
    axes[0,0].set_title('H3a: Market Concentration (HHI)', fontweight='bold', pad=15)
    axes[0,0].set_ylabel('Average HHI')
    
    # Add value labels on bars
    for i, v in enumerate(hhi_data):
        axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # H3b: Load Factor comparison
    lf_data = [competitive_results['LF_without_ULCC'], competitive_results['LF_with_ULCC']]
    bars2 = axes[0,1].bar(['No ULCC', 'With ULCC'], lf_data, 
                         color=['lightblue', ulcc_color], alpha=0.8, 
                         edgecolor='black', linewidth=0.5, width=0.6)
    axes[0,1].set_title('H3b: Load Factor Impact', fontweight='bold', pad=15)
    axes[0,1].set_ylabel('Average Load Factor (%)')
    
    # Add value labels on bars
    for i, v in enumerate(lf_data):
        axes[0,1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Route distribution pie chart
    route_counts = [
        competitive_results['Routes_analyzed'] - competitive_results['Routes_with_ULCC'],
        competitive_results['Routes_with_ULCC']
    ]
    axes[0,2].pie(route_counts, labels=['No ULCC', 'With ULCC'], 
                  colors=['lightblue', ulcc_color], alpha=0.8, autopct='%1.1f%%',
                  startangle=90, textprops={'fontsize': 9})
    axes[0,2].set_title('Route Distribution', fontweight='bold', pad=15)
    
    # Scatter plots for detailed analysis
    if combined_analysis is not None and len(combined_analysis) > 0:
        # ULCC share vs Load Factor
        ulcc_data = combined_analysis[combined_analysis['ULCC_Share'] > 0]
        if len(ulcc_data) > 0:
            axes[1,0].scatter(ulcc_data['ULCC_Share'] * 100, ulcc_data['Load Factor'], 
                            alpha=0.6, color=ulcc_color, s=20)
            axes[1,0].set_xlabel('ULCC Market Share (%)')
            axes[1,0].set_ylabel('Load Factor (%)')
            axes[1,0].set_title('ULCC Share vs Load Factor', fontweight='bold', pad=15)
        
        # ULCC share vs HHI
        if len(ulcc_data) > 0:
            axes[1,1].scatter(ulcc_data['ULCC_Share'] * 100, ulcc_data['HHI'], 
                            alpha=0.6, color=ulcc_color, s=20)
            axes[1,1].set_xlabel('ULCC Market Share (%)')
            axes[1,1].set_ylabel('HHI')
            axes[1,1].set_title('ULCC Share vs Market Concentration', fontweight='bold', pad=15)
        
        # Load Factor distribution comparison
        no_ulcc = combined_analysis[combined_analysis['Has_ULCC'] == False]['Load Factor']
        with_ulcc = combined_analysis[combined_analysis['Has_ULCC'] == True]['Load Factor']
        
        axes[1,2].hist(no_ulcc, bins=30, alpha=0.7, label='No ULCC', 
                      color='lightblue', density=True, edgecolor='black', linewidth=0.5)
        axes[1,2].hist(with_ulcc, bins=30, alpha=0.7, label='With ULCC', 
                      color=ulcc_color, density=True, edgecolor='black', linewidth=0.5)
        axes[1,2].set_xlabel('Load Factor (%)')
        axes[1,2].set_ylabel('Density')
        axes[1,2].set_title('Load Factor Distribution', fontweight='bold', pad=15)
        axes[1,2].legend(frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save in both formats
    plt.savefig('figures/Figure_H3_Competitive_Impact.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figures/Figure_H3_Competitive_Impact.eps', format='eps', bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

#num4: Main H3 analysis function
def run_h3_analysis(base_data):
    """Run complete H3 analysis"""
    
    print("RUNNING H3: COMPETITIVE IMPACT ANALYSIS")
    print("=" * 50)
    
    # Main competitive impact analysis
    competitive_results, combined_analysis = analyze_competitive_impact_h3(base_data)
    
    if competitive_results is None:
        print("H3 analysis failed - no data available")
        return None
    
    # Create visualization
    fig = create_h3_figure(competitive_results, combined_analysis)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Save competitive results as DataFrame
    results_df = pd.DataFrame([competitive_results])
    results_df.to_csv('results/H3_Competitive_Impact_Results.csv', index=False)
    
    # Save detailed route analysis
    if combined_analysis is not None:
        combined_analysis.to_csv('results/H3_Route_Analysis_Data.csv', index=False)
    
    print("\nH3 Analysis Complete!")
    print("Results saved in 'results/' directory")
    print("Figures saved in 'figures/' directory")
    
    return {
        'competitive_results': competitive_results,
        'route_data': combined_analysis,
        'figure': fig
    }

if __name__ == "__main__":
    from basecode import prepare_base_data
    base_data = prepare_base_data()
    if base_data:
        h3_results = run_h3_analysis(base_data)