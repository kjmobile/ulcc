#num1: Import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import os

#num2: Data preparation helper functions
def calculate_route_market_shares(od_data, classification_map):
    """Helper function to calculate route-level market shares"""
    
    # Apply business model classification
    od_data['Business_Model'] = od_data['Mkt'].map(classification_map)
    od_data = od_data.dropna(subset=['Business_Model'])
    
    valid_types = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    od_data = od_data[od_data['Business_Model'].isin(valid_types)]
    
    # Calculate route-level market shares
    route_shares = od_data.groupby(['Org', 'Dst', 'Business_Model']).agg({
        'Passengers': 'sum'
    }).reset_index()
    
    route_totals = route_shares.groupby(['Org', 'Dst'])['Passengers'].sum().reset_index()
    route_totals.rename(columns={'Passengers': 'Total_Passengers'}, inplace=True)
    
    route_shares = route_shares.merge(route_totals, on=['Org', 'Dst'])
    route_shares['Market_Share'] = route_shares['Passengers'] / route_shares['Total_Passengers']
    
    return route_shares[['Org', 'Dst', 'Business_Model', 'Market_Share', 'Passengers']]

def prepare_recent_panel_data(base_data):
    """Prepare panel data for H4a & H4b (2022-2024 only)"""
    
    print("\n=== PREPARING H4a & H4b PANEL DATA ===")
    
    od_years = base_data['od_years']
    t100_years = base_data['t100_years']
    classification_map = base_data['classification_map']
    
    # Recent years only for competitive analysis
    recent_years = [2022, 2023, 2024]
    print(f"Processing years: {recent_years}")
    
    all_data = []
    
    for year in recent_years:
        if year not in od_years:
            print(f"Missing OD data for {year}")
            continue
            
        print(f"Processing {year}...")
        
        # Process market shares
        od_data = od_years[year].copy()
        route_shares = calculate_route_market_shares(od_data, classification_map)
        
        # Add year identifier
        route_shares['Year'] = year
        route_shares['Route_ID'] = route_shares['Org'].astype(str) + '-' + route_shares['Dst'].astype(str)
        
        # Get T-100 data if available
        if year in t100_years:
            t100_data = t100_years[year].copy()
            t100_data.rename(columns={'Orig': 'Org', 'Dest': 'Dst'}, inplace=True)
            
            # Route-level T-100 aggregations
            route_t100 = t100_data.groupby(['Org', 'Dst']).agg({
                'Load Factor': 'mean',
                'ASMs': 'sum',
                'Onboards': 'sum'
            }).reset_index()
            
            route_t100['Route_ID'] = route_t100['Org'].astype(str) + '-' + route_t100['Dst'].astype(str)
            
            # Merge with market share data
            route_shares = route_shares.merge(
                route_t100[['Route_ID', 'Load Factor', 'ASMs', 'Onboards']], 
                on='Route_ID', how='left'
            )
        
        all_data.append(route_shares)
    
    if not all_data:
        print("No data available for panel construction")
        return None
    
    # Combine all years
    panel_data = pd.concat(all_data, ignore_index=True)
    
    # Create wide format for market shares
    shares_wide = panel_data.pivot_table(
        index=['Route_ID', 'Org', 'Dst', 'Year'], 
        columns='Business_Model', 
        values='Market_Share', 
        fill_value=0
    ).reset_index()
    
    # Add T-100 data back
    t100_cols = ['Load Factor', 'ASMs', 'Onboards']
    t100_data = panel_data[['Route_ID', 'Year'] + t100_cols].drop_duplicates()
    
    panel_final = shares_wide.merge(t100_data, on=['Route_ID', 'Year'], how='left')
    
    # Calculate additional metrics
    panel_final['Total_Passengers'] = panel_data.groupby(['Route_ID', 'Year'])['Passengers'].sum().values
    panel_final['HHI'] = (panel_final[['Legacy', 'ULCC', 'LCC', 'Hybrid']] ** 2).sum(axis=1)
    panel_final['Has_ULCC'] = panel_final['ULCC'] > 0
    panel_final['Num_Carriers'] = (panel_final[['Legacy', 'ULCC', 'LCC', 'Hybrid']] > 0).sum(axis=1)
    
    print(f"H4a/H4b panel data created: {len(panel_final)} route-year observations")
    print(f"Unique routes: {panel_final['Route_ID'].nunique()}")
    
    return panel_final

#num3: H4a & H4b analysis with statistical tests
def analyze_h4a_h4b_with_tests(panel_data, base_data):
    """Analyze market concentration and load factor effects with statistical tests"""
    
    print("\n=== H4a & H4b: STATISTICAL ANALYSIS ===")
    
    # Clean data
    recent_data = panel_data[
        (panel_data['Load Factor'] > 0) & 
        (panel_data['Load Factor'] <= 100) &
        (panel_data['Total_Passengers'] >= 1000) &
        panel_data['HHI'].notna()
    ].copy()
    
    print(f"Routes analyzed: {len(recent_data)}")
    print(f"Routes with ULCC: {recent_data['Has_ULCC'].sum()}")
    
    # H4a: Market Concentration Analysis
    print("\n--- H4a: Market Concentration (HHI) ---")
    
    no_ulcc_hhi = recent_data[~recent_data['Has_ULCC']]['HHI']
    with_ulcc_hhi = recent_data[recent_data['Has_ULCC']]['HHI']
    
    hhi_mean_no_ulcc = no_ulcc_hhi.mean()
    hhi_mean_with_ulcc = with_ulcc_hhi.mean()
    
    # Independent t-test for HHI
    t_stat_hhi, p_val_hhi = stats.ttest_ind(no_ulcc_hhi, with_ulcc_hhi)
    
    print(f"Average HHI without ULCC: {hhi_mean_no_ulcc:.3f}")
    print(f"Average HHI with ULCC: {hhi_mean_with_ulcc:.3f}")
    print(f"Difference: {hhi_mean_with_ulcc - hhi_mean_no_ulcc:.3f}")
    print(f"T-statistic: {t_stat_hhi:.3f}, P-value: {p_val_hhi:.3f}")
    print(f"Statistically significant: {'Yes' if p_val_hhi < 0.05 else 'No'}")
    
    # H4b: Load Factor Analysis (INCUMBENT CARRIERS ONLY)
    print("\n--- H4b: Load Factor Impact (Incumbent Analysis) ---")
    
    # Filter to incumbent carriers only (Legacy + Hybrid)
    incumbent_carriers = base_data['classification_map']
    legacy_hybrid = {k: v for k, v in incumbent_carriers.items() if v in ['Legacy', 'Hybrid']}
    
    # Get incumbent-only load factor data from T-100
    incumbent_lf_data = []
    for year in [2022, 2023, 2024]:
        if year in base_data['t100_years']:
            t100_year = base_data['t100_years'][year].copy()
            t100_year.rename(columns={'Orig': 'Org', 'Dest': 'Dst', 'Mkt Al': 'Mkt'}, inplace=True)
            
            # Filter to incumbent carriers only
            incumbent_t100 = t100_year[t100_year['Mkt'].isin(legacy_hybrid.keys())]
            
            if len(incumbent_t100) > 0:
                # Route-level incumbent load factors
                incumbent_routes = incumbent_t100.groupby(['Org', 'Dst']).agg({
                    'Load Factor': 'mean',
                    'Onboards': 'sum'
                }).reset_index()
                incumbent_routes['Route_ID'] = incumbent_routes['Org'].astype(str) + '-' + incumbent_routes['Dst'].astype(str)
                incumbent_routes['Year'] = year
                incumbent_lf_data.append(incumbent_routes)
    
    if incumbent_lf_data:
        incumbent_panel = pd.concat(incumbent_lf_data, ignore_index=True)
        
        # Merge with ULCC presence data
        recent_routes = recent_data[['Route_ID', 'Has_ULCC', 'Year']].drop_duplicates()
        incumbent_analysis = incumbent_panel.merge(recent_routes, on=['Route_ID', 'Year'], how='inner')
        
        # Clean data
        incumbent_analysis = incumbent_analysis[
            (incumbent_analysis['Load Factor'] > 0) & 
            (incumbent_analysis['Load Factor'] <= 100) &
            incumbent_analysis['Load Factor'].notna()
        ]
        
        if len(incumbent_analysis) > 0:
            no_ulcc_lf = incumbent_analysis[~incumbent_analysis['Has_ULCC']]['Load Factor']
            with_ulcc_lf = incumbent_analysis[incumbent_analysis['Has_ULCC']]['Load Factor']
            
            lf_mean_no_ulcc = no_ulcc_lf.mean()
            lf_mean_with_ulcc = with_ulcc_lf.mean()
            
            # Independent t-test for Load Factor (Incumbent only)
            t_stat_lf, p_val_lf = stats.ttest_ind(no_ulcc_lf, with_ulcc_lf)
            
            print(f"Incumbent Load Factor without ULCC: {lf_mean_no_ulcc:.1f}%")
            print(f"Incumbent Load Factor with ULCC: {lf_mean_with_ulcc:.1f}%")
            print(f"Difference: {lf_mean_with_ulcc - lf_mean_no_ulcc:.1f} percentage points")
            print(f"T-statistic: {t_stat_lf:.3f}, P-value: {p_val_lf:.3f}")
            print(f"Statistically significant: {'Yes' if p_val_lf < 0.05 else 'No'}")
            print(f"Routes analyzed (incumbent): {len(incumbent_analysis)}")
        else:
            print("No incumbent data available for load factor analysis")
            lf_mean_no_ulcc = lf_mean_with_ulcc = t_stat_lf = p_val_lf = 0
    else:
        print("No T-100 data available for incumbent analysis")
        lf_mean_no_ulcc = lf_mean_with_ulcc = t_stat_lf = p_val_lf = 0
    
    # Correlation analysis for ULCC routes
    ulcc_routes = recent_data[recent_data['ULCC'] > 0]
    
    correlations = {}
    if len(ulcc_routes) > 10:
        corr_lf, p_lf = pearsonr(ulcc_routes['ULCC'], ulcc_routes['Load Factor'])
        corr_hhi, p_hhi = pearsonr(ulcc_routes['ULCC'], ulcc_routes['HHI'])
        
        correlations = {
            'ULCC_LF_corr': corr_lf,
            'ULCC_LF_p': p_lf,
            'ULCC_HHI_corr': corr_hhi,
            'ULCC_HHI_p': p_hhi
        }
        
        print(f"\nCorrelation between ULCC share and Load Factor: {corr_lf:.3f} (p={p_lf:.3f})")
        print(f"Correlation between ULCC share and HHI: {corr_hhi:.3f} (p={p_hhi:.3f})")
    
    results = {
        'HHI_no_ULCC': hhi_mean_no_ulcc,
        'HHI_with_ULCC': hhi_mean_with_ulcc,
        'HHI_difference': hhi_mean_with_ulcc - hhi_mean_no_ulcc,
        'HHI_t_stat': t_stat_hhi,
        'HHI_p_value': p_val_hhi,
        'LF_no_ULCC': lf_mean_no_ulcc,
        'LF_with_ULCC': lf_mean_with_ulcc,
        'LF_difference': lf_mean_with_ulcc - lf_mean_no_ulcc,
        'LF_t_stat': t_stat_lf,
        'LF_p_value': p_val_lf,
        'routes_analyzed': len(recent_data),
        'routes_with_ULCC': recent_data['Has_ULCC'].sum(),
        'correlations': correlations
    }
    
    return results, recent_data

#num4: H4a & H4b visualization
def create_h4ab_figure(h4ab_results, panel_data):
    """Create H4a & H4b visualization - Market Competition Effects"""
    
    os.makedirs('paper_1_outputs', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Import colors
    from basecode import CARRIER_COLORS
    
    # Define consistent color scheme for ULCC presence comparison
    no_ulcc_color = '#E8F4FD'  # Light blue consistent with other analyses
    ulcc_color = CARRIER_COLORS['ULCC']
    
    # Get data for effect size calculation
    recent_data = panel_data[
        (panel_data['Load Factor'] > 0) & 
        (panel_data['Load Factor'] <= 100) &
        (panel_data['Total_Passengers'] >= 1000) &
        panel_data['HHI'].notna()
    ].copy()
    
    no_ulcc_hhi = recent_data[~recent_data['Has_ULCC']]['HHI']
    with_ulcc_hhi = recent_data[recent_data['Has_ULCC']]['HHI']
    
    # Panel A: HHI comparison with significance and effect size
    hhi_data = [h4ab_results['HHI_no_ULCC'], h4ab_results['HHI_with_ULCC']]
    bars1 = axes[0].bar(['No ULCC', 'With ULCC'], hhi_data, 
                         color=[no_ulcc_color, ulcc_color], alpha=0.8, 
                         edgecolor='black', linewidth=0.5, width=0.6)
    axes[0].set_title('Panel A: Market Concentration (HHI)', fontweight='bold', pad=15)
    axes[0].set_ylabel('Average HHI')
    axes[0].set_ylim(0, 1.0)
    
    # Calculate Cohen's d for effect size
    pooled_std = np.sqrt(((len(no_ulcc_hhi)-1)*no_ulcc_hhi.std()**2 + (len(with_ulcc_hhi)-1)*with_ulcc_hhi.std()**2) / 
                        (len(no_ulcc_hhi) + len(with_ulcc_hhi) - 2))
    cohens_d = abs(h4ab_results['HHI_with_ULCC'] - h4ab_results['HHI_no_ULCC']) / pooled_std
    
    if h4ab_results['HHI_p_value'] < 0.001:
        sig_text = f"p < 0.001***, d = {cohens_d:.2f}"
    elif h4ab_results['HHI_p_value'] < 0.01:
        sig_text = f"p = {h4ab_results['HHI_p_value']:.3f}**, d = {cohens_d:.2f}"
    elif h4ab_results['HHI_p_value'] < 0.05:
        sig_text = f"p = {h4ab_results['HHI_p_value']:.3f}*, d = {cohens_d:.2f}"
    else:
        sig_text = f"p = {h4ab_results['HHI_p_value']:.3f}, d = {cohens_d:.2f}"
    
    axes[0].text(0.5, 1.01, sig_text, ha='center', va='center', 
                   transform=axes[0].transAxes, fontsize=10, fontweight='bold')
    
    for i, v in enumerate(hhi_data):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Panel B: Load Factor comparison with significance and effect size
    lf_data = [h4ab_results['LF_no_ULCC'], h4ab_results['LF_with_ULCC']]
    bars2 = axes[1].bar(['No ULCC', 'With ULCC'], lf_data, 
                         color=[no_ulcc_color, ulcc_color], alpha=0.8, 
                         edgecolor='black', linewidth=0.5, width=0.6)
    axes[1].set_title('Panel B: Load Factor Impact', fontweight='bold', pad=15)
    axes[1].set_ylabel('Average Load Factor (%)')
    axes[1].set_ylim(0, 100)
    
    # Approximate Cohen's d calculation for Load Factor
    lf_diff = abs(h4ab_results['LF_with_ULCC'] - h4ab_results['LF_no_ULCC'])
    cohens_d_lf = lf_diff / 15.0  # Typical load factor std dev
    
    if h4ab_results['LF_p_value'] < 0.001:
        sig_text = f"p < 0.001***, d ≈ {cohens_d_lf:.2f}"
    elif h4ab_results['LF_p_value'] < 0.01:
        sig_text = f"p = {h4ab_results['LF_p_value']:.3f}**, d ≈ {cohens_d_lf:.2f}"
    elif h4ab_results['LF_p_value'] < 0.05:
        sig_text = f"p = {h4ab_results['LF_p_value']:.3f}*, d ≈ {cohens_d_lf:.2f}"
    else:
        sig_text = f"p = {h4ab_results['LF_p_value']:.3f}, d ≈ {cohens_d_lf:.2f}"
    
    axes[1].text(0.5, 1.01, sig_text, ha='center', va='center', 
                   transform=axes[1].transAxes, fontsize=10, fontweight='bold')
    
    for i, v in enumerate(lf_data):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Panel C: Enhanced scatter plot with bubble features
    if h4ab_results['correlations'] and panel_data is not None:
        ulcc_data = panel_data[panel_data['ULCC'] > 0]
        
        if len(ulcc_data) > 0:
            # Bubble size based on Total_Passengers (market size)
            sizes = ulcc_data['Total_Passengers'] / 1000
            sizes = np.clip(sizes, 10, 200)
            
            # Color intensity based on HHI (market concentration)
            colors = ulcc_data['HHI']
            
            scatter = axes[2].scatter(ulcc_data['ULCC'] * 100, ulcc_data['Load Factor'], 
                            s=sizes, c=colors, alpha=0.6, 
                            cmap='Reds', edgecolor='w', linewidth=0.5)
            
            axes[2].set_xlabel('ULCC Market Share (%)')
            axes[2].set_ylabel('Load Factor (%)')
            axes[2].set_title('Panel C: ULCC Share vs Load Factor', fontweight='bold', pad=15)
            
            # Add colorbar for HHI
            cbar = plt.colorbar(scatter, ax=axes[2], shrink=0.8)
            cbar.set_label('Market Concentration (HHI)', fontsize=9)
            
            # Add correlation info
            corr = h4ab_results['correlations']['ULCC_LF_corr']
            p_val = h4ab_results['correlations']['ULCC_LF_p']
            
            if p_val < 0.001:
                corr_text = f"r = {corr:.3f}, p < 0.001***"
            elif p_val < 0.01:
                corr_text = f"r = {corr:.3f}, p = {p_val:.3f}**"
            elif p_val < 0.05:
                corr_text = f"r = {corr:.3f}, p = {p_val:.3f}*"
            else:
                corr_text = f"r = {corr:.3f}, p = {p_val:.3f}"
            
            axes[2].text(0.5, 1.01, corr_text, ha='center', va='center', 
                        transform=axes[2].transAxes, fontsize=10, fontweight='bold')
            
            # Add size legend
            sizes_legend = [50, 100, 150]
            labels_legend = ['Small', 'Medium', 'Large']
            legend_elements = [plt.scatter([], [], s=s, c='gray', alpha=0.6, edgecolor='w') 
                             for s in sizes_legend]
            legend1 = axes[2].legend(legend_elements, labels_legend, 
                                   title='Market Size', loc='lower right', fontsize=8)
            legend1.get_title().set_fontsize(9)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('paper_1_outputs/Figure_4.8_H4ab_Market_Competition.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

#num5: Main H4a & H4b analysis function
def run_h4ab_analysis(base_data):
    """Run H4a & H4b analysis - Market Competition Effects"""
    
    print("RUNNING H4a & H4b: MARKET COMPETITION ANALYSIS")
    print("=" * 50)
    
    # Step 1: Prepare panel data for recent years
    panel_data = prepare_recent_panel_data(base_data)
    
    if panel_data is None:
        print("Failed to prepare panel data")
        return None
    
    # Step 2: H4a & H4b analysis with statistical tests
    h4ab_results, recent_data = analyze_h4a_h4b_with_tests(panel_data, base_data)
    
    # Step 3: Create visualization
    fig = create_h4ab_figure(h4ab_results, panel_data)
    
    # Step 4: Save results
    os.makedirs('paper_1_outputs', exist_ok=True)
    
    # Save H4a & H4b results
    h4ab_df = pd.DataFrame([h4ab_results])
    
    # Display Table 4.7 results
    print("\n=== TABLE 4.7: H4a & H4b Statistical Results (ULCC Competitive Impact Analysis) ===")
    print(h4ab_df.round(3).to_string(index=False))

    h4ab_df.to_csv('paper_1_outputs/Table_4.7_H4ab_Statistical_Results.csv', index=False)
    print(f"\nTable 4.7 saved: paper_1_outputs/Table_4.7_H4ab_Statistical_Results.csv")
    
    # Save panel data
    panel_data.to_csv('paper_1_outputs/H4ab_Panel_Data.csv', index=False)
    
    print("\n" + "=" * 50)
    print("H4a & H4b ANALYSIS COMPLETE!")
    print("Results saved in 'paper_1_outputs/' directory")
    print("- Statistical tests with p-values included")
    print("- Incumbent-only analysis for load factor effects")
    print("- Figure 4.6 created with significance indicators")
    print("=" * 50)
    
    return {
        'h4ab_results': h4ab_results,
        'panel_data': panel_data,
        'figure': fig
    }

if __name__ == "__main__":
    from basecode import prepare_base_data
    base_data = prepare_base_data()
    if base_data:
        h4ab_results = run_h4ab_analysis(base_data)