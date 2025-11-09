#num1: Import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

#num2: Data preparation for COVID analysis
def prepare_covid_panel_data(base_data):
    """Prepare panel data for H4c & H4d (2019, 2023 only)"""
    
    print("\n=== PREPARING H4c & H4d COVID PANEL DATA ===")
    
    od_years = base_data['od_years']
    t100_years = base_data['t100_years']
    classification_map = base_data['classification_map']
    
    # COVID analysis years only
    covid_years = [2019, 2023]
    print(f"Processing COVID analysis years: {covid_years}")
    
    all_data = []
    
    for year in covid_years:
        if year not in od_years:
            print(f"Missing OD data for {year}")
            continue
            
        print(f"Processing {year}...")
        
        # Apply business model classification
        od_data = od_years[year].copy()
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
        
        # Add identifiers
        route_shares['Year'] = year
        route_shares['Route_ID'] = route_shares['Org'].astype(str) + '-' + route_shares['Dst'].astype(str)
        
        # Get T-100 data for capacity analysis
        if year in t100_years:
            t100_data = t100_years[year].copy()
            t100_data.rename(columns={'Orig': 'Org', 'Dest': 'Dst'}, inplace=True)
            
            route_t100 = t100_data.groupby(['Org', 'Dst']).agg({
                'ASMs': 'sum'
            }).reset_index()
            
            route_t100['Route_ID'] = route_t100['Org'].astype(str) + '-' + route_t100['Dst'].astype(str)
            
            # Merge with market share data
            route_shares = route_shares.merge(
                route_t100[['Route_ID', 'ASMs']], 
                on='Route_ID', how='left'
            )
        
        all_data.append(route_shares)
    
    if not all_data:
        print("No data available for COVID panel construction")
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
    
    # Add ASMs data back
    asms_data = panel_data[['Route_ID', 'Year', 'ASMs']].drop_duplicates()
    panel_final = shares_wide.merge(asms_data, on=['Route_ID', 'Year'], how='left')
    
    print(f"COVID panel data created: {len(panel_final)} route-year observations")
    print(f"Unique routes: {panel_final['Route_ID'].nunique()}")
    
    return panel_final

#num3: H4c & H4d analysis with DiD regression models (FIXED)
def analyze_h4c_h4d_with_did(panel_data):
    """Analyze COVID effects using Difference-in-Differences regression models"""
    
    print("\n=== H4c & H4d: COVID ANALYSIS WITH DiD REGRESSION ===")
    
    if panel_data is None or len(panel_data) == 0:
        print("No data available for COVID analysis")
        return None
    
    # Identify treatment and control groups for H4c
    routes_2019 = panel_data[panel_data['Year'] == 2019]
    ulcc_routes_2019 = set(routes_2019[routes_2019['ULCC'] > 0]['Route_ID'])
    
    panel_data['Had_ULCC_2019'] = panel_data['Route_ID'].isin(ulcc_routes_2019).astype(int)
    panel_data['Post_COVID'] = (panel_data['Year'] == 2023).astype(int)
    panel_data['Treatment_Post'] = (panel_data['Had_ULCC_2019'] & panel_data['Post_COVID']).astype(int)
    
    print(f"Routes with ULCC in 2019: {len(ulcc_routes_2019)}")
    
    # H4c: DiD Regression Analysis for existing ULCC routes
    print("\n--- H4c: DiD Regression Results (Existing ULCC Routes) ---")
    
    h4c_results = {}
    h4c_regression_results = {}
    
    for carrier in ['Legacy', 'ULCC', 'LCC', 'Hybrid']:
        if carrier in panel_data.columns:
            # Prepare regression data - FIXED: Remove Route_ID and ensure numeric types
            reg_data = panel_data[[carrier, 'Had_ULCC_2019', 'Post_COVID', 'Treatment_Post']].copy()
            reg_data = reg_data.dropna()
            
            # Ensure all columns are numeric
            for col in ['Had_ULCC_2019', 'Post_COVID', 'Treatment_Post']:
                reg_data[col] = pd.to_numeric(reg_data[col], errors='coerce')
            reg_data[carrier] = pd.to_numeric(reg_data[carrier], errors='coerce')
            reg_data = reg_data.dropna()
            
            if len(reg_data) > 10:
                # DiD Regression: Y = β0 + β1*Treatment + β2*Post + β3*(Treatment*Post) + ε
                X = reg_data[['Had_ULCC_2019', 'Post_COVID', 'Treatment_Post']].copy()
                X = sm.add_constant(X)  # Add intercept
                y = reg_data[carrier].copy()
                
                try:
                    model = sm.OLS(y, X).fit()
                    
                    # Extract DiD coefficient (β3 - Treatment*Post interaction)
                    did_coef = model.params['Treatment_Post']
                    did_pvalue = model.pvalues['Treatment_Post']
                    did_ci = model.conf_int().loc['Treatment_Post']
                    
                    # Manual calculation for comparison using original panel_data
                    pre_treatment = panel_data[(panel_data['Had_ULCC_2019'] == 1) & (panel_data['Post_COVID'] == 0)][carrier].mean()
                    post_treatment = panel_data[(panel_data['Had_ULCC_2019'] == 1) & (panel_data['Post_COVID'] == 1)][carrier].mean()
                    pre_control = panel_data[(panel_data['Had_ULCC_2019'] == 0) & (panel_data['Post_COVID'] == 0)][carrier].mean()
                    post_control = panel_data[(panel_data['Had_ULCC_2019'] == 0) & (panel_data['Post_COVID'] == 1)][carrier].mean()
                    
                    treatment_change = post_treatment - pre_treatment
                    control_change = post_control - pre_control
                    manual_did = treatment_change - control_change
                    
                    h4c_results[carrier] = {
                        'pre_covid': pre_treatment if not pd.isna(pre_treatment) else 0,
                        'post_covid': post_treatment if not pd.isna(post_treatment) else 0,
                        'treatment_change': treatment_change if not pd.isna(treatment_change) else 0,
                        'control_change': control_change if not pd.isna(control_change) else 0,
                        'did_effect': manual_did if not pd.isna(manual_did) else 0,
                        'did_coefficient': did_coef,
                        'did_pvalue': did_pvalue,
                        'did_ci_lower': did_ci[0],
                        'did_ci_upper': did_ci[1]
                    }
                    
                    h4c_regression_results[carrier] = model
                    
                    # Display results
                    sig_stars = ""
                    if did_pvalue < 0.001:
                        sig_stars = "***"
                    elif did_pvalue < 0.01:
                        sig_stars = "**"
                    elif did_pvalue < 0.05:
                        sig_stars = "*"
                    
                    print(f"{carrier}: DiD = {did_coef:+.3f}{sig_stars} (p={did_pvalue:.3f}) "
                          f"[{did_ci[0]:+.3f}, {did_ci[1]:+.3f}]")
                    
                except Exception as e:
                    print(f"{carrier}: Regression failed - {str(e)}")
                    h4c_results[carrier] = {
                        'pre_covid': 0, 'post_covid': 0, 'treatment_change': 0,
                        'control_change': 0, 'did_effect': 0, 'did_coefficient': 0,
                        'did_pvalue': 1, 'did_ci_lower': 0, 'did_ci_upper': 0
                    }
            else:
                print(f"{carrier}: Insufficient data for regression")
                h4c_results[carrier] = {
                    'pre_covid': 0, 'post_covid': 0, 'treatment_change': 0,
                    'control_change': 0, 'did_effect': 0, 'did_coefficient': 0,
                    'did_pvalue': 1, 'did_ci_lower': 0, 'did_ci_upper': 0
                }
    
    # H4d: Capacity reduction analysis with regression
    print("\n--- H4d: DiD Regression Results (Capacity-Reduced Routes) ---")
    
    h4d_results = {}
    h4d_regression_results = {}
    
    # Identify routes with capacity reduction
    if 'ASMs' in panel_data.columns:
        capacity_2019 = panel_data[panel_data['Year'] == 2019].groupby('Route_ID')['ASMs'].sum()
        capacity_2023 = panel_data[panel_data['Year'] == 2023].groupby('Route_ID')['ASMs'].sum()
        
        capacity_change = pd.merge(capacity_2019, capacity_2023, left_index=True, right_index=True, suffixes=('_2019', '_2023'))
        capacity_change['pct_change'] = (capacity_change['ASMs_2023'] - capacity_change['ASMs_2019']) / capacity_change['ASMs_2019']
        
        # Routes with >20% capacity reduction
        reduced_routes = set(capacity_change[capacity_change['pct_change'] < -0.2].index)
        panel_data['Capacity_Reduced'] = panel_data['Route_ID'].isin(reduced_routes).astype(int)
        panel_data['Treatment_Post_H4d'] = (panel_data['Capacity_Reduced'] & panel_data['Post_COVID']).astype(int)
        
        print(f"Routes with >20% capacity reduction: {len(reduced_routes)}")
        
        for carrier in ['Legacy', 'ULCC', 'LCC', 'Hybrid']:
            if carrier in panel_data.columns:
                # Prepare regression data - FIXED: Remove Route_ID and ensure numeric types
                reg_data = panel_data[[carrier, 'Capacity_Reduced', 'Post_COVID', 'Treatment_Post_H4d']].copy()
                reg_data = reg_data.dropna()
                
                # Ensure all columns are numeric
                for col in ['Capacity_Reduced', 'Post_COVID', 'Treatment_Post_H4d']:
                    reg_data[col] = pd.to_numeric(reg_data[col], errors='coerce')
                reg_data[carrier] = pd.to_numeric(reg_data[carrier], errors='coerce')
                reg_data = reg_data.dropna()
                
                if len(reg_data) > 10:
                    # DiD Regression
                    X = reg_data[['Capacity_Reduced', 'Post_COVID', 'Treatment_Post_H4d']].copy()
                    X = sm.add_constant(X)
                    y = reg_data[carrier].copy()
                    
                    try:
                        model = sm.OLS(y, X).fit()
                        
                        did_coef = model.params['Treatment_Post_H4d']
                        did_pvalue = model.pvalues['Treatment_Post_H4d']
                        did_ci = model.conf_int().loc['Treatment_Post_H4d']
                        
                        # Manual calculation using original panel_data
                        pre_reduced = panel_data[(panel_data['Capacity_Reduced'] == 1) & (panel_data['Post_COVID'] == 0)][carrier].mean()
                        post_reduced = panel_data[(panel_data['Capacity_Reduced'] == 1) & (panel_data['Post_COVID'] == 1)][carrier].mean()
                        pre_stable = panel_data[(panel_data['Capacity_Reduced'] == 0) & (panel_data['Post_COVID'] == 0)][carrier].mean()
                        post_stable = panel_data[(panel_data['Capacity_Reduced'] == 0) & (panel_data['Post_COVID'] == 1)][carrier].mean()
                        
                        treatment_change = post_reduced - pre_reduced
                        control_change = post_stable - pre_stable
                        manual_did = treatment_change - control_change
                        
                        h4d_results[carrier] = {
                            'pre_covid': pre_reduced if not pd.isna(pre_reduced) else 0,
                            'post_covid': post_reduced if not pd.isna(post_reduced) else 0,
                            'treatment_change': treatment_change if not pd.isna(treatment_change) else 0,
                            'control_change': control_change if not pd.isna(control_change) else 0,
                            'did_effect': manual_did if not pd.isna(manual_did) else 0,
                            'did_coefficient': did_coef,
                            'did_pvalue': did_pvalue,
                            'did_ci_lower': did_ci[0],
                            'did_ci_upper': did_ci[1]
                        }
                        
                        h4d_regression_results[carrier] = model
                        
                        # Display results
                        sig_stars = ""
                        if did_pvalue < 0.001:
                            sig_stars = "***"
                        elif did_pvalue < 0.01:
                            sig_stars = "**"
                        elif did_pvalue < 0.05:
                            sig_stars = "*"
                        
                        print(f"{carrier}: DiD = {did_coef:+.3f}{sig_stars} (p={did_pvalue:.3f}) "
                              f"[{did_ci[0]:+.3f}, {did_ci[1]:+.3f}]")
                        
                    except Exception as e:
                        print(f"{carrier}: Regression failed - {str(e)}")
                        h4d_results[carrier] = {
                            'pre_covid': 0, 'post_covid': 0, 'treatment_change': 0,
                            'control_change': 0, 'did_effect': 0, 'did_coefficient': 0,
                            'did_pvalue': 1, 'did_ci_lower': 0, 'did_ci_upper': 0
                        }
                else:
                    print(f"{carrier}: Insufficient data for regression")
                    h4d_results[carrier] = {
                        'pre_covid': 0, 'post_covid': 0, 'treatment_change': 0,
                        'control_change': 0, 'did_effect': 0, 'did_coefficient': 0,
                        'did_pvalue': 1, 'did_ci_lower': 0, 'did_ci_upper': 0
                    }
    else:
        print("No capacity data available for H4d analysis")
    
    return {
        'h4c_results': h4c_results, 
        'h4d_results': h4d_results,
        'h4c_models': h4c_regression_results,
        'h4d_models': h4d_regression_results
    }

#num4: H4c & H4d visualization (FIXED for 1x4 layout)
def create_h4cd_figure(covid_results, panel_data):
    """Create H4c & H4d visualization - COVID Impact Analysis"""
    
    os.makedirs('paper_1_outputs', exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    # Import colors
    from basecode import CARRIER_COLORS
    
    # Panel A: H4c DiD results with significance indicators
    if covid_results and 'h4c_results' in covid_results:
        h4c_data = covid_results['h4c_results']
        carriers = list(h4c_data.keys())
        did_effects = [h4c_data[c]['did_coefficient'] for c in carriers]
        colors = [CARRIER_COLORS.get(c, 'gray') for c in carriers]
        
        bars = axes[0].bar(carriers, did_effects, color=colors, alpha=0.8, 
                            edgecolor='black', linewidth=0.5)
        axes[0].set_title('Panel A: COVID DiD Effects\n(Existing ULCC Routes)', fontweight='bold', pad=15)
        axes[0].set_ylabel('DiD Coefficient (Market Share)')
        axes[0].set_ylim(-0.04, 0.03)
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add significance stars and values
        for i, (carrier, v) in enumerate(zip(carriers, did_effects)):
            p_val = h4c_data[carrier]['did_pvalue']
            sig_stars = ""
            if p_val < 0.001:
                sig_stars = "***"
            elif p_val < 0.01:
                sig_stars = "**"
            elif p_val < 0.05:
                sig_stars = "*"
            
            axes[0].text(i, v + 0.002 if v >= 0 else v - 0.002, 
                          f'{v:+.3f}{sig_stars}', 
                          ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    
    # Panel B: H4d DiD results with significance indicators
    if covid_results and 'h4d_results' in covid_results and covid_results['h4d_results']:
        h4d_data = covid_results['h4d_results']
        carriers = list(h4d_data.keys())
        did_effects = [h4d_data[c]['did_coefficient'] for c in carriers]
        colors = [CARRIER_COLORS.get(c, 'gray') for c in carriers]
        
        bars = axes[1].bar(carriers, did_effects, color=colors, alpha=0.8, 
                            edgecolor='black', linewidth=0.5)
        axes[1].set_title('Panel B: COVID DiD Effects\n(Capacity-Reduced Routes)', fontweight='bold', pad=15)
        axes[1].set_ylabel('DiD Coefficient (Market Share)')
        axes[1].set_ylim(-0.04, 0.03)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add significance stars and values
        for i, (carrier, v) in enumerate(zip(carriers, did_effects)):
            p_val = h4d_data[carrier]['did_pvalue']
            sig_stars = ""
            if p_val < 0.001:
                sig_stars = "***"
            elif p_val < 0.01:
                sig_stars = "**"
            elif p_val < 0.05:
                sig_stars = "*"
            
            axes[1].text(i, v + 0.002 if v >= 0 else v - 0.002, 
                          f'{v:+.3f}{sig_stars}', 
                          ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'No Capacity Data\nAvailable', ha='center', va='center', 
                      transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Panel B: COVID DiD Effects\n(Capacity-Reduced Routes)', fontweight='bold', pad=15)
    
    # Panel C: Treatment Group Analysis
    if panel_data is not None:
        routes_2019 = panel_data[panel_data['Year'] == 2019]
        
        # H4c treatment: Routes with ULCC in 2019
        ulcc_routes_2019 = set(routes_2019[routes_2019['ULCC'] > 0]['Route_ID'])
        h4c_treatment_count = len(ulcc_routes_2019)
        h4c_control_count = routes_2019['Route_ID'].nunique() - h4c_treatment_count
        
        # H4d treatment: Routes with >20% capacity reduction (if available)
        h4d_treatment_count = 0
        h4d_control_count = 0
        if 'ASMs' in panel_data.columns:
            capacity_2019 = panel_data[panel_data['Year'] == 2019].groupby('Route_ID')['ASMs'].sum()
            capacity_2023 = panel_data[panel_data['Year'] == 2023].groupby('Route_ID')['ASMs'].sum()
            capacity_change = pd.merge(capacity_2019, capacity_2023, left_index=True, right_index=True, suffixes=('_2019', '_2023'))
            capacity_change['pct_change'] = (capacity_change['ASMs_2023'] - capacity_change['ASMs_2019']) / capacity_change['ASMs_2019']
            reduced_routes = set(capacity_change[capacity_change['pct_change'] < -0.2].index)
            h4d_treatment_count = len(reduced_routes)
            h4d_control_count = len(capacity_change) - h4d_treatment_count
        
        # Create side-by-side bar chart
        categories = ['H4c: Existing\nULCC Routes', 'H4d: Capacity\nReduced Routes']
        treatment_counts = [h4c_treatment_count, h4d_treatment_count]
        control_counts = [h4c_control_count, h4d_control_count]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = axes[2].bar(x - width/2, treatment_counts, width, label='Treatment Group', 
                             color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = axes[2].bar(x + width/2, control_counts, width, label='Control Group', 
                             color='#E8F4FD', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        axes[2].set_xlabel('Analysis Type')
        axes[2].set_ylabel('Number of Routes')
        axes[2].set_title('Panel C: Treatment Group Analysis', fontweight='bold', pad=15)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(categories)
        axes[2].set_ylim(0, 35000)
        axes[2].legend(frameon=False, loc='upper center', ncol=2)
        
        # Add value labels on bars
        for i, (t, c) in enumerate(zip(treatment_counts, control_counts)):
            axes[2].text(i - width/2, t + max(treatment_counts + control_counts) * 0.01, 
                          f'{t:,}', ha='center', va='bottom', fontsize=9)
            axes[2].text(i + width/2, c + max(treatment_counts + control_counts) * 0.01, 
                          f'{c:,}', ha='center', va='bottom', fontsize=9)
    
    # Panel D: Before/After comparison for H4c
    if covid_results and 'h4c_results' in covid_results:
        h4c_data = covid_results['h4c_results']
        carriers = list(h4c_data.keys())
        
        pre_values = [h4c_data[c]['pre_covid'] for c in carriers]
        post_values = [h4c_data[c]['post_covid'] for c in carriers]
        
        x = np.arange(len(carriers))
        width = 0.35
        
        bars1 = axes[3].bar(x - width/2, pre_values, width, label='2019 (Pre-COVID)', 
                             color='lightgray', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = axes[3].bar(x + width/2, post_values, width, label='2023 (Post-COVID)', 
                             color=[CARRIER_COLORS.get(c, 'gray') for c in carriers], 
                             alpha=0.8, edgecolor='black', linewidth=0.5)
        
        axes[3].set_xlabel('Carrier Type')
        axes[3].set_ylabel('Market Share')
        axes[3].set_title('Panel D: Before/After Comparison\n(Existing ULCC Routes)', fontweight='bold', pad=15)
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(carriers)
        axes[3].legend(frameon=False)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('paper_1_outputs/Figure_4.9_H4cd_COVID_Impact.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

#num5: Main H4c & H4d analysis function
def run_h4cd_analysis(base_data):
    """Run H4c & H4d analysis - COVID Impact Analysis"""
    
    print("RUNNING H4c & H4d: COVID IMPACT ANALYSIS")
    print("=" * 50)
    
    # Step 1: Prepare COVID panel data
    panel_data = prepare_covid_panel_data(base_data)
    
    if panel_data is None:
        print("Failed to prepare COVID panel data")
        return None
    
    # Step 2: H4c & H4d DiD regression analysis
    covid_results = analyze_h4c_h4d_with_did(panel_data)
    
    if covid_results is None:
        print("Failed to complete COVID analysis")
        return None
    
    # Step 3: Create visualization
    fig = create_h4cd_figure(covid_results, panel_data)
    
    # Step 4: Save results with formatted tables
    os.makedirs('paper_1_outputs', exist_ok=True)
    
    # Save COVID DiD results with regression details and formatted tables
    if covid_results:
        if 'h4c_results' in covid_results:
            h4c_df = pd.DataFrame(covid_results['h4c_results']).T
            
            # Display Table 4.8 results
            print("\n=== TABLE 4.8: H4c DiD Regression Results (COVID-Period DiD Results) ===")
            print(h4c_df.round(3).to_string(index=True))

            h4c_df.to_csv('paper_1_outputs/Table_4.8_H4c_DiD_Regression_Results.csv')
            print(f"\nTable 4.8 saved: paper_1_outputs/Table_4.8_H4c_DiD_Regression_Results.csv")

            # Create formatted H4c table for display
            print("\n" + "=" * 70)
            print("TABLE 4.8: H4c DiD Regression Results (COVID-Period DiD Results)")
            print("=" * 70)
            print(f"{'Carrier':<8} {'DiD Coeff':<12} {'95% CI':<20} {'P-value':<10} {'Sig':<5}")
            print("-" * 70)
            
            for carrier in ['Legacy', 'ULCC', 'LCC', 'Hybrid']:
                if carrier in covid_results['h4c_results']:
                    data = covid_results['h4c_results'][carrier]
                    coeff = data['did_coefficient']
                    ci_lower = data['did_ci_lower']
                    ci_upper = data['did_ci_upper']
                    p_val = data['did_pvalue']
                    
                    # Significance stars
                    sig_stars = ""
                    if p_val < 0.001:
                        sig_stars = "***"
                    elif p_val < 0.01:
                        sig_stars = "**"
                    elif p_val < 0.05:
                        sig_stars = "*"
                    
                    p_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
                    ci_str = f"[{ci_lower:+.3f}, {ci_upper:+.3f}]"
                    
                    print(f"{carrier:<8} {coeff:+.3f}{'':>5} {ci_str:<20} {p_str:<10} {sig_stars:<5}")
            
            print("-" * 70)
            print("Note: *p<0.05, **p<0.01, ***p<0.001")
            print("DiD = Difference-in-Differences coefficient")
            print("Treatment: Routes with ULCC presence in 2019")
            print("=" * 70)
        
        if 'h4d_results' in covid_results and covid_results['h4d_results']:
            h4d_df = pd.DataFrame(covid_results['h4d_results']).T
            
            # Display Table 4.9 results
            print("\n=== TABLE 4.9: H4d DiD Regression Results (ULCC Strategic Portfolio Rebalancing) ===")
            print(h4d_df.round(3).to_string(index=True))

            h4d_df.to_csv('paper_1_outputs/Table_4.9_H4d_DiD_Regression_Results.csv')
            print(f"\nTable 4.9 saved: paper_1_outputs/Table_4.9_H4d_DiD_Regression_Results.csv")

            # Create formatted H4d table for display
            print("\n" + "=" * 70)
            print("TABLE 4.9: H4d DiD Regression Results (ULCC Strategic Portfolio Rebalancing)")
            print("=" * 70)
            print(f"{'Carrier':<8} {'DiD Coeff':<12} {'95% CI':<20} {'P-value':<10} {'Sig':<5}")
            print("-" * 70)
            
            for carrier in ['Legacy', 'ULCC', 'LCC', 'Hybrid']:
                if carrier in covid_results['h4d_results']:
                    data = covid_results['h4d_results'][carrier]
                    coeff = data['did_coefficient']
                    ci_lower = data['did_ci_lower']
                    ci_upper = data['did_ci_upper']
                    p_val = data['did_pvalue']
                    
                    # Significance stars
                    sig_stars = ""
                    if p_val < 0.001:
                        sig_stars = "***"
                    elif p_val < 0.01:
                        sig_stars = "**"
                    elif p_val < 0.05:
                        sig_stars = "*"
                    
                    p_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
                    ci_str = f"[{ci_lower:+.3f}, {ci_upper:+.3f}]"
                    
                    print(f"{carrier:<8} {coeff:+.3f}{'':>5} {ci_str:<20} {p_str:<10} {sig_stars:<5}")
            
            print("-" * 70)
            print("Note: *p<0.05, **p<0.01, ***p<0.001")
            print("DiD = Difference-in-Differences coefficient")
            print("Treatment: Routes with >20% capacity reduction 2019-2023")
            print("=" * 70)
    
    # Save panel data
    panel_data.to_csv('paper_1_outputs/H4cd_Panel_Data.csv', index=False)
    
    print("\n" + "=" * 50)
    print("H4c & H4d ANALYSIS COMPLETE!")
    print("Results saved in 'paper_1_outputs/' directory")
    print("- DiD regression analysis with significance tests")
    print("- COVID impact analysis with p-values")
    print("- Figure 4.7 created with significance indicators")
    print("=" * 50)
    
    return {
        'covid_results': covid_results,
        'panel_data': panel_data,
        'figure': fig
    }

if __name__ == "__main__":
    from basecode import prepare_base_data
    base_data = prepare_base_data()
    if base_data:
        h4cd_results = run_h4cd_analysis(base_data)