#num1: Import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

#num2: Additional analysis - Shock sensitivity
def analyze_shock_sensitivity(base_data):
    """Analyze carrier sensitivity to external shocks"""
    
    print("\n=== ADDITIONAL: SHOCK SENSITIVITY ANALYSIS ===")
    
    od_years = base_data['od_years']
    shock_data = base_data['shock_data']
    classification_map = base_data['classification_map']
    
    # Prepare monthly traffic data by carrier type
    monthly_traffic = []
    
    for year in range(2014, 2025):
        if year not in od_years:
            continue
            
        year_data = od_years[year].copy()
        year_data['Business_Model'] = year_data['Mkt'].map(classification_map)
        year_data = year_data.dropna(subset=['Business_Model'])
        
        valid_types = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
        year_data = year_data[year_data['Business_Model'].isin(valid_types)]
        
        for month in range(1, 13):
            month_data = year_data[year_data['Month'] == month]
            
            if len(month_data) == 0:
                continue
                
            model_traffic = month_data.groupby('Business_Model')['Passengers'].sum()
            
            for model in valid_types:
                traffic = model_traffic.get(model, 0)
                monthly_traffic.append({
                    'Year': year,
                    'Month': month,
                    'Business_Model': model,
                    'Passengers': traffic
                })
    
    traffic_df = pd.DataFrame(monthly_traffic)
    
    # Merge with shock data
    shock_data['Date'] = pd.to_datetime(shock_data.index)
    shock_data['Year'] = shock_data['Date'].dt.year
    shock_data['Month'] = shock_data['Date'].dt.month
    
    combined_shock = traffic_df.merge(
        shock_data[['Year', 'Month', 'WTI_Real', 'JetFuel_Real', 'COVID_Dummy', 'Workplace_Mobility']], 
        on=['Year', 'Month'], 
        how='left'
    )
    
    # Calculate correlations with shock variables
    print("\nCorrelations with External Shocks:")
    
    shock_correlations = {}
    for model in ['Legacy', 'ULCC', 'LCC', 'Hybrid']:
        model_data = combined_shock[combined_shock['Business_Model'] == model]
        
        if len(model_data) > 10:
            # Oil price correlation
            oil_corr, oil_p = stats.pearsonr(model_data['WTI_Real'].fillna(0), model_data['Passengers'])
            
            # Jet fuel correlation
            fuel_corr, fuel_p = stats.pearsonr(model_data['JetFuel_Real'].fillna(0), model_data['Passengers'])
            
            # COVID period impact
            covid_data = model_data[model_data['COVID_Dummy'] == 1]
            normal_data = model_data[model_data['COVID_Dummy'] == 0]
            
            if len(covid_data) > 0 and len(normal_data) > 0:
                covid_impact = (covid_data['Passengers'].mean() / normal_data['Passengers'].mean() - 1) * 100
            else:
                covid_impact = 0
            
            shock_correlations[model] = {
                'Oil_Correlation': oil_corr,
                'Oil_P_Value': oil_p,
                'Fuel_Correlation': fuel_corr,
                'Fuel_P_Value': fuel_p,
                'COVID_Impact': covid_impact
            }
            
            print(f"\n{model}:")
            print(f"  Oil Price Correlation: {oil_corr:.3f} (p={oil_p:.3f})")
            print(f"  Jet Fuel Correlation: {fuel_corr:.3f} (p={fuel_p:.3f})")
            print(f"  COVID Impact: {covid_impact:.1f}%")
    
    return shock_correlations, combined_shock

#num3: Additional analysis - Route concentration
def analyze_route_concentration(base_data):
    """Analyze route concentration patterns by carrier type"""
    
    print("\n=== ADDITIONAL: ROUTE CONCENTRATION ANALYSIS ===")
    
    combined_od = base_data['combined_od']
    valid_types = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    
    # Calculate route-level traffic by carrier type
    route_analysis = combined_od.groupby(['Org', 'Dst', 'Business_Model']).agg({
        'Passengers': 'sum'
    }).reset_index()
    
    concentration_results = {}
    
    for model in valid_types:
        model_routes = route_analysis[route_analysis['Business_Model'] == model]
        
        if len(model_routes) == 0:
            continue
        
        # Calculate concentration metrics
        total_passengers = model_routes['Passengers'].sum()
        model_routes_sorted = model_routes.sort_values('Passengers', ascending=False)
        
        # Top routes concentration
        top_10_share = model_routes_sorted.head(10)['Passengers'].sum() / total_passengers * 100
        top_25_share = model_routes_sorted.head(25)['Passengers'].sum() / total_passengers * 100
        top_50_share = model_routes_sorted.head(50)['Passengers'].sum() / total_passengers * 100
        
        # Route size distribution
        route_passengers = model_routes['Passengers'].values
        mean_route_size = np.mean(route_passengers)
        median_route_size = np.median(route_passengers)
        std_route_size = np.std(route_passengers)
        
        # Gini coefficient for route concentration
        sorted_passengers = np.sort(route_passengers)
        n = len(sorted_passengers)
        cumsum = np.cumsum(sorted_passengers)
        gini = (n + 1 - 2 * np.sum((n + 1 - np.arange(1, n + 1)) * sorted_passengers) / np.sum(sorted_passengers)) / n
        
        concentration_results[model] = {
            'Total_Routes': len(model_routes),
            'Top_10_Share': top_10_share,
            'Top_25_Share': top_25_share,
            'Top_50_Share': top_50_share,
            'Mean_Route_Size': mean_route_size,
            'Median_Route_Size': median_route_size,
            'Route_Gini': gini
        }
        
        print(f"\n{model} Route Concentration:")
        print(f"  Total Routes: {len(model_routes):,}")
        print(f"  Top 10 Routes Share: {top_10_share:.1f}%")
        print(f"  Top 25 Routes Share: {top_25_share:.1f}%")
        print(f"  Route Gini Coefficient: {gini:.3f}")
    
    return concentration_results

#num4: Additional analysis - Market penetration
def analyze_market_penetration(base_data):
    """Analyze market penetration patterns"""
    
    print("\n=== ADDITIONAL: MARKET PENETRATION ANALYSIS ===")
    
    combined_od = base_data['combined_od']
    valid_types = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    
    # Airport presence analysis
    airport_presence = {}
    
    for model in valid_types:
        model_data = combined_od[combined_od['Business_Model'] == model]
        
        if len(model_data) == 0:
            continue
        
        # Origin airports
        origin_airports = set(model_data['Org'].unique())
        destination_airports = set(model_data['Dst'].unique())
        all_airports = origin_airports.union(destination_airports)
        
        # Market presence metrics
        total_routes = len(model_data.groupby(['Org', 'Dst']).size())
        airports_served = len(all_airports)
        
        # Average routes per airport
        avg_routes_per_airport = total_routes / airports_served if airports_served > 0 else 0
        
        # Hub analysis - airports with most routes
        origin_counts = model_data['Org'].value_counts()
        dest_counts = model_data['Dst'].value_counts()
        
        # Combine origin and destination counts
        all_airport_counts = pd.concat([origin_counts, dest_counts]).groupby(level=0).sum()
        top_hubs = all_airport_counts.head(5)
        
        airport_presence[model] = {
            'Total_Routes': total_routes,
            'Airports_Served': airports_served,
            'Avg_Routes_Per_Airport': avg_routes_per_airport,
            'Top_Hub': top_hubs.index[0] if len(top_hubs) > 0 else 'None',
            'Top_Hub_Routes': top_hubs.iloc[0] if len(top_hubs) > 0 else 0
        }
        
        print(f"\n{model} Market Penetration:")
        print(f"  Total Routes: {total_routes:,}")
        print(f"  Airports Served: {airports_served}")
        print(f"  Avg Routes per Airport: {avg_routes_per_airport:.1f}")
        if len(top_hubs) > 0:
            print(f"  Top Hub: {top_hubs.index[0]} ({top_hubs.iloc[0]} routes)")
    
    return airport_presence

#num5: Create additional analysis visualization
def create_additional_figures(shock_correlations, concentration_results, airport_presence, combined_shock):
    """Create additional analysis visualizations"""
    
    os.makedirs('figures', exist_ok=True)
    
    # Import colors from basecode
    from basecode import CARRIER_COLORS as colors
    
    # Figure 1: Shock Sensitivity Analysis
    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle('Additional Analysis: Shock Sensitivity', fontsize=16, fontweight='bold')
    
    models = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    
    # Oil price correlations
    if shock_correlations:
        oil_corrs = [shock_correlations.get(m, {}).get('Oil_Correlation', 0) for m in models]
        bars1 = axes[0,0].bar(models, oil_corrs, color=[colors[m] for m in models], 
                             alpha=0.8, width=0.6, edgecolor='black', linewidth=0.5)
        axes[0,0].set_title('Oil Price Correlations', fontweight='bold', pad=15)
        axes[0,0].set_ylabel('Correlation Coefficient')
        axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        
        # Add value labels
        for bar, corr in zip(bars1, oil_corrs):
            height = bar.get_height()
            y_pos = height + 0.01 if height > 0 else height - 0.02
            axes[0,0].text(bar.get_x() + bar.get_width()/2., y_pos,
                          f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # Fuel price correlations
        fuel_corrs = [shock_correlations.get(m, {}).get('Fuel_Correlation', 0) for m in models]
        bars2 = axes[0,1].bar(models, fuel_corrs, color=[colors[m] for m in models], 
                             alpha=0.8, width=0.6, edgecolor='black', linewidth=0.5)
        axes[0,1].set_title('Fuel Price Correlations', fontweight='bold', pad=15)
        axes[0,1].set_ylabel('Correlation Coefficient')
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        
        # Add value labels
        for bar, corr in zip(bars2, fuel_corrs):
            height = bar.get_height()
            y_pos = height + 0.01 if height > 0 else height - 0.02
            axes[0,1].text(bar.get_x() + bar.get_width()/2., y_pos,
                          f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # COVID impact
        covid_impacts = [shock_correlations.get(m, {}).get('COVID_Impact', 0) for m in models]
        bars3 = axes[1,0].bar(models, covid_impacts, color=[colors[m] for m in models], 
                             alpha=0.8, width=0.6, edgecolor='black', linewidth=0.5)
        axes[1,0].set_title('COVID-19 Traffic Impact', fontweight='bold', pad=15)
        axes[1,0].set_ylabel('Impact (%)')
        axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        
        # Add value labels
        for bar, impact in zip(bars3, covid_impacts):
            height = bar.get_height()
            y_pos = height + 1 if height > 0 else height - 2
            axes[1,0].text(bar.get_x() + bar.get_width()/2., y_pos,
                          f'{impact:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Time series plot
    if combined_shock is not None and len(combined_shock) > 0:
        for model in models:
            model_data = combined_shock[combined_shock['Business_Model'] == model]
            if len(model_data) > 0:
                model_data = model_data.sort_values(['Year', 'Month'])
                axes[1,1].plot(range(len(model_data)), model_data['Passengers']/1000000, 
                              label=model, color=colors[model], linewidth=2)
        
        axes[1,1].set_title('Traffic Time Series', fontweight='bold', pad=15)
        axes[1,1].set_xlabel('Time Period')
        axes[1,1].set_ylabel('Passengers (Millions)')
        axes[1,1].legend(frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig('figures/Additional_Shock_Sensitivity.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figures/Additional_Shock_Sensitivity.eps', format='eps', bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Figure 2: Route Concentration Analysis
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle('Additional Analysis: Route Concentration', fontsize=16, fontweight='bold')
    
    if concentration_results:
        # Top 10 route concentration
        top10_shares = [concentration_results.get(m, {}).get('Top_10_Share', 0) for m in models]
        bars1 = axes[0,0].bar(models, top10_shares, color=[colors[m] for m in models], 
                             alpha=0.8, width=0.6, edgecolor='black', linewidth=0.5)
        axes[0,0].set_title('Top 10 Routes Concentration', fontweight='bold', pad=15)
        axes[0,0].set_ylabel('Share of Total Traffic (%)')
        
        # Add value labels
        for bar, share in zip(bars1, top10_shares):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                          f'{share:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Route Gini coefficients
        route_ginis = [concentration_results.get(m, {}).get('Route_Gini', 0) for m in models]
        bars2 = axes[0,1].bar(models, route_ginis, color=[colors[m] for m in models], 
                             alpha=0.8, width=0.6, edgecolor='black', linewidth=0.5)
        axes[0,1].set_title('Route Traffic Inequality (Gini)', fontweight='bold', pad=15)
        axes[0,1].set_ylabel('Gini Coefficient')
        
        # Add value labels
        for bar, gini in zip(bars2, route_ginis):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{gini:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Total routes
        total_routes = [concentration_results.get(m, {}).get('Total_Routes', 0) for m in models]
        bars3 = axes[1,0].bar(models, total_routes, color=[colors[m] for m in models], 
                             alpha=0.8, width=0.6, edgecolor='black', linewidth=0.5)
        axes[1,0].set_title('Total Routes Operated', fontweight='bold', pad=15)
        axes[1,0].set_ylabel('Number of Routes')
        
        # Add value labels
        for bar, routes in zip(bars3, total_routes):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 50,
                          f'{routes:,}', ha='center', va='bottom', fontsize=9)
        
        # Mean route size
        mean_sizes = [concentration_results.get(m, {}).get('Mean_Route_Size', 0)/1000 for m in models]
        bars4 = axes[1,1].bar(models, mean_sizes, color=[colors[m] for m in models], 
                             alpha=0.8, width=0.6, edgecolor='black', linewidth=0.5)
        axes[1,1].set_title('Average Route Size', fontweight='bold', pad=15)
        axes[1,1].set_ylabel('Passengers (Thousands)')
        
        # Add value labels
        for bar, size in zip(bars4, mean_sizes):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 1,
                          f'{size:.1f}K', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig('figures/Additional_Route_Concentration.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figures/Additional_Route_Concentration.eps', format='eps', bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig1, fig2

#num6: Main additional analysis function
def run_additional_analysis(base_data):
    """Run complete additional analysis"""
    
    print("RUNNING ADDITIONAL ANALYSIS")
    print("=" * 50)
    
    # Shock sensitivity analysis
    shock_correlations, combined_shock = analyze_shock_sensitivity(base_data)
    
    # Route concentration analysis
    concentration_results = analyze_route_concentration(base_data)
    
    # Market penetration analysis
    airport_presence = analyze_market_penetration(base_data)
    
    # Create visualizations
    fig1, fig2 = create_additional_figures(shock_correlations, concentration_results, 
                                         airport_presence, combined_shock)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Save shock correlations
    if shock_correlations:
        shock_df = pd.DataFrame(shock_correlations).T
        shock_df.to_csv('results/Additional_Shock_Sensitivity.csv')
    
    # Save concentration results
    if concentration_results:
        concentration_df = pd.DataFrame(concentration_results).T
        concentration_df.to_csv('results/Additional_Route_Concentration.csv')
    
    # Save airport presence
    if airport_presence:
        airport_df = pd.DataFrame(airport_presence).T
        airport_df.to_csv('results/Additional_Market_Penetration.csv')
    
    print("\nAdditional Analysis Complete!")
    print("Results saved in 'results/' directory")
    print("Figures saved in 'figures/' directory")
    
    return {
        'shock_correlations': shock_correlations,
        'concentration_results': concentration_results,
        'airport_presence': airport_presence,
        'figures': [fig1, fig2]
    }

if __name__ == "__main__":
    from basecode import prepare_base_data
    base_data = prepare_base_data()
    if base_data:
        additional_results = run_additional_analysis(base_data)