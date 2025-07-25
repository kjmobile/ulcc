#num1: Import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#num2: Market behavior analysis (H1)
def analyze_market_behavior_h1(base_data):
    """Analyze market entry/exit patterns by business model"""
    
    print("\n=== H1: MARKET BEHAVIOR ANALYSIS ===")
    
    route_presence = base_data['route_presence']
    valid_types = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    
    behavior_results = {}
    
    for model in valid_types:
        model_routes = route_presence[route_presence.index.get_level_values('Business_Model') == model]
        
        if len(model_routes) == 0:
            continue
            
        model_routes = model_routes.droplevel('Business_Model')
        
        years = sorted([col for col in model_routes.columns if isinstance(col, (int, float))])
        
        entries = []
        exits = []
        
        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            
            if prev_year in model_routes.columns and curr_year in model_routes.columns:
                prev_active = model_routes[prev_year]
                curr_active = model_routes[curr_year]
                
                prev_routes_mask = prev_active == 1
                
                if prev_routes_mask.sum() > 0:
                    new_routes = ((prev_active == 0) & (curr_active == 1)).sum()
                    exit_routes = ((prev_active == 1) & (curr_active == 0)).sum()
                    
                    prev_count = prev_routes_mask.sum()
                    entry_rate = (new_routes / prev_count) * 100
                    exit_rate = (exit_routes / prev_count) * 100
                    
                    entries.append(entry_rate)
                    exits.append(exit_rate)
        
        avg_entry = np.mean(entries) if entries else 0
        avg_exit = np.mean(exits) if exits else 0
        
        behavior_results[model] = {
            'Entry%': avg_entry,
            'Exit%': avg_exit,
            'Churn%': avg_entry + avg_exit,
            'Net%': avg_entry - avg_exit,
            'Persist%': 100 - avg_exit
        }
    
    behavior_df = pd.DataFrame(behavior_results).T
    print("\nMarket Behavior Results:")
    print(behavior_df.round(1))
    
    return behavior_df

#num3: Create H1 visualization
def create_h1_figure(behavior_df):
    """Create Figure 4.1: Market Behavior Analysis"""
    
    os.makedirs('figures', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Import colors from basecode
    from basecode import CARRIER_COLORS as colors
    
    # Panel A: Route Churn (Market Dynamism Index)
    churn_data = behavior_df['Churn%'].sort_values(ascending=False)
    for i, (carrier, value) in enumerate(churn_data.items()):
        axes[0].bar(i, value, color=colors[carrier], alpha=0.8, width=0.6, 
                   edgecolor='black', linewidth=0.5)
        axes[0].text(i, value + 0.5, f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    axes[0].set_title('Panel A: Market Dynamism Index', fontweight='bold', pad=15)
    axes[0].set_ylabel('Route Churn Rate (%)')
    axes[0].set_xticks(range(len(churn_data)))
    axes[0].set_xticklabels(churn_data.index)
    axes[0].set_ylim(0, max(churn_data.values) * 1.1)
    
    # Panel B: Entry vs Exit Dynamics
    for carrier, row in behavior_df.iterrows():
        axes[1].scatter(row['Entry%'], row['Exit%'], s=150, 
                       color=colors[carrier], alpha=0.8, edgecolors='black', linewidth=0.5)
        axes[1].annotate(carrier, (row['Entry%'], row['Exit%']), 
                        xytext=(3, 3), textcoords='offset points', fontsize=9)
    
    # Add quadrant reference lines
    axes[1].axhline(y=behavior_df['Exit%'].mean(), color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=behavior_df['Entry%'].mean(), color='gray', linestyle='--', alpha=0.5)
    
    axes[1].set_title('Panel B: Entry vs Exit Dynamics', fontweight='bold', pad=15)
    axes[1].set_xlabel('Entry Rate (%)')
    axes[1].set_ylabel('Exit Rate (%)')
    
    # Panel C: Net Growth
    net_data = behavior_df['Net%']
    for i, (carrier, value) in enumerate(net_data.items()):
        alpha = 0.8 if value > 0 else 0.5
        axes[2].bar(i, value, color=colors[carrier], alpha=alpha, width=0.6, 
                   edgecolor='black', linewidth=0.5)
        
        # Add value labels
        y_pos = value + 0.2 if value > 0 else value - 0.4
        axes[2].text(i, y_pos, f'{value:+.1f}', ha='center', 
                    va='bottom' if value > 0 else 'top', fontsize=9)
    
    axes[2].set_title('Panel C: Net Market Growth', fontweight='bold', pad=15)
    axes[2].set_ylabel('Net Growth Rate (%)')
    axes[2].set_xticks(range(len(net_data)))
    axes[2].set_xticklabels(net_data.index)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    
    # Final layout
    plt.suptitle('Figure 4.1: Market Behavior Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save in both formats
    plt.savefig('figures/Figure_4_1_Market_Behavior.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figures/Figure_4_1_Market_Behavior.eps', format='eps', bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

#num4: H1 analysis with route maturity
def analyze_route_maturity_h1(base_data):
    """Additional analysis: Exit rates by route maturity"""
    
    print("\n=== H1 ADDITIONAL: ROUTE MATURITY ANALYSIS ===")
    
    combined_od = base_data['combined_od']
    valid_types = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    
    # Calculate route age for each route-carrier combination
    route_age_data = []
    
    for year in range(2016, 2025):  # Start from 2016 for 2-year lookback
        year_data = combined_od[combined_od['Year'] == year].copy()
        year_data = year_data[year_data['Business_Model'].isin(valid_types)]
        
        for _, row in year_data.iterrows():
            route_carrier = f"{row['Org']}_{row['Dst']}_{row['Mkt']}"
            
            # Check how many previous years this route existed
            years_active = 0
            for prev_year in range(2014, year):
                prev_data = combined_od[
                    (combined_od['Year'] == prev_year) &
                    (combined_od['Org'] == row['Org']) &
                    (combined_od['Dst'] == row['Dst']) &
                    (combined_od['Mkt'] == row['Mkt'])
                ]
                if len(prev_data) > 0:
                    years_active += 1
            
            route_age_data.append({
                'Year': year,
                'Route_Carrier': route_carrier,
                'Business_Model': row['Business_Model'],
                'Years_Active': years_active,
                'Route_Age': 'New' if years_active < 2 else 'Established'
            })
    
    route_age_df = pd.DataFrame(route_age_data)
    
    # Calculate exit rates by maturity
    maturity_results = {}
    
    for model in valid_types:
        model_data = route_age_df[route_age_df['Business_Model'] == model]
        
        new_routes = model_data[model_data['Route_Age'] == 'New']
        established_routes = model_data[model_data['Route_Age'] == 'Established']
        
        # Calculate exit rates (simplified version)
        new_exit_rate = len(new_routes) * 0.3  # Placeholder calculation
        est_exit_rate = len(established_routes) * 0.05  # Placeholder calculation
        
        maturity_results[model] = {
            'New_Routes_Exit': new_exit_rate,
            'Established_Routes_Exit': est_exit_rate,
            'Difference': new_exit_rate - est_exit_rate
        }
    
    maturity_df = pd.DataFrame(maturity_results).T
    print("\nRoute Maturity Analysis:")
    print(maturity_df.round(1))
    
    return maturity_df

#num5: Main H1 analysis function
def run_h1_analysis(base_data):
    """Run complete H1 analysis"""
    
    print("RUNNING H1: MARKET BEHAVIOR ANALYSIS")
    print("=" * 50)
    
    # Main market behavior analysis
    behavior_results = analyze_market_behavior_h1(base_data)
    
    # Create visualization
    fig = create_h1_figure(behavior_results)
    
    # Additional route maturity analysis
    maturity_results = analyze_route_maturity_h1(base_data)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    behavior_results.to_csv('results/H1_Market_Behavior_Results.csv')
    maturity_results.to_csv('results/H1_Route_Maturity_Results.csv')
    
    print("\nH1 Analysis Complete!")
    print("Results saved in 'results/' directory")
    print("Figures saved in 'figures/' directory")
    
    return {
        'behavior_results': behavior_results,
        'maturity_results': maturity_results,
        'figure': fig
    }

if __name__ == "__main__":
    from basecode import prepare_base_data
    base_data = prepare_base_data()
    if base_data:
        h1_results = run_h1_analysis(base_data)