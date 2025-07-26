# #002 - H1 Analysis Fixed Version
# Memory-efficient market behavior analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_market_behavior_h1(base_data):
    """Memory-efficient market entry/exit rate calculation"""
    
    combined_od = base_data['combined_od']
    valid_types = ['ULCC', 'LCC', 'Hybrid', 'Legacy']
    
    # Filter and prepare data efficiently
    analysis_data = combined_od[combined_od['Business_Model'].isin(valid_types)].copy()
    analysis_data['Route'] = analysis_data['Org'] + '-' + analysis_data['Dst']
    analysis_data['Quarter'] = (analysis_data['Year'].astype(str) + 'Q' + 
                               ((analysis_data['Month'] - 1) // 3 + 1).astype(str))
    
    # Aggregate to route-carrier-quarter level with traffic threshold
    route_activity = (analysis_data.groupby(['Business_Model', 'Route', 'Opr', 'Quarter'])
                     ['Passengers'].sum().reset_index())
    route_activity = route_activity[route_activity['Passengers'] >= 100]
    
    # Calculate entry/exit rates by business model
    behavior_results = {}
    
    for business_model in valid_types:
        bm_data = route_activity[route_activity['Business_Model'] == business_model]
        
        if len(bm_data) == 0:
            behavior_results[business_model] = {
                'Entry%': 0.0, 'Exit%': 0.0, 'Churn%': 0.0, 'Net%': 0.0, 'Persist%': 100.0
            }
            continue
        
        # Create quarterly route sets (route-carrier combinations)
        quarterly_routes = bm_data.groupby('Quarter').apply(
            lambda x: set(zip(x['Route'], x['Opr']))
        ).to_dict()
        
        quarters = sorted(quarterly_routes.keys())
        entry_rates = []
        exit_rates = []
        
        # Calculate quarter-to-quarter transitions
        for i in range(1, len(quarters)):
            prev_quarter = quarters[i-1]
            curr_quarter = quarters[i]
            
            prev_routes = quarterly_routes[prev_quarter]
            curr_routes = quarterly_routes[curr_quarter]
            
            if len(prev_routes) == 0:
                continue
            
            # Entry: routes that exist in current but not previous quarter
            entries = len(curr_routes - prev_routes)
            # Exit: routes that existed in previous but not current quarter
            exits = len(prev_routes - curr_routes)
            
            entry_rate = (entries / len(prev_routes)) * 100
            exit_rate = (exits / len(prev_routes)) * 100
            
            entry_rates.append(entry_rate)
            exit_rates.append(exit_rate)
        
        # Calculate average rates
        avg_entry = np.mean(entry_rates) if entry_rates else 0
        avg_exit = np.mean(exit_rates) if exit_rates else 0
        
        behavior_results[business_model] = {
            'Entry%': round(avg_entry, 1),
            'Exit%': round(avg_exit, 1),
            'Churn%': round(avg_entry + avg_exit, 1),
            'Net%': round(avg_entry - avg_exit, 1),
            'Persist%': round(100 - avg_exit, 1)
        }
    
    # Create results DataFrame
    behavior_df = pd.DataFrame.from_dict(behavior_results, orient='index')
    behavior_df = behavior_df.reindex(['ULCC', 'LCC', 'Hybrid', 'Legacy'])
    
    print("TABLE 4.1: MARKET BEHAVIOR PATTERNS BY BUSINESS MODEL")
    print(behavior_df.to_string())
    
    return behavior_df

def analyze_route_maturity_h1(base_data):
    """Route maturity analysis - new vs established routes"""
    
    combined_od = base_data['combined_od']
    valid_types = ['ULCC', 'LCC', 'Hybrid', 'Legacy']
    
    # Filter and prepare data
    analysis_data = combined_od[combined_od['Business_Model'].isin(valid_types)].copy()
    analysis_data['Route'] = analysis_data['Org'] + '-' + analysis_data['Dst']
    analysis_data['Quarter'] = (analysis_data['Year'].astype(str) + 'Q' + 
                               ((analysis_data['Month'] - 1) // 3 + 1).astype(str))
    
    # Find first operation year for each route-carrier combination
    route_first_year = (analysis_data.groupby(['Route', 'Opr'])['Year'].min()
                       .reset_index().rename(columns={'Year': 'First_Year'}))
    
    # Merge and calculate route age
    analysis_data = analysis_data.merge(route_first_year, on=['Route', 'Opr'])
    analysis_data['Route_Age'] = analysis_data['Year'] - analysis_data['First_Year']
    analysis_data['Route_Maturity'] = np.where(analysis_data['Route_Age'] < 2, 'New', 'Established')
    
    # Aggregate by maturity
    route_activity = (analysis_data.groupby(['Business_Model', 'Route_Maturity', 'Route', 'Opr', 'Quarter'])
                     ['Passengers'].sum().reset_index())
    route_activity = route_activity[route_activity['Passengers'] >= 100]
    
    # Calculate exit rates by maturity
    maturity_results = {}
    
    for business_model in valid_types:
        bm_data = route_activity[route_activity['Business_Model'] == business_model]
        
        new_exit_rates = []
        established_exit_rates = []
        
        for maturity in ['New', 'Established']:
            maturity_data = bm_data[bm_data['Route_Maturity'] == maturity]
            
            if len(maturity_data) == 0:
                continue
            
            # Create quarterly route sets
            quarterly_routes = maturity_data.groupby('Quarter').apply(
                lambda x: set(zip(x['Route'], x['Opr']))
            ).to_dict()
            
            quarters = sorted(quarterly_routes.keys())
            exit_rates = []
            
            for i in range(1, len(quarters)):
                prev_routes = quarterly_routes[quarters[i-1]]
                curr_routes = quarterly_routes[quarters[i]]
                
                if len(prev_routes) == 0:
                    continue
                
                exits = len(prev_routes - curr_routes)
                exit_rate = (exits / len(prev_routes)) * 100
                exit_rates.append(exit_rate)
            
            avg_exit = np.mean(exit_rates) if exit_rates else 0
            
            if maturity == 'New':
                new_exit_rates.append(avg_exit)
            else:
                established_exit_rates.append(avg_exit)
        
        avg_new_exit = np.mean(new_exit_rates) if new_exit_rates else 0
        avg_established_exit = np.mean(established_exit_rates) if established_exit_rates else 0
        
        maturity_results[business_model] = {
            'New_Routes_Exit': round(avg_new_exit, 1),
            'Established_Routes_Exit': round(avg_established_exit, 1),
            'Difference': round(avg_new_exit - avg_established_exit, 1)
        }
    
    maturity_df = pd.DataFrame.from_dict(maturity_results, orient='index')
    maturity_df = maturity_df.reindex(['ULCC', 'LCC', 'Hybrid', 'Legacy'])
    
    print("\nTable 4.2: Exit Rates by Route Maturity")
    print(maturity_df.to_string())
    
    return maturity_df

def create_h1_figure(behavior_df):
    """Create Figure 4.1: Market Behavior Analysis"""
    
    colors = {
        'ULCC': '#d62728',
        'LCC': '#ff7f0e', 
        'Hybrid': '#1f77b4',
        'Legacy': '#2ca02c'
    }
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Market Dynamism Index
    bar_colors = [colors[bm] for bm in behavior_df.index]
    x_pos = np.arange(len(behavior_df))
    
    bars1 = ax1.bar(x_pos, behavior_df['Churn%'], color=bar_colors, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    ax1.set_title('Panel A: Market Dynamism Index', fontweight='bold')
    ax1.set_ylabel('Route Churn Rate (%)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(behavior_df.index)
    
    for i, v in enumerate(behavior_df['Churn%']):
        ax1.text(i, v + 0.5, f'{v}%', ha='center', va='bottom', fontweight='bold')
    
    # Panel B: Entry vs Exit Dynamics
    ax2.scatter(behavior_df['Entry%'], behavior_df['Exit%'], 
               s=150, c=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for i, (bm, row) in enumerate(behavior_df.iterrows()):
        ax2.annotate(bm, (row['Entry%'], row['Exit%']), 
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    ax2.set_title('Panel B: Entry vs Exit Dynamics', fontweight='bold')
    ax2.set_xlabel('Entry Rate (%)')
    ax2.set_ylabel('Exit Rate (%)')
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Net Market Growth
    net_values = behavior_df['Net%'].values
    bars3 = ax3.bar(x_pos, net_values, color=bar_colors, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    
    ax3.set_title('Panel C: Net Market Growth', fontweight='bold')
    ax3.set_ylabel('Net Growth Rate (%)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(behavior_df.index)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    for i, v in enumerate(net_values):
        y_pos = v + 0.2 if v >= 0 else v - 0.3
        va = 'bottom' if v >= 0 else 'top'
        ax3.text(i, y_pos, f'{v:+.1f}%', ha='center', va=va, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    Path('figures').mkdir(exist_ok=True)
    plt.savefig('figures/Figure_4_1_Market_Behavior.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def run_h1_analysis(base_data):
    """Main H1 analysis function"""
    
    print("H1: MARKET ENTRY AND EXIT HYPOTHESIS")
    print("="*50)
    
    # Market behavior analysis
    behavior_results = analyze_market_behavior_h1(base_data)
    
    # Route maturity analysis
    maturity_results = analyze_route_maturity_h1(base_data)
    
    # Create visualization
    figure = create_h1_figure(behavior_results)
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    behavior_results.to_csv('results/H1_Market_Behavior_Results.csv')
    maturity_results.to_csv('results/H1_Route_Maturity_Results.csv')
    
    print(f"\nH1 Analysis Complete!")
    print("Results saved to results/ and figures/")
    
    return {
        'behavior_results': behavior_results,
        'maturity_results': maturity_results,
        'figure': figure
    }

# Usage example:
# from basecode import prepare_base_data
# base_data = prepare_base_data()
# h1_results = run_h1_analysis(base_data)