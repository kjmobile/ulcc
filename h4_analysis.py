#num1: Import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#num2: COVID recovery analysis (H4)
def analyze_covid_recovery_h4(base_data):
    """Analyze COVID-19 market share changes and recovery patterns"""
    
    print("\n=== H4: COVID RECOVERY ANALYSIS ===")
    
    od_years = base_data['od_years']
    classification_map = base_data['classification_map']
    
    # Define periods
    period_years = {
        'pre_covid': [2019],
        'covid': [2020, 2021],
        'recovery': [2022, 2023],
        'current': [2024]
    }
    
    def calculate_market_shares(year_list):
        """Calculate market share for given years"""
        period_data = []
        for year in year_list:
            if year in od_years:
                df = od_years[year].copy()
                df['Business_Model'] = df['Mkt'].map(classification_map)
                period_data.append(df)
        
        if not period_data:
            return {}
        
        combined = pd.concat(period_data, ignore_index=True)
        combined = combined.dropna(subset=['Business_Model'])
        
        valid_types = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
        combined = combined[combined['Business_Model'].isin(valid_types)]
        
        model_passengers = combined.groupby('Business_Model')['Passengers'].sum()
        total_passengers = model_passengers.sum()
        
        shares = (model_passengers / total_passengers * 100).to_dict()
        
        return shares
    
    # Calculate market shares for all periods
    period_shares = {}
    for period_name, years in period_years.items():
        period_shares[period_name] = calculate_market_shares(years)
    
    # Display results
    models = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    
    for period_name, shares in period_shares.items():
        if shares:
            print(f"\n{period_name.replace('_', ' ').title()} Market Shares:")
            for model in models:
                share = shares.get(model, 0)
                print(f"  {model}: {share:.1f}%")
    
    # Calculate changes
    covid_changes = {}
    if period_shares['pre_covid'] and period_shares['recovery']:
        print(f"\nMarket Share Changes (2019 vs 2022-23):")
        for model in models:
            pre_share = period_shares['pre_covid'].get(model, 0)
            recovery_share = period_shares['recovery'].get(model, 0)
            change = recovery_share - pre_share
            covid_changes[model] = change
            print(f"  {model}: {change:+.1f} pp")
    
    return period_shares, covid_changes

#num3: Recovery timeline analysis
def analyze_recovery_timeline_h4(base_data):
    """Analyze recovery timeline for each business model"""
    
    print("\n=== H4 RECOVERY TIMELINE ANALYSIS ===")
    
    od_years = base_data['od_years']
    classification_map = base_data['classification_map']
    models = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    
    recovery_timeline = {}
    recovery_speeds = {}
    
    if 2019 in od_years:
        # Calculate baseline (2019)
        base_data_2019 = od_years[2019].copy()
        base_data_2019['Business_Model'] = base_data_2019['Mkt'].map(classification_map)
        base_data_2019 = base_data_2019[base_data_2019['Business_Model'].isin(models)]
        base_traffic = base_data_2019.groupby('Business_Model')['Passengers'].sum()
        
        print(f"\nTraffic Recovery Analysis:")
        print(f"2019 Baseline Traffic (millions):")
        for model in models:
            traffic = base_traffic.get(model, 0) / 1_000_000
            print(f"  {model}: {traffic:.1f}M")
        
        # Calculate yearly recovery rates
        for year in range(2020, 2025):
            if year in od_years:
                year_data = od_years[year].copy()
                year_data['Business_Model'] = year_data['Mkt'].map(classification_map)
                year_data = year_data[year_data['Business_Model'].isin(models)]
                year_traffic = year_data.groupby('Business_Model')['Passengers'].sum()
                
                print(f"\n{year} vs 2019 Recovery Rates:")
                
                for model in models:
                    base_value = base_traffic.get(model, 0)
                    year_value = year_traffic.get(model, 0)
                    
                    if base_value > 0:
                        recovery_rate = (year_value / base_value) * 100
                        print(f"  {model}: {recovery_rate:.1f}%")
                        
                        if model not in recovery_timeline:
                            recovery_timeline[model] = {}
                        recovery_timeline[model][year] = recovery_rate
        
        # Calculate recovery speeds (months to 90% recovery)
        print(f"\nRecovery Speed Analysis (months to 90% recovery):")
        for model in models:
            if model in recovery_timeline:
                model_timeline = recovery_timeline[model]
                
                months_to_90 = None
                for year in sorted(model_timeline.keys()):
                    if model_timeline[year] >= 90:
                        # Estimate months from 2020 start
                        months_to_90 = (year - 2020) * 12
                        if year == 2020:
                            months_to_90 = 6  # Mid-year estimate
                        elif year == 2021:
                            months_to_90 = 18  # Estimate
                        elif year == 2022:
                            months_to_90 = 24  # Estimate
                        break
                
                if months_to_90 is not None:
                    recovery_speeds[model] = months_to_90
                    print(f"  {model}: {months_to_90} months")
                else:
                    print(f"  {model}: Not yet reached 90%")
    
    return recovery_timeline, recovery_speeds

#num4: Create H4 visualization
def create_h4_figure(period_shares, covid_changes, recovery_timeline, recovery_speeds):
    """Create Figure 4.4: COVID Recovery Analysis"""
    
    os.makedirs('figures', exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Figure 4.4: COVID Recovery Analysis', fontsize=16, fontweight='bold')
    
    models = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    colors = {'Legacy': 'green', 'ULCC': 'red', 'LCC': 'orange', 'Hybrid': 'blue'}
    
    # Panel A: Pre-COVID vs Recovery shares
    if period_shares.get('pre_covid') and period_shares.get('recovery'):
        pre_shares = [period_shares['pre_covid'].get(m, 0) for m in models]
        rec_shares = [period_shares['recovery'].get(m, 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = axes[0,0].bar(x - width/2, pre_shares, width, label='2019', alpha=0.8, color='lightgray')
        bars2 = axes[0,0].bar(x + width/2, rec_shares, width, label='2022-23', alpha=0.8, 
                             color=[colors[m] for m in models])
        
        axes[0,0].set_title('Panel A: Market Share Evolution')
        axes[0,0].set_ylabel('Market Share (%)')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(models)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
    
    # Panel B: Market share changes
    if covid_changes:
        changes = [covid_changes.get(m, 0) for m in models]
        bar_colors = ['green' if x > 0 else 'red' for x in changes]
        
        bars = axes[0,1].bar(models, changes, color=bar_colors, alpha=0.7)
        axes[0,1].set_title('Panel B: Market Share Changes')
        axes[0,1].set_ylabel('Change (percentage points)')
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.2),
                          f'{change:+.1f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # Panel C: Recovery speeds
    if recovery_speeds:
        speeds = [recovery_speeds.get(m, 0) for m in models if recovery_speeds.get(m, 0) > 0]
        speed_models = [m for m in models if recovery_speeds.get(m, 0) > 0]
        
        if speeds:
            bars = axes[0,2].bar(speed_models, speeds, color=[colors[m] for m in speed_models], alpha=0.7)
            axes[0,2].set_title('Panel C: Recovery Speed')
            axes[0,2].set_ylabel('Months to 90% Recovery')
            axes[0,2].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, speed in zip(bars, speeds):
                height = bar.get_height()
                axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'{speed}', ha='center', va='bottom')
    
    # Panel D: Recovery timeline
    if recovery_timeline:
        for model in models:
            if model in recovery_timeline:
                timeline = recovery_timeline[model]
                years = sorted(timeline.keys())
                rates = [timeline[y] for y in years]
                axes[1,0].plot(years, rates, marker='o', label=model, color=colors[model], linewidth=2)
        
        axes[1,0].axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Full Recovery')
        axes[1,0].axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% Recovery')
        axes[1,0].set_title('Panel D: Recovery Timeline')
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Recovery Rate (%)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylim(0, 120)
    
    # Panel E: Trough analysis (lowest point during pandemic)
    if recovery_timeline:
        trough_data = {}
        for model in models:
            if model in recovery_timeline:
                timeline = recovery_timeline[model]
                min_rate = min(timeline.values())
                trough_data[model] = min_rate
        
        if trough_data:
            models_trough = list(trough_data.keys())
            troughs = list(trough_data.values())
            
            bars = axes[1,1].bar(models_trough, troughs, color=[colors[m] for m in models_trough], alpha=0.7)
            axes[1,1].set_title('Panel E: Trough Performance')
            axes[1,1].set_ylabel('Minimum Recovery Rate (%)')
            axes[1,1].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, trough in zip(bars, troughs):
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 1,
                              f'{trough:.1f}%', ha='center', va='bottom')
    
    # Panel F: Market share evolution all periods
    if period_shares:
        period_names = ['pre_covid', 'covid', 'recovery', 'current']
        period_labels = ['2019', '2020-21', '2022-23', '2024']
        
        x_pos = np.arange(len(period_labels))
        
        for i, model in enumerate(models):
            shares = []
            for period in period_names:
                share = period_shares.get(period, {}).get(model, 0)
                shares.append(share)
            
            axes[1,2].plot(x_pos, shares, marker='o', label=model, color=colors[model], linewidth=2)
        
        axes[1,2].set_title('Panel F: Market Share Evolution')
        axes[1,2].set_xlabel('Period')
        axes[1,2].set_ylabel('Market Share (%)')
        axes[1,2].set_xticks(x_pos)
        axes[1,2].set_xticklabels(period_labels)
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
    
#num4: Create H4 visualization
def create_h4_figure(period_shares, covid_changes, recovery_timeline, recovery_speeds):
    """Create Figure 4.4: COVID Recovery Analysis"""
    
    os.makedirs('figures', exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Figure 4.4: COVID Recovery Analysis', fontsize=16, fontweight='bold')
    
    models = ['Legacy', 'ULCC', 'LCC', 'Hybrid']
    
    # Import colors from basecode
    from basecode import CARRIER_COLORS as colors
    
    # Panel A: Pre-COVID vs Recovery shares
    if period_shares.get('pre_covid') and period_shares.get('recovery'):
        pre_shares = [period_shares['pre_covid'].get(m, 0) for m in models]
        rec_shares = [period_shares['recovery'].get(m, 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = axes[0,0].bar(x - width/2, pre_shares, width, label='2019', 
                             color='lightgray', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = axes[0,0].bar(x + width/2, rec_shares, width, label='2022-23', 
                             color=[colors[m] for m in models], alpha=0.8, 
                             edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (pre, rec) in enumerate(zip(pre_shares, rec_shares)):
            axes[0,0].text(i - width/2, pre + 0.5, f'{pre:.1f}%', ha='center', va='bottom', fontsize=8)
            axes[0,0].text(i + width/2, rec + 0.5, f'{rec:.1f}%', ha='center', va='bottom', fontsize=8)
        
        axes[0,0].set_title('Panel A: Market Share Evolution', fontweight='bold', pad=15)
        axes[0,0].set_ylabel('Market Share (%)')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(models)
        axes[0,0].legend(frameon=False)
    
    # Panel B: Market share changes
    if covid_changes:
        changes = [covid_changes.get(m, 0) for m in models]
        alphas = [0.8 if x > 0 else 0.5 for x in changes]
        
        bars = axes[0,1].bar(models, changes, color=[colors[m] for m in models], 
                            alpha=0.8, width=0.6, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            y_pos = height + 0.1 if height > 0 else height - 0.2
            axes[0,1].text(bar.get_x() + bar.get_width()/2., y_pos,
                          f'{change:+.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        axes[0,1].set_title('Panel B: Market Share Changes', fontweight='bold', pad=15)
        axes[0,1].set_ylabel('Change (percentage points)')
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    
    # Panel C: Recovery speeds
    if recovery_speeds:
        speeds = [recovery_speeds.get(m, 0) for m in models if recovery_speeds.get(m, 0) > 0]
        speed_models = [m for m in models if recovery_speeds.get(m, 0) > 0]
        
        if speeds:
            bars = axes[0,2].bar(speed_models, speeds, color=[colors[m] for m in speed_models], 
                               alpha=0.8, width=0.6, edgecolor='black', linewidth=0.5)
            axes[0,2].set_title('Panel C: Recovery Speed', fontweight='bold', pad=15)
            axes[0,2].set_ylabel('Months to 90% Recovery')
            
            # Add value labels
            for bar, speed in zip(bars, speeds):
                height = bar.get_height()
                axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'{speed}', ha='center', va='bottom', fontsize=9)
    
    # Panel D: Recovery timeline
    if recovery_timeline:
        for model in models:
            if model in recovery_timeline:
                timeline = recovery_timeline[model]
                years = sorted(timeline.keys())
                rates = [timeline[y] for y in years]
                axes[1,0].plot(years, rates, marker='o', label=model, color=colors[model], 
                             linewidth=2, markersize=6, markeredgecolor='white', markeredgewidth=1)
        
        axes[1,0].axhline(y=100, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        axes[1,0].axhline(y=90, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        axes[1,0].set_title('Panel D: Recovery Timeline', fontweight='bold', pad=15)
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Recovery Rate (%)')
        axes[1,0].legend(frameon=False, loc='lower right')
        axes[1,0].set_ylim(0, 120)
        
        # Add threshold annotations
        axes[1,0].text(2023.5, 100, '100%', fontsize=8, alpha=0.7)
        axes[1,0].text(2023.5, 90, '90%', fontsize=8, alpha=0.7)
    
    # Panel E: Trough analysis
    if recovery_timeline:
        trough_data = {}
        for model in models:
            if model in recovery_timeline:
                timeline = recovery_timeline[model]
                min_rate = min(timeline.values())
                trough_data[model] = min_rate
        
        if trough_data:
            models_trough = list(trough_data.keys())
            troughs = list(trough_data.values())
            
            bars = axes[1,1].bar(models_trough, troughs, color=[colors[m] for m in models_trough], 
                               alpha=0.8, width=0.6, edgecolor='black', linewidth=0.5)
            axes[1,1].set_title('Panel E: Trough Performance', fontweight='bold', pad=15)
            axes[1,1].set_ylabel('Minimum Recovery Rate (%)')
            
            # Add value labels
            for bar, trough in zip(bars, troughs):
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 1,
                              f'{trough:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Panel F: Market evolution across all periods
    if period_shares:
        period_names = ['pre_covid', 'covid', 'recovery', 'current']
        period_labels = ['2019', '2020-21', '2022-23', '2024']
        
        x_pos = np.arange(len(period_labels))
        
        for model in models:
            shares = []
            for period in period_names:
                share = period_shares.get(period, {}).get(model, 0)
                shares.append(share)
            
            axes[1,2].plot(x_pos, shares, marker='o', label=model, color=colors[model], 
                         linewidth=2, markersize=6, markeredgecolor='white', markeredgewidth=1)
        
        axes[1,2].set_title('Panel F: Market Share Evolution', fontweight='bold', pad=15)
        axes[1,2].set_xlabel('Period')
        axes[1,2].set_ylabel('Market Share (%)')
        axes[1,2].set_xticks(x_pos)
        axes[1,2].set_xticklabels(period_labels)
        axes[1,2].legend(frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save in both formats
    plt.savefig('figures/Figure_4_4_COVID_Recovery.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figures/Figure_4_4_COVID_Recovery.eps', format='eps', bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

#num5: Main H4 analysis function
def run_h4_analysis(base_data):
    """Run complete H4 analysis"""
    
    print("RUNNING H4: COVID RECOVERY ANALYSIS")
    print("=" * 50)
    
    # Main COVID recovery analysis
    period_shares, covid_changes = analyze_covid_recovery_h4(base_data)
    
    # Recovery timeline analysis
    recovery_timeline, recovery_speeds = analyze_recovery_timeline_h4(base_data)
    
    # Create visualization
    fig = create_h4_figure(period_shares, covid_changes, recovery_timeline, recovery_speeds)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Save period shares
    if period_shares:
        period_df = pd.DataFrame(period_shares).T
        period_df.to_csv('results/H4_Period_Market_Shares.csv')
    
    # Save COVID changes
    if covid_changes:
        changes_df = pd.DataFrame([covid_changes])
        changes_df.to_csv('results/H4_COVID_Changes.csv', index=False)
    
    # Save recovery timeline
    if recovery_timeline:
        timeline_data = []
        for model, timeline in recovery_timeline.items():
            for year, rate in timeline.items():
                timeline_data.append({
                    'Business_Model': model,
                    'Year': year,
                    'Recovery_Rate': rate
                })
        timeline_df = pd.DataFrame(timeline_data)
        timeline_df.to_csv('results/H4_Recovery_Timeline.csv', index=False)
    
    # Save recovery speeds
    if recovery_speeds:
        speeds_df = pd.DataFrame([recovery_speeds])
        speeds_df.to_csv('results/H4_Recovery_Speeds.csv', index=False)
    
    print("\nH4 Analysis Complete!")
    print("Results saved in 'results/' directory")
    print("Figures saved in 'figures/' directory")
    
    return {
        'period_shares': period_shares,
        'covid_changes': covid_changes,
        'recovery_timeline': recovery_timeline,
        'recovery_speeds': recovery_speeds,
        'figure': fig
    }

if __name__ == "__main__":
    from basecode import prepare_base_data
    base_data = prepare_base_data()
    if base_data:
        h4_results = run_h4_analysis(base_data)