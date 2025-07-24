# run_analysis.py  
# #num8: Simple script to run the complete analysis

from main_analysis import AirlineAnalysisController
from data_loader import DataLoader
import sys
from pathlib import Path

def main():
    """Run the complete airline strategy analysis"""
    print("="*80)
    print("AIRLINE STRATEGIC VOLATILITY ANALYSIS")
    print("Ultra-Low-Cost Carriers in U.S. Domestic Networks (2014-2024)")
    print("="*80)
    
    # Check if data directory exists
    data_path = "data"
    if not Path(data_path).exists():
        print(f"Error: Data directory '{data_path}' not found!")
        print("Please ensure your data files are in the correct location:")
        print("  data/od/od_YYYY.parquet")
        print("  data/t_100/t_100_YYYY.parquet") 
        print("  data/analysis/shock_2014_2024.parquet")
        return False
        
    # Create data loader and check available data
    data_loader = DataLoader(data_path)
    data_summary = data_loader.get_data_summary()
    
    print("\nData Availability Check:")
    print(f"O&D files: {len(data_summary['od_files'])}")
    print(f"T-100 files: {len(data_summary['t100_files'])}")
    print(f"Analysis files: {len(data_summary['analysis_files'])}")
    
    if len(data_summary['od_files']) == 0:
        print("Warning: No O&D data files found!")
        return False
        
    # Create output directory
    Path("report").mkdir(exist_ok=True)
    
    # Initialize and run analysis
    try:
        controller = AirlineAnalysisController(data_path)
        results = controller.run_complete_analysis()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nResults saved in 'report/' directory:")
        print("📊 Figures: Fig_4_1 through Fig_4_5")
        print("📋 Tables: Table_4_1 through Table_4_4") 
        print("📈 Data: CSV files with detailed results")
        print("\n" + "="*80)
        
        return True
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("Please check your data files and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
