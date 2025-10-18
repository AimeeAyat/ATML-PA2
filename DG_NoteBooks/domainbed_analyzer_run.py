# from DomainBedAnalyzer import DomainBedAnalyzer

# # Initialize
# analyzer = DomainBedAnalyzer(base_results_dir=r"G:\Rabia-Salman\DomainBed\domainbed\results")

# # Generate full report (addresses all research questions)
# analyzer.generate_report(output_file="domain_generalization_analysis.txt")

# # Generate comparison plots
# analyzer.generate_comparison_plots(save_dir="./analysis_plots")



from DomainBedAnalyzer1 import DomainBedAnalyzer1


analyzer = DomainBedAnalyzer1(
    base_results_dir="./domainbed/results",
    data_dir="./domainbed/data"
)

# Run complete analysis - FAST! (1-5 seconds)
analyzer.run_complete_analysis(
    generate_plots=True,
    save_dir="./analysis_plots"
)