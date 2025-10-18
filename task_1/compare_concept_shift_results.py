"""
Utility script to compare concept shift evaluation results across all methods.
Run this after executing all four concept shift evaluation scripts.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Define methods and their output directories
METHODS = {
    'Baseline': 'baseline_concept_shift_results/baseline_concept_shift_summary.txt',
    'DAN-MMD': 'DAN_concept_shift_results/dan_concept_shift_summary.txt',
    'DANN': 'DANN_concept_shift_results/dann_concept_shift_summary.txt',
    'CDAN': 'CDAN_concept_shift_results/cdan_concept_shift_summary.txt'
}

def parse_summary_file(filepath):
    """Parse summary text file and extract accuracy metrics"""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Run the corresponding evaluation script first.")
        return None
    
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            if 'Original Target Distribution Accuracy:' in line:
                metrics['original'] = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Label Shift Scenario Accuracy:' in line:
                metrics['label_shift'] = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Rare-Class Scenario Accuracy:' in line:
                metrics['rare_class'] = float(line.split(':')[1].strip().replace('%', ''))
    
    return metrics

def create_comparison_plots():
    """Create comparison visualizations across all methods"""
    
    # Parse all results
    results = {}
    for method, filepath in METHODS.items():
        metrics = parse_summary_file(filepath)
        if metrics:
            results[method] = metrics
    
    if not results:
        print("No results found. Please run the concept shift evaluation scripts first.")
        return
    
    # Prepare data for plotting
    methods = list(results.keys())
    original_accs = [results[m]['original'] for m in methods]
    label_shift_accs = [results[m]['label_shift'] for m in methods]
    rare_class_accs = [results[m]['rare_class'] for m in methods]
    
    # Calculate degradation
    label_shift_degradation = [results[m]['original'] - results[m]['label_shift'] for m in methods]
    rare_class_degradation = [results[m]['original'] - results[m]['rare_class'] for m in methods]
    
    # Create comprehensive comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Accuracy Comparison
    x = np.arange(len(methods))
    width = 0.25
    
    ax1 = axes[0, 0]
    ax1.bar(x - width, original_accs, width, label='Original', color='green', alpha=0.7)
    ax1.bar(x, label_shift_accs, width, label='Label Shift', color='orange', alpha=0.7)
    ax1.bar(x + width, rare_class_accs, width, label='Rare Class', color='red', alpha=0.7)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Comparison Across Concept Shifts')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Add value labels on bars
    for i, method in enumerate(methods):
        ax1.text(i - width, original_accs[i] + 1, f'{original_accs[i]:.1f}', 
                ha='center', va='bottom', fontsize=8)
        ax1.text(i, label_shift_accs[i] + 1, f'{label_shift_accs[i]:.1f}', 
                ha='center', va='bottom', fontsize=8)
        ax1.text(i + width, rare_class_accs[i] + 1, f'{rare_class_accs[i]:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Performance Degradation
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, label_shift_degradation, width, label='Label Shift Impact', 
           color='orange', alpha=0.7)
    ax2.bar(x + width/2, rare_class_degradation, width, label='Rare Class Impact', 
           color='red', alpha=0.7)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Accuracy Drop (%)')
    ax2.set_title('Performance Degradation Under Concept Shifts')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for i, method in enumerate(methods):
        ax2.text(i - width/2, label_shift_degradation[i] + 0.5, 
                f'{label_shift_degradation[i]:.1f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + width/2, rare_class_degradation[i] + 0.5, 
                f'{rare_class_degradation[i]:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Robustness Score (inverse of degradation)
    ax3 = axes[1, 0]
    robustness_label = [100 - d for d in label_shift_degradation]
    robustness_rare = [100 - d for d in rare_class_degradation]
    avg_robustness = [(l + r) / 2 for l, r in zip(robustness_label, robustness_rare)]
    
    ax3.bar(x, avg_robustness, color='purple', alpha=0.7)
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Robustness Score (%)')
    ax3.set_title('Overall Robustness to Concept Shifts\n(Higher is Better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 100])
    
    # Add value labels
    for i, score in enumerate(avg_robustness):
        ax3.text(i, score + 1, f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Relative Performance Matrix
    ax4 = axes[1, 1]
    
    # Calculate relative performance (% of original accuracy retained)
    label_shift_retention = [(results[m]['label_shift'] / results[m]['original']) * 100 
                             for m in methods]
    rare_class_retention = [(results[m]['rare_class'] / results[m]['original']) * 100 
                           for m in methods]
    
    retention_data = np.array([label_shift_retention, rare_class_retention])
    
    im = ax4.imshow(retention_data, cmap='RdYlGn', aspect='auto', vmin=70, vmax=100)
    ax4.set_xticks(np.arange(len(methods)))
    ax4.set_yticks(np.arange(2))
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.set_yticklabels(['Label Shift', 'Rare Class'])
    ax4.set_title('Accuracy Retention Rate (%)\n(% of Original Accuracy Maintained)')
    
    # Add text annotations
    for i in range(2):
        for j in range(len(methods)):
            text = ax4.text(j, i, f'{retention_data[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=10, weight='bold')
    
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('concept_shift_comparison_all_methods.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plot saved as: concept_shift_comparison_all_methods.png")
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("CONCEPT SHIFT EVALUATION - SUMMARY TABLE")
    print("="*80)
    print(f"{'Method':<12} | {'Original':>10} | {'Label Shift':>12} | {'Rare Class':>11} | {'Avg Robustness':>15}")
    print("-"*80)
    for i, method in enumerate(methods):
        print(f"{method:<12} | {original_accs[i]:>9.2f}% | {label_shift_accs[i]:>11.2f}% | "
              f"{rare_class_accs[i]:>10.2f}% | {avg_robustness[i]:>14.2f}%")
    print("="*80)
    
    # Identify best and worst performers
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    best_original_idx = np.argmax(original_accs)
    best_robustness_idx = np.argmax(avg_robustness)
    worst_label_shift_idx = np.argmax(label_shift_degradation)
    worst_rare_class_idx = np.argmax(rare_class_degradation)
    
    print(f"\n✓ Best Original Accuracy: {methods[best_original_idx]} ({original_accs[best_original_idx]:.2f}%)")
    print(f"✓ Most Robust to Shifts: {methods[best_robustness_idx]} ({avg_robustness[best_robustness_idx]:.2f}% robustness)")
    print(f"⚠ Most Affected by Label Shift: {methods[worst_label_shift_idx]} (-{label_shift_degradation[worst_label_shift_idx]:.2f}%)")
    print(f"⚠ Most Affected by Rare Classes: {methods[worst_rare_class_idx]} (-{rare_class_degradation[worst_rare_class_idx]:.2f}%)")
    
    # Domain adaptation effectiveness
    if 'Baseline' in results:
        print("\n" + "="*80)
        print("DOMAIN ADAPTATION EFFECTIVENESS")
        print("="*80)
        baseline_orig = results['Baseline']['original']
        for method in methods:
            if method != 'Baseline':
                improvement = results[method]['original'] - baseline_orig
                print(f"{method} vs Baseline: {improvement:+.2f}% on original target distribution")
        
        # Robustness comparison
        baseline_robustness = avg_robustness[methods.index('Baseline')]
        print("\nRobustness Comparison to Baseline:")
        for i, method in enumerate(methods):
            if method != 'Baseline':
                robustness_diff = avg_robustness[i] - baseline_robustness
                print(f"{method}: {robustness_diff:+.2f}% {'more' if robustness_diff > 0 else 'less'} robust")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    print("\n" + "="*80)
    print("CONCEPT SHIFT RESULTS COMPARISON TOOL")
    print("="*80)
    print("\nThis script compares evaluation results across all domain adaptation methods.")
    print("Make sure you have run all four concept shift evaluation scripts first:\n")
    for method, filepath in METHODS.items():
        status = "✓" if os.path.exists(filepath) else "✗"
        print(f"  {status} {method}: {filepath}")
    print()
    
    create_comparison_plots()



