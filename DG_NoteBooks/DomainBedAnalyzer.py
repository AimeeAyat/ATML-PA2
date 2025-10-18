import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class DomainBedAnalyzer:
    """
    Wrapper class to analyze Domain Generalization experiments on PACS dataset.
    Addresses research questions for ERM, IRM, and GroupDRO algorithms.
    """
    
    def __init__(self, base_results_dir="./domainbed/results"):
        """
        Initialize the analyzer with the base results directory.
        
        Args:
            base_results_dir: Path to the directory containing algorithm results
        """
        self.base_results_dir = Path(base_results_dir)
        self.algorithms = ["ERM", "IRM", "GroupDRO"]
        self.domains = ["art_painting", "cartoon", "photo", "sketch"]
        self.domain_indices = {0: "art_painting", 1: "cartoon", 2: "photo", 3: "sketch"}
        self.results = {}
        
    def load_results(self):
        """Load all results from JSONL files for each algorithm and test environment."""
        for algo in self.algorithms:
            self.results[algo] = {}
            for test_env in range(4):  # 4 domains in PACS
                folder_name = f"{algo}_RESULTS_{test_env}"
                results_path = self.base_results_dir / folder_name / f"RESULTS_{algo}_{test_env}.jsonl"
                
                if results_path.exists():
                    self.results[algo][test_env] = self._parse_jsonl(results_path)
                else:
                    print(f"Warning: Results not found at {results_path}")
    
    def _parse_jsonl(self, filepath):
        """Parse JSONL file and extract relevant metrics."""
        records = []
        with open(filepath, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        return records
    
    def get_best_checkpoint(self, algo, test_env):
        """
        Get the best checkpoint based on validation accuracy.
        
        Args:
            algo: Algorithm name (ERM, IRM, GroupDRO)
            test_env: Test environment index (0-3)
            
        Returns:
            Dictionary containing best checkpoint metrics
        """
        if algo not in self.results or test_env not in self.results[algo]:
            return None
        
        records = self.results[algo][test_env]
        if not records:
            return None
        
        # Find checkpoint with best average in-domain validation accuracy
        best_record = max(records, key=lambda x: np.mean([
            x.get(f'env{i}_in_acc', 0) 
            for i in range(4) if i != test_env
        ]))
        
        return best_record
    
    def analyze_erm(self):
        """
        Analyze ERM baseline performance addressing research questions:
        1. How well does ERM perform on held-out domain?
        2. Does it overfit to source idiosyncrasies?
        3. Evidence of spurious correlations
        """
        print("="*80)
        print("ERM BASELINE ANALYSIS")
        print("="*80)
        
        erm_results = []
        
        for test_env in range(4):
            best_checkpoint = self.get_best_checkpoint("ERM", test_env)
            if not best_checkpoint:
                continue
            
            target_domain = self.domain_indices[test_env]
            target_acc = best_checkpoint.get(f'env{test_env}_out_acc', 0)
            
            # Source domain accuracies
            source_accs = []
            source_domains = []
            for i in range(4):
                if i != test_env:
                    source_accs.append(best_checkpoint.get(f'env{i}_in_acc', 0))
                    source_domains.append(self.domain_indices[i])
            
            avg_source_acc = np.mean(source_accs)
            source_std = np.std(source_accs)
            generalization_gap = avg_source_acc - target_acc
            
            result = {
                'target_domain': target_domain,
                'target_accuracy': target_acc,
                'source_accuracies': dict(zip(source_domains, source_accs)),
                'avg_source_accuracy': avg_source_acc,
                'source_std': source_std,
                'generalization_gap': generalization_gap,
                'step': best_checkpoint.get('step', 0)
            }
            erm_results.append(result)
            
            print(f"\nTest Domain: {target_domain}")
            print(f"  Target Accuracy: {target_acc:.2%}")
            print(f"  Average Source Accuracy: {avg_source_acc:.2%}")
            print(f"  Source Std Dev: {source_std:.4f}")
            print(f"  Generalization Gap: {generalization_gap:.2%}")
            print(f"  Individual Source Accuracies:")
            for domain, acc in zip(source_domains, source_accs):
                print(f"    {domain}: {acc:.2%}")
        
        # Overall statistics
        all_target_accs = [r['target_accuracy'] for r in erm_results]
        all_source_accs = [r['avg_source_accuracy'] for r in erm_results]
        all_gaps = [r['generalization_gap'] for r in erm_results]
        
        print(f"\n{'='*40}")
        print("OVERALL ERM STATISTICS")
        print(f"{'='*40}")
        print(f"Average Target Accuracy: {np.mean(all_target_accs):.2%} ± {np.std(all_target_accs):.4f}")
        print(f"Average Source Accuracy: {np.mean(all_source_accs):.2%} ± {np.std(all_source_accs):.4f}")
        print(f"Average Generalization Gap: {np.mean(all_gaps):.2%} ± {np.std(all_gaps):.4f}")
        
        # Insights
        print(f"\n{'='*40}")
        print("KEY INSIGHTS")
        print(f"{'='*40}")
        if np.mean(all_gaps) > 0.05:
            print("⚠ Large generalization gap detected!")
            print("  → Model likely overfits to source-specific features")
            print("  → Evidence of spurious correlations that don't transfer")
        
        if np.mean([r['source_std'] for r in erm_results]) > 0.1:
            print("⚠ High variance in source domain performance")
            print("  → Model struggles with certain source domains")
            print("  → May benefit from domain-aware training strategies")
        
        return erm_results
    
    def analyze_irm(self):
        """
        Analyze IRM performance addressing research questions:
        1. Does IRM improve generalization vs ERM?
        2. Check for trivial solutions (zero penalty + low accuracy)
        3. Trade-off between source fit and invariance
        4. Domain balance improvements
        """
        print("\n" + "="*80)
        print("INVARIANT RISK MINIMIZATION (IRM) ANALYSIS")
        print("="*80)
        
        irm_results = []
        erm_results = []
        
        for test_env in range(4):
            irm_checkpoint = self.get_best_checkpoint("IRM", test_env)
            erm_checkpoint = self.get_best_checkpoint("ERM", test_env)
            
            if not irm_checkpoint or not erm_checkpoint:
                continue
            
            target_domain = self.domain_indices[test_env]
            
            # IRM metrics
            irm_target_acc = irm_checkpoint.get(f'env{test_env}_out_acc', 0)
            irm_penalty = irm_checkpoint.get('irm_penalty', None)
            
            # Source accuracies
            irm_source_accs = [
                irm_checkpoint.get(f'env{i}_in_acc', 0) 
                for i in range(4) if i != test_env
            ]
            erm_source_accs = [
                erm_checkpoint.get(f'env{i}_in_acc', 0) 
                for i in range(4) if i != test_env
            ]
            
            irm_avg_source = np.mean(irm_source_accs)
            erm_avg_source = np.mean(erm_source_accs)
            
            irm_source_balance = np.std(irm_source_accs)
            erm_source_balance = np.std(erm_source_accs)
            
            # ERM comparison
            erm_target_acc = erm_checkpoint.get(f'env{test_env}_out_acc', 0)
            improvement = irm_target_acc - erm_target_acc
            
            result = {
                'target_domain': target_domain,
                'irm_target_acc': irm_target_acc,
                'erm_target_acc': erm_target_acc,
                'improvement': improvement,
                'irm_penalty': irm_penalty,
                'irm_avg_source': irm_avg_source,
                'erm_avg_source': erm_avg_source,
                'source_fit_tradeoff': erm_avg_source - irm_avg_source,
                'irm_source_balance': irm_source_balance,
                'erm_source_balance': erm_source_balance,
                'balance_improvement': erm_source_balance - irm_source_balance
            }
            irm_results.append(result)
            
            print(f"\nTest Domain: {target_domain}")
            print(f"  IRM Target Accuracy: {irm_target_acc:.2%}")
            print(f"  ERM Target Accuracy: {erm_target_acc:.2%}")
            print(f"  Improvement over ERM: {improvement:+.2%}")
            
            if irm_penalty is not None:
                print(f"  IRM Penalty: {irm_penalty:.6f}")
                if irm_penalty < 0.01 and irm_target_acc < 0.5:
                    print("  ⚠ WARNING: Near-zero penalty + low accuracy = possible trivial solution!")
            
            print(f"  Source Fit Trade-off (ERM - IRM): {result['source_fit_tradeoff']:+.2%}")
            print(f"  IRM Source Balance (std): {irm_source_balance:.4f}")
            print(f"  ERM Source Balance (std): {erm_source_balance:.4f}")
            print(f"  Balance Improvement: {result['balance_improvement']:+.4f}")
        
        # Overall comparison
        avg_improvement = np.mean([r['improvement'] for r in irm_results])
        wins = sum(1 for r in irm_results if r['improvement'] > 0)
        
        print(f"\n{'='*40}")
        print("OVERALL IRM COMPARISON")
        print(f"{'='*40}")
        print(f"Average Improvement over ERM: {avg_improvement:+.2%}")
        print(f"IRM wins on {wins}/4 domains")
        print(f"Average Source Fit Trade-off: {np.mean([r['source_fit_tradeoff'] for r in irm_results]):+.2%}")
        print(f"Average Balance Improvement: {np.mean([r['balance_improvement'] for r in irm_results]):+.4f}")
        
        # Insights
        print(f"\n{'='*40}")
        print("KEY INSIGHTS")
        print(f"{'='*40}")
        
        if avg_improvement < 0:
            print("⚠ IRM underperforms ERM on average")
            print("  Possible reasons:")
            print("  → Optimization difficulty with IRM penalty")
            print("  → Insufficient domains for learning invariances")
            print("  → Hyperparameter tuning needed (penalty weight, learning rate)")
            print("  → Dataset may not have strong spurious correlations")
        elif avg_improvement > 0.02:
            print("✓ IRM shows meaningful improvement!")
            print("  → Successfully learned domain-invariant features")
            print("  → Better generalization to unseen distributions")
        
        avg_tradeoff = np.mean([r['source_fit_tradeoff'] for r in irm_results])
        if avg_tradeoff > 0.05:
            print(f"\n✓ IRM trades off source fit for invariance ({avg_tradeoff:.2%})")
            print("  → This is expected and desirable for better generalization")
        
        return irm_results
    
    def analyze_group_dro(self):
        """
        Analyze Group DRO performance addressing research questions:
        1. Does worst-case training improve unseen domain performance?
        2. Does it balance source domain performance?
        3. Connection to distributionally robust optimization
        4. Relationship to spurious correlations
        """
        print("\n" + "="*80)
        print("GROUP DRO (WORST-CASE TRAINING) ANALYSIS")
        print("="*80)
        
        dro_results = []
        erm_results = []
        
        for test_env in range(4):
            dro_checkpoint = self.get_best_checkpoint("GroupDRO", test_env)
            erm_checkpoint = self.get_best_checkpoint("ERM", test_env)
            
            if not dro_checkpoint or not erm_checkpoint:
                continue
            
            target_domain = self.domain_indices[test_env]
            
            # Target accuracies
            dro_target_acc = dro_checkpoint.get(f'env{test_env}_out_acc', 0)
            erm_target_acc = erm_checkpoint.get(f'env{test_env}_out_acc', 0)
            
            # Source domain metrics
            dro_source_accs = [
                dro_checkpoint.get(f'env{i}_in_acc', 0) 
                for i in range(4) if i != test_env
            ]
            erm_source_accs = [
                erm_checkpoint.get(f'env{i}_in_acc', 0) 
                for i in range(4) if i != test_env
            ]
            
            dro_worst_source = min(dro_source_accs)
            erm_worst_source = min(erm_source_accs)
            
            dro_best_source = max(dro_source_accs)
            erm_best_source = max(erm_source_accs)
            
            dro_source_gap = dro_best_source - dro_worst_source
            erm_source_gap = erm_best_source - erm_worst_source
            
            result = {
                'target_domain': target_domain,
                'dro_target_acc': dro_target_acc,
                'erm_target_acc': erm_target_acc,
                'improvement': dro_target_acc - erm_target_acc,
                'dro_worst_source': dro_worst_source,
                'erm_worst_source': erm_worst_source,
                'worst_source_improvement': dro_worst_source - erm_worst_source,
                'dro_source_gap': dro_source_gap,
                'erm_source_gap': erm_source_gap,
                'gap_reduction': erm_source_gap - dro_source_gap,
                'dro_avg_source': np.mean(dro_source_accs),
                'erm_avg_source': np.mean(erm_source_accs)
            }
            dro_results.append(result)
            
            print(f"\nTest Domain: {target_domain}")
            print(f"  GroupDRO Target Accuracy: {dro_target_acc:.2%}")
            print(f"  ERM Target Accuracy: {erm_target_acc:.2%}")
            print(f"  Improvement over ERM: {result['improvement']:+.2%}")
            print(f"\n  Worst-Case Source Performance:")
            print(f"    GroupDRO: {dro_worst_source:.2%}")
            print(f"    ERM: {erm_worst_source:.2%}")
            print(f"    Improvement: {result['worst_source_improvement']:+.2%}")
            print(f"\n  Source Domain Gap (Best - Worst):")
            print(f"    GroupDRO: {dro_source_gap:.2%}")
            print(f"    ERM: {erm_source_gap:.2%}")
            print(f"    Gap Reduction: {result['gap_reduction']:+.2%}")
        
        # Overall statistics
        print(f"\n{'='*40}")
        print("OVERALL GROUP DRO COMPARISON")
        print(f"{'='*40}")
        
        avg_target_improvement = np.mean([r['improvement'] for r in dro_results])
        avg_worst_improvement = np.mean([r['worst_source_improvement'] for r in dro_results])
        avg_gap_reduction = np.mean([r['gap_reduction'] for r in dro_results])
        
        print(f"Average Target Improvement: {avg_target_improvement:+.2%}")
        print(f"Average Worst-Source Improvement: {avg_worst_improvement:+.2%}")
        print(f"Average Gap Reduction: {avg_gap_reduction:+.2%}")
        print(f"Wins on target domain: {sum(1 for r in dro_results if r['improvement'] > 0)}/4")
        
        # Theoretical insights
        print(f"\n{'='*40}")
        print("KEY INSIGHTS & CONNECTIONS")
        print(f"{'='*40}")
        
        print("\n📊 Connection to Distributionally Robust Optimization:")
        print("  GroupDRO optimizes a min-max objective: min_θ max_d L_d(θ)")
        print("  → Guards against worst-case domain shifts")
        print("  → Theoretically robust to distributional uncertainty")
        
        if avg_gap_reduction > 0:
            print(f"\n✓ Successfully reduced domain performance gap by {avg_gap_reduction:.2%}")
            print("  → More balanced performance across sources")
            print("  → Less reliance on easy-to-learn spurious features")
        
        if avg_worst_improvement > 0.02:
            print(f"\n✓ Significant worst-case improvement ({avg_worst_improvement:.2%})")
            print("  → Model learned features useful across all domains")
            print("  → Reduced exploitation of domain-specific shortcuts")
        
        print("\n🔗 Relationship to Spurious Correlations:")
        print("  → Easy domains often have stronger spurious features")
        print("  → Hard domains force model to learn robust features")
        print("  → Focusing on worst-case prevents over-reliance on spurious cues")
        print("  → This naturally addresses the spurious correlation problem")
        
        if avg_target_improvement > 0:
            print(f"\n✓ Improved unseen domain generalization ({avg_target_improvement:.2%})")
            print("  → Worst-case training transfers to new distributions")
        else:
            print("\n⚠ Limited improvement on unseen domains")
            print("  Possible reasons:")
            print("  → May need stronger regularization (L2, dropout)")
            print("  → Early stopping might be necessary")
            print("  → Overparameterized network can still memorize patterns")
        
        return dro_results
    
    def generate_comparison_plots(self, save_dir="./analysis_plots"):
        """Generate comparative visualizations for all algorithms."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Collect data
        data = []
        for algo in self.algorithms:
            for test_env in range(4):
                checkpoint = self.get_best_checkpoint(algo, test_env)
                if checkpoint:
                    data.append({
                        'Algorithm': algo,
                        'Target Domain': self.domain_indices[test_env],
                        'Target Accuracy': checkpoint.get(f'env{test_env}_out_acc', 0),
                        'Avg Source Accuracy': np.mean([
                            checkpoint.get(f'env{i}_in_acc', 0) 
                            for i in range(4) if i != test_env
                        ])
                    })
        
        df = pd.DataFrame(data)
        
        # Plot 1: Target accuracy comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Target Domain', y='Target Accuracy', hue='Algorithm')
        plt.title('Target Domain Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend(title='Algorithm')
        plt.tight_layout()
        plt.savefig(save_dir / 'target_accuracy_comparison.png', dpi=300)
        plt.close()
        
        # Plot 2: Source vs Target accuracy
        plt.figure(figsize=(12, 6))
        for algo in self.algorithms:
            algo_data = df[df['Algorithm'] == algo]
            plt.scatter(algo_data['Avg Source Accuracy'], 
                       algo_data['Target Accuracy'],
                       label=algo, s=100, alpha=0.7)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Generalization')
        plt.xlabel('Average Source Accuracy')
        plt.ylabel('Target Accuracy')
        plt.title('Generalization Gap Analysis', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'generalization_gap.png', dpi=300)
        plt.close()
        
        print(f"\n📊 Plots saved to {save_dir}")
    
    def generate_report(self, output_file="analysis_report.txt"):
        """Generate a comprehensive text report addressing all research questions."""
        import sys
        from io import StringIO
        
        # Capture all print output
        old_stdout = sys.stdout
        sys.stdout = report_buffer = StringIO()
        
        # Run all analyses
        self.load_results()
        erm_results = self.analyze_erm()
        irm_results = self.analyze_irm()
        dro_results = self.analyze_group_dro()
        
        # Restore stdout
        sys.stdout = old_stdout
        report_content = report_buffer.getvalue()
        
        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        
        # Also print to console
        print(report_content)
        print(f"\n📄 Full report saved to {output_file}")
        
        return report_content


# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = DomainBedAnalyzer(base_results_dir="./domainbed/results")
    
    # Generate comprehensive report
    analyzer.generate_report(output_file="domain_generalization_analysis.txt")
    
    # Generate comparison plots
    analyzer.generate_comparison_plots(save_dir="./analysis_plots")