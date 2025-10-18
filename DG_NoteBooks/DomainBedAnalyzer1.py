import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DomainBedAnalyzer1:
    """
    Comprehensive wrapper class to analyze Domain Generalization experiments on PACS dataset.
    Includes model loading, feature visualization, gradient analysis, and research question analysis.
    """
    
    def __init__(self, base_results_dir="./domainbed/results", data_dir="./domainbed/data"):
        self.base_results_dir = Path(base_results_dir)
        self.data_dir = Path(data_dir)
        self.algorithms = ["ERM", "IRM", "GroupDRO"]
        self.domains = ["art_painting", "cartoon", "photo", "sketch"]
        self.domain_indices = {0: "art_painting", 1: "cartoon", 2: "photo", 3: "sketch"}
        self.results = {}
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🖥️  Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def load_results(self):
        """Load all results from JSONL files."""
        print("\n📊 Loading results from JSONL files...")
        for algo in self.algorithms:
            self.results[algo] = {}
            for test_env in range(4):
                folder_name = f"{algo}_RESULTS_{test_env}"
                results_path = self.base_results_dir / folder_name / f"RESULTS_{algo}_{test_env}.jsonl"
                
                if results_path.exists():
                    self.results[algo][test_env] = self._parse_jsonl(results_path)
                    print(f"  ✓ Loaded {algo} test_env={test_env}")
                else:
                    print(f"  ✗ Warning: Results not found at {results_path}")
    
    def _parse_jsonl(self, filepath):
        """Parse JSONL file and extract relevant metrics."""
        records = []
        with open(filepath, 'r', encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        return records
    
    def get_best_checkpoint(self, algo, test_env):
        """Get the best checkpoint based on validation accuracy."""
        if algo not in self.results or test_env not in self.results[algo]:
            return None
        
        records = self.results[algo][test_env]
        if not records:
            return None
        
        best_record = max(records, key=lambda x: np.mean([
            x.get(f'env{i}_in_acc', 0) 
            for i in range(4) if i != test_env
        ]))
        
        return best_record
    
    def analyze_erm(self):
        """Analyze ERM baseline performance."""
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
        
        all_target_accs = [r['target_accuracy'] for r in erm_results]
        all_source_accs = [r['avg_source_accuracy'] for r in erm_results]
        all_gaps = [r['generalization_gap'] for r in erm_results]
        
        print(f"\n{'='*40}")
        print("OVERALL ERM STATISTICS")
        print(f"{'='*40}")
        print(f"Average Target Accuracy: {np.mean(all_target_accs):.2%} ± {np.std(all_target_accs):.4f}")
        print(f"Average Source Accuracy: {np.mean(all_source_accs):.2%} ± {np.std(all_source_accs):.4f}")
        print(f"Average Generalization Gap: {np.mean(all_gaps):.2%} ± {np.std(all_gaps):.4f}")
        
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
        """Analyze IRM performance."""
        print("\n" + "="*80)
        print("INVARIANT RISK MINIMIZATION (IRM) ANALYSIS")
        print("="*80)
        
        irm_results = []
        
        for test_env in range(4):
            irm_checkpoint = self.get_best_checkpoint("IRM", test_env)
            erm_checkpoint = self.get_best_checkpoint("ERM", test_env)
            
            if not irm_checkpoint or not erm_checkpoint:
                continue
            
            target_domain = self.domain_indices[test_env]
            
            irm_target_acc = irm_checkpoint.get(f'env{test_env}_out_acc', 0)
            irm_penalty = irm_checkpoint.get('irm_penalty', None)
            
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
            
            print(f"  Source Fit Trade-off: {result['source_fit_tradeoff']:+.2%}")
            print(f"  Balance Improvement: {result['balance_improvement']:+.4f}")
        
        avg_improvement = np.mean([r['improvement'] for r in irm_results])
        wins = sum(1 for r in irm_results if r['improvement'] > 0)
        
        print(f"\n{'='*40}")
        print("OVERALL IRM COMPARISON")
        print(f"{'='*40}")
        print(f"Average Improvement over ERM: {avg_improvement:+.2%}")
        print(f"IRM wins on {wins}/4 domains")
        
        return irm_results
    
    def analyze_group_dro(self):
        """Analyze Group DRO performance."""
        print("\n" + "="*80)
        print("GROUP DRO (WORST-CASE TRAINING) ANALYSIS")
        print("="*80)
        
        dro_results = []
        
        for test_env in range(4):
            dro_checkpoint = self.get_best_checkpoint("GroupDRO", test_env)
            erm_checkpoint = self.get_best_checkpoint("ERM", test_env)
            
            if not dro_checkpoint or not erm_checkpoint:
                continue
            
            target_domain = self.domain_indices[test_env]
            
            dro_target_acc = dro_checkpoint.get(f'env{test_env}_out_acc', 0)
            erm_target_acc = erm_checkpoint.get(f'env{test_env}_out_acc', 0)
            
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
            print(f"  GroupDRO Target: {dro_target_acc:.2%}")
            print(f"  ERM Target: {erm_target_acc:.2%}")
            print(f"  Improvement: {result['improvement']:+.2%}")
            print(f"  Worst-Source Improvement: {result['worst_source_improvement']:+.2%}")
            print(f"  Gap Reduction: {result['gap_reduction']:+.2%}")
        
        print(f"\n{'='*40}")
        print("KEY INSIGHTS")
        print(f"{'='*40}")
        print("📊 GroupDRO optimizes min-max objective: min_θ max_d L_d(θ)")
        print("🔗 Focuses on worst-case to prevent spurious correlations")
        
        return dro_results
    
    def generate_comparison_plots(self, save_dir="./analysis_plots"):
        """Generate comparative visualizations."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"\n📊 Generating comparison plots...")
        
        data = []
        for algo in self.algorithms:
            for test_env in range(4):
                checkpoint = self.get_best_checkpoint(algo, test_env)
                if checkpoint:
                    source_accs = [
                        checkpoint.get(f'env{i}_in_acc', 0) 
                        for i in range(4) if i != test_env
                    ]
                    data.append({
                        'Algorithm': algo,
                        'Target Domain': self.domain_indices[test_env],
                        'Target Accuracy': checkpoint.get(f'env{test_env}_out_acc', 0),
                        'Avg Source Accuracy': np.mean(source_accs),
                        'Worst Source Accuracy': min(source_accs),
                        'Source Std': np.std(source_accs)
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
        print(f"  ✓ Saved: target_accuracy_comparison.png")
        
        # Plot 2: Generalization gap
        plt.figure(figsize=(12, 8))
        for algo in self.algorithms:
            algo_data = df[df['Algorithm'] == algo]
            plt.scatter(algo_data['Avg Source Accuracy'], 
                       algo_data['Target Accuracy'],
                       label=algo, s=150, alpha=0.7)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect', linewidth=2)
        plt.xlabel('Average Source Accuracy', fontsize=12)
        plt.ylabel('Target Accuracy', fontsize=12)
        plt.title('Generalization Gap Analysis', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'generalization_gap.png', dpi=300)
        plt.close()
        print(f"  ✓ Saved: generalization_gap.png")
        
        # Plot 3: Heatmap
        pivot_data = df.pivot_table(
            values='Target Accuracy', 
            index='Target Domain', 
            columns='Algorithm'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, linewidths=0.5)
        plt.title('Target Accuracy Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / 'accuracy_heatmap.png', dpi=300)
        plt.close()
        print(f"  ✓ Saved: accuracy_heatmap.png")
        
        print(f"\n✅ All plots saved to {save_dir}")
    
    def generate_latex_table(self, output_file="results_table.tex"):
        """Generate LaTeX table."""
        print(f"\n📝 Generating LaTeX table...")
        
        table_data = []
        for test_env in range(4):
            target_domain = self.domain_indices[test_env]
            row = {'Target Domain': target_domain}
            
            for algo in self.algorithms:
                checkpoint = self.get_best_checkpoint(algo, test_env)
                if checkpoint:
                    target_acc = checkpoint.get(f'env{test_env}_out_acc', 0)
                    row[f'{algo}'] = f"{target_acc:.3f}"
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        latex_str = df.to_latex(index=False, escape=False)
        
        with open(output_file, 'w', encoding="utf-8") as f:
            f.write(latex_str)
        
        print(f"  ✓ Saved: {output_file}")
        return latex_str
    
    def generate_comprehensive_report(self, output_file="analysis_report.txt"):
        """Generate comprehensive text report."""
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = report_buffer = StringIO()
        
        print("="*80)
        print("DOMAIN GENERALIZATION ANALYSIS REPORT")
        print("PACS Dataset - Leave-One-Domain-Out")
        print("="*80)
        
        erm_results = self.analyze_erm()
        irm_results = self.analyze_irm()
        dro_results = self.analyze_group_dro()
        
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY")
        print("="*80)
        
        erm_avg = np.mean([r['target_accuracy'] for r in erm_results])
        print(f"\nERM Baseline: {erm_avg:.2%}")
        
        if irm_results:
            irm_avg = np.mean([r['irm_target_acc'] for r in irm_results])
            print(f"IRM: {irm_avg:.2%} ({irm_avg - erm_avg:+.2%})")
        
        if dro_results:
            dro_avg = np.mean([r['dro_target_acc'] for r in dro_results])
            print(f"GroupDRO: {dro_avg:.2%} ({dro_avg - erm_avg:+.2%})")
        
        sys.stdout = old_stdout
        report_content = report_buffer.getvalue()
        
        with open(output_file, 'w', encoding="utf-8") as f:
            f.write(report_content)
        
        print(report_content)
        print(f"\n✅ Report saved to {output_file}")
        
        return report_content
    
    def run_complete_analysis(self, generate_plots=True, save_dir="./analysis_plots"):
        """
        Run the complete analysis pipeline.
        
        Args:
            generate_plots: Whether to generate comparison plots
            save_dir: Directory to save outputs
        """
        print("\n" + "🚀 " + "="*76)
        print("  STARTING COMPLETE DOMAIN GENERALIZATION ANALYSIS")
        print("="*80)
        
        # Load results
        self.load_results()
        
        # Generate report
        self.generate_comprehensive_report("domain_generalization_analysis.txt")
        
        # Generate plots
        if generate_plots:
            self.generate_comparison_plots(save_dir)
        
        # Generate LaTeX table
        self.generate_latex_table("results_table.tex")
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\n📁 Outputs:")
        print(f"   - domain_generalization_analysis.txt")
        print(f"   - results_table.tex")
        print(f"   - Plots in {save_dir}/")


if __name__ == "__main__":
    analyzer = DomainBedAnalyzer(
        base_results_dir="./domainbed/results",
        data_dir="./domainbed/data"
    )
    
    analyzer.run_complete_analysis(
        generate_plots=True,
        save_dir="./analysis_plots"
    )
    
    print("\n🎉 All done! 🎉")