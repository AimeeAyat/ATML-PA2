# Concept Shift Evaluation for Domain Adaptation Methods

## Overview
This directory contains scripts to evaluate four domain adaptation methods under concept shift scenarios:
1. **Baseline (Source-Only)** - Model trained only on source domain
2. **DAN-MMD** - Domain Adversarial Network with Maximum Mean Discrepancy
3. **DANN** - Domain Adversarial Neural Network with Gradient Reversal Layer
4. **CDAN** - Conditional Domain Adversarial Network with multilinear conditioning

## Files Created

### 1. `concept_shift.py` (Updated)
- **Model**: Baseline Source-Only
- **Checkpoints**: `baseline_checkpoints/`
- **Source → Target**: art_painting → cartoon
- **Output**: `baseline_concept_shift_results/`

### 2. `concept_shift_dan.py` (New)
- **Model**: DAN-MMD (Domain Alignment with MMD loss)
- **Checkpoints**: `DAN_checkpoints/`
- **Source → Target**: art_painting → cartoon
- **Output**: `DAN_concept_shift_results/`

### 3. `concept_shift_dann.py` (New)
- **Model**: DANN (Domain Adversarial with Gradient Reversal)
- **Checkpoints**: `DANN_checkpoints/`
- **Source → Target**: art_painting → photo
- **Output**: `DANN_concept_shift_results/`

### 4. `concept_shift_cdan.py` (New)
- **Model**: CDAN (Conditional Domain Adversarial with Multilinear Map)
- **Checkpoints**: `CDAN_checkpoints/`
- **Source → Target**: art_painting → photo
- **Output**: `CDAN_concept_shift_results/`

## Concept Shift Scenarios Evaluated

Each script evaluates the model under three scenarios:

### 1. **Original Target Distribution**
- Baseline evaluation on unmodified target domain test set
- Provides reference accuracy for comparison

### 2. **Label Shift Scenario**
- **Downsample**: Reduces one class (e.g., class 0) to 10% of original samples
- **Oversample**: Duplicates another class (e.g., class 1) by 2x
- Tests model robustness to class imbalance

### 3. **Rare-Class Scenario**
- **Downsample**: Reduces one class (e.g., class 2) to 5% of original samples
- Tests model performance on severely underrepresented classes
- Identifies potential negative transfer effects

## Model Architectures

### Baseline & DAN-MMD
```
FeatureExtractor (ResNet18) → Classifier
```
- Same architecture, different training objectives
- Baseline: Only classification loss
- DAN: Classification + MMD loss for domain alignment

### DANN
```
FeatureExtractor (ResNet18) → Classifier
                            → GRL → DomainDiscriminator
```
- Additional domain discriminator with gradient reversal layer
- Adversarial training for domain-invariant features

### CDAN
```
FeatureExtractor (ResNet18) → Classifier → Softmax
                            ↓              ↓
                    MultilinearMap (features ⊗ predictions)
                            ↓
                    GRL → DomainDiscriminator
```
- Most complex: conditions domain adaptation on class predictions
- Uses randomized multilinear map for efficiency
- Better alignment of class-specific features

## Output Structure

Each evaluation script generates:

### Visualizations
1. **Confusion Matrices** (3 files)
   - `{method}_original_target_confusion_matrix.png`
   - `{method}_label_shift_target_confusion_matrix.png`
   - `{method}_rare_class_target_confusion_matrix.png`

2. **Class-wise Accuracy Heatmaps** (3 files)
   - `{method}_original_target_class_accuracy_heatmap.png`
   - `{method}_label_shift_target_class_accuracy_heatmap.png`
   - `{method}_rare_class_target_class_accuracy_heatmap.png`

### Summary Metrics
- `{method}_concept_shift_summary.txt`
  - Original accuracy
  - Label shift accuracy
  - Rare-class accuracy
  - Performance degradation metrics

## Usage

### Run Baseline Evaluation
```bash
python concept_shift.py
```

### Run DAN-MMD Evaluation
```bash
python concept_shift_dan.py
```

### Run DANN Evaluation
```bash
python concept_shift_dann.py
```

### Run CDAN Evaluation
```bash
python concept_shift_cdan.py
```

## Configuration Parameters

Each script has configurable parameters in the `Config` class:

```python
# Concept Shift parameters
downsample_class_label_shift = 0      # Class to downsample for label shift
downsample_ratio_label_shift = 0.1    # Keep 10% of this class
oversample_class_label_shift = 1      # Class to oversample
oversample_factor_label_shift = 2     # Multiply samples by 2

rare_class_label = 2                  # Class to make rare
rare_class_ratio = 0.05              # Keep 5% of this class
```

You can modify these to test different concept shift scenarios.

## Key Findings to Analyze

After running all evaluations, compare:

1. **Overall Accuracy**
   - Which method handles concept shifts best?
   - Does domain adaptation help or hurt under distribution shift?

2. **Label Shift Robustness**
   - How does each method handle class imbalance?
   - Impact of downsampling vs. oversampling

3. **Rare-Class Performance**
   - Which method maintains performance on rare classes?
   - Evidence of negative transfer in domain adaptation methods?

4. **Class-wise Analysis**
   - Which classes are most affected?
   - Confusion patterns across methods

## Expected Behavior

- **Baseline**: May perform poorly on target domain but potentially more robust to distribution shifts
- **DAN-MMD**: Better target accuracy but may be sensitive to distribution changes
- **DANN**: Similar to DAN but with adversarial training benefits/drawbacks
- **CDAN**: Best target accuracy on original distribution but may suffer on shifted distributions due to complex conditioning

## Notes

- All scripts use the same seed (42) for reproducibility
- Batch size: 128 (can be adjusted based on GPU memory)
- Models are loaded from pre-trained checkpoints (no training occurs)
- Visualizations use `plt.close()` to avoid memory issues with multiple plots
- Results are saved automatically, no manual intervention needed

## Troubleshooting

If checkpoint loading fails:
1. Check that checkpoint files exist in the respective directories
2. Verify source and target domain names match checkpoint filenames
3. Scripts try `_final.pth` first, then fall back to latest epoch checkpoint

## Comparison Analysis

To compare all methods:
1. Run all four scripts
2. Compare the summary `.txt` files
3. Visually inspect confusion matrices and class-wise accuracy heatmaps
4. Calculate performance gaps: `Original - Shifted`
5. Identify which method is most robust to concept shifts



