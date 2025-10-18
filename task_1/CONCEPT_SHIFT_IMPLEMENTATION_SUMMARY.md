# Concept Shift Implementation Summary

## What Was Done

I've successfully created a complete evaluation framework to test how **Baseline (Source-Only)**, **DAN-MMD**, **DANN**, and **CDAN** domain adaptation models perform under **concept shift** scenarios.

---

## 📁 Files Created/Modified

### 1. **Modified**: `concept_shift.py`
**Changes Made:**
- ✅ Fixed checkpoint loading to use `baseline_checkpoints/` directory instead of `concept_checkpoints/`
- ✅ Updated target domain from `photo` to `cartoon` (matching baseline checkpoint)
- ✅ Changed output directory to `baseline_concept_shift_results/`
- ✅ Removed unnecessary checkpoint saving (only evaluation, no training)
- ✅ Added summary metrics file export
- ✅ Fixed visualization file paths
- ✅ Changed `plt.show()` to `plt.close()` to prevent memory issues

**Loading Strategy:**
```python
checkpoint_path = baseline_checkpoints/source_only_art_painting_to_cartoon_final.pth
# Falls back to: source_only_art_painting_to_cartoon_epoch_10.pth
```

---

### 2. **Created**: `concept_shift_dan.py`
**Purpose:** Evaluate DAN-MMD (Domain Alignment with MMD loss) under concept shifts

**Key Details:**
- **Checkpoint Dir**: `DAN_checkpoints/`
- **Domain Pair**: art_painting → cartoon
- **Model Architecture**: Same as baseline (FeatureExtractor + Classifier)
- **Checkpoint Loading**: `dan_art_painting_to_cartoon_final.pth`
- **Output Dir**: `DAN_concept_shift_results/`

**Note:** DAN has the same architecture as baseline but was trained with additional MMD loss for domain alignment.

---

### 3. **Created**: `concept_shift_dann.py`
**Purpose:** Evaluate DANN (Domain Adversarial Neural Network) under concept shifts

**Key Details:**
- **Checkpoint Dir**: `DANN_checkpoints/`
- **Domain Pair**: art_painting → photo ⚠️ (Different from baseline/DAN)
- **Model Architecture**: FeatureExtractor + Classifier + DomainDiscriminator + GRL
- **Checkpoint Loading**: `dann_art_painting_to_photo_final.pth`
- **Output Dir**: `DANN_concept_shift_results/`

**Special Components:**
- `GradientReversalLayer`: Reverses gradients during backprop
- `DomainDiscriminator`: Binary classifier for source vs target
- Model returns: `(class_output, domain_output, features)`

---

### 4. **Created**: `concept_shift_cdan.py`
**Purpose:** Evaluate CDAN (Conditional Domain Adversarial Network) under concept shifts

**Key Details:**
- **Checkpoint Dir**: `CDAN_checkpoints/`
- **Domain Pair**: art_painting → photo ⚠️ (Different from baseline/DAN)
- **Model Architecture**: FeatureExtractor + Classifier + MultilinearMap + DomainDiscriminator + GRL
- **Checkpoint Loading**: `cdan_art_painting_to_photo_final.pth`
- **Output Dir**: `CDAN_concept_shift_results/`

**Special Components:**
- `RandomizedMultiLinearMap`: Conditions domain adaptation on class predictions
  - Input: features ⊗ class_predictions
  - Output: Lower-dimensional projection for discriminator
- Model returns: `(class_output, domain_output, features, class_predictions)`

**Most Complex Architecture:** CDAN is the most sophisticated, conditioning alignment on predicted classes.

---

### 5. **Created**: `CONCEPT_SHIFT_EVALUATION_README.md`
Comprehensive documentation explaining:
- Purpose of each script
- Model architectures
- Concept shift scenarios
- Expected outputs
- Usage instructions
- Configuration parameters

---

### 6. **Created**: `compare_concept_shift_results.py`
**Utility Script** to compare all methods after evaluation:

**Features:**
- ✅ Parses all summary `.txt` files
- ✅ Creates comparison visualizations:
  - Accuracy comparison bar chart
  - Performance degradation analysis
  - Robustness score comparison
  - Retention rate heatmap
- ✅ Generates comprehensive summary table
- ✅ Identifies best/worst performers
- ✅ Analyzes domain adaptation effectiveness vs baseline

**Output:** `concept_shift_comparison_all_methods.png`

---

## 🧪 Concept Shift Scenarios

All scripts evaluate models on **three scenarios**:

### 1. **Original Target Distribution**
- No modifications to target test set
- Provides baseline performance for comparison

### 2. **Label Shift Scenario**
- **Downsample** class 0 (e.g., 'dog') → Keep only 10%
- **Oversample** class 1 (e.g., 'elephant') → Duplicate 2×
- **Tests**: Model robustness to class imbalance

### 3. **Rare-Class Scenario**
- **Downsample** class 2 (e.g., 'giraffe') → Keep only 5%
- **Tests**: Performance on severely underrepresented classes
- **Identifies**: Potential negative transfer from domain adaptation

---

## 📊 Output Structure

Each script generates:

### Visualizations (6 images per method)
1. `{method}_original_target_confusion_matrix.png`
2. `{method}_original_target_class_accuracy_heatmap.png`
3. `{method}_label_shift_target_confusion_matrix.png`
4. `{method}_label_shift_target_class_accuracy_heatmap.png`
5. `{method}_rare_class_target_confusion_matrix.png`
6. `{method}_rare_class_target_class_accuracy_heatmap.png`

### Summary File (1 text file per method)
- `{method}_concept_shift_summary.txt`
  - Original target accuracy
  - Label shift accuracy
  - Rare-class accuracy
  - Performance degradation metrics

---

## 🔄 Model Loading Strategy

Each script follows this fallback pattern:

```python
# Try final checkpoint first
checkpoint_path = {checkpoint_dir}/{method}_{source}_to_{target}_final.pth

# If not found, try latest epoch checkpoint
if not exists:
    checkpoint_path = {checkpoint_dir}/{method}_{source}_to_{target}_epoch_20.pth
```

---

## 🎯 Expected Checkpoint Structure

```
baseline_checkpoints/
  └── source_only_art_painting_to_cartoon_final.pth  ✅

DAN_checkpoints/
  └── dan_art_painting_to_cartoon_final.pth  ✅

DANN_checkpoints/
  └── dann_art_painting_to_photo_final.pth  ✅

CDAN_checkpoints/
  └── cdan_art_painting_to_photo_final.pth  ✅
```

---

## 🚀 How to Run

### Step 1: Run All Evaluations
```bash
cd task_1

# Evaluate Baseline
python concept_shift.py

# Evaluate DAN-MMD
python concept_shift_dan.py

# Evaluate DANN
python concept_shift_dann.py

# Evaluate CDAN
python concept_shift_cdan.py
```

### Step 2: Compare Results
```bash
python compare_concept_shift_results.py
```

This will generate a comprehensive comparison plot and summary table.

---

## 📈 Key Research Questions

After running all scripts, you can analyze:

### 1. **Accuracy vs Robustness Trade-off**
- Does better target domain accuracy come at the cost of robustness?
- Which method maintains performance under distribution shift?

### 2. **Domain Adaptation Effectiveness**
- Do DAN/DANN/CDAN improve over baseline on original distribution?
- Do they maintain this advantage under concept shifts?

### 3. **Negative Transfer Analysis**
- Do domain adaptation methods hurt rare-class performance?
- Evidence of negative transfer in complex methods (CDAN)?

### 4. **Class-Specific Patterns**
- Which classes are most affected by shifts?
- Are certain classes consistently misclassified across methods?

---

## ⚙️ Configuration

All scripts share the same concept shift parameters (configurable in `Config` class):

```python
# Label Shift Parameters
downsample_class_label_shift = 0      # Class index to downsample
downsample_ratio_label_shift = 0.1    # Keep 10% of samples
oversample_class_label_shift = 1      # Class index to oversample
oversample_factor_label_shift = 2     # Multiply by 2

# Rare-Class Parameters
rare_class_label = 2                  # Class index to make rare
rare_class_ratio = 0.05              # Keep 5% of samples
```

You can modify these to test different shift intensities.

---

## 🔍 Model Architecture Comparison

| Model | Components | Forward Pass Returns | Training Objective |
|-------|-----------|---------------------|-------------------|
| **Baseline** | FE + C | (output, features) | Classification only |
| **DAN-MMD** | FE + C | (output, features) | Classification + MMD |
| **DANN** | FE + C + DD + GRL | (class_out, domain_out, feat) | Class + Adversarial |
| **CDAN** | FE + C + MLM + DD + GRL | (class_out, domain_out, feat, pred) | Class + Conditional Adv |

**Legend:**
- FE = FeatureExtractor
- C = Classifier
- DD = DomainDiscriminator
- GRL = Gradient Reversal Layer
- MLM = MultiLinear Map

---

## ✅ Verification Checklist

Before running, verify:

- [ ] All checkpoint files exist in their respective directories
- [ ] PACS dataset is at the correct path: `pacs_data/pacs_data/`
- [ ] GPU available (or set `device = 'cpu'` in Config)
- [ ] Sufficient disk space for visualizations (~50-100 MB total)
- [ ] Python packages installed: torch, torchvision, sklearn, matplotlib, seaborn

---

## 📝 Notes

1. **No Training Occurs**: All scripts only load and evaluate pre-trained models
2. **Reproducibility**: Seed = 42 for all random operations
3. **Memory Efficiency**: Using `plt.close()` instead of `plt.show()` to prevent memory buildup
4. **Batch Size**: Default 128 (adjust if GPU memory limited)
5. **Different Target Domains**: 
   - Baseline & DAN: art_painting → cartoon
   - DANN & CDAN: art_painting → photo
   - This is intentional based on your checkpoint structure

---

## 🎓 Research Implications

This evaluation framework allows you to:

1. **Quantify** the accuracy-robustness trade-off in domain adaptation
2. **Identify** when domain adaptation helps vs. hurts performance
3. **Understand** negative transfer effects on rare classes
4. **Compare** simple (DAN) vs complex (CDAN) adaptation methods
5. **Demonstrate** the importance of evaluating beyond standard test accuracy

---

## 📧 Support

If you encounter issues:

1. Check checkpoint file names match expected patterns
2. Verify PACS dataset structure
3. Review error messages for missing files
4. Ensure all dependencies are installed
5. Check that target domain names match checkpoint files

---

## 🎉 Summary

**What We Built:**
- ✅ 4 evaluation scripts (1 per method)
- ✅ 1 comparison utility
- ✅ 2 documentation files
- ✅ Comprehensive concept shift testing framework

**What You Get:**
- 📊 24 visualization images (6 per method)
- 📄 4 summary text files
- 📈 1 comprehensive comparison plot
- 📋 Detailed performance analysis

**Next Steps:**
1. Run all 4 evaluation scripts
2. Run comparison script
3. Analyze results
4. Draw conclusions about domain adaptation robustness
5. Write research findings! 🚀



