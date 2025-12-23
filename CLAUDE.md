# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepIMAGER is a supervised deep-learning model for predicting cell-specific gene regulatory networks (GRNs) from scRNA-seq and ChIP-seq data. It uses ResNet-based CNNs to learn patterns in gene expression that indicate regulatory relationships between transcription factors (TFs) and target genes.

## Execution Commands

The project uses Python 3.8 with command-line scripts. The pipeline has three main steps:

### Step 1: Generate Input Representations
```bash
python3.8 generate_input_realdata.py \
  -out_dir OUTPUT_DIR \
  -expr_file EXPRESSION_DATA \
  -pairs_for_predict_file GENE_PAIRS \
  -geneName_map_file GENE_NAME_MAP \
  -flag_load_from_h5 True \
  -TF_divide_pos_file TF_PARTITION_FILE \
  -TF_num NUMBER_OF_TFS \
  -top_or_random top_cov
```

### Step 2: Cross-Validation Training
```bash
python3.8 DeepIMAGER.py \
  -num_batches NUMBER_OF_TFS \
  -data_path PATH_TO_REPRESENTATIONS \
  -output_dir OUTPUT_DIR \
  -cross_validation_fold_divide_file cross_validation_fold_divide.txt
```

### Step 3: Model Prediction
```bash
python3.8 DeepIMAGER.py \
  -to_predict True \
  -num_batches NUMBER_OF_TFS \
  -data_path PATH_TO_DATA \
  -output_dir OUTPUT_DIR \
  -weight_path TRAINED_MODEL.h5

# Alternative prediction script:
python3.8 predict_use_model.py \
  -num_batches NUMBER_OF_TFS \
  -data_path PATH_TO_DATA \
  -output_dir OUTPUT_DIR \
  -weight_path TRAINED_MODEL.h5
```

## Architecture

### Core Scripts
- **DeepIMAGER.py** - Main training/inference engine with cross-validation
- **generate_input_realdata.py** - Converts gene expression data to image representations
- **predict_use_model.py** - Standalone inference wrapper

### Model Architecture (model/)
- **resnet50.py** - ResNet50 with bottleneck blocks
- **resnet_18.py** - ResNet18 with residual blocks
- Both support a dual-branch architecture: single-image branch (primary TF) + multi-image branch (neighboring genes), merged via concatenation and dense layers

### Key Classes
- `direct_model1_squarematrix` (DeepIMAGER.py) - Training class with 3-fold CV
- `use_model_predict` (predict_use_model.py) - Inference class
- `RepresentationTest2` (generate_input_realdata.py) - Data loading and image generation

### Data Flow
1. Gene expression (H5/CSV) + gene pair labels → Image representations
2. Representations split by TF → Batch .npy files
3. Cross-validation training → Model evaluation (accuracy, AUC, ROC)

### Pre-trained Models (model.h5/)
- `dendritic.h5` - Dendritic cells
- `mHSC_L.h5` - Mouse hematopoietic stem cells

### Test Datasets (data_evaluation/)
- Bone marrow, dendritic cells, hESC, mESC, mHSC variants

## Configuration

- **cross_validation_fold_divide.txt** - 3-fold CV splits (comma-separated TF indices per line)
- GPU selection: Set `CUDA_VISIBLE_DEVICES` in DeepIMAGER.py line 4

## Dependencies

- Python 3.8
- TensorFlow 2.4.1 / TensorFlow-GPU 2.4.1
- NumPy 1.23.5
- scikit-learn 1.2.2
- pandas, matplotlib, scipy
- numba 0.51.2
