# HybridCA-Net: Multimodal Fusion Framework  

## Overview  
HybridCA-Net is an innovative multimodal fusion solution designed to tackle four critical challenges in multimodal data integration: **modality heterogeneity**, **adaptive feature weighting**, **missing modalities**, and **effective feature integration**. Through specially designed modules, it achieves significant performance improvements in medical image classification and other tasks.  


## Core Technologies and Advantages  
### 1. **Modality Heterogeneity Handling**  
- **Solution**: The **Self-Supervised Consistency (SSC) Module** aligns feature representations across modalities using mean squared error (MSE), balancing the extraction of common pathological features with the preservation of modality-specific information.  
- **Performance**: In the MCI vs. CN classification task, accuracy improved from **66.30% to 90.86%**.  

### 2. **Adaptive Feature Weighting**  
- **Solution**: The **Cross-Modal Fusion (CMF) Module** employs cross-attention mechanisms to dynamically emphasize relevant features and suppress irrelevant information.  
- **Performance**: Ablation studies show this approach outperforms traditional addition/multiplication fusion methods in all classification tasks.  

### 3. **Missing Modalities Recovery**  
- **Solution**: The **Recovery Module** reconstructs features of missing modalities using high-level features from available modalities, enhancing practicality in clinical scenarios with data limitations.  
- **Performance**: In the AD vs. CN task with only fMRI data, accuracy improved from **68.46% to 93.23%**.  

### 4. **Deep Feature Integration**  
- **Solution**: A custom architecture fuses structural information from sMRI and functional connectivity data from fMRI, preserving the unique value of each modality.  
- **Performance**: Compared to single-modality results (fMRI: 58.81%, sMRI: 88.65%), the fusion approach achieved **90.86% accuracy** in the MCI vs. CN task, demonstrating superior complementary information utilization.  


## Code Release Notice  
The complete codebase (including module implementations, training scripts, and test cases) will be released **shortly**! Stay tuned to this repository for updates.  


## Usage and Citation  
- **Testing**: Free to download and use for algorithm validation once released.  
- **Academic Citation**: If used in research papers, please contact us in advance for citation guidelines.  
- **Contact**: For inquiries, monitor this repository for updated contact information.  

**Updated Date**: May 28, 2025
