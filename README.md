
# DTBNet: Multimodal 3D Geometric Perception with Hierarchical Attention for Robust Drug-Target Interaction and Binding Affinity Prediction

## Overview  
Accurately predicting **drug-target interactions (DTIs)** and **binding affinity (DTA)** is crucial for drug discovery. However, existing models often lack **3D interaction modeling** and perform poorly in **cold-start scenarios** (new drugs/targets).  

**DTBNet** is a novel framework that:  
- Fuses **protein 3D geometry** (e.g., dihedral angles) with **multilevel drug representations** (fingerprints, molecular graphs, 3D conformations).  
- Uses **hierarchical attention** to enhance interaction modeling.  
- Advances **structure-based drug design**, aiding **virtual screening** and **target discovery**.  

## Dataset & Implementation  
This repository demonstrates DTBNet on the **Davis dataset**, including:  
- **Multi-feature extraction** for drugs and targets.  
- **Multi-network architecture** and **feature fusion**.  

### File Structure  
- `data/` – Contains dataset files.  
- `graph_model/` – Stores graph neural network models.  

## Dependencies  

| Package        | Version  |
|---------------|----------|
| numpy         | 1.26.4   |
| pandas        | 1.4.2    |
| rdkit         | 2022.03.2|
| scikit-learn  | 1.1.1    |
| scipy         | 1.12.0   |
| torch-cluster | 1.6.3    |
| torch-scatter | 2.1.2    |

## Quick Start  
 **Install dependencies** (Python 3.9.21 recommended):  
   ```bash
   python main.py
```
## Citation  
If you use this code in your research, please cite our paper:  

**DTBNet: A Unified Deep Learning Framework with Multimodal Geometric Awareness and Attention Enhancement for Drug-Target Interaction and Binding Affinity Prediction**  

Happy researching! 🚀  
