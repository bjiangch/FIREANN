# Field-Induced Recursively embedded atom neural network for Charge Density and Response
## Introduction
Electron density is a fundamental quantity, which can in principle determine all ground state electronic properties of a given system. Although machine learning (ML) models for electron density based on either an atom-centered basis or a real-space grid have been proposed, the demand for the number of high-order basis functions or grid points is enormous. In this work, we propose an efficient grid-point sampling strategy that combines a targeted sampling favoring large density and a screening of grid points associated with linearly independent atomic features. This new sampling strategy is integrated with a field-induced recursively embedded atom neural network model to develop a real-space grid-based ML model for electron density and its response to an electric field. This approach is applied to a QM9 molecular dataset, a H2O/Pt(111) interfacial system, an Au(100) electrode, and an Au nanoparticle under an electric field. The number of training points is found much smaller than previous models, when yielding comparably accurate predictions for the electron density of the entire grid. The resultant machine learned electron density model enables us to properly partition partial charge onto each atom and analyze the charge variation upon proton transfer in the H2O/Pt(111) system. The machined learned electronic response model allows us to predict charge transfer and the electrostatic potential change induced by an electric field applying to an Au(100) electrode or an Au nanoparticle. 

## Requirements
1. PyTorch 2.0.0
2. LibTorch 2.0.0
3. cmake 3.1.0
4. opt_einsum 3.2.0
5. ase 3.22.1

## Data Description
The charge density and response datasets used in this work are provided in the *data* folder. 
- **Pt_H2O**: The VASP input parameters and coordinates files necessary for generating the charge density for the H2O/Pt(111) dataset.
- **Au_slab**: The CP2K inpput parameters and coordinates files necessary for generating the charge response for the Au slab dataset under an applied field Ez=-1V/Angstrom.
- **QM10**: The VASP input parameters and coordinates files necessary for generating the charge density for the independent test dataset with both PBE and PBE0 functional.
- **Au_nanoparticle**: The CP2K inpput parameters and coordinates files necessary for generating the charge response for the Au nanoparticle dataset under an applied field Ez=1V/Angstrom.
- 
