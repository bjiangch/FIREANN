# Field-Induced Recursively embedded atom neural network for Charge Density and Response

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
- **QM9**: The Charge Density of QM9 molecules calculated with VASP are available at [QM9 Charge Densities and Energies Calculated with VASP](https://data.dtu.dk/articles/dataset/QM9_Charge_Densities_and_Energies_Calculated_with_VASP/16794500)


# Machine learning accelerated finite-field simulations for electrochemical interfaces

Electrochemical interfaces are of fundamental importance in electrocatalysis, batteries, and metal corrosion. Finite-field methods are one of most reliable approaches for modeling electrochemical interfaces in complete cells under realistic constant-potential conditions. However, previous finite-field studies have been limited to either expensive ab initio molecular dynamics or less accurate classical descriptions of electrodes and electrolytes. To overcome these limitations, we present a machine learning-based finite-field approach that combines two neural network models: one predicts atomic forces under applied electric fields, while the other describes the corresponding charge response. Both models are trained entirely on first-principles data without employing any classical approximations. As a proof-of-concept demonstration in a prototypical Au(100)/NaCl(aq) system, this approach not only dramatically accelerates fully first-principles finite-field simulations but also successfully extrapolates to cell potentials beyond the training range while accurately predicting key electrochemical properties. Interestingly, we reveal a turnover of orientation distributions of interfacial water molecules at the anode, arising from competing interactions between the positively charged anode and adsorbed Cl- ions with water molecules as the applied potential increases. This novel computational scheme shows great promise in efficient first-principles modelling of large-scale electrochemical interfaces under potential control.

## Data Description
The datasets used in this work are also provided in the *data* folder. 
- **Au100_NaCl_Interface**: The atomic forces datasets used for the FIREANN PES model training under cell potential of 0.0 V, 1.0 V, 2.0 V.
- **Au100_NaCl_Interface_response**: The CP2k input parameters for generating the charge response files. The configuration coordinates are consistent with the configurations within the **Au100_NaCl_Interface**.  We do not provide the charge response cube files directly here, as the corresponding charge response data are too large (~400 GB).
