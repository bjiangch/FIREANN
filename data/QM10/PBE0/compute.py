import os
from ase.io.vasp import read_vasp
from ase.calculators.vasp import Vasp

numatoms = 1000
atoms_list = list()
for i in range(numatoms):
    filepath = "/data/home/fcqustc/density/QM10/POSCAR/{}.POSCAR".format(i)
    atoms = read_vasp(filepath)
    atoms_list.append(atoms)

calc = Vasp(
    command="mpirun -n $SLURM_NPROCS vasp_std > log",
    xc='pbe0', 
    istart=0, 
    algo='Normal', 
    icharg=2, 
    nelm=180, 
    ispin=1, 
    nelmdl=6, 
    isym=0, 
    lcorr=True, 
    potim=0.1, 
    nelmin=5, 
    kpts=[1,1,1], 
    ismear=0, 
    ediff=0.1E-05, 
    sigma=0.1, 
    nsw=0, 
    ldiag=True, 
    lreal='Auto', 
    lwave=False, 
    lcharg=True, 
    encut=400
)
for i in range(100, 200):
    atoms = atoms_list[i]
    atoms.calc = calc
    atoms.get_potential_energy()
    
    os.popen('cp CHGCAR ../{}.CHGCAR'.format(i))  
    os.popen('cp log ../{}.log'.format(i)) 
