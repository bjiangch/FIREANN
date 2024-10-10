import os
import math
import sys
import datetime
import numpy as np
import torch
torch.set_printoptions(profile="full")
# from tqdm import tqdm
from gpu_sel import *
from calculate import *
import getneigh as getneigh
from write_format import *
import re
from matplotlib import pyplot as plt
from ase.atom import Atom
from ase.io.cube import read_cube, write_cube
import ase
from ase.calculators.vasp import VaspChargeDensity
# from EADestcriptor import EADestriptor
#====================================Prop_list================================================
# used for select a unoccupied GPU
gpu_sel()
# gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype
torch_dtype=torch.double
calculator=Calculator(device,torch_dtype)
cutoff = calculator.cutoff

# same as the atomtype in the file input_density
# atomtype=["Au","X"]
atomtype = ["C","H","O","N","F","X"]
# save the lattic parameters

filepath = str(sys.argv[1])
with open(filepath, "r") as fd:
    if filepath.endswith("CHGCAR"):
        vasp_charge = VaspChargeDensity(filename=filepath)
        density = vasp_charge.chg[-1]
        atoms = vasp_charge.atoms[-1]
        cell = np.array(atoms.get_cell())
        origin = np.zeros(3)    # 假定晶格原点为(0,0,0)
    elif filepath.endswith("cube"):
        cube = read_cube(fd)
        density = cube["data"] 
        atoms = cube["atoms"] 
        cell = np.array(atoms.get_cell())  
        origin = np.zeros(3)
    
    ngridpts = np.array(density.shape)
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / density.shape[0],
        np.arange(ngridpts[1]) / density.shape[1],
        np.arange(ngridpts[2]) / density.shape[2],
        indexing='ij'
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, cell)
    grid_pos = grid_pos + origin
    grid_size = np.prod(np.diagonal(cell) / np.array(density.shape))
    grid_num = np.prod(np.array(density.shape))
print(density.shape, flush=True)

num=0
sort_shape = np.sort(np.array(density.shape))
batchsize=sort_shape[2]
maxneigh = batchsize *100
nbatch= math.ceil(np.prod(np.array(density.shape))/batchsize)
# nbatch = 1

# 输入变量构造
# species = list()
atoms_species = [atomtype.index(atom.symbol) for atom in atoms]
atoms_species.extend([atomtype.index("X") for _ in range(batchsize)])
species = [atoms_species for _ in range(nbatch)]
# atoms_pbc = atoms.get_pbc()
atoms_pbc = [True, True, True]
pbc = [atoms_pbc for _ in range(nbatch)]
atoms_cart = np.array([atoms.get_positions() for _ in range(nbatch)])
cart = np.concatenate((atoms_cart, grid_pos.reshape(-1, batchsize, 3)), axis=1)
atoms_cell = atoms.get_cell()
cell = [atoms_cell for _ in range(nbatch)]
# cell=list()
atoms_ef = [0.0, 0.0, 1.0]
ef = [atoms_ef for _ in range(nbatch)]
abprop = density.reshape(-1)

# 计算输出结果
num=0
totnum=grid_num
batchsize=1
numatom=len(atoms) + 1
charge_pred = []
for m in range(nbatch):
    print("batch:", m, flush=True)
    if num >= totnum:
        break
    neigh_list=torch.empty(2,0).to(torch.long).to(device)
    shifts=torch.empty(0,3).to(torch_dtype).to(device)
    index_cell=torch.empty(0).to(device).to(torch.long)
    num_up=min(num+batchsize,totnum)
    bcart=cart[num:num_up]
    bcart=torch.from_numpy(bcart).to(device).to(torch_dtype)
    # print("bcart", bcart)  
    bpbc=pbc[num:num_up]
    bpbc=torch.from_numpy(np.array(bpbc)).to(device).to(torch_dtype).view(-1,3)
    bcell=cell[num:num_up]
    bcell=torch.from_numpy(np.array(bcell)).to(device).to(torch_dtype)
    bspecies=species[num:num_up]
    bspecies=torch.from_numpy(np.array(bspecies)).to(device)
    dummy_mask = (bspecies == atomtype.index("X"))
    bef=ef[num:num_up]
    bef=torch.from_numpy(np.array(bef)).to(device).to(torch_dtype).view(-1,3)
    c_cart = []
    for i in range(num_up-num):
        getneigh.init_neigh(cutoff,cutoff/2.0,bcell[i].cpu().T)
        coor,neighlist,shiftimage,scutnum=getneigh.get_neigh(bcart[i].T.cpu(), dummy_mask[i].cpu(), maxneigh)
        # print(neighlist)
        index_cell=torch.cat((index_cell,torch.ones(scutnum).to(device).to(torch.long)*i),0)
        c_cart.append(coor.T)
        tmp_neigh=neighlist+i*bcart.shape[1]
        neigh_list=torch.cat((neigh_list,torch.from_numpy(tmp_neigh)[:,:scutnum].to(device).to(torch.long)),1)
        shifts=torch.cat((shifts,torch.from_numpy(shiftimage).T[:scutnum].to(device).to(torch_dtype)),0)
        num+=1

    bef.requires_grad=False
    bcart.requires_grad=False
    bcart=torch.from_numpy(np.array(c_cart)).contiguous().to(device).to(torch_dtype)
    # bcell=torch.from_numpy(bcell).to(device).to(torch_dtype)
    disp_cell=torch.zeros_like(bcell)
    # print(index_cell, flush=True)
    # print(disp_cell.isnan().any(), bcell.isnan().any(),bcart.isnan().any(),bef.isnan().any(),index_cell.isnan().any(),neigh_list.isnan().any(),shifts.isnan().any(),bspecies.isnan().any())
    # print(index_cell)    
    varene=calculator.get_charge_response(bcell,disp_cell,bcart,bef,index_cell,neigh_list,shifts,bspecies.view(-1))
    init_num=num-bef.shape[0]
    charge_pred.append(varene.cpu().numpy().reshape(-1))
    # print("charge_pred", charge_pred)
    
charge_pred = np.concatenate(tuple(charge_pred), axis=0)
np.savetxt("./refer_charge_density.txt", abprop)
np.savetxt("./pred_charge_density.txt", charge_pred)
rmse = np.sqrt(np.mean(np.square(charge_pred-abprop)))
mae = np.mean(np.abs(charge_pred-abprop))
error_max = np.max(charge_pred-abprop)
error_min = np.min(charge_pred-abprop)
print("error_max", error_max, "error_min", error_min, flush=True)
print("rmse", rmse, "mae", mae, flush=True)

if filepath.endswith("CHGCAR"):
    pred_charge = vasp_charge
    pred_charge.chg[-1] = charge_pred.reshape(density.shape)
    pred_charge.write("pred_CHGCAR")
elif filepath.endswith("cube"):
    with open("pred.cube", "w") as fd:
        write_cube(fd, atoms, charge_pred.reshape(density.shape))

