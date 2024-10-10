import torch
# from get_neigh import *
from torch.autograd.functional import jacobian,hessian

class Calculator():
    def __init__(self,device,torch_dtype):
        #load the serilizable model
        pes=torch.jit.load("PES.pt")
        # FLOAT: torch.float32; DOUBLE:torch.double for using float/double in inference
        pes.to(device).to(torch_dtype)
        # set the eval mode
        pes.eval()
        print("initstd", pes.nnmod.initstd, flush=True)
        print("initpot", pes.nnmod.initpot, flush=True)
        self.cutoff=pes.cutoff
        self.pes=torch.jit.optimize_for_inference(pes)

    def get_dipole(self,disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy)
        dipole=torch.autograd.grad(varene,ef,create_graph=create_graph)[0]
        return dipole,

    def get_force(self,disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy)
        dipole=torch.autograd.grad(varene,cart,create_graph=create_graph)[0]
        return force

    def get_ene_dipole(self,disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy.view(-1,cart.shape[1]),dim=1)
        dipole=torch.autograd.grad(torch.sum(varene),ef,create_graph=create_graph)[0]
        return varene,dipole,
         
    def get_ene(self,disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy.view(-1,cart.shape[1]),dim=1)
        return varene,
    
    def get_pol(self,disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy)
        dipole=torch.sum(torch.autograd.grad(varene,ef,create_graph=True)[0],dim=0)
        pol=torch.cat([torch.autograd.grad(idipole,ef,create_graph=True)[0].view(-1,1,3) for idipole in dipole],dim=1)
        return pol,

    def get_stresstensor(self,disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy.view(-1,cart.shape[1]),dim=1)
        stress=torch.autograd.grad(torch.sum(varene),disp_cell,create_graph=create_graph)[0]/(cell[:,0,0]*cell[:,1,1]*cell[:,2,2])
        return stress

    def get_ene_stress(self,disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        atomic_energy=self.pes(disp_cell,cell,cart,ef,index_cell,neigh_list,shifts,species)
        varene=torch.sum(atomic_energy.view(-1,cart.shape[1]),dim=1)
        stress=torch.autograd.grad(torch.sum(varene),disp_cell,create_graph=create_graph)[0]/(cell[:,0,0]*cell[:,1,1]*cell[:,2,2])
        return varene,stress

    def get_charge_response(self,cell,disp_cell,cart,ef,index_cell,neigh_list,shifts,species,create_graph=False):
        dummy_index = (species==torch.max(species)).nonzero()
        atomic_charge=self.pes(cell,disp_cell,cart,ef,index_cell,neigh_list,shifts,species)
        # print(atomic_charge, flush=True)
        varene = atomic_charge[dummy_index].view(-1)
        # print(varene)
        return varene
        
#===============bug need to be fixed in jit for vectorize=======================================     
    #def get_pol(self,cart,ef,neigh_list,shifts,species,create_graph=False):
    #    pol=jacobian(lambda x: self.get_dipole_for_pol(cart,x,neigh_list,shifts,species),ef,\
    #    create_graph=create_graph,vectorize=True)[0].view(3,-1,3).permute(1,0,2)
    #    return (pol,)
    #
    #def get_dipole_for_pol(self,cart,ef,neigh_list,shifts,species):
    #    atomic_energy=self.pes(cart,ef,neigh_list,shifts,species)
    #    varene=torch.sum(atomic_energy)
    #    dipole=torch.autograd.grad(varene,ef,create_graph=True)[0]
    #    return torch.sum(dipole,dim=0),
#=======================================================================================
