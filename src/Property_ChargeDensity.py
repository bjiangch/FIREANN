import numpy as np
import torch 
import opt_einsum as oe

#============================calculate the energy===================================
class Property(torch.nn.Module):
    def __init__(self,density,nnmod):
        super(Property,self).__init__()
        self.density=density
        self.nnmod=nnmod

    def forward(self,cart,ef,numatoms,species,atom_index,shifts,create_graph=None):
        dummy_index = (species==torch.max(species)).nonzero()
        species=species.view(-1)
        density = self.density(cart,ef,numatoms,species,atom_index,shifts)
        output=self.nnmod(density,species).view(numatoms.shape[0],-1)
        varene = output[dummy_index[:, 0], dummy_index[:, 1]].view(numatoms.shape[0], -1)
        return (varene,)

