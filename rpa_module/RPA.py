# python libs
from itertools import product
# Installed packages
import graphviz
import cantera as ct
import numpy as np
from scipy.interpolate import simpson
# Imported from this module
from utils import format_value

def compute_reaction_graph(flame, element):
    """
    Compute the reaction graph between species
    containing {element} in {flame}
    """
    # What does this do ?
    ix_in = 0
    ix_out = len(flame.grid)
    
    # Get the names of species containing element
    species = [sp.name for sp in gas.species() if gas.n_atoms(sp.name, element)]
    
    # Get the index of the species from the gas instance
    species_idx = [gas.species_index(sp) for sp in species]
    
    # Empty array for the reaction graph
    graph = np.zeros((len(species), len(species)))
        
    ne_s1s2 = e_transfer_matrix(flame.gas, element)
    
    ne_rate = np.zeros((len(species),len(species),len(flame.grid)))
    
    
    for irx in range(0,np.shape(ne_s1s2)[0]):
        idx_r = np.where(species_idx == ne_s1s2[irx,1])[0]
        idx_p = np.where(species_idx == ne_s1s2[irx,2])[0]
    
        ne_rate[idx_r,idx_p,:] += ne_s1s2[irx,3]*flame.net_rates_of_progress[ne_s1s2[irx,0].astype(int)]
    
    for idx_r in range(np.shape(ne_rate)[0]):
        for idx_p in range(np.shape(ne_rate)[1]):
            graph[idx_r,idx_p] = simpson(ne_rate[idx_r,idx_p,:], flame.grid, even="first")
    
    ## in/out fluxes
    Flux_net = np.zeros((len(species),))
    
    for i_sp, sp_name in enumerate(species):
        Flux_net[i_sp] = simpson(flame.net_production_rates[species_idx[i_sp]], 
                                 flame.grid,
                                 even="first")*gas.n_atoms(species_idx[i_sp],element)
    
    Flux_in = np.maximum(np.zeros(np.shape(Flux_net)), -Flux_net)
    Flux_out = np.reshape(np.append(np.maximum(np.zeros(np.shape(Flux_net)), Flux_net),[0,0]),(len(Flux_net)+2,1))

    
    residuals = abs((graph.sum(axis = 1) - graph.sum(axis = 0) + Flux_net)/Flux_in.sum())
    print("Maximum atom flux imbalance is {:.2e}%".format(residuals.max()*100))

    if residuals.max() > 1e-9:
        raise ValueError("Imbalance in atom flux detected")    
    

    graph = np.vstack((graph,Flux_in))
    graph = np.hstack((graph,np.zeros((np.shape(graph)[0],1))))

    graph = np.vstack((graph,np.zeros((1,np.shape(graph)[1]))))
    graph = np.hstack((graph,Flux_out))

    species.append('influx')
    species.append('outflux')
    
    
    # Put rates in new forward (positive) direction
    graph = graph - graph.T
    graph[graph < 0] = 0
    
    # add balance check
    
    # Normalize
    graph /= Flux_in.sum()
    #graph /= np.max(graph)
    # For plotting
    return species, graph


def visualize_reaction_graph(flame, element, 
                             color="black", max_width=10, tol=1e-1, write_fraction=False):
    """
    Display a reaction path analysis diagram
    for a Cantera flame.
    flame: Solved Cantera flame instance
    element: Element for which the flow is computed
    color: Color of the lines in the graph
    max_width: Width of the largest arrow in the graph
    tol: Cutoff for which reactions are included (in width)
    write_fraction: Whether the flow percentages are written
    """
    # Compute the graph matrix
    element_species, graph = compute_reaction_graph(flame, element)
    # Normalize by the max width (for line thicknesses)
    graph *= max_width
    # Empty graph instance
    f = graphviz.Digraph()

    # For every "reactant" specie
    for i, spr in enumerate(element_species):
        # For every "product" specie 
        for j, spp in enumerate(element_species):
            # width is graph value
            width = graph[i, j]

            if write_fraction:
                # Convert to % string
                label = f"  {format_value(width, max_width)}\n "
            else:
                label = None
                
            if width > tol:
                f.edge(spr, spp, label=label, penwidth=str(width), color=color)
    return f


