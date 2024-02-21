import graphviz
import cantera as ct
import numpy as np
from itertools import product

def format_value(value, line_width):
    """
    Format normalized reaction rate value
    """
    value *= (100/line_width)
    if value > 1.0:
        label = f"{value:.2f}%"
    elif value > 0.1:
        label = f"{value:.2f}%"
    elif value > 0.01:
        label = f"{value:.3f}%"
    else:
        label = f"{value:.0e}%"
    return label


def compute_reaction_graph(flame, element):
    """
    Compute the reaction graph between species
    containing {element} in {flame}
    """
    ix_in = 0
    ix_out = len(flame.grid)
    
    # Get the relevant species names (special case for H and HE)
    # Before 2024.02.20 : species = [sp.name for sp in ct_species if (element in sp.name) and (sp.name != 'HE')]
    species = [sp.name for sp in gas.species() if gas.n_atoms(sp.name,element)]
    
    species_idx = [gas.species_index(sp) for sp in species]
    
    graph = np.zeros((len(species), len(species)))
        
    ne_s1s2 = e_transfer_matrix(flame.gas, element)
    
    ne_rate = np.zeros((len(species),len(species),len(flame.grid)))
    
    
    for irx in range(0,np.shape(ne_s1s2)[0]):
        idx_r = np.where(species_idx == ne_s1s2[irx,1])[0]
        idx_p = np.where(species_idx == ne_s1s2[irx,2])[0]
    
        ne_rate[idx_r,idx_p,:] += ne_s1s2[irx,3]*flame.net_rates_of_progress[ne_s1s2[irx,0].astype(int)]
    
    for idx_r in range(np.shape(ne_rate)[0]):
        for idx_p in range(np.shape(ne_rate)[1]):
            graph[idx_r,idx_p] = Simpson_13_comp(ne_rate[idx_r,idx_p,:], flame.grid)
            #graph[idx_r,idx_p] = np.trapz(ne_rate[idx_r,idx_p,:], flame.grid)
    
    


    ## in/out fluxes

    Flux_net = np.zeros((len(species),))
    
    for i_sp, sp_name in enumerate(species):
        Flux_net[i_sp] = Simpson_13_comp(flame.net_production_rates[species_idx[i_sp]], flame.grid)*gas.n_atoms(species_idx[i_sp],element)
    
    Flux_in = np.maximum(np.zeros(np.shape(Flux_net)), -Flux_net)
    Flux_out = np.reshape(np.append(np.maximum(np.zeros(np.shape(Flux_net)), Flux_net),[0,0]),(len(Flux_net)+2,1))
    

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
    graph /= np.max(graph)
    # For plotting
    return species, graph
    
    
    
    
    """
    # Integrate the reaction rates
    int_NRR = {}
    for i, rate  in enumerate(flame.net_rates_of_progress):
        int_NRR[f'R{i}'] = np.trapz(rate, flame.grid)

    # Construct the graph
    ct_species = flame.gas.species()
    # Get the relevant species names (special case for H and HE)
    # Before 2024.02.20 : species = [sp.name for sp in ct_species if (element in sp.name) and (sp.name != 'HE')]
    species = [sp.name for sp in ct_species if gas.n_atoms(sp.name,element)]
    
    # Reindex with the species of interest
    species_indexes = {sp:i for i, sp in enumerate(species)}
    # Empty graph
    graph = np.zeros((len(species), len(species)))
    # All reactions
    ct_reactions = flame.gas.reactions()
    # For each reaction
    for i, rkey in enumerate(int_NRR):
        reaction = ct_reactions[i]
        # For each reactant
        for ri in reaction.reactants:
            # If the reactant contains element
            if ri in species:
                # For each product
                for rj in reaction.products:
                    # If the product contains element
                    if rj in species:
                        # Reactant species index
                        idx_r = species_indexes[ri]
                        # Product species index
                        idx_p = species_indexes[rj]
                        # Integrated net reaction rate
                        rk = int_NRR[rkey]
                        # Cantera index of reactant
                        ct_index = flame.gas.species_index(ri)
                        # Number of atoms of element
                        # PV 2024.02.20 - Implement n_e here
                        ni = ct_species[ct_index].composition[element]
                        # Reactant has a flow of n_ele * net_rate
                        # Towards the product
                        graph[idx_r, idx_p] += rk * ni
    
    # Remove reverse direction
    # PV 2024.02.20: to double check - C'est bon!
    graph = graph - graph.T
    graph[graph < 0] = 0
    # Normalize
    graph /= np.max(graph)
    # For plotting
    return species, np.around(graph, 3)"""


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




def e_transfer_matrix(gas, element):
    """
    Determine the number of atoms of element "e" transfered from species 1 to species 2
    through the reaction i
    
    Output is n_i(e,sp1,sp2). See:
    Joseph F. Grcar , Marcus S. Day & John B. Bell (2006) A taxonomy of integral reaction path analysis,
    Combustion Theory and Modelling, 10:4, 559-579, DOI: 10.1080/13647830600551917
    """
    ne_s1s2 = np.empty((0, 4))
    
    for irx, reaction in enumerate(gas.reactions()):
        if np.any([gas.n_atoms(reac,element) for reac in reaction.reactants]):
    
            # Generate all possible and valid combinations of atom transfer
            combinations, dlt_sp_cmp, dlt_W = generate_valid_combinations(reaction,element)
            
            # minimize change in species composition (element count)
            if combinations.shape[0]-1:
                combinations = combinations[(np.multiply(combinations,dlt_sp_cmp).sum(axis = 1).min() == np.multiply(combinations,dlt_sp_cmp).sum(axis = 1))]
            
            # favour carbon atom over oxygen atom transfer
            if combinations.shape[0]-1 and  'O' in gas.element_names and 'C' in gas.element_names and (element == 'O' or element == 'C'):
                comb_fltr = np.zeros((np.shape(combinations)[0],1), dtype=bool)
            
                element_2 = 'C' if element == 'O' else 'O'
                
                combinations_2  = generate_valid_combinations(reaction,element_2)[0]
            
                for i in range(np.shape(combinations)[0]):
                    for j in range(np.shape(combinations_2)[0]):
                        if element == 'O':
                            if ((combinations_2[j] >= combinations[i]) & (combinations[i] > 0)).any():
                                comb_fltr[i] = True        
                        
                        else:
                            if ((combinations[i] >= combinations_2[j]) & (combinations_2[j] > 0)).any():
                                comb_fltr[i] = True        
                    
                combinations = combinations[comb_fltr]    
            
            # minimize change in molar mass
            if combinations.shape[0]-1:
                combinations = combinations[(np.multiply(combinations,dlt_W).sum(axis = 1).min() == np.multiply(combinations,dlt_W).sum(axis = 1))]
    
            # minimize split        
            if combinations.shape[0]-1:
                combinations = combinations[(combinations > 0).sum(axis = 1).min() == (combinations > 0).sum(axis = 1)]
    
         
            if combinations.shape[0]-1:
            #****Output warning
                combinations = combinations[0]
            
            
            ic = 0
            for reac in reaction.reactants:
                for prod in reaction.products: #for k=1:size(Prod_RR_n,2)
                    if combinations[0][ic]:
                        ne_s1s2 = np.vstack((ne_s1s2,[irx, gas.species_index(reac), gas.species_index(prod), combinations[0][ic]]))
                    ic += 1
    return ne_s1s2
    
def generate_valid_combinations(reaction,element):

    fct_lvls    = []
    fct_lvls_v  = []
    dlt_sp_cmp  = []
    dlt_W       = []
    dlt_bd      = []
    
    n_e = 0
    for reac in reaction.reactants: #for j=1:size(Reac_RR_n,2)
        n_e +=  gas.n_atoms(reac,element) * reaction.reactants[reac] 
        for prod in reaction.products: #for k=1:size(Prod_RR_n,2)
            
            fct_lvls_v = np.append(fct_lvls_v, [reaction.reactants[reac],reaction.products[prod]][np.argmin(([gas.n_atoms(reac,element) * reaction.reactants[reac],gas.n_atoms(prod,element) * reaction.products[prod]]))])
            fct_lvls   = np.append(fct_lvls,   [gas.n_atoms(reac,element),gas.n_atoms(prod,element)][np.argmin(([gas.n_atoms(reac,element) * reaction.reactants[reac],gas.n_atoms(prod,element) * reaction.products[prod]]))])
            
            dlt_sp_cmp = np.append(dlt_sp_cmp, np.abs([gas.n_atoms(reac,elem)-gas.n_atoms(prod,elem) for elem in gas.element_names]).sum())
            dlt_W      = np.append(dlt_W     , abs(gas.molecular_weights[gas.species_index(reac)]-gas.molecular_weights[gas.species_index(prod)]))
            dlt_bd     = np.append(dlt_bd    , abs(gas.molecular_weights[gas.species_index(reac)]-gas.molecular_weights[gas.species_index(prod)]))
    
    
    fct_lvls = fct_lvls.astype(np.int64)

    # Generate all possible combinations
    combinations = np.multiply(np.array(list(product(*[range(level + 1) for level in fct_lvls]))),fct_lvls_v)

    # check that the total number of exchanged atom matches the total number of atoms in the reactants (and hence products)
    if combinations.shape[0]-1:
        combinations = combinations[combinations.sum(axis = 1) == n_e]

    # Check for:
    #           1) a given reactant that the total number of atoms given matches its elemental composition
    #           2) a given product that the total number of atoms received matches its elemental composition
    
    if combinations.shape[0]-1:
        combinations = combinations[[np.all([abs(np.sum(np.reshape(combinations[iC],(len(reaction.reactants),len(reaction.products))),axis = 1) - [gas.n_atoms(reac,element) * reaction.reactants[reac] for reac in reaction.reactants]).sum() < RES,abs(np.sum(np.reshape(combinations[iC],(len(reaction.reactants),len(reaction.products))),axis = 0) - [gas.n_atoms(prod,element) * reaction.products[prod] for prod in reaction.products]).sum() < RES]) for iC in range(np.shape(combinations)[0])]]
    
    return combinations, dlt_sp_cmp, dlt_W    
    
    

def Simpson_13(fx,x):
    """
    Simpson's 1/3 integration for a 3-point irregular grid

    Parameters:
    - x: grid points
    - fx: function evaluated at each grid point

    Returns:
    - Integral estimate using Simpson's 1/3 rule.


    Reference: P. Versailles, CH formation in Premixed Flames of C1–C4 Alkanes: Assessment of Current Chemical
               Modelling Capability Against Experiments, McGill University, 2017 Ph.D. thesis. http://digitool.library.mcgill.ca/R/
    """
    if len(x) != 3 or len(fx) != 3:
        raise ValueError("x and fx must be 1x3 or 3x1 matrices")
    
    
    fd_x0x1 = (fx[1] - fx[0]) / (x[1] - x[0])
    fd_x1x2 = (fx[2] - fx[1]) / (x[2] - x[1])
    fd_x0x1x2 = (fd_x1x2 - fd_x0x1) / (x[2] - x[0])
    
    A = fx[0] - fd_x0x1 * x[0] + fd_x0x1x2 * x[0] * x[1]
    B = fd_x0x1 - (x[0] + x[1]) * fd_x0x1x2
    C = fd_x0x1x2
    
    Int = A * (x[2] - x[0]) + B * (x[2] ** 2 - x[0] ** 2) / 2 + C * (x[2] ** 3 - x[0] ** 3) / 3
    
    return Int


def Simpson_13_comp(fx, x):
    """
    Composite Simpson's 1/3 integration on an irregular grid of arbitrary length

    Parameters:
    - x: grid points
    - fx: function evaluated at each grid point

    Returns:
    - Integral estimate using composite Simpson's 1/3 rule.


    Reference: P. Versailles, CH formation in Premixed Flames of C1–C4 Alkanes: Assessment of Current Chemical
               Modelling Capability Against Experiments, McGill University, 2017 Ph.D. thesis. http://digitool.library.mcgill.ca/R/
    """

    
    if len(x) != len(fx):
        raise ValueError("x and fx must be of the same size")
    
    # Remove duplicate x-values
    unq_x = []
    unq_fx = []
    
    for i, val in enumerate(x):
        if val not in unq_x:
            unq_x.append(val)
            unq_fx.append(fx[i])

    x = unq_x
    fx = unq_fx
 
      
    if len(x) <= 3 and len(fx) <= 3:
        raise ValueError("x and fx must be 1x>3 or >3x1 matrices")
    
    nb_gp = len(x)
    nb_Intint = (nb_gp - 1) // 2

    Int = 0
    
    # Apply Simpson's 1/3 rule to each interval
    for i in range(nb_Intint):
        Int += Simpson_13(fx[2*i:2*i+3], x[2*i:2*i+3])
        
    
    # Handle the last interval if the number of 3-gridpoint intervals is not a round number
    if 2 * nb_Intint + 1 < nb_gp:
        Int += (fx[-1] + fx[-2]) / 2 * (x[-1] - x[-2])
    
    return Int