from itertools import product
import cantera as ct
import numpy as np
import warnings

def e_transfer_matrix(gas, element):
    """
    Determine the number of atoms of element "e" transfered 
    from species 1 to species 2 through the reaction i

    gas: cantera.Solution instance containing the reactions
         for which the reaction pathway analysis is
         computed
    element: string of the element for which the flux is 
             computed
    
    Output is n_i(e,sp1,sp2). 

    See:
    Joseph F. Grcar , Marcus S. Day & John B. Bell (2006) 
    A taxonomy of integral reaction path analysis,
    Combustion Theory and Modelling, 10:4, 559-579, 
    DOI: 10.1080/13647830600551917
    """
    ne_s1s2 = np.empty((0, 4))
    
    for irx, reaction in enumerate(gas.reactions()):
        if np.any([gas.n_atoms(reac,element) for reac in reaction.reactants]):
    
            # Generate all possible and valid combinations of atom transfer
            combinations, dlt_sp_cmp, dlt_W = generate_valid_combinations(reaction,element, gas)
            
            # minimize change in species composition (element count)
            if combinations.shape[0]-1:
                combinations = combinations[(np.multiply(combinations,dlt_sp_cmp).sum(axis = 1).min() == np.multiply(combinations,dlt_sp_cmp).sum(axis = 1))]
            
            # favour carbon atom over oxygen atom transfer
            if combinations.shape[0]-1 and  'O' in gas.element_names and 'C' in gas.element_names and (element == 'O' or element == 'C'):
                if np.any([gas.n_atoms(reac,'O') for reac in reaction.reactants]) and np.any([gas.n_atoms(reac,'C') for reac in reaction.reactants]):
                    comb_fltr = np.zeros((np.shape(combinations)[0]), dtype=bool)
                
                    element_2 = 'C' if element == 'O' else 'O'
                    
                    combinations_2  = generate_valid_combinations(reaction, element_2, gas)[0]
                
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
                comb_fltr = np.zeros((np.shape(combinations)[0]), dtype=bool)
                comb_fltr[0] = True
                combinations = combinations[comb_fltr]
                warnings.warn("Script could not find a unique distribution of " + element + " for reaction " + str(reaction) + ".\n First distribution selected : " + str(combinations))
            
            
            ic = 0
            for reac in reaction.reactants:
                for prod in reaction.products: #for k=1:size(Prod_RR_n,2)
                    if combinations[0][ic]:
                        ne_s1s2 = np.vstack((ne_s1s2,[irx, gas.species_index(reac), gas.species_index(prod), combinations[0][ic]]))
                    ic += 1
    return ne_s1s2


def generate_valid_combinations(reaction, element, gas):

    fct_lvls    = []
    fct_lvls_v  = []
    dlt_sp_cmp  = []
    dlt_W       = []
    dlt_bd      = []
    RES = 1e-9
    n_e = 0
    # For each reactant
    for reac in reaction.reactants: #for j=1:size(Reac_RR_n,2)
        # Increment the number of atoms in the reactants
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

    #return combinations, n_e
    # check that the total number of exchanged atom matches the total number of atoms in the reactants (and hence products)
    if combinations.shape[0]-1:
        combinations = combinations[combinations.sum(axis = 1) == n_e]

    # Check for:
    #           1) a given reactant that the total number of atoms given matches its elemental composition
    #           2) a given product that the total number of atoms received matches its elemental composition
    
    if combinations.shape[0]-1:
        combinations = combinations[[np.all([abs(np.sum(np.reshape(combinations[iC],(len(reaction.reactants),len(reaction.products))),axis = 1) - [gas.n_atoms(reac,element) * reaction.reactants[reac] for reac in reaction.reactants]).sum() < RES,abs(np.sum(np.reshape(combinations[iC],(len(reaction.reactants),len(reaction.products))),axis = 0) - [gas.n_atoms(prod,element) * reaction.products[prod] for prod in reaction.products]).sum() < RES]) for iC in range(np.shape(combinations)[0])]]
    
    return combinations, dlt_sp_cmp, dlt_W     

def generate_valid_combinations_new(reaction, element, gas):

    # What is this
    fct_lvls    = []
    fct_lvls_v  = []
    dlt_sp_cmp  = []
    dlt_W       = []
    dlt_bd      = []
    
    RES = 1e-9
    
    # Number of elements in the reactants
    n_e = 0
    # For each reactant specie and molar coefficient in the reaction
    for spec_r, coef_r in reaction.reactants.items():
        # Number of element in the current reactant
        n_atoms_spec_r = gas.n_atoms(spec_r, element)
        # Increment the number of elements by the
        # contribution of spec_r to the number of elements
        n_e += n_atoms_spec_r * coef_r
        # For each product and molar coefficient in the reaction
        for spec_p, coef_p in reaction.products.items(): #for k=1:size(Prod_RR_n,2)
            # Number of atoms of element in the current product
            n_atoms_spec_p = gas.n_atoms(spec_p, element)
            # Total number of atoms from each species
            atoms_in_r_and_p = [n_atoms_spec_r * coef_r,
                                n_atoms_spec_p * coef_p]
            # Append the molar coefficient with the least atoms
            fct_lvls_v.append([coef_r, coef_p][np.argmin(atoms_in_r_and_p)])
            # Append the number of atoms for the species contributing the
            # Least number of atoms in the reaction
            fct_lvls.append([n_atoms_spec_r, 
                             n_atoms_spec_p][np.argmin(atoms_in_r_and_p)])
            # All elements balance between the current reactant and product
            element_balance = []
            for element in gas.element_names:
                # Difference in number of elements between each specie
                diff = gas.n_atoms(spec_r, element) - gas.n_atoms(spec_p, element)
                element_balance.append(np.abs(diff))
            # Append to the list for each reactant/product pair
            dlt_sp_cmp.append(np.sum(element_balance))
            # Difference in molecular weight between the current
            # Reactants and products
            W_reactant = gas.molecular_weights[gas.species_index(spec_r)]
            W_product = gas.molecular_weights[gas.species_index(spec_p)]
            # Store the absolute value
            dlt_W.append(np.abs(W_reactant - W_product))
            # Store the difference
            dlt_bd.append(W_reactant - W_product)
    
    # Convert the molar coefficients to integers
    fct_lvls = np.array(fct_lvls, dtype=int)

    # Generate all possible combinations
    level_ranges = []  
    # For each lowest number of atoms between reactant and product
    for level in fct_lvls:
        # Append a range between zero and the number of atoms
        level_ranges.append(range(level + 1))
    # Combinations between ranges of number of atoms
    level_combinations = list(product(*level_ranges))
    # Broadcasted multiplication with the corresponding molar coefficients
    combinations = np.multiply(level_combinations, fct_lvls_v)

    # If more than one combination
    if len(combinations) > 1:
        # check that the total number of exchanged atom matches 
        # the total number of atoms in the reactants (and hence products)
        combinations = combinations[combinations.sum(axis=1) == int(n_e)]

    # Check for:
    # If more than one combination
    if len(combinations) > 1:
        # Total number of atoms of element in the reactants
        reactant_atoms = []
        for sp, mcoef in reaction.reactants.items():
            reactant_atoms.append(gas.n_atoms(sp, element) * mcoef)
        # Total number of atoms of element in the products
        products_atoms = []
        for sp, mcoef in reaction.products.items():
            products_atoms.append(gas.n_atoms(sp, element) * mcoef)
        # Shape of a combination from the number of reactants
        comb_shape = (len(reaction.reactants), len(reaction.products))
        # For each combination
        new_indexes = []
        for comb in combinations:
            # Array for each combination axis
            bool_array = [0, 0]
            # Reshape according to species in reaction
            comp = comb.reshape(comb_shape)
            # Validation with first axis (reactants)
            # 1) a given reactant that the total number of atoms given 
            # matches its elemental composition
            residual = np.abs(np.sum(comp, axis=1) - reactants_atoms)
            bool_array[0] = np.sum(residu) < RES
            # Validation with second axis (products)
            # 2) a given product that the total number of atoms received 
            # matches its elemental composition
            residual = np.abs(np.sum(comp, axis=0) - products_atoms)
            bool_array[1] = np.sum(residu) < RES
            # If the residual is lower for both we keep
            new_indexes.append(np.all(bool_array))
        # Reindex
        combinations = combinations[new_indexes]

    return combinations, dlt_sp_cmp, dlt_W    

    # Original absolutely cursed line for legacy:
    # new_indexes = [np.all([abs(np.sum(np.reshape(combinations[iC],
    #                                             (len(reaction.reactants),
    #                                              len(reaction.products))),
    #                                  axis = 1)\ 
    #                           - [gas.n_atoms(reac, element) * reaction.reactants[reac]\ 
    #                                                    for reac in reaction.reactants]).sum()\ 
    #                       < RES, 
    #                       abs(np.sum(np.reshape(combinations[iC],
    #                                             (len(reaction.reactants),
    #                                              len(reaction.products))),
    #                                  axis = 0)\
    #                           - [gas.n_atoms(prod,element)  * reaction.products[prod]  
    #                                                    for prod in reaction.products]).sum()\
    #                       < RES]) for iC in range(np.shape(combinations)[0])]
