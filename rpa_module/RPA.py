"""
Reaction Path Analysis for Cantera Flames
"""
import graphviz
import cantera as ct
import numpy as np

def format_value(value, line_width):
    """
    Format normalized reaction rate value
    """
    value *= (100/line_width)
    if value > 1.0:
        label = f"{value:.0f}%"
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
    # Integrate the reaction rates
    int_NRR = {}
    for i, rate  in enumerate(flame.net_rates_of_progress):
        int_NRR[f'R{i}'] = np.trapz(rate, flame.grid)

    # Construct the graph
    ct_species = flame.gas.species()
    # Get the relevant species names (special case for H and HE)
    species = [sp.name for sp in ct_species if (element in sp.name) and (sp.name != 'HE')]
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
                        ni = ct_species[ct_index].composition[element]
                        # Reactant has a flow of n_ele * net_rate
                        # Towards the product
                        graph[idx_r, idx_p] += rk * ni
    
    # Remove reverse direction
    graph = graph - graph.T
    graph[graph < 0] = 0
    # Normalize
    graph /= np.max(graph)
    # For plotting
    return species, np.around(graph, 3)


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
