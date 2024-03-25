import numpy as np
import cantera as ct
from scipy.integrate import simpson
from .element_transfert import e_transfer_matrix


class ReactionGraph(np.ndarray):
    """
    Reaction graph object for the transfert of a given element 
    that can be constructed from multiple input sources: 
    - Cantera.Solution (0D)
    - Cantera.Flame (1D) 
    - User supplied integrated reaction rates of progress
    The latter requires suppliying the corresponding
    Cantera.Solution instance
    """
	# Cantera objects for isinstance() tests
    supported_cantera_1D = [ct.onedim.FreeFlame]
    supported_cantera_0D = [ct.composite.Solution]
    ct_sol = ct.composite.Solution
    
    def __new__(cls, rdata=None, sol=None, element=None, sdata=None):
        """
        Creates a new ReactionGraph instance from reaction data
		(Required)
		rdata: reaction data can be either:
			   - Cantera.Solution
			   - Cantera.FreeFlame	
			   - User supplied integrated net rates of progress
			     with shape 1 x len(sol.reactions)
		sol: Cantera.Solution instance to access reaction data for
			 reaction pathways with user supplied data
		element: String of the element ('N', 'C', 'H', ...) for which
				 the reaction graph is computed
		(Optional)
		sdata: User supplied species reaction rate data to validate the
			   reaction graph produced with user suplied reaction data
        """
        # Bogus graph so we can instanciate the class
        graph = np.ones((2, 2))
        # Define obj once
        obj = np.asarray(graph).view(cls)
        # Check if the input is ok
        obj.check_for_bad_input(rdata=rdata, sol=sol, element=element)
        # Find out how we are gonna build the reaction graph
        input_type = obj.find_out_input_type(rdata)
        
        # Case for Solution input
        if input_type == 'cantera0D':
            # Solution data is also the input data
            species, graph = obj.compute_reaction_graph_0D(rdata, 
                                                           element)
            # Update the superclassed numpy array
            obj = np.asarray(graph).view(cls)
            obj.species = species
            
        # Case for 1D flame input
        elif input_type == 'cantera1D':
            species, graph = obj.compute_reaction_graph_1D(rdata, 
                                                       element)
            # Update the superclassed numpy array
            obj = np.asarray(graph).view(cls)
            obj.species = species
            
        # Case for custom reaction data (from integrated nD sims)
        elif input_type == 'user':
            species, graph = obj.compute_reaction_graph_user(rdata, 
                                                             sol, 
                                                             element,
                                                             sdata,
                                                             do_residuals=False)
            # Update the superclassed numpy array
            obj = np.asarray(graph).view(cls)
            obj.species = species
        
        return obj
    
    def check_for_bad_input(self, **kwargs):
        """
        Just making sure the input is ok
        """
        # Check if data was supplied
        if kwargs['rdata'] is None:
            raise ValueError("Must supply reaction data")
        # Check if an element was supplied
        if kwargs['element'] is None:
            raise ValueError("Must supply an element string")
        # If the sol argument is not a Cantera Solution
        if not isinstance(kwargs['sol'], self.ct_sol):
            # Reaction data could be a gas instance
            if isinstance(kwargs['rdata'], self.ct_sol):
                return
            # Try getting it from a 1D flame input
            else:
                try:
                    _ = kwargs['rdata'].gas
                # Maybe fail
                except AttributeError as e:
                    raise ValueError(("Could not find Cantera.Solution "
                                      "object in the reaction data. It "
                                      "sould be supplied with the sol argument"))
                    

    def find_out_input_type(self, rdata):
        """
        Parse the keyword input to find out
        what input we are dealing with
        """
        for ctobj in self.supported_cantera_0D:
            if isinstance(rdata, ctobj):
                return 'cantera0D'
        for ctobj in self.supported_cantera_1D:
            if isinstance(rdata, ctobj):
                return 'cantera1D'
        return 'user'
    
    @staticmethod
    def compute_reaction_graph_0D(gas, element):
        """
        Compute the reaction graph between species
        containing {element} in a gas instance
        """
        # Get the names of species containing element
        species = [sp.name for sp in gas.species() if gas.n_atoms(sp.name, element)]

        # Get the index of the species from the gas instance
        species_idx = [gas.species_index(sp) for sp in species]

        # Empty array for the reaction graph
        graph = np.zeros((len(species), len(species)))

        ne_s1s2 = e_transfer_matrix(gas, element)
        
        for irx in range(0,np.shape(ne_s1s2)[0]):
            idx_r = np.where(species_idx == ne_s1s2[irx,1])[0]
            idx_p = np.where(species_idx == ne_s1s2[irx,2])[0]

            graph[idx_r, idx_p] += ne_s1s2[irx,3]*gas.net_rates_of_progress[ne_s1s2[irx,0].astype(int)]

        ## in/out fluxes
        Flux_net = np.zeros((len(species),))

        for i_sp, sp_name in enumerate(species):
            Flux_net[i_sp] = gas.net_production_rates[species_idx[i_sp]]*gas.n_atoms(species_idx[i_sp], element)

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
    
    @staticmethod
    def compute_reaction_graph_1D(flame, element):
        """
        Compute the reaction graph between species
        containing {element} in {flame}
        """
        # What does this do ?
        ix_in = 0
        ix_out = len(flame.grid)
        gas = flame.gas

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
                graph[idx_r,idx_p] = simpson(ne_rate[idx_r,idx_p,:], 
                                             x=flame.grid)

        ## in/out fluxes
        Flux_net = np.zeros((len(species),))

        for i_sp, sp_name in enumerate(species):
            Flux_net[i_sp] = simpson(flame.net_production_rates[species_idx[i_sp]], 
                                     x=flame.grid)*gas.n_atoms(species_idx[i_sp],element)

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


    @staticmethod
    def compute_reaction_graph_user(rdata, gas, element, sdata=None, do_residuals=False):
        """
        Compute the reaction graph between species
        containing {element} in {flame}
        rdata: reaction data 1 x n_reactions array in the same order as in the
               Cantera.Solution instance
        gas: Cantera.Solution instance
        element: Element string for which the graph is computed
        sdata: Species reaction rate data for validation
        """
        # Get the names of species containing element
        species = [sp.name for sp in gas.species() if gas.n_atoms(sp.name, element)]

        # Get the index of the species from the gas instance
        species_idx = [gas.species_index(sp) for sp in species]

        # Empty array for the reaction graph
        graph = np.zeros((len(species), len(species)))

        ne_s1s2 = e_transfer_matrix(gas, element)

        #ne_rate = np.zeros((len(species), len(species)))


        for irx in range(0,np.shape(ne_s1s2)[0]):
            idx_r = np.where(species_idx == ne_s1s2[irx, 1])[0]
            idx_p = np.where(species_idx == ne_s1s2[irx, 2])[0]
            
            graph[idx_r, idx_p] += ne_s1s2[irx, 3]*rdata[ne_s1s2[irx, 0].astype(int)]
            
        ## in/out fluxes
        if do_residuals:
            Flux_net = np.zeros((len(species), ))

            for i_sp, sp_name in enumerate(species):
                Flux_net[i_sp] = sdata[species_idx[i_sp]]*gas.n_atoms(species_idx[i_sp],element)

            Flux_in = np.maximum(np.zeros(np.shape(Flux_net)), -Flux_net)
            Flux_out = np.reshape(np.append(np.maximum(np.zeros(np.shape(Flux_net)), Flux_net),[0,0]),(len(Flux_net)+2,1))


            residuals = abs((graph.sum(axis = 1) - graph.sum(axis = 0) + Flux_net)/Flux_in.sum())
            print("Maximum atom flux imbalance is {:.2e}%".format(residuals.max()*100))

            if residuals.max() > 1e-9:
                raise ValueError("Imbalance in atom flux detected")    

            graph = np.vstack((graph, Flux_in))
            graph = np.hstack((graph, np.zeros((np.shape(graph)[0], 1))))
            graph = np.vstack((graph, np.zeros((1, np.shape(graph)[1]))))
            graph = np.hstack((graph,Flux_out))
            species.append('Influx')
            species.append('Outflux')

        # Put rates in new forward (positive) direction
        graph = graph - graph.T
        graph[graph < 0] = 0

        # add balance check

        # Normalize
        if do_residuals:
            graph /= Flux_in.sum()
        #graph /= np.max(graph)
        # For plotting
        return species, graph
