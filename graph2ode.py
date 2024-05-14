# This script converts a graphical edgelist representation 
# of a chemical reaction network into a system of ODEs
# Obin Sturm (psturm@usc.edu) May 2024
import numpy as np
import pandas as pd
from ChemicalCase import ChemicalCase
from scipy.integrate import solve_ivp

# read in the csv file using the ChemicalCase class
case = ChemicalCase('LosAngeles_L1_20180702_1900.txt')

# read in a csv file of the edgelist
edgelist = pd.read_csv('gckpp_EdgeList.csv',skiprows=25)

# find max species index in edgelist
number_active_spc = max(edgelist['# species index (starts from 1)'])
# only keep the active species
c0 = case.concentrations[:number_active_spc]
dcdt = np.zeros_like(c0)

# rate constants
k = case.rate_constants

# find where the reaction rates are not using the mass action law
def f_check(t, c0):
    # initialize the rates array to 1
    rates = np.ones(len(k))
    dcdt = np.zeros_like(c0)
    for rxn in range(1, len(rates)+1):
        # Step 1: define reaction rates
        # find instance of each reaction "R{integer}" in the "to" column and get all the species involved in that reaction
        reaction = 'R{}'.format(rxn)
        to_indices = edgelist[edgelist['to'] == reaction].index

        # then get the species to_indices of the reactants only
        reactants = edgelist['# species index (starts from 1)'][to_indices]-1

        # mass action law; multiply reactant concentrations (raise to the power of their stoich)
        rates[rxn-1] *= k[rxn-1]*np.prod([c0[reactants.iloc[r]]**edgelist['stoichiometric value '][to_indices[r]] for r in range(len(to_indices))])
        # rates = np.array(case.reaction_rates)

        # Step 2: assign species consumption rates
        for r in reactants:
            dcdt[r] -= rates[rxn-1]
            # print("reaction" + str(rxn) + "has reactants " + str(reactants+1))
        
        # Step 3: assign species production rates
        # find instance of each reaction "R{integer}" in the "from" column and get all the species involved in that reaction
        from_indices = edgelist[edgelist['from'] == reaction].index

        # then get the species to_indices of the products only
        products = edgelist['# species index (starts from 1)'][from_indices]-1
        for p in products:
            dcdt[p] += rates[rxn-1]
            # print("reaction" + str(rxn) + "has products " + str(products+1))

    return dcdt, rates

# find where rates is not close (<1e-8 percent) to np.array(case.reaction_rates)
# how many elements is that?
dcdt, rates = f_check(0, c0)
num_rxn_notmassaction = len(np.where((np.abs(rates - np.array(case.reaction_rates))/(np.array(case.reaction_rates)+1e-8)) > 1e-8)[0])
notmassaction = np.where((np.abs(rates - np.array(case.reaction_rates))/(np.array(case.reaction_rates)+1e-8)) > 1e-8)[0]
print("there are " + str(num_rxn_notmassaction) + " reactions that are not using mass action law:")
print(notmassaction)
print("assuming that these reaction rates instead are constant")
print("and equal to the values in case.reaction_rates")



def f(t, c0):
    # initialize the rates array to 1
    rates = np.ones(len(k))
    dcdt = np.zeros_like(c0)
    for rxn in range(1, len(rates)+1):
        # Step 1: define reaction rates
        # find instance of each reaction "R{integer}" in the "to" column and get all the species involved in that reaction
        reaction = 'R{}'.format(rxn)
        to_indices = edgelist[edgelist['to'] == reaction].index

        # then get the species to_indices of the reactants only
        reactants = edgelist['# species index (starts from 1)'][to_indices]-1

        # mass action law; multiply reactant concentrations (raise to the power of their stoich)
        rates[rxn-1] *= k[rxn-1]*np.prod([c0[reactants.iloc[r]]**edgelist['stoichiometric value '][to_indices[r]] for r in range(len(to_indices))])
        rates[notmassaction] = np.array(case.reaction_rates)[notmassaction]
        # rates = np.array(case.reaction_rates)

        # Step 2: assign species consumption rates
        for r in reactants:
            dcdt[r] -= rates[rxn-1]
            # print("reaction" + str(rxn) + "has reactants " + str(reactants+1))
        
        # Step 3: assign species production rates
        # find instance of each reaction "R{integer}" in the "from" column and get all the species involved in that reaction
        from_indices = edgelist[edgelist['from'] == reaction].index

        # then get the species to_indices of the products only
        products = edgelist['# species index (starts from 1)'][from_indices]-1
        for p in products:
            dcdt[p] += rates[rxn-1]
            # print("reaction" + str(rxn) + "has products " + str(products+1))

    return dcdt

# solve the system of ODEs
# sol = solve_ivp(f, [0, 1], c0, method='LSODA', t_eval=np.linspace(0, 1, 2))
