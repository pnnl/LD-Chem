#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve process ODE

@author: Laura Fierce
"""


from dataclasses import dataclass
from dataclasses import replace

from typing import Tuple
from typing import Callable

from particles import ParticlePopulation

from processes import water_uptake
from processes import air_thermo
from processes import fluctuations
from processes.water_uptake import dlnr_dt
import constants as c

from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from scipy.integrate import solve_ivp

@dataclass
class Problem:
    xvarnames: Tuple[str, ...]
    x0: Tuple[float, ...]
    rhs_funs: Tuple[Callable, ...] # Time derivative of state x0 with respect to t
    t0: float = 0.
    tf: float
    solver: str = 'BDF' # Set default to BDF method, which works for stiff problems


def initialize_problem(settings,processes):
    # rhs is the same at all timesteps
    # x0 and args are updated at each timestep
    problem = Problem(xvarnames=[],x0=[],rhs_funs=[],t0=0.,tf=settings.dt)
    # get xvarnames (the unique set of variable names, with mAeroSpec_PartID as an example)
    # get rhs for each of the varnames, and the lists xvarnmaes and argnames
    # [LATER] for (xvarnames,argnames) get (x0,args) within each solver loop 
    return problem



# first try

def ammend_problem(problem, additional_xvarnames, additional_x0, additional_rhs_funs):
    # note: xvarnames needs to be Dwet_0001, etc. 
    xvarnames = problem.xvarnames
    var_idx = []
    for ii,add_xvarname in enumerate(additional_xvarnames):
        if add_xvarname in xvaranmes:
            idx_in_orig = np.where([xvarname == add_xvarname for xvarname in xvarnames])
            problem.rhs_funs[idx_in_orig] += additional_rhs_funs[ii]
            if problem.x0[idx_in_orig] != additional_x0[ii]:
                print('error! There is a problem here.')
        else:
            var_idx = len(xvarnames)
            xvarnames.append(add_xvarname)
    replace(problem.xvarnames, xvarnames)

def retrieve_xs(xvarnames,parcel_state):
    xs = []
    for xvarname in xvarnames:
        xs.append(get_vardat(varname,parcel_state))
    return xs

def retrieve_rhs(xvarnames,parcel_state,processes,feedbacks):
    rhs = np.zeros(len(xvarnames))
    for xvarname in xvarnames:
        if processes.condensation:
            dDdt = 2.*water_uptake.dr_dt(D/2., r_dry_i, kappa_i, P, T, S, wv, accom=1.)
        
    return rhs, args, feedbacks

def get_ode_onevar(xvarname, parcel_state):
    x0 = get_vardat(varname,parcel_state)
    
    
    return rhs, x0, args
    
def get_vardat(varname,parcel_state):
    try:
        ii = retrieve_ii(varname)
    except:
        ii = None
    
    if varname.startswith('Dwet'):
        vardat = parcel_state.population.particles[ii].get_Dwet()
    elif varname == 'Ddry':
        vardat = parcel_state.population.particles[ii].get_Ddry()
    elif varname == 'S':
        vardat=parcel_state.S
    elif varname == 'T':
        vardat=parcel_state.T
    elif varname == 'P':
        vardat=parcel_state.P
    elif varname == 'wv':
        vardat=parcel_state.wv
    # elif varname.startswith('m_'): # mass of component in particle
    return vardat
        
def retrieve_ii(varname):
    idx_ = varname.find('_')
    ii = int(varname[idx_+1:])
    return ii
        
def make_iterable(some_object):
    try:
        some_object_iterator = iter(some_object)
    except TypeError as te:
        some_object = np.array([some_object])
    return some_object

def build_problem(parcel_state, coupled_processes, settings):
    # this step builds 
    
    for process in coupled_processes:
        if process.condensation:
            
    return problem    