'''
=======================================
Runs nonlinear Stokes solver for a prolbem with vonMises yield criterion.

Author:            Melody Shih
=======================================
'''

from firedrake import *
from firedrake.petsc import PETSc
from alfi.transfer import *
from alfi import *
from functools import reduce
from firedrake.mg.utils import get_level
from balance import load_balance, rebalance

from VariableViscosityStokes import *
import WeakForm
import Abstract.Vector

import copy
import argparse
import numpy as np
PETSc.Sys.popErrorHandler()

import logging
#logging.basicConfig(level="INFO")

#======================================
# Parsing input arguments
#======================================

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--nref", type=int, default=1)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--solver-type", type=str, default="almg")
parser.add_argument("--linearization", type=str, default="stdnewton")
parser.add_argument("--gamma", type=float, default=1e4)
parser.add_argument("--itref", type=int, default=0)
parser.add_argument("--case", type=int, default=3)
parser.add_argument("--discretisation", type=str, default="hdiv")
parser.add_argument("--quad", dest="quad", default=False, action="store_true")
parser.add_argument("--quad-deg", type=int, dest="quad_deg", default=20)
parser.add_argument("--rebalance", dest="rebalance", default=False, action="store_true")
parser.add_argument("--asmbackend", type=str, choices=['tinyasm', 'petscasm'], 
                                                             default="tinyasm")
args, _ = parser.parse_known_args()


nref = args.nref
k = args.k
case = args.case
gamma = Constant(args.gamma)
deg = args.quad_deg
divdegree = None
quad = args.quad

#======================================
# Parameters
#======================================
# Stokes problem parameters
VISC_TYPE = 'const' # const, anomaly
VISC_REG  = 1.e-3
VISC_MAX  = 1.e3
VISC_REG  = 1e-3

RHO = 2700.0
GRAVITY = 9.81
YEAR_PER_SEC = 1./3600/365.25/24

REF_VELOCITY    = 2.5e-3
REF_HEIGHT      = 30000.0 #1.0
REF_VISCOSITY   = 1.e22
REF_STRAIN_RATE = REF_VELOCITY*YEAR_PER_SEC/REF_HEIGHT
REF_STRESS_RATE = 2.0*REF_STRAIN_RATE*REF_VISCOSITY

VISC_UPPER_SCALED   = 1.e24/REF_VISCOSITY
VISC_MIDDLE_SCALED  = 1.e21/REF_VISCOSITY
VISC_LOWER_SCALED   = 1.e19/REF_VISCOSITY
BOUNDARY_INFLOW_VELOCITY = 1.0

# nolinear solver parameters
MONITOR_NL_ITER=True
MONITOR_NL_STEPSEARCH=False
NL_SOLVER_GRAD_RTOL = 1e-8
NL_SOLVER_GRAD_STEP_RTOL = 1e-8
NL_SOLVER_MAXITER = 100
NL_SOLVER_STEP_MAXITER = 15
NL_SOLVER_STEP_ARMIJO    = 1.0e-4

# output
OUTPUT_VTK=True
#======================================
# Setup VariableViscosityStokesProblem
#======================================
vvstokesprob = VariableViscosityStokesProblem(2, # dimension of the problem 
                                    quad, #triangular mesh
                                    args.discretisation, # finite elems spaces
                                    k, # order of discretisation
                                    quaddegree=deg, #quadrature degree
                                    quaddivdegree=divdegree) # qaudrature divdeg                      
if quad:
    basemesh = Mesh('mesh/land_quad_simple.msh')
    #basemesh = Mesh('mesh/land_quad.msh')
else:
    basemesh = Mesh('mesh/land.msh')
vvstokesprob.set_meshhierarchy(basemesh, nref)

mesh = vvstokesprob.get_mesh()
#dx_upper  = Measure("dx", domain=mesh, subdomain_id=2)
#dx_middle = Measure("dx", domain=mesh, subdomain_id=3)
#dx_lower  = Measure("dx", domain=mesh, subdomain_id=1)
dx_upper  = Measure("dx", domain=mesh, subdomain_id=6)
dx_middle = Measure("dx", domain=mesh, subdomain_id=5)
dx_lower  = Measure("dx", domain=mesh, subdomain_id=7)
dx = Measure("dx", domain=mesh, subdomain_id="everywhere")
#vvstokesprob.set_measurelist([dx_upper, dx_lower])
vvstokesprob.set_measurelist([dx])

#--------------------------------------
# Setup boundary condition
#--------------------------------------
mesh = vvstokesprob.get_mesh()
V, Q, Vd = vvstokesprob.get_functionspace(mesh,info=True, dualFncSp=True)
VQ = V*Q

# set functions for boundary conditions
vel_noslip = Constant((0.0, 0.0))
vel_inflow = Constant((BOUNDARY_INFLOW_VELOCITY, 0.0))

def bc_fun(mesh):
    V, Q = vvstokesprob.get_functionspace(mesh)
    VQ = V*Q

    # construct boundary conditions
    bc_walls    = DirichletBC(VQ.sub(0).sub(1), 0.0, sub_domain=4)
    bc_left     = DirichletBC(VQ.sub(0), vel_inflow, sub_domain=2)
    bc_right    = DirichletBC(VQ.sub(0),-vel_inflow, sub_domain=3)
    #bc_walls    = DirichletBC(VQ.sub(0).sub(1), 0.0, sub_domain=6)
    #bc_left     = DirichletBC(VQ.sub(0), vel_inflow, sub_domain=4)
    #bc_right    = DirichletBC(VQ.sub(0),-vel_inflow, sub_domain=5)
    #bc_outflow  = DirichletBC(VQ.sub(1), 0.0       , sub_domain=4)
    bcs = [bc_left, bc_right, bc_walls]
    return bcs

def bcstep_fun(mesh):
    V, Q = vvstokesprob.get_functionspace(mesh)
    VQ = V*Q

    # construct homogeneous Dirichlet BC's at inflow boundary for Newton steps
    bc_walls      = DirichletBC(VQ.sub(0).sub(1), 0.0, sub_domain=4)
    bc_step_left  = DirichletBC(VQ.sub(0), vel_noslip, sub_domain=2)
    bc_step_right = DirichletBC(VQ.sub(0), vel_noslip, sub_domain=3)
    #bc_walls      = DirichletBC(VQ.sub(0).sub(1), 0.0, sub_domain=6)
    #bc_step_left  = DirichletBC(VQ.sub(0), vel_noslip, sub_domain=4)
    #bc_step_right = DirichletBC(VQ.sub(0), vel_noslip, sub_domain=5)
    bcs_step = [bc_step_left, bc_step_right, bc_walls]
    return bcs_step

vvstokesprob.set_bcsfun(bc_fun)
bcs =  vvstokesprob.get_bcs(mesh)

#--------------------------------------
# Setup viscosity, right hand side
#--------------------------------------
# rhs
rhs = Constant((0.0, 0.0))

# set viscosity field
visc_upper  = Constant(VISC_UPPER_SCALED)
visc_middle = Constant(VISC_MIDDLE_SCALED)
visc_lower  = Constant(VISC_LOWER_SCALED)
def visc_fun(mesh):
    V, Q = vvstokesprob.get_functionspace(mesh)
    return Constant(1.0)
vvstokesprob.set_viscosity(visc_fun) #used for W

#--------------------------------------
# Setup weak form
#--------------------------------------
# create solution vectors
sol, sol_prev, step    = Function(VQ), Function(VQ), Function(VQ)
sol_u      = sol.split()[0]
sol_p      = sol.split()[1]
sol_prev_u = sol_prev.split()[0]
sol_prev_p = sol_prev.split()[1]
step_u     = step.split()[0]
step_p     = step.split()[1]

# initialize dual variable
S = None
S_prev = None
S_step = None
S_proj = None

#sol2 = Function(VQ)
#step2 = Function(VQ)

phi = 0
C = 1.e8
A = C
yield_strength = A/REF_VISCOSITY/REF_STRAIN_RATE

# set weak forms of objective functional and gradient
obj  = WeakForm.objective(sol_u, sol_p, rhs, visc_upper, VISC_REG, 
                          yield_strength, dx, dx_upper, visc_lower, dx_lower,
                          visc_middle, dx_middle)
grad = WeakForm.gradient(sol_u, sol_p, rhs, VQ, visc_upper, VISC_REG, 
                         yield_strength, dx, dx_upper, visc_lower, dx_lower,
                         visc_middle, dx_middle)

# set weak form of Hessian and forms related to the linearization
if args.linearization == 'stdnewton':
    hess = WeakForm.hessian_NewtonStandard(sol_u, sol_p, VQ, visc_upper, VISC_REG, 
                                       yield_strength, dx, dx_upper, visc_lower, 
                                       dx_lower,visc_middle, dx_middle)
elif args.linearization == 'stressvel':
    if Vd is None:
        raise ValueError("stressvel not implemented for discretisation %s" \
                                   % vvstokesprob.discretisation)
    S      = Function(Vd)
    S_step = Function(Vd)
    S_proj = Function(Vd)
    S_prev = Function(Vd)
    dualStep = WeakForm.hessian_dualStep(
        sol_u, step_u, S, Vd, visc_upper, VISC_REG, yield_strength,
        dx, dx_upper, visc_lower, dx_lower, visc_middle, dx_middle)
    dualres = WeakForm.dualresidual(S, sol_u, Vd, visc_upper,
        VISC_REG, yield_strength, dx, dx_upper, visc_lower, dx_lower,
        visc_middle, dx_middle)
    hess = WeakForm.hessian_NewtonStressvel(
        sol_u, sol_p, VQ, S_proj, visc_upper, VISC_REG,
        yield_strength, dx, dx_upper, visc_lower, dx_lower,
        visc_middle, dx_middle)
else:
    raise ValueError("unknown type of linearization %s" % args.linearization)

# preconditioner viscosity
#previsc1expr, previsc2expr = WeakForm.precondvisc(sol_u, sol_p, VQ, visc_upper, 
#                                                  VISC_REG, yield_strength, 
#                                                  visc_lower)

#======================================
# Solve the nonlinear problem
#======================================
# initialize solution
#TODO add stablization term for hdiv discretisation
(a,l) = WeakForm.linear_stokes(rhs, VQ, visc_upper, dx, dx_upper,
                               visc_lower, dx_lower, visc_middle, dx_middle)

vvstokesprob.set_linearvariationalproblem(a, l, sol, bcs)
vvstokessolver = VariableViscosityStokesSolver(vvstokesprob, 
                                               args.solver_type, 
                                               args.case,
                                               args.gamma,
                                               args.asmbackend)
vvstokessolver.set_linearvariationalsolver()
vvstokessolver.set_transfers()

for i in range(args.itref+1):
    vvstokessolver.solve()

## uncomment to compare solutions between augmented/unaugmented sys
#solve(a==l, sol, bcs)
#PETSc.Sys.Print("absolute diff in vel:",\
#       norm(sol.split()[0]-sol2.split()[0]))
#PETSc.Sys.Print("relative diff in vel:",\
#       norm(sol.split()[0]-sol2.split()[0])\
#       /norm(sol.split()[0]))
#PETSc.Sys.Print("absolute diff in pre: ",\
#       norm(sol.split()[1]-sol2.split()[1]))
#PETSc.Sys.Print("relative diff in pre: ",\
#       norm(sol.split()[1]-sol2.split()[1])\
#       /norm(sol.split()[1]))

# initialize gradient
g = assemble(grad, bcs=bcstep_fun(mesh))
g_norm_init = g_norm = norm(g)
angle_grad_step_init = angle_grad_step = np.nan

# initialize solver statistics
lin_it       = 0
lin_it_total = 0
obj_val      = assemble(obj)
step_length  = 0.0

# assemble mass matrices
if args.linearization == 'stressvel':
    Md = assemble(Abstract.WeakForm_Phi.mass(Vd))

if MONITOR_NL_ITER:
    PETSc.Sys.Print('{0:<3} "{1:>6}"{2:^20}{3:^14}{4:^15}{5:^10}'.format(
          "Itn", vvstokessolver.solver_type, "Energy", "||g||_l2", 
           "(grad,step)", "step len"))

for itn in range(NL_SOLVER_MAXITER+1):
    # print iteration line
    if MONITOR_NL_ITER:
        PETSc.Sys.Print("{0:>3d} {1:>6d}{2:>20.12e}{3:>14.6e}{4:>+15.6e}{5:>10f}".format(
              itn, lin_it, obj_val, g_norm, angle_grad_step, step_length))

    # stop if converged
    if g_norm < NL_SOLVER_GRAD_RTOL*g_norm_init:
        PETSc.Sys.Print("Stop reason: Converged to rtol; ||g|| reduction %3e." % g_norm/g_norm_init)
        break
    if np.abs(angle_grad_step) < NL_SOLVER_GRAD_STEP_RTOL*np.abs(angle_grad_step_init):
        PETSc.Sys.Print("Stop reason: Converged to rtol; (grad,step) reduction %3e." % \
              np.abs(angle_grad_step/angle_grad_step_init))
        break
    # stop if step search failed
    if 0 < itn and not step_success:
        PETSc.Sys.Print("Stop reason: Step search reached maximum number of backtracking.")
        break

    # set up the linearized system
    if args.linearization == 'stressvel':
        if 0 == itn:
            Abstract.Vector.setZero(S)
            Abstract.Vector.setZero(S_step)
            Abstract.Vector.setZero(S_proj)
        else:
            # project S to unit sphere
            Sprojweak = WeakForm.hessian_dualUpdate_boundMaxMagnitude(S, Vd, 1.0)
            b = assemble(Sprojweak)
            solve(Md, S_proj.vector(), b)

    # assemble linearized system
    vvstokesprob.set_bcsfun(bcstep_fun)
    bcs_step = vvstokesprob.get_bcs(mesh)
    vvstokesprob.set_linearvariationalproblem(hess, grad, step, bcs_step)
    vvstokessolver = VariableViscosityStokesSolver(vvstokesprob, 
                                                   args.solver_type, 
                                                   args.case,
                                                   args.gamma,
                                                   args.asmbackend)
    #vvstokessolver.set_precondviscosity([previsc1expr])
    vvstokessolver.set_linearvariationalsolver()
    vvstokessolver.set_transfers()
    vvstokessolver.solve()
    lin_it=vvstokessolver.get_iterationnum()
    lin_it_total += lin_it
    
    ## uncomment to compare solutions between augmented/unaugmented sys
    #solve(hess == grad, step2, bcs_step)
    #PETSc.Sys.Print("abstepute diff in vel:",\
    #       norm(step.split()[0]-step2.split()[0]))
    #PETSc.Sys.Print("relative diff in vel:",\
    #       norm(step.split()[0]-step2.split()[0])\
    #       /norm(step.split()[0]))
    #PETSc.Sys.Print("abstepute diff in pre: ",\
    #       norm(step.split()[1]-step2.split()[1]))
    #PETSc.Sys.Print("relative diff in pre: ",\
    #       norm(step.split()[1]-step2.split()[1])\
    #       /norm(step.split()[1]))

    # solve dual variable step
    if args.linearization == 'stressvel':
        Abstract.Vector.scale(step, -1.0)
        b = assemble(dualStep)
        solve(Md, S_step.vector(), b)
        Abstract.Vector.scale(step, -1.0)

    # compute the norm of the gradient
    g = assemble(grad, bcs=bcs_step)
    g_norm = norm(g)
    # compute angle between step and (negative) gradient
    angle_grad_step = -step.vector().inner(g)
    if 0 == itn:
        angle_grad_step_init = angle_grad_step

    # initialize backtracking line search
    sol_prev.assign(sol)
    step_length = 1.0
    step_success = False

    # run backtracking line search
    for j in range(NL_SOLVER_STEP_MAXITER):
        sol.vector().axpy(-step_length, step.vector())
        obj_val_next = assemble(obj)
        if MONITOR_NL_STEPSEARCH and 0 < j:
           PETSc.Sys.Print("Step search: {0:>2d}{1:>10f}{2:>20.12e}{3:>20.12e}".format(
                 j, step_length, obj_val_next, obj_val))
        if obj_val_next < obj_val + step_length*NL_SOLVER_STEP_ARMIJO*angle_grad_step:
            if args.linearization == 'stressvel':
                S.vector().axpy(step_length, S_step.vector())
            obj_val = obj_val_next
            step_success = True
            break
        step_length *= 0.5
        sol.assign(sol_prev)
    if not step_success:
        sol.assign(sol_prev)
    Abstract.Vector.scale(step, -step_length)

PETSc.Sys.Print("%s: #iter %i, ||g|| reduction %3e, (grad,step) reduction %3e, #total linear iter %i." % \
    (
        args.linearization,
        itn,
        g_norm/g_norm_init,
        np.abs(angle_grad_step/angle_grad_step_init),
        lin_it_total
    )
)

#======================================
# Output
#======================================

# set the 2nd invariant of the strain rate
strainrateII = WeakForm.strainrateII(sol_u)
visceff = WeakForm.visceff(sol_u, visc_upper, VISC_REG,
                           yield_strength)

# output vtk file for strain rate 
if OUTPUT_VTK:
    Vd1 = FunctionSpace(mesh, "DG", 0)
    edotp   = Function(Vd1)
    edotp_t = TestFunction(Vd1)
    solve(inner(edotp_t, (edotp - strainrateII))*dx == 0.0, edotp)
    Abstract.Vector.scale(edotp, REF_STRAIN_RATE)
    File("vtk/land_strainrateII.pvd").write(edotp)

    Vd1 = FunctionSpace(mesh, "DG", 0)
    edotp   = Function(Vd1)
    edotp_t = TestFunction(Vd1)
    solve((inner(edotp_t, (edotp - visceff))*dx_upper+ \
           inner(edotp_t, (edotp - visc_lower))*dx_lower+ \
           inner(edotp_t, (edotp - visc_middle))*dx_middle) == 0.0, edotp)
    File("vtk/land_visceff.pvd").write(edotp)

    File("vtk/multinotch_solution_u.pvd").write(sol_u)
    File("vtk/multinotch_solution_p.pvd").write(sol_p)
