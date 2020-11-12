'''
=======================================
Defines abstract weak forms.
(adapted from perturbed_netwon repo
 see https://bitbucket.org/johannrudi/perturbed_newton.git)

Author:             Johann Rudi
                    Melody Shih
=======================================
'''

import firedrake as fd
import math
import sys


#=======================================
# Objective
#=======================================

def objective(U, Phi, reg=None, dx=None):
    '''
    Creates the weak form for the objetive functional:

        \int reg/2*|U|^2 + Phi
    '''
    obj = Phi

    if reg is not None:
        assert isinstance(reg, float) and 0.0 <= reg
        obj = obj + 0.5*reg*fd.inner(U, U)

    if dx is not None:
        return (obj)*dx

    return (obj)*fd.dx

#=======================================
# Linearization
#=======================================

def gradient(U, U_test, dPhi, reg=None, dx=None):
    '''
    Creates the weak form for the gradient:

        \int reg*(U,U_test) + dPhi(sigma)/(2*sigma)*(U,U_test)
    '''
    sigma = fd.sqrt(fd.inner(0.5*U, U))
    grad = dPhi/(2*sigma)*fd.inner(U, U_test)

    if reg is not None:
        assert isinstance(reg, float) and 0.0 <= reg
        grad = grad + reg*fd.inner(U, U_test)
    if dx is not None:
        return (grad)*dx
    return (grad)*fd.dx

def hessian_Picard(U, U_trial, U_test, dPhi, reg=None, dx=None):
    '''
    Creates the weak form for the Hessian of the Picard linearization:

        \int reg*(U_trial,U_test) + dPhi/(2*sigma)*(U_trial,U_test)
    '''
    sigma = fd.sqrt(fd.inner(0.5*U, U))
    hess = dPhi/(2*sigma)*fd.inner(U_trial, U_test)

    if reg is not None:
        assert isinstance(reg, float) and 0.0 <= reg
        hess = hess + reg*fd.inner(U_trial, U_test)
    if dx is not None:
        return (hess)*dx
    return (hess)*fd.dx

def hessian_NewtonStandard(U, U_trial, U_test, dPhi, dsqPhi, reg=None, dx=None):
    '''
    Creates the weak form for the Hessian of the standard Newton linearization:

        \int reg*(Utrial,U_test) + 
             dPhi/(2*sigma)*(I - (dPhi-sigma*dsqPhi)/dPhi * U(x)U/|U|^2)*
                                                               (Utrial, U_test)

    '''
    sigma = fd.sqrt(fd.inner(0.5*U, U))
    magn_sq = fd.inner(U, U)
    hess = dPhi/(2*sigma)*(fd.inner(U_trial, U_test) - \
           (dPhi-sigma*dsqPhi)/dPhi* \
           fd.inner(U_trial, U)*fd.inner(U, U_test)/magn_sq)

    if reg is not None:
        assert isinstance(reg, float) and 0.0 <= reg
        hess = hess + reg*fd.inner(U_trial, U_test)

    if dx is not None:
        return (hess)*dx
    return (hess)*fd.dx

def hessian_NewtonStressvel(U, U_trial, U_test, S, dPhi, dsqPhi, 
    scale=1.0, reg=None, dx=None):
    '''
    Creates the weak form for the Hessian of the stress-vel Newton 
    linearization:

        \int reg*(Utrial,U_test) + 
             dPhi/(2*sigma)*(I - (dPhi-sigma*dsqPhi)/dPhi * sqrt(2)/dPhi(sigma)
                                         * U(x)S/|U|)*(Utrial, U_test)
    '''
    sigma = fd.sqrt(fd.inner(0.5*U, U))
    magn = fd.sqrt(fd.inner(U, U))
    hess = dPhi/(2*sigma)*(fd.inner(U_trial, U_test) - \
           (dPhi-sigma*dsqPhi)/dPhi*math.sqrt(2)/dPhi\
                *scale*fd.inner(U_trial, S)*fd.inner(U, U_test)/magn)
        
    if reg is not None:
        assert isinstance(reg, float) and 0.0 <= reg
        hess = hess + reg*fd.inner(U_trial, U_test)
    if dx is not None:
        return (hess)*dx
    return (hess)*fd.dx

def hessian_NewtonStressvelSym(U, U_trial, U_test, S, dPhi, dsqPhi, 
    scale=1.0, reg=None, dx=None):
    '''
    Creates the weak form for the Hessian of the standard Newton linearization:

        \int reg*(Utrial,U_test) + 
             dPhi/(2*sigma)*(I - (dPhi-sigma*dsqPhi)/dPhi * sqrt(2)/dPhi(sigma)
                                          * (U(x)S+S(x)U)/|U|)*(Utrial, U_test)
    '''
    sigma = fd.sqrt(fd.inner(0.5*U, U))
    magn = fd.sqrt(fd.inner(U, U))
    hess = dPhi/(2*sigma)*(fd.inner(U_trial, U_test) - \
          (dPhi-sigma*dsqPhi)/dPhi*math.sqrt(2)/dPhi\
               *scale*0.5*(fd.inner(U_trial, S)*fd.inner(U, U_test)+
                           fd.inner(U_trial, U)*fd.inner(S, U_test))/magn)

    if reg is not None:
        assert isinstance(reg, float) and 0.0 <= reg
        hess = hess + reg*fd.inner(U_trial, U_test)
    if dx is not None:
        return (hess)*dx
    return (hess)*fd.dx

def dualstepNewtonStressvel(S, S_test, U, U_step, dPhi, dsqPhi, 
    scale=1.0, dx=None):
    '''
    Creates the weak form for dual variable step of the stress-vel
    Newton linearization:

       \int -(S, S_test) + 
            dPhi/(2*sigma) * ((U_step,S_test) + (U, S_test)) + 
            -(dPhi-sigma*dsqPhi)/(dPhi*sigma) * U(x)S/(2*sigma)*(U_step,S_test)
    '''
    sigma = fd.sqrt(fd.inner(0.5*U, U))
    if isinstance(scale, float):
        if scale > 1e15:
            dual = 0.0
        else:
            dual = dPhi/(2*sigma*scale)*fd.inner(U_step, S_test) + \
                   dPhi/(2*sigma*scale)*fd.inner(U, S_test) - \
                    (dPhi-sigma*dsqPhi)/(dPhi*sigma)* \
                    fd.inner(U_step, S)*fd.inner(U, S_test)/(2*sigma)
    else:
        dual = dPhi/(2*sigma*scale)*fd.inner(U_step, S_test) + \
               dPhi/(2*sigma*scale)*fd.inner(U, S_test) - \
               (dPhi-sigma*dsqPhi)/(dPhi*sigma)* \
               fd.inner(U_step, S)*fd.inner(U, S_test)/(2*sigma)
    dual = dual - fd.inner(S, S_test)

    if dx is not None:
        return (dual)*dx
    return (dual)*fd.dx

def dualResidual(S, S_test, U, dPhi, scale=1.0, dx=None):
    '''
    Creates the weak form for residual of dual variable:
  
        \int dPhi/(2*sigma)*(U,S_test) - (S,S_test)
    '''
    sigma = fd.sqrt(fd.inner(0.5*U, U))
    dualres = fd.inner(U, S_test)/scale*dPhi/(2*sigma)
    dualres -= fd.inner(S, S_test)

    if dx is not None:
        return (dualres)*dx
    return (dualres)*fd.dx

#=======================================
# Basic Weak Forms
#=======================================

def mass(FncSp):
    '''
    Creates the weak form for a mass matrix.
    '''
    return fd.inner(fd.TrialFunction(FncSp), fd.TestFunction(FncSp)) * fd.dx

def magnitude(U, FncSpScalar):
    '''
    Creates the weak form for computing a pointwise magnitude.
    '''
    return fd.sqrt(fd.inner(U, U)) * fd.TestFunction(FncSpScalar) * fd.dx

def magnitude_normalize(U, FncSp):
    '''
    Creates the weak form for normalizing a function by its magnitude.
    '''
    magn     = fd.sqrt(fd.inner(U, U))
    magn_min = 1.0e20 * sys.float_info.min
    return fd.inner( fd.conditional(fd.lt(magn_min, magn), U/magn, 0.0*U),
                     fd.TestFunction(FncSp) ) * fd.dx

def magnitude_scale(U_magn, U_scal, FncSp):
    '''
    Creates the weak form for scaling a function by the magnitude of another function.
    '''
    magn = fd.sqrt(fd.inner(U_magn, U_magn))
    return fd.inner( fd.conditional(fd.lt(1.0, magn), magn*U_scal, U_scal),
                     fd.TestFunction(FncSp) ) * fd.dx
