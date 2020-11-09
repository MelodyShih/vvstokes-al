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
