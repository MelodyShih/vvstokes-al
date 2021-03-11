'''
=======================================
Defines weak forms.
(adapted from perturbed_netwon repo
 see https://bitbucket.org/johannrudi/perturbed_newton.git)

Author:             Johann Rudi
                    Melody Shih
=======================================
'''

import firedrake as fd
import math
import sys

import Abstract.WeakForm_Phi

PHICASE = 'Ideal'
#=======================================
# Basic Weak Forms
#=======================================

def grad_u(u, FncSpVelGrad):
    '''
    Creates the weak form for grad(u).
    '''
    return fd.inner(fd.nabla_grad(u), fd.TestFunction(FncSpVelGrad)) * fd.dx

def strainrate(u, FncSpVelGrad):
    '''
    Creates the weak form for the strain rate tensor.
    '''
    return fd.inner(fd.sym(fd.nabla_grad(u)), fd.TestFunction(FncSpVelGrad)) * fd.dx

def viscstress(u, FncSpVelGrad, viscosity):
    '''
    Creates the weak form for the viscous stress tensor.
    '''
    return 2.0 * viscosity * strainrate(u, FncSpVelGrad)

def strainrateII(u, FncSpScalar=None):
    '''
    Creates the weak form for the second invariant of the strain rate tensor.
    '''
    srII = fd.sqrt(0.5*fd.inner( fd.sym(fd.nabla_grad(u)), fd.sym(fd.nabla_grad(u)) ))
    if FncSpScalar is None:
        return srII
    else:
        return srII * fd.TestFunction(FncSpScalar) * fd.dx

def visceff(u, visc1, visc_min, yield_strength, FncSpScalar=None):
    '''
    Creates the weak form for the second invariant of the viscous stress tensor.
    '''
    srII = fd.sqrt(0.5*fd.inner( fd.sym(fd.nabla_grad(u)), fd.sym(fd.nabla_grad(u)) ))
    if PHICASE == 'Ideal':
        visc_eff = visc_min + fd.conditional(
                          fd.lt(2.0*visc1*srII, yield_strength),
                          visc1, 0.5*yield_strength/srII)
    elif PHICASE == 'Comp':
        visc_eff = yield_strength*visc1/(2*srII*visc1+yield_strength)
    else: 
        raise ValueError("Unknown type of Phi")
    if FncSpScalar is None:
        return visc_eff
    else:
        return visc_eff* fd.TestFunction(FncSpScalar) * fd.dx

def viscstressII(u, FncSpScalar, viscosity):
    '''
    Creates the weak form for the second invariant of the viscous stress tensor.
    '''
    if FncSpScalar is None:
        return srII
    else:
        return srII * fd.TestFunction(FncSpScalar) * fd.dx

def linear_stokes(rhs_mom, FncSp, visc1, dx=None, dx1=None, visc2=0.0, dx2=None, visc3=0.0, dx3=None):
    '''
    Creates the weak form for linear stokes equation
    '''
    if dx is None:
        dx = fd.dx
        dx1 = dx

    (u_trial,p_trial) = fd.TrialFunctions(FncSp)
    (u_test,p_test)   = fd.TestFunctions(FncSp)
    U_trial   = fd.sym(fd.nabla_grad(u_trial))
    U_test    = fd.sym(fd.nabla_grad(u_test))

    visc1_stress = 2.0*visc1*fd.inner(U_trial, U_test)*dx1
    if dx2 is not None:
      visc2_stress = 2.0*visc2*fd.inner(U_trial, U_test)*dx2
    else:
      visc2_stress = 0.0

    if dx3 is not None:
      visc3_stress = 2.0*visc3*fd.inner(U_trial, U_test)*dx3
    else:
      visc3_stress = 0.0

    grad_press  = -p_trial*fd.div(u_test)*dx
    div_vel     = -fd.div(u_trial)*p_test*dx
    stokes      = visc1_stress + visc2_stress + grad_press + div_vel + visc3_stress
    rhs_vel     = fd.inner(rhs_mom, u_test)*dx

    return (stokes, rhs_vel)

#==========================================
# Phi, derivatives of Phi, Regularization
#==========================================

def Phi(sigma, yield_strength, viscosity, shift=None):
    if PHICASE == 'Ideal':
        phi =  fd.conditional( fd.lt(2*viscosity*sigma, yield_strength), \
          2*viscosity*sigma*sigma,
          2*yield_strength*sigma - 0.5*yield_strength*yield_strength/viscosity)
        if shift is not None:
            if isinstance(shift, float):
                phi = phi + fd.Constant(shift)
            else:
                phi = phi + shift
    elif PHICASE == 'Comp':
        if isinstance(yield_strength, float):
            if yield_strength < 1e15:
                phi = 2*yield_strength*sigma - yield_strength*yield_strength/\
                      viscosity*fd.ln(yield_strength + 2*viscosity*sigma)
            else: 
                phi = 2*viscosity*sigma*sigma
        else:
            phi = 2*yield_strength*sigma - yield_strength*yield_strength/\
                  viscosity*fd.ln(yield_strength + 2*viscosity*sigma)
    else:
        raise ValueError("Unknown type of Phi")

    return phi
    
def dPhi(sigma, yield_strength, viscosity):
    if PHICASE == 'Ideal':
        dphi = fd.conditional( fd.lt(2*viscosity*sigma, yield_strength), \
                    4*viscosity*sigma, 2*yield_strength)
    elif PHICASE == 'Comp':
        if isinstance(yield_strength, float):
            if yield_strength < 1e15:
                dphi = 4*viscosity*yield_strength*sigma/(yield_strength+ \
                       2*sigma*viscosity)
            else: 
                dphi = 4*viscosity*sigma
        else:
            dphi = 4*viscosity*yield_strength*sigma/(yield_strength+ \
                   2*sigma*viscosity)
            
    else:
        raise ValueError("Unknown type of Phi")
        
    return dphi
        
def dsqPhi(sigma, yield_strength, viscosity):
    if PHICASE == 'Ideal':
        dsqphi = fd.conditional( fd.lt(2*viscosity*sigma, yield_strength), \
                4*viscosity, 0.0)
    elif PHICASE == 'Comp':
        if isinstance(yield_strength, float):
            if yield_strength < 1e15:
                dsqphi = 4*viscosity*yield_strength*yield_strength /\
                        (yield_strength+2*sigma*viscosity) /\
                        (yield_strength+2*sigma*viscosity)
            else: 
                dsqphi = 4*viscosity
        else:
            dsqphi = 4*viscosity*yield_strength*yield_strength /\
                    (yield_strength+2*sigma*viscosity) /\
                    (yield_strength+2*sigma*viscosity)
    else:
        raise ValueError("Unknown type of Phi")
    return dsqphi

def dsqPhidtauy(sigma, yield_strength, viscosity):
    if PHICASE == 'Comp':
        if isinstance(yield_strength, float):
            dsqphidtauy = 0.0
        else:
            dsqphidtauy = 8*viscosity*viscosity*sigma*sigma /\
                    (yield_strength+2*sigma*viscosity) /\
                    (yield_strength+2*sigma*viscosity)
    else:
        raise ValueError("Unknown type of Phi")
    return dsqphidtauy

def Reg(viscosity_min):
    if PHICASE == 'Ideal':
        reg = 2*viscosity_min
    elif PHICASE == 'Comp':
        reg = None
    else:
        raise ValueError("Unknown type of Phi")
    return reg

#=======================================
# Objective 
#=======================================
def objective(u, p, rhs_mom, visc1, viscosity_min, 
              yield_strength, dx=None, dx1=None, visc2=0.0, dx2=None,
              visc3=0.0, dx3=None):
    '''
    Creates the weak form for the objective functional:

        \int viscmin*(grad_s u,grad_s u) + Phi(eII(u)) - p*div(u) - (f,u)

    where
        eII(u) = sqrt of the 2nd invariant of the strain rate (of u)
        viscmin = viscosity_min
    '''
    assert 0.0 <= viscosity_min
    if dx is None:
        dx  = fd.dx
        dx1 = dx

    U         = fd.sym(fd.nabla_grad(u))
    sigma     = fd.sqrt(fd.inner(0.5*U, U))
    shift1  = 0.5*yield_strength*yield_strength/visc1

    obj1 = Abstract.WeakForm_Phi.objective(U=U,
              Phi=Phi(sigma, yield_strength, visc1, shift=shift1),
              reg=Reg(viscosity_min), dx=dx1)
    if dx2 is not None:
        obj2 = Abstract.WeakForm_Phi.objective(U=U,
               Phi=Phi(sigma, 1e16, visc2), 
               reg=Reg(viscosity_min), dx=dx2)
    else:
        obj2 = 0.0

    if dx3 is not None:
        obj3 = Abstract.WeakForm_Phi.objective(U=U,
               Phi=Phi(sigma, 1e16, visc3), 
               reg=Reg(viscosity_min), dx=dx3)
    else:
        obj3 = 0.0
        
    return obj1 + obj2 + obj3 - fd.inner(rhs_mom, u)*dx
        
#=======================================
# Linearization
#=======================================

def gradient(u, p, rhs_mom, FncSp, visc1, viscosity_min, yield_strength, 
             dx=None, dx1=None, visc2=0.0, dx2=None, 
             visc3=0.0, dx3=None, stab=None):
    '''
    Creates the weak form for the gradient:

    (1) Viscoplastic domain (dx1)
      \int (2*viscmin + dPhi(eII(u))/(2*eII(u)))*(grad_s u,grad_s ute) - 
                                                        p*div(ute) - div(u)*pte
    (2) Isoviscous domain (dx2)
        \int 2*visc2*(grad_s utr,grad_s ute) - ptr*div(ute) - div(utr)*pte

    where

        ute = u_test = TestFunction of the velocity space
        pte = p_test = TestFunction of the pressure space
        eII(u) = sqrt of the 2nd invariant of the strain rate (of u)
        dPhi(eII(u)) = derivative of Phi with respect to eII(u)
        viscmin = viscosity_min
    '''
    assert 0.0 <= viscosity_min
    if dx is None:
        dx  = fd.dx
        dx1 = dx

    (u_test,p_test) = fd.TestFunctions(FncSp)
    U         = fd.sym(fd.nabla_grad(u))
    U_test    = fd.sym(fd.nabla_grad(u_test))

    sigma     = fd.sqrt(fd.inner(0.5*U, U))
    visc1_stress = Abstract.WeakForm_Phi.gradient(U=U, U_test=U_test,
        dPhi=dPhi(sigma, yield_strength, visc1),
        reg=Reg(viscosity_min), dx=dx1)
    if dx2 is not None:
        visc2_stress = Abstract.WeakForm_Phi.gradient(U=U, U_test=U_test,
            dPhi=dPhi(sigma, 1e16, visc2),
            reg=Reg(viscosity_min), dx=dx2)
    else:
        visc2_stress = 0.0

    if dx3 is not None:
        visc3_stress = Abstract.WeakForm_Phi.gradient(U=U, U_test=U_test,
            dPhi=dPhi(sigma, 1e16, visc3),
            reg=Reg(viscosity_min), dx=dx3)
    else:
        visc3_stress = 0.0

    grad_press  = -p*fd.div(u_test)*dx
    div_vel     = -fd.div(u)*p_test*dx
    rhs_vel     = fd.inner(rhs_mom, u_test)*dx

    grad = visc1_stress + visc2_stress + visc3_stress + grad_press + div_vel - rhs_vel
    #if stab is not None:
    #    h = fd.CellDiameter(FncSp.mesh())
    #    grad = grad + _stabilization(p, p_test, h, stab) - \
    #                  _stabilizationRhs(rhs_mom, p_test, h, stab)
    return grad

def precondvisc(u, p, visc1, viscosity_min, yield_strength, visc2=None):

    U         = fd.sym(fd.nabla_grad(u))
    sigma     = fd.sqrt(fd.inner(0.5*U, U))

    precondvisc1 = Abstract.WeakForm_Phi.precondvisc(U, 
                               dPhi=dPhi(sigma, yield_strength, visc1), 
                               reg=Reg(viscosity_min))

    if visc2 is not None:
        precondvisc2 = Abstract.WeakForm_Phi.precondvisc(U, 
                                   dPhi=dPhi(sigma, 1e16, visc2), 
                                   reg=Reg(viscosity_min))
        precondvisc2 = 2*visc2
        return precondvisc1, precondvisc2
    
    return precondvisc1

def hessian_Picard(u, p, FncSp, visc1, viscosity_min, yield_strength, dx=None, 
                  dx1=None, visc2=0.0, dx2=None, visc3=0.0, dx3=None,):
    '''
    Creates the weak form for the Hessian of the Picard linearization:

    (1) Viscoplastic domain (dx1)
        \int (2*viscmin + dPhi(eII(u))/(2*eII(u)))*(grad_s utr,grad_s ute) 
                                                  - ptr*div(ute) - div(utr)*pte

    (2) Isoviscous domain (dx2)
        \int 2*visc2*(grad_s utr,grad_s ute) - ptr*div(ute) - div(utr)*pte
    where

        utr = u_trial = TrialFunction of the velocity space
        ptr = p_trial = TrialFunction of the pressure space
        ute = u_test = TestFunction of the velocity space
        pte = p_test = TestFunction of the pressure space
        eII(u) = sqrt of the 2nd invariant of the strain rate (of u)
        dPhi(eII(u)) = derivative of Phi with respect to eII(u)
        viscmin = viscosity_min
    '''
    assert 0.0 <= viscosity_min
    if dx is None:
        dx  = fd.dx
        dx1 = dx

    (u_trial,p_trial) = fd.TrialFunctions(FncSp)
    (u_test,p_test)   = fd.TestFunctions(FncSp)
    U         = fd.sym(fd.nabla_grad(u))
    U_trial   = fd.sym(fd.nabla_grad(u_trial))
    U_test    = fd.sym(fd.nabla_grad(u_test))

    sigma     = fd.sqrt(fd.inner(0.5*U, U))
    visc1_stress = Abstract.WeakForm_Phi.hessian_Picard(U=U, 
        U_trial=U_trial, U_test=U_test,
        dPhi=dPhi(sigma, yield_strength, visc1),
        reg=Reg(viscosity_min), dx=dx1)

    if dx2 is not None:
        visc2_stress = Abstract.WeakForm_Phi.hessian_Picard(U=U, 
            U_trial=U_trial, U_test=U_test,
            dPhi=dPhi(sigma, 1e16, visc2),
            reg=Reg(viscosity_min), dx=dx2)
    else:
        visc2_stress = 0.0

    if dx3 is not None:
        visc3_stress = Abstract.WeakForm_Phi.hessian_Picard(U=U, 
            U_trial=U_trial, U_test=U_test,
            dPhi=dPhi(sigma, 1e16, visc3),
            reg=Reg(viscosity_min), dx=dx3)
    else:
        visc3_stress = 0.0

    grad_press   = -p_trial*fd.div(u_test)*dx
    div_vel      = -fd.div(u_trial)*p_test*dx
    hess         = visc1_stress + visc2_stress + visc3_stress + grad_press + div_vel
    return hess

    

def hessian_NewtonStandard(u, p, FncSp, visc1, viscosity_min, yield_strength, 
                           dx=None, dx1=None, visc2=0.0, dx2=None, 
                           visc3=0.0, dx3=None, stab=None):
    '''
    Creates the weak form for the Hessian of the standard Newton linearization:

    (1) Viscoplastic domain (dx1)
        \int 2*viscmin*(grad_s utr,grad_s ute) + 
             dPhi(eII(u))/(2*eII(u))*(I - 
                 Psi(eII(u))*(grad_s u (x) grad_s u)/(grad_s u,grad_s u)))*
                 (grad_s utr,grad_s ute) - ptr*div(ute) - div(utr)*pte

    (2) Isoviscous domain (dx2)
        \int 2*visc2*(grad_s utr,grad_s ute) - ptr*div(ute) - div(utr)*pte

    where

        utr = u_trial = TrialFunction of the velocity space
        ptr = p_trial = TrialFunction of the pressure space
        ute = u_test = TestFunction of the velocity space
        pte = p_test = TestFunction of the pressure space
        eII(u) = sqrt of the 2nd invariant of the strain rate (of u)
        dPhi(eII(u)) = first derivative of Phi with respect to eII(u)
        dsqPhi(eII(u)) = second derivative of Phi with respect to eII(u)
        Psi(eII(u)) = (dPhi(eII(u)) - eII(u)*dsqPhi(eII(u)))/dPhi(eII(u))
        viscmin = viscosity_min
    '''
    assert 0.0 <= viscosity_min
    if dx is None:
        dx  = fd.dx
        dx1 = dx

    (u_trial,p_trial) = fd.TrialFunctions(FncSp)
    (u_test,p_test)   = fd.TestFunctions(FncSp)

    U         = fd.sym(fd.nabla_grad(u))
    U_trial   = fd.sym(fd.nabla_grad(u_trial))
    U_test    = fd.sym(fd.nabla_grad(u_test))
    sigma     = fd.sqrt(fd.inner(0.5*U, U))

    visc1_stress = Abstract.WeakForm_Phi.hessian_NewtonStandard(U=U, 
                       U_trial=U_trial, U_test=U_test, 
                       dPhi=dPhi(sigma, yield_strength, visc1),
                       dsqPhi=dsqPhi(sigma, yield_strength, visc1),
                       reg=Reg(viscosity_min), dx=dx1)

    if dx2 is not None:
        visc2_stress = Abstract.WeakForm_Phi.hessian_NewtonStandard(U=U,
                           U_trial=U_trial, U_test=U_test,
                           dPhi=dPhi(sigma, 1e16, visc2),
                           dsqPhi=dsqPhi(sigma, 1e16, visc2),
                           reg=Reg(viscosity_min), dx=dx2)
    else:
        visc2_stress = 0.0

    if dx3 is not None:
        visc3_stress = Abstract.WeakForm_Phi.hessian_NewtonStandard(U=U,
                           U_trial=U_trial, U_test=U_test,
                           dPhi=dPhi(sigma, 1e16, visc3),
                           dsqPhi=dsqPhi(sigma, 1e16, visc3),
                           reg=Reg(viscosity_min), dx=dx3)
    else:
        visc3_stress = 0.0

    grad_press = -p_trial*fd.div(u_test)*dx
    div_vel    = -fd.div(u_trial)*p_test*dx

    hess         = visc1_stress + visc2_stress + visc3_stress + grad_press + div_vel
    #if stab is not None:
    #    h = fd.CellDiameter(FncSp.mesh())
    #    hess = hess + _stabilization(p_trial, p_test, h, stab)
    return hess

def hessian_NewtonStressvel(u, p, PrimalFncSp, S, visc1, viscosity_min, 
                            yield_strength, dx=None, dx1=None, visc2=0.0, 
                            dx2=None, visc3=0.0, dx3=None, stab=None):
    '''
    Creates the weak form for the Hessian of the stress-vel Newton linearization:

    (1) Viscoplastic domain (dx1)
        \int 2*viscmin*(grad_s utr,grad_s ute) + 
             dPhi(eII(u))/(2*eII(u))*(I -Psi(eII(u))*sqrt(2)/dPhi(eII(u))*
                 (grad_s u (x) S)/(grad_s u,grad_s u))*(grad_s utr,grad_s ute) - 
             ptr*div(ute) - div(utr)*pte

    (2) Isoviscous domain (dx2)
        \int 2*visc2*(grad_s utr,grad_s ute) - ptr*div(ute) - div(utr)*pte

    where

        utr = u_trial = TrialFunction of the velocity space
        ptr = p_trial = TrialFunction of the pressure space
        ute = u_test = TestFunction of the velocity space
        pte = p_test = TestFunction of the pressure space
        S = viscous stress tensor (tau), also called dual variable
        eII(u) = sqrt of the 2nd invariant of the strain rate (of u)
        dPhi(eII(u)) = first derivative of Phi with respect to eII(u)
        dsqPhi(eII(u)) = second derivative of Phi with respect to eII(u)
        Psi(eII(u)) = (dPhi(eII(u)) - eII(u)*dsqPhi(eII(u)))/dPhi(eII(u))
        viscmin = viscosity_min
    '''
    assert 0.0 <= viscosity_min
    if dx is None:
        dx = fd.dx
        dx1 = dx

    (u_trial,p_trial) = fd.TrialFunctions(PrimalFncSp)
    (u_test,p_test)   = fd.TestFunctions(PrimalFncSp)
    U         = fd.sym(fd.nabla_grad(u))
    U_trial   = fd.sym(fd.nabla_grad(u_trial))
    U_test    = fd.sym(fd.nabla_grad(u_test))

    sigma     = fd.sqrt(fd.inner(0.5*U, U))
    scale1 = math.sqrt(2)*yield_strength
    visc1_stress = Abstract.WeakForm_Phi.hessian_NewtonStressvel(U=U,
                      U_trial=U_trial, U_test=U_test, S=S,
                      dPhi=dPhi(sigma, yield_strength, visc1),
                      dsqPhi=dsqPhi(sigma, yield_strength, visc1),
                      scale=scale1,
                      reg=Reg(viscosity_min), dx=dx1)
    if dx2 is not None:
        scale2 = math.sqrt(2)*1e16
        visc2_stress = Abstract.WeakForm_Phi.hessian_NewtonStressvel(U=U,
            U_trial=U_trial, U_test=U_test, S=S,
            dPhi=dPhi(sigma, 1e16, visc2),
            dsqPhi=dsqPhi(sigma, 1e16, visc2),
            scale=scale2,
            reg=Reg(viscosity_min), dx=dx2)
    else:
        visc2_stress = 0.0

    if dx3 is not None:
        scale3 = math.sqrt(3)*1e16
        visc3_stress = Abstract.WeakForm_Phi.hessian_NewtonStressvel(U=U,
            U_trial=U_trial, U_test=U_test, S=S,
            dPhi=dPhi(sigma, 1e16, visc3),
            dsqPhi=dsqPhi(sigma, 1e16, visc3),
            scale=scale3,
            reg=Reg(viscosity_min), dx=dx3)
    else:
        visc3_stress = 0.0

    grad_press   = -p_trial*fd.div(u_test)*dx
    div_vel      = -fd.div(u_trial)*p_test*dx
    hess         = visc1_stress + visc2_stress + visc3_stress + grad_press + div_vel
    #if stab is not None:
    #    h = fd.CellDiameter(PrimalFncSp.mesh())
    #    hess = hess + _stabilization(p_trial, p_test, h, stab)
    return hess

def hessian_NewtonStressvelSym(u, p, PrimalFncSp, S, visc1, viscosity_min, 
                               yield_strength, dx=None, dx1=None, visc2=0.0, 
                               dx2=None, 
                               visc3=0.0, dx3=None, stab=None):
    '''
    Creates the weak form for the Hessian of the symmetrized stress-vel Newton 
    linearization:

    (1) Viscoplastic domain (dx1)
        \int 2*viscmin*(grad_s utr,grad_s ute) + 
             dPhi(eII(u))/(2*eII(u))*(I -Psi(eII(u))*sqrt(2)/dPhi(eII(u))*
                 (grad_s u (x) S + S (x) grad_s u)/(grad_s u,grad_s u))*
             (grad_s utr,grad_s ute) - ptr*div(ute) - div(utr)*pte

    (2) Isoviscous domain (dx2)
        \int 2*visc2*(grad_s utr,grad_s ute) - ptr*div(ute) - div(utr)*pte

    where

        utr = u_trial = TrialFunction of the velocity space
        ptr = p_trial = TrialFunction of the pressure space
        ute = u_test = TestFunction of the velocity space
        pte = p_test = TestFunction of the pressure space
        S = viscous stress tensor (tau), also called dual variable
        eII(u) = sqrt of the 2nd invariant of the strain rate (of u)
        dPhi(eII(u)) = first derivative of Phi with respect to eII(u)
        dsqPhi(eII(u)) = second derivative of Phi with respect to eII(u)
        Psi(eII(u)) = (dPhi(eII(u)) - eII(u)*dsqPhi(eII(u)))/dPhi(eII(u))
        viscmin = viscosity_min
    '''
    assert 0.0 <= viscosity_min
    if dx is None:
        dx = fd.dx
        dx1 = dx

    (u_trial,p_trial) = fd.TrialFunctions(PrimalFncSp)
    (u_test,p_test)   = fd.TestFunctions(PrimalFncSp)
    U         = fd.sym(fd.nabla_grad(u))
    U_trial   = fd.sym(fd.nabla_grad(u_trial))
    U_test    = fd.sym(fd.nabla_grad(u_test))

    sigma     = fd.sqrt(fd.inner(0.5*U, U))
    scale1 = math.sqrt(2)*yield_strength
    visc1_stress = Abstract.WeakForm_Phi.hessian_NewtonStressvelSym(U=U,
        U_trial=U_trial, U_test=U_test, S=S,
        dPhi=dPhi(sigma, yield_strength, visc1),
        dsqPhi=dsqPhi(sigma, yield_strength, visc1),
        scale=scale1,
        reg=Reg(viscosity_min), dx=dx1)
    grad_press = - p_trial*fd.div(u_test)*fd.dx
    
    grad_press   = -p_trial*fd.div(u_test)*dx
    div_vel      = -fd.div(u_trial)*p_test*dx
    if dx2 is not None:
        scale2 = math.sqrt(2)*1e16
        visc2_stress = Abstract.WeakForm_Phi.hessian_NewtonStressvelSym(U=U,
            U_trial=U_trial, U_test=U_test, S=S,
            dPhi=dPhi(sigma, 1e16, visc2),
            dsqPhi=dsqPhi(sigma, 1e16, visc2),
            scale=scale2,
            reg=Reg(viscosity_min), dx=dx2)
    else:
        visc2_stress = 0.0

    if dx3 is not None:
        scale3 = math.sqrt(2)*1e16
        visc3_stress = Abstract.WeakForm_Phi.hessian_NewtonStressvelSym(U=U,
            U_trial=U_trial, U_test=U_test, S=S,
            dPhi=dPhi(sigma, 1e16, visc3),
            dsqPhi=dsqPhi(sigma, 1e16, visc3),
            scale=scale3,
            reg=Reg(viscosity_min), dx=dx3)
    else:
        visc3_stress = 0.0

    hess         = visc1_stress + visc2_stress + visc3_stress + grad_press + div_vel
    #if stab is not None:
    #    h = fd.CellDiameter(PrimalFncSp.mesh())
    #    hess = hess + _stabilization(p_trial, p_test, h, stab)
    return hess

def hessian_dualStep(u, u_step, S, DualFncSp, visc1, viscosity_min, 
                     yield_strength, dx=None, dx1=None, visc2=0.0, dx2=None,
                     visc3=0.0, dx3=None):
    '''
    Creates the weak form for step of dual variable (viscous stress tensor, tau)
    '''
    if dx is None:
        dx = fd.dx
        dx1 = dx

    U         = fd.sym(fd.nabla_grad(u))
    U_step    = fd.sym(fd.nabla_grad(u_step))
    sigma     = fd.sqrt(fd.inner(0.5*U, U))

    scale1 = math.sqrt(2)*yield_strength
    S_step = Abstract.WeakForm_Phi.dualstepNewtonStressvel(S=S, 
                  S_test=fd.TestFunction(DualFncSp),U=U, U_step=U_step, 
                  dPhi=dPhi(sigma, yield_strength, visc1),
                  dsqPhi=dsqPhi(sigma, yield_strength, visc1),
                  scale=scale1, dx=dx1)
    if dx2 is not None:
        scale2 = math.sqrt(2)*1e16
        S_step = S_step + Abstract.WeakForm_Phi.dualstepNewtonStressvel(S=S, 
                          S_test=fd.TestFunction(DualFncSp),U=U, U_step=U_step, 
                          dPhi=dPhi(sigma, 1e16, visc2),
                          dsqPhi=dsqPhi(sigma, 1e16, visc2),
                          scale=scale2, dx=dx2)
    if dx3 is not None:
        scale3 = math.sqrt(2)*1e16
        S_step = S_step + Abstract.WeakForm_Phi.dualstepNewtonStressvel(S=S, 
                          S_test=fd.TestFunction(DualFncSp),U=U, U_step=U_step, 
                          dPhi=dPhi(sigma, 1e16, visc3),
                          dsqPhi=dsqPhi(sigma, 1e16, visc3),
                          scale=scale2, dx=dx3)
    return S_step

def dualresidual(S, u, DualFncSp, visc1, viscosity_min, yield_strength, dx=None, 
                 dx1=None, visc2=0.0, dx2=None, visc3=0.0, dx3=None):
    '''
    Creates the weak form for residual of dual variable (viscous stress tensor, 
    tau)
    '''

    if dx is None:
        dx = fd.dx
        dx1 = dx

    U = fd.sym(fd.nabla_grad(u))

    scale1 = math.sqrt(2)*yield_strength
    sigma     = fd.sqrt(fd.inner(0.5*U, U))
    res = Abstract.WeakForm_Phi.dualResidual(S=S, 
                S_test=fd.TestFunction(DualFncSp),U=U, 
                dPhi=dPhi(sigma, yield_strength, visc1),
                scale=scale1, dx=dx1)
    if dx2 is not None:
        scale2 = math.sqrt(2)*1e16
        res = res + Abstract.WeakForm_Phi.dualResidual(S=S, 
                        S_test=fd.TestFunction(DualFncSp),U=U, 
                        dPhi=dPhi(sigma, 1e16, visc2),
                        scale=scale2, dx=dx2)
    if dx3 is not None:
        scale3 = math.sqrt(2)*1e16
        res = res + Abstract.WeakForm_Phi.dualResidual(S=S, 
                        S_test=fd.TestFunction(DualFncSp),U=U, 
                        dPhi=dPhi(sigma, 1e16, visc3),
                        scale=scale3, dx=dx3)
    return res

def hessian_dualUpdate_boundMaxMagnitude(S, DualFncSp, max_magn):
    S_test = fd.TestFunction(DualFncSp)
    S_rescaled = fd.conditional( fd.lt(fd.inner(S, S), max_magn*max_magn),
                    fd.inner(S, S_test),
                    fd.inner(S, S_test)/fd.sqrt(fd.inner(S,S))*max_magn)*fd.dx
    return S_rescaled
