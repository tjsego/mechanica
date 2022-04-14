"""
Demonstrates how to simulate 1D particles in a custom pseudo-Gaussian potential well
"""
import mechanica as mx
from math import exp, factorial
import sys

mx.init(cutoff=5, bc={'x': 'noslip'})


class WellType(mx.ParticleType):
    frozen = True
    style = {'visible': False}


class SmallType(mx.ParticleType):
    radius = 0.1


well_type, small_type = WellType.get(), SmallType.get()
small_type.frozen_y = True

# Build functions for custom potential
lam, mu, s = -0.5, 1.0, 3


def He(r, n):
    """nth Hermite polynomial evaluated at r"""
    if n == 0:
        return 1.0
    elif n == 1:
        return r
    return r * He(r, n-1) - (n-1) * He(r, n-2)


def dgdr(r, n):
    """Utility function for simplifying potential calculations"""
    r = max(r, sys.float_info.min)
    result = 0.0
    for k in range(1, s+1):
        if 2*k - n >= 0:
            result += factorial(2*k) / factorial(2*k - n) * (lam + k) * mu ** k / factorial(k) * r ** (2*k)
    return result / r ** n


def f_n(r: float, n: int):
    """nth derivative of potential function evaluated at r"""
    u_n = lambda r, n: (-1) ** n * He(r, n) * lam * exp(-mu * r ** 2.0)
    w_n = 0.0
    for j in range(0, n+1):
        w_n += factorial(n) / factorial(j) / factorial(n-j) * dgdr(r, j) * u_n(r, n-j)
    return 10.0 * (u_n(r, n) + w_n / lam)


pot_c = mx.Potential.custom(min=0, max=5, f=lambda r: f_n(r, 0), fp=lambda r: f_n(r, 1), f6p=lambda r: f_n(r, 6))
pot_c.name = "Pseudo-Gaussian"
# pot_c.plot(min=0, max=5, potential=True, force=False)
mx.bind.types(p=pot_c, a=well_type, b=small_type)

# Create particles
well_type(position=mx.Universe.center, velocity=mx.MxVector3f(0))
for i in range(20):
    small_type(position=mx.MxVector3f((i+1)/21 * mx.Universe.dim()[0], mx.Universe.center[1], mx.Universe.center[2]),
               velocity=mx.MxVector3f(0))

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
