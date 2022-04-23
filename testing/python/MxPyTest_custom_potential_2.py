"""
Demonstrates how to simulate a particle in custom distance and angle potentials using built-in approximations.

The particle is bound by distance to the center of the domain with a cosine potential.

The particle is bound by angle to the x-axis with a cosine potential.
"""
import mechanica as mx
from math import acos, cos, pi, sin
from random import random

mx.init(bc={'x': 'noslip', 'y': 'noslip', 'z': 'noslip'}, cutoff=5.0, windowless=True)


class FrozenType(mx.ParticleType):
    frozen = True
    style = {'visible': False}


class BondedType(mx.ParticleType):
    radius = 0.1


frozen_type, bonded_type = FrozenType.get(), BondedType.get()

# Construct bond potential
rad_a, rad_l = 10.0, 2.0
rad_f = lambda r: rad_a * cos(2 * pi / rad_l * r)
rad_fp = lambda r: - 2 * pi / rad_l * rad_a * sin(2 * pi / rad_l * r)
rad_f6p = lambda r: - (2 * pi / rad_l) ** 6.0 * rad_a * cos(2 * pi / rad_l * r)
pot_rad = mx.Potential.custom(f=rad_f, fp=rad_fp, f6p=rad_f6p, min=0.0, max=2*rad_l)

# Construct angle potential
#   Apporoximating deriviatives, since passed argument is cosine of the angle, which makes differentiation tedious
ang_f = lambda r: 10.0 * cos(6.0 * acos(r))
pot_ang = mx.Potential.custom(min=-0.999, max=0.999, f=ang_f, flags=mx.Potential.Flags.angle.value)

# Create particles
ftp0 = frozen_type(position=mx.Universe.center, velocity=mx.MxVector3f(0))
ftp1 = frozen_type(position=mx.Universe.center + mx.MxVector3f(1, 0, 0), velocity=mx.MxVector3f(0))
btp = bonded_type(position=mx.Universe.center + mx.MxVector3f((random() - 0.5) * mx.Universe.dim()[0] / 4,
                                                              (random() - 0.5) * mx.Universe.dim()[1] / 4, 0))
btp.frozen_z = True

# Bind particles
mx.Bond.create(pot_rad, btp, ftp0)
mx.Angle.create(pot_ang, btp, ftp0, ftp1)

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
