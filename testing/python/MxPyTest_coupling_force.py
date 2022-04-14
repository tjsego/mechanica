"""
This example demonstrates making force magnitudes depend on concentrations
"""

import mechanica as mx

mx.init(dim=[6.5, 6.5, 6.5], bc=mx.FREESLIP_FULL)


class AType(mx.ParticleType):
    """A particle type carrying a species"""
    radius = 0.1
    species = ['S1']
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": (0, 1)}}
    dynamics = mx.Overdamped


class BType(AType):
    """A particle type that acts as a constant source"""

    dynamics = mx.Newtonian

    @classmethod
    def get(cls):
        result = super().get()
        result.species.S1.constant = True
        return result


A, B = AType.get(), BType.get()

# Particles are randomly perturbed with increasing S1
force = mx.Force.random(0.1, 1.0)
mx.bind.force(force, A, 'S1')
mx.bind.force(force, B, 'S1')

# S1 diffuses between particles
mx.Fluxes.flux(A, A, "S1", 1)
mx.Fluxes.flux(A, B, "S1", 1)

# Make a lattice of stationary particles
uc = mx.lattice.sc(0.25, A)
parts = mx.lattice.create_lattice(uc, [25, 25, 25])

# Grab a particle to act as a constant source
o = parts[24, 0, 24][0]
o.become(B)
o.species.S1 = 10.0

mx.step(100 * mx.Universe.dt)


def test_pass():
    pass
