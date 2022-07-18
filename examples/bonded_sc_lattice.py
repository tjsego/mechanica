import mechanica as mx
import numpy as np

mx.init(dt=0.1, dim=[15, 12, 10])

# lattice spacing
a = 0.65


class AType(mx.ParticleType):
    radius = 0.3
    style = {"color": "seagreen"}
    dynamics = mx.Overdamped


A = AType.get()


class BType(mx.ParticleType):
    radius = 0.3
    style = {"color": "red"}
    dynamics = mx.Overdamped


B = BType.get()


class FixedType(mx.ParticleType):
    radius = 0.3
    style = {"color": "blue"}
    frozen = True


Fixed = FixedType.get()

repulse = mx.Potential.coulomb(q=0.08, min=0.01, max=2 * a)

mx.bind.types(repulse, A, A)
mx.bind.types(repulse, A, B)

f = mx.ConstantForce(lambda: [0.3, 1 * np.sin(0.4 * mx.Universe.time), 0], 0.01)

mx.bind.force(f, B)

pot = mx.Potential.power(r0=0.5 * a, alpha=2, max=10 * a)

uc = mx.lattice.sc(a, A, lambda i, j: mx.Bond.create(pot, i, j, dissociation_energy=100.0))

parts = mx.lattice.create_lattice(uc, [15, 15, 15])

for p in parts[14, :].flatten():
    p[0].become(B)

for p in parts[0, :].flatten():
    p[0].become(Fixed)

mx.run()
