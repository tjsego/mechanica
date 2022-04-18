import mechanica as mx
import numpy as np

# dimensions of universe
dim = [30., 30., 30.]

mx.init(dim=dim,
        cutoff=10,
        integrator=mx.FORWARD_EULER,
        dt=0.005, windowless=True)


class AType(mx.ParticleType):
    radius = 0.5
    dynamics = mx.Newtonian
    mass = 30
    style = {"color": "MediumSeaGreen"}


class SphereType(mx.ParticleType):
    radius = 3
    frozen = True
    style = {"color": "orange"}


class TestType(mx.ParticleType):
    radius = 0
    frozen = True
    style = {"color": "orange"}


A = AType.get()
Sphere = SphereType.get()
Test = TestType.get()

p = mx.Potential.glj(e=1, m=2, max=10)

mx.bind.types(p, A, Sphere)
mx.bind.types(p, A, Test)
mx.bind.cuboid(p, A)
mx.bind.types(p, A, A)


# above the sphere
# Sphere(mx.Universe.center + [5, 0, 0])

# A(mx.Universe.center + [5, 0, 5.8])

# above the test
Test(mx.Universe.center + [0, -10, 3])
# A(mx.Universe.center + [0, -10, 5.8])

# above the scube
c = mx.Cuboid.create(mx.Universe.center + [0, 0, 0],
                     size=[13, 13, 15],
                     orientation=[0, -np.pi/1.8, 0])

c.rotate([0, 0.1, 0])

c.spin = [0, 0.2, 0]

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
