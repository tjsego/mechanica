import mechanica as mx
import numpy as np

# dimensions of universe
dim = [30., 30., 30.]

mx.init(dim=dim,
        cutoff=10,
        integrator=mx.FORWARD_EULER,
        cells=[1, 1, 1],
        dt=0.005)


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


# above the scube
# c2 = mx.Cuboid.create(mx.Universe.center + [7, 0, 0],
#                      size=[6, 6, 6],
#                      orientation=[-0.3, np.pi/4, 0])


A(mx.Universe.center + [0, 0, 5], velocity=[0, 0, -5])

# A(mx.Universe.center + [-8, 0, 5.8], velocity=[0, 0, -2])

# A(mx.Universe.center + [-5, 0, -5.8], velocity=[-1, 0, 2])

# A(mx.Universe.center + [-3, 3, -5.8], velocity=[0, 0, 2])

# A(mx.Universe.center + [-4, -4, 0], velocity=[0, 3, 0])

# A(mx.Universe.center + [-4,  4, 0], velocity=[0, -3, 0])

# uniform random cube
positions = np.random.uniform(low=0, high=30, size=(1000, 3))

for p in positions:
    A(p, velocity=[0, 0, 0])

mx.step(20*mx.Universe.dt)


def test_pass():
    pass
