import mechanica as mx

# dimensions of universe
dim = [30., 30., 30.]

dist = 3

mx.init(dim=dim,
        cutoff=7,
        integrator=mx.FORWARD_EULER,
        cells=[3, 3, 3],
        dt=0.01, windowless=True)


class AType(mx.ParticleType):
    radius = 0.5
    dynamics = mx.Newtonian
    mass = 5
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

p = mx.Potential.glj(e=100, m=3, max=7)

mx.bind.types(p, A, Sphere)
mx.bind.types(p, A, Test)
mx.bind.cuboid(p, A)

# above the sphere
Sphere(mx.Universe.center + [6, 0, 0])
A(mx.Universe.center + [6, 0, Sphere.radius + dist])

# above the test
Test(mx.Universe.center + [0, -10, 3])
A(mx.Universe.center + [0, -10, 3 + dist])

# above the cube
c = mx.Cuboid.create(mx.Universe.center + [-5, 0, 0], size=[6, 6, 6])
A(mx.Universe.center + [-5, 0, 3 + dist])

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
