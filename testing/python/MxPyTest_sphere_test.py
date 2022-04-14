import mechanica as mx

# dimensions of universe
dim = [30., 30., 30.]

mx.init(dim=dim,
        cutoff=5,
        integrator=mx.FORWARD_EULER,
        dt=0.001)


class AType(mx.ParticleType):
    radius = 0.1
    dynamics = mx.Overdamped
    mass = 5
    style = {"color": "MediumSeaGreen"}


class CType(mx.ParticleType):
    radius = 10
    frozen = True
    style = {"color": "orange"}


A = AType.get()
C = CType.get()
C(mx.Universe.center)

pos = [p * (C.radius+A.radius) + mx.Universe.center for p in mx.random_points(mx.PointsType.Sphere, 500)]

[A(p) for p in pos]

pc = mx.Potential.glj(e=30, m=2, max=5)
pa = mx.Potential.coulomb(q=100, min=0.01, max=5)

mx.bind.types(pc, A, C)
mx.bind.types(pa, A, A)

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
