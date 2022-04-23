import mechanica as mx

mx.init(cutoff=3, windowless=True)


class AType(mx.ParticleType):
    radius = 0.1
    dynamics = mx.Overdamped
    style = {"color": "MediumSeaGreen"}


class BType(mx.ParticleType):
    radius = 0.1
    dynamics = mx.Overdamped
    style = {"color": "skyblue"}


A = AType.get()
B = BType.get()

p = mx.Potential.coulomb(q=0.5, min=0.01, max=3)
q = mx.Potential.coulomb(q=0.5, min=0.01, max=3)
r = mx.Potential.coulomb(q=2.0, min=0.01, max=3)

mx.bind.types(p, A, A)
mx.bind.types(q, B, B)
mx.bind.types(r, A, B)

pos = [x * 10 + mx.Universe.center for x in mx.random_points(mx.PointsType.SolidCube, 1000)]

[A(p) for p in pos]

a = A.items()[0]

[p.become(B) for p in a.neighbors(5)]

mx.step(mx.Universe.dt * 100)


def test_pass():
    pass
