import mechanica as mx

mx.init()


class AType(mx.ParticleType):
    radius = 0.1
    dynamics = mx.Overdamped
    mass = 5
    style = {"color": "MediumSeaGreen"}


class BType(mx.ParticleType):
    radius = 0.1
    dynamics = mx.Overdamped
    mass = 10
    style = {"color": "skyblue"}


A = AType.get()
B = BType.get()

p = mx.Potential.coulomb(q=2, min=0.01, max=3)

mx.bind.types(p, A, A)
mx.bind.types(p, B, B)
mx.bind.types(p, A, B)

r = mx.Force.random(mean=0, std=1)

mx.bind.force(r, A)

pos = [x * 10 + mx.Universe.center for x in mx.random_points(mx.PointsType.SolidCube, 50000)]

[A(p) for p in pos]

a = A.items()[0]

[p.become(B) for p in a.neighbors(3)]

a.radius = 2

mx.show()
