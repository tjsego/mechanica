import mechanica as mx

# dimensions of universe
dim = [30., 30., 30.]

mx.init(dim=dim,
        cutoff=10,
        integrator=mx.FORWARD_EULER,
        dt=0.0005, windowless=True)


class AType(mx.ParticleType):
    radius = 0.5
    dynamics = mx.Overdamped
    mass = 10
    style = {"color": "MediumSeaGreen"}


class BType(mx.ParticleType):
    radius = 0.5
    dynamics = mx.Overdamped
    mass = 10
    style = {"color": "skyblue"}


A, B = AType.get(), BType.get()


class CType(mx.ClusterType):
    radius = 3
    types = [A, B]


C = CType.get()

c1 = C(position=mx.Universe.center - (3, 0, 0))
c2 = C(position=mx.Universe.center + (7, 0, 0))

[A(clusterId=c1.id) for _ in range(2000)]
[B(clusterId=c2.id) for _ in range(2000)]

p1 = mx.Potential.morse(d=0.5, a=5, max=3)
p2 = mx.Potential.morse(d=0.5, a=2.5, max=3)
mx.bind.types(p1, A, A, bound=True)
mx.bind.types(p2, B, B, bound=True)

rforce = mx.Force.random(mean=0, std=10)
mx.bind.force(rforce, A)
mx.bind.force(rforce, B)

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
