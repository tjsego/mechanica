import mechanica as mx

# dimensions of universe
dim = [30., 30., 30.]

mx.init(dim=dim,
        cutoff=10,
        integrator=mx.FORWARD_EULER,
        dt=0.0005)


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


A = AType.get()
B = BType.get()


class CType(mx.ClusterType):
    radius = 3
    types = [A, B]


C = CType.get()

c1 = C(position=mx.Universe.center - (3, 0, 0))
c2 = C(position=mx.Universe.center + (7, 0, 0))

[A() for _ in range(2000)]
[B() for _ in range(2000)]

p1 = mx.Potential.glj(e=7, m=1, max=1)
p2 = mx.Potential.glj(e=7, m=1, max=2)
mx.bind.types(p1, A, A, bound=True)
mx.bind.types(p2, B, B, bound=True)

rforce = mx.Force.random(mean=0, std=10)
mx.bind.force(rforce, A)
mx.bind.force(rforce, B)

mx.run()
