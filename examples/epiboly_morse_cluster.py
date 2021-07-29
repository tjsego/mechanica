import mechanica as mx

# dimensions of universe
dim = [30., 30., 30.]

mx.init(dim=dim,
        cutoff=3,
        integrator=mx.FORWARD_EULER,
        dt=0.001,
        cells=[6, 6, 6])


class BType(mx.ParticleType):
    radius = 0.25
    dynamics = mx.Overdamped
    mass = 15
    style = {"color": "skyblue"}


B = BType.get()


class CType(mx.ClusterType):
    radius = 5


C = CType.get()


def split(event: mx.ParticleTimeEvent):
    particle: mx.ClusterHandle = event.targetParticle
    ptype: mx.ClusterType = event.targetType

    print("split(" + str(ptype.name) + ")")
    axis = particle.position - yolk.position
    print("axis: " + str(axis))
    particle.split(axis=axis)

    print("new cluster count: ", len(C.items()))


mx.on_particletime(ptype=C, invoke_method=split, period=1, selector="largest")


class YolkType(mx.ParticleType):
    radius = 10
    mass = 1000000
    dynamics = mx.Overdamped
    flozen = True
    style = {"color": "gold"}


Yolk = YolkType.get()

total_height = 2 * Yolk.radius + 2 * C.radius
yshift = 1.5 * (total_height/2 - Yolk.radius)
cshift = total_height/2 - C.radius - 1

yolk = Yolk(position=mx.Universe.center + [0., 0., -yshift])

c = C(position=yolk.position + [0, 0, yolk.radius + C.radius - 5])

[B() for _ in range(8000)]

pb = mx.Potential.morse(d=15, a=5.5, min=0.1, max=2)
pub = mx.Potential.morse(d=1, a=6, min=0.1, max=2)
py = mx.Potential.morse(d=3, a=3, max=30)

rforce = mx.Force.random(0, 500, 0.0001)

mx.bind.force(rforce, B)
mx.bind.types(pb, B, B, bound=True)
mx.bind.types(pub, B, B, bound=False)
mx.bind.types(py, Yolk, B)

print("initial cluster count: ", len(C.items()))

mx.run()
