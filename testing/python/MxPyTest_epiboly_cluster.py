import mechanica as mx

# dimensions of universe
dim = [30., 30., 30.]

mx.init(dim=dim,
        cutoff=3,
        integrator=mx.FORWARD_EULER,
        dt=0.001)


class BType(mx.ParticleType):
    radius = 0.25
    dynamics = mx.Overdamped
    mass = 15
    style = {"color": "skyblue"}


B = BType.get()


class CType(mx.ClusterType):
    radius = 2.3

    types = [B]


C = CType.get()


def split(event: mx.ParticleTimeEvent):
    particle: mx.ClusterHandle = event.targetParticle
    ptype: mx.ClusterType = event.targetType

    print("split(" + str(ptype.name) + ")")
    axis = particle.position - yolk.position
    print("axis: " + str(axis.as_list()))

    particle.split(axis=axis)


mx.on_particletime(ptype=C, invoke_method=split, period=0.2, selector="largest")


class YolkType(mx.ParticleType):
    radius = 10
    mass = 1000000
    dynamics = mx.Overdamped
    frozen = True
    style = {"color": "gold"}


Yolk = YolkType.get()

total_height = 2 * Yolk.radius + 2 * C.radius
yshift = total_height/2 - Yolk.radius
cshift = total_height/2 - C.radius - 1

yolk = Yolk(position=mx.Universe.center + [0., 0., -yshift])

c = C(position=mx.Universe.center + [0., 0., cshift])

[c(B) for _ in range(4000)]

pb = mx.Potential.soft_sphere(kappa=300, epsilon=6, r0=0.5, eta=2, tol=0.05, min=0.01, max=3)

pub = mx.Potential.soft_sphere(kappa=400, epsilon=0, r0=0.5, eta=2, tol=0.05, min=0.01, max=1.5)

py = mx.Potential.soft_sphere(kappa=300, epsilon=25, r0=1, eta=2, tol=0.04, min=0.01, max=10, shift=True)

rforce = mx.Force.random(mean=0, std=1)

mx.bind.force(rforce, B)
mx.bind.types(pb, C, B, bound=True)
mx.bind.types(pub, C, B, bound=False)
mx.bind.types(py, Yolk, B)

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
