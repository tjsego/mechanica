import mechanica as mx

# dimensions of universe
dim = [30., 30., 30.]

mx.init(dim=dim,
        cutoff=5,
        integrator=mx.FORWARD_EULER,
        dt=0.0005)


class AType(mx.ParticleType):
    radius = 0.5
    dynamics = mx.Overdamped
    mass = 5
    style = {"color": "MediumSeaGreen"}


class BType(mx.ParticleType):
    radius = 0.2
    dynamics = mx.Overdamped
    mass = 1
    style = {"color": "skyblue"}


class CType(mx.ParticleType):
    radius = 10
    frozen = True
    style = {"color": "orange"}


A = AType.get()
B = BType.get()
C = CType.get()

C(mx.Universe.center)

# make a ring of of 50 particles
pts = mx.Points(mx.PointsType.Ring, 100)
pts = [x * (C.radius+B.radius) + mx.Universe.center - mx.MxVector3f(0, 0, 1) for x in pts]
[B(p) for p in pts]

pc = mx.Potential.glj(e=30, m=2, max=5)
pa = mx.Potential.glj(e=3, m=2.5, max=3)
pb = mx.Potential.glj(e=1, m=4, max=1)
pab = mx.Potential.glj(e=1, m=2, max=1)
ph = mx.Potential.harmonic(r0=0.001, k=200)

mx.bind.types(pc, A, C)
mx.bind.types(pc, B, C)
mx.bind.types(pa, A, A)
mx.bind.types(pab, A, B)

r = mx.Force.random(mean=0, std=5)

mx.bind.force(r, A)
mx.bind.force(r, B)

mx.bind.bonds(ph, B.items(), 1)


def update(e: mx.Event):
    """Callback to report the center of mass of all B-type particles during simulation"""
    print(e.times_fired, B.items().center_of_mass.as_list())


# Implement the callback
mx.on_time(period=0.01, invoke_method=update)

mx.run()
