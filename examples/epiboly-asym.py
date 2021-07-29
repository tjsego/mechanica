import mechanica as mx
import numpy as np

# dimensions of universe
dim = [30., 30., 30.]

mx.init(dim=dim,
        cutoff=5,
        integrator=mx.FORWARD_EULER,
        dt=0.001)


class AType(mx.ParticleType):
    radius = 0.5
    dynamics = mx.Overdamped
    mass = 5
    style = {"color": "MediumSeaGreen"}


class BType(mx.ParticleType):
    radius = 0.2
    dynamics = mx.Overdamped
    mass = 10
    style = {"color": "skyblue"}


class CType(mx.ParticleType):
    radius = 10
    frozen = True
    style = {"color": "orange"}


A = AType.get()
B = BType.get()
C = CType.get()

pc = mx.Potential.glj(e=10, m=3, max=5)
pa = mx.Potential.glj(e=2, m=4, max=3.0)
pb = mx.Potential.glj(e=1, m=4, max=1)
pab = mx.Potential.harmonic(k=10, r0=0, min=0.01, max=0.55)


# simple harmonic potential to pull particles
h = mx.Potential.harmonic(k=40, r0=0.001, max=5)

mx.bind.types(pc, A, C)
mx.bind.types(pc, B, C)
mx.bind.types(pa, A, A)

r = mx.Force.random(0, 5)

mx.bind.force(r, A)


c = C(mx.Universe.center)

pos_a = [p * ((1 + 0.125) * C.radius) + mx.Universe.center for p in mx.random_points(mx.PointsType.SolidSphere,
                                                                                     3000,
                                                                                     dr=0.25,
                                                                                     phi1=0.60 * np.pi)]

parts, bonds = mx.bind.sphere(h, type=B, n=4, phi=(0.6 * np.pi, np.pi), radius=C.radius + B.radius)

[A(p) for p in pos_a]

# grab a vertical slice of the neighbors of the yolk:
slice_parts = [p for p in c.neighbors() if p.sphericalPosition()[1] > 0]

mx.bind.bonds(pab, slice_parts, cutoff=5 * A.radius, pairs=[(A, B)])

C.style.visible = False
B.style.visible = False


def update(e):
    print(B.items().center_of_mass.as_list())


mx.on_time(invoke_method=update, period=0.01)


mx.run()
