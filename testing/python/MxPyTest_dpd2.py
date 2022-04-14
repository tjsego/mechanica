import mechanica as mx

mx.init(dt=0.1, dim=[15, 12, 10],
        bc={'x': 'no_slip',
            'y': 'periodic',
            'bottom': 'no_slip',
            'top': {'velocity': [-0.1, 0, 0]}},
        perfcounter_period=100)

# lattice spacing
a = 0.3

mx.Universe.boundary_conditions.left.restore = 0.5


class AType(mx.ParticleType):
    radius = 0.2
    style = {"color": "seagreen"}
    dynamics = mx.Newtonian
    mass = 10


A = AType.get()

dpd = mx.Potential.dpd(alpha=0.5, gamma=1, sigma=0.1, cutoff=0.5)

mx.bind.types(dpd, A, A)

uc = mx.lattice.sc(a, A)

parts = mx.lattice.create_lattice(uc, [25, 25, 25])

print(mx.Universe.boundary_conditions)

mx.step(20*mx.Universe.dt)


def test_pass():
    pass
