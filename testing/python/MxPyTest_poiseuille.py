import mechanica as mx

mx.init(dt=0.1, dim=[15, 12, 10], cells=[7, 6, 5], cutoff=0.5,
        bc={'x': 'periodic',
            'y': 'periodic',
            'top': {'velocity': [0, 0, 0]},
            'bottom': {'velocity': [0, 0, 0]}})

# lattice spacing
a = 0.15


class AType(mx.ParticleType):
    radius = 0.05
    style = {"color": "seagreen"}
    dynamics = mx.Newtonian
    mass = 10


A = AType.get()

dpd = mx.Potential.dpd(alpha=10, sigma=1)

mx.bind.types(dpd, A, A)

# driving pressure
pressure = mx.ConstantForce([0.1, 0, 0])

mx.bind.force(pressure, A)

uc = mx.lattice.sc(a, A)

parts = mx.lattice.create_lattice(uc, [20, 20, 20])

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
