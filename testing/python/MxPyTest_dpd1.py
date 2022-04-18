import mechanica as mx

mx.init(dt=0.1, dim=[15, 12, 10],
        bc={'x': 'periodic', 'y': 'periodic', 'z': 'no_slip'},
        perfcounter_period=100, windowless=True)

# lattice spacing
a = 0.7


class AType(mx.ParticleType):
    radius = 0.3
    style = {"color": "seagreen"}
    dynamics = mx.Newtonian
    mass = 10


A = AType.get()

dpd = mx.Potential.dpd(sigma=1.5)

mx.bind.types(dpd, A, A)

f = mx.ConstantForce([0.01, 0, 0])

mx.bind.force(f, A)

uc = mx.lattice.sc(a, A)

parts = mx.lattice.create_lattice(uc, [15, 15, 15])

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
