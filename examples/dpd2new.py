import mechanica as mx

mx.init(dt=0.1, dim=[15, 12, 10],
        bc={'x': 'periodic',
            'y': 'periodic',
            'bottom': 'no_slip',
            'top': {'velocity': [-0.4, 0, 0]}},
        perfcounter_period=100)

# lattice spacing
a = 0.3


class AType(mx.ParticleType):
    radius = 0.2
    style = {"color": "seagreen"}
    dynamics = mx.Newtonian
    mass = 10


A = AType.get()

dpd = mx.Potential.dpd(alpha=0.3, gamma=1, sigma=1, cutoff=0.6)
dpd_wall = mx.Potential.dpd(alpha=0.5, gamma=10, sigma=1, cutoff=0.1)

mx.bind.types(dpd, A, A)
mx.bind.boundaryCondition(dpd_wall, mx.Universe.boundary_conditions.top, A)


uc = mx.lattice.sc(a, A)

parts = mx.lattice.create_lattice(uc, [25, 25, 25])

print(mx.Universe.boundary_conditions)

mx.show()
