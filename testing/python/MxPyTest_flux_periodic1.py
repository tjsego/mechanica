import mechanica as mx

mx.init(dt=0.1, dim=[15, 6, 6], cells=[9, 3, 3], bc={'x': ('periodic', 'reset')}, cutoff=3, windowless=True)


class AType(mx.ParticleType):
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1",
                          "map": "rainbow"}}


A = AType.get()

mx.Fluxes.flux(A, A, "S1", 2)

a1 = A(mx.Universe.center - [0, 1, 0])
a2 = A(mx.Universe.center + [-5, 1, 0], velocity=[0.5, 0, 0])

a1.species.S1 = 3
a2.species.S1 = 0

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
