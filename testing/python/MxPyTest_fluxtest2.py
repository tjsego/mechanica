import mechanica as mx

mx.init(dim=[6.5, 6.5, 6.5], bc=mx.FREESLIP_FULL)


class AType(mx.ParticleType):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1",
                          "map": "rainbow",
                          "range": (0, 10)}}


A = AType.get()
mx.Fluxes.flux(A, A, "S1", 5, 0.001)

uc = mx.lattice.sc(0.25, A)

parts = mx.lattice.create_lattice(uc, [25, 25, 25])

parts[24, 0, 24][0].species.S1 = 5000

mx.step(10*mx.Universe.dt)


def test_pass():
    pass
