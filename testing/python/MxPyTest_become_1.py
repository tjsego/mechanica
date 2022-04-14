import mechanica as mx

mx.init()


class AType(mx.ParticleType):

    radius = 1

    species = ['S1', 'S2', 'S3']

    style = {"colormap": {"species": "S2", "map": "rainbow", "range": (0, 1)}}


A = AType.get()


class BType(mx.ParticleType):

    radius = 4

    species = ['S2', 'S3', 'S4']

    style = {"colormap": {"species": "S2", "map": "rainbow", "range": (0, 1)}}


B = BType.get()

o = A()

o.species.S2 = 0.5

o.become(B)

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
