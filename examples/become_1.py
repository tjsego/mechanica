import mechanica as mx

mx.init()


class AType(mx.ParticleType):

    radius = 1

    species = ['S1', 'S2', 'S3']

    style = {"colormap": {"species": "S2", "map": "rainbow", "range": "auto"}}


A = AType.get()


class BType(mx.ParticleType):

    radius = 4

    species = ['S2', 'S3', 'S4']

    style = {"colormap": {"species": "S2", "map": "rainbow", "range": "auto"}}


B = BType.get()

o = A()

o.species.S2 = 0.5

o.become(B)

mx.show()
