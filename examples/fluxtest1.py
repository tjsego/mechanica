import mechanica as mx

mx.init()


class AType(mx.ParticleType):
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1",
                          "map": "rainbow",
                          "range": "auto"}}


A = AType.get()
mx.Fluxes.flux(A, A, "S1", 5)

a1 = A(mx.Universe.center)
a2 = A(mx.Universe.center + [0, 0.5, 0])

a1.species.S1 = 0
a2.species.S1 = 1

mx.run()
