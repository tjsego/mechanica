import mechanica as mx

mx.init(dt=0.1, dim=[15, 5, 5],
        bc={'x': ('periodic', 'reset')})


class AType(mx.ParticleType):
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1",
                          "map": "rainbow"}}


A = AType.get()

a1 = A(mx.Universe.center + [0, -1, 0])
a2 = A(mx.Universe.center + [-5, 1, 0], velocity=[0.5, 0, 0])

pressure = mx.ConstantForce([0.1, 0, 0])

mx.bind.force(pressure, A, "S1")

a1.species.S1 = 0
a2.species.S1 = 0.1

mx.run()
