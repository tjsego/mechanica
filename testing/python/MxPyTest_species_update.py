import mechanica as mx
import numpy as n

mx.init()


class AType(mx.ParticleType):

    radius = 5

    species = ['S1', 'S2', 'S3']

    style = {"colormap": {"species": "S1", "map": "rainbow"}}

    @staticmethod
    def on_register(ptype):
        def update(event: mx.ParticleTimeEvent):
            for p in ptype.items():
                p.species.S1 = (1 + n.sin(2. * mx.Universe.time)) / 2

        mx.on_particletime(ptype=ptype, invoke_method=update, period=0.01)


A = AType.get()
a = A(mx.Universe.center)

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
