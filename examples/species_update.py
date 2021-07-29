import mechanica as m
import numpy as n

m.init()


class AType(m.ParticleType):

    radius = 5

    species = ['S1', 'S2', 'S3']

    style = {"colormap": {"species": "S1", "map": "rainbow", "range": "auto"}}

    @staticmethod
    def on_register(ptype):
        def update(event: m.ParticleTimeEvent):
            for p in ptype.items():
                p.species.S1 = (1 + n.sin(2. * m.Universe.time))/2

        m.on_particletime(ptype=ptype, invoke_method=update, period=0.01)


A = AType.get()
a = A(m.Universe.center)

m.run()
