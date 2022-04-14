import mechanica as mx
import numpy as n

mx.init()

print(mx.Species("S1"))

s1 = mx.Species("$S1")

s1 = mx.Species("const S1")

s1 = mx.Species("const $S1")

s1 = mx.Species("S1 = 1")

s1 = mx.Species("const S1 = 234234.5")


class AType(mx.ParticleType):

    species = ['S1', 'S2', 'S3']

    style = {"colormap": {"species": "S1", "map": "rainbow"}}

    @staticmethod
    def on_register(ptype):
        def update(event: mx.ParticleTimeEvent):
            for p in ptype.items():
                p.species.S1 = (1 + n.sin(2. * mx.Universe.time)) / 2

        mx.on_particletime(ptype=ptype, invoke_method=update, period=0.01)


A = AType.get()


print("A.species:")
print(A.species)

print("making f")
a = A()

print("f.species")
print(a.species)

print("A.species.S1: ", A.species.S1)

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
