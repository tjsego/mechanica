import mechanica as mx

cutoff = 8
count = 3

# dimensions of universe
dim = [20., 20., 20.]

mx.init(dim=dim, cutoff=cutoff)


class BType(mx.ParticleType):
    mass = 1
    dynamics = mx.Overdamped


B = BType.get()
# make a glj potential, this automatically reads the
# particle radius to determine rest distance.
pot = mx.Potential.glj(e=1)

mx.bind.types(pot, B, B)

p1 = B(mx.Universe.center + (-2, 0, 0))
p2 = B(mx.Universe.center + (2, 0, 0))
p1.radius = 1
p2.radius = 2

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
