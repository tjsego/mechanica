import mechanica as mx

mx.init()


class AType(mx.ParticleType):
    radius = 0.2


A = AType.get()

uc = mx.lattice.hex2d(1, A)

print(uc)

mx.lattice.create_lattice(uc, [6, 4, 6])

mx.step(100*mx.Universe.dt)


def test_pass():
    pass
