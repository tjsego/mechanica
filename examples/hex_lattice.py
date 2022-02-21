import mechanica as mx

mx.init()


class AType(mx.ParticleType):
    radius = 0.2


A = AType.get()

uc = mx.lattice.hex(1, A)

print(uc)

mx.lattice.create_lattice(uc, [6, 4, 6])

mx.show()
