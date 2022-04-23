import mechanica as mx

mx.init(windowless=True)


class NaType(mx.ParticleType):
    radius = 0.4
    style = {"color": "orange"}


class ClType(mx.ParticleType):
    radius = 0.25
    style = {"color": "spablue"}


Na, Cl = NaType.get(), ClType.get()

uc = mx.lattice.bcc(0.9, [Na, Cl])

mx.lattice.create_lattice(uc, [10, 10, 10])


def test_pass():
    pass
