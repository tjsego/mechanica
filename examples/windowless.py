import mechanica as mx

mx.init(windowless=True,
        window_size=[1024, 1024],
        clip_planes=[([2, 2, 2], [1, 1, 0]), ([5, 5, 5], [-1, -1, 0])])

print(mx.system.gl_info())


class NaType(mx.ParticleType):
    radius = 0.4
    style = {"color": "orange"}


class ClType(mx.ParticleType):
    radius = 0.25
    style = {"color": "spablue"}


Na = NaType.get()
Cl = ClType.get()

uc = mx.lattice.bcc(0.9, [Na, Cl])

mx.lattice.create_lattice(uc, [10, 10, 10])

# mx.system.image_data() is a jpg byte stream of the
# contents of the frame buffer.

with open('system.jpg', 'wb') as f:
    f.write(mx.system.image_data())
