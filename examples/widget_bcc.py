import mechanica as mx
from ipywidgets import widgets

mx.init(windowless=True, window_size=[1024, 1024])


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

w = widgets.Image(value=mx.system.image_data(), width=600)

display(w)
