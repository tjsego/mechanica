import mechanica as mx
import numpy as np

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
mx.init(dim=dim, windowless=True)


class PType(mx.ParticleType):
    radius = 2


P = PType.get()

# Get all available color names
color3_names = mx.color3_names()

# loop over arays for x/y coords
for x in np.arange(0., 20., 2.):
    for y in np.arange(0., 20., 2.):

        # create and register a new particle type based on PType
        # all we need is a unique name to create a type on the fly
        PP = P.newType(f'PP{mx.Universe.num_types}')
        PP.registerType()

        # instantiate that type
        PP = PP.get()
        # set a new style, since the current style is the same as PType
        PP.style = mx.MxStyle(color3_names[np.random.randint(len(color3_names))])
        PP([x+1.5, y+1.5, 10.])

# run the simulator interactive
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
