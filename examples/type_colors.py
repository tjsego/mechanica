import mechanica as mx
import numpy as np

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
mx.init(dim=dim)

# loop over arays for x/y coords
for x in np.arange(0., 20., 2.):
    for y in np.arange(0., 20., 2.):

        # create a new particle type, chooses next default color
        class PType(mx.ParticleType):
            radius = 2

        # instantiate that type
        P = PType.get()
        P([x+1.5, y+1.5, 10.])

# run the simulator interactive
mx.run()
