import mechanica as mx
import numpy as np

cutoff = 1

mx.init(dim=[10., 10., 10.])


class ArgonType(mx.ParticleType):
    mass = 39.4
    target_temperature = 100


Argon = ArgonType.get()


# hook up the destroy method on the Argon type to the
# on_time event
def destroy(event: mx.ParticleTimeEvent):
    if event.targetParticle:
        print('destroy....')
        event.targetParticle.destroy()
        print('destroy:', len(event.targetType.items()))


mx.on_particletime(ptype=Argon, invoke_method=destroy, period=1, distribution='exponential')

pot = mx.Potential.lennard_jones_12_6(0.275, cutoff, 9.5075e-06, 6.1545e-03, 1.0e-3)
mx.bind.types(pot, Argon, Argon)

tstat = mx.Force.berenderson_tstat(10)

mx.bind.force(tstat, Argon)

size = 100

# uniform random cube
positions = np.random.uniform(low=0, high=10, size=(size, 3))
velocities = np.random.normal(0, 0.2, size=(size, 3))

for pos, vel in zip(positions, velocities):
    # calling the particle constructor implicitly adds
    # the particle to the universe
    Argon(pos, vel)

# run the simulator interactive
mx.run()
