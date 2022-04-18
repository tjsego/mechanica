import mechanica as mx

# potential cutoff distance
cutoff = 1

# dimensions of universe
dim = [10., 10., 10.]

# new simulator
mx.init(dim=dim, windowless=True)


class MyCellType(mx.ParticleType):

    mass = 39.4
    target_temperature = 50
    radius = 0.2


MyCell = MyCellType.get()

# create a potential representing a 12-6 Lennard-Jones potential
# A The first parameter of the Lennard-Jones potential.
# B The second parameter of the Lennard-Jones potential.
# cutoff
pot = mx.Potential.lennard_jones_12_6(0.275, cutoff, 9.5075e-06, 6.1545e-03, 1.0e-3)


# bind the potential with the *TYPES* of the particles
mx.bind.types(pot, MyCell, MyCell)

# create a thermostat, coupling time constant determines how rapidly the
# thermostat operates, smaller numbers mean thermostat acts more rapidly
tstat = mx.Force.berendsen_tstat(10)

# bind it just like any other force
mx.bind.force(tstat, MyCell)


# create a new particle every 0.05 time units. The 'on_particletime' function
# binds the MyCell object and callback 'split' with the event, and is called
# at periodic intervals based on the exponential distribution,
# so the mean time between particle creation is 0.05
def split(event: mx.ParticleTimeEvent):
    if event.targetParticle is None:
        event.targetType(mx.Universe.center)
    else:
        p = event.targetParticle.split()
        p.radius = event.targetType.radius
        event.targetParticle.radius = event.targetType.radius
    print('split:', len(event.targetType.items()))


mx.on_particletime(ptype=MyCell, invoke_method=split, period=0.05, distribution="exponential")

# run the simulator interactive
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
