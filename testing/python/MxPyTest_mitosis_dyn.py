import mechanica as mx

# potential cutoff distance
cutoff = 1

# new simulator
mx.init(dim=[20., 20., 20.], windowless=True)

pot = mx.Potential.soft_sphere(kappa=10, epsilon=0.1, r0=0.6, eta=3, tol=0.1, min=0.05, max=4)


class CellType(mx.ParticleType):
    mass = 20
    target_temperature = 0
    radius = 0.5


Cell = CellType.get()


def fission(event: mx.ParticleTimeEvent):
    event.targetParticle.fission()
    print('fission: ', len(event.targetType.items()))


mx.on_particletime(ptype=Cell, invoke_method=fission, period=1, distribution='exponential')

mx.bind.types(pot, Cell, Cell)

Cell([10., 10., 10.])

# run the simulator interactive
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
