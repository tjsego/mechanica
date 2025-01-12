import mechanica as mx

# potential cutoff distance
cutoff = 1

# new simulator
mx.init(dim=[20., 20., 20.])

pot = mx.Potential.morse(d=1, a=5, r0=0.6, min=0, max=4, shifted=False)


class CellType(mx.ParticleType):
    mass = 20
    target_temperature = 0
    radius = 0.5


Cell = CellType.get()


# Callback for time- and particle-dependent events
def fission(e: mx.ParticleTimeEvent):
    e.targetParticle.fission()


mx.on_particletime(ptype=Cell, period=1, invoke_method=fission, distribution='exponential')

mx.bind.types(pot, Cell, Cell)

Cell([10., 10., 10.])

# run the simulator interactive
mx.run()
