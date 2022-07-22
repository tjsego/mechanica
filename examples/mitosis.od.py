import mechanica as mx

# potential cutoff distance
cutoff = 1

# new simulator
mx.init(dim=[20., 20., 20.])


pot = mx.Potential.morse(d=0.1, a=6, min=-1, max=1)


class CellType(mx.ParticleType):
    mass = 20
    target_temperature = 0
    radius = 0.5
    dynamics = mx.Overdamped

    @staticmethod
    def on_register(ptype):
        def fission(event: mx.ParticleTimeEvent):
            m = event.targetParticle
            d = m.fission()
            m.radius = d.radius = CellType.radius
            m.mass = d.mass = CellType.mass

            print('fission:', len(event.targetType.items()))

        mx.on_particletime(ptype=ptype, invoke_method=fission, period=1, distribution='exponential')


Cell = CellType.get()

mx.Universe.bind(pot, Cell, Cell)

rforce = mx.Force.random(mean=0, std=0.5)

mx.bind.force(rforce, Cell)

Cell([10., 10., 10.])

# run the simulator interactive
mx.run()
