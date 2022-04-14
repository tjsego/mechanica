import mechanica as mx

# potential cutoff distance
cutoff = 1

# new simulator
mx.init(dim=[20., 20., 20.])


pot = mx.Potential.soft_sphere(kappa=10, epsilon=0.1, r0=0.6, eta=3, tol=0.1, min=0.05, max=4)


class CellType(mx.ParticleType):
    mass = 20
    target_temperature = 0
    radius = 0.5
    # events = [mx.on_time(mx.Particle.fission, period=1, distribution='exponential')]
    dynamics = mx.Overdamped

    @staticmethod
    def on_register(ptype):
        def fission(event: mx.ParticleTimeEvent):
            event.targetParticle.fission()

            print('fission:', len(event.targetType.items()))

        mx.on_particletime(ptype=ptype, invoke_method=fission, period=1, distribution='exponential')


Cell = CellType.get()

mx.Universe.bind(pot, Cell, Cell)

rforce = mx.Force.random(mean=0, std=0.5)

mx.bind.force(rforce, Cell)

Cell([10., 10., 10.])

# run the simulator interactive
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
