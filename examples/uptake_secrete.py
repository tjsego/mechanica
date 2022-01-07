import mechanica as mx

mx.init(dim=[6.5, 6.5, 6.5], bc=mx.FREESLIP_FULL)


class AType(mx.ParticleType):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": (0, 1)}}


class ProducerType(mx.ParticleType):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": (0, 1)}}


class ConsumerType(mx.ParticleType):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": (0, 1)}}


A, Producer, Consumer = AType.get(), ProducerType.get(), ConsumerType.get()

# define fluxes between objects types
mx.Fluxes.flux(A, A, "S1", 5, 0)
mx.Fluxes.secrete(Producer, A, "S1", 5, 0)
mx.Fluxes.uptake(A, Consumer, "S1", 10, 500)

# make a lattice of objects
uc = mx.lattice.sc(0.25, A)
parts = mx.lattice.create_lattice(uc, [25, 25, 1])

# grap the left part
left = parts[0, 12, 0][0]

# grab the right part
right = parts[24, 12, 0][0]

# change types
left.become(Producer)
right.become(Consumer)

left.species.S1 = 200  # set initial condition

mx.run()
