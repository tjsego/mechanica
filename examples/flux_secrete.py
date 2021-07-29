import mechanica as mx

mx.init(dim=[6.5, 6.5, 6.5], bc=mx.FREESLIP_FULL)


class AType(mx.ParticleType):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": "auto"}}


class BType(mx.ParticleType):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap": {"species": "S1", "map": "rainbow", "range": "auto"}}


A = AType.get()
B = BType.get()

mx.Fluxes.flux(A, A, "S1", 5, 0.005)

uc = mx.lattice.sc(0.25, A)

parts = mx.lattice.create_lattice(uc, [25, 25, 25])

# grap the particle at the top cornder
o = parts[24, 0, 24][0]

print("secreting pos: ", o.position.as_list())

# change type to B, since there is no flux rule between A and B
o.become(B)


def spew(event: mx.ParticleTimeEvent):

    print("spew...")

    # reset the value of the species
    # secrete consumes material...
    event.targetParticle.species.S1 = 500
    event.targetParticle.species.S1.secrete(250, distance=1)


mx.on_particletime(ptype=B, invoke_method=spew, period=0.3)
mx.run()
