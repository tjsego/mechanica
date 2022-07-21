import mechanica as mx

mx.init(windowless=True)


class BeadType(mx.ParticleType):
    species = ['S1']
    radius = 3

    style = {"colormap": {"species": "S1", "map": "rainbow"}}

    def __init__(self, pos, value):
        super().__init__(pos)
        self.species.S1 = value


Bead = BeadType.get()

# make a ring of of 50 particles
pts = [x * 4 + mx.Universe.center for x in mx.points(mx.PointsType.Ring, 100)]

# constuct a particle for each position, make
# a list of particles
beads = [Bead(p) for p in pts]

Bead.i = 0


def keypress(e):
    names = mx.ColorMap.names
    name = None

    if e.key_name == "n":
        Bead.i = (Bead.i + 1) % len(names)
        name = names[Bead.i]
        print("setting colormap to: ", name)

    elif e.key_name == "p":
        Bead.i = (Bead.i - 1) % len(names)
        name = names[Bead.i]
        print("setting colormap to: ", name)

    if name:
        Bead.style.colormap = name


mx.on_keypress(keypress)

# run the model
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
