import mechanica as mx
import numpy as np

# the cell initial positions
# use the random_points function to make a group of points in the shape of a solid sphere
# parameters:
# count:  number of initial cells
count = 6000

# dr: thickness of epithelial cell sheet. This is a number beteen 0 and 1, smaller number
# make the initial sheet thinner, larger makes it thicker.
dr = 0.7

# epiboly_percent: percentage coverage of the yolk by epetheilial cells, betwen 0 and 1. Smaller
# numbers make the sheet smaller, larger means more initial coverage
epiboly_percent = 0.55

# percent of the *reamining* yolk that is covered by actin mesh
actin_percent = 1

# potential from yolk to other types. 'e' is the interaction strentgh, larger
# e will cause the other objects to get pulled in harder aganst the yolk
# The 'k' term here is the strength of the long-range harmonic attractin
# of the yolk and other objects, we use this longer range attractioin
# to model the effects of the external eveloping membrane
p_yolk = mx.Potential.glj(e=35, m=6, k=10, max=10)

# potential between the cell objects. Increasing e here makes the cell sheet want
# to pull in and clump up into a ball. Decreasing it makes the cell sheet softer,
# but also makes it easier to tear apparts.
#
# max (cutoff) distance here is very important, too large a value makes the
# cell sheet pull into itself and compress, but small lets it tear.

p_cell = mx.Potential.glj(e=10, m=1, n=2, max=1.5)

# potential between the actin and the cell. Can vary this parameter to see
# how strongly the actin mesh binds to the cells.
p_cellactin = mx.Potential.harmonic(k=50, r0=0.5, min=0.5, max=0.9)

# simple harmonic potential to pull particles
# strength of the key here determines how strongly the actin mesh pulls in
# if this is too high, it pulls in too fast, and tears the cell sheet,
# if this is too low, the mesh does not pull in fast enough.
p_actin = mx.Potential.harmonic(k=40, r0=0.001, max=5)

# dimensions of universe
dim = [30., 30., 40.]

mx.init(dim=dim,
        cutoff=5,
        integrator=mx.FORWARD_EULER,
        dt=0.001)


class CellType(mx.ParticleType):
    """
    The main developing embryo cell type, this is the initial mass of
    cells at the top.
    """

    radius = 0.5
    dynamics = mx.Overdamped
    mass = 5
    style = {"color": "MediumSeaGreen"}


class ActinType(mx.ParticleType):
    """
    Actin mesh particle types. The actin mesh / ring needs to be connected to
    nodes, and this is the node type for the actin mesh.
    """

    radius = 0.2
    dynamics = mx.Overdamped
    mass = 10
    style = {"color": "skyblue"}


class YolkType(mx.ParticleType):
    """
    The yolk type, only have a single instance of this
    """

    radius = 10
    frozen = True
    style = {"color": "orange"}


Cell = CellType.get()
Actin = ActinType.get()
Yolk = YolkType.get()

# Make some potentials (forces) to connect the different cell types together.

mx.bind.types(p_yolk, Cell, Yolk)
mx.bind.types(p_yolk, Actin, Yolk)
mx.bind.types(p_cell, Cell, Cell)
mx.bind.types(p_cellactin, Cell, Actin)

r = mx.Force.random(mean=0, std=20)

mx.bind.force(r, Cell)

# make a yolk at the center of the simulation domain
yolk = Yolk(mx.Universe.center)

# create the initial cell positions
cell_positions = [p * ((1 + dr/2) * Yolk.radius) + yolk.position for p in mx.random_points(mx.PointsType.SolidSphere,
                                                                                           6000,
                                                                                           dr=dr,
                                                                                           phi1=epiboly_percent * np.pi)]

# create an actin mesh that covers a section of a sphere.
parts, bonds = mx.bind.sphere(p_actin, type=Actin, n=3,
                              phi=(0.9 * epiboly_percent * np.pi,
                                   (epiboly_percent + actin_percent * (1 - epiboly_percent)) * np.pi),
                              radius=Yolk.radius + Actin.radius)

# create the initial cell cells
[Cell(p) for p in cell_positions]

# mx.bind_pairwise(h, c.neighbors(2* Cell.radius), cutoff=2*Cell.radius, pairs=[(Cell,Actin)])

# set visiblity on the different objects
Yolk.style.visible = True
Actin.style.visible = True
Cell.style.visible = True


# display function to write out values, any code can go here.
def update(e):
    print(Actin.items().center_of_mass)


# hook up the 'update' function to the on_time event to disply output.
mx.on_time(invoke_method=update, period=0.01)

# display the model (space bar starts / pauses the simulation)
mx.run()
