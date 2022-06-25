import mechanica as mx


# dimensions of universe
dim = [30., 30., 30.]

dist = 3.9

offset = 6

mx.init(dim=dim,
        cutoff=7,
        cells=[3, 3, 3],
        integrator=mx.FORWARD_EULER,
        dt=0.01,
        bc={'z': 'potential', 'x': 'potential', 'y': 'potential'})


class AType(mx.ParticleType):
    radius = 1
    dynamics = mx.Newtonian
    mass = 2.5
    style = {"color": "MediumSeaGreen"}


class SphereType(mx.ParticleType):
    radius = 3
    frozen = True
    style = {"color": "orange"}


A = AType.get()
Sphere = SphereType.get()

p = mx.Potential.morse(d=100, a=1, min=-3, max=4)

mx.bind.types(p, A, Sphere)

mx.bind.boundaryCondition(p, mx.Universe.boundary_conditions.bottom, A)
mx.bind.boundaryCondition(p, mx.Universe.boundary_conditions.top, A)
mx.bind.boundaryCondition(p, mx.Universe.boundary_conditions.left, A)
mx.bind.boundaryCondition(p, mx.Universe.boundary_conditions.right, A)
mx.bind.boundaryCondition(p, mx.Universe.boundary_conditions.front, A)
mx.bind.boundaryCondition(p, mx.Universe.boundary_conditions.back, A)


# above the sphere
Sphere(mx.Universe.center + [5, 0, 0])
A(mx.Universe.center + [5, 0, Sphere.radius + dist])

# bottom of simulation
A([mx.Universe.center[0], mx.Universe.center[1], dist])

# top of simulation
A([mx.Universe.center[0], mx.Universe.center[1], dim[2] - dist])

# left of simulation
A([dist, mx.Universe.center[1] - offset, mx.Universe.center[2]])

# right of simulation
A([dim[0] - dist, mx.Universe.center[1] + offset, mx.Universe.center[2]])

# front of simulation
A([mx.Universe.center[0], dist, mx.Universe.center[2]])

# back of simulation
A([mx.Universe.center[0], dim[1] - dist, mx.Universe.center[2]])

mx.run()
