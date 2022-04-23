"""
This example demonstrates constructing a lattice from a custom unit cell to simulate
fracture in a two-dimensional elastic sheet.
"""

import math
import mechanica as mx

mx.init(dim=[40, 20, 3])


class MtlType(mx.ParticleType):
    """Basic material type"""

    radius = 0.1


class BoundaryType(MtlType):
    """Material type with a zero-displacement condition along ``y``"""

    style = {'color': 'orange'}

    @staticmethod
    def apply_boundary(p: mx.ParticleHandle):
        p.frozen_y = True


class LoadedType(MtlType):
    """Material type on which an external load is applied"""

    style = {'color': 'darkgreen'}


mtl_type = MtlType.get()
boundary_type = BoundaryType.get()
loaded_type = LoadedType.get()

n_lattice = [60, 20]
a = 6 * mtl_type.radius
stiffness = 1E2
pot_border = mx.Potential.harmonic(k=stiffness, r0=a/2)
pot_cross = mx.Potential.harmonic(k=stiffness, r0=a/2*math.sqrt(2))
bcb_border = lambda i, j: mx.Bond.create(pot_border, i, j)
bcb_cross = lambda i, j: mx.Bond.create(pot_cross, i, j)

uc_sqcross = mx.lattice.unitcell(N=4,
                                 types=[mtl_type] * 4,
                                 a1=[a, 0, 0],
                                 a2=[0, a, 0],
                                 a3=[0, 0, 1],
                                 dimensions=2,
                                 position=[[0, 0, 0],
                                           [a/2, 0, 0],
                                           [0, a/2, 0],
                                           [a/2, a/2, 0]],
                                 bonds=[
                                     mx.lattice.BondRule(bcb_border, (0, 1), (0, 0, 0)),
                                     mx.lattice.BondRule(bcb_border, (0, 2), (0, 0, 0)),
                                     mx.lattice.BondRule(bcb_cross,  (0, 3), (0, 0, 0)),
                                     mx.lattice.BondRule(bcb_cross,  (1, 2), (0, 0, 0)),
                                     mx.lattice.BondRule(bcb_border, (1, 3), (0, 0, 0)),
                                     mx.lattice.BondRule(bcb_border, (2, 3), (0, 0, 0)),
                                     mx.lattice.BondRule(bcb_border, (1, 0), (1, 0, 0)),
                                     mx.lattice.BondRule(bcb_cross,  (1, 2), (1, 0, 0)),
                                     mx.lattice.BondRule(bcb_cross,  (3, 0), (1, 0, 0)),
                                     mx.lattice.BondRule(bcb_border, (3, 2), (1, 0, 0)),
                                     mx.lattice.BondRule(bcb_border, (2, 0), (0, 1, 0)),
                                     mx.lattice.BondRule(bcb_cross,  (2, 1), (0, 1, 0)),
                                     mx.lattice.BondRule(bcb_cross,  (3, 0), (0, 1, 0)),
                                     mx.lattice.BondRule(bcb_border, (3, 1), (0, 1, 0)),
                                     mx.lattice.BondRule(bcb_cross,  (3, 0), (1, 1, 0)),
                                     mx.lattice.BondRule(bcb_cross,  (2, 1), (-1, 1, 0))
                                 ])

parts = mx.lattice.create_lattice(uc_sqcross, n_lattice)

p_back, p_front = [], []
for i in range(2):
    p_front.extend([p[i] for p in parts[:, 0, :].flatten().tolist()])
for i in range(4):
    p_back.extend([p[i] for p in parts[:, n_lattice[1]-1, :].flatten().tolist()])

# Apply types

for p in p_front:
    p.become(boundary_type)
    BoundaryType.apply_boundary(p)
for p in p_back:
    p.become(loaded_type)

# Apply fracture criterion to material type

mtl_ids = [p.id for p in mtl_type.parts]
for p in mtl_type.items():
    for b in p.bonds:
        p1id, p2id = b.parts
        if p1id in mtl_ids and p2id in mtl_ids:
            b.dissociation_energy = 1E-2

# Apply force with damping

f_load = mx.ConstantForce([0, 2, 0])
f_friction = mx.Force.friction(coef=1000.0)
mx.bind.force(f_load + f_friction, loaded_type)

mx.system.cameraViewTop()

mx.show()
