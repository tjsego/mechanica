import pickle
from typing import List
import mechanica as mx


def validate_copy(obj0, obj1, attr_list: List[str]):
    for attr in attr_list:
        v0 = getattr(obj0, attr)
        v1 = getattr(obj1, attr)
        if v0 != v1:
            if isinstance(v0, float):
                num = abs(v0 - v1)
                den = max(abs(v0), abs(v1))
                if den < 1E-12:
                    den = 1.0
                if num / den < 1E-6:
                    return

            print('Difference found:', attr)
            print('\tValue 0:', v0)
            print('\tValue 1:', v1)
            raise ValueError


def validate(obj, attr_list: List[str]):
    validate_copy(obj, pickle.loads(pickle.dumps(obj)), attr_list)


mx.init(bc={'z': 'potential', 'x': 'potential', 'y': 'potential'}, windowless=True)


class AType(mx.ParticleType):
    style = {"colormap": {"species": "S1",
                          "map": "rainbow",
                          "range": (0, 1)}}
    species = ['S1']


A = AType.get()


class BType(mx.ClusterType):
    types = [A]
    style = {"color": "MediumSeaGreen"}


B = BType.get()

# test particle
particle = A().part()
validate(particle, ['clusterId',
                    'flags',
                    'id',
                    'imass',
                    'mass',
                    'nr_parts',
                    'q',
                    'radius',
                    'typeId'])

# test cluster
b = B()
cluster_constituent = b(A).part()
cluster_particle = b.cluster()
validate(b.part(), ['clusterId',
                    'flags',
                    'id',
                    'imass',
                    'mass',
                    'q',
                    'radius',
                    'typeId'])

# test particle type
validate(A, ['frozen',
             'frozen_x',
             'frozen_y',
             'frozen_z',
             'temperature',
             'target_temperature',
             'mass',
             'charge',
             'radius',
             'kinetic_energy',
             'potential_energy',
             'target_energy',
             'minimum_radius',
             'dynamics',
             'name'])

# test cluster type
validate(B, ['frozen',
             'frozen_x',
             'frozen_y',
             'frozen_z',
             'temperature',
             'target_temperature',
             'mass',
             'charge',
             'radius',
             'kinetic_energy',
             'potential_energy',
             'target_energy',
             'minimum_radius',
             'dynamics',
             'name'])

# validate particle constructor from string on particle type
particle_clone = A(particle.toString()).part()
validate_copy(particle, particle_clone, ['clusterId',
                                         'flags',
                                         'imass',
                                         'mass',
                                         'nr_parts',
                                         'q',
                                         'radius',
                                         'typeId'])
if particle_clone.id < 0:
    raise ValueError
elif particle_clone.id == particle.id:
    raise ValueError

# validate particle constructor from string on cluster
cluster_constituent_clone = b(A, cluster_constituent.toString()).part()
validate_copy(cluster_constituent, cluster_constituent_clone, ['clusterId',
                                                               'flags',
                                                               'imass',
                                                               'mass',
                                                               'q',
                                                               'radius',
                                                               'typeId'])
if cluster_constituent_clone.id < 0:
    raise ValueError
elif cluster_constituent_clone.id == cluster_constituent.id:
    raise ValueError

# test particle list
validate(b.items(), ['radius_of_gyration', 'nr_parts'])

# test particle type list
validate(B.types, ['radius_of_gyration', 'nr_parts'])

# test regular potential
pot_bb = mx.Potential.soft_sphere(kappa=0.2, epsilon=0.05, r0=0.2, eta=4, tol=0.01, min=0.01, max=0.5)
validate(pot_bb, ['min',
                  'max',
                  'cutoff',
                  'intervals',
                  'bound',
                  'shifted',
                  'periodic',
                  'r_square',
                  'name',
                  'domain',
                  'n'])

pot_dpd = mx.Potential.dpd(alpha=1.0, gamma=2.0, sigma=3.0)
validate(mx.DPDPotential.fromPot(pot_dpd), ['alpha', 'gamma', 'sigma'])

# test random force
rforce = mx.Force.random(mean=0, std=50)
validate(mx.Gaussian.fromForce(rforce), ['mean', 'std', 'durration_steps'])

# test tstat force
tforce = mx.Force.berendsen_tstat(10)
validate(mx.Berendsen.fromForce(tforce), ['itau'])

# test friction force
fforce = mx.Force.friction(1.0)
validate(mx.Friction.fromForce(fforce), ['coef'])

# test bond
p0 = A()
p1 = A()
bh: mx.BondHandle = mx.Bond.create(pot_bb, p0, p1, half_life=1.0, dissociation_energy=1000.0)
validate(bh.get(), ['id',
                    'i',
                    'j',
                    'creation_time',
                    'half_life',
                    'dissociation_energy'])

# test angle
p2 = A()
ah: mx.AngleHandle = mx.Angle.create(pot_bb, p0, p1, p2)
ah.half_life = 1.0
ah.dissociation_energy = 1000.0
validate(ah.get(), ['i',
                    'j',
                    'k',
                    'creation_time',
                    'half_life',
                    'dissociation_energy'])

# test dihedral
p3 = A()
dh: mx.DihedralHandle = mx.Dihedral.create(pot_bb, p0, p1, p2, p3)
dh.half_life = 1.0
dh.dissociation_energy = 1000.0
validate(dh.get(), ['i',
                    'j',
                    'k',
                    'l',
                    'creation_time',
                    'half_life',
                    'dissociation_energy'])

# test species
validate(p0.species.species.S1, ['boundary_condition',
                                 'charge',
                                 'compartment',
                                 'constant',
                                 'conversion_factor',
                                 'name',
                                 'substance_units',
                                 'units'])

# test state vector
validate(p0.species, ['q', 'size'])

# test style
validate(A.style, ['visible'])

# test boundary conditions
mx.bind.boundaryCondition(pot_bb, mx.Universe.boundary_conditions.bottom, A)
mx.bind.boundaryCondition(pot_bb, mx.Universe.boundary_conditions.top, A)
mx.bind.boundaryCondition(pot_bb, mx.Universe.boundary_conditions.left, A)
mx.bind.boundaryCondition(pot_bb, mx.Universe.boundary_conditions.right, A)
mx.bind.boundaryCondition(pot_bb, mx.Universe.boundary_conditions.front, A)
mx.bind.boundaryCondition(pot_bb, mx.Universe.boundary_conditions.back, A)
import os
pid = os.getpid()
bc_new = pickle.loads(pickle.dumps(mx.Universe.boundary_conditions))
for side_name in ['bottom', 'top', 'left', 'right', 'front', 'back']:
    validate_copy(getattr(mx.Universe.boundary_conditions, side_name),
                  getattr(bc_new, side_name),
                  ['id', 'kind', 'kind_str', 'name', 'radius', 'restore'])


def test_pass():
    pass
