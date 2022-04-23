"""
Reproduces cylindrical symmetry demonstrated in Figure 4 of

    Nissen, Silas Boye, et al.
    "Theoretical tool bridging cell polarities with development of robust morphologies."
    Elife 7 (2018): e38407.
"""
import mechanica as mx
from mechanica.models.center.cell_polarity import CellPolarity
import numpy as np

mx.init(dim=[10., 10., 10.], dt=1.0)


class PolarType(mx.ParticleType):
    dynamics = mx.Overdamped
    radius = 0.25


polar_type = PolarType.get()
CellPolarity.load()
CellPolarity.setArrowScale(0.25)
CellPolarity.setArrowLength(polar_type.radius)
CellPolarity.registerType(pType=polar_type)

f_random = mx.Force.random(1E-3, 0)
mx.bind.force(f_random, polar_type)

pot_contact = mx.Potential.morse(d=5E-4, a=5, max=1.0)
pot_polar = CellPolarity.potentialContact(cutoff=2.5*polar_type.radius, mag=2.5E-3, rate=0,
                                          distanceCoeff=10.0*polar_type.radius, couplingFlat=1.0)
mx.bind.types(pot_contact + pot_polar, polar_type, polar_type)

for pos in mx.random_points(mx.PointsType.SolidSphere, 500, 5.0):
    p = polar_type(position=pos + mx.Universe.center, velocity=mx.MxVector3f(0.0))
    CellPolarity.registerParticle(p)
    # Assign cylindrical polarity orientations
    pos_rel = p.position - mx.Universe.center
    ang = np.arctan2(pos_rel[1], pos_rel[0])
    pvec_ab = mx.MxVector3f(np.cos(ang), np.sin(ang), 0.0)
    pvec_pcp = mx.MxVector3f(np.cos(ang+np.pi/2), np.sin(ang+np.pi/2), 0.0)

    CellPolarity.setVectorAB(p.id, pvec_ab, init=True)
    CellPolarity.setVectorPCP(p.id, pvec_pcp, init=True)

mx.show()
