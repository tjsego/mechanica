"""
This example demonstrates basic usage of boundary 'reset' conditions.

In 'reset' conditions, the initial concentration of a species is restored for a particle that crosses a
restoring boundary.
"""

import mechanica as mx

# Initialize a domain like a tunnel, with flow along the x-direction
mx.init(dim=[20, 10, 10],
        cells=[5, 5, 5],
        cutoff=5,
        bc={'x': ('periodic', 'reset'), 'y': 'free_slip', 'z': 'free_slip'})

# Make a clip plane through the middle of the domain to better visualize transport
mx.ClipPlanes.create(mx.Universe.center, mx.MxVector3f(0, 1, 0))


class CarrierType(mx.ParticleType):
    """A particle type to carry stuff"""

    radius = 0.5
    mass = 0.1
    species = ['S1']
    style = {'colormap': {'species': 'S1', range: (0, 1)}}


class SinkType(mx.ParticleType):
    """A particle type to absorb stuff"""

    frozen = True
    radius = 1.0
    species = ['S1']
    style = {'colormap': {'species': 'S1', range: (0, 1)}}


carrier_type, sink_type = CarrierType.get(), SinkType.get()

# Carrier type begins carrying stuff
carrier_type.species.S1.initial_concentration = 1.0
# Sink type absorbs stuff by acting as a void
sink_type.species.S1.constant = True

# Carrier type like a fluid
dpd = mx.Potential.dpd(alpha=1, gamma=1, sigma=0.1, cutoff=3 * CarrierType.radius)
mx.bind.types(dpd, carrier_type, carrier_type)

# Sink type like a barrier
rep = cp = mx.Potential.harmonic(k=1000,
                                 r0=carrier_type.radius + 1.05 * sink_type.radius,
                                 min=carrier_type.radius + sink_type.radius,
                                 max=carrier_type.radius + 1.05 * sink_type.radius,
                                 tol=0.001)
mx.bind.types(rep, carrier_type, sink_type)

# Flow to the right
force = mx.ConstantForce([0.01, 0, 0])
mx.bind.force(force, carrier_type)

# Diffusive transport and high flux into sink type
mx.Fluxes.flux(carrier_type, carrier_type, "S1", 0.001)
mx.Fluxes.flux(carrier_type, sink_type, "S1", 0.5)

# Put a sink at the center and carrier types randomly, though not in the sink
st = sink_type(mx.Universe.center)
[carrier_type() for _ in range(2000)]
to_destroy = []
for p in carrier_type.items():
    if p.relativePosition(mx.Universe.center).length() < (sink_type.radius + carrier_type.radius) * 1.1:
        to_destroy.append(p)
[p.destroy() for p in to_destroy]

# Run it!
mx.run()
