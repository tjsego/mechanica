"""
This example demonstrates how Mechanica objects can be serialized for use in multiprocessing applications.

Special care must be taken to account for that deserialized Mechanica objects are copies of their original
object, and that the Mechanica engine is not available in separate processes. As such, calls to methods that
require the engine in a spawned process will fail.
"""
import mechanica as mx
from multiprocessing import Pool


def calc_energy_diff(bond: mx.Bond):
    # This call will fail in a spawned process, since a handle requires the engine
    # bh = mx.BondHandle(bond.id)

    # This call is ok in a spawned process, since accessed members are just values
    result = bond.dissociation_energy - bond.potential_energy
    return result


# Protect the entry point using __main__ for safe multiprocessing
if __name__ == '__main__':
    mx.init()
    # Get the default particle type
    ptype = mx.ParticleType.get()
    # Construct some bonds
    pot = mx.Potential.harmonic(k=1.0, r0=1.0)
    [mx.Bond.create(potential=pot, i=ptype(), j=ptype(), dissociation_energy=10.0) for _ in range(1000)]
    # Do a step
    mx.step()
    # Do calculations in 8 processes
    with Pool(8) as p:
        diff = p.map(calc_energy_diff, [bh.get() for bh in mx.Universe.bonds()])
