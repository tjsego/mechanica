import mechanica as mx
import csv
import os


# set the output file name here,
# when running in Python windowed mode, working directory can do
# strange things.
fname = os.path.join(os.getcwd(), "positions.csv")

print(fname)


# potential cutoff distance
cutoff = 1

# dimensions of universe
dim = [10., 10., 10.]

# new simulator
mx.init(dim=dim)


class MyCellType(mx.ParticleType):

    mass = 39.4
    target_temperature = 50
    radius = 0.2


MyCell = MyCellType.get()


# create a potential representing a 12-6 Lennard-Jones potential
# A The first parameter of the Lennard-Jones potential.
# B The second parameter of the Lennard-Jones potential.
# cutoff
pot = mx.Potential.lennard_jones_12_6(0.275, cutoff, 9.5075e-06, 6.1545e-03, 1.0e-3)


# bind the potential with the *TYPES* of the particles
mx.bind.types(pot, MyCell, MyCell)

# create a thermostat, coupling time constant determines how rapidly the
# thermostat operates, smaller numbers mean thermostat acts more rapidly
tstat = mx.Force.berendsen_tstat(10)

# bind it just like any other force
mx.bind.force(tstat, MyCell)


# create a new particle every 0.05 time units. The 'on_time' event
# here binds the constructor of the MyCell object with the event, and
# calls at periodic intervals based on the exponential distribution,
# so the mean time between particle creation is 0.05
def write_data(event: mx.TimeEvent):
    time = mx.Universe.time
    print("time is now: ", time)

    positions = [list(p.position) for p in mx.Universe.particles()]

    print(positions)

    with open(fname, "a") as f:
        writer = csv.writer(f)
        writer.writerow([time, positions])

    print("wrote positions...")


mx.on_time(invoke_method=write_data, period=0.05)


# run the simulator interactive
mx.run()
