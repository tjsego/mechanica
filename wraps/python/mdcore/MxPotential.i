%{
    #include "MxPotential.h"
    #include <langs/py/MxPotentialPy.h>

%}
%rename(_call) MxPotential::operator();
%ignore MxPotential::custom;
%rename(custom) MxPotentialPy::customPy(double, double, PyObject*, PyObject*, PyObject*, double*, uint32_t*);

%include "MxPotential.h"
%include <langs/py/MxPotentialPy.h>

%extend MxPotential {
    %pythoncode %{
        from enum import Enum as EnumPy

        class Constants(EnumPy):
            degree = potential_degree
            chunk = potential_chunk
            ivalsa = potential_ivalsa
            ivalsb = potential_ivalsb
            N = potential_N
            align = potential_align
            ivalsmax = potential_ivalsmax

        class Flags(EnumPy):
            none = POTENTIAL_NONE
            lj126 = POTENTIAL_LJ126
            ewald = POTENTIAL_EWALD
            coulomb = POTENTIAL_COULOMB
            single = POTENTIAL_SINGLE
            r2 = POTENTIAL_R2
            r = POTENTIAL_R
            angle = POTENTIAL_ANGLE
            harmonic = POTENTIAL_HARMONIC
            dihedral = POTENTIAL_DIHEDRAL
            switch = POTENTIAL_SWITCH
            reactive = POTENTIAL_REACTIVE
            scaled = POTENTIAL_SCALED
            shifted = POTENTIAL_SHIFTED
            bound = POTENTIAL_BOUND
            psum = POTENTIAL_SUM
            periodic = POTENTIAL_PERIODIC
            coulombr = POTENTIAL_COULOMBR

        class Kind(EnumPy):
            potential = POTENTIAL_KIND_POTENTIAL
            dpd = POTENTIAL_KIND_DPD
            byparticles = POTENTIAL_KIND_BYPARTICLES
            combination = POTENTIAL_KIND_COMBINATION

        def __call__(self, *args):
            return self._call(*args)

        @property
        def min(self) -> float:
            """Minimum distance of evaluation"""
            return self.getMin()

        @property
        def max(self) -> float:
            """Maximum distance of evaluation"""
            return self.getMax()

        @property
        def cutoff(self) -> float:
            """Cutoff distance"""
            return self.getCutoff()

        @property
        def domain(self) -> (float, float):
            """Evaluation domain"""
            return self.getDomain()

        @property
        def intervals(self) -> int:
            """Evaluation intervals"""
            return self.getIntervals()

        @property
        def bound(self) -> bool:
            """Bound flag"""
            return self.getBound()

        @bound.setter
        def bound(self, bound: bool):
            self.setBound(bound)

        @property
        def r0(self) -> float:
            """Potential r0 value"""
            return self.getR0()

        @r0.setter
        def r0(self, r0: float):
            self.setR0(r0)

        @property
        def shifted(self) -> bool:
            """Shifted flag"""
            return self.getShifted()

        @property
        def periodic(self) -> bool:
            """Periodic flag"""
            return self.getPeriodic()

        @property
        def r_square(self) -> bool:
            """Potential r2 value"""
            return self.getRSquare()

        def plot(self, s=None, force=True, potential=False, show=True, ymin=None, ymax=None, *args, **kwargs):
            """Potential plot function"""
            import matplotlib.pyplot as plt
            import numpy as n
            import warnings

            min = kwargs["min"] if "min" in kwargs else 0.00001
            max = kwargs["max"] if "max" in kwargs else self.max
            step = kwargs["step"] if "step" in kwargs else 0.001
            range = kwargs["range"] if "range" in kwargs else (min, max, step)

            if isinstance(min, float) or isinstance(min, int):
                xx = n.arange(*range)
            else:
                t = 0
                xx = list()
                while t <= 1:
                    xx.append((min + (max - min) * t).asVector())
                    t += step

            yforce = None
            ypot = None

            if self.flags & POTENTIAL_SCALED or self.flags & POTENTIAL_SHIFTED:
                if not s:
                    warnings.warn("""plotting scaled function,
                    but no 's' parameter for sum of radii given,
                    using value of 1 as s""")
                    s = 1

                if force:
                    yforce = [self.force(x, s) for x in xx]

                if potential:
                    ypot = [self(x, s) for x in xx]

            else:

                if force:
                    yforce = [self.force(x) for x in xx]

                if potential:
                    ypot = [self(x) for x in xx]

            if not isinstance(xx[0], float):
                xx = [MxVector3f(xxx).length() for xxx in xx]

            if not ymin:
                y = n.array([])
                if yforce:
                    y = n.concatenate((y, n.asarray(yforce).flat))
                if ypot:
                    y = n.concatenate((y, ypot))
                ymin = n.amin(y)

            if not ymax:
                y = n.array([])
                if yforce:
                    y = n.concatenate((y, n.asarray(yforce).flat))
                if ypot:
                    y = n.concatenate((y, ypot))
                ymax = n.amax(y)

            yrange = n.abs(ymax - ymin)

            lines = None

            print("ymax: ", ymax, "ymin:", ymin, "yrange:", yrange)

            print("Ylim: ", ymin - 0.1 * yrange, ymax + 0.1 * yrange )

            plt.ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange )

            if yforce and not ypot:
                lines = plt.plot(xx, yforce, label='force')
            elif ypot and not yforce:
                lines = plt.plot(xx, ypot, label='potential')
            elif yforce and ypot:
                lines = [plt.plot(xx, yforce, label='force'), plt.plot(xx, ypot, label='potential')]

            plt.legend()

            plt.title(self.name)

            if show:
                plt.show()

            return lines

        def __reduce__(self):
            return MxPotentialPy.fromString, (self.toString(),)
    %}
}

%pythoncode %{
    Potential = MxPotentialPy
%}
