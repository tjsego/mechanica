%{
    #include "MxPotential.h"

%}
%rename(_call) MxPotential::operator();

%include "MxPotential.h"

%extend MxPotential {
    %pythoncode %{
        def __call__(self, r: float, r0: float = -1.0):
            return self._call(r, r0)

        @property
        def min(self) -> float:
            return self.getMin()

        @property
        def max(self) -> float:
            return self.getMax()

        @property
        def cutoff(self) -> float:
            return self.getCutoff()

        @property
        def domain(self) -> (float, float):
            return self.getDomain()

        @property
        def intervals(self) -> int:
            return self.getIntervals()

        @property
        def bound(self) -> bool:
            return self.getBound()

        @bound.setter
        def bound(self, bound: bool):
            self.setBound(bound)

        @property
        def r0(self) -> float:
            return self.getR0()

        @r0.setter
        def r0(self, r0: float):
            self.setR0(r0)

        @property
        def shifted(self) -> bool:
            return self.getShifted()

        @shifted.setter
        def shifted(self, shifted: bool):
            self.setShifted(shifted)

        @property
        def r_square(self) -> bool:
            return self.getRSquare()

        @r_square.setter
        def r_square(self, r_square: bool):
            self.setRSquare(r_square)

        def plot(self, s=None, force=True, potential=False, show=True, ymin=None, ymax=None, *args, **kwargs):
            import matplotlib.pyplot as plt
            import numpy as n
            import warnings

            min = kwargs["min"] if "min" in kwargs else 0.00001
            max = kwargs["max"] if "max" in kwargs else self.max
            step = kwargs["step"] if "step" in kwargs else 0.001
            range = kwargs["range"] if "range" in kwargs else (min, max, step)

            xx = n.arange(*range)

            yforce = None
            ypot = None

            if self.flags & POTENTIAL_SCALED or self.flags & POTENTIAL_SHIFTED:
                if not s:
                    warnings.warn("""plotting scaled function,
                    but no 's' parameter for sum of radii given,
                    using value of 1 as s""")
                    s = 1

                if force:
                    yforce = [self(x, s)[1] for x in xx]

                if potential:
                    ypot = [self(x, s)[0] for x in xx]

            else:

                if force:
                    yforce = [self(x)[1] for x in xx]

                if potential:
                    ypot = [self(x)[0] for x in xx]

            if not ymin:
                y = n.array([])
                if yforce:
                    y = n.concatenate((y, yforce))
                if ypot:
                    y = n.concatenate((y, ypot))
                ymin = n.amin(y)

            if not ymax:
                y = n.array([])
                if yforce:
                    y = n.concatenate((y, yforce))
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
    %}
}

%pythoncode %{
    Potential = MxPotential
%}
// todo: ensure PotentialFlags is being wrapped and all enums are available
