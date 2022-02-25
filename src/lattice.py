# Coypright (c) 2020 Andy Somogyi (somogyie at indiana dot edu)
# this is a port of the HOOMMD unit cell code
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# original Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
"""
Defines lattices.
"""

import numpy
import math
from typing import Callable, List, Tuple, Union
from collections import namedtuple

import mechanica as m


# make a types vector of the requested size
def _make_types(n, types):

    try:
        if len(types):
            return types
    except TypeError:
        pass

    if types is None:
        return [m.MxParticleType] * n

    return [types] * n


_BondRule = namedtuple('_BondRule', ['func', 'part_ids', 'cell_offset'])
"""
hold bond rule info,

*func: function of func(p1, p2) that accepts two particle handles and
       returns a bond.

*parts: pair of particle ids in current current and other unit cell
        must be tuple.

*cell_offset: offset vector of other unit cell relative to current
        unit cell. Must be a tuple
"""


class unitcell(object):
    """
    A unit cell

    A unit cell is a box definition (*a1*, *a2*, *a3*, *dimensions*), and particle properties
    for *N* particles. You do not need to specify all particle properties. Any property omitted
    will be initialized to defaults. The function :py:func:`create_lattice` initializes the system with
    many copies of a unit cell.

    :py:class:`unitcell` is a completely generic unit cell representation. See other classes in
    the :py:mod:`lattice` module for convenience wrappers for common lattices.

    Example::

        uc = lattice.unitcell(N=2,
                              a1=[1, 0, 0],
                              a2=[0.2, 1.2, 0],
                              a3=[-0.2, 0, 1.0],
                              dimensions=3,
                              position=[[0, 0, 0], [0.5, 0.5, 0.5]],
                              types=[A, B])

    Note:
        *a1*, *a2*, *a3* must define a right handed coordinate system.
    """

    def __init__(self,
                 N: int,
                 a1: List[float],
                 a2: List[float],
                 a3: List[float],
                 dimensions: int = 3,
                 position: List[List[float]] = None,
                 types: List[m.MxParticleType] = None,
                 bonds: Tuple[_BondRule] = None):
        """

        :param N: Number of particles in the unit cell.
        :type N: int
        :param a1: Lattice vector (3-vector).
        :type a1: List[float]
        :param a2: Lattice vector (3-vector).
        :type a2: List[float]
        :param a3: Lattice vector (3-vector).
        :type a3: List[float]
        :param dimensions: Dimensionality of the lattice (2 or 3).
        :type dimensions: int
        :param position: List of particle positions.
        :type position: List[List[float]]
        :param types: List of particle types
        :type types: List[m.MxParticleType]
        :param bonds: bond constructors rules
        :type bonds: Tuple[_BondRule]
        """

        self.N = N
        self.a1 = numpy.asarray(a1, dtype=numpy.float64)
        self.a2 = numpy.asarray(a2, dtype=numpy.float64)
        self.a3 = numpy.asarray(a3, dtype=numpy.float64)
        self.dimensions = dimensions
        self.bonds = bonds

        if position is None:
            self.position = numpy.array([(0, 0, 0)] * self.N, dtype=numpy.float64)
        else:
            self.position = numpy.asarray(position, dtype=numpy.float64)
            if len(self.position) != N:
                raise ValueError("Particle properties must have length N")

        if types is None:
            self.types = [m.Particle] * self.N
        else:
            self.types = types
            if len(self.types) != N:
                raise ValueError("Particle properties must have length N")


def sc(a: float,
       types: m.MxParticleType = None,
       bond: Union[Callable[[m.MxParticleHandle, m.MxParticleHandle], m.MxBondHandle],
                   List[Callable[[m.MxParticleHandle, m.MxParticleHandle], m.MxBondHandle]]] = None,
       bond_vector: Tuple[bool] = (True, True, True)) -> unitcell:
    """
    Create a unit cell for a simple cubic lattice (3D).

    The simple cubic lattice unit cell has one particle:

    - ``[0, 0, 0]``

    And the box matrix:

    - ``[[a, 0, 0]``
      ``[0, a, 0]``
      ``[0, 0, a]]``

    Example::

        uc = mechanica.lattice.sc(1.0, A, lambda i, j: mx.Bond.create(pot, i, j, dissociation_energy=100.0))

    :param a: lattice constant
    :param types: particle type
    :param bond: bond constructor(s)
    :param bond_vector: flags for creating bonds in the 1-, 2-, and 3-directions
    :return: a simple cubic lattice unit cell
    """

    bonds = None
    if bond:
        try:
            iter(bond)
            bond_funcs = bond
        except TypeError:
            bond_funcs = [bond] * 3

        bonds = []

        if bond_vector[0]:
            bonds.append(_BondRule(bond_funcs[0], (0, 0), (1, 0, 0)))

        if bond_vector[1]:
            bonds.append(_BondRule(bond_funcs[1], (0, 0), (0, 1, 0)))

        if bond_vector[2]:
            bonds.append(_BondRule(bond_funcs[2], (0, 0), (0, 0, 1)))

        bonds = tuple(bonds)

    return unitcell(N=1,
                    types=_make_types(1, types),
                    a1=[a, 0, 0],
                    a2=[0, a, 0],
                    a3=[0, 0, a],
                    dimensions=3,
                    bonds=bonds)


def bcc(a: float,
        types: Union[m.MxParticleType, List[m.MxParticleType]] = None,
        bond: Union[Callable[[m.MxParticleHandle, m.MxParticleHandle], m.MxBondHandle],
                    List[Callable[[m.MxParticleHandle, m.MxParticleHandle], m.MxBondHandle]]] = None,
        bond_vector: Tuple[bool] = (True, True)) -> unitcell:
    """
    Create a unit cell for a body centered cubic lattice (3D).

    The body centered cubic lattice unit cell has two particles:

    - ``[0, 0, 0]``
    - ``[a/2, a/2, a/2]``

    And the box matrix:

    - ``[[a, 0, 0]``
      ``[0, a, 0]``
      ``[0, 0, a]]``

    Example::

        uc = mechanica.lattice.bcc(1.0, A, lambda i, j: mx.Bond.create(pot, i, j, dissociation_energy=100.0))

    :param a: lattice constant
    :param types: particle type or list of particle types
    :param bond: bond constructor(s)
    :param bond_vector: flags for creating bonds
        - between the corner particles
        - between the corner and center particles
    :return: a body centered cubic lattice unit cell
    """

    bonds = None
    if bond:
        try:
            iter(bond)
            bond_funcs = bond
        except TypeError:
            bond_funcs = [bond] * 2

        bonds = []

        if bond_vector[0]:
            bonds.append(_BondRule(bond_funcs[0], (0, 0), (1, 0, 0)))
            bonds.append(_BondRule(bond_funcs[0], (0, 0), (0, 1, 0)))
            bonds.append(_BondRule(bond_funcs[0], (0, 0), (0, 0, 1)))

        if bond_vector[1]:
            bonds.append(_BondRule(bond_funcs[1], (0, 1), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (1, 0, 0)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (0, 1, 0)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (0, 0, 1)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (1, 1, 0)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (1, 0, 1)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (0, 1, 1)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (1, 1, 1)))

        bonds = tuple(bonds)

    return unitcell(N=2,
                    types=_make_types(2, types),
                    position=[[0, 0, 0], [a/2, a/2, a/2]],
                    a1=[a, 0, 0],
                    a2=[0, a, 0],
                    a3=[0, 0, a],
                    dimensions=3,
                    bonds=bonds)


def fcc(a: float,
        types: Union[m.MxParticleType, List[m.MxParticleType]] = None,
        bond: Union[Callable[[m.MxParticleHandle, m.MxParticleHandle], m.MxBondHandle],
                    List[Callable[[m.MxParticleHandle, m.MxParticleHandle], m.MxBondHandle]]] = None,
        bond_vector: Tuple[bool] = (True, True)) -> unitcell:
    """
    Create a unit cell for a face centered cubic lattice (3D).

    The face centered cubic lattice unit cell has four particles:

    - ``[0, 0, 0]``
    - ``[0, a/2, a/2]``
    - ``[a/2, 0, a/2]``
    - ``[a/2, a/2, 0]]``

    And the box matrix:

    - ``[[a, 0, 0]``
      ``[0, a, 0]``
      ``[0, 0, a]]``

    Example::

        uc = mechanica.lattice.fcc(1.0, A, lambda i, j: mx.Bond.create(pot, i, j, dissociation_energy=100.0))

    :param a: lattice constant
    :param types: particle type or list of particle types
    :param bond: bond constructor(s)
    :param bond_vector: flags for creating bonds
        - between the corner particles
        - between the corner and the center particles
    :return: a face centered cubic lattice unit cell
    """

    bonds = None
    if bond:
        try:
            iter(bond)
            bond_funcs = bond
        except TypeError:
            bond_funcs = [bond] * 2

        bonds = []

        if bond_vector[0]:
            bonds.append(_BondRule(bond_funcs[0], (0, 0), (1, 0, 0)))
            bonds.append(_BondRule(bond_funcs[0], (0, 0), (0, 1, 0)))
            bonds.append(_BondRule(bond_funcs[0], (0, 0), (0, 0, 1)))

        if bond_vector[1]:
            bonds.append(_BondRule(bond_funcs[1], (0, 1), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[1], (0, 2), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[1], (0, 3), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (0, 1, 0)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (0, 0, 1)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (0, 1, 1)))
            bonds.append(_BondRule(bond_funcs[1], (2, 0), (1, 0, 0)))
            bonds.append(_BondRule(bond_funcs[1], (2, 0), (0, 0, 1)))
            bonds.append(_BondRule(bond_funcs[1], (2, 0), (1, 0, 1)))
            bonds.append(_BondRule(bond_funcs[1], (3, 0), (1, 0, 0)))
            bonds.append(_BondRule(bond_funcs[1], (3, 0), (0, 1, 0)))
            bonds.append(_BondRule(bond_funcs[1], (3, 0), (1, 1, 0)))

        bonds = tuple(bonds)

    return unitcell(N=4,
                    types=_make_types(4, types),
                    position=[[0, 0, 0], [0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]],
                    a1=[a, 0, 0],
                    a2=[0, a, 0],
                    a3=[0, 0, a],
                    dimensions=3,
                    bonds=bonds)


def sq(a: float,
       types: Union[m.MxParticleType, List[m.MxParticleType]] = None,
       bond: Union[Callable[[m.MxParticleHandle, m.MxParticleHandle], m.MxBondHandle],
                   List[Callable[[m.MxParticleHandle, m.MxParticleHandle], m.MxBondHandle]]] = None,
       bond_vector: Tuple[bool] = (True, True, False)) -> unitcell:
    """
    Create a unit cell for a square lattice (2D).

    The square lattice unit cell has one particle:

    - ``[0, 0]``

    And the box matrix:

    - ``[[a, 0]``
      ``[0, a]]``

    Example::

        uc = mechanica.lattice.sq(1.0, A, lambda i, j: mx.Bond.create(pot, i, j, dissociation_energy=100.0))

    :param a: lattice constant
    :param types: particle type or list of particle types
    :param bond: bond constructor(s)
    :param bond_vector: flags for creating bonds along the 1-, 2- and 3-directions
    :return: a square lattice unit cell
    """

    bonds = None
    if bond:
        try:
            iter(bond)
            bond_funcs = bond
        except TypeError:
            bond_funcs = [bond] * 3

        bonds = []

        if bond_vector[0]:
            bonds.append(_BondRule(bond_funcs[0], (0, 0), (1, 0, 0)))

        if bond_vector[1]:
            bonds.append(_BondRule(bond_funcs[1], (0, 0), (0, 1, 0)))

        if bond_vector[2]:
            bonds.append(_BondRule(bond_funcs[2], (0, 0), (0, 0, 1)))

        bonds = tuple(bonds)

    return unitcell(N=1,
                    types=_make_types(1, types),
                    a1=[a, 0, 0],
                    a2=[0, a, 0],
                    a3=[0, 0, 1],
                    dimensions=2,
                    bonds=bonds)


def hex(a: float,
        types: Union[m.MxParticleType, List[m.MxParticleType]] = None,
        bond: Union[Callable[[m.MxParticleHandle, m.MxParticleHandle], m.MxBondHandle],
                    List[Callable[[m.MxParticleHandle, m.MxParticleHandle], m.MxBondHandle]]] = None,
        bond_vector: Tuple[bool] = (True, True, False)) -> unitcell:
    """
    Create a unit cell for a hexagonal lattice (2D).

    The hexagonal lattice unit cell has two particles:

    - ``[0, 0]``
    - ``[0, a*sqrt(3)/2]``

    And the box matrix:

    - ``[[a, 0]``
      ``[0, a*sqrt(3)]]``

    Example::

        uc = mechanica.lattice.hex(1.0, A, lambda i, j: mx.Bond.create(pot, i, j, dissociation_energy=100.0))

    :param a: lattice constant
    :param types: particle type or list of particle types
    :param bond: bond constructor(s)
    :param bond_vector: flags for creating bonds along the 1-, 2- and 3-directions
    :return: a hexagonal lattice unit cell
    """

    bonds = None
    if bond:
        try:
            iter(bond)
            bond_funcs = bond
        except TypeError:
            bond_funcs = [bond] * 3

        bonds = []

        if bond_vector[0]:
            bonds.append(_BondRule(bond_funcs[0], (0, 0), (1, 0, 0)))
            bonds.append(_BondRule(bond_funcs[0], (1, 1), (1, 0, 0)))

        if bond_vector[1]:
            bonds.append(_BondRule(bond_funcs[1], (0, 1), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (1, 0, 0)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (0, 1, 0)))
            bonds.append(_BondRule(bond_funcs[1], (1, 0), (1, 1, 0)))

        if bond_vector[2]:
            bonds.append(_BondRule(bond_funcs[2], (0, 0), (0, 0, 1)))
            bonds.append(_BondRule(bond_funcs[2], (1, 1), (0, 0, 1)))

        bonds = tuple(bonds)

    return unitcell(N=2,
                    types=_make_types(2, types),
                    position=[[0, 0, 0], [a/2, math.sqrt(3)*a/2, 0]],
                    a1=[a, 0, 0],
                    a2=[0, math.sqrt(3)*a, 0],
                    a3=[0, 0, 1],
                    dimensions=2,
                    bonds=bonds)


def hcp(a: float,
        c: float = None,
        types: Union[m.MxParticleType, List[m.MxParticleType]] = None,
        bond: Union[Callable[[m.MxParticleHandle, m.MxParticleHandle], m.MxBondHandle],
                    List[Callable[[m.MxParticleHandle, m.MxParticleHandle], m.MxBondHandle]]] = None,
        bond_vector: Tuple[bool] = (True, True, True)) -> unitcell:
    """
    Create a unit cell for a hexagonal close pack lattice (3D).

    The hexagonal close pack lattice unit cell has seven particles:

    - ``[0, a*sqrt(3)/2, 0]``
    - ``[a/2, 0, 0]``
    - ``[a, a*sqrt(3)/2, 0]``
    - ``[a*3/2, 0, 0]``
    - ``[a/2, a*2/sqrt(3), c/2]``
    - ``[a, a/2/sqrt(3), c/2]``
    - ``[a*3/2, a*2/sqrt(3), c/2]``

    And the box matrix:

    - ``[[2*a, 0, 0]``
      ``[0, a*sqrt(3), 0]``
      ``[0, 0, c]]``

    Example::

        uc = mechanica.lattice.hcp(1.0, 2.0, A, lambda i, j: mx.Bond.create(pot, i, j, dissociation_energy=100.0))

    :param a: lattice constant
    :param c: height of lattice (default ``a``)
    :param types: particle type or list of particle types
    :param bond: bond constructor(s)
    :param bond_vector: flags for creating bonds
        - between the outer particles
        - between the inner particles
        - between the outer and inner particles
    :return: a hexagonal close pack lattice unit cell
    """

    if c is None:
        c = a

    bonds = None
    if bond:
        try:
            iter(bond)
            bond_funcs = bond
        except TypeError:
            bond_funcs = [bond] * 3

        bonds = []

        if bond_vector[0]:
            bonds.append(_BondRule(bond_funcs[0], (0, 1), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[0], (0, 2), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[0], (1, 2), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[0], (1, 3), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[0], (2, 3), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[0], (2, 0), (1, 0, 0)))
            bonds.append(_BondRule(bond_funcs[0], (3, 0), (1, 0, 0)))
            bonds.append(_BondRule(bond_funcs[0], (3, 1), (1, 0, 0)))
            bonds.append(_BondRule(bond_funcs[0], (0, 1), (0, 1, 0)))
            bonds.append(_BondRule(bond_funcs[0], (2, 1), (0, 1, 0)))
            bonds.append(_BondRule(bond_funcs[0], (2, 3), (0, 1, 0)))

        if bond_vector[1]:
            bonds.append(_BondRule(bond_funcs[1], (4, 5), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[1], (4, 6), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[1], (5, 6), (0, 0, 0)))

        if bond_vector[2]:
            bonds.append(_BondRule(bond_funcs[2], (0, 4), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[2], (4, 1), (0, 1, 0)))
            bonds.append(_BondRule(bond_funcs[2], (2, 4), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[2], (1, 5), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[2], (2, 5), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[2], (3, 5), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[2], (2, 6), (0, 0, 0)))
            bonds.append(_BondRule(bond_funcs[2], (6, 0), (1, 0, 0)))
            bonds.append(_BondRule(bond_funcs[2], (6, 3), (0, 1, 0)))

            bonds.append(_BondRule(bond_funcs[2], (4, 0), (0, 0, 1)))
            bonds.append(_BondRule(bond_funcs[2], (4, 1), (0, 1, 1)))
            bonds.append(_BondRule(bond_funcs[2], (4, 2), (0, 0, 1)))
            bonds.append(_BondRule(bond_funcs[2], (5, 1), (0, 0, 1)))
            bonds.append(_BondRule(bond_funcs[2], (5, 2), (0, 0, 1)))
            bonds.append(_BondRule(bond_funcs[2], (5, 3), (0, 0, 1)))
            bonds.append(_BondRule(bond_funcs[2], (6, 2), (0, 0, 1)))
            bonds.append(_BondRule(bond_funcs[2], (6, 0), (1, 0, 1)))
            bonds.append(_BondRule(bond_funcs[2], (6, 3), (0, 1, 1)))

        bonds = tuple(bonds)

    return unitcell(N=7,
                    types=_make_types(7, types),
                    position=[[0, a*math.sqrt(3)/2, 0], [a/2, 0, 0], [a, a*math.sqrt(3)/2, 0], [a*3/2, 0, 0],
                              [a/2,   a*2/math.sqrt(3), c/2],
                              [a,     a/2/math.sqrt(3), c/2],
                              [a*3/2, a*2/math.sqrt(3), c/2]],
                    a1=[2*a, 0, 0],
                    a2=[0, math.sqrt(3)*a, 0],
                    a3=[0, 0, c],
                    dimensions=3,
                    bonds=bonds)


def create_lattice(uc: unitcell, n: Union[int, List[int]], origin: List[float] = None) -> numpy.ndarray:
    """
    Create a lattice

    Takes a unit cell and replicates it the requested
    number of times in each direction. A generic :py:class:`unitcell`
    may have arbitrary vectors ``a1``, ``a2``, and ``a3``.
    :py:func:`create_lattice` will rotate the unit cell so
    that ``a1`` points in the ``x`` direction and
    ``a2`` is in the ``xy`` plane so that the lattice may be
    represented as a simulation box. When ``n`` is a single value, the lattice is
    replicated ``n`` times in each direction. When ``n`` is a list, the
    lattice is replicated ``n[i]`` times in each ``i``th direction.

    Examples::

        mechanica.lattice.create_lattice(uc=mechanica.lattice.sc(a=1.0), n=[2,4,2])
        mechanica.lattice.create_lattice(uc=mechanica.lattice.bcc(a=1.0), n=10)
        mechanica.lattice.create_lattice(uc=mechanica.lattice.sq(a=1.2), n=[100,10])
        mechanica.lattice.create_lattice(uc=mechanica.lattice.hex(a=1.0), n=[100,58])

    :param uc: unit cell
    :param n: number of unit cells to create along all/each direction(s)
    :param origin: origin to begin creating lattice (default centered about simulation origin)
    :return: particles created in every unit cell
    """

    if isinstance(n, int):
        n = [n, n, 1] if uc.dimensions == 2 else [n] * 3

    if len(n) == 2 and uc.dimensions == 2:
        n.append(1)

    if origin is None:
        cell_half_size = (uc.a1 + uc.a2 + uc.a3) / 2
        extents = n[0] * uc.a1 + n[1] * uc.a2 + n[2] * uc.a3
        origin = m.Universe.center - extents / 2 + cell_half_size

    lattice = numpy.empty(n, dtype=numpy.object)

    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                pos = origin + uc.a1 * i + uc.a2 * j + uc.a3 * k
                parts = [type(pos.tolist()) for (type, pos) in zip(uc.types, uc.position + pos)]
                lattice[i, j, k] = parts

    if uc.bonds:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    for bond in uc.bonds:
                        ii = (i, j, k)  # index of first unit cell, needs to be tuple
                        jj = (ii[0] + bond.cell_offset[0], ii[1] + bond.cell_offset[1], ii[2] + bond.cell_offset[2])
                        # check if next unit cell index is valid
                        if any([not 0 <= jj[i] < n[i] for i in range(3)]):
                            continue

                        # grap the parts out of the lattice
                        ci = lattice[ii]
                        cj = lattice[jj]

                        m.Logger.log(m.Logger.TRACE, f"bonding: {ci[bond.part_ids[0]]}, {cj[bond.part_ids[1]]}")

                        bond.func(ci[bond.part_ids[0]], cj[bond.part_ids[1]])

    return lattice


