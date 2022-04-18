%{
    #include "MxUtil.h"

%}

%ignore Differentiator;
%ignore MxColor3Names;
%ignore MxMath_FindPrimes(uint64_t, int, uint64_t*);

%include "MxUtil.h"

%pythoncode %{
    class PointsType:
        Sphere = MxPointsType_Sphere
        """Unit sphere

        :meta hide-value:
        """

        SolidSphere = MxPointsType_SolidSphere
        """Unit sphere shell

        :meta hide-value:
        """

        Disk = MxPointsType_Disk
        """Unit disk

        :meta hide-value:
        """

        Cube = MxPointsType_Cube
        """Unit hollow cube

        :meta hide-value:
        """

        SolidCube = MxPointsType_SolidCube
        """Unit solid cube

        :meta hide-value:
        """

        Ring = MxPointsType_Ring
        """Unit ring

        :meta hide-value:
        """

    def random_point(kind: int, dr: float = None, phi0: float = None, phi1: float = None):
        """
        Get the coordinates of a random point in a kind of shape.
    
        Currently supports :attr:`PointsType.Sphere`, :attr:`PointsType.Disk`, :attr:`PointsType.SolidCube` and :attr:`PointsType.SolidSphere`.
    
        :param kind: kind of shape
        :param dr: thickness parameter; only applicable to solid sphere kind
        :param phi0: angle lower bound; only applicable to solid sphere kind
        :param phi1: angle upper bound; only applicable to solid sphere kind
        :return: coordinates of random points
        :rtype: :class:`MxVector3f`
        """
        
        args = [kind]
        if dr is not None:
            args.append(dr)
            if phi0 is not None:
                args.append(phi0)
                if phi1 is not None:
                    args.append(phi1)
        return MxRandomPoint(*args)

    def random_points(kind: int, n: int = 1, dr: float = None, phi0: float = None, phi1: float = None):
        """
        Get the coordinates of random points in a kind of shape.
    
        Currently supports :attr:`PointsType.Sphere`, :attr:`PointsType.Disk`, :attr:`PointsType.SolidCube` and :attr:`PointsType.SolidSphere`.
    
        :param kind: kind of shape
        :param n: number of points
        :param dr: thickness parameter; only applicable to solid sphere kind
        :param phi0: angle lower bound; only applicable to solid sphere kind
        :param phi1: angle upper bound; only applicable to solid sphere kind
        :return: coordinates of random points
        :rtype: list of :class:`MxVector3f`
        """
        
        args = [kind, n]
        if dr is not None:
            args.append(dr)
            if phi0 is not None:
                args.append(phi0)
                if phi1 is not None:
                    args.append(phi1)
        return list(MxRandomPoints(*args))

    def points(kind: int, n: int = 1):
        """
        Get the coordinates of uniform points in a kind of shape.
    
        Currently supports :attr:`PointsType.Ring` and :attr:`PointsType.Sphere`.
    
        :param kind: kind of shape
        :param n: number of points
        :return: coordinates of uniform points
        :rtype: list of :class:`MxVector3f`
        """
    
        return list(MxPoints(kind, n))

    def filled_cube_uniform(corner1,
                            corner2,
                            num_parts_x: int = 2,
                            num_parts_y: int = 2,
                            num_parts_z: int = 2):
        """
        Get the coordinates of a uniformly filled cube.
    
        :param corner1: first corner of cube
        :type corner1: list of float or :class:`MxVector3f`
        :param corner2: second corner of cube
        :type corner2: list of float or :class:`MxVector3f`
        :param num_parts_x: number of particles along x-direction of filling axes (>=2)
        :param num_parts_y: number of particles along y-direction of filling axes (>=2)
        :param num_parts_z: number of particles along z-direction of filling axes (>=2)
        :return: coordinates of uniform points
        :rtype: list of :class:`MxVector3f`
        """
    
        return list(MxFilledCubeUniform(MxVector3f(corner1), MxVector3f(corner2), num_parts_x, num_parts_y, num_parts_z))

    def filled_cube_random(corner1, corner2, num_particles: int):
        """
        Get the coordinates of a randomly filled cube.
    
        :param corner1: first corner of cube
        :type corner1: list of float or :class:`MxVector3f`
        :param corner2: second corner of cube
        :type corner2: list of float or :class:`MxVector3f`
        :param num_particles: number of particles
        :return: coordinates of random points
        :rtype: list of :class:`MxVector3f`
        """
    
        return list(MxFilledCubeUniform(MxVector3f(corner1), MxVector3f(corner2), num_particles))

    def icosphere(subdivisions: int, phi0: float, phi1: float):
        """
        Get the coordinates of an icosphere.
    
        :param subdivisions: number of subdivisions
        :param phi0: angle lower bound
        :param phi1: angle upper bound
        :return: vertices and indices
        :rtype: (list of :class:`MxVector3f`, list of int)
        """

        verts = vectorMxVector3f()
        inds = vectorl()
        Mx_Icosphere(subdivisions, phi0, phi1, verts, inds)
        return list(verts), list(inds)

    def color3_names():
        """
        Get the names of all available colors

        :rtype: list of str
        """
    
        return list(MxColor3_Names())

    def primes(start_prime: int, n: int):
        """
        Get prime numbers, beginning with a starting prime number.
        
        :param start_prime: Starting prime number
        :param n: Number of prime numbers to get
        :return: Requested prime numbers
        :rtype: list of int
        """
        
        return list(MxMath_FindPrimes(start_prime, n))

    random_vector = MxRandomVector

    random_unit_vector = MxRandomUnitVector

    get_seed = MxGetSeed

    set_seed = MxSetSeed
%}
