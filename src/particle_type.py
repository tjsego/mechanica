from . import mechanica as mx


class ParticleType:
    """
    Interface for class-centric design of MxParticleType
    """

    mass = None
    """Particle type mass"""

    charge = None
    """Particle type charge"""

    radius = None
    """Particle type radius"""

    target_energy = None
    """Particle type target energy"""

    minimum_radius = None
    """Particle type minimum radius"""

    eps = None
    """Particle type nonbonded interaction parameter"""

    rmin = None
    """Particle type nonbonded interaction parameter"""

    dynamics = None
    """Particle type dynamics flag"""

    frozen = None
    """Particle type frozen flag"""

    name = None
    """Particle type name"""

    name2 = None
    """Particle type second name"""

    style = None
    """
    Particle type style dictionary specification. 
    
    Basic rendering details can be specified as a dictionary, like color and visibility, 
    
    .. code:: python
    
        style = {'color': 'CornflowerBlue', 'visible': False}

    This declaration is the same as performing operations on a type after registration, 
    
    .. code:: python
    
        ptype: MxParticleType
        ptype.style.setColor('CornflowerBlue')
        ptype.style.setVisible(False)
    
    Rendering instead by species and species amount uses specification for a color mapper, 
    
    .. code:: python
    
        style = {'colormap': {'species': 'S1', 'map': 'rainbow', 'range': (0, 10)}}

    This declaration is the same as performing operations on a type after registration, 
    
    .. code:: python
    
        ptype: MxParticleType
        ptype.style.newColorMapper(partType=ptype, speciesName='S1', name='rainbow', min=0, max=10)
    
    """

    species = None
    """
    Particle type list of species by name, if any. Species are automatically created and populated in the state 
    vector of the type and all created particles. 
    """

    __mx_properties__ = [
        'mass',
        'charge',
        'radius',
        'target_energy',
        'minimum_radius',
        'eps',
        'rmin',
        'dynamics',
        'frozen',
        'name',
        'name2',
        'style',
        'species'
    ]
    """All defined particle type properties"""

    @classmethod
    def get(cls):
        """
        Get the engine type that corresponds to this class.

        The type is automatically registered as necessary.

        :return: registered type instance
        :rtype: mechanica.MxParticleType
        """

        name = cls.name
        if name is None:
            name = cls.__name__

        type_instance = mx.MxParticleType_FindFromName(name)
        if type_instance is not None:
            return type_instance

        type_instance = mx.MxParticleType(noReg=True)

        props_to_copy = [n for n in cls.__mx_properties__ if n not in ['name', 'species', 'style', 'types']]
        props_to_assign = {prop_name: getattr(cls, prop_name) for prop_name in props_to_copy}
        props_to_assign['name'] = name

        for prop_name, prop_value in props_to_assign.items():
            if prop_value is not None:
                setattr(type_instance, prop_name, prop_value)

        if cls.species is not None:
            type_instance.species = mx.MxSpeciesList()
            for s in cls.species:
                type_instance.species.insert(s)

        if cls.style is not None:
            cls.style: dict
            if 'color' in cls.style.keys():
                type_instance.style.setColor(cls.style['color'])
            if 'visible' in cls.style.keys():
                type_instance.style.setVisible(cls.style['visible'])
            if 'colormap' in cls.style.keys():
                kwargs = {'partType': type_instance,
                          'speciesName': cls.style['colormap']['species']}
                if 'map' in cls.style['colormap'].keys():
                    kwargs['name'] = cls.style['colormap']['map']
                if 'range' in cls.style['colormap'].keys():
                    r = cls.style['colormap']['range']
                    kwargs['min'] = r[0]
                    kwargs['max'] = r[1]
                type_instance.style.newColorMapper(**kwargs)

        type_instance.registerType()
        type_instance = type_instance.get()

        if hasattr(cls, 'on_register'):
            cls.on_register(type_instance)

        return type_instance


class ClusterType(ParticleType):
    """
    Interface for class-centric design of MxClusterParticleType
    """

    types = None
    """List of constituent types of the cluster, if any"""

    __mx_properties__ = ParticleType.__mx_properties__ + [
        'types'
    ]
    """All defined cluster type properties"""

    @classmethod
    def get(cls):
        """
        Get the engine type that corresponds to this class.

        The type is automatically registered as necessary.

        :return: registered type instance
        :rtype: mechanica.MxClusterParticleType
        """

        name = cls.name
        if name is None:
            name = cls.__name__

        type_instance = mx.MxClusterParticleType_FindFromName(name)
        if type_instance is not None:
            return type_instance

        type_instance = mx.MxClusterParticleType(noReg=True)

        props_to_copy = [n for n in cls.__mx_properties__ if n not in ['name', 'species', 'style', 'types']]
        props_to_assign = {prop_name: getattr(cls, prop_name) for prop_name in props_to_copy}
        props_to_assign['name'] = name

        for prop_name, prop_value in props_to_assign.items():
            if prop_value is not None:
                setattr(type_instance, prop_name, prop_value)

        if cls.species is not None:
            type_instance.species = mx.MxSpeciesList()
            for s in cls.species:
                type_instance.species.insert(s)

        if cls.style is not None:
            cls.style: dict
            if 'color' in cls.style.keys():
                type_instance.style.setColor(cls.style['color'])
            if 'visible' in cls.style.keys():
                type_instance.style.setVisible(cls.style['visible'])
            if 'colormap' in cls.style.keys():
                kwargs = {'partType': type_instance,
                          'speciesName': cls.style['colormap']['species']}
                if 'map' in cls.style['colormap'].keys():
                    kwargs['name'] = cls.style['colormap']['map']
                if 'range' in cls.style['colormap'].keys():
                    r = cls.style['colormap']['range']
                    kwargs['min'] = r[0]
                    kwargs['max'] = r[1]
                type_instance.style.newColorMapper(**kwargs)

        type_instance.registerType()
        type_instance = type_instance.get()

        if cls.types is not None:
            for t in cls.types:
                type_instance.types.insert(t.id)

        if hasattr(cls, 'on_register'):
            cls.on_register(type_instance)

        return type_instance
