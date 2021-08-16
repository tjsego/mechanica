from . import mechanica as mx


class ParticleType:
    """Interface for class-centric design of MxParticleType"""
    mass = None
    charge = None
    radius = None
    target_energy = None
    minimum_radius = None
    eps = None
    rmin = None
    dynamics = None
    frozen = None
    name = None
    name2 = None
    style = None  # Dictionary specification
    species = None

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

    @classmethod
    def get(cls):
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
                type_instance.style.newColorMapper(**kwargs)
        
        type_instance.registerType()
        type_instance = type_instance.get()

        if hasattr(cls, 'on_register'):
            cls.on_register(type_instance)

        return type_instance


class ClusterType(ParticleType):

    types = None

    __mx_properties__ = ParticleType.__mx_properties__ + [
        'types'
    ]

    @classmethod
    def get(cls):
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
                type_instance.style.newColorMapper(**kwargs)

        type_instance.registerType()
        type_instance = type_instance.get()

        if cls.types is not None:
            for t in cls.types:
                type_instance.types.insert(t.id)

        if hasattr(cls, 'on_register'):
            cls.on_register(type_instance)

        return type_instance
