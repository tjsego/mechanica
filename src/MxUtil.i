%{
    #include "MxUtil.h"

%}

%include "MxUtil.h"

%pythoncode %{
    class PointsType:
        Sphere = MxPointsType_Sphere
        SolidSphere = MxPointsType_SolidSphere
        Disk = MxPointsType_Disk
        SolidCube = MxPointsType_SolidCube
        Cube = MxPointsType_Cube
        Ring = MxPointsType_Ring

    RandomPoints = MxRandomPoints
    Points = MxPoints
    FilledCubeUniform = MxFilledCubeUniform
    FilledCubeRandom = MxFilledCubeRandom

    color3_names = MxColor3_Names
%}
