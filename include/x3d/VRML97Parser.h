/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Parser.h
*
******************************************************************/

#ifndef _CX3D_VRML97PARSER_H_
#define _CX3D_VRML97PARSER_H_

#include <x3d/Parser.h>

namespace CyberX3D {

class VRML97Parser : public Parser {

public:

	VRML97Parser();
	virtual ~VRML97Parser();

	bool load(const char *fileName, void (*callbackFn)(int nLine, void *info) = NULL, void *callbackFnInfo = NULL);
};

enum {
VRML97_ANCHOR,
VRML97_ANCHOR_PARAMETER,
VRML97_ANCHOR_URL,
VRML97_APPEARANCE,
VRML97_AUDIOCLIP,
VRML97_AUDIOCLIP_URL,
VRML97_BACKGROUND,
VRML97_BACKGROUND_BACKURL,
VRML97_BACKGROUND_BOTTOMURL,
VRML97_BACKGROUND_FRONTURL,
VRML97_BACKGROUND_GROUNDANGLE,
VRML97_BACKGROUND_GROUNDCOLOR,
VRML97_BACKGROUND_LEFTURL,
VRML97_BACKGROUND_RIGHTURL,
VRML97_BACKGROUND_SKYANGLE,
VRML97_BACKGROUND_SKYCOLOR,
VRML97_BACKGROUND_TOPURL,
VRML97_BILLBOARD,
VRML97_BOX,
VRML97_CHILDREN,
VRML97_COLLISION,
VRML97_COLLISION_PROXY,
VRML97_COLOR,
VRML97_COLOR_INDEX,
VRML97_COLORINTERPOLATOR,
VRML97_CONE,
VRML97_COORDINATE,
VRML97_COORDINATE_INDEX,
VRML97_COORDINATEINTERPOLATOR,
VRML97_CUBE,
VRML97_CYLINDER,
VRML97_CYLINDERSENSOR,
VRML97_DIRECTIONALLIGHT,
VRML97_ELEVATIONGRID,
VRML97_ELEVATIONGRID_HEIGHT,
VRML97_EXTRUSION,
VRML97_EXTRUSION_CROSSSECTION,
VRML97_EXTRUSION_ORIENTATION,
VRML97_EXTRUSION_SCALE,
VRML97_EXTRUSION_SPINE,
VRML97_FOG,
VRML97_FONTSTYLE,
VRML97_FONTSTYLE_JUSTIFY,
VRML97_GROUP,
VRML97_IMAGETEXTURE,
VRML97_IMAGETEXTURE_URL,
VRML97_INDEXEDFACESET,
VRML97_INDEXEDLINESET,
VRML97_INLINE,
VRML97_INLINE_URL,
VRML97_INTERPOLATOR_KEY,
VRML97_INTERPOLATOR_KEYVALUE,
VRML97_LOD,
VRML97_LOD_LEVEL,
VRML97_LOD_RANGE,
VRML97_MATERIAL,
VRML97_MOVIETEXTURE,
VRML97_MOVIETEXTURE_URL,
VRML97_NAVIGATIONINFO,
VRML97_NAVIGATIONINFO_AVATARSIZE,
VRML97_NAVIGATIONINFO_TYPE,
VRML97_NORMAL,
VRML97_NORMAL_INDEX,
VRML97_NORMALINTERPOLATOR,
VRML97_ORIENTATIONINTERPOLATOR,
VRML97_PIXELTEXTURE,
VRML97_PIXELTEXTURE_IMAGE,
VRML97_PLANESENSOR,
VRML97_POINTLIGHT,
VRML97_POINTSET,
VRML97_POSITIONINTERPOLATOR,
VRML97_PROXIMITYSENSOR,
VRML97_ROOT,
VRML97_SCALARINTERPOLATOR,
VRML97_SCRIPT,
VRML97_SCRIPT_URL,
VRML97_SHAPE,
VRML97_SOUND,
VRML97_SPHERE,
VRML97_SPHERESENSOR,
VRML97_SPOTLIGHT,
VRML97_SWITCH,
VRML97_SWITCH_CHOICE,
VRML97_TEXT,
VRML97_TEXT_LENGTH,
VRML97_TEXT_STRING,
VRML97_TEXTURECOODINATE,
VRML97_TEXTURECOODINATE_INDEX,
VRML97_TEXTURETRANSFORM,
VRML97_TIMESENSOR,
VRML97_TOUCHSENSOR,
VRML97_TRANSFORM,
VRML97_TRANSLATION,
VRML97_VIEWPOINT,
VRML97_VISIBILITYSENSOR,
VRML97_WORLDINFO,
VRML97_WORLDINFO_INFO,
};

////////////////////////////////////////////////
// Parser Buffer Size
////////////////////////////////////////////////

const int VRML97_PARSER_DEFAULT_BUF_SIZE =  512 * 1024;

void VRML97ParserSetBufSize(int bufSize);
int VRML97ParserGetBufSize();

}

#endif


