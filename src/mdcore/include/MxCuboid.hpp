/*
 * MxCuboid.h
 *
 *  Created on: Jan 17, 2021
 *      Author: andy
 */
#pragma once
#ifndef SRC_MDCORE_CUBOID_H_
#define SRC_MDCORE_CUBOID_H_

#include <platform.h>
#include <MxBody.hpp>

struct MxCuboidHandle;

struct MxCuboid : MxBody
{
    MxCuboid();

    static MxCuboidHandle *create(MxVector3f *pos=NULL, MxVector3f *size=NULL, MxVector3f *orientation=NULL);
    
    // extents / size of the cuboid
    MxVector3f size;
};

struct MxCuboidHandle : MxBodyHandle
{
    MxCuboid *cuboid();

    MxCuboidHandle() : MxBodyHandle() {}
    MxCuboidHandle(const int32_t &id) : MxBodyHandle(id) {}
    MxCuboidHandle(const MxVector3f &pos, const MxVector3f &size, const MxVector3f &orientation);
    
    /**
     * @brief For initializing a cuboid after constructing with default constructor
     * 
     * @param pos cuboid position
     * @param size cuboid size
     * @param orientation cuboid orientation
     */
    void init(const MxVector3f &pos, const MxVector3f &size={1, 1, 1}, const MxVector3f &orientation={0, 0, 0});
    void scale(const MxVector3f &scale);
};

void MxCuboid_UpdateAABB(MxCuboid *c);

#endif /* SRC_MDCORE_CUBOID_H_ */
