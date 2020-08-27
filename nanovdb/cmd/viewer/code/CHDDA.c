// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

////////////////////////////////////////////////////////

CNANOVDB_INLINE HDDA HDDA_create(nanovdb_Ray ray, int dim)
{
    vec3 dir = ray.mDir;
    vec3 inv = vec3_fdiv(1.0f, dir);
    vec3 pos = vec3_add(ray.mEye, vec3_fmul(ray.mT0, dir));

    HDDA v;
    v.mDim = dim;
    v.mT0 = ray.mT0;
    v.mT1 = ray.mT1;
    v.mVoxel = ivec3_andi(CNANOVDB_MAKE_IVEC3(floor(pos.x), floor(pos.y), floor(pos.z)), (~(dim - 1)));

    {
        if (fabs(dir.x) < DeltaFloat) { //handles dir = +/- 0
            v.mNext.x = MaxFloat; //i.e. disabled!
        } else if (inv.x > 0) {
            v.mStep.x = dim;
            v.mNext.x = v.mT0 + ((float)(v.mVoxel.x) + (float)(dim)-pos.x) * inv.x;
            v.mDelta.x = dim * inv.x;
        } else {
            v.mStep.x = -dim;
            v.mNext.x = v.mT0 + (v.mVoxel.x - pos.x) * inv.x;
            v.mDelta.x = -dim * inv.x;
        }
    }
    {
        if (fabs(dir.y) < DeltaFloat) { //handles dir = +/- 0
            v.mNext.y = MaxFloat; //i.e. disabled!
        } else if (inv.y > 0) {
            v.mStep.y = dim;
            v.mNext.y = v.mT0 + (v.mVoxel.y + dim - pos.y) * inv.y;
            v.mDelta.y = dim * inv.y;
        } else {
            v.mStep.y = -dim;
            v.mNext.y = v.mT0 + (v.mVoxel.y - pos.y) * inv.y;
            v.mDelta.y = -dim * inv.y;
        }
    }
    {
        if (fabs(dir.z) < DeltaFloat) { //handles dir = +/- 0
            v.mNext.z = MaxFloat; //i.e. disabled!
        } else if (inv.z > 0) {
            v.mStep.z = dim;
            v.mNext.z = v.mT0 + (v.mVoxel.z + dim - pos.z) * inv.z;
            v.mDelta.z = dim * inv.z;
        } else {
            v.mStep.z = -dim;
            v.mNext.z = v.mT0 + (v.mVoxel.z - pos.z) * inv.z;
            v.mDelta.z = -dim * inv.z;
        }
    }
    return v;
}

CNANOVDB_INLINE boolean HDDA_step(CNANOVDB_REF(HDDA) v)
{
    const int stepAxis = MinIndex(CNANOVDB_DEREF(v).mNext);
    if (stepAxis == 0) {
        CNANOVDB_DEREF(v).mT0 = CNANOVDB_DEREF(v).mNext.x;
        CNANOVDB_DEREF(v).mNext.x += CNANOVDB_DEREF(v).mDelta.x;
        CNANOVDB_DEREF(v).mVoxel.x += CNANOVDB_DEREF(v).mStep.x;
    } else if (stepAxis == 1) {
        CNANOVDB_DEREF(v).mT0 = CNANOVDB_DEREF(v).mNext.y;
        CNANOVDB_DEREF(v).mNext.y += CNANOVDB_DEREF(v).mDelta.y;
        CNANOVDB_DEREF(v).mVoxel.y += CNANOVDB_DEREF(v).mStep.y;
    } else if (stepAxis == 2) {
        CNANOVDB_DEREF(v).mT0 = CNANOVDB_DEREF(v).mNext.z;
        CNANOVDB_DEREF(v).mNext.z += CNANOVDB_DEREF(v).mDelta.z;
        CNANOVDB_DEREF(v).mVoxel.z += CNANOVDB_DEREF(v).mStep.z;
    }
    return CNANOVDB_DEREF(v).mT0 <= CNANOVDB_DEREF(v).mT1;
}

CNANOVDB_INLINE boolean HDDA_update(CNANOVDB_REF(HDDA) v, nanovdb_Ray ray, int32_t dim)
{
    if (CNANOVDB_DEREF(v).mDim == dim)
        return CNANOVDB_FALSE;
    CNANOVDB_DEREF(v).mDim = dim;

    vec3 dir = ray.mDir;
    vec3 inv = vec3_fdiv(1.0f, dir);
    vec3 pos = vec3_add(ray.mEye, vec3_fmul(ray.mT0, dir));
    CNANOVDB_DEREF(v).mVoxel = ivec3_andi(CNANOVDB_MAKE_IVEC3(floor(pos.x), floor(pos.y), floor(pos.z)), (~(dim - 1)));

    if (fabs(dir.x) >= DeltaFloat) {
        float tmp = dim * inv.x;
        CNANOVDB_DEREF(v).mStep.x = dim; //>0
        CNANOVDB_DEREF(v).mDelta.x = tmp; //>0
        CNANOVDB_DEREF(v).mNext.x = CNANOVDB_DEREF(v).mT0 + (CNANOVDB_DEREF(v).mVoxel.x + dim - pos.x) * inv.x;
        if (inv.x <= 0) {
            CNANOVDB_DEREF(v).mStep.x = -dim; //<0
            CNANOVDB_DEREF(v).mDelta.x = -tmp; //>0
            CNANOVDB_DEREF(v).mNext.x -= tmp;
        }
    }

    if (fabs(dir.y) >= DeltaFloat) {
        float tmp = dim * inv.y;
        CNANOVDB_DEREF(v).mStep.y = dim; //>0
        CNANOVDB_DEREF(v).mDelta.y = tmp; //>0
        CNANOVDB_DEREF(v).mNext.y = CNANOVDB_DEREF(v).mT0 + (CNANOVDB_DEREF(v).mVoxel.y + dim - pos.y) * inv.y;
        if (inv.y <= 0) {
            CNANOVDB_DEREF(v).mStep.y = -dim; //<0
            CNANOVDB_DEREF(v).mDelta.y = -tmp; //>0
            CNANOVDB_DEREF(v).mNext.y -= tmp;
        }
    }

    if (fabs(dir.z) >= DeltaFloat) {
        float tmp = dim * inv.z;
        CNANOVDB_DEREF(v).mStep.z = dim; //>0
        CNANOVDB_DEREF(v).mDelta.z = tmp; //>0
        CNANOVDB_DEREF(v).mNext.z = CNANOVDB_DEREF(v).mT0 + (CNANOVDB_DEREF(v).mVoxel.z + dim - pos.z) * inv.z;
        if (inv.z <= 0) {
            CNANOVDB_DEREF(v).mStep.z = -dim; //<0
            CNANOVDB_DEREF(v).mDelta.z = -tmp; //>0
            CNANOVDB_DEREF(v).mNext.z -= tmp;
        }
    }

    return CNANOVDB_TRUE;
}

////////////////////////////////////////////////////////

CNANOVDB_INLINE boolean nanovdb_ZeroCrossing(CNANOVDB_CONTEXT cxt, nanovdb_Ray ray, nanovdb_ReadAccessor acc, CNANOVDB_REF(nanovdb_Coord) ijk, CNANOVDB_REF(float) v)
{
    // intersect with bounds...
    boolean hit = nanovdb_Ray_clip(CNANOVDB_ADDRESS(ray),
                                nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).root.mBBox_min),
                                nanovdb_CoordToVec3f(CNANOVDB_ROOTDATA(cxt).root.mBBox_max));

    if (!hit || ray.mT1 > 1000000.f)
        return CNANOVDB_FALSE;

    // intersect with levelset...
    CNANOVDB_DEREF(ijk) = nanovdb_Vec3fToCoord(nanovdb_Ray_start(CNANOVDB_ADDRESS(ray)));
    float v0 = nanovdb_ReadAccessor_getValue(cxt, CNANOVDB_ADDRESS(acc), CNANOVDB_DEREF(ijk));
    int   n = (int)(ray.mT1 - ray.mT0);
    if (n <= 1)
        return CNANOVDB_FALSE;

#if 0
	HDDA hdda = HDDA_create(ray, nanovdb_ReadAccessor_getDim(cxt, CNANOVDB_ADDRESS(acc), CNANOVDB_DEREF(ijk), ray));
	while (HDDA_step(CNANOVDB_ADDRESS(hdda)) && (--n > 0))
	{
		CNANOVDB_DEREF(ijk) = nanovdb_Vec3fToCoord(vec3_add(ray.mEye, vec3_fmul((hdda.mT0 + 1.0f), ray.mDir)));
		HDDA_update(CNANOVDB_ADDRESS(hdda), ray, nanovdb_ReadAccessor_getDim(cxt, CNANOVDB_ADDRESS(acc), CNANOVDB_DEREF(ijk), ray));
		if ( hdda.mDim > 1 || !nanovdb_ReadAccessor_isActive(acc, CNANOVDB_DEREF(ijk) ) )
			continue;

		while(HDDA_step(CNANOVDB_ADDRESS(hdda)) && nanovdb_ReadAccessor_isActive(acc, nanovdb_Vec3iToCoord(hdda.mVoxel)) )
		{
			// in the narrow band
			CNANOVDB_DEREF(v) = nanovdb_ReadAccessor_getValue(cxt, CNANOVDB_ADDRESS(acc), nanovdb_Vec3iToCoord(hdda.mVoxel));
			if (CNANOVDB_DEREF(v)*v0 < 0)
			{
				CNANOVDB_DEREF(ijk) = nanovdb_Vec3iToCoord(hdda.mVoxel);
				return CNANOVDB_TRUE;
			}
		}
	}
#elif 1
    HDDA hdda = HDDA_create(ray, 1);
    while (HDDA_step(CNANOVDB_ADDRESS(hdda)) && (--n > 0)) {
        CNANOVDB_DEREF(ijk) = nanovdb_Vec3fToCoord(vec3_add(ray.mEye, vec3_fmul((hdda.mT0 + 1.0f), ray.mDir)));
        if (hdda.mDim < 1 || !nanovdb_ReadAccessor_isActive(acc, CNANOVDB_DEREF(ijk)))
            continue;

        while (HDDA_step(CNANOVDB_ADDRESS(hdda)) && nanovdb_ReadAccessor_isActive(acc, nanovdb_Vec3iToCoord(hdda.mVoxel))) {
            // in the narrow band
            CNANOVDB_DEREF(ijk) = nanovdb_Vec3iToCoord(hdda.mVoxel);
            CNANOVDB_DEREF(v) = nanovdb_ReadAccessor_getValue(cxt, CNANOVDB_ADDRESS(acc), CNANOVDB_DEREF(ijk));
            if (CNANOVDB_DEREF(v) * v0 < 0)
                return CNANOVDB_TRUE;
        }
    }
#else
    return CNANOVDB_TRUE;
#endif
    return CNANOVDB_FALSE;
}

////////////////////////////////////////////////////////