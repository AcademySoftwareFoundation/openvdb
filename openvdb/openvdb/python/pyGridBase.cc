// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/shared_ptr.h>
#include <openvdb/openvdb.h>
#include <variant>
#include "pyTypeCasters.h"

namespace nb = nanobind;
using namespace openvdb::OPENVDB_VERSION_NAME;

/// Create a Python wrapper for GridBase.
void
exportGridBase(nb::module_ m)
{
    // Add a module-level list that gives the types of all supported Grid classes.
    m.attr("GridTypes") = nb::list();

    auto setName = [](GridBase::Ptr grid, const std::string& name) {
        if (name.empty()) {
            grid->removeMeta(GridBase::META_GRID_NAME);
        } else {
            grid->setName(name);
        }
    };

    auto setCreator = [](GridBase::Ptr grid, const std::string& creator) {
        if (creator.empty()) {
            grid->removeMeta(GridBase::META_GRID_CREATOR);
        } else {
            grid->setCreator(creator);
        }
    };

    auto getGridClass = [](GridBase::ConstPtr grid) {
        return GridBase::gridClassToString(grid->getGridClass());
    };

    auto setGridClass = [](GridBase::Ptr grid, const std::string& className)
    {
        if (className.empty()) {
            grid->clearGridClass();
        } else {
            grid->setGridClass(GridBase::stringToGridClass(className));
        }
    };

    auto getVecType = [](GridBase::ConstPtr grid) {
        return GridBase::vecTypeToString(grid->getVectorType());
    };

    auto setVecType = [](GridBase::Ptr grid, const std::string& vecTypeName) {
        if (vecTypeName.empty()) {
            grid->clearVectorType();
        } else {
            grid->setVectorType(GridBase::stringToVecType(vecTypeName));
        }
    };

    auto setGridTransform = [](GridBase::Ptr grid, math::Transform::Ptr xform) {
        grid->setTransform(xform);
    };

    auto gridInfo = [](GridBase::ConstPtr grid, int verbosity) {
        std::ostringstream ostr;
        grid->print(ostr, std::max<int>(1, verbosity));
        return ostr.str();
    };

    auto getStatsMetadata = [](GridBase::ConstPtr grid) {
        if (MetaMap::ConstPtr metadata = grid->getStatsMetadata())
            return *metadata;
        else
            return MetaMap();
    };

    auto getAllMetadata = [](GridBase::ConstPtr grid) {
        return static_cast<const MetaMap&>(*grid);
    };

    auto replaceAllMetadata = [](GridBase::Ptr grid, const MetaMap& metadata) {
        grid->clearMetadata();
        for (MetaMap::ConstMetaIterator it = metadata.beginMeta();
            it != metadata.endMeta(); ++it) {
            if (it->second)
                grid->insertMeta(it->first, *it->second);
        }
    };


    auto updateMetadata = [](GridBase::Ptr grid, const MetaMap& metadata) {
        for (MetaMap::ConstMetaIterator it = metadata.beginMeta();
            it != metadata.endMeta(); ++it) {
            if (it->second) {
                grid->removeMeta(it->first);
                grid->insertMeta(it->first, *it->second);
            }
        }
    };


    auto getMetadataKeys = [](GridBase::ConstPtr grid) {
        // Return an iterator over the "keys" view of a dict.
        return nb::make_key_iterator(nb::type<MetaMap>(), "key_iterator", static_cast<const MetaMap&>(*grid).beginMeta(), static_cast<const MetaMap&>(*grid).endMeta());
    };


    auto getMetadata = [](GridBase::ConstPtr grid, const std::string& name) {
        Metadata::ConstPtr metadata = (*grid)[name];
        if (!metadata) {
            throw nb::key_error(name.c_str());
        }

        MetaMap metamap;
        metamap.insertMeta(name, *metadata);
        // todo: Add/refactor out type_casters for each TypedMetadata from MetaMap's type_caster
        return nb::cast<nb::object>(nb::cast<nb::dict>(nb::cast(metamap))[nb::str(name.c_str())]);
    };


    auto setMetadata = [](GridBase::Ptr grid, const std::string& name,
            const std::variant<bool, int32_t, int64_t,
                               float, double, std::string,
                               Vec2d, Vec2i, Vec2s,
                               Vec3d, Vec3i, Vec3s,
                               Vec4d, Vec4i, Vec4s,
                               Mat4s, Mat4d>& value) {
        // Insert the Python object into a Python dict, then use the dict-to-MetaMap
        // converter (see pyOpenVDBModule.cc) to convert the dict to a MetaMap
        // containing a Metadata object of the appropriate type.
        // todo: Add/refactor out type_casters for each TypedMetadata from MetaMap's type_caster
        nb::dict dictObj;
        dictObj[nb::str(name.c_str())] = value;
        MetaMap metamap = nb::cast<MetaMap>(dictObj);

        if (Metadata::Ptr metadata = metamap[name]) {
            grid->removeMeta(name);
            grid->insertMeta(name, *metadata);
        }
    };


    auto removeMetadata = [](GridBase::Ptr grid, const std::string& name) {
        Metadata::Ptr metadata = (*grid)[name];
        if (!metadata) {
            throw nb::key_error(name.c_str());
        }
        grid->removeMeta(name);
    };


    auto hasMetadata = [](GridBase::ConstPtr grid, const std::string& name) {
        return ((*grid)[name] != nullptr);
    };

    auto evalActiveVoxelBoundingBox = [](GridBase::ConstPtr grid) {
        CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
        return nb::make_tuple(bbox.min(), bbox.max());
    };

    // Export GridBase in order to properly support inheritance for typed Grids
    // and expose the corresponding base-class properties.
    nb::class_<GridBase>(m, "GridBase")
        .def("empty", &GridBase::empty,
            "Return True if this grid contains only background voxels.")
        .def("__nonzero__", [](GridBase::ConstPtr grid) { return !grid->empty(); })
        .def("clear", &GridBase::clear,
            "Remove all tiles from this grid and all nodes other than the root node.")
        .def_prop_rw("name", &GridBase::getName, setName,
            "this grid's user-specified name")
        .def_prop_rw("creator", &GridBase::getCreator, setCreator,
            "user-specified description of this grid's creator")
        .def_prop_rw("gridClass", getGridClass, setGridClass,
            "the class of volumetric data (level set, fog volume, etc.)\n"
            "stored in this grid")
        .def_prop_rw("vectorType", getVecType, setVecType,
            "how transforms are applied to values stored in this grid")
        .def_prop_rw("transform", static_cast<math::Transform::Ptr (GridBase::*)()>(&GridBase::transformPtr),
            setGridTransform, "transform associated with this grid")
        .def("info", gridInfo, nb::arg("verbosity") = 1,
            "Return a string containing information about this grid\n"
            "with a specified level of verbosity.")
        .def("activeVoxelCount", &GridBase::activeVoxelCount,
            "Return the number of active voxels in this grid.")
        .def("evalActiveVoxelBoundingBox", evalActiveVoxelBoundingBox,
            "Return the coordinates of opposite corners of the axis-aligned\n"
            "bounding box of all active voxels.")
        .def("evalActiveVoxelDim", &GridBase::evalActiveVoxelDim,
            "Return the dimensions of the axis-aligned bounding box of all\n"
            "active voxels.")
        .def("memUsage", &GridBase::memUsage,
            "Return the memory usage of this grid in bytes.")
        .def("addStatsMetadata", &GridBase::addStatsMetadata,
            "Add metadata to this grid comprising the current values\n"
            "of statistics like the active voxel count and bounding box.\n"
            "(This metadata is not automatically kept up-to-date with\n"
            "changes to this grid.)")
        .def("getStatsMetadata", getStatsMetadata,
            "Return a (possibly empty) dict containing just the metadata\n"
            "that was added to this grid with addStatsMetadata().")
        .def_prop_rw("metadata", getAllMetadata, replaceAllMetadata,
            "dict of this grid's metadata\n\n"
            "Setting this attribute replaces all of this grid's metadata,\n"
            "but mutating it in place has no effect on the grid, since\n"
            "the value of this attribute is a only a copy of the metadata.\n"
            "Use either indexing or updateMetadata() to mutate metadata in place.")
        .def("updateMetadata", updateMetadata,
            "Add metadata to this grid, replacing any existing items\n"
            "having the same names as the new items.")
        .def("__getitem__", getMetadata,
            "Return the metadata value associated with the given name.")
        .def("__setitem__", setMetadata,
            "Add metadata to this grid, replacing any existing item having\n"
            "the same name as the new item.")
        .def("__delitem__", removeMetadata,
            "Remove the metadata with the given name.")
        .def("__contains__", hasMetadata,
            "Return True if this grid contains metadata with the given name.")
        .def("__iter__", getMetadataKeys,
            "Return an iterator over this grid's metadata keys.")
        .def("iterkeys", getMetadataKeys,
            "Return an iterator over this grid's metadata keys.")
        .def_prop_rw("saveFloatAsHalf",
            &GridBase::saveFloatAsHalf, &GridBase::setSaveFloatAsHalf,
            "if True, write floating-point voxel values as 16-bit half floats");
}

