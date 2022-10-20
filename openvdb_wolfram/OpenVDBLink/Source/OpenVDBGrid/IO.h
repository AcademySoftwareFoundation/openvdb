// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_IO_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_IO_HAS_BEEN_INCLUDED

/* OpenVDBGrid public member function list

const char* detectFileVDBType(const char* file_path, const char* grid_name)

bool importVDB(const char* file_path, const char* grid_name)

void exportVDB(const char* file_path)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
const char*
openvdbmma::OpenVDBGrid<V>::importVDBType(const char* file_path, const char* grid_name)
{
    using NameIter = openvdb::io::File::NameIterator;

    openvdb::io::File file(file_path);
    const std::string filename(file_path);
    mma::disownString(file_path);

    const std::string name(grid_name);
    mma::disownString(grid_name);

    if (!file.open())
        throw mma::LibraryError("Unable to read " + filename + ".");

    bool read_grid = false;
    openvdb::GridBase::Ptr baseGrid;
    for (NameIter nameIter = file.beginName(); nameIter != file.endName(); ++nameIter) {
        if (name == "" || nameIter.gridName() == name) {
            baseGrid = file.readGridMetadata(nameIter.gridName());
            read_grid = true;
            break;
        }
    }

    file.close();

    if (!read_grid)
        throw mma::LibraryError("Grid" + (name == "" ? "" : (" named " + name)) + " not found.");

    //Let the class handle memory management when passing a string to WL
    return WLString(baseGrid->type());
}

template<typename V>
bool
openvdbmma::OpenVDBGrid<V>::importVDB(const char* file_path, const char* grid_name)
{
    using NameIter = openvdb::io::File::NameIterator;

    openvdb::io::File file(file_path);
    const std::string filename(file_path);
    mma::disownString(file_path);

    const std::string name(grid_name);
    mma::disownString(grid_name);

    if (!file.open())
        throw mma::LibraryError("Unable to read " + filename + ".");

    bool read_grid = false;
    openvdb::GridBase::Ptr baseGrid;
    for (NameIter nameIter = file.beginName(); nameIter != file.endName(); ++nameIter) {
        if (name == "" || nameIter.gridName() == name) {
            baseGrid = file.readGrid(nameIter.gridName());
            read_grid = true;
            break;
        }
    }

    file.close();

    if (!read_grid)
        throw mma::LibraryError("Grid" + (name == "" ? "" : (" named " + name)) + " not found.");

    if (!baseGrid->isType<wlGridType>()) {
        throw mma::LibraryError("Type mismatch: expected " + grid()->gridType() +
            " but encountered " + baseGrid->type() + ".");
    }

    wlGridPtr grid = openvdb::gridPtrCast<wlGridType>(baseGrid);
    setGrid(grid, false);

    return true;
}

template<typename V>
void
openvdbmma::OpenVDBGrid<V>::exportVDB(const char* file_path) const
{
    // How can we determine of vdb export is successful?
    openvdb::io::File(file_path).write({grid()});

    mma::disownString(file_path);
}

#endif // OPENVDBLINK_OPENVDBGRID_IO_HAS_BEEN_INCLUDED
