#include <nanovdb/io/IO.h> // this is required to read (and write) NanoVDB files on the host

/// @brief Read a NanoVDB grid from a file and print out multiple values.
///
/// @note Note This example does NOT depend on OpenVDB (nor CUDA), only NanoVDB.
int main()
{
    try {
        auto handle = nanovdb::io::readGrid("data/sphere.nvdb"); // reads first grid from file

        auto* grid = handle.grid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float

        if (!grid)
            throw std::runtime_error("File did not contain a grid with value type float");

        auto acc = grid->getAccessor(); // create an accessor for fast access to multiple values
        for (int i = 97; i < 104; ++i) {
            printf("(%3i,0,0) NanoVDB cpu: % -4.2f\n", i, acc.getValue(nanovdb::Coord(i, 0, 0)));
        }
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}