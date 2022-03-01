#include <fstream>
#include <iostream>
#include <cstdio>

#include "Util.h"
#include "Tool.h"

int main (int argc, char *argv[])
{
    if (argc<2) {
        std::cerr << "Usage: " << argv[0] << " conf1.txt conf2.txt ...\n";
        return 1;
    }
    for (int i=1; i<argc; ++i) {
        std::string line;
        std::ifstream file(argv[i]);
        if (!file.is_open() || !getline (file, line)) {
            std::cerr << "Failed to read file " << argv[i] << std::endl;
            return 1;
        }
        auto header = openvdb::vdb_tool::tokenize(line, " .");
        int v[3];
        if (header.size()==4 && header[0]=="vdb_tool" &&
            openvdb::vdb_tool::is_int(header[1], v[0]) &&
            openvdb::vdb_tool::is_int(header[2], v[1]) &&
            openvdb::vdb_tool::is_int(header[3], v[2]) &&
            v[0] != openvdb::vdb_tool::Tool::major())  {
            std::ofstream tmp("tmp.txt");
            tmp << "vdb_tool " << openvdb::vdb_tool::Tool::version() << std::endl;
            tmp << file.rdbuf();
            tmp.close();
            file.close();
            std::remove(argv[i]);
            std::rename("tmp.txt", argv[i]);
        }
    }
    return 0;
}