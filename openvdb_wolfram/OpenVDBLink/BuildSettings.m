(* ::Package:: *)

Switch[$OperatingSystem,
    "MacOSX",
        $buildSettings = {
            "CompileOptions" -> {"-std=c++14 -ltbb -lHalf -lopenvdb -flto -L/usr/local/opt/ilmbase/lib -I/usr/local/opt/ilmbase/include"},
            "Compiler" -> CCompilerDriver`ClangCompiler`ClangCompiler
        };
        $libraryName = "OpenVDBLink.dylib",
    "Windows",
        $buildSettings = {
            "CompileOptions" -> {"/bigobj /EHsc /showIncludes /FC /diagnostics:classic "},
            "IncludeDirectories" -> {
                (* these might be avoidable with proper path variables? *)
                "C:\\Program Files\\OpenVDB\\include",
                "C:\\src\\vcpkg\\installed\\x64-windows\\include",
                "C:\\src\\openvdb\\include",
                "C:\\src\\vcpkg\\packages",
                "C:\\src\\vcpkg\\installed\\x64-windows\\bin",
                "C:\\src\\vcpkg\\installed\\x64-windows\\lib",
                "C:\\src\\vcpkg\\installed\\x64-windows\\include",
                "C:\\src\\vcpkg\\installed\\x64-windows\\include\\boost",
                "C:\\src\\vcpkg\\installed\\x64-windows\\include\\boost\\type_traits",
                "C:\\src\\vcpkg\\packages\\boost-numeric-conversion_x64-windows\\include",
                "C:\\src\\vcpkg\\packages\\boost-numeric-conversion_x64-windows\\include\\boost"
            },
            "ExtraObjectFiles" -> {
                "C:\\src\\openvdb\\build\\openvdb\\openvdb\\Release\\libopenvdb.lib",
                "C:\\src\\openvdb\\build\\openvdb\\openvdb\\Release\\openvdb.lib",
                "C:\\src\\vcpkg\\installed\\x64-windows\\lib\\blosc.lib",
                "C:\\src\\vcpkg\\installed\\x64-windows\\lib\\snappy.lib",
                "C:\\src\\vcpkg\\installed\\x64-windows\\lib\\tbbmalloc_proxy.lib",
                "C:\\src\\vcpkg\\installed\\x64-windows\\lib\\tbb.lib",
                "C:\\src\\vcpkg\\installed\\x64-windows\\lib\\tbbmalloc.lib",
                "C:\\src\\vcpkg\\installed\\x64-windows\\lib\\zlib.lib",
                "C:\\src\\vcpkg\\installed\\x64-windows\\lib\\zstd.lib"
            }
        };
        $libraryName = "OpenVDBLink.dll",
    _, (* TODO "Unix", etc. *)
        $buildSettings = None;
        $libraryName = ""
];
