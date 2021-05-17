{ nixpkgs, ... }:
let
  pkgs = import nixpkgs { system = "x86_64-linux"; };
in
  with pkgs;
  stdenv.mkDerivation {
    name = "openvdb";
    version = "8.0.1";

  # https://nixos.org/nix/manual/#builtin-filterSource
  src = builtins.filterSource
  (path: type: lib.cleanSourceFilter path type
  && baseNameOf path != "doc/*"
  && baseNameOf path != "openvdb_houdini/*"
  && baseNameOf path != "openvdb_maya/*"
  && baseNameOf path != "pendingchanges/*"
  && baseNameOf path != "tsc/*") ./.;

  cmakeFlags =["-DOPENVDB_BUILD_VDB_VIEW=ON -DOPENVDB_BUILD_VDB_RENDER=ON -DOPENVDB_BUILD_VDB_LOD=ON"];

  enableParallelBuilding = true;
  nativeBuildInputs = [ cmake pkg-config ];

    # required dependencies for downstream development
    propagatedBuildInputs = [
      openexr
      tbb
      c-blosc
    ];

    buildInputs = [
      unzip boost jemalloc ilmbase
      # for the optional VDB_VIEW binary opengl related dependencies:
      libGL glfw3 x11 libGLU xorg.libXdmcp
    ];

  }
