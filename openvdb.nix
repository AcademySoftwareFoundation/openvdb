{ nixpkgs, ... }:
let
  pkgs = import nixpkgs {
    system = "x86_64-linux";
    overlays = [ (import ./overlay.nix) ];
  };
in
  with pkgs;
  stdenv.mkDerivation {
    name = "openvdb";
    version = "9.0.0";

  # https://nixos.org/nix/manual/#builtin-filterSource
  src = builtins.filterSource
  (path: type: lib.cleanSourceFilter path type
  && baseNameOf path != "doc/*"
  && baseNameOf path != "openvdb_houdini/*"
  && baseNameOf path != "openvdb_maya/*"
  && baseNameOf path != "pendingchanges/*"
  && baseNameOf path != "tsc/*") ./.;

  cmakeFlags =["-DOPENVDB_BUILD_VDB_VIEW=ON"];

  # easily maxes out RAM on github actions or systems <64 GB
  enableParallelBuilding = true;
  nativeBuildInputs = [ cmake pkg-config ];

    # required dependencies for downstream development
    propagatedBuildInputs = [
      openexr
      tbb
      c-blosc
      boost175
    ];

    buildInputs = [
      unzip jemalloc ilmbase
      # for the optional VDB_VIEW binary opengl related dependencies:
      libGL glfw3 x11 libGLU xorg.libXdmcp
    ];

  }
