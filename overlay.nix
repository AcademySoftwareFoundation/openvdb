final: prev: {
  tbb = prev.stdenv.mkDerivation {
    name="tbb";
    version = "2021.4";

    src = prev.fetchFromGitHub {
      owner = "oneapi-src";
      repo = "oneTBB";
      rev = "v2021.4.0";
      sha256 = "eJ/NQ1XkWWlioBu05zbtZ/EwVxCAQzz5pkkKgN4RB0Y=";
    };

    nativeBuildInputs = [ prev.pkgs.cmake prev.pkgs.pkg-config ];

    makeFlags = prev.lib.optionals prev.stdenv.cc.isClang [
      "compiler=clang"
    ];

    cmakeFlags =["-DTBB_TEST=OFF"];

    enableParallelBuilding = true;

    meta = with prev.lib; {
      description = "Intel Thread Building Blocks C++ Library";
      homepage = "http://threadingbuildingblocks.org/";
      license = licenses.asl20;
      longDescription = ''
        Intel Threading Building Blocks offers a rich and complete approach to
        expressing parallelism in a C++ program. It is a library that helps you
        take advantage of multi-core processor performance without having to be a
        threading expert. Intel TBB is not just a threads-replacement library. It
        represents a higher-level, task-based parallelism that abstracts platform
        details and threading mechanisms for scalability and performance.
      '';
      platforms = platforms.unix;
      maintainers = with maintainers; [ thoughtpolice dizfer ];
    };
  };
}
