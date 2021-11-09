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

    nativeBuildInputs = prev.lib.optionals prev.stdenv.isDarwin [
      prev.fixDarwinDylibNames
    ];

    buildInputs = [ ];

    makeFlags = prev.lib.optionals prev.stdenv.cc.isClang [
      "compiler=clang"
    ];

    enableParallelBuilding = true;

    installPhase = ''
        runHook preInstall
        mkdir -p $out/lib
        cp "build/"*release*"/"*${prev.stdenv.hostPlatform.extensions.sharedLibrary}* $out/lib/
        mv include $out/
        rm $out/include/index.html
        runHook postInstall
    '';

    postInstall = let
      pcTemplate = prev.fetchurl {
        url = "https://github.com/oneapi-src/oneTBB/raw/master/integration/pkg-config/tbb.pc.in";
        sha256 = "2pCad9txSpNbzac0vp/VY3x7HNySaYkbH3Rx8LK53pI=";
      };
  in ''
    # Generate pkg-config file based on upstream template.
    # It should not be necessary with tbb after 2021.2.
    mkdir -p "$out/lib/pkgconfig"
    substitute "${pcTemplate}" "$out/lib/pkgconfig/tbb.pc" \
      --subst-var-by CMAKE_INSTALL_PREFIX "$out" \
      --subst-var-by CMAKE_INSTALL_LIBDIR "lib" \
      --subst-var-by CMAKE_INSTALL_INCLUDEDIR "include" \
      --subst-var-by TBB_VERSION 2021.4 \
      --subst-var-by TBB_LIB_NAME "tbb"
  '';

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
