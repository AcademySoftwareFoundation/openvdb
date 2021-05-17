{
  description = "openvdb";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/733e537a8ad76fd355b6f501127f7d0eb8861775";
  };

  outputs = inputs: {

    defaultPackage.x86_64-linux = import ./openvdb.nix {
      nixpkgs = inputs.nixpkgs;
    };
  };
}
