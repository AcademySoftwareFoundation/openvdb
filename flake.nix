{
  description = "openvdb";
  inputs = {
    # pick latest commit from stable branch and test it so no suprises
    nixpkgs.url = "github:NixOS/nixpkgs/d53978239b265066804a45b7607b010b9cb4c50c";
  };

  outputs = inputs: {

    defaultPackage.x86_64-linux = import ./openvdb.nix {
      nixpkgs = inputs.nixpkgs;
    };
  };
}
