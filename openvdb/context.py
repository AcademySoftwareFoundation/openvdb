
from dd.runtime import Context


class Openvdb(Context):

    def __init__(self, version=None):
        super(Openvdb, self).__init__(package='openvdb', version=version)

    def setupEnvironment(self):
        self.environ["OPENVDB_PACKAGE_ROOT"] = self.package_root
        self.environ["OPENVDB_VERSION"] = self.version

        self.environ["OPENVDB_INCLUDE_PATH"] = self.expandPaths(
          "$OPENVDB_PACKAGE_ROOT/include"
        )
        self.environ["OPENVDB_LIBRARY_PATH"] = self.expandPaths(
          "$OPENVDB_PACKAGE_ROOT/lib"
        )
        # LD_LIBRARY path
        self.environ[ 'LD_LIBRARY_PATH' ] = self.expandPaths(
          '$OPENVDB_PACKAGE_ROOT/lib',
            '$LD_LIBRARY_PATH'
        )
            
