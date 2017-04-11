
from dd.runtime import Context


class Pyopenvdb(Context):

    def __init__(self, version=None):
        super(Pyopenvdb, self).__init__(package='pyopenvdb', version=version)

    def setupEnvironment(self):
        self.environ["PYOPENVDB_PACKAGE_ROOT"] = self.package_root
        self.environ["PYOPENVDB_VERSION"] = self.version

        self.environ["PYOPENVDB_INCLUDE_PATH"] = self.expandPaths(
          "$PYOPENVDB_PACKAGE_ROOT/include"
        )
            
        self.environ["PYTHONPATH"] = self.expandPaths(
            os.path.join(self.package_root, "python", self.python_version), "$PYTHONPATH"
        )

