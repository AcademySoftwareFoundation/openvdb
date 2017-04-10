
from dd.runtime import Context


class Openvdb_maya(Context):

    def __init__(self, version=None):
        super(Openvdb_maya, self).__init__(package='openvdb_maya', version=version)

    def setupEnvironment(self):
        self.environ["OPENVDB_MAYA_PACKAGE_ROOT"] = self.package_root
        self.environ["OPENVDB_MAYA_VERSION"] = self.version

        self.environ["OPENVDB_MAYA_INCLUDE_PATH"] = self.expandPaths(
            '$OPENVDB_MAYA_PACKAGE_ROOT/maya/$MAYA_VERSION_MAJOR/include'
        )
        self.environ["OPENVDB_MAYA_LIBRARY_PATH"] = self.expandPaths(
            '$OPENVDB_MAYA_PACKAGE_ROOT/maya/$MAYA_VERSION_MAJOR/lib'
        )

        # LD_LIBRARY path
        self.environ[ 'LD_LIBRARY_PATH' ] = self.expandPaths(
            '$OPENVDB_MAYA_PACKAGE_ROOT/maya/$MAYA_VERSION_MAJOR/lib',
            '$LD_LIBRARY_PATH'
        )
            
        self.environ[ 'MAYA_PLUG_IN_PATH'] = self.expandPaths(
            '$OPENVDB_MAYA_PACKAGE_ROOT/maya/$MAYA_VERSION_MAJOR/plug-ins',
            '$MAYA_PLUG_IN_PATH'
        )

        self.environ[ 'MAYA_SCRIPT_PATH'] = self.expandPaths(
            '$OPENVDB_MAYA_PACKAGE_ROOT/maya/$MAYA_VERSION_MAJOR/scripts',
            '$MAYA_SCRIPT_PATH'          
        )
