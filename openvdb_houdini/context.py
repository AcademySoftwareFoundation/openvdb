
from dd.runtime import Context


class Openvdb_houdini(Context):

    def __init__(self, version=None):
        super(Openvdb_houdini, self).__init__(package='openvdb_houdini',
                                              version=version)

    def setupEnvironment(self):
        self.environ['OPENVDB_HOUDINI_PACKAGE_ROOT'] = self.package_root
        self.environ['OPENVDB_HOUDINI_VERSION'] = self.version

        path = self.expandPaths(
            "$OPENVDB_HOUDINI_PACKAGE_ROOT/houdini/$HOUDINI_VERSION"
        )

        # Fallback to the major release if openvdb_houdini isn't built
        # for the specific version.
        if not path:
            # DD_HOUD_MAJ_VERSION -> $HOUDINI_MAJOR_RELEASE.$HOUDINI_MINIR_RELEASE
            path = self.expandPaths(
                '$OPENVDB_HOUDINI_PACKAGE_ROOT/houdini/$DD_HOUD_MAJ_VERSION'
            )
        
        self.environ['OPENVDB_HOUDINI_INCLUDE_PATH'] = '%/include'%path
        self.environ['OPENVDB_HOUDINI_LIBRARY_PATH'] = '%/lib'%path

        self.environ[ 'HOUDINI_SCRIPT_PATH' ] = self.expandPaths(
            '%/scripts'%path,
            '$HOUDINI_SCRIPT_PATH'
        )
        
        self.environ[ 'HOUDINI_DSO_PATH' ] = self.expandPaths(
            '%/dso'%path,
            '$HOUDINI_DSO_PATH'
        )
        
        self.environ[ 'LD_LIBRARY_PATH' ] = self.expandPaths(
            '$OPENVDB_HOUDINI_LIBRARY_PATH', 
            '$LD_LIBRARY_PATH'
        )
