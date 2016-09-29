
from dd.runtime import Context


class Openvdb_houdini(Context):

    def __init__(self, version=None):
        super(Openvdb_houdini, self).__init__(package='openvdb_houdini',
                                              version=version)

    def setupEnvironment(self):
        self.environ['OPENVDB_HOUDINI_PACKAGE_ROOT'] = self.package_root
        self.environ['OPENVDB_HOUDINI_VERSION'] = self.version

        # DD_HOUD_MAJ_VERSION -> $HOUDINI_MAJOR_RELEASE.$HOUDINI_MINIR_RELEASE
        
        self.environ['OPENVDB_HOUDINI_INCLUDE_PATH'] = self.expandPaths(
            self.package_root + '/houdini/$DD_HOUD_MAJ_VERSION/include'
        )

        self.environ['OPENVDB_HOUDINI_LIBRARY_PATH'] = self.expandPaths(
            self.package_root + '/houdini/$DD_HOUD_MAJ_VERSION/lib'
        )

        # Fallback to the major release if openvdb_houdini isn't built
        # for the specific version.
        self.environ[ 'LD_LIBRARY_PATH' ] = self.expandPaths(
            self.package_root + '/houdini/$DD_HOUD_MAJ_VERSION/lib',
            '$LD_LIBRARY_PATH'
        )
            
        self.environ[ 'LD_LIBRARY_PATH' ] = self.expandPaths(
            self.package_root + '/houdini/$HOUDINI_VERSION/lib',
            '$LD_LIBRARY_PATH'
        )

        # Script path
        self.environ[ 'HOUDINI_SCRIPT_PATH' ] = self.expandPaths(
            self.package_root + '/houdini/$DD_HOUD_MAJ_VERSION/scripts',
            '$HOUDINI_SCRIPT_PATH',
        )

        self.environ[ 'HOUDINI_SCRIPT_PATH' ] = self.expandPaths(
            self.package_root + '/houdini/$HOUDINI_VERSION/scripts',
            '$HOUDINI_SCRIPT_PATH',
        )

        # DSO path
        self.environ[ 'HOUDINI_DSO_PATH' ] = self.expandPaths(
            self.package_root + '/houdini/$DD_HOUD_MAJ_VERSION/dso',
            '$HOUDINI_DSO_PATH',
        )

        self.environ[ 'HOUDINI_DSO_PATH' ] = self.expandPaths(
            self.package_root + '/houdini/$HOUDINI_VERSION/dso',
            '$HOUDINI_DSO_PATH',
        )



