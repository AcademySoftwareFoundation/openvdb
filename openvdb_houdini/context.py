
from dd.runtime import Context
from houdini_plugin_util import findNearestBuild


class Openvdb_houdini(Context):

    def __init__(self, version=None):
        super(Openvdb_houdini, self).__init__(package='openvdb_houdini',
                                              version=version)

    def setupEnvironment(self):
        self.environ['OPENVDB_HOUDINI_PACKAGE_ROOT'] = self.package_root
        self.environ['OPENVDB_HOUDINI_VERSION'] = self.version
        
        path = findNearestBuild( self.expandPaths('$OPENVDB_HOUDINI_PACKAGE_ROOT/houdini'),
                                 self.environ['HOUDINI_VERSION'])

        if path:
        
            self.environ['OPENVDB_HOUDINI_INCLUDE_PATH'] = '%s/include'%path
            self.environ['OPENVDB_HOUDINI_LIBRARY_PATH'] = '%s/lib'%path

            self.environ[ 'HOUDINI_SCRIPT_PATH' ] = self.expandPaths(
                '%s/scripts'%path,
                '$HOUDINI_SCRIPT_PATH'
            )
        
            self.environ[ 'HOUDINI_DSO_PATH' ] = self.expandPaths(
                '%s/dso'%path,
                '$HOUDINI_DSO_PATH'
            )
        
            self.environ[ 'LD_LIBRARY_PATH' ] = self.expandPaths(
                '$OPENVDB_HOUDINI_LIBRARY_PATH', 
                '$LD_LIBRARY_PATH'
            )
