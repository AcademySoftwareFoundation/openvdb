import numpy as np


have_pxr_lib = False
try:
    # conditionally load so it is an optional dependency
    import pxr  # USD support
    from pxr import Usd, UsdGeom
    have_pxr_lib = True
except ImportError:
    pass

# See https://github.com/ColinKennedy/USD-Cookbook/blob/master/tricks/traverse_instanced_prims/README.md
def traverse_instanced_children(prim):
    """Get every Prim child beneath `prim`, even if `prim` is instanced.

    Important:
        If `prim` is instanced, any child that this function yields will
        be an instance proxy.

    Args:
        prim (`pxr.Usd.Prim`): Some Prim to check for children.

    Yields:
        `pxr.Usd.Prim`: The children of `prim`.

    """
    for child in prim.GetFilteredChildren(Usd.TraverseInstanceProxies()):
        yield child

        for subchild in traverse_instanced_children(child):
            yield subchild


def simple_triangulate_faces(face_vertex_counts, face_vertex_inds_flat):
    """
    Given a polygonal mesh (with arbitrary degree faces) in a flat list representation, triangulates the faces, and returns a new (Tx3) numpy array.

    This is a naive pure-python for-loop, so it will be slow for large meshes.
    TODO write some clever numpy or something.
    """

    tri_faces = []
    i_start = 0
    for i_face in range(len(face_vertex_counts)):
        D = face_vertex_counts[i_face]
        i_end = i_start + D
        for k in range(1, D - 1):
            # fan tesselation with first vertex as root
            tri_faces.append((face_vertex_inds_flat[i_start],
                              face_vertex_inds_flat[i_start + k],
                              face_vertex_inds_flat[i_start + k + 1]))

        i_start = i_end

    return np.array(tri_faces, dtype=face_vertex_inds_flat.dtype)


def merge_meshes(verts_list, faces_list):
    """Merge multiple meshes into a single mesh.

    Concatenates vertices and adjusts face indices to account for the offset
    from combining vertex arrays.

    Args:
        verts_list: List of vertex arrays, each (N_i, 3)
        faces_list: List of face index arrays, each (M_i, 3) or (M_i, 4)

    Returns:
        Tuple of (combined_verts, combined_faces):
        - combined_verts: (N_total, 3) array with all vertices
        - combined_faces: (M_total, 3/4) array with adjusted face indices
    """
    N = len(verts_list)

    # Shift face indices to account for vertex concatenation
    # Each mesh's face indices need to be offset by the total number
    # of vertices from all previous meshes
    v_sum = 0  # Running sum of vertex counts
    faces_list_shifted = []
    for i in range(N):
        # Add vertex offset to all face indices in this mesh
        faces_list_shifted.append(faces_list[i] + v_sum)
        v_sum += verts_list[i].shape[0]

    # Concatenate all vertices and adjusted faces
    combined_verts = np.concatenate(verts_list, axis=0)
    combined_faces = np.concatenate(faces_list_shifted, axis=0)

    return combined_verts, combined_faces


def get_world_transform(prim):
    """
    Get the local transformation of a prim using Xformable.
    See https://graphics.pixar.com/usd/release/api/class_usd_geom_xformable.html
    Args:
        prim: The prim to calculate the world transformation.
    Returns:
        4x4 world object --> world transformation matrix
    """

    # https://docs.omniverse.nvidia.com/prod_kit/prod_kit/programmer_ref/usd/transforms/get-world-transforms.html

    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default(
    )  # The time at which we compute the bounding box
    world_transform = xform.ComputeLocalToWorldTransform(time)
    mat = np.array(world_transform)
    return mat

    # translation: Gf.Vec3d = world_transform.ExtractTranslation()
    # rotation: Gf.Rotation = world_transform.ExtractRotation()
    # scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    # return translation, rotation, scale


class USDInterface:

    def __init__(self, usd_path, verbose=True):

        if not have_pxr_lib:
            raise ValueError("USD support is not available, try `pip install usd-core`")

        self.usd_path = usd_path
        self.verbose = verbose
        self.load_file()

    def load_file(self):
        """Load and process all meshes from the USD file.

        This method:
        1. Opens the USD stage and discovers all mesh prims (including instanced)
        2. Extracts vertices and faces from each mesh
        3. Triangulates all n-gon faces using fan tessellation
        4. Applies world transformations to vertices
        5. Merges all meshes into single arrays
        6. Converts coordinate system from Z-up to Y-up

        Sets instance variables:
            self.stage: USD stage object
            self.root_prim: Root prim of the stage
            self.mesh_prims: List of UsdGeom.Mesh prims found
            self.merged_verts: (N, 3) array of all vertices in world space
            self.merged_faces: (M, 3) array of all triangle faces
            self.merged_faces_source_prim: (M,) array indicating source prim index for each face
        """
        # Open the USD file
        if self.verbose:
            print(f"USD: loading {self.usd_path}")
        self.stage = Usd.Stage.Open(self.usd_path)
        self.root_prim = self.stage.GetPseudoRoot()

        # Discover all mesh prims in the scene hierarchy
        # traverse_instanced_children handles instanced geometry properly
        self.mesh_prims = []
        for prim in traverse_instanced_children(self.root_prim):
            if self.verbose:
                print(f"  USD traversing: {prim.GetPath()}")

            if prim.IsA(UsdGeom.Mesh):
                self.mesh_prims.append(prim)

        if self.verbose:
            print(f"USD: found {len(self.mesh_prims)} mesh prims in file")

        # Collect vertices and faces from all meshes
        verts_list = []
        faces_list = []
        face_source_prim_list = []  # Track which prim each face came from

        for ip, p in enumerate(self.mesh_prims):
            if self.verbose:
                print(f"  mesh prim: {p.GetPath()}")
            mesh = UsdGeom.Mesh(p)

            # Extract face topology (USD stores as flat arrays)
            # face_vertex_counts: [3, 4, 3, ...] - vertices per face
            # face_vertex_inds_flat: [0, 1, 2, 0, 2, 3, 4, ...] - flattened indices
            face_vertex_counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.longlong)
            face_vertex_inds_flat = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.longlong)

            # Triangulate all faces (converts quads/n-gons to triangles)
            tri_faces = simple_triangulate_faces(face_vertex_counts, face_vertex_inds_flat)
            vertices = np.array(mesh.GetPointsAttr().Get())

            # Apply world transformation to get vertices in world space
            # This handles translation, rotation, scale, and parenting
            tmat = get_world_transform(p)
            # Convert to homogeneous coordinates (add w=1)
            vertices_homog = np.concatenate((vertices, np.ones_like(vertices[:, 0:1])), axis=-1)
            # Apply 4x4 transform and extract xyz
            vertices = (vertices_homog @ tmat)[:, :3]

            verts_list.append(vertices)
            faces_list.append(tri_faces)

            # Tag each face with its source prim index for debugging
            face_source_prim = np.zeros_like(tri_faces[:, 0]) + ip
            face_source_prim_list.append(face_source_prim)

        # Merge all meshes into single vertex/face arrays
        self.merged_verts, self.merged_faces = merge_meshes(verts_list, faces_list)
        self.merged_faces_source_prim = np.concatenate(face_source_prim_list, axis=0)

        # Convert from Z-up (USD convention) to Y-up (OpenVDB convention)
        # np.roll shifts axis order: [x, y, z] -> [z, x, y] -> use as [x, y, z] in Y-up
        self.merged_verts = np.roll(self.merged_verts, 2, axis=-1)

        if False:
            import polyscope as ps  # for debugging only
            ps.init()
            ps.set_up_dir("y_up")
            ps_mesh = ps.register_surface_mesh("usd mesh", self.merged_verts, self.merged_faces)
            ps_mesh.add_scalar_quantity("source prim", self.merged_faces_source_prim, defined_on='faces')
            ps.show()

        if self.verbose:
            print(f"USD: loaded full mesh of {self.merged_verts.shape[0]} verts and {self.merged_faces.shape[0]} faces.")

    @staticmethod
    def write_file(usd_path, points, triangles=None, quads=None, mesh_name="Mesh"):
        """Write mesh to USD file using Pixar's USD library.

        Preserves both triangles and quads without triangulation.

        Args:
            usd_path: Output USD file path
            points: (N, 3) numpy array of vertices
            triangles: (M, 3) numpy array or None
            quads: (K, 4) numpy array or None
            mesh_name: Name for the mesh prim (default: "Mesh")

        Raises:
            ValueError: If USD support not available or no triangles/quads provided
        """
        if not have_pxr_lib:
            raise ValueError("USD support is not available, try `pip install usd-core`")

        from pxr import Vt, Sdf

        # Create new stage
        stage = Usd.Stage.CreateNew(usd_path)

        # Create mesh prim
        mesh_path = Sdf.Path(f"/{mesh_name}")
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        # Set vertices
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))

        # Combine triangles and quads into face data
        face_vertex_indices = []
        face_vertex_counts = []

        if triangles is not None and len(triangles) > 0:
            for tri in triangles:
                face_vertex_indices.extend([int(x) for x in tri])
                face_vertex_counts.append(3)

        if quads is not None and len(quads) > 0:
            for quad in quads:
                face_vertex_indices.extend([int(x) for x in quad])
                face_vertex_counts.append(4)

        if not face_vertex_indices:
            raise ValueError("No triangles or quads provided")

        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))

        # Save stage
        stage.GetRootLayer().Save()
