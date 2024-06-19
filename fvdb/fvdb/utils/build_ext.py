import os

from torch.utils import cpp_extension
import fvdb


def FVDBExtension(name, sources, *args, **kwargs):
    """
    Utility function for creating pytorch extensions that depend on fvdb. You then have access to all fVDB's internal
    headers to program with. Example usage:

    .. code-block:: python

            from fvdb.utils import FVDBExtension

            ext = FVDBExtension(
                name='my_extension',
                sources=['my_extension.cpp'],
                extra_compile_args={'cxx': ['-std=c++17']},
                libraries=['mylib'],
            )

    :param name: The name of the extension.
    :param sources: The list of source files.
    :param args: Other arguments to pass to :func:`torch.utils.cpp_extension.CppExtension`.
    :param kwargs: Other keyword arguments to pass to :func:`torch.utils.cpp_extension.CppExtension`.
    :return: A :class:`torch.utils.cpp_extension.CppExtension` object.
    """

    libraries = kwargs.get('libraries', [])
    libraries.append('fvdb')
    kwargs['libraries'] = libraries

    library_dirs = kwargs.get('library_dirs', [])
    library_dirs.append(os.path.dirname(fvdb.__file__))
    kwargs['library_dirs'] = library_dirs

    include_dirs = kwargs.get('include_dirs', [])
    include_dirs.append(os.path.join(os.path.dirname(fvdb.__file__), 'include'))

    # We also need to add this because fvdb internally will refer to their headers without the fvdb/ prefix.
    include_dirs.append(os.path.join(os.path.dirname(fvdb.__file__), 'include/fvdb'))
    kwargs['include_dirs'] = include_dirs

    extra_link_args = kwargs.get('extra_link_args', [])
    extra_link_args.append(f'-Wl,-rpath={os.path.dirname(fvdb.__file__)}')
    kwargs['extra_link_args'] = extra_link_args

    extra_compile_args = kwargs.get('extra_compile_args', {})
    extra_compile_args['nvcc'] = extra_compile_args.get('nvcc', [])
    if '--extended-lambda' not in extra_compile_args['nvcc']:
        extra_compile_args['nvcc'].append('--extended-lambda')
    kwargs['extra_compile_args'] = extra_compile_args

    return cpp_extension.CUDAExtension(name, sources, *args, **kwargs)
