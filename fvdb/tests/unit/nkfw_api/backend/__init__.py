
def load_backend(backend: str):
    if backend == 'hash_table':
        from backend import hash_table
        return hash_table
    elif backend == 'fvdb':
        from backend import fvdb
        return fvdb
    else:
        raise NotImplementedError
