import torch
import subprocess
from typing import Optional

import pandas as pd
from rich.table import Table


def encode_range_name(name: str, method: str, attr: dict) -> str:
    """Encode the range name."""
    return f"{name}::{method}::{attr}"


def decode_range_name(name: str) -> tuple:
    """Decode the range name."""
    return tuple(name.split("::"))


def is_range_name(name: str) -> bool:
    """Check if the name is a range name."""
    return len(name.split("::")) == 3


def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Optional[Table] = None,
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    rich_table = rich_table or Table()

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)
        rich_indexes = pandas_dataframe.index.to_list()

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(rich_indexes[index])] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table


L2_CACHE_TABLE = {'NVIDIA RTX 6000 Ada Generation':96,
                'NVIDIA GeForce RTX 4090':72,}
L2_CACHE_BUFFER = None


def create_l2_cache():
    global L2_CACHE_BUFFER

    device_name = torch.cuda.get_device_name(0)

    if device_name not in L2_CACHE_TABLE:
        raise NotImplementedError(f"Cache size for {device_name} is not known.")

    L2_CACHE_SIZE = L2_CACHE_TABLE[device_name] # MB.
    L2_CACHE_BUFFER = torch.empty(int(L2_CACHE_SIZE * (1024 ** 2)), dtype=torch.int8, device='cuda')


def flush_l2_cache():
    assert isinstance(L2_CACHE_BUFFER, torch.Tensor), "L2 cache not initialized."
    L2_CACHE_BUFFER.zero_()


def current_gpu_memory_usage() -> Optional[int]:
    try:
        # Run the nvidia-smi command and capture the output
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )

        # Check if the command was successful
        if result.returncode == 0:
            # Parse the output to extract memory usage
            memory_usage = [int(value.strip()) for value in result.stdout.strip().split('\n')]

            # Return the memory usage
            return memory_usage[0]
        else:
            # Print an error message if the command failed
            print(f"Error running nvidia-smi: {result.stderr}")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
