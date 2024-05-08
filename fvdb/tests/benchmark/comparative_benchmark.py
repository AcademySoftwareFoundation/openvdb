import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import rich
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torchsparse
import torchsparse.nn.functional
import fvdb.nn as fvnn
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity

from fvdb_benchmark.dataset import CoordsDataset
from fvdb_benchmark.configs import all_configs, BaseConfig
from fvdb_benchmark.utils import current_gpu_memory_usage, create_l2_cache, flush_l2_cache, \
    df_to_table, decode_range_name, encode_range_name, is_range_name



@torch.no_grad()
def main(args) -> None:

    # Setting the libraries up.
    torch.backends.cudnn.benchmark = False
    torchsparse.backends.allow_tf32 = True
    torchsparse.backends.hash_rsv_ratio = 4
    torchsparse.nn.functional.set_kmap_mode("hashmap_on_the_fly")
    conv_mode = torchsparse.nn.functional.ConvMode.mode1
    ts_cfg = torchsparse.nn.functional.conv_config.get_default_conv_config(conv_mode=conv_mode, training=False)
    ts_cfg.dataflow = torchsparse.nn.functional.Dataflow.ImplicitGEMM

    # Determine the configuration.
    if args.config not in all_configs:
        raise ValueError(f"Config {args.config} not found, available configs are {list(all_configs.keys())}")
    config = all_configs[args.config]()

    assert isinstance(config, BaseConfig), "Config not valid."

    # Iterate over all baselines.
    gross_data = {}
    detailed_data = {}

    pbar_baselines = tqdm(config.baselines)
    for baseline in pbar_baselines:
        pbar_baselines.set_description(f"Baseline: {baseline}")

        # Dataset
        dataset = CoordsDataset(
            config.dataset_paths,
            in_channels=config.in_channels,
            max_files=args.limit,
        )
        assert len(dataset) > 0, "No data found. Check if data is available."

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda batch: batch[0],
        )

        # Model
        model = config.make_model(baseline)
        model.cuda().eval()

        pbar = tqdm(dataloader)
        baseline_gross_data = []
        baseline_detailed_data = []

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start_gpu_memory_usage = current_gpu_memory_usage()

        for i, batch in enumerate(pbar):
            ijk_name, input_grid, input_feature = batch
            input_grid, input_feature = input_grid.to("cuda"), input_feature.cuda()
            pbar.set_description(f"{ijk_name}: voxel count {input_grid.total_voxels:,}")

            # Prepare inputs
            fvdb_input = fvnn.VDBTensor(input_grid, input_grid.jagged_like(input_feature))
            aux_dict = config.get_aux_inputs(fvdb_input)
            baseline_input = config.to_baseline_input(fvdb_input, baseline)

            # warmup only for the first data point for each baseline.
            if i == 0:
                for _ in range(5):
                    _ = model(baseline_input, **aux_dict)
                if isinstance(baseline_input, torchsparse.SparseTensor):
                    baseline_input._caches.cmaps.clear()
                    baseline_input._caches.kmaps.clear()
                    baseline_input._caches.hashmaps.clear()

            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
            detailed_record = defaultdict(list)

            for j in range(args.repeats):
                flush_l2_cache()
                torch.cuda.synchronize()
                torch.cuda._sleep(1_000_000)

                start_events[j].record()

                if args.detail:

                    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                        with record_function(encode_range_name("main", '-', {})):
                            baseline_out = model(baseline_input, **aux_dict)

                    all_events = prof.key_averages()
                    for event in all_events:
                        if is_range_name(event.key):
                            range_name, _, info_dict = decode_range_name(event.key)
                            table_name = f"{range_name}::{info_dict}"
                            detailed_record[table_name].append(event.cpu_time)

                else:
                    baseline_out = model(baseline_input, **aux_dict)

                end_events[j].record()
                torch.cuda.synchronize()

                config.post_measure(baseline_input, baseline_out)

            my_time = np.mean(
                [start_events[j].elapsed_time(end_events[j]) for j in range(args.repeats)]
            )
            my_memory = current_gpu_memory_usage() - start_gpu_memory_usage

            baseline_gross_data.append({
                "key": ijk_name,
                "time": my_time,
                # "memory": my_memory,      # We shouldn't use nvidia-smi to obtain memory usage.
            })
            baseline_detailed_data.append({
                "key": ijk_name,
                **{k: np.mean(v) for k, v in detailed_record.items()}
            })

        gross_data[baseline] = pd.DataFrame(baseline_gross_data)
        gross_data[baseline].set_index("key", inplace=True)
        detailed_data[baseline] = pd.DataFrame(baseline_detailed_data)
        detailed_data[baseline].set_index("key", inplace=True)

    # Combine and compare the results.
    gross_data_sample = next(iter(gross_data.values()))
    gross_col_names = list(gross_data_sample.columns)

    for col_name in gross_col_names:
        rich.print(f"------ Comparing {col_name} ------")
        full_dataframes = pd.concat(
            [df[col_name].rename(baseline) for baseline, df in gross_data.items()],
            axis=1
        )
        full_dataframes.loc['mean'] = full_dataframes.mean()
        rich.print(df_to_table(full_dataframes))

    if args.detail:
        detailed_data_sample = next(iter(detailed_data.values()))
        detailed_col_names = list(detailed_data_sample.columns)

        for col_name in detailed_col_names:
            rich.print(f"------ Comparing {col_name} ------")
            full_dataframes = pd.concat(
                [df[col_name].rename(baseline) for baseline, df in detailed_data.items() if col_name in df.columns],
                axis=1
            )
            full_dataframes.loc['mean'] = full_dataframes.mean()
            rich.print(df_to_table(full_dataframes))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="name of the config.")
    parser.add_argument("--no-clb", action="store_true", help="disable CUDA_LAUNCH_BLOCKING.")
    parser.add_argument("--detail", action="store_true", help="enable detailed profiling.")
    parser.add_argument("--repeats", type=int, default=1, help="number of repeats.")
    parser.add_argument("--limit", type=int, default=-1, help="limit the number of data points.")
    options = parser.parse_args()

    if not options.no_clb:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    create_l2_cache()

    main(options)
