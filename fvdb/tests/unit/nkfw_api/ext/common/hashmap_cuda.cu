#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <c10/cuda/CUDAException.h>

#include "hashmap_cuda.cuh"

__device__ static uint64_t atomicExch(uint64_t *addr, uint64_t val) {
    return (uint64_t) atomicExch((unsigned long long int *) addr,
                                 (unsigned long long int) val);
}

__global__ void cuckooBucketKernel_Multi(
        uint64_t *const key_buf, uint64_t *const val_buf, const int size,
        const uint64_t *const keys, const uint64_t *const vals, const int n,
        int *const counters, const int num_buckets) {
    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Only threads within range are active.
    if (idx < n) {
        // Do 1st-level hashing to get bucket id, then do atomic add to get index
        // inside the bucket.
        uint64_t key = keys[idx];
        uint64_t val = vals ? vals[idx] : idx;

        int bucket_num = do_1st_hash(key, num_buckets);
        int bucket_ofs = atomicAdd(&counters[bucket_num], 1);

        // Directly write the key into the table buffer.
        if (bucket_ofs >= BUCKET_SIZE) {
            printf("%d/%d ERROR: bucket overflow! (n=%d, bucket_num=%d/%d, key=%d)\n",
                   bucket_ofs, BUCKET_SIZE, n, bucket_num, num_buckets, key);
        } else {
            key_buf[bucket_num * BUCKET_SIZE + bucket_ofs] = key;
            val_buf[bucket_num * BUCKET_SIZE + bucket_ofs] = val;
        }
    }
}

__global__ void cuckooInsertKernel_Multi(
        uint64_t *const key, uint64_t *const val, const uint64_t *const key_buf,
        const uint64_t *const val_buf, const int size,
        const FuncConfig *const hash_func_configs, const int num_funcs,
        const int *const counters, const int num_buckets, const int evict_bound,
        const int pos_width, int *const rehash_requests) {
    // Create local cuckoo table in shared memory. Size passed in as the third
    // kernel parameter.
    extern __shared__ uint64_t local_key[];
    for (int i = 0; i < num_funcs; ++i) {
        local_key[i * BUCKET_SIZE + threadIdx.x] = EMPTY_CELL;
    }

    // might be useful
    __syncthreads();

    // Get thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t cur_idx = idx;

    // Only threads within local bucket range are active.
    if (threadIdx.x < counters[blockIdx.x]) {
        // Set initial conditions.
        uint64_t cur_key = key_buf[cur_idx];
        int cur_func = 0;
        int evict_count = 0;

        // Start the test-kick-and-reinsert loops.
        do {
            int pos = do_2nd_hash(cur_key, hash_func_configs, cur_func, BUCKET_SIZE);

            uint64_t new_data = make_data(cur_idx + 1, cur_func, pos_width);

            uint64_t old_idx =
                    atomicExch(&local_key[cur_func * BUCKET_SIZE + pos], new_data);

            if (old_idx != EMPTY_CELL) {
                cur_idx = fetch_val(old_idx, pos_width) - 1;
                // potential overflow here. It seems that cur_idx < 0 is possible!
                cur_key = key_buf[cur_idx];
                cur_func = (fetch_func(old_idx, pos_width) + 1) % num_funcs;
                evict_count++;
            } else {
                break;
            }

        } while (evict_count < num_funcs * evict_bound);

        // If exceeds eviction bound, then needs rehashing.
        if (evict_count >= num_funcs * evict_bound) {
            atomicAdd(rehash_requests, 1);
        }
    }

    // Every thread write its responsible local slot into the global data table.
    __syncthreads();
    for (int i = 0; i < num_funcs; ++i) {
        uint64_t cur_idx = local_key[i * BUCKET_SIZE + threadIdx.x];
        if (cur_idx == EMPTY_CELL) {
            continue;
        }
        int cur_func = fetch_func(cur_idx, pos_width);
        cur_idx = fetch_val(cur_idx, pos_width) - 1;
        key[i * size + idx] = key_buf[cur_idx];
        val[i * size + idx] = val_buf[cur_idx];
    }
}

__global__ void cuckooLookupKernel_Multi(
        const uint64_t *const keys, uint64_t *const results, const int n,
        const uint64_t *const all_keys, const uint64_t *const all_vals,
        const int size, const FuncConfig *const hash_func_configs,
        const int num_funcs, const int num_buckets, const int pos_width) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Only threads within range are active.
    if (idx < n) {
        uint64_t key = keys[idx];
        results[idx] = hashtable_lookup(all_keys, all_vals, size, hash_func_configs,
                                        num_funcs, num_buckets, key);
    }
}

int CuckooHashTableCuda_Multi::insert_vals(const uint64_t *const keys,
                                           const uint64_t *const vals,
                                           uint64_t *d_key_buf,
                                           uint64_t *d_val_buf, uint64_t *d_key,
                                           uint64_t *d_val, const int n) {
    //
    // Phase 1: Distribute keys into buckets.
    //

    // Allocate GPU memory.

    int *d_counters = NULL;

    cudaMalloc((void **) &d_counters, _num_buckets * sizeof(int));

    cudaMemset(d_counters, 0, _num_buckets * sizeof(int));

    // Invoke bucket kernel.
    if (n > 0) {
        cuckooBucketKernel_Multi<<<ceil((double) n / BUCKET_SIZE), BUCKET_SIZE>>>(
                d_key_buf, d_val_buf, _size, keys, vals, n, d_counters, _num_buckets);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    //
    // Phase 2: Local cuckoo hashing.
    //

    // Allocate GPU memory.

    cudaDeviceSynchronize();
    int *d_rehash_requests = NULL;

    cudaMalloc((void **) &d_rehash_requests, sizeof(int));

    // Copy values onto GPU memory.
    cudaMemcpy(_d_hash_func_configs, _hash_func_configs,
               _num_funcs * sizeof(FuncConfig), cudaMemcpyHostToDevice);

    // Invoke insert kernel. Passes shared memory table size by the third
    // argument. Loops until no rehashing needed.

    int rehash_count = 0;
    do {
        int rehash_requests = 0;
        cudaMemset(d_rehash_requests, 0, sizeof(int));
        cuckooInsertKernel_Multi<<<ceil((double) _size / BUCKET_SIZE), BUCKET_SIZE,
        _num_funcs * BUCKET_SIZE * sizeof(uint64_t)>>>(
                d_key, d_val, d_key_buf, d_val_buf, _size, _d_hash_func_configs,
                _num_funcs, d_counters, _num_buckets, _evict_bound, _pos_width,
                d_rehash_requests);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cudaMemcpy(&rehash_requests, d_rehash_requests, sizeof(int),
                   cudaMemcpyDeviceToHost);

        if (rehash_requests == 0) {
            break;
        } else {
            rehash_count++;
            gen_hash_funcs();
            cudaMemcpy(_d_hash_func_configs, _hash_func_configs,
                       _num_funcs * sizeof(FuncConfig), cudaMemcpyHostToDevice);
        }
    } while (rehash_count < MAX_DEPTH);

    cudaDeviceSynchronize();

    // Free GPU resources.

    if (d_counters != NULL) {
        cudaFree(d_counters);
    }
    if (d_rehash_requests != NULL) {
        cudaFree(d_rehash_requests);
    }

    _inserted_size = n;
    return (rehash_count < MAX_DEPTH) ? rehash_count : ERR_DEPTH;
}

// kernel hashing: given data D and offset map K, generate D x K
// input N*4 int32 tensor, |K|*3 int32 tensor, output |K|*N int64 tensor
template<class T, int DimSize>
__global__ void kernel_hash_kernel(int N, int K, const T *__restrict__ data,
                                   const int *__restrict__ kernel_offset,
                                   int64_t *__restrict__ out) {
    extern __shared__ int kernel_offset_local[];

    for (int i = 0; i < K * 3; i++) {
        kernel_offset_local[i] = kernel_offset[i];
    }
    __syncthreads();

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int k = idx % K;
    int i = idx / K;
    T cur_coord[DimSize];
    if (i < N) {
        data += i * DimSize;
        for (int j = 0; j < 3; j++) {
            cur_coord[j] = data[j] + kernel_offset[k * 3 + j];
        }
        if (DimSize == 4) {
            cur_coord[3] = (T) data[3];
        }
        uint64_t hash = 14695981039346656037UL;
        for (int j = 0; j < DimSize; j++) {
            hash ^= (unsigned int) cur_coord[j];
            hash *= 1099511628211UL;
        }
        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
        out[k * N + i] = hash;
    }
}

template<class T, int DimSize>
void kernel_hash_wrapper(int N, int K, const T *data,
                         const int *kernel_offset, int64_t *out) {
    kernel_hash_kernel<T, DimSize><<<ceil((double) (N * K) / 512), 512, K * 3 * sizeof(int)>>>(
            N, K, data, kernel_offset, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// hashing
// input N*4 int32 tensor output N*1 int64 tensor
template<class T, int DimSize>
__global__ void hash_kernel(int N, const T *__restrict__ data,
                            int64_t *__restrict__ out) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    data += i * DimSize;
    uint64_t hash = 14695981039346656037UL;
    for (int j = 0; j < DimSize; j++) {
      hash ^= (unsigned int)data[j];
      hash *= 1099511628211UL;
    }
    hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
    out[i] = hash;
  }
}

template<class T, int DimSize>
void hash_wrapper(int N, const T *data, int64_t *out) {
    hash_kernel<T, DimSize><<<ceil((double) N / 512), 512>>>(N, data, out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor hash_cuda(const at::Tensor idx) {
    int N = idx.size(0);
    at::Tensor out =
            torch::zeros({N}, at::device(idx.device()).dtype(at::ScalarType::Long));

    switch (idx.size(1)) {
        case 2: {
            if (idx.dtype() == torch::ScalarType::Long) {
                hash_wrapper<int64_t, 2>(N, idx.data_ptr<int64_t>(), out.data_ptr<int64_t>());
            } else if (idx.dtype() == torch::ScalarType::Int) {
                hash_wrapper<int, 2>(N, idx.data_ptr<int>(), out.data_ptr<int64_t>());
            }
            break;
        }
        case 3: {
            if (idx.dtype() == torch::ScalarType::Long) {
                hash_wrapper<int64_t, 3>(N, idx.data_ptr<int64_t>(), out.data_ptr<int64_t>());
            } else if (idx.dtype() == torch::ScalarType::Int) {
                hash_wrapper<int, 3>(N, idx.data_ptr<int>(), out.data_ptr<int64_t>());
            }
            break;
        }
        case 4: {
            if (idx.dtype() == torch::ScalarType::Long) {
                hash_wrapper<int64_t, 4>(N, idx.data_ptr<int64_t>(), out.data_ptr<int64_t>());
            } else if (idx.dtype() == torch::ScalarType::Int) {
                hash_wrapper<int, 4>(N, idx.data_ptr<int>(), out.data_ptr<int64_t>());
            }
            break;
        }
        default: {
            std::cerr << "Error. Not compiled" << std::endl;
        }
    }

    return out;
}

at::Tensor kernel_hash_cuda(const at::Tensor idx,
                            const at::Tensor kernel_offset) {
    int N = idx.size(0);
    int K = kernel_offset.size(0);
    at::Tensor out = torch::zeros(
            {K, N}, at::device(idx.device()).dtype(at::ScalarType::Long));
    switch (idx.size(1)) {
        case 3: {
            if (idx.dtype() == torch::ScalarType::Long) {
                kernel_hash_wrapper<int64_t, 3>(
                        N, K, idx.data_ptr<int64_t>(), kernel_offset.data_ptr<int>(), out.data_ptr<int64_t>());
            } else if (idx.dtype() == torch::ScalarType::Int) {
                kernel_hash_wrapper<int, 3>(
                        N, K, idx.data_ptr<int>(), kernel_offset.data_ptr<int>(), out.data_ptr<int64_t>());
            }
            break;
        }
        case 4: {
            if (idx.dtype() == torch::ScalarType::Long) {
                kernel_hash_wrapper<int64_t, 4>(
                        N, K, idx.data_ptr<int64_t>(), kernel_offset.data_ptr<int>(), out.data_ptr<int64_t>());
            } else if (idx.dtype() == torch::ScalarType::Int) {
                kernel_hash_wrapper<int, 4>(
                        N, K, idx.data_ptr<int>(), kernel_offset.data_ptr<int>(), out.data_ptr<int64_t>());
            }
            break;
        }
        default: {
            std::cerr << "Error. Not compiled" << std::endl;
        }
    }
    return out;
}

HashLookupData build_hash_table(const at::Tensor hash_target, const at::Tensor idx_target, bool enlarge) {
    // When n is large, the hash values tend to be more evenly distrubuted and
    // choosing table_size to be 2 * nextPow2 typically suffices. For smaller n,
    // the effect of uneven distribution of hash values is more pronounced and
    // hence we choose table_size to be 4 * nextPow2 to reduce the chance of
    // bucket overflow.
    int n_source = hash_target.size(0);

    const int nextPow2 = pow(2, ceil(log2((double) n_source)));
    int table_size = (n_source < 2048) ? 4 * nextPow2 : 2 * nextPow2;
    if (enlarge) {
        table_size = 4 * nextPow2;
    }

    if (table_size < 512) {
        table_size = 512;
    }
    int num_funcs = 3;
    CuckooHashTableCuda_Multi in_hash_table(table_size, 8 * ceil(log2((double) n_source)),
                                            num_funcs);

    auto long_tensor_option = at::device(hash_target.device()).dtype(at::ScalarType::Long);
    at::Tensor key_buf = torch::zeros({table_size}, long_tensor_option);
    at::Tensor val_buf = torch::zeros({table_size}, long_tensor_option);
    at::Tensor hash_key = torch::zeros({num_funcs * table_size}, long_tensor_option);
    at::Tensor hash_val = torch::zeros({num_funcs * table_size}, long_tensor_option);

    bool default_idx = idx_target.size(0) == 0;
    in_hash_table.insert_vals((uint64_t * )(hash_target.data_ptr<int64_t>()),
                              default_idx ? nullptr : (uint64_t * )(idx_target.data_ptr<int64_t>()),
                              (uint64_t * )(key_buf.data_ptr<int64_t>()),
                              (uint64_t * )(val_buf.data_ptr<int64_t>()),
                              (uint64_t * )(hash_key.data_ptr<int64_t>()),
                              (uint64_t * )(hash_val.data_ptr<int64_t>()), n_source);

    auto hash_data = in_hash_table.get_data(hash_key, hash_val);

    return hash_data;
}

at::Tensor hash_table_query(const HashLookupData &hash_data, const at::Tensor hash_query) {
    int n1 = hash_query.size(0);
    auto hash_param = hash_data.get_param();

    at::Tensor out = torch::zeros(
            {n1}, at::device(hash_query.device()).dtype(at::ScalarType::Long));

    cuckooLookupKernel_Multi<<<ceil((double) n1 / BUCKET_SIZE), BUCKET_SIZE>>>(
            (uint64_t * )(hash_query.data_ptr<int64_t>()),
            (uint64_t * )(out.data_ptr<int64_t>()), n1,
            hash_param.d_key, hash_param.d_val,
            hash_param.size, hash_param.config, hash_param.num_funcs,
            hash_param.num_buckets, 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    hash_data.release_param(hash_param);
    return out;
}
