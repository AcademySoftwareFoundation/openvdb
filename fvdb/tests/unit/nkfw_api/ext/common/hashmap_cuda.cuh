#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <torch/torch.h>

#include "cuda_runtime.h"

/** Reserved value for indicating "empty". */
#define EMPTY_CELL (0)
/** Max rehashing depth, and error depth. */
#define MAX_DEPTH (100)
#define ERR_DEPTH (-1)
/** CUDA naive thread block size. */
#define BLOCK_SIZE (256)
/** CUDA multi-level thread block size = bucket size. */
#define BUCKET_SIZE (512)

/** Struct of a hash function config. */
typedef struct {
  int rv;  // Randomized XOR value.
  int ss;  // Randomized shift filter start position.
} FuncConfig;

/** Hard code hash functions and all inline helper functions for CUDA kernels'
 * use. */
inline __device__ int do_1st_hash(const uint64_t val, const int num_buckets) {
  return val % num_buckets;
}

inline __device__ int do_2nd_hash(const uint64_t val,
                                  const FuncConfig *const hash_func_configs,
                                  const int func_idx, const int size) {
  FuncConfig fc = hash_func_configs[func_idx];
  return ((val ^ fc.rv) >> fc.ss) % size;  // XOR function as 2nd-level hashing.
}

// trying to ignore EMPTY_CELL by adding 1 at make_data.
inline __device__ uint64_t fetch_val(const uint64_t data, const int pos_width) {
  return data >> pos_width;
}

inline __device__ int fetch_func(const uint64_t data, const int pos_width) {
  return data & ((0x1 << pos_width) - 1);
}

inline __device__ uint64_t make_data(const uint64_t val, const int func,
                                     const int pos_width) {
  return (val << pos_width) ^ func;
}

struct HashLookupParam {
    uint64_t *d_key;
    uint64_t *d_val;
    int size;
    FuncConfig* config = nullptr;
    int num_funcs;
    int num_buckets;
};

class HashLookupData {
    HashLookupParam _param;
    friend class CuckooHashTableCuda_Multi;
public:
    int inserted_size;
    torch::Tensor keys;
    torch::Tensor vals;
    std::vector<FuncConfig> config;
    HashLookupParam get_param() const {
        HashLookupParam p = _param;
        cudaMalloc((void **)&p.config, p.num_funcs * sizeof(FuncConfig));
        cudaMemcpy(p.config, config.data(),
                   p.num_funcs * sizeof(FuncConfig), cudaMemcpyHostToDevice);
        p.d_key = (uint64_t *)(keys.data_ptr<int64_t>());
        p.d_val = (uint64_t *)(vals.data_ptr<int64_t>());
        return p;
    }
    torch::Device device() const {
        return this->keys.device();
    }
    void release_param(const HashLookupParam& param) const {
        cudaFree(param.config);
    }
};

class CuckooHashTableCuda_Multi {
 private:
  const int _size;
  const int _evict_bound;
  const int _num_funcs;
  const int _pos_width;
  const int _num_buckets;
  int _inserted_size;

  FuncConfig *_d_hash_func_configs;

  /** Cuckoo hash function set. */
  FuncConfig *_hash_func_configs;

  /** Private operations. */
  void gen_hash_funcs() {
    // Calculate bit width of value range and table size.
    int val_width = 8 * sizeof(uint64_t) - ceil(log2((double)_num_funcs));
    int bucket_width = ceil(log2((double)_num_buckets));
    int size_width = ceil(log2((double)BUCKET_SIZE));
    // Generate randomized configurations.
    for (int i = 0; i < _num_funcs; ++i) {  // At index 0 is a dummy function.
      if (val_width - bucket_width <= size_width)
        _hash_func_configs[i] = {rand(), 0};
      else {
        _hash_func_configs[i] = {
            rand(), rand() % (val_width - bucket_width - size_width + 1) +
                        bucket_width};
      }
    }
  };

  inline uint64_t fetch_val(const uint64_t data) { return data >> _pos_width; }
  inline int fetch_func(const uint64_t data) {
    return data & ((0x1 << _pos_width) - 1);
  }

 public:
  CuckooHashTableCuda_Multi(const int size, const int evict_bound,
                            const int num_funcs)
      : _size(size),
        _evict_bound(evict_bound),
        _num_funcs(num_funcs),
        _pos_width(ceil(log2((double)_num_funcs))),
        _num_buckets(ceil((double)_size / BUCKET_SIZE)) {
    srand(time(NULL));
    _d_hash_func_configs = NULL;
    _hash_func_configs = NULL;
    _hash_func_configs = new FuncConfig[num_funcs];

    gen_hash_funcs();

    cudaMalloc((void **)&_d_hash_func_configs, _num_funcs * sizeof(FuncConfig));
    cudaMemcpy(_d_hash_func_configs, _hash_func_configs,
               _num_funcs * sizeof(FuncConfig), cudaMemcpyHostToDevice);
  };
  ~CuckooHashTableCuda_Multi() {
    if (_hash_func_configs != NULL) delete[] _hash_func_configs;

    if (_d_hash_func_configs != NULL) cudaFree(_d_hash_func_configs);
  };

  HashLookupData get_data(torch::Tensor keys, torch::Tensor vals) const {
      HashLookupData data;
      data._param.size = _size;
      data._param.num_funcs = _num_funcs;
      data._param.num_buckets = _num_buckets;
      for (int i = 0; i < _num_funcs; ++i) {
          data.config.push_back(_hash_func_configs[i]);
      }
      data.vals = vals;
      data.keys = keys;
      data.inserted_size = _inserted_size;
      return data;
  }

  int insert_vals(const uint64_t *const keys, const uint64_t *const vals,
                  uint64_t *d_key_buf, uint64_t *d_val_buf, uint64_t *d_key,
                  uint64_t *d_val, const int n);

};

__global__ void cuckooBucketKernel_Multi(
    uint64_t *const key_buf, uint64_t *const val_buf, const int size,
    const uint64_t *const keys, const uint64_t *const vals, const int n,
    int *const counters, const int num_buckets);

__global__ void cuckooInsertKernel_Multi(
    uint64_t *const key, uint64_t *const val, const uint64_t *const key_buf,
    const uint64_t *const val_buf, const int size,
    const FuncConfig *const hash_func_configs, const int num_funcs,
    const int *const counters, const int num_buckets, const int evict_bound,
    const int pos_width, int *const rehash_requests);

__global__ void cuckooLookupKernel_Multi(
    const uint64_t *const keys, uint64_t *const results, const int n,
    const uint64_t *const all_keys, const uint64_t *const all_vals,
    const int size, const FuncConfig *const hash_func_configs,
    const int num_funcs, const int num_buckets, const int pos_width);


inline __device__ uint64_t hashtable_lookup(
    const uint64_t *const all_keys, const uint64_t *const all_vals,
    const int size, const FuncConfig *const hash_func_configs,
    const int num_funcs, const int num_buckets, uint64_t key) {

    int bucket_num = do_1st_hash(key, num_buckets);
    for (int i = 0; i < num_funcs; ++i) {
        int pos = bucket_num * BUCKET_SIZE +
                  do_2nd_hash(key, hash_func_configs, i, BUCKET_SIZE);
        if (all_keys[i * size + pos] == key) {
            return all_vals[i * size + pos] + 1;
        }
    }
    return EMPTY_CELL;
}

inline __device__ uint64_t hash2(unsigned int a, unsigned int b) {
    uint64_t hash = 14695981039346656037UL;
    hash ^= a;
    hash *= 1099511628211UL;
    hash ^= b;
    hash *= 1099511628211UL;
    hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
    return hash;
}

template<class T>
inline __device__ uint64_t hash3(const T a, const T b, const T c) {
    uint64_t hash = 14695981039346656037UL;
    hash ^= (unsigned int) a;
    hash *= 1099511628211UL;
    hash ^= (unsigned int) b;
    hash *= 1099511628211UL;
    hash ^= (unsigned int) c;
    hash *= 1099511628211UL;
    hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
    return hash;
}
