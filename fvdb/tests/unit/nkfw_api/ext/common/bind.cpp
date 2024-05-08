#include <torch/extension.h>
#include <optional>
#include "hashmap_cuda.cuh"

// Hash computation
at::Tensor hash_cuda(const at::Tensor idx);
at::Tensor kernel_hash_cuda(const at::Tensor idx, const at::Tensor kernel_offset);

// Hash queries (kernel queries should be flattened beforehand)
HashLookupData build_hash_table(const at::Tensor hash_target, const at::Tensor idx_target, bool enlarge);
at::Tensor hash_table_query(const HashLookupData& hash_data, const at::Tensor hash_query);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hash_cuda", &hash_cuda, "");
    m.def("kernel_hash_cuda", &kernel_hash_cuda, "");
    m.def("build_hash_table", &build_hash_table);
    m.def("hash_table_query", &hash_table_query, "");
    py::class_<HashLookupData>(m, "HashLookupData");
}
