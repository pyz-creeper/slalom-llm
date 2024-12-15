#ifndef SGXDNN_EMBEDDING_H_
#define SGXDNN_EMBEDDING_H_

#include "assert.h"
#include <iostream>
#include <string>
#include "../mempool.hpp"
#include "layer.hpp"

#include "../Crypto.h"

#ifdef USE_SGX
#include "Enclave.h"
#endif


using namespace tensorflow;

namespace SGXDNN
{

	template <typename T> class Embedding: public Layer<T>
	{
	public:
		explicit Embedding(
				const std::string& name,
				const array4d input_shape,
                const int num_feature,
                const int max_idx,
                const int seq_len,
				MemPool* mem_pool,
                T* kernel
				): Layer<T>(name, input_shape),
				mem_pool_(mem_pool),
				kernel_data_(nullptr),
				kernel_(NULL, num_feature),
				num_feature_(num_feature),
                max_idx_(max_idx)
		{
			output_shape_ = {1,1,seq_len, num_feature};
			output_size_ = output_shape_[2] * output_shape_[3];
            // kernel_data_ = mem_pool -> alloc<T>(num_feature * max_idx);
            // std::copy(kernel, kernel + num_feature*max_idx, kernel_data_);
			kernel_data_ = kernel;
		}

		array4d output_shape() override
		{
			return output_shape_;
		}

		int output_size() override
		{
			return output_size_;
		}

		int num_linear() override
		{
			return 2;
		}

	protected:

		TensorMap<T, 4> apply_impl(TensorMap<T, 4> input, void* device_ptr = NULL, bool release_input = true) override
		{
            // const int batch_size = input.dimension(2);
            const int seq_len = input.dimension(3);    // Sequence length (e.g., number of tokens per sample)

            // Allocate output for the embedding results
            auto output_data = mem_pool_->alloc<T>(seq_len * num_feature_);
            auto output = TensorMap<T, 4>(output_data, output_shape_);
            for (int t = 0; t < seq_len; ++t) {  // Loop over sequence length (e.g., tokens per sample)
				int idx = (int)(*(input.data() + t));
                if (idx >= max_idx_ || idx < 0) {
                    throw std::out_of_range("Index out of bounds in embedding lookup.");
                }
                auto embedding_ptr = kernel_data_ + idx * num_feature_;
                for (int f = 0; f < num_feature_; ++f) {
                    *(output.data() + f + t*num_feature_) = embedding_ptr[f];
                }
            }
            if (release_input) {
                mem_pool_->release(input.data());
            }
            return output;
        }

		TensorMap<T, 4> fwd_verify_impl(TensorMap<T, 4> input, float** aux_data, int linear_idx, void* device_ptr = NULL, bool release_input = true) override
		{
            return input;
		}

		array4d output_shape_;
		int output_size_;
		MemPool* mem_pool_;

		const int num_feature_;
        const int max_idx_;

		T* kernel_data_;
        TensorMap<T, 1> kernel_;
	};
}
#endif