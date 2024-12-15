#ifndef SGXDNN_LAYERNORM_H_
#define SGXDNN_LAYERNORM_H_

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

	template <typename T> class LayerNorm: public Layer<T>
	{
	public:
		explicit LayerNorm(
				const std::string& name,
				const array4d input_shape,
                const int num_feature,
				MemPool* mem_pool,
                T* kernel, T* bias
				): Layer<T>(name, input_shape),
				mem_pool_(mem_pool),
				kernel_data_(nullptr),
			    bias_data_(nullptr),
				kernel_(NULL, num_feature),
				bias_(NULL, num_feature),
				num_feature_(num_feature)
		{
			output_shape_ = input_shape;
			output_size_ = input_shape[2] * input_shape[3];
            auto kernel_data_ = mem_pool -> alloc<T>(num_feature);
            auto bias_data_ = mem_pool -> alloc<T>(num_feature);
            std::copy(kernel, kernel + num_feature, kernel_data_);
            std::copy(bias, bias + num_feature, bias_data_);
            new (&kernel_)  TensorMap<T, 1>(kernel_data_, num_feature);
            new (&bias_) TensorMap<T, 1>(bias_data_, num_feature);
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
            const int seq_len = input.dimension(2);
            const int num_feature = input.dimension(3);
            array4d dims4d = {1, 1, seq_len, 1};
            array4d bcast = {1, 1, 1, num_feature};
            array1d depth_dim = {3};
            input = input-input.mean(depth_dim).eval().reshape(dims4d).broadcast(bcast);
            input = input/(input.sum(depth_dim).eval().sqrt().eval().reshape(dims4d).broadcast(bcast));
            array1d rbcast = {seq_len};
            array1d one_d = {input.size()};
            input.reshape(one_d) = input.reshape(one_d) * kernel_.broadcast(rbcast).reshape(one_d) + bias_.broadcast(rbcast).reshape(one_d);
            return input;
        }

		TensorMap<T, 4> fwd_verify_impl(TensorMap<T, 4> input, float** aux_data, int linear_idx, void* device_ptr = NULL, bool release_input = true) override
		{
            return input;
		}

		array4d output_shape_;
		int output_size_;
		MemPool* mem_pool_;

		const int num_feature_;

		T* kernel_data_;
		T* bias_data_;
        TensorMap<T, 1> kernel_;
		TensorMap<T, 1> bias_;
	};
}
#endif