#ifndef SGXDNN_MLP_H_
#define SGXDNN_MLP_H_

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

	template <typename T> class LlamaMLP: public Layer<T>
	{
	public:
		explicit LlamaMLP(
				const std::string& name,
				const array4d input_shape,
				std::shared_ptr<Layer<T>> gate_proj,
				std::shared_ptr<Layer<T>> up_proj,
				std::shared_ptr<Layer<T>> down_proj,
				MemPool* mem_pool
				): Layer<T>(name, input_shape),
                gate_proj_(gate_proj),
                up_proj_(up_proj),
                down_proj_(down_proj),
				mem_pool_(mem_pool)
		{
			output_shape_ = input_shape;
			output_size_ = input_shape[2] * input_shape[3];
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
			int seq_len = input.dimension(2);
            auto gate = gate_proj_ -> apply(input, device_ptr, false);
			auto up = up_proj_ -> apply(input, device_ptr, false);
			for (int i = 0; i < gate.size(); ++i) {
				T x = gate.data()[i];  // Get the value at index i
				T sigmoid_x = 1.0f / (1.0f + std::exp(-x));  // Compute sigmoid(x)
				gate.data()[i] = x * sigmoid_x * up.data()[i];  // SiLU(x) = x * sigmoid(x)
			}
			mem_pool_ -> release(up.data());
			if (release_input) mem_pool_ -> release(input.data());
			return down_proj_ -> apply(gate, device_ptr);
		}

		TensorMap<T, 4> fwd_verify_impl(TensorMap<T, 4> input, float** aux_data, int linear_idx, void* device_ptr = NULL, bool release_input = true) override
		{
			return input;
		}

		array4d output_shape_;
		int output_size_;
		MemPool* mem_pool_;

        std::shared_ptr<Layer<T>> gate_proj_;
        std::shared_ptr<Layer<T>> up_proj_;
        std::shared_ptr<Layer<T>> down_proj_;
	};
}
#endif