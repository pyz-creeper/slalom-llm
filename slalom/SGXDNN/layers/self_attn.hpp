#ifndef SGXDNN_SELFATTN_H_
#define SGXDNN_SELFATTN_H_

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

	template <typename T> class SelfAttention: public Layer<T>
	{
	public:
		explicit SelfAttention(
				const std::string& name,
				const array4d input_shape,
				std::shared_ptr<Layer<T>> in_qproj,
				std::shared_ptr<Layer<T>> in_kproj,
				std::shared_ptr<Layer<T>> in_vproj,
				std::shared_ptr<Layer<T>> out_proj,
				int head_dim,
				int num_head,
				MemPool* mem_pool
				): Layer<T>(name, input_shape),
				in_qproj_(in_qproj),
				in_kproj_(in_kproj),
				in_vproj_(in_vproj),
				out_proj_(out_proj),
				head_dim_(head_dim),
				num_head_(num_head),
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
            auto temp_q = in_qproj_ -> apply(input, device_ptr, false);
			auto temp_k = in_kproj_ -> apply(input, device_ptr, false);
			auto temp_v = in_vproj_ -> apply(input, device_ptr, false);

			auto temp_attn = mem_pool_->alloc<T>(num_head_ * seq_len * seq_len);
			for (int h=0; h < num_head_ ; h++) { // attn with causual mask
				auto attn_pt = temp_attn + h *  seq_len * seq_len;
				for (int s=0; s < seq_len*seq_len; s++) {
					int q_id = s / seq_len;
					int k_id = s % seq_len;
					auto q_matrix_map = VectorMap<T>(temp_q.data() + h*head_dim_ + q_id*head_dim_*num_head_, head_dim_);
					auto k_matrix_map = VectorMap<T>(temp_k.data() + h*head_dim_ + k_id*head_dim_*num_head_, head_dim_);
					*(attn_pt + q_id*seq_len + k_id) = (q_id > k_id) ? -1000 : q_matrix_map.dot(k_matrix_map);
				}
				for (int q_id = 0; q_id < seq_len; q_id++) {
					T max_val = *(attn_pt + q_id * seq_len);
					for (int k_id = 1; k_id < seq_len; k_id++) {
						max_val = std::max(max_val, *(attn_pt + q_id * seq_len + k_id));
					}
					T sum_exp = 0.0f;
					for (int k_id = 0; k_id < seq_len; k_id++) {
						*(attn_pt + q_id * seq_len + k_id) -= max_val;  // Avoid overflow
						sum_exp += std::exp(*(attn_pt + q_id * seq_len + k_id));
					}
					for (int k_id = 0; k_id < seq_len; k_id++) {
						*(attn_pt + q_id * seq_len + k_id) = std::exp(*(attn_pt + q_id * seq_len + k_id)) / sum_exp;
					}
				}
			}
			mem_pool_->release(temp_q.data());
			mem_pool_->release(temp_k.data());
			auto out_mat =  mem_pool_->alloc<T>(num_head_ * seq_len * head_dim_);
			for (int h = 0; h < num_head_; h++) {
				auto attn_pt = temp_attn + h * seq_len * seq_len;
				auto v_pt = temp_v.data() + h * seq_len * head_dim_ * num_head_;
				auto out_pt = out_mat + h * seq_len * head_dim_;

				// Perform matrix multiplication: attention * value for each head
				for (int q_id = 0; q_id < seq_len; q_id++) {
					// For each query, compute the weighted sum of values
					for (int d = 0; d < head_dim_; d++) {
						T sum = 0.0f;
						for (int k_id = 0; k_id < seq_len; k_id++) {
							sum += *(attn_pt + q_id * seq_len + k_id) * *(v_pt + k_id * head_dim_ + d);
						}
						*(out_pt + q_id * head_dim_ + d) = sum;
					}
				}
			}
			mem_pool_ -> release(temp_v.data());
			mem_pool_ -> release(temp_attn);
			return out_proj_ -> apply(TensorMap<T, 4>(out_mat, {1, 1, seq_len, num_head_ * head_dim_}), device_ptr, true);
		}

		TensorMap<T, 4> fwd_verify_impl(TensorMap<T, 4> input, float** aux_data, int linear_idx, void* device_ptr = NULL, bool release_input = true) override
		{
			return input;
		}

		array4d output_shape_;
		int output_size_;
		MemPool* mem_pool_;

		const int num_head_;
		const int head_dim_;

		std::shared_ptr<Layer<T>> in_qproj_;
		std::shared_ptr<Layer<T>> in_kproj_;
		std::shared_ptr<Layer<T>> in_vproj_;
		std::shared_ptr<Layer<T>> out_proj_;
	};
}
#endif