/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer {

template<typename T>
void GptContextAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                          TensorMap*                input_tensors,
                                          const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [token_num, hidden_dimension]
    //      attention_mask [batch_size, 1, seq_len, seq_len + max_prompt_length]
    //      attention_type [1]
    //      is_final_layer [1], bool on cpu
    //      layer_id [1], int on cpu
    //      padding_offset, int, [token_num] (optional)
    //      cu_seqlens, int, [batch_size] (optional)
    //      d_prefix_prompt_batch [global_batch_size], (optional)
    //          each element contains ptr with buffer shape[2, local_head_num_, prompt_length, size_per_head]
    //      d_prefix_prompt_lengths [batch_size], int (optional)
    //      linear_bias_slopes [head_num] (optional)

    // output_tensors:
    //      hidden_features [token_num, hidden_dimension]
    //      key_cache [batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK(output_tensors->at("key_cache").shape.size() == 5);
    FT_CHECK(output_tensors->at("value_cache").shape.size() == 4
             || output_tensors->at("value_cache").shape.size() == 3);
    const int request_batch_size = input_tensors->at("attention_mask").shape[0];
    const int request_seq_len    = input_tensors->at("attention_mask").shape[2];
    const int max_prompt_length =
        input_tensors->at("attention_mask").shape[3] - input_tensors->at("attention_mask").shape[2];
    // add by wuziyang
    const int  layer_id                = input_tensors->getVal<int>("layer_id");
    const int  position_id             = input_tensors->getVal<int>("position_id");
    const int  block_position_id       = input_tensors->getVal<int>("block_position_id");

    const T**  d_prefix_prompt_batch   = input_tensors->getPtr<const T*>("d_prefix_prompt_batch", nullptr);
    const int* d_prefix_prompt_lengths = input_tensors->getPtr<int>("d_prefix_prompt_lengths", nullptr);
    const int* padding_offset          = input_tensors->getPtr<int>("padding_offset", nullptr);
    int*       cu_seqlens              = input_tensors->getPtr<int>("cu_seqlens", nullptr);
    T*         linear_bias_slopes      = input_tensors->getPtr<T>("linear_bias_slopes", nullptr);
    /* float*     attention_query_dynamic_scale = input_tensors->getPtr<float>("attention_query_dynamic_scale",
     * nullptr); */

    T* attention_out   = output_tensors->at("hidden_features").getPtr<T>();
    T* attention_input = input_tensors->at("input_query").getPtr<T>();
    T* attention_mask  = input_tensors->at("attention_mask").getPtr<T>();

    const AttentionType attention_type = input_tensors->getVal<AttentionType>("attention_type");
    FT_CHECK_WITH_INFO(attention_type != AttentionType::FUSED_PADDED_MHA,
                       "Gpt Context FUSED_PADDED_MHA is not supported !");

    PUSH_RANGE("attention buffer alloc");
    // allocateBuffer(request_batch_size, request_seq_len + max_prompt_length, attention_type != AttentionType::FUSED_MHA);
    // std::cout << "request_seq_len + max_prompt_length + position_id + block_position_id: " << request_seq_len + max_prompt_length + position_id + block_position_id << ", attention_type != AttentionType::FUSED_MHA: " << (attention_type != AttentionType::FUSED_MHA) << ", request_seq_len: " << request_seq_len << ", position_id : " << position_id << ", block_position_id: " << block_position_id << std::endl;
    if (layer_id < 1){
        allocateBuffer(request_batch_size, request_seq_len + max_prompt_length + position_id + block_position_id, attention_type != AttentionType::FUSED_MHA);
    }
    POP_RANGE;
    sync_check_cuda_error();

    const bool is_final = input_tensors->at("is_final_layer").getVal<bool>();

    const int m = input_tensors->at("input_query").shape[0];

    PUSH_RANGE("qkv_gemm");

#ifdef SPARSITY_ENABLED
    const int m_padded   = 8 * div_up(m, 8);
    bool      use_sparse = sparse_ && cublas_wrapper_->isUseSparse(1, 3 * local_hidden_units_, m_padded, hidden_units_);
#else
    constexpr bool use_sparse = false;
#endif

    if (use_sparse) {
#ifdef SPARSITY_ENABLED
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                3 * local_hidden_units_,
                                m_padded,
                                hidden_units_,
                                attention_weights->query_weight.sp_kernel,
                                attention_input,
                                qkv_buf_);
#endif
    }
    else if (int8_mode_ == 1) {
        FT_CHECK(weight_only_int8_fc_runner_.get() != NULL && attention_weights->query_weight.int8_kernel != NULL
                 && attention_weights->query_weight.weight_only_quant_scale != NULL);

        weight_only_int8_fc_runner_->gemm(attention_input,
                                          reinterpret_cast<const uint8_t*>(attention_weights->query_weight.int8_kernel),
                                          attention_weights->query_weight.weight_only_quant_scale,
                                          qkv_buf_,
                                          m,
                                          3 * local_hidden_units_,
                                          hidden_units_,
                                          mixed_gemm_workspace_,
                                          mixed_gemm_ws_bytes_,
                                          stream_);
    }
    else if (int8_mode_ == 2) {
        cublas_wrapper_->Int8Gemm(3 * local_hidden_units_,
                                  m,
                                  hidden_units_,
                                  attention_weights->query_weight.int8_kernel,
                                  hidden_units_,
                                  input_tensors->at("input_query").getPtr<int8_t>(),
                                  hidden_units_,
                                  reinterpret_cast<int8_t*>(qkv_buf_),
                                  3 * local_hidden_units_,
                                  attention_weights->query_weight.scale_inter,
                                  true);
    }
    else {
        // // m = 4
        // cublas_wrapper_->Gemm(CUBLAS_OP_N,
        //                       CUBLAS_OP_N,
        //                       3 * local_hidden_units_,  // m
        //                       m, // n
        //                       hidden_units_,  // k
        //                       attention_weights->query_weight.kernel,
        //                       3 * local_hidden_units_,  // n
        //                       attention_input,
        //                       hidden_units_,  // k
        //                       qkv_buf_,
        //                       3 * local_hidden_units_ /* n */);

        // 特殊处理，专为half计算设计
        cublas_wrapper_->Gemm(CUBLAS_OP_T,
            CUBLAS_OP_N,
            3 * local_hidden_units_,  // m
            m, // n
            hidden_units_,  // k
            &alphah,
            attention_weights->query_weight.kernel,
            CUDA_R_16F,
            hidden_units_,  // m
            attention_input,
            CUDA_R_16F,
            hidden_units_,  // k
            &betah,
            qkv_buf_,
            CUDA_R_16F,
            3 * local_hidden_units_,
            CUDA_R_16F,
            cublasGemmAlgo_t(-1));
    }

    sync_check_cuda_error();

    // if (layer_id == 0){
    //     if (wzy_flag < 1){
    //         int attention_size = 2 * 4096;
    //         __half qkv_cpu[attention_size];
    //         __half inp_cpu[attention_size];
    //         cudaMemcpy(qkv_cpu, qkv_buf_, attention_size * sizeof(__half), cudaMemcpyDeviceToHost);
    //         cudaMemcpy(inp_cpu, attention_weights->query_weight.bias, attention_size * sizeof(__half), cudaMemcpyDeviceToHost);

            // std::cout << "qkv_buf_ res:" << layer_id << std::endl;
            // for (int i = 0; i < 128; i++)
            //     std::cout << ", " << qkv_cpu[i];
            // std::cout << std::endl << std::endl;
            // for (int i = 0; i < 128; i++)
            //     std::cout << ", " << qkv_cpu[i + 128];
            // std::cout << std::endl << std::endl;
            // for (int i = 0; i < 128; i++)
            //     std::cout << ", " << qkv_cpu[i + 4096];
            // std::cout << std::endl << std::endl;
            // std::cout << "qkv bias res:" << layer_id << std::endl;
            // for (int i = 0; i < 128; i++)
            //     std::cout << ", " << inp_cpu[i];
            // std::cout << std::endl << std::endl;
            // for (int i = 0; i < 128; i++)
            //     std::cout << ", " << inp_cpu[i + 128];
            // std::cout << std::endl << std::endl;
            // for (int i = 0; i < 128; i++)
            //     std::cout << ", " << inp_cpu[i + 4096];
            // std::cout << std::endl << std::endl;
    //     }
    // }

    // IDEA: append prefix prompt key value here
    // all is 0, no prompt
    PrefixPromptBatchWeightsParam<T> param{d_prefix_prompt_batch,
                                            // 0
                                           d_prefix_prompt_lengths,
                                            // 0
                                           max_prompt_length,
                                           (size_t)layer_id * 2 * local_head_num_ * size_per_head_,
                                           // 增加layerid参数，用于计算attention
                                           layer_id,
                                           position_id,
                                           block_position_id};

    // true
    if (padding_offset != nullptr) {
        // request_batch_size * request_seq_len * 3 * local_hidden_units_ = 49152
        // q_buf_2_, k_buf_2_ and v_buf_2_ are continuous
        cudaMemsetAsync(
            q_buf_2_, 0, request_batch_size * request_seq_len * 3 * local_hidden_units_ * sizeof(T), stream_);
    }

    // request_seq_len = 4 => 1;
    // rotary_embedding_dim_:64,neox_rotary_style_: 1, size_per_head_: 128, m: 4 => 1
    // std::cout << "request_seq_len:" << request_seq_len << ", m: " << m << std::endl;
    invokeAddFusedQKVBiasTranspose(q_buf_2_,
                                   k_buf_2_,
                                   v_buf_2_,
                                   param,  // prefix prompt
                                   qkv_buf_,
                                   attention_weights->query_weight.bias,
                                   padding_offset,
                                   request_batch_size,
                                   request_seq_len,
                                   m,
                                   local_head_num_,
                                   size_per_head_,
                                   rotary_embedding_dim_,
                                   neox_rotary_style_,
                                   attention_weights->query_weight.scale_out,
                                   int8_mode_,
                                   stream_);
    sync_check_cuda_error();    

    // 36
    const int max_seq_len = (int)(output_tensors->at("key_cache").shape[3]);  // max output seq length
    // Use batch major
    // put k/v_buf from shape [B, H, PL + L, Dh]
    // to cache [B, H, Dh/x, PL + L, x]  and [B, H, PL + L, Dh/x, x], PL denotes prompt length
    const int cache_idx = position_id + block_position_id;
    invokeTranspose4dBatchMajor(output_tensors->getPtr<T>("key_cache"),
                                output_tensors->getPtr<T>("value_cache"),
                                k_buf_2_,
                                v_buf_2_,
                                request_batch_size,
                                max_prompt_length + request_seq_len,  // max input length + prefix prompt length
                                cache_idx,
                                max_seq_len,
                                size_per_head_,
                                local_head_num_,
                                stream_);
    // IDEA : after this, k_cache = (batch_size, num_heads, Dh/x, prefix_prompt_len + L, x)
    // k_cache = (batch_size, num_heads, prefix_prompt_len + L, Dh)
    sync_check_cuda_error();

    // TODO: fmha kernels doesn't support different seq lengths of q and kv
    if (attention_type == AttentionType::FUSED_MHA) {
        dispatcher_fp16->setup_causal_masked_fmha(request_seq_len, request_batch_size);
        dispatcher_fp16->run_causal_masked_fmha(qkv_buf_, cu_seqlens, qkv_buf_3_, true, stream_);
    }
    // NOTE: qkv buffer shape (batch_size, num_heads,L or prompt_len + L, Dh)

    // if (layer_id == 0){
    //     // if (wzy_flag <= 1){
    //         int attention_sizep = 8 * 4096;
    //         __half k_cpu[attention_sizep];
    //         __half k_cpu2[attention_sizep];
    //         __half v_cpu[attention_sizep];
    //         __half v_cpu2[attention_sizep];
    //         cudaMemcpy(k_cpu, k_buf_2_, attention_sizep * sizeof(__half), cudaMemcpyDeviceToHost);
    //         cudaMemcpy(k_cpu2, output_tensors->getPtr<T>("key_cache"), attention_sizep * sizeof(__half), cudaMemcpyDeviceToHost);
    //         // cudaMemcpy(v_cpu, v_buf_2_, attention_sizep * sizeof(__half), cudaMemcpyDeviceToHost);
    // //         cudaMemcpy(v_cpu2, output_tensors->getPtr<T>("value_cache"), attention_sizep * sizeof(__half), cudaMemcpyDeviceToHost);

    //         std::cout << "kv cache res:" << std::endl;
    //         for (int i = 0; i < 128; i++)
    //             std::cout << ", " << k_cpu[i];
    //         std::cout << std::endl << std::endl;
    //         for (int i = 0; i < 128; i++)
    //             std::cout << ", " << k_cpu[i + 128];
    //         std::cout << std::endl << std::endl;
    //         for (int i = 0; i < 128; i++)
    //             std::cout << ", " << k_cpu[i + 128 * 2];
    //         std::cout << std::endl << std::endl;
    //         for (int i = 0; i < 128; i++)
    //             std::cout << ", " << k_cpu[i + 128 * 3];
    //         std::cout << std::endl << std::endl;
    //         for (int i = 0; i < 128; i++)
    //             std::cout << ", " << k_cpu[i + 128 * 4];
    //         std::cout << std::endl << std::endl;

    //         // std::cout << "v cache res:" << std::endl;
    //         // for (int i = 0; i < 128; i++)
    //         //     std::cout << ", " << v_cpu[i];
    //         // std::cout << std::endl << std::endl;
    //         // for (int i = 0; i < 128; i++)
    //         //     std::cout << ", " << v_cpu[i + 128];
    //         // std::cout << std::endl << std::endl;
    //         // for (int i = 0; i < 128; i++)
    //         //     std::cout << ", " << v_cpu[i + 128 * 2];
    //         // std::cout << std::endl << std::endl;

    //         std::cout << "kv cache22222222222222222222222222222222222222222 res:" << std::endl;
    //         for (int i = 0; i < 128; i++)
    //             std::cout << ", " << k_cpu2[i];
    //         std::cout << std::endl << std::endl;
    // //         // for (int i = 0; i < 128; i++)
    // //         //     std::cout << ", " << k_cpu2[i + 128];
    // //         // std::cout << std::endl << std::endl;
    // //         // for (int i = 0; i < 128; i++)
    // //         //     std::cout << ", " << k_cpu2[i + 128 * 2];
    // //         // std::cout << std::endl << std::endl;
    // //         // for (int i = 0; i < 128; i++)
    // //         //     std::cout << ", " << k_cpu2[i + 128 * 3];
    // //         // std::cout << std::endl << std::endl;
    // //         // for (int i = 0; i < 128; i++)
    // //         //     std::cout << ", " << k_cpu2[i + 128 * 4];
    // //         // std::cout << std::endl << std::endl;
    //         for (int i = 0; i < 128; i++)
    //             std::cout << ", " << k_cpu2[i + 128 * 32];
    //         std::cout << std::endl << std::endl;
    //         for (int i = 0; i < 128; i++)
    //             std::cout << ", " << k_cpu2[i + 128 * 32 * 2];
    //         std::cout << std::endl << std::endl;
    //         for (int i = 0; i < 128; i++)
    //             std::cout << ", " << k_cpu2[i + 128 * 32 * 3];
    //         std::cout << std::endl << std::endl;
    //         for (int i = 0; i < 128; i++)
    //             std::cout << ", " << k_cpu2[i + 128 * 32 * 4];
    //         std::cout << std::endl << std::endl;

    // //         std::cout << "kv cache22222222222222222222222222222222222222222 res:" << std::endl;
    // //         for (int i = 0; i < 128; i++)
    // //             std::cout << ", " << v_cpu2[i];
    // //         std::cout << std::endl << std::endl;
    // //         for (int i = 0; i < 128; i++)
    // //             std::cout << ", " << v_cpu2[i + 128];
    // //         std::cout << std::endl << std::endl;
    // //         // for (int i = 0; i < 128; i++)
    // //         //     std::cout << ", " << v_cpu2[i + 128 * 2];
    // //         // std::cout << std::endl << std::endl;
    // //         // for (int i = 0; i < 128; i++)
    // //         //     std::cout << ", " << v_cpu2[i + 128 * 3];
    // //         // std::cout << std::endl << std::endl;
    // //         // for (int i = 0; i < 128; i++)
    // //         //     std::cout << ", " << v_cpu2[i + 128 * 4];
    // //         // std::cout << std::endl << std::endl;
    // //         for (int i = 0; i < 128; i++)
    // //             std::cout << ", " << v_cpu2[i + 128 * 32];
    // //         std::cout << std::endl << std::endl;
    // //         for (int i = 0; i < 128; i++)
    // //             std::cout << ", " << v_cpu2[i + 128 * 32 * 2];
    // //         std::cout << std::endl << std::endl;
    // //         for (int i = 0; i < 128; i++)
    // //             std::cout << ", " << v_cpu2[i + 128 * 32 * 3];
    // //         std::cout << std::endl << std::endl;
    // //         for (int i = 0; i < 128; i++)
    // //             std::cout << ", " << v_cpu2[i + 128 * 32 * 4];
    // //         std::cout << std::endl << std::endl;

    // //     }
    // //     if (wzy_flag < 2){
    // //         wzy_flag += 1;
    // //     }
    // }


    /*
        这里下面对应torch版本中的attention_fn函数，所以上面必须要完成rotary_embedding
    */

    // __half a = static_cast<__half>(1.0f);
    // __half b = static_cast<__half>(2.0f);
    // __half alpfhah[2] = {a, b};
    // __half betfah  = static_cast<__half>(0.0f);
    // float  alphaf = 1.0;
    // float  aph2f[2]  = {0.0, 0.0};
    // aph2f[0] = static_cast<float>(alpfhah[0]);
    // aph2f[1] = static_cast<float>(alpfhah[1]);

    // std::cout << "123412424" << std::endl << std::endl;
    // std::cout << aph2f[0] << ", " << aph2f[1] << std::endl << std::endl;



    POP_RANGE;
    // true
    if (is_final == false) {
        const cudaDataType_t gemm_data_type      = getCudaDataType<T>();
        const int            attention_seq_len_1 = request_seq_len;                      // q length
        // const int            attention_seq_len_2 = max_prompt_length + request_seq_len;  // kv length
        const int            attention_seq_len_2 = max_prompt_length + request_seq_len + position_id + block_position_id;  // kv length
        const T              qk_scale            = static_cast<T>(1.0f / sqrtf(size_per_head_ * 1.0f));
        
        // attention_seq_len_1: 4, attention_seq_len_2: 4, size_per_head_: 128, request_batch_size: 1, local_head_num_: 32

        // 手动调整，使用half推理
        is_qk_buf_float_ = false;
        if (attention_type != AttentionType::FUSED_MHA) {
            // false
            if (is_qk_buf_float_ == true && gemm_data_type != CUDA_R_32F) {
                PUSH_RANGE("Q*K batch gemm");
                cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    attention_seq_len_2,  // n
                                                    attention_seq_len_1,  // m
                                                    size_per_head_,       // k
                                                    // 1.0f,
                                                    alphah,
                                                    k_buf_2_,
                                                    gemm_data_type,
                                                    size_per_head_,                        // k
                                                    attention_seq_len_2 * size_per_head_,  // n * k
                                                    q_buf_2_,
                                                    gemm_data_type,
                                                    size_per_head_,                        // k
                                                    attention_seq_len_1 * size_per_head_,  // m * k
                                                    // 0.0f,
                                                    betah,
                                                    qk_buf_float_,
                                                    // CUDA_R_32F,
                                                    CUDA_R_16F,
                                                    attention_seq_len_2,  // n
                                                    attention_seq_len_2 * attention_seq_len_1,
                                                    request_batch_size * local_head_num_,  // global batch size
                                                    // CUDA_R_32F);
                                                    CUDA_R_16F);

                sync_check_cuda_error();
                POP_RANGE;

                PUSH_RANGE("softmax");
                MaskedSoftmaxParam<T, float> param;
                param.attention_score    = qk_buf_;         // (batch_size, head_num, q_length, k_length)
                param.qk                 = qk_buf_float_;   // (batch_size, head_num, q_length, k_length)
                param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
                param.batch_size         = request_batch_size;
                param.q_length           = attention_seq_len_1;
                param.k_length           = attention_seq_len_2;
                param.num_heads          = local_head_num_;
                param.qk_scale           = qk_scale;
                param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);  // (head_num,), optional
                invokeMaskedSoftmax(param, stream_);
                sync_check_cuda_error();
                POP_RANGE;
            }
            // true
            else {
                // std::cout << "gemm in here , and :" << attention_seq_len_2  << ", and :" << attention_seq_len_1 << std::endl;
                PUSH_RANGE("Q*K batch gemm");
                // cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                //                                     CUBLAS_OP_N,
                //                                     attention_seq_len_2,
                //                                     attention_seq_len_1,
                //                                     size_per_head_,
                //                                     k_buf_2_,
                //                                     size_per_head_,
                //                                     attention_seq_len_2 * size_per_head_,
                //                                     q_buf_2_,
                //                                     size_per_head_,
                //                                     attention_seq_len_1 * size_per_head_,
                //                                     qk_buf_,
                //                                     attention_seq_len_2,
                //                                     attention_seq_len_2 * attention_seq_len_1,
                //                                     request_batch_size * local_head_num_);

                cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    attention_seq_len_2,  // n
                                                    attention_seq_len_1,  // m
                                                    size_per_head_,       // k
                                                    // 1.0f,
                                                    alphah,
                                                    k_buf_2_,
                                                    gemm_data_type,
                                                    size_per_head_,  // n * k
                                                    attention_seq_len_2 * size_per_head_,       // k
                                                    q_buf_2_,
                                                    gemm_data_type,
                                                    size_per_head_,                        // k
                                                    attention_seq_len_1 * size_per_head_,  // m * k
                                                    // 0.0f,
                                                    betah,
                                                    qk_buf_,
                                                    // CUDA_R_32F,
                                                    CUDA_R_16F,
                                                    attention_seq_len_2,  // n
                                                    attention_seq_len_2 * attention_seq_len_1,
                                                    request_batch_size * local_head_num_,  // global batch size
                                                    // CUDA_R_32F);
                                                    CUDA_R_16F);

                if (request_seq_len == 1){
                    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    attention_seq_len_2,  // n
                                    attention_seq_len_1,  // m
                                    size_per_head_,       // k
                                    // 1.0f,
                                    alphah,
                                    k_buf_2_,
                                    gemm_data_type,
                                    size_per_head_,  // n * k
                                    attention_seq_len_2 * size_per_head_,       // k
                                    q_buf_2_,
                                    gemm_data_type,
                                    size_per_head_,                        // k
                                    attention_seq_len_1 * size_per_head_,  // m * k
                                    // 0.0f,
                                    betah,
                                    qkv_buf_3_,
                                    // CUDA_R_32F,
                                    CUDA_R_16F,
                                    attention_seq_len_2,  // n
                                    attention_seq_len_2 * attention_seq_len_1,
                                    request_batch_size * local_head_num_,  // global batch size
                                    // CUDA_R_32F);
                                    CUDA_R_16F);


                    // int qk_s2iz2e = 4096;
                    // __half qek22_cpu[qk_s2iz2e];
                    // cudaMemcpy(qek22_cpu, qkv_buf_3_, qk_s2iz2e * sizeof(__half), cudaMemcpyDeviceToHost);

                    // std::cout << "qkv_buf_3_ res:" << std::endl;
                    // for (int i = 0; i < 16; i++)
                    //     std::cout << ", " << qek22_cpu[i];
                    // std::cout << std::endl << std::endl;
                    // for (int i = 0; i < 16; i++)
                    //     std::cout << ", " << qek22_cpu[i + 16];
                    // std::cout << std::endl << std::endl;
                    // for (int i = 0; i < 16; i++)
                    //     std::cout << ", " << qek22_cpu[i + 16 * 2];
                    // std::cout << std::endl << std::endl;
                    // for (int i = 0; i < 16; i++)
                    //     std::cout << ", " << qek22_cpu[i + 16 * 3];
                    // std::cout << std::endl << std::endl;
                }

                // if (layer_id == 0){
                //     int qk_s2ize = 4096;
                //     __half qk22_cpu[qk_s2ize];
                //     cudaMemcpy(qk22_cpu, qk_buf_, qk_s2ize * sizeof(__half), cudaMemcpyDeviceToHost);

                    // std::cout << "qk buf  res:" << std::endl;
                    // for (int i = 0; i < 128; i++)
                    //     std::cout << ", " << qk22_cpu[i];
                    // std::cout << std::endl << std::endl;

                //     __half qk3_cpu[qk_s2ize];
                //     cudaMemcpy(qk3_cpu, qkv_buf_3_, qk_s2ize * sizeof(__half), cudaMemcpyDeviceToHost);

                //     std::cout << "qk buf3  res:" << std::endl;
                //     for (int i = 0; i < 128; i++)
                //         std::cout << ", " << qk3_cpu[i];
                //     std::cout << std::endl << std::endl;
                // }

                POP_RANGE;
                PUSH_RANGE("softmax");
                MaskedSoftmaxParam<T, T> param;
                param.attention_score    = request_seq_len == 1 ? qkv_buf_3_ : qk_buf_;         // (batch_size, head_num, q_length, k_length)
                param.qk                 = request_seq_len == 1 ? qkv_buf_3_ : qk_buf_;         // (batch_size, head_num, q_length, k_length)
                param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
                param.batch_size         = request_batch_size;
                param.q_length           = attention_seq_len_1;
                param.k_length           = attention_seq_len_2;
                param.num_heads          = local_head_num_;
                param.qk_scale           = qk_scale;
                param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);  // (head_num,), optional
                invokeMaskedSoftmax(param, stream_);
                sync_check_cuda_error();
                POP_RANGE;
            }

            PUSH_RANGE("QK*V batch gemm");

            // if (layer_id == 0){
            //     int att_s2ize = 4096;
            //     __half att_cpu[att_s2ize];
            //     cudaMemcpy(att_cpu, attention_mask, att_s2ize * sizeof(__half), cudaMemcpyDeviceToHost);

            //     std::cout << "att res:" << std::endl;
            //     for (int i = 0; i < 128; i++)
            //         std::cout << ", " << att_cpu[i];
            //     std::cout << std::endl << std::endl;

            // }



            // cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
            //                                     CUBLAS_OP_N,
            //                                     size_per_head_,
            //                                     attention_seq_len_1,
            //                                     attention_seq_len_2,
            //                                     v_buf_2_,
            //                                     size_per_head_,
            //                                     attention_seq_len_2 * size_per_head_,
            //                                     qk_buf_,
            //                                     attention_seq_len_2,
            //                                     attention_seq_len_1 * attention_seq_len_2,
            //                                     qkv_buf_2_,
            //                                     size_per_head_,
            //                                     attention_seq_len_1 * size_per_head_,
            //                                     request_batch_size * local_head_num_);

            // local_head_num_: 32, size_per_head_: 128, attention_seq_len_1: 4, attention_seq_len_2: 4
            // 结果是32 * 4 * 128， 4 * 128 列优先排列
            if (request_seq_len > 1){
                cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                    CUBLAS_OP_T,
                                                    attention_seq_len_1,  // m
                                                    size_per_head_,  // n
                                                    attention_seq_len_2,       // k
                                                    // 1.0f,
                                                    alphah,
                                                    request_seq_len == 1 ? qkv_buf_3_ : qk_buf_,
                                                    CUDA_R_16F,
                                                    attention_seq_len_2,                        // k
                                                    attention_seq_len_1 * attention_seq_len_2,  // m * k
                                                    v_buf_2_,
                                                    CUDA_R_16F,
                                                    size_per_head_,                        // k
                                                    attention_seq_len_2 * size_per_head_,  // n * k
                                                    // 0.0f,
                                                    betah,
                                                    qkv_buf_2_,
                                                    // CUDA_R_32F,
                                                    CUDA_R_16F,
                                                    attention_seq_len_2,  // n
                                                    attention_seq_len_1 * size_per_head_,
                                                    request_batch_size * local_head_num_,  // global batch size
                                                    // CUDA_R_32F);
                                                    CUDA_R_16F);
            } else {
                cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                    CUBLAS_OP_T,
                                                    attention_seq_len_1,  // m
                                                    size_per_head_,  // n
                                                    attention_seq_len_2,       // k
                                                    // 1.0f,
                                                    alphah,
                                                    request_seq_len == 1 ? qkv_buf_3_ : qk_buf_,
                                                    CUDA_R_16F,
                                                    attention_seq_len_2,                        // k
                                                    attention_seq_len_1 * attention_seq_len_2,  // m * k
                                                    v_buf_2_,
                                                    CUDA_R_16F,
                                                    size_per_head_,                        // k
                                                    attention_seq_len_2 * size_per_head_,  // n * k
                                                    // 0.0f,
                                                    betah,
                                                    qkv_buf_2_,
                                                    // CUDA_R_32F,
                                                    CUDA_R_16F,
                                                    attention_seq_len_1,  // n
                                                    attention_seq_len_1 * size_per_head_,
                                                    request_batch_size * local_head_num_,  // global batch size
                                                    // CUDA_R_32F);
                                                    CUDA_R_16F);
            }

            // if (layer_id == 0){
            //     int qkv_size = 2 * 4096;
            //     __half qkv_cpu[qkv_size];
            //     cudaMemcpy(qkv_cpu, qkv_buf_2_, qkv_size * sizeof(__half), cudaMemcpyDeviceToHost);

                // std::cout << "qkv buf2 res:" << std::endl;
                // for (int i = 0; i < 128; i++)
                //     std::cout << ", " << qkv_cpu[i];
                // std::cout << std::endl << std::endl;
                // for (int i = 0; i < 128; i++)
                //     std::cout << ", " << qkv_cpu[i + 128];
                // std::cout << std::endl << std::endl;
                // for (int i = 0; i < 128; i++)
                //     std::cout << ", " << qkv_cpu[i + 128 * 2];
                // std::cout << std::endl << std::endl;
                // for (int i = 0; i < 128; i++)
                //     std::cout << ", " << qkv_cpu[i + 128 * 31];
                // std::cout << std::endl << std::endl;


                // int qk_s2ize = 4096;
                // __half qk22_cpu[qk_s2ize];
                // cudaMemcpy(qk22_cpu, (request_seq_len == 1 ? qkv_buf_3_ : qk_buf_), qk_s2ize * sizeof(__half), cudaMemcpyDeviceToHost);

                // std::cout << "qk buf  res:" << std::endl;
                // for (int i = 0; i < 128; i++)
                //     std::cout << ", " << qk22_cpu[i];
                // std::cout << std::endl << std::endl;

                // __half v_cpu[qk_s2ize];
                // cudaMemcpy(v_cpu, v_buf_2_, qk_s2ize * sizeof(__half), cudaMemcpyDeviceToHost);

                // std::cout << "v buf  res:" << std::endl;
                // for (int i = 0; i < 128; i++)
                //     std::cout << ", " << v_cpu[i];
                // std::cout << std::endl << std::endl;
            // }


            // transpose (batch_size, num_heads, L, Dh) to (batch_size, L, num_heads * Dh)
            // false
            if (padding_offset == nullptr) {
                invokeTransposeQKV(qkv_buf_3_,
                                   qkv_buf_2_,
                                   request_batch_size,
                                   attention_seq_len_1,
                                   local_head_num_,
                                   size_per_head_,
                                   attention_weights->attention_output_weight.scale,
                                   int8_mode_,
                                   stream_);
                sync_check_cuda_error();
            }
            // in here
            else {
                invokeTransposeAttentionOutRemovePadding(qkv_buf_2_,
                                                         qkv_buf_3_,
                                                         m,
                                                         request_batch_size,
                                                         attention_seq_len_1,
                                                         local_head_num_,
                                                         size_per_head_,
                                                         padding_offset,
                                                         attention_weights->attention_output_weight.scale,
                                                         int8_mode_,
                                                         stream_);
            }
            POP_RANGE;
        }
        sync_check_cuda_error();


        // if (layer_id == 0){
        //     int qk_s2ize = 4096;
        //     __half qk222_cpu[qk_s2ize];
        //     cudaMemcpy(qk222_cpu, qkv_buf_3_, qk_s2ize * sizeof(__half), cudaMemcpyDeviceToHost);

            // std::cout << "qkv buf 3 res:" << std::endl;
            // for (int i = 0; i < 128; i++)
            //     std::cout << ", " << qk222_cpu[i];
            // std::cout << std::endl << std::endl;


            // for (int i = 0; i < 128; i++)
            //     std::cout << ", " << qk222_cpu[i + 128];
            // std::cout << std::endl << std::endl;
            // for (int i = 0; i < 128; i++)
            //     std::cout << ", " << qk222_cpu[i + 128 * 2];
            // std::cout << std::endl << std::endl;
            // for (int i = 0; i < 128; i++)
            //     std::cout << ", " << qk222_cpu[i + 128 * 31];
            // std::cout << std::endl << std::endl;
        // }



        PUSH_RANGE("proj gemm");
#ifdef SPARSITY_ENABLED
        bool use_sparse = sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m_padded, local_hidden_units_);
#endif

        if (use_sparse) {
#ifdef SPARSITY_ENABLED
            cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    hidden_units_,
                                    m_padded,
                                    local_hidden_units_,
                                    attention_weights->attention_output_weight.sp_kernel,
                                    qkv_buf_3_,
                                    attention_out);
#endif
        }
        else {
            if (int8_mode_ == 1) {
                FT_CHECK(weight_only_int8_fc_runner_.get() != NULL
                         && attention_weights->attention_output_weight.int8_kernel != NULL
                         && attention_weights->attention_output_weight.weight_only_quant_scale != NULL);

                weight_only_int8_fc_runner_->gemm(
                    qkv_buf_3_,
                    reinterpret_cast<const uint8_t*>(attention_weights->attention_output_weight.int8_kernel),
                    attention_weights->attention_output_weight.weight_only_quant_scale,
                    attention_out,
                    m,
                    hidden_units_,
                    local_hidden_units_,
                    mixed_gemm_workspace_,
                    mixed_gemm_ws_bytes_,
                    stream_);
            }
            else if (int8_mode_ == 2) {
                int8_fc_runner_->gemm(reinterpret_cast<int8_t*>(qkv_buf_3_),
                                      attention_weights->attention_output_weight.int8_kernel,
                                      QuantMode::PerTensorQuant,
                                      attention_weights->attention_output_weight.scale_inter,
                                      attention_weights->attention_output_weight.scale_out,
                                      output_tensors->at("hidden_features").getPtr<T>(),
                                      m,
                                      hidden_units_,
                                      local_hidden_units_,
                                      nullptr,
                                      0,
                                      stream_);
            }
            // in here
            else {
                // cublas_wrapper_->Gemm(CUBLAS_OP_T,
                //                       CUBLAS_OP_N,
                //                       m,
                //                       hidden_units_,
                //                       local_hidden_units_,
                //                       &alphah,
                //                       qkv_buf_3_,
                //                       CUDA_R_16F,
                //                       hidden_units_,
                //                       attention_weights->attention_output_weight.kernel,
                //                       CUDA_R_16F,
                //                       hidden_units_,
                //                       &betah,
                //                       attention_out,
                //                       CUDA_R_16F,
                //                       m,
                //                       CUDA_R_16F,
                //                       cublasGemmAlgo_t(-1));

                // if (layer_id == 0){
                //     int qk_s2iz2e = 4096;
                //     __half qek22_cpu[qk_s2iz2e];
                //     cudaMemcpy(qek22_cpu, qkv_buf_3_, qk_s2iz2e * sizeof(__half), cudaMemcpyDeviceToHost);

                //     std::cout << "qkv_buf_3_ res:" << std::endl;
                //     for (int i = 0; i < 128; i++)
                //         std::cout << ", " << qek22_cpu[i];
                //     std::cout << std::endl << std::endl;
                // }

                cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      hidden_units_,
                                      m,
                                      local_hidden_units_,
                                      &alphah,
                                      attention_weights->attention_output_weight.kernel,
                                      CUDA_R_16F,
                                      hidden_units_,
                                      qkv_buf_3_,
                                      CUDA_R_16F,
                                      hidden_units_,
                                      &betah,
                                      attention_out,
                                      CUDA_R_16F,
                                      hidden_units_,
                                      CUDA_R_16F,
                                      cublasGemmAlgo_t(-1));

                // if (layer_id == 0){
                //     int attention_size = 4096 * 8;
                //     __half attention_cpu[attention_size];

                //     cudaMemcpy(attention_cpu, attention_out, attention_size * sizeof(__half), cudaMemcpyDeviceToHost);

                    // std::cout << "attention_cpu output:" << std::endl;
                    // for (int i = 0; i < 128; i++)
                    //     std::cout << ", " << attention_cpu[i];
                    // std::cout << std::endl << std::endl;
                    
                    // for (int i = 0; i < 128; i++)
                    //     std::cout << ", " << attention_cpu[i + 128];
                    // std::cout << std::endl;
                    // for (int i = 0; i < 128; i++)
                    //     std::cout << ", " << attention_cpu[i + 128 * 4];
                    // std::cout << std::endl;
                    // for (int i = 0; i < 128; i++)
                    //     std::cout << ", " << attention_cpu[i + 128 * 32];
                    // std::cout << std::endl;
                // }
            }
        }
        POP_RANGE;
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t           max_batch_size,
                                                      size_t           max_seq_len,
                                                      size_t           head_num,
                                                      size_t           size_per_head,
                                                      cudaStream_t     stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator*      allocator,
                                                      bool             is_free_buffer_after_forward,
                                                      bool             is_qk_buf_float,
                                                      bool             sparse,
                                                      int              int8_mode):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(0),
    neox_rotary_style_(false),
    is_qk_buf_float_(is_qk_buf_float || int8_mode == 2),
    weight_only_int8_fc_runner_(int8_mode == 1 ? std::make_shared<CutlassFpAIntBGemmRunner<T, uint8_t>>() : nullptr),
    int8_fc_runner_(int8_mode == 2 ? std::make_shared<CutlassInt8GemmRunner<T>>() : nullptr),
    int8_mode_(int8_mode)
{
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t           max_batch_size,
                                                      size_t           max_seq_len,
                                                      size_t           head_num,
                                                      size_t           size_per_head,
                                                      size_t           local_head_num,
                                                      cudaStream_t     stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator*      allocator,
                                                      bool             is_free_buffer_after_forward,
                                                      bool             is_qk_buf_float,
                                                      bool             sparse,
                                                      int              int8_mode):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(0),
    neox_rotary_style_(false),
    is_qk_buf_float_(is_qk_buf_float || int8_mode == 2),
    weight_only_int8_fc_runner_(int8_mode == 1 ? std::make_shared<CutlassFpAIntBGemmRunner<T, uint8_t>>() : nullptr),
    int8_fc_runner_(int8_mode == 2 ? std::make_shared<CutlassInt8GemmRunner<T>>() : nullptr),
    int8_mode_(int8_mode)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    dispatcher_fp16.reset(new FusedMHARunnerFP16v2(local_head_num_, size_per_head_, sm_, 1.0f));
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t           max_batch_size,
                                                      size_t           max_seq_len,
                                                      size_t           head_num,
                                                      size_t           size_per_head,
                                                      size_t           local_head_num,
                                                      size_t           rotary_embedding_dim,
                                                      bool             neox_rotary_style,
                                                      cudaStream_t     stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator*      allocator,
                                                      bool             is_free_buffer_after_forward,
                                                      bool             is_qk_buf_float,
                                                      bool             sparse,
                                                      int              int8_mode):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(rotary_embedding_dim),
    neox_rotary_style_(neox_rotary_style),
    is_qk_buf_float_(is_qk_buf_float),
    weight_only_int8_fc_runner_(int8_mode == 1 ? std::make_shared<CutlassFpAIntBGemmRunner<T, uint8_t>>() : nullptr),
    int8_fc_runner_(int8_mode == 2 ? std::make_shared<CutlassInt8GemmRunner<T>>() : nullptr),
    int8_mode_(int8_mode)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    dispatcher_fp16.reset(new FusedMHARunnerFP16v2(local_head_num_, size_per_head_, sm_, 1.0f));
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(GptContextAttentionLayer<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_,
                          attention_layer.sparse_),
    max_batch_size_(attention_layer.max_batch_size_),
    max_seq_len_(attention_layer.max_seq_len_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    local_head_num_(attention_layer.local_head_num_),
    local_hidden_units_(attention_layer.local_hidden_units_),
    rotary_embedding_dim_(attention_layer.rotary_embedding_dim_),
    neox_rotary_style_(attention_layer.neox_rotary_style_),
    is_qk_buf_float_(attention_layer.is_qk_buf_float_),
    weight_only_int8_fc_runner_(attention_layer.weight_only_int8_fc_runner_),
    int8_fc_runner_(attention_layer.int8_fc_runner_),
    int8_mode_(attention_layer.int8_mode_)
{
}

template<typename T>
GptContextAttentionLayer<T>::~GptContextAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void GptContextAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void GptContextAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t seq_len, bool allocate_qk_buf)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // const auto type_size = int8_mode_ == 2 ? sizeof(int8_t) : sizeof(T);
    // NOTE (perkzz): use sizeof(T) here for cutlass int8 kernels.
    const auto type_size = sizeof(T);
    qkv_buf_ = (T*)allocator_->reMalloc(qkv_buf_, type_size * 3 * batch_size * seq_len * local_hidden_units_, true);
    q_buf_2_ = (T*)allocator_->reMalloc(q_buf_2_, sizeof(T) * batch_size * seq_len * 3 * local_hidden_units_, true);
    k_buf_2_ = q_buf_2_ + batch_size * seq_len * local_hidden_units_;
    v_buf_2_ = k_buf_2_ + batch_size * seq_len * local_hidden_units_;

    // save memory usage when using fmha
    // if (allocate_qk_buf) {
    qk_buf_ = (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * local_head_num_ * seq_len * seq_len, true);
    // }
    // else {
    //     allocator_->free((void**)(&qk_buf_));
    // }
    qkv_buf_2_ = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * seq_len * local_hidden_units_, true);
    qkv_buf_3_ = (T*)allocator_->reMalloc(qkv_buf_3_, type_size * batch_size * seq_len * local_hidden_units_, true);

    // if (is_qk_buf_float_ == true) {
    //     if (allocate_qk_buf) {
    //         qk_buf_float_ = (float*)allocator_->reMalloc(
    //             qk_buf_float_, sizeof(float) * batch_size * local_head_num_ * seq_len * seq_len, true);
    //     }
    //     else {
    //         allocator_->free((void**)(&qk_buf_float_));
    //     }
    // }

    if (int8_mode_ == 1) {
        // We use max_size for n and k since we reuse buffers for both FCs and want to allocate the max
        // possible memory that would be required by any of the individual gemms.
        const int max_size    = std::max(hidden_units_, 3 * local_hidden_units_);
        mixed_gemm_ws_bytes_  = weight_only_int8_fc_runner_->getWorkspaceSize(batch_size * seq_len, max_size, max_size);
        mixed_gemm_workspace_ = (char*)allocator_->reMalloc(mixed_gemm_workspace_, mixed_gemm_ws_bytes_, false);
    }

    if (int8_mode_ == 1) {
        // We use max_size for n and k since we reuse buffers for both FCs and want to allocate the max
        // possible memory that would be required by any of the individual gemms.
        const int max_size    = std::max(hidden_units_, 3 * local_hidden_units_);
        mixed_gemm_ws_bytes_  = weight_only_int8_fc_runner_->getWorkspaceSize(batch_size * seq_len, max_size, max_size);
        mixed_gemm_workspace_ = (char*)allocator_->reMalloc(mixed_gemm_workspace_, mixed_gemm_ws_bytes_, false);
    }
    else if (int8_mode_ == 2) {
        const int max_size   = std::max(hidden_units_, 3 * local_hidden_units_);
        int8_gemm_ws_bytes_  = int8_fc_runner_->getWorkspaceSize(batch_size * seq_len, max_size, max_size);
        int8_gemm_workspace_ = (char*)allocator_->reMalloc(int8_gemm_workspace_, int8_gemm_ws_bytes_, false);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void GptContextAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&q_buf_2_));
        allocator_->free((void**)(&qk_buf_));
        allocator_->free((void**)(&qkv_buf_2_));
        allocator_->free((void**)(&qkv_buf_3_));

        if (is_qk_buf_float_ == true) {
            allocator_->free((void**)(&qk_buf_float_));
        }

        allocator_->free((void**)(&mixed_gemm_workspace_));
        mixed_gemm_ws_bytes_ = 0;

        allocator_->free((void**)(&int8_gemm_workspace_));
        int8_gemm_ws_bytes_ = 0;

        is_allocate_buffer_ = false;
    }
}

template class GptContextAttentionLayer<float>;
template class GptContextAttentionLayer<half>;
#ifdef ENABLE_BF16
template class GptContextAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
