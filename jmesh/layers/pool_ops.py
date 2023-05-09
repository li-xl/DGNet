import jittor as jt 
import numpy as np 

class PoolFuncV2(jt.Function):
    def execute(self,feats,mask,adj):
        head = r'''
        #include<iostream>
        #define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
        '''
        src = r'''
        __global__ void pool(const int nthreads, 
                            const int N,
                            const int C,
                            const int F,
                            const float* feats,
                            const int* adj,
                            const int* mask,
                            float * out_feats,
                            int* indices){
            CUDA_KERNEL_LOOP(index, nthreads){
                int f = index % F;
                int c = (index / F) % C;
                int n =  index / F / C;
                out_feats[index] = feats[index];
                indices[index] = f;

                if(mask[n*F+f]==1){
                    for(int i=0;i<3;i++){
                        int j = n*F*3+f*3+i;
                        int a = adj[j];
                        int k = n*F+a;
                        if(a>=0 && mask[k]==0){
                            int k_index = n*C*F+c*F+a;
                            if(out_feats[index]<feats[k_index]){
                                out_feats[index] = feats[k_index];
                                indices[index] = a;
                            }
                        }
                    }
                        
                }
            }
        }
        @alias(feats_t,in0);
        @alias(adj_t,in1);
        @alias(mask_t,in2);
        @alias(output_t,out0);
        @alias(indices_t,out1)
        const int N = feats_t_shape0;
        const int C = feats_t_shape1;
        const int F = feats_t_shape2;
        const int output_size = N*C*F;
        const int thread_per_block = 1024;
        const int block_count = (output_size + thread_per_block - 1) / thread_per_block;
        pool<<<block_count,thread_per_block>>>(output_size,N,C,F,feats_t_p,adj_t_p,mask_t_p,output_t_p,indices_t_p);
        '''
        N,C,F = feats.shape
        feats,indices =  jt.code([(N,C,F),(N,C,F)],[feats.dtype,adj.dtype],inputs=[feats,adj,mask],cuda_header=head,cuda_src=src)
        
        self.indices = indices

        return feats

    def grad(self,grad_output):
        head = r'''
        #include<iostream>
        #define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
        '''
        src = r'''
        __global__ void pool_backward(const int nthreads, 
                            const int N,
                            const int C,
                            const int F,
                            const float* grad_output,
                            const int* indices,
                            float * out_grad){
            CUDA_KERNEL_LOOP(index, nthreads){
                int c = (index / F) % C;
                int n =  index / F / C;
                int src_f = indices[index];
                atomicAdd(out_grad+n*F*C+c*F+src_f,grad_output[index]);
            }
        }
        @alias(feats_t,in0);
        @alias(indices_t,in1);
        @alias(output_t,out0);
        const int N = feats_t_shape0;
        const int C = feats_t_shape1;
        const int F = feats_t_shape2;
        const int output_size = N*C*F;
        const int thread_per_block = 1024;
        const int block_count = (output_size + thread_per_block - 1) / thread_per_block;
        cudaMemsetAsync(output_t_p,0,output_t->size);
        pool_backward<<<block_count,thread_per_block>>>(output_size,N,C,F,feats_t_p,indices_t_p,output_t_p);
        '''
        
        grad_feats =  jt.code(grad_output.shape,grad_output.dtype,inputs=[grad_output,self.indices],cuda_header=head,cuda_src=src)

        return grad_feats

pool_funcv2 = PoolFuncV2.apply

class UnPoolFuncV2(jt.Function):
    def execute(self,feats,mask,adj,is_bilinear=1):
        head = r'''
        #include<iostream>
        #define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
        '''+f'''
        #define IS_BILI {is_bilinear}
        '''
        src = r'''
        __global__ void pool(const int nthreads, 
                            const int N,
                            const int C,
                            const int F,
                            const float* feats,
                            const int* adj,
                            const int* mask,
                            float * out_feats,
                            int* indices){
            CUDA_KERNEL_LOOP(index, nthreads){
                int f = index % F;
                int c = (index / F) % C;
                int n =  index / F / C;
                int i_index = n*F*3+f*3;
                out_feats[index] = feats[index];
                if(c==0){
                    indices[i_index] = f;
                    indices[i_index+1] = -1;
                    indices[i_index+2] = -1;
                }
                if(mask[n*F+f]==0){
                    float val = 0;
                    int count = 0;
                    for(int i=0;i<3;i++){
                        int j = n*F*3+f*3+i;
                        int a = adj[j];
                        int k = n*F+a;
                        if(a>=F){
                            printf("Error!!!!!!!!!\n");
                        }
                        if(a>=0 && mask[k]==1){
                            int k_index = n*C*F+c*F+a;
                            if (c==0)
                                indices[i_index+count] = a;
                            val +=feats[k_index];
                            count += 1;   
                            if(!IS_BILI)
                                break;
                        }
                    }
                    if(count>0)
                        out_feats[index] = val/count;
                    else
                        out_feats[index] = 0.0f;
                }
            }
        }
        @alias(feats_t,in0);
        @alias(adj_t,in1);
        @alias(mask_t,in2);
        @alias(output_t,out0);
        @alias(indices_t,out1)
        const int N = feats_t_shape0;
        const int C = feats_t_shape1;
        const int F = feats_t_shape2;
        const int output_size = N*C*F;
        const int thread_per_block = 1024;
        const int block_count = (output_size + thread_per_block - 1) / thread_per_block;
        pool<<<block_count,thread_per_block>>>(output_size,N,C,F,feats_t_p,adj_t_p,mask_t_p,output_t_p,indices_t_p);
        '''
        N,C,F = feats.shape
        feats,indices =  jt.code([(N,C,F),(N,F,3)],[feats.dtype,adj.dtype],inputs=[feats,adj,mask],cuda_header=head,cuda_src=src)
        
        self.indices = indices

        return feats

    def grad(self,grad_output):
        head = r'''
        #include<iostream>
        #define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
        '''
        src = r'''
        __global__ void pool_backward(const int nthreads, 
                            const int N,
                            const int C,
                            const int F,
                            const float* grad_output,
                            const int* indices,
                            float * out_grad){
            CUDA_KERNEL_LOOP(index, nthreads){
                int f = index % F;
                int c = (index / F) % C;
                int n =  index / F / C;
                int i_index = n*F*3+f*3;
                int count = (indices[i_index]>0)+(indices[i_index+1]>0)+(indices[i_index+2]>0);
                for(int i=0;i<count;i++){
                    atomicAdd(out_grad+n*F*C+c*F+indices[i_index+i],grad_output[index]/count);
                }                
            }
        }
        @alias(feats_t,in0);
        @alias(indices_t,in1);
        @alias(output_t,out0);
        const int N = feats_t_shape0;
        const int C = feats_t_shape1;
        const int F = feats_t_shape2;
        const int output_size = N*C*F;
        const int thread_per_block = 1024;
        const int block_count = (output_size + thread_per_block - 1) / thread_per_block;
        cudaMemsetAsync(output_t_p,0,output_t->size);
        pool_backward<<<block_count,thread_per_block>>>(output_size,N,C,F,feats_t_p,indices_t_p,output_t_p);
        '''
        
        grad_feats =  jt.code(grad_output.shape,grad_output.dtype,inputs=[grad_output,self.indices],cuda_header=head,cuda_src=src)

        return grad_feats

unpool_funcv2 = UnPoolFuncV2.apply