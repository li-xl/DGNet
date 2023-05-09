from jittor import nn 
import jittor as jt 
import math
from jmesh.utils.registry import MODELS 
from jmesh.layers.pool_ops import pool_funcv2,unpool_funcv2 

class MeshReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def execute(self, mesh_tensor):
        feats = self.relu(mesh_tensor.feats)
        return mesh_tensor.updated(feats=feats)

class MeshDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def execute(self, mesh_tensor):
        feats = self.dropout(mesh_tensor.feats)
        return mesh_tensor.updated(feats=feats)


class MeshUnpool(nn.Module):
    def __init__(self,mode="nearest"):
        super().__init__()
        self.mode = mode
        assert mode in ["nearest","bilinear"]

    def execute(self,mesh_tensor):
        feats = mesh_tensor.feats
        N,C,F = feats.shape 

        pool_mask = mesh_tensor.last_pool_mask
        adj = mesh_tensor.last_face_adjacency
        max_f = mesh_tensor.last_Mf
        
        indexes = mesh_tensor.last_indexes
        # TODO update with jittor
        # indexes = (pool_mask.cumsum(dim=1)*pool_mask)-1
        
        feats = feats.reindex(
                shape=[N,C,max_f],
                indexes=[
                    'i0',
                    'i1',
                    '@e0(i0,i2)'
                ],
                extras=[indexes],
                overflow_conditions=['@e0(i0,i2)<0'],
                overflow_value=0,
        )
        is_bilinear=1 if self.mode == "bilinear" else 0
        feats = unpool_funcv2(feats,pool_mask,adj,is_bilinear)

        return mesh_tensor.updated(feats=feats,level=mesh_tensor.level-1)
          

def mesh_concat(meshes):
    new_feats = jt.concat([mesh.feats for mesh in meshes], dim=1)
    return meshes[0].updated(feats=new_feats)

class MeshPool(nn.Module):
    def __init__(self,op="max"):
        super().__init__()
        assert op in ["max","none"]
        self.op = op
    
    def execute(self,mesh_tensor):
        feats = mesh_tensor.feats
        N,C,F = feats.shape 

        pool_mask = mesh_tensor.pool_mask
        adj = mesh_tensor.face_adjacency
        max_f = mesh_tensor.next_Mf
        
        if self.op == "max":
            feats = pool_funcv2(feats,pool_mask,adj)

        indexes = -jt.ones((N,max_f),dtype="int32")
        for i in range(N):
            index, = jt.where(pool_mask[i])
            indexes[i,:index.shape[0]] = index 
        
        feats = feats.reindex(
                shape=[N,C,max_f],
                indexes=[
                    'i0',
                    'i1',
                    '@e0(i0,i2)'
                ],
                extras=[indexes],
                overflow_conditions=['@e0(i0,i2)<0'],
                overflow_value=0,
        )
        
        return mesh_tensor.updated(feats=feats,level=mesh_tensor.level+1)
    
class MeshLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def execute(self, mesh_tensor):
        feats = self.conv1d(mesh_tensor.feats)
        return mesh_tensor.updated(feats=feats)
    
class MeshBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm(num_features, eps, momentum)

    def execute(self, mesh_tensor):
        feats = self.bn(mesh_tensor.feats)
        return mesh_tensor.updated(feats=feats)
    
class MeshInstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.ln = nn.InstanceNorm(num_features, eps)

    def execute(self, mesh_tensor):
        feats = self.ln(mesh_tensor.feats)
        return mesh_tensor.updated(feats=feats)


def dilated_face_adjacencies(FAF, dilation):
    if dilation <= 1:
        return FAF
    N,F,_ = FAF.shape 
    DFA = jt.code(
        shape=[N, F, 3],
        dtype=jt.int32,
        inputs=[FAF, jt.zeros((dilation, 0), dtype=jt.int32)],
        cpu_src="""
            @alias(FAF, in0)
            int dilation = in1_shape0;

            for (int bs = 0; bs < out_shape0; ++bs)
                for (int f = 0; f < out_shape1; ++f)
                    for (int k = 0; k < out_shape2; ++k) {
                        int a = f;
                        int b = @FAF(bs, f, k);
                        for (int d = 1; d < dilation; ++d) {
                            int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                            a = b;
                            if ((d & 1) == 0) {       // go to next
                                b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                            } else {                // go to previous
                                b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                            }
                        }
                        @out(bs, f, k) = b;
                    }
        """,
        cuda_src="""
            __global__ void dilated_face_adjacencies_kernel(@ARGS_DEF) {
                @PRECALC
                @alias(FAF, in0)
                int dilation = in1_shape0;
                int N = in0_shape0;
                int F = in0_shape1;

                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int bs = idx / (F * 3);
                int f = idx / 3 % F;
                int k = idx % 3;

                if (bs >= N)
                    return;

                int a = f;
                int b = @FAF(bs, f, k);
                for (int d = 1; d < dilation; ++d) {
                    if(b==-1) 
                        break;
                    int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                    a = b;
                    if ((d & 1) == 0) {     // go to next
                        b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                    } else {                // go to previous
                        b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                    }
                }
                @out(bs, f, k) = b;
            }

            dilated_face_adjacencies_kernel<<<(in0_shape0*in0_shape1*3-1)/1024+1, 1024>>>(@ARGS);
        """
    )

    return DFA
        

def dilation_radius(index,dis,radius,sample):
    N,F,_ = index.shape
    ret = jt.code((N,F,sample),index.dtype,
               inputs=[index,dis],
               cuda_header=f'''
               #define RADIUS {radius}
               #define SAMPLE {sample}
               #define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
               ''',
               cuda_src=r'''
    __global__ void dilation_index(const int N,const int F,const int M,const int* index,const float* dis,int* out_index){
        CUDA_KERNEL_LOOP(i,N*F){
            int n = i/F;
            int f = i%F;
            int l = 0;
            int r = M;
            int mid = M/2;
            while(l<r){
                mid = (l+r) / 2;
                if(dis[n*F*M+f*M+mid]>RADIUS)
                    r = mid;
                else
                    l = mid+1;
            }
            // printf("%f %f\n",RADIUS,dis[n*F*M+f*M+mid]);
            mid -= SAMPLE/2;
            mid = max(mid,0);
            mid = min(mid,M-SAMPLE);
            for(int j=0;j<SAMPLE;j++){
                out_index[n*F*SAMPLE+f*SAMPLE+j]=index[n*F*M+f*M+j+mid];
            }
        }
    }
    const int* index = in0_p;
    const float* dis = in1_p;
    int* out_index = out0_p;
    const int N = in0_shape0;
    const int F = in0_shape1;
    const int M = in0_shape2;
    const int output_size = N*F;
    const int thread_per_block = 1024;
    const int block_count = (output_size + thread_per_block - 1) / thread_per_block;
    dilation_index<<<block_count,thread_per_block>>>(N,F,M,index,dis,out_index);
               ''')
    return ret 



def sample_adj(feats,adj,Fs,merge_op=None):
    N,C,F = feats.shape
    QN = adj.shape[2]
    adj_feats = feats.reindex(
        shape=[N,C,F,QN],
        indexes=[
            'i0',
            'i1',
            '@e0(i0,i2,i3)'
        ],
        extras=[adj,Fs],
        overflow_conditions=['i2 >= @e1(i0)','@e0(i0,i2,i3)<0'],
        overflow_value=0,
    )
    if merge_op == 'max':
        feats = adj_feats.max(dim=-1)
    elif merge_op == "mean":
        feats = adj_feats.mean(dim=-1)
    else:
        feats = adj_feats
    return feats 

def sample_adj2(feats,adj,Fs):
    N,C,F = feats.shape
    QN = adj.shape[2]
    adj_feats = feats.reindex(
        shape=[N,C,F,QN],
        indexes=[
            'i0',
            'i1',
            '@e0(i0,i2,i3)'
        ],
        extras=[adj,Fs],
        overflow_conditions=['i2 >= @e1(i0)','@e0(i0,i2,i3)<0'],
        overflow_value=0,
    )
    adj_feats -= feats[:,:,:,None]
    return adj_feats

def sample_vbuffer(xyz,
                    direction=jt.array([
                            [0,0,1],
                            [0,0,-1],
                            [0,1,0],
                            [0,-1,0],
                            [1,0,0],
                            [-1,0,0]]),
                    vsize=0.1,
                    sample_size=5,
                    dilation=1):
    assert dilation<=8
    def vhash(i,dims):
        ii = i[...,0]*dims[1]*dims[2]+i[...,1]*dims[2]+i[...,2]
        return ii 

    xyz -= xyz.min(0)
    index = (xyz/vsize).int32()+dilation
    dims = index.max(0)-index.min(0)+1+dilation*2
    adj_index = index[:,None]+direction[None]*dilation
    adj_index = vhash(adj_index,dims)
    index = vhash(index,dims)
    M = (dims[2]*dims[1]*dims[0]).item()
    # assert adj_index.max()<M and adj_index.min()>=0,f"{adj_index.max()},{adj_index.min()}"
    counts = jt.zeros((M,),dtype="int32")
    vbuffer = -jt.ones((M,sample_size),dtype="int32")
    vbuffer,counts = jt.code(
         outputs = [vbuffer,counts],
         inputs=[index],
         cuda_header=r'''
         #define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
         ''',
         cuda_src=r'''
         __global__ void build_vbuffer(const int N,const int M,const int* index,int* counts,int* vbuffer){
            CUDA_KERNEL_LOOP(i,N){
                int c = atomicAdd(counts+index[i],1);
                if(c<M)
                    vbuffer[index[i]*M+c] = i;
            }
        }
        const int* index = in0_p;
        const int N = in0_shape0;
        const int M = out0_shape1;
        int* vbuffer = out0_p;
        int* counts = out1_p;
        const int thread_per_block = 1024;
        const int block_count = (N + thread_per_block - 1) / thread_per_block;
        build_vbuffer<<<block_count,thread_per_block>>>(N,M,index,counts,vbuffer);
         ''',
         )
    adj = vbuffer[adj_index]
    return adj 

def sample_spatialface(mesh,dilation=1,sample_size=5):
    centers = mesh.center
    feats = mesh.feats
    N,C,F = feats.shape
    Fs = jt.array([len(c) for c in centers])
    assert (mesh.Fs!=Fs).sum().item() == 0
    
    direction = jt.array([
                        [0,0,0],
                        [0,0,1],
                        [0,0,-1],
                        [0,1,0],
                        [0,-1,0],
                        [1,0,0],
                        [-1,0,0]])
    # dilation -= 1
    if dilation<=1:
        direction = jt.array([[0,0,0]])
        # dilation=1
        vsize = 0.1
        sample_size *= 6
    else:
        vsize = 0.05*dilation
        dilation = 1
    D = len(direction)
    S = sample_size
    all_adj = -jt.ones((N,F,D,S),dtype="int32")
    for i,c in enumerate(centers):
        adj = sample_vbuffer(c,direction,vsize=vsize,
                    sample_size=sample_size,
                    dilation=dilation)
        all_adj[i,:len(adj),:,:]=adj
    adj_feats = feats.reindex(
        shape=[N,C,F,D,S],
        indexes=[
            'i0',
            'i1',
            '@e0(i0,i2,i3,i4)'
        ],
        extras=[all_adj,Fs],
        overflow_conditions=['i2 >= @e1(i0)','@e0(i0,i2,i3,i4)<0'],
        overflow_value=0,
    )
    adj_feats = adj_feats.max(dim=-1)
    return adj_feats 


class SpatialConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3, dilation=1, stride=1,merge_op="max",groups=1,with_spatial=False,radius=0.1,max_sample=20,temp_sample=1000,bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.merge_op = merge_op
        self.radius = radius 
        self.max_sample = max_sample
        self.with_spatial = with_spatial
        self.temp_sample = temp_sample 
        assert stride == 1 and kernel_size in [1,3]

        assert merge_op in ['max','mean']

        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias,groups=groups)

        ratio_num = 1

        if kernel_size>1:
            self.conv_mf = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False,groups=groups)
            ratio_num+=1
        if with_spatial:
            self.spatial_dilation = math.ceil(radius*10) 
            sd = 7 if self.spatial_dilation>1 else 1
            self.conv_sf = nn.Conv2d(in_channels, out_channels, kernel_size=(1,sd), bias=False,groups=groups)
            
         
            ratio_num+=1

        self.se = nn.Sequential([
            nn.Linear(out_channels, out_channels // 16, bias=False),
            nn.ReLU(),
            nn.Linear(out_channels // 16, out_channels, bias=False),
            nn.Sigmoid()
        ])
        
    def execute(self,mesh):
        feats = mesh.feats
        Fs = mesh.Fs 
        f1 = self.conv_f(feats)

        if self.kernel_size == 3:
            f2 = self.conv_mf(feats)
            adj = mesh.face_adjacency
            adj = dilated_face_adjacencies(adj,self.dilation)
            f2 = sample_adj(f2,adj,Fs,self.merge_op)
        else:
            f2 = 0.
        
        if self.with_spatial:
          
            sample_size = 5

            adj_feats = sample_spatialface(mesh,dilation=self.spatial_dilation,sample_size=sample_size)
            f3 = self.conv_sf(adj_feats)
            assert f3.shape[-1]==1
            f3 = f3[...,0]

        else:
            f3 = 0.
        ff = [f1]
        if isinstance(f2,jt.Var):
            ff.append(f2)
        if isinstance(f3,jt.Var):
            ff.append(f3)
        
        feats = sum(ff)
        attn_c = self.se(feats.mean(dim=-1))
        feats = attn_c[:,:,None]*feats

        return mesh.updated(feats=feats)

class MLP(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,with_bn=True,dropout=0.,bias=False):
        super().__init__()
        self.mlp = SpatialConv(in_channels,out_channels,kernel_size=kernel_size,bias=bias)
        self.relu = MeshReLU()
        if with_bn:
            self.bn = MeshBatchNorm(out_channels)
        else:
            self.bn = nn.Identity()
        
        self.dropout = MeshDropout(dropout)

    def execute(self,mesh):
        mesh = self.mlp(mesh)
        mesh = self.relu(mesh)
        mesh = self.bn(mesh)
        mesh = self.dropout(mesh)
        return mesh 


class SpatialBlock(nn.Module):
    def __init__(self,in_channels,out_channels,dilation=1,radius=None,max_sample=20,temp_sample=1000,merge_op='max'):
        super().__init__()
        with_spatial = radius is not None
        self.conv1 = nn.Sequential([
            SpatialConv(in_channels,out_channels,
                        dilation=dilation,
                        with_spatial=with_spatial,
                        radius=radius,
                        max_sample=max_sample,
                        temp_sample=temp_sample,
                        merge_op=merge_op),
            MeshReLU(),
            MeshBatchNorm(out_channels),
        ])

        self.res1 = nn.Sequential([
            SpatialConv(out_channels,out_channels),
            MeshReLU(),
            MeshInstanceNorm(out_channels),
        ])

    
    def execute(self,mesh):
        mesh = self.conv1(mesh)
        mesh = mesh+self.res1(mesh)
        return  mesh


@MODELS.register_module()
class DGNet(nn.Module):
    def __init__(self,
                 in_channels=16,
                 encoder_channels=[32, 64, 96, 128,128],
                 dilations = [1,1,1,1],
                 radius = [0.,0.2,0.4,0.8],
                 dropouts = [0.,0.,0.,0.],
                 cls_dropouts = None,
                 decoder_channels=[128, 128,128, 96, 96],
                 max_sample = 30,
                 temp_sample = 1000,
                 merge_op="max",
                 use_pool = False,
                 num_classes=21):
        super().__init__()

        self.block1 = nn.Sequential([
            MLP(in_channels,encoder_channels[0],with_bn=False),
            SpatialBlock(encoder_channels[0],encoder_channels[0]),
        ])

        pool_f = MeshPool
        unpool_f = MeshUnpool

        if use_pool:
            self.pool = pool_f(op="max")
            self.unpool = unpool_f(mode="bilinear")
        else:
            self.pool = nn.Identity()
            self.unpool = nn.Identity()
        
        self.depth = len(encoder_channels)-1

        assert len(encoder_channels) == len(decoder_channels)
        assert encoder_channels[-1] == decoder_channels[0]
        assert self.depth == len(radius) and self.depth == len(dilations) and self.depth == len(dropouts)

        
        self.encoders = nn.Sequential()
        for i in range(self.depth):
            self.encoders.append(SpatialBlock(encoder_channels[i],encoder_channels[i+1],dilation=dilations[i],radius=radius[i],max_sample=max_sample,temp_sample=temp_sample,merge_op=merge_op))
        
        self.decoders = nn.Sequential()
        for i in range(self.depth):
            self.decoders.append(SpatialBlock(decoder_channels[i]+encoder_channels[self.depth-1-i],decoder_channels[i+1],dilation=dilations[-i],radius=radius[-i],max_sample=max_sample,temp_sample=temp_sample,merge_op=merge_op))
        
        if cls_dropouts is None:
            self.predict = MeshLinear(decoder_channels[-1],num_classes)
        else:
            self.predict = nn.Sequential()
            for d in cls_dropouts:
                self.predict.append(MeshLinear(decoder_channels[-1],decoder_channels[-1]))
                self.predict.append(MeshDropout(d))
            self.predict.append(MeshLinear(decoder_channels[-1],num_classes))



    def execute(self,mesh):
        # initial
        mesh = self.block1(mesh)
        
        enc_meshs = [mesh]

        # encoder
        for i in range(self.depth):
            mesh = self.pool(mesh)
            mesh = self.encoders[i](mesh)
            if i<self.depth:
                enc_meshs.append(mesh)
        # decoder
        for i in range(self.depth):
            mesh = self.unpool(mesh)    
            enc_mesh = enc_meshs[self.depth-i-1]
            concat_mesh = enc_mesh.updated(feats=jt.concat([mesh.feats,enc_mesh.feats],dim=1))
            mesh = self.decoders[i](concat_mesh)
        
        mesh = self.predict(mesh)

        return mesh.feats


