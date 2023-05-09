import jittor as jt 
import numpy as np 
import trimesh 

def map_face(adj,strict=True):
    # 2 represent pooled, 1 represent reserved, 0 represent unlabeled;
    head=f'''
    #define STRICT {1 if strict else 0}
    '''+r'''
    int check(int now,int last,int A,const int* adj,const int* color){
        if (!STRICT || color[last]==2)
            return !(color[last]-1)+1;
        for(int j=0;j<A;j++){
            int a = adj[now*A+j];
            if(a>=0){
                if(color[a]==2)
                    return 1;
            }
        }
        return 2;
    }
    '''
    src =r'''
    const int* adj = in0_p;
    int* q = in1_p;
    int* color = out0_p;
    
    const int F = in0_shape0;
    const int A = in0_shape1;
    memset(color,0,out0->size);

    for (int i=0;i<F;i++){
        if(color[i]==0){
            int l=0,r=0;
            q[r++]=i;
            color[i] = 1;
            while(l<r){
                int now = q[l++];
                for(int j=0;j<A;j++){
                    int a = adj[now*A+j];
                    if(a>=0 && color[a]==0){
                        q[r++]=a;
                        color[a] = check(a,now,A,adj,color);
                    }
                }
            }
        }
    }
    '''
    F,_ = adj.shape
    q = jt.zeros((F,),dtype="int32")
    color = jt.code((F,),adj.dtype,inputs=[adj,q],cpu_header=head,cpu_src=src)
    assert (color == 0).sum()==0
    return color  

def update_adj(pool_mask,adj):
    '''
    pool_mask: 1 -> reserve 0-> pooled
    '''
    new_adj = -jt.ones_like(adj)
    new_adj, = jt.code(outputs=[new_adj,],
                       inputs=[pool_mask,adj],
                       cpu_src=r'''
    const int* pool_mask = in0_p;
    const int* in_adj = in1_p;
    int* out_adj = out0_p;
    const int F = in0_shape0;
    const int A = in1_shape1;
    for(int f=0;f<F;f++){
        int* oadj = out_adj+f*A;
        if(pool_mask[f]){
            for(int i=0;i<A;i++){
                int a = in_adj[f*A+i];
                if(a>=0 && !pool_mask[a]){
                    
                    /*int k = in_adj[a*A] == f ?1 : (in_adj[a*A+1] == f ? 2 : 0);
                    int next_a = in_adj[a*A+k];
                    if(next_a==-1 || next_a == f || !pool_mask[next_a]){
                        next_a = in_adj[a*A+(k+1)%3];
                    }*/
                    
                    int next_a = -1;
                    for(int j=0;j<A;j++){
                        int now = in_adj[a*A+j];
                        if(now>=0 && now!=f && pool_mask[now]){
                            next_a = now;
                            break;
                        }
                    }
                    oadj[i] = next_a;
                }else{
                    oadj[i] = a;
                }
            }
        }
    }
    ''')
    return new_adj

def face_pool_single(adj):
    '''
    inputs:
        adj:[F,3],
    '''
    color = map_face(adj,strict=True)
    pool_mask = (color ==1).int()
    new_adj = update_adj(pool_mask,adj)
    new_adj = new_adj[color ==1,:]
    remap = jt.array(pool_mask.numpy().cumsum())*pool_mask-1
    mask = new_adj == -1
    new_adj = remap[new_adj]
    new_adj[mask]=-1
    assert mask.sum() == (new_adj==-1).sum()

    return pool_mask,new_adj

def build_face_adjacency(faces):
    edges = faces[:,[0,1,1,2,2,0]].reshape(-1,2)
    edges = np.sort(edges,axis=1)
    m = edges.max().item()+10
    hash_edge = m*edges[:,0]+edges[:,1]
    index = np.argsort(hash_edge)
    face_adjacency = -np.ones_like(faces)
    for i in range(len(index)-1):
        i1 = index[i]
        i2 = index[i+1]
        if hash_edge[i1]==hash_edge[i2]:
            face_adjacency[i1//3,i1%3]=i2//3 
            face_adjacency[i2//3,i2%3]=i1//3
    return face_adjacency

def build_mesh_level(faces,level=9):
    face_adj = build_face_adjacency(faces)
    face_adj = jt.array(face_adj)
    levels = []
    for i in range(level):
        pool_mask,new_adj = face_pool_single(face_adj)
        levels.append([pool_mask.numpy(),face_adj.numpy()])
        face_adj = new_adj
    return levels


