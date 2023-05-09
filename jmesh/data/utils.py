import trimesh
import os 
import glob 
import os.path as osp 
import numpy as np 
import math
import random
import scipy
from scipy.spatial.transform import Rotation as R
from jmesh.utils.registry import build_from_cfg,TRANSFORMS

def browse_dataroot(dataroot,mode="train",ext=".obj"):
    files = glob.glob(osp.join(dataroot,"*",mode,f"*{ext}"))
    classes = os.listdir(dataroot)
    labels = [classes.index(f.split("/")[-3]) for f in files]
    return files,classes,labels

def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh

def random_rotation(mesh: trimesh.Trimesh,max_rot_ang_deg=360):
    vertices = np.array(mesh.vertices)
    x = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
    y = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
    z = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
    A = np.array(((np.cos(x), -np.sin(x), 0),
                    (np.sin(x), np.cos(x), 0),
                    (0, 0, 1)),
                dtype=vertices.dtype)
    B = np.array(((np.cos(y), 0, -np.sin(y)),
                    (0, 1, 0),
                    (np.sin(y), 0, np.cos(y))),
                dtype=vertices.dtype)
    C = np.array(((1, 0, 0),
                    (0, np.cos(z), -np.sin(z)),
                    (0, np.sin(z), np.cos(z))),
                dtype=vertices.dtype)
    
    np.dot(vertices, A, out=vertices)
    np.dot(vertices, B, out=vertices)
    np.dot(vertices, C, out=vertices)
    mesh.vertices = vertices
    return mesh


def mesh_normalize(mesh: trimesh.Trimesh):
    vertices = mesh.vertices - mesh.vertices.min(axis=0)
    vertices = vertices / vertices.max()
    mesh.vertices = vertices-0.5
    return mesh


def color_distort(color, trans_range_ratio=0.1, jitter_std=0.05):
  def _color_autocontrast(color, rand_blend_factor=True, blend_factor=0.5):
    assert color.shape[1] >= 3
    lo = color[:, :3].min(0, keepdims=True)
    hi = color[:, :3].max(0, keepdims=True)
    assert hi.max() > 1

    scale = 255 / (hi - lo)
    contrast_feats = (color[:, :3] - lo) * scale

    blend_factor = random.random() if rand_blend_factor else blend_factor
    color[:, :3] = (1 - blend_factor) * color + blend_factor * contrast_feats
    return color

  def _color_translation(color, trans_range_ratio=0.1):
    assert color.shape[1] >= 3
    if random.random() < 0.95:
      tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * trans_range_ratio
      color[:, :3] = np.clip(tr + color[:, :3], 0, 255)
    return color

  def _color_jiter(color, std=0.01):
    if random.random() < 0.95:
      noise = np.random.randn(color.shape[0], 3)
      noise *= std * 255
      color[:, :3] = np.clip(noise + color[:, :3], 0, 255)
    return color

  color = _color_autocontrast(color)
  color = _color_translation(color, trans_range_ratio)
  color = _color_jiter(color, jitter_std)
  return color


def elastic_distort(points, distortion_params=np.array([[0.2, 0.4], [0.8, 1.6]], np.float32)):
  def _elastic_distort(coords, granularity, magnitude):
    blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
    blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
    blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
    coords_min = coords.min(0)

    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    convolve = scipy.ndimage.filters.convolve
    for _ in range(2):
      noise = convolve(noise, blurx, mode='constant', cval=0)
      noise = convolve(noise, blury, mode='constant', cval=0)
      noise = convolve(noise, blurz, mode='constant', cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [np.linspace(d_min, d_max, d)
          for d_min, d_max, d in zip(coords_min - granularity,
                                     coords_min + granularity*(noise_dim - 2),
                                     noise_dim)]
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0)
    coords += interp(coords) * magnitude
    return coords

  assert distortion_params.shape[1] == 2
  if random.random() < 0.95:
    for granularity, magnitude in distortion_params:
      points = _elastic_distort(points, granularity, magnitude)
  return points

@TRANSFORMS.register_module()
class Distort:
    def __init__(self,use_color=True) -> None:
        self.use_color = use_color 

    def __call__(self,mesh):
        scale_factor = 5.12
        xyz = np.array(mesh.vertices)/scale_factor
        if self.use_color:
            color = np.array(mesh.visual.vertex_colors)[:,:3]
            color = color_distort(color)
        xyz = elastic_distort(xyz)
        mesh.vertices = xyz*scale_factor
        if self.use_color:
            mesh.visual.vertex_colors = color
        return mesh

@TRANSFORMS.register_module()
class Scale:
    def __init__(self,):
        pass 
    
    def __call__(self,mesh:trimesh.Trimesh):
        mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
        return mesh 

@TRANSFORMS.register_module()
class Scale2:
    def __init__(self,scale=0.1):
        self.scale=scale 
    
    def __call__(self,mesh:trimesh.Trimesh):
        mesh.vertices = mesh.vertices * np.random.normal(1, self.scale, size=(1, 3))
        return mesh 

@TRANSFORMS.register_module()
class Noise:
    def __call__(self,mesh:trimesh.Trimesh):
        N = len(mesh.vertices)
        mesh.vertices = mesh.vertices * np.random.normal(1, 0.01, size=(N, 3))
        return mesh 

@TRANSFORMS.register_module()
class Rotation:
    def __init__(self,):
        pass 
    
    def __call__(self,mesh:trimesh.Trimesh):
        mesh = random_rotation(mesh)
        return mesh 

@TRANSFORMS.register_module()
class Rotation2:
    def __init__(self,):
        pass 
    
    def __call__(self,mesh:trimesh.Trimesh):
        axis_seq = ''.join(random.sample('xyz', 3))
        angles = [random.choice([0, 90, 180, 270]) for _ in range(3)]
        rotation = R.from_euler(axis_seq, angles, degrees=True)
        mesh.vertices = rotation.apply(mesh.vertices)
        return mesh 

@TRANSFORMS.register_module()
class RandomAxis:
    def __init__(self,):
        pass 
    
    def __call__(self,mesh:trimesh.Trimesh):
        vertices = np.array(mesh.vertices)
        indexes = np.array([0,1,2])
        np.random.shuffle(indexes)
        vertices = vertices[:,indexes]
        mesh.vertices = vertices
        return mesh 

@TRANSFORMS.register_module()
class Rotation3:
    def __init__(self,):
        pass 
    
    def __call__(self,mesh:trimesh.Trimesh):
        theta = np.random.randn(1) * 2 * math.pi

        rot_matrix = np.array([[math.cos(theta), math.sin(theta), 0],
                                        [-math.sin(theta), math.cos(theta), 0],
                                        [0, 0, 1]])
        vertices = np.array(mesh.vertices) @ rot_matrix
        # print(vertices.shape)
        # print(vertices.min(axis=0),vertices.max(axis=0))

        mesh.vertices = vertices
        return mesh 

@TRANSFORMS.register_module()
class Rotation4:
    def __init__(self,):
        pass 
    
    def __call__(self,mesh:trimesh.Trimesh):
        theta = np.random.randint(0,4)*math.pi /2

        rot_matrix = np.array([[math.cos(theta), math.sin(theta), 0],
                                        [-math.sin(theta), math.cos(theta), 0],
                                        [0, 0, 1]])
        vertices = np.array(mesh.vertices)[:,[0,2,1]] @ rot_matrix
        vertices = vertices[:,[0,2,1]]
        # print(vertices.shape)
        # print(vertices.min(axis=0),vertices.max(axis=0))

        mesh.vertices = vertices
        return mesh 

@TRANSFORMS.register_module()
class Normalize:
    def __init__(self,):
        pass 
    
    def __call__(self,mesh:trimesh.Trimesh):
        # vertices = mesh.vertices - mesh.vertices.min(axis=0)
        # vertices = vertices / vertices.max()
        # mesh.vertices = vertices-0.5
        vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        vertices = 2*vertices / (vertices.max()-vertices.min())
        mesh.vertices = vertices
        return mesh

@TRANSFORMS.register_module()
class Normalize2:
    def __init__(self,):
        pass 
    
    def __call__(self,mesh:trimesh.Trimesh):
        vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        vertices = vertices / vertices.max()
        mesh.vertices = vertices
        # vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        # vertices = vertices / (vertices.max()-vertices.min())
        # mesh.vertices = vertices
        return mesh

@TRANSFORMS.register_module()
class Normalize3: 
    def __call__(self,mesh:trimesh.Trimesh):
        vertices = np.array(mesh.vertices)
        max_v = vertices.max(axis=0,keepdims=True)
        min_v = vertices.min(axis=0,keepdims=True)
        center = (max_v+min_v)/2
        center[0,-1] = min_v[0,-1]
        vertices = vertices - center
        mesh.vertices = vertices
        return mesh

@TRANSFORMS.register_module()
class Scale3:
    def __init__(self) -> None:
        self.jitter_sigma  = 0.01
        self.jitter_clip = 0.05

    def __call__(self,mesh):
        vertices = np.array(mesh.vertices)
        jittered_data = np.clip(self.jitter_sigma * np.random.randn(*vertices.shape), -1 * self.jitter_clip, self.jitter_clip)
        vertices = vertices+jittered_data
        mesh.vertices = vertices
        return mesh 
        
@TRANSFORMS.register_module()
class SceneNormalize: 
    def __call__(self,mesh:trimesh.Trimesh):
        vertices = np.array(mesh.vertices)
        center = vertices.mean(axis=0)
        # center[-1] = vertices[:,-1].min()
        vertices = vertices - center
        mesh.vertices = vertices#/vertices.max()
        return mesh

@TRANSFORMS.register_module()
class Collect:
    def __init__(self,feats=[""]):
        pass 

    def __call__(self,mesh:trimesh.Trimesh):
        pass 

@TRANSFORMS.register_module()
class Compose:
    def __init__(self, transforms=None):
        self.transforms = []
        if transforms is None:
            transforms = []
        for transform in transforms:
            if isinstance(transform,dict):
                transform = build_from_cfg(transform,TRANSFORMS)
            elif not callable(transform):
                raise TypeError('transform must be callable or a dict')
            self.transforms.append(transform)

    def __call__(self, mesh: trimesh.Trimesh):
        for t in self.transforms:
            mesh = t(mesh)
        
        return mesh


@TRANSFORMS.register_module()
class LinearTransform:
    def __call__(self, mesh:trimesh.Trimesh):
        m = np.eye(3) + np.random.randn(3, 3) * 0.1
        m[0][0] *= np.random.randint(0, 2) * 2 - 1

        # m *= 0.05
        theta = np.random.randn(1) * 2 * math.pi

        m1 = np.array([[math.cos(theta), math.sin(theta), 0],
                                        [-math.sin(theta), math.cos(theta), 0],
                                        [0, 0, 1]])
        m = np.matmul(m,m1)
        vertices = np.array(mesh.vertices) @ m
        mesh.vertices = vertices

        return mesh

@TRANSFORMS.register_module()
class RandomTranslate:
    def __init__(self,random=True):
        self.random = random
    def __call__(self,mesh):
        vertices = np.array(mesh.vertices)
        vertices = vertices - vertices.mean(axis=0)
        if self.random:
            vertices += np.random.randn(3)
        mesh.vertices = vertices
        return mesh

@TRANSFORMS.register_module()
class RandomTranslate2:
    def __init__(self,random=True):
        self.random = random
    def __call__(self,mesh):
        vertices = np.array(mesh.vertices)
        vertices = vertices - vertices.mean(axis=0)
        if self.random:
            vertices += np.random.uniform(low=-0.2, high=0.2, size=[3])
        mesh.vertices = vertices
        return mesh

@TRANSFORMS.register_module()
class RandomDrop:
    def __init__(self,drop_ratio=0.875) -> None:
        self.drop_ratio = drop_ratio
    def __call__(self,mesh):
        max_dropout_ratio = self.drop_ratio
        faces = mesh.faces 
        dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
        drop_idx = np.where(np.random.random((faces.shape[0]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            faces[drop_idx,:] = faces[0,:] # set to the first point
            mesh.faces = faces
        return mesh

        
@TRANSFORMS.register_module()
class RandomCrop:
    def __init__(self,min_size=0.5,max_size=1):
        assert max_size>=min_size and  min_size>0.1 and max_size<=1
        self.min_size = min_size
        self.max_size = max_size 

    def __call__(self, mesh:trimesh.Trimesh,labels=None):
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        vertex_colors = np.array(mesh.visual.vertex_colors)

        minx,miny,_ = vertices.min(axis=0)
        maxx,maxy,_ = vertices.max(axis=0)
        
        randx = np.random.uniform(minx,maxx)
        randy = np.random.uniform(miny,maxy)
        
        block_x = np.random.uniform(0,1)*(self.max_size-self.min_size)+self.min_size
        block_y = np.random.uniform(0,1)*(self.max_size-self.min_size)+self.min_size

        block_x *= (maxx-minx)
        block_y *= (maxy-miny)

        l = randx-block_x/2
        r = randx+block_x/2
        t = randy-block_y/2
        b = randy+block_y/2

        valid = ((vertices[:,0]>=l) & (vertices[:,0]<=r) & (vertices[:,1]>=t) & (vertices[:,1]<=b))
        valid2 = valid[faces].sum(axis=1) == 3

        valid_vertices = vertices[valid]
        valid_colors  = vertex_colors[valid]
        valid_faces = faces[valid2]
        index_map = np.cumsum(valid)-1
        valid_faces = index_map[valid_faces]

        if labels is not None:
            labels = labels[valid2]

        mesh = trimesh.Trimesh(vertices=valid_vertices,faces=valid_faces,vertex_colors=valid_colors)
        
        if labels is not None:
            return mesh,labels

        return mesh