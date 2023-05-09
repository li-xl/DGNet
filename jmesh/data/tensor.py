class MeshTensor:
    def __init__(self,feats,level,levels,**propertys):
        self._level = level
        self._feats = feats
        self._levels = levels
        self._property = propertys
        
        assert level==None or (level<len(levels) and level>=0)
    
    @property
    def face_adjacency(self,):
        adj =  self._levels[self._level][1]
        N,C,F = self._feats.shape
        assert tuple(adj.shape) == (N,F,3)
        return adj
    
    @property
    def center(self,):
        center =  self._levels[self._level][3]
        return center
    
    @property
    def last_face_adjacency(self,):
        mask = self._levels[self._level-1][1]
        return mask
    
    @property
    def pool_mask(self,):
        mask = self._levels[self._level][0]
        N,C,F = self._feats.shape
        assert tuple(mask.shape) == (N,F)
        return mask
    
    @property
    def last_pool_mask(self,):
        mask = self._levels[self._level-1][0]
        return mask
    
    @property
    def last_indexes(self,):
        indexes = self._levels[self._level-1][4]
        return indexes
    
    @property
    def Fs(self,):
        Fs = self._levels[self._level][2]
        assert self._feats.shape[0] == Fs.shape[0]
        return Fs
    
    @property
    def last_Fs(self,):
        Fs = self._levels[self._level-1][2]
        return Fs

    @property
    def next_Mf(self,):
        assert self._level+1<len(self._levels),f"{self._level},{len(self.levels)}"
        return self._levels[self._level+1][0].shape[1]
    
    @property
    def last_Mf(self,):
        assert self._level>0
        return self._levels[self._level-1][0].shape[1]

    @property
    def feats(self,):
        return self._feats
    
    @property
    def levels(self,):
        return self._levels

    @property
    def level(self,):
        return self._level
    
    def get(self,key):
        if hasattr(self,key):
            return getattr(self,key)
        if key in self._property:
            return self._property[key]
        return None 

    def updated(self,**kwargs):
        data = {
            "feats":self._feats,
            "level":self._level,
            "levels":self._levels,
        }
        data.update(self._property)
        data.update(kwargs)
        return MeshTensor(**data)

    def __add__(self,mesh):
        return self.updated(feats=self.feats+mesh.feats)

    def __mul__(self,mesh):
        return self.updated(feats=self.feats*mesh.feats)
    
    