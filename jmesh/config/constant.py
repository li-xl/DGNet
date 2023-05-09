import numpy as np 

SCANNET_CLASS_REMAP = np.zeros(41,dtype=np.int32)
SCANNET_CLASS_REMAP[1] = 1
SCANNET_CLASS_REMAP[2] = 2
SCANNET_CLASS_REMAP[3] = 3
SCANNET_CLASS_REMAP[4] = 4
SCANNET_CLASS_REMAP[5] = 5
SCANNET_CLASS_REMAP[6] = 6
SCANNET_CLASS_REMAP[7] = 7
SCANNET_CLASS_REMAP[8] = 8
SCANNET_CLASS_REMAP[9] = 9
SCANNET_CLASS_REMAP[10] = 10
SCANNET_CLASS_REMAP[11] = 11
SCANNET_CLASS_REMAP[12] = 12
SCANNET_CLASS_REMAP[14] = 13
SCANNET_CLASS_REMAP[16] = 14
SCANNET_CLASS_REMAP[24] = 15
SCANNET_CLASS_REMAP[28] = 16
SCANNET_CLASS_REMAP[33] = 17
SCANNET_CLASS_REMAP[34] = 18
SCANNET_CLASS_REMAP[36] = 19
SCANNET_CLASS_REMAP[39] = 20

SCANNET_MAP = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

SCANNET_COLOR=np.array([
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
])

SCANNET_REMAP_COLOR=np.array([
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (247, 182, 210),		# desk
       (219, 219, 141),		# curtain
       (255, 127, 14), 		# refrigerator
       (158, 218, 229),		# shower curtain
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (227, 119, 194),		# bathtub
       (82, 84, 163),  		# otherfurn
])

SCANNET_CLS_WEIGHT = np.array([0.,          1.25089798,  1.42715083,  3.14175996,  3.44036699,  2.31996208,
  3.50786119 , 3.22160315,  3.12658513 , 3.11693438,  3.69456181,  5.40694323,
  5.50372611 , 3.81466853 , 3.72706873 , 5.1808677  , 5.49659028, 5.36131712,
  5.73269378 , 5.64253038 , 3.18236783])