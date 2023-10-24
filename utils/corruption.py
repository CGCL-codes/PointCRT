import numpy as np
from utils.corrupt_utils import *
from utils import distortion


class corrupt():
    def __init__(self, seed=2023, origin_num=1024) -> None:
        self.seed = seed
        np.random.seed(self.seed)
        self.ORIG_NUM = origin_num

    def __call__(self, corruption):
        MAP = {'uniform': self.uniform_noise,
            'gaussian': self.gaussian_noise,
            'background': self.background_noise,
            'impulse': self.impulse_noise,
            'scale': self.scale,
            'upsampling': self.upsampling,
            'ufsampling': self.uniform_sampling,
            'shear': self.shear,
            'rotation': self.rotation,
            'cutout': self.cutout,
            'density': self.density,
            'density_inc': self.density_inc,
            'distortion': self.ffd_distortion,
            'distortion_rbf': self.rbf_distortion,
            'distortion_rbf_inv': self.rbf_distortion_inv,
            'original': self.original,
        }
        return MAP[corruption]

    def original(self, pointcloud, severity):
        return pointcloud

    ### Noise ###
    '''
    Add Uniform noise to point cloud 
    '''
    def uniform_noise(self, pointcloud, severity):
        #TODO
        N, C = pointcloud.shape
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
        jitter = np.random.uniform(-c,c,(N, C))
        new_pc = (pointcloud + jitter).astype('float32')
        return normalize(new_pc)

    '''
    Add Gaussian noise to point cloud 
    '''
    def gaussian_noise(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
        jitter = np.random.normal(size=(N, C)) * c
        new_pc = (pointcloud + jitter).astype('float32')
        new_pc = np.clip(new_pc,-1,1)
        return new_pc

    '''
    Add noise to the edge-length-2 cude
    '''
    def background_noise(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [N//50, N//40, N//30, N//20, N//10][severity-1]
        jitter = np.random.uniform(-1,1,(c, C))
        new_pc = np.concatenate((pointcloud,jitter),axis=0).astype('float32')
        return normalize(new_pc)

    '''
    Add impulse noise
    '''
    def impulse_noise(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [N//5, N//4, N//3, N//2, N][severity-1]
        index = np.random.choice(self.ORIG_NUM, c, replace=False)
        pointcloud[index] += np.random.choice([-1,1], size=(c,C)) * 0.1
        return normalize(pointcloud)
        

    ### Transformation ###
    '''
    Rotate the point cloud
    '''
    def rotation(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [10, 20, 30, 40, 50][severity-1]
        theta = np.random.uniform(c-2.5,c+2.5) * np.random.choice([-1,1]) * np.pi / 180.
        gamma = np.random.uniform(c-2.5,c+2.5) * np.random.choice([-1,1]) * np.pi / 180.
        beta = np.random.uniform(c-2.5,c+2.5) * np.random.choice([-1,1]) * np.pi / 180.

        matrix_1 = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
        matrix_2 = np.array([[np.cos(gamma),0,np.sin(gamma)],[0,1,0],[-np.sin(gamma),0,np.cos(gamma)]])
        matrix_3 = np.array([[np.cos(beta),-np.sin(beta),0],[np.sin(beta),np.cos(beta),0],[0,0,1]])
        
        new_pc = np.matmul(pointcloud,matrix_1)
        new_pc = np.matmul(new_pc,matrix_2)
        new_pc = np.matmul(new_pc,matrix_3).astype('float32')

        return normalize(new_pc)
    '''
    Shear the point cloud
    '''
    def shear(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [0.1, 0.3, 0.5, 0.7, 0.9][severity-1]
        a = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        b = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        d = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        e = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        f = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        g = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])

        matrix = np.array([[1,0,b],[d,1,e],[f,0,1]])
        new_pc = np.matmul(pointcloud,matrix).astype('float32')
        return normalize(new_pc)

    '''
    Scale the point cloud
    '''
    def scale(self, pointcloud, severity):
        #TODO
        N, C = pointcloud.shape
        c = [0.1, 0.3, 0.5, 0.7, 0.9][severity-1]
        a=b=d=1
        r = np.random.randint(0,3)
        t = np.random.choice([-1,1])
        if r == 0:
            a += c * t
            b += c * (-t)
        elif r == 1:
            b += c * t
            d += c * (-t)
        elif r == 2:
            a += c * t
            d += c * (-t)

        matrix = np.array([[a,0,0],[0,b,0],[0,0,d]])
        new_pc = np.matmul(pointcloud,matrix).astype('float32')
        return normalize(new_pc)


    ### Point Number Modification ###
    '''
    Cutout several part in the point cloud
    '''
    def cutout(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [(10, 30), (15, 40), (15, 45), (18, 45), (16, 56)][severity-1]
        for _ in range(c[0]):
            i = np.random.choice(pointcloud.shape[0],1)
            picked = pointcloud[i]
            dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            # pointcloud[idx.squeeze()] = 0
            pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        # print(pointcloud.shape)
        return normalize(pointcloud)


    '''
    Uniformly sampling the point cloud
    '''
    def uniform_sampling(self, pointcloud, severity):
        N, C = pointcloud.shape
        #! pointnet++ take at least 128 points as input
        c = [200, 400, 600, 800, 896][severity-1]
        index = np.random.choice(self.ORIG_NUM, self.ORIG_NUM - c, replace=False)
        return normalize(pointcloud[index])

    '''
    Upsampling
    '''
    def upsampling(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [N//5, N//4, N//3, N//2, N][severity-1]
        index = np.random.choice(self.ORIG_NUM, c, replace=False)
        add = pointcloud[index] + np.random.uniform(-0.1,0.1,(c, C))
        new_pc = np.concatenate((pointcloud,add),axis=0).astype('float32')
        return normalize(new_pc)

    '''
    Density-based up-sampling the point cloud
    '''
    def density_inc(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [(1,150), (3,150), (4,150), (4,200), (5,200)][severity-1]
        # idx = np.random.choice(N,c[0])
        temp = []
        for _ in range(c[0]):
            i = np.random.choice(pointcloud.shape[0],1)
            picked = pointcloud[i]
            dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            # idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
            # idx = idx[idx_2]
            temp.append(pointcloud[idx.squeeze()])
            pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        
        idx = np.random.choice(pointcloud.shape[0], 1024 - c[0] * c[1])
        temp.append(pointcloud[idx.squeeze()])

        pointcloud = np.concatenate(temp)
        # print(pointcloud.shape)
        return pointcloud

    '''
    Density-based sampling the point cloud
    '''
    def density(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [(1,200), (2,200), (3,200), (4,200), (5,200)][severity-1]
        for _ in range(c[0]):
            i = np.random.choice(pointcloud.shape[0],1)
            picked = pointcloud[i]
            dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
            idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
            idx = idx[idx_2]
            pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
            # pointcloud[idx.squeeze()] = 0
        # print(pointcloud.shape)
        return pointcloud


    def ffd_distortion(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [0.1,0.3,0.4,0.6,0.9][severity-1]
        new_pc = distortion.distortion(pointcloud,severity=c)
        return normalize(new_pc)

    def rbf_distortion(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [(0.1,5),(0.2,5),(0.3,5),(0.4,5),(0.5,5)][severity-1]
        new_pc = distortion.distortion_2(pointcloud,severity=c,func='multi_quadratic_biharmonic_spline')
        return normalize(new_pc).astype('float32')

    def rbf_distortion_inv(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [(0.1,5),(0.2,5),(0.3,5),(0.4,5),(0.5,5)][severity-1]
        new_pc = distortion.distortion_2(pointcloud,severity=c,func='inv_multi_quadratic_biharmonic_spline')
        return normalize(new_pc).astype('float32')


    def SOR(self, pointcloud, severity):
        N, C = pointcloud.shape
        c = [(2, 1.1), (12, 1.2), (18, 1.1), (20, 1.1), (30, 1.2)][severity-1]
        pointcloud = pointcloud.T # [3, N]
        inner = -2. * np.dot(pointcloud.T, pointcloud)  # [N, N]
        xx = np.sum(pointcloud ** 2, axis=0, keepdims=True)  # [1, N]
        dist = xx + inner + xx.T 
        idx = np.argpartition(dist,c[0]+1, axis=-1)[:, 1:c[0]+1]
        n_idx = np.arange(N).reshape(N, 1)
        dist = np.mean(dist[n_idx, idx], axis=-1)
        dist_std = np.std(dist, axis=-1)
        dist_mean = np.mean(dist, axis=-1)
        idx_mask = np.where(dist<= (dist_mean + c[1] * dist_std))[0]
        pointcloud = pointcloud[:,idx_mask]
        num = N - len(idx_mask)
        duplicate_pc = pointcloud[:, :num]
        pointcloud = np.concatenate([pointcloud, duplicate_pc], axis=1)
        return pointcloud.T
