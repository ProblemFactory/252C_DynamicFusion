import numpy as np
from numpy.linalg import norm
from numba import autojit, prange


def toHomo(x):
    # converts points from inhomogeneous to homogeneous coordinates
    if x.ndim == 1:
        return np.hstack((x,1))
    else:
        return np.vstack((x,np.ones((1,x.shape[1]))))


def fromHomo(x):
    # converts points from homogeneous to inhomogeneous coordinates
    return x[:-1] / x[-1]
from numba import autojit, prange

@autojit
def dense_render(K, verts, faces, shape=(480,640)):
    result = np.zeros((shape[0], shape[1], 5))
    verts_sensor = K@verts.T
    verts_sensor = verts_sensor[:-1, :]/verts_sensor[-1:, :]
    verts_sensor = verts_sensor.T
    for v in prange(faces.shape[0]):
        min_i = max(int(np.floor(verts_sensor[faces[v],0].min())), 0)
        min_j = max(int(np.floor(verts_sensor[faces[v],1].min())), 0)
        max_i = min(int(np.ceil(verts_sensor[faces[v],0].max())), result.shape[0]-1)
        max_j = min(int(np.ceil(verts_sensor[faces[v],1].max())), result.shape[1]-1)
        for i in prange(min_i, max_i+1):
            for j in prange(min_j, max_j+1):
                a = verts_sensor[faces[v,0],0] - i
                b = verts_sensor[faces[v,1],0] - verts_sensor[faces[v,0],0]
                c = verts_sensor[faces[v,2],0] - verts_sensor[faces[v,0],0]
                d = verts_sensor[faces[v,0],1] - j
                e = verts_sensor[faces[v,1],1] - verts_sensor[faces[v,0],1]
                f = verts_sensor[faces[v,2],1] - verts_sensor[faces[v,0],1]
                s = (a*f-c*d)/(c*e-b*f)
                t = (b*d-a*e)/(c*e-b*f)
                if s>=0 and s<=1 and t>=0 and t<=1 and s+t<=1:
                    vet = s*verts[faces[v,1], :]+t*verts[faces[v,2], :]+(1-s-t)*verts[faces[v,0], :]
                    if result[i,j,0]==0 or vet[2]<result[i,j,0]:
                        result[i,j,:] = [vet[2],1-s-t,s,t,v]
    return result
    
@autojit
def render(verts, K, shape=(480,640)):
    sensor_plane_pts = K@verts.T
    sensor_plane_pts /= sensor_plane_pts[2,:]
    sensor_plane_pts = sensor_plane_pts[:2,...]
    It = np.zeros(shape, dtype=np.int)-1
    It_mask = np.zeros(shape, dtype=np.bool)
    sensor_plane_pts_int = np.round(sensor_plane_pts).astype(np.int)
    mask = np.all([sensor_plane_pts_int[0,:]>=0, 
                   sensor_plane_pts_int[0,:]<It.shape[0], 
                   sensor_plane_pts_int[1,:]>=0, 
                   sensor_plane_pts_int[1,:]<It.shape[1]], axis=0)

    for i in np.arange(verts.shape[0])[mask]:
        pix_pos = (sensor_plane_pts_int[0, i], sensor_plane_pts_int[1, i])
        if It[pix_pos]==-1 or verts[It[pix_pos], 2]>verts[i,2]:
            It[pix_pos] = i
            It_mask[pix_pos] = 1
        else:
            mask[i] = False
    return sensor_plane_pts_int.T, mask

def cross2mat(v):
    if v.ndim==1:
        w = v[None,:]
    else:
        w = v
    zeros = np.zeros((w.shape[0],))
    result = np.array([[zeros, -w[:,2], w[:,1]], [w[:,2], zeros, -w[:,0]], [-w[:,1], w[:,0], zeros]]).transpose([2,0,1])
    if v.ndim==1:
        return result[0,...]
    return result


# Not used
def quatCross(q, p):
#     Quaternion multiply
    return np.hstack([q[0]*p[0]-np.dot(q[1:], p[1:]), q[0]*p[1:] + \
                    p[0]*q[1:] + np.cross(q[1:], p[1:])])

# Verified
def se32dq(se3):
#     Transform se3(nx6) elements to unit dual quaternions(nx8)
    if se3.ndim == 1:
        theta = norm(se3[:3])
        dq = np.zeros(8)
        dq[0] = np.cos(theta/2)
        dq[1:4] = np.sinc(theta/(2*np.pi))*se3[:3]/2
        a, b, c, d = dq[:4]
        t1, t2, t3 = se3[3:]
        dq[4:] = np.array([
            - b*t1 - c*t2 - d*t3,
              a*t1 + d*t2 - c*t3,
            - d*t1 + a*t2 + b*t3,
              c*t1 - b*t2 + a*t3
        ]) / 2
#         dq[4:] = quatCross(np.hstack((0, se3[3:])), dq[:4]) / 2
        
    else:
        dq = np.zeros((*se3.shape[:-1], 8))
        theta = norm(se3[...,:3], axis=-1, keepdims=True)
        dq[...,:4] = np.concatenate([np.cos(theta/2), np.sinc(theta/(2*np.pi))*se3[...,:3]/2], axis=-1)
#             dq[i,4:] = quatCross(np.hstack((0, se3[i,3:])), dq[i,:4]) / 2
        a, b, c, d = dq[...,:4].T
        t1, t2, t3 = se3[...,3:].T
        dq[...,4:] = np.array([
            - b*t1 - c*t2 - d*t3,
              a*t1 + d*t2 - c*t3,
            - d*t1 + a*t2 + b*t3,
              c*t1 - b*t2 + a*t3
        ]).T / 2
    return dq


def dq2SE3(dq):
    
#     Transform a unit dual quaternion to R and t
#     dq = dq / norm(dq[:4])
    w, x, y, z = dq[...,:4].T
    #Extract rotational information into the new matrix
    mat = np.array([[w*w + x*x - y*y - z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                    [2*x*y + 2*w*z, w*w + y*y - x*x - z*z, 2*y*z - 2*w*x],
                    [2*x*z - 2*w*y, 2*y*z + 2*w*x, w*w + z*z - x*x - y*y]]).transpose([2,0,1])
    #Extract translation information into t
#     t = 2*quatCross(dq[4:], dq[:4]*np.array([1.,-1.,-1.,-1.]))[1:]
    t = 2*(dq[...,0:1]*dq[...,5:] - dq[...,4:5]*dq[...,1:4] + np.cross(dq[...,1:4], dq[...,5:]))

    return mat, t


# This better not be used, since it uses arccos
def dq2se3(dq):
    dq = dq / norm(dq[:4])
    theta = 2 * np.arccos(dq[0])   
    if norm(dq[1:4]) ==0:
        w = np.zeros(3)
        
    else:
        w = dq[1:4] / norm(dq[1:4]) * theta
        
    t = 2*(dq[0]*dq[5:] - dq[4]*dq[1:4] + np.cross(dq[1:4], dq[5:]))
    
    return np.hstack((w, t))

def R2w(R):
    # given a rotation matrix R return the angle-axis representation
    u, s, v = np.linalg.svd(R - np.eye(3))
    v = v[np.argmin(s)]
    vh = np.array([[R[2,1]-R[1,2]], [R[0,2]-R[2,0]], [R[1,0]-R[0,1]]])
    sin = 1/2 * v @ vh
    cos = (np.trace(R) - 1) / 2
    theta = np.arctan2(sin, cos)
    return v * theta


def w2R(w):
    # given the angle-axis representation w return the rotation matrix
    w = dg_se3_test[:, :3]
    theta = norm(w, axis=1)
    R = np.eye(3)[None,...] * np.cos(theta)[:,None,None] +\
        np.sinc(theta/np.pi)[:,None,None]*cross2mat(w) +\
        ((1 - np.cos(theta)) / theta**2)[:,None,None]*(w[...,None]@w[:,None,:])
    return R

# Verified
def drotated_dq0(v, q):
#     v is the UNROTATED 3x1 vector, q is the quaternion(q0)
    w, x, y, z = q
    v1, v2, v3 = v
    return 2*np.array([
        [w*v1 - z*v2 + y*v3, x*v1 + y*v2 + z*v3,
         -y*v1 + x*v2 + w*v3, -z*v1 - w*v2 + x*v3],
        [z*v1 + w*v2 - x*v3, y*v1 - x*v2 - w*v3,
         x*v1 + y*v2 + z*v3, w*v1 - z*v2 + y*v3],
        [-y*v1 + x*v2 + w*v3, z*v1 + w*v2 - x*v3,
         -w*v1 + z*v2 - y*v3, x*v1 + y*v2 + z*v3]
    ])

# Verified
# d_q0 / d_t = 0
def dqe_dt(w):
#     Note that this is actually only a function of w
    w1, w2, w3 = w
#     Small angle approximation
    if norm(w) <= 0.6:
        return np.array([
            [-w1/4, -w2/4, -w3/4],
            [1/2, w3/4, -w2/4],
            [-w3/4, 1/2, w1/4],
            [w2/4, -w1/4, 1/2]
        ])
    
    else:
        theta_2 = 0.5 * norm(w)
        sinc_theta_2 = np.sinc(theta_2/np.pi)
        cos_theta_2 = np.cos(theta_2)

        return np.array([
            [-w1*sinc_theta_2/4, -w2*sinc_theta_2/4, -w3*sinc_theta_2/4],
            [cos_theta_2/2, w3*sinc_theta_2/4, -w2*sinc_theta_2/4],
            [-w3*sinc_theta_2/4, cos_theta_2/2, w1*sinc_theta_2/4],
            [w2*sinc_theta_2/4, -w1*sinc_theta_2/4, cos_theta_2/2]
        ])

def dt_ddq(dq):
#     dq = dq / norm(dq[:4])
    q01, q02, q03, q04, qe1, qe2, qe3, qe4 = dq
    
    dt_dq0 = 2 * np.array([
        [qe2, -qe1, qe4, -qe3],
        [qe3, -qe4, -qe1, qe2],
        [qe4, qe3, -qe2, -qe1]
    ])
    dt_dqe = 2 * np.array([
        [-q02, q01, -q04, q03],
        [-q03, q04, q01, -q02],
        [-q04, -q03, q02, q01]
    ])
    return np.hstack((dt_dq0, dt_dqe))  
  
# Verified
def ddq_dw(w, t):
#     Derivative of dual quaternion WRT w
    w1, w2, w3 = w
    t1, t2, t3 = t
    theta = norm(w)
    dq0_dw = np.zeros((4, 3))
    dqe_dw = np.zeros((4, 3))
#     Small angle approximation
    if theta <= 0.6:
        
        dq0_dw[0] = np.array([-w1/4, -w2/4, -w3/4])
        dq0_dw[1:4] =  0.5 * np.eye(3) - w.reshape(-1,1) @ w.reshape(1,-1) / 24.
        
        dqe_dw[0,0] = 1/48 * (t1*(w1**2 - 12) + w1*(t2*w2 + t3*w3))
        dqe_dw[0,1] = 1/48 * (t2*(w2**2 - 12) + w2*(t1*w1 + t3*w3))
        dqe_dw[0,2] = 1/48 * (t3*(w3**2 - 12) + w3*(t2*w2 + t1*w1))
        dqe_dw[1,0] = -1/48 * w1*(6*t1 - t3*w2 + t2*w3)
        dqe_dw[2,1] = -1/48 * w2*(6*t2 + t3*w1 - t1*w3)
        dqe_dw[3,2] = -1/48 * w3*(6*t3 - t2*w1 + t1*w2)
        dqe_dw[1,1] = 1/48 * (t3*(w2**2 - 12) - w2*(6*t1 + t2*w3))
        dqe_dw[1,2] = 1/48 * (-t2*(w3**2 - 12) - w3*(6*t1 - t3*w2))
        dqe_dw[2,0] = 1/48 * (-t3*(w1**2 - 12) - w1*(6*t2 - t1*w3))
        dqe_dw[2,2] = 1/48 * (t1*(w3**2 - 12) - w3*(6*t2 + t3*w1))
        dqe_dw[3,0] = 1/48 * (t2*(w1**2 - 12) - w1*(6*t3 + t1*w2))
        dqe_dw[3,1] = 1/48 * (-t1*(w2**2 - 12) - w2*(6*t3 - t2*w1))
        
    else:
        
        theta_2 = 0.5 * norm(w)
        sinc_theta_2 = np.sinc(theta_2/np.pi)
        cos_theta_2 = np.cos(theta_2)
        sin_theta_2 = np.sin(theta_2)
        d_sinc = (theta * cos_theta_2 - 2 * sin_theta_2) / theta**3

        dq0_dw =  0.5 * np.array([
            [-w1*sin_theta_2/theta, -w2*sin_theta_2/theta, -w3*sin_theta_2/theta],
            [d_sinc * w1**2 + sinc_theta_2, w1 * w2 * d_sinc, w1 * w3 * d_sinc],
            [w1 * w2 * d_sinc, d_sinc * w2**2 + sinc_theta_2, w2 * w3 * d_sinc],
            [w1 * w3 * d_sinc, w2 * w3 * d_sinc, d_sinc * w3**2 + sinc_theta_2]
        ])

        dqe_dw[0] = - 0.25 * t@w * d_sinc * w - 0.25 *sinc_theta_2 * t
        dqe_dw[1] = w * (t2*w3 - t3*w2) * d_sinc / 4
        dqe_dw[2] = w * (t3*w1 - t1*w3) * d_sinc / 4
        dqe_dw[3] = w * (t1*w2 - t2*w1) * d_sinc / 4

        dqe_dw[1:] -= 0.25 * t.reshape(-1,1) @ w.reshape(1,-1) * sin_theta_2 / theta
        dqe_dw[1:] += cross2mat(t) * sinc_theta_2 / 4

    return np.vstack((dq0_dw, dqe_dw))

