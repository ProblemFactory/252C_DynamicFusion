import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule
mod = SourceModule("""
#include <stdlib.h>
#include <math.h>

__global__ void dense_render_gpu(float *result, float *verts_sensor, int *faces, int n_rows, int n_cols, int N)
{
    const int v = threadIdx.x + blockIdx.x * blockDim.x;
    if(v>=N)
        return;
    int min_i = max((int)min(min(verts_sensor[faces[v*3]*3], verts_sensor[faces[v*3+1]*3]), verts_sensor[faces[v*3+2]*3]), 0);
    int min_j = max((int)min(min(verts_sensor[faces[v*3]*3+1], verts_sensor[faces[v*3+1]*3+1]), verts_sensor[faces[v*3+2]*3+1]), 0);
    int max_i = min((int)ceil(max(max(verts_sensor[faces[v*3]*3], verts_sensor[faces[v*3+1]*3]), verts_sensor[faces[v*3+2]*3])), n_rows-1);
    int max_j = min((int)ceil(max(max(verts_sensor[faces[v*3]*3+1], verts_sensor[faces[v*3+1]*3+1]), verts_sensor[faces[v*3+2]*3+1])), n_cols-1);
    for(int i=min_i; i<=max_i; ++i)
    {
        for(int j=min_j; j<=max_j; ++j)
        {
            float a = verts_sensor[faces[v*3]*3] - i;
            float b = verts_sensor[faces[v*3+1]*3] - verts_sensor[faces[v*3]*3];
            float c = verts_sensor[faces[v*3+2]*3] - verts_sensor[faces[v*3]*3];
            float d = verts_sensor[faces[v*3]*3+1] - j;
            float e = verts_sensor[faces[v*3+1]*3+1] - verts_sensor[faces[v*3]*3+1];
            float f = verts_sensor[faces[v*3+2]*3+1] - verts_sensor[faces[v*3]*3+1];
            float s = (a*f-c*d)/(c*e-b*f);
            float t = (b*d-a*e)/(c*e-b*f);
            if(s>=0 && s<=1 && t>=0 && t<=1 && s+t<=1)
            {
                float depth = s*verts_sensor[faces[v*3+1]*3+2]+t*verts_sensor[faces[v*3+2]*3+2]+(1-s-t)*verts_sensor[faces[v*3]*3+2];
                if(result[i*n_cols*5+j*5]==0 || depth<result[i*n_cols*5+j*5])
                {
                    result[i*n_cols*5+j*5] = depth;
                    result[i*n_cols*5+j*5+1] = 1-s-t;
                    result[i*n_cols*5+j*5+2] = s;
                    result[i*n_cols*5+j*5+3] = t;
                    result[i*n_cols*5+j*5+4] = v;
                }
            }
        
        }
    }
}
""")

dense_render_gpu_helper = mod.get_function("dense_render_gpu")
def dense_render_gpu(K, verts, faces, shape=(480,640)):
    N = faces.shape[0]

    nThreads = 1024
    nBlocks = (N+nThreads-1)//nThreads

    n_rows, n_cols = shape
    result = np.zeros((n_rows, n_cols, 5), dtype=np.float32)

    verts_sensor = K@verts.T
    verts_sensor = verts_sensor[:-1, :]/verts_sensor[-1:, :]
    verts_sensor = np.ascontiguousarray(np.hstack([verts_sensor.T, verts[:, -1:]]), np.float32)
    faces_cont = np.ascontiguousarray(faces)

    start = time()
    dense_render_gpu_helper(
            drv.Out(result), drv.In(verts_sensor), drv.In(faces_cont), np.int32(n_rows), np.int32(n_cols), np.int32(N),
            block=(nThreads,1,1), grid=(nBlocks,1))
    return result