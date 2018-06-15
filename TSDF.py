import numpy as np
from numba import autojit

@autojit
def update_TSDF(R, t, Dt, K, TSDF_v, TSDF_w, tau, idces, dc_idces):
    #print(R.shape, t.shape)
    #print(R.min(), R.max(), t.min())
    xt = (R@dc_idces+t)[...,0].T
    uc = K@xt
    xt = xt[2,:]
    #print(xt.min(), xt.max())
    uc = uc[:-1,:]/uc[-1,:]
    uc = uc.astype(np.int)
    mask = ~np.any(np.vstack([np.isnan(uc.sum(0)), uc[0,:]<0, uc[1,:]<0, uc[0,:]>=Dt.shape[0], uc[1,:]>=Dt.shape[1]]), axis=0)
    #print(mask.sum())
    uc = uc[:,mask]
    psdf = Dt[uc[0,:],uc[1,:]]-xt[mask]
    psdf[psdf>tau] = tau
    #print(psdf.min(), psdf.max())
    psdf_mask = psdf>-tau
    #print(psdf_mask.sum())
    mask[mask] = np.logical_and(mask[mask], psdf_mask)
    masked_idces = idces[:,mask]
    TSDF_v[masked_idces[0,:], masked_idces[1,:], masked_idces[2,:]] *= TSDF_w[masked_idces[0,:], masked_idces[1,:], masked_idces[2,:]]
    TSDF_v[masked_idces[0,:], masked_idces[1,:], masked_idces[2,:]] += psdf[psdf_mask]
    TSDF_w[masked_idces[0,:], masked_idces[1,:], masked_idces[2,:]] += 1
    TSDF_v[masked_idces[0,:], masked_idces[1,:], masked_idces[2,:]] /= TSDF_w[masked_idces[0,:], masked_idces[1,:], masked_idces[2,:]]
    #print(masked_idces.shape)
