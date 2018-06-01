import numpy as np

def update_TSDF(Wt, Dt, K, TSDF_v, TSDF_w, tau, idces, dc_idces, t=0):
    xt = np.matmul(Wt,np.hstack([dc_idces.T, np.ones([dc_idces.shape[1], 1])])[...,None])[:,:,0].T
    uc = K.dot(xt)
    xt = xt[2,:]
    uc = (uc/uc[-1:,:]).astype(int)
    mask = ~np.any(np.vstack([np.isnan(uc.sum(0)), uc[0,:]<0, uc[1,:]<0, uc[0,:]>=Dt.shape[0], uc[1,:]>=Dt.shape[1]]), axis=0)
    psdf = Dt[uc[0,:][mask],uc[1,:][mask]]-xt[mask]
    psdf[psdf>tau] = tau
    psdf_mask = psdf>-tau
    mask[mask] = np.logical_and(mask[mask], psdf_mask)
    masked_idces = idces[:,mask]
    TSDF_v[masked_idces[0,:], masked_idces[1,:], masked_idces[2,:]] += psdf[psdf_mask]
    TSDF_w[masked_idces[0,:], masked_idces[1,:], masked_idces[2,:]] += 1
    TSDF_v[masked_idces[0,:], masked_idces[1,:], masked_idces[2,:]] /= TSDF_w[masked_idces[0,:], masked_idces[1,:], masked_idces[2,:]]