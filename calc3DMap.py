def calc3DMap_opt(depthmap,K):
    K_inv=LA.inv(K)
    vl_check=np.ones((480,640,3))
    vl_check[:,:,0]=np.tile(np.arange(480),(640,1)).T
    vl_check[:,:,1]=np.tile(np.arange(640),(480,1))
    worldPts=np.transpose(np.matmul(K,np.transpose(vl_check,(1,2,0))),(2,0,1))
    return worldPts*np.reshape(depthmap,(480,640,1))
start_time=time()
depthmap = np.array(Image.open('./photos/frame-000000.depth.png'))
vl_u_check = calc3DMap_opt(depthmap, K)
print(vl_u_check)
print("--- %s seconds ---" % (time() - start_time))