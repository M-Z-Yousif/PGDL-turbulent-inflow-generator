import numpy as np

data=np.load(file=".npy") ## Input the file's path and name to load initial data.

u,v,w,p=np.split(data,4,axis=-1)

mean_u=np.mean(u,axis=0)
mean_v=np.mean(v,axis=0)
mean_w=np.mean(w,axis=0)
mean_p=np.mean(p,axis=0)

u_fluc=u-mean_u
v_fluc=v-mean_v
w_fluc=w-mean_w
p_fluc=p-mean_p

def maxmin(x):
    max_=[]
    min_=[]
    x=x.reshape((4000,128*256))
    for i in range(4000):
        max_.append(max(x[i]))
        min_.append(min(x[i]))
    maxv=max(max_)
    minv=min(min_)
    return maxv,minv

u_max,u_min=maxmin(u_fluc)
v_max,v_min=maxmin(v_fluc)
w_max,w_min=maxmin(w_fluc)
p_max,p_min=maxmin(p_fluc)


data1=np.load(file=".npy") ## Input the file's path and name to load new fluctuation_normalization data from prediction.
u_fnp,v_fnp,w_fnp,p_fnp=np.split(data1,4,axis=-1) 


def denor(x,minn, maxn):
    y=x*(maxn-minn)+minn
    return y

ufn=denor(u_fnp,u_min,u_max)
vfn=denor(v_fnp,v_min,v_max)
wfn=denor(w_fnp,w_min,w_max)
pfn=denor(p_fnp,p_min,p_max)

# s=np.concatenate((uu,vv,ww,pp),axis=-1)
# print(s.shape)


u_p=ufn+mean_u
v_p=vfn+mean_v
w_p=wfn+mean_w
p_p=pfn+mean_p
s=np.concatenate((u_p,v_p,w_p,p_p),axis=-1)


np.save(file=".npy",arr=s) ## Input the file's path and name to save prediction data







