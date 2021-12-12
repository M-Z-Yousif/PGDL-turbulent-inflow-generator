import numpy as np
import sys
import urllib.request

def dl_progress(count, block_size, total_size):
    sys.stdout.write('\r %d%% of %d MB' %(100*count*block_size/total_size, total_size/1024/1024))
savename = "./U_p_1000.npy"## Input the file's path and name to save initial data.

url = "https://dl.dropboxusercontent.com/s/ttjx549flh3hn69/U_p_1000.npy" ## dowmloading link
print('Downloading:',savename)

urllib.request.urlretrieve(url, savename, dl_progress)


data=np.load(file="./U_p_1000.npy") ## Input the file's path and name to load initial data which is same as last one.

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

def nor(x,minn, maxn):
    xx=(x-minn)/(maxn-minn)
    return xx

ufn=nor(u_fluc,u_min,u_max)
vfn=nor(v_fluc,v_min,v_max)
wfn=nor(w_fluc,w_min,w_max)
pfn=nor(p_fluc,p_min,p_max)

s=np.concatenate((ufn,vfn,wfn,pfn),axis=-1)

np.save(file=".npy",arr=data) ## Input the file's path and name to save fluctuation_normalization data














