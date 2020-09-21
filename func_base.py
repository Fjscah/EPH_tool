import numpy as np
import itertools
def expand_mask(array,lp=0,rp=0,up=0,dp=0):
    # array is bool
    lp=int(lp)
    rp=int(rp)
    up=int(up)
    dp=int(dp)
    for n in range(lp):
        array = array +np.roll(array,-1-n,axis=1)
    for n in range(rp):
        array = array +np.roll(array,1+n,axis=1)
    for n in range(up):
        array = array +np.roll(array,-1-n,axis=0)
    for n in range(dp):
        array = array +np.roll(array,1+n,axis=0)
    return array

def find_levels(data,threshold,edge=0,larger=False,dura_p=0):
    '''
    edge:
    0: all plat
    1: left edge
    -1: right edge 
    dura: 
    need keep point time

    return :
    mask , index of finds
    '''
    diff=data
    if isinstance(threshold,(list,np.ndarray)):
        threshold=np.array(threshold)
        threshold=threshold[:,None]
        if larger:
            diff=(diff>threshold)*1
        else:
            diff=(diff<threshold)*1
    else:
        if larger:
            diff=(diff>threshold)*1
        else:
            diff=(diff<threshold)*1
    
    #diff=diff-np.roll(diff,edge,axis=1)
    #diff=(diff>0.5)*1
    dura_p=int(dura_p)
    diff_t=diff
    for n in range(dura_p):
        diff_t=diff_t+np.roll(diff,int(-1-n),axis=1)
    diff=(diff_t>dura_p)*1
    if edge:
        return (diff-np.roll(diff,edge,axis=1))>0.5 #,np.where(diff_t>0.5)
    else:
        return (diff>0.5)#, np.where(diff>0.5)




def array_vec_thre(array,vector,axis=1,large=True):
    if isinstance(vector,(list,np.ndarray)):
        vector =np.array(vector)
        if axis==1:
            vector=vector[:,None]
        elif axis==0:
            vector=vector[None,:]
    #print(vector)
    #print('finds threshold positive',np.where(array>vector))
    if large:
        return array>vector
    else:
        return array<vector

def tuple_to_dict(ttuplt):
    ddict={}
    for n,p in ttuplt:
        if n in ddict:
            ddict[n].append(p)
        else:
            ddict[n]=[p]
    return ddict


def point_in_mask(mask1,mask2,dura_p):
    length_row=len(mask1[1])
    print('00000length_row',length_row)
    print('00000durap',dura_p)

    l=r=0
    find_levels=[]
    nrow=0
    for row1,row2 in zip(mask1,mask2):
        for cur in range(length_row):
            if (not l) and row1[cur]:
                l=cur
            elif l and (not row1[cur]):
                r=cur
                if r-l> dura_p:
                    #print('bigger')
                    if (row2[l:r]>0).any():
                        #print((l,r))
                        find_levels.append((nrow,l,r-1))
                l=0
        nrow=nrow+1
    return find_levels


def boolean_indexing(v):
    '''
    Convert Python sequence to NumPy array, filling missing values
    equal to a=np.array(list(itertools.zip_longest(*v, fillvalue=0))).T
    '''
    lens = np.array([len(item) for item in v])
    #print(lens)
    mask = lens[:,None] > np.arange(lens.max())
    #print(lens[:,None])
    #print(np.arange(lens.max()))
    out = np.zeros(mask.shape,dtype=int)
    out[mask] = np.concatenate(v)
    return out

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
    
   
def interval_length():
    """
    @param row: 0 1 1-D array
    """
    np.roll()