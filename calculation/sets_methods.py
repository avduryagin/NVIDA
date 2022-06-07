import numpy as np

def interseption(C, D,shape=3):
    if shape==3:
        A = np.array(C)
        X = np.array(D)
    else:
        A = np.array(C,dtype=float)
        X = np.array(D,dtype=float)
    a = A[0]
    b = A[1]
    x = X[0]
    y = X[1]
    mask1 = (a < x) & (x < b)
    mask2 = (a < y) & (y < b)
    mask3 = ((x <= a) & (a <= y)) & ((x <= b) & (b <= y))
    # print(A)
    # print(X)
    if mask1 & mask2:
        A[0] = x
        A[1] = y
        # print('returned ',A)
        return A
    if mask1:
        A[0] = x
        # print('returned ',A)
        return A
    if mask2:
        A[1] = y
        # print('returned ',A)
        return A
    if mask3:
        # print('returned ',A)
        return A.reshape(-1, shape)
    return np.array([],dtype=float)

def merge(A=np.array([]),B=np.array([]),shape=2):

    if (A[0] < B[1]) & (A[1] == B[0]):
        return np.array([A[0], B[1]])
    if (B[0] < A[1]) & (A[0] == B[1]):
        return np.array([B[0], A[1]])

    isp=interseption(A,B,shape=2)
    if isp.shape[0]>0:
        if A[0]<B[0]:
            return np.array([A[0],B[1]])
        else:
            return np.array([B[0], A[1]])

    else:
        return np.array([A,B])

def residual(C, D,shape=3):
    if shape==3:
        A = np.array(C)
        X = np.array(D)
    else:
        A = np.array(C,dtype=float)
        X = np.array(D,dtype=float)
    a = A[0]
    b = A[1]
    x = X[0]
    y = X[1]
    mask1 = (a < x) & (x < b)
    mask2 = (a < y) & (y < b)
    mask3 = ((x <= a) & (a <= y)) & ((x <= b) & (b <= y))
    if mask1 & mask2:
        #print('m12')
        A[1] = x
        if A.shape[0] == 3:
            B = np.array([y, b, A[2]])
        else:
            B = np.array([y, b],dtype=float)
        if (A[1]-A[0]>0)&(B[1]-B[0]>0):
            #print('both')
            return np.array([A, B])
        elif (A[1]-A[0]>0):
            #print('A')
            return np.array([A])
        elif (B[1]-B[0]>0):
            #print('B')
            return np.array([B])
        else:
            #print('empty')
            return np.array([],dtype=float)



    if mask1:
        A[1] = x
        #print(A)
        #print('mask1')
        if A[1]-A[0]>0:
            return A
        else: return np.array([],dtype=float)
    if mask2:
        A[0] = y
        #print('mask2')
        if A[1]-A[0]>0:
            return A
        else: return np.array([],dtype=float)
    if mask3:
        #print('mask3')
        return np.array([],dtype=float)
    return A.reshape(-1, shape)



def get_sets_residual(L, X, f=residual,shape=3):
    if shape==3:
        Y = np.array([], dtype=[('a', float), ('b', float), ('date', np.datetime64)]).reshape(-1, shape)
    else:
        Y = np.array([]).reshape(-1, shape)

    for l in L:
        y = f(l, X,shape=shape)

        if len(y) > 0:
            Y = np.vstack((Y, y))
    Y = np.vstack((Y, X.reshape(-1, shape)))
    return Y

def get_disjoint_sets(x=np.array([]),shape=2):
    if x.shape[0]>1:
        a=x[0]
        b=x[0:]
        x_=get_sets_residual(b,a,shape=shape)[:-1]
        y=get_disjoint_sets(x_,shape=shape)
        return np.vstack((a,y))
    else:
        return x

class linear_transform:
    def __init__(self, x=np.array([0,1]),y=np.array([0,1])):
        self.x1 = x[0]
        self.x2 = x[1]
        self.y1 = y[0]
        self.y2 = y[1]
        self.a2 = (self.y1 - self.y2) / (self.x1 - self.x2)
        self.a1 = (self.y1+self.y2 - self.a2*(self.x1+self.x2))*0.5

    def value(self, x):
        y=self.a1 + self.a2 * x
        return y





def mean_approach(*args,**kwargs):
    k=0
    s=0.
    for a in args:
        #if ~np.isnan(a):
        s+=a
        k+=1
    if k>0:
        s=s/k

    def value(x=0):
        return s

    return value
