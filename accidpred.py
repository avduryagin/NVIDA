import engineering as en
import generator as gn
import pandas as pd
import numpy as np
class predictor:
    def __init__(self,*args,**kwargs):
        self.data=pd.DataFrame([])
        self.feat = en.features()
        self.gen = gn.Generator()
    def fit(self,data,*args,**kwargs):
        self.data=data
        en.inscribing(self.data,*args,**kwargs)
        self.feat.fit(self.data,*args,**kwargs)

    def predict(self):
        self.predicted = self.gen.predict(self.feat.ClRe, self.feat.horizon)
        self.probab=np.cumsum(np.cumprod(self.gen.p.T,axis=1),axis=1)
        self.time_series=np.multiply(self.gen.r.T,self.feat.s.reshape(-1,1))

    def fill(self):
        self.data['predicted']=np.nan
        self.data['time']=np.nan
        self.data['probab']=np.nan
        index=self.feat.data['index'].reshape(-1)
        self.data.loc[index,'predicted']=self.predicted
        for i,j in enumerate(index):
            self.data.loc[j,'time']=self.time_series[i]
            self.data.loc[j, 'probab'] = self.probab[i]






