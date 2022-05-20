import numpy as np
import pandas as pd
#import joblib
import pickle

class ClRe:
    def __init__(self,c=np.array([],dtype=float),r=np.array([],dtype=float),
                 t=np.array([],dtype=float),s=np.array([],dtype=float),shape=np.array([],dtype=int)):
        self.c=c
        self.r=r
        self.t=t
        self.s=s
        self.shape=shape
        self.indices=np.arange(c.shape[0])

    def get_items(self,mask=np.array([],dtype=bool),indices=np.array([],dtype=int)):
        if mask.shape[0]>0:
            return ClRe(self.c[mask],self.r[mask],self.t[mask],self.s[mask],self.shape[mask])
        elif indices.shape[0]>0:
            #print(indices)
            return ClRe(self.c[indices],self.r[indices],self.t[indices],self.s[mask],self.shape[indices])
        else:
            return None
class Generator:
    def __init__(self, classifier=None, regressor=None, col=None,path='D:\\ml\\models\\sklearn\\settings\\'):
        if regressor is not None:
            self.regressor=regressor
        else:
            self.regressor=pickle.load(open(path+'rfreg.sav', 'rb'))
        if classifier is not None:
            self.classifier=classifier
        else:
            self.classifier = pickle.load(open(path + 'rfc.sav', 'rb'))

        if col is not None:
            self.col=col
        else:
            self.col = np.load(path + 'col.npy',allow_pickle=True)[()]

        #self.classifier = classifier
        #self.regressor = regressor
        #self.col = col
        self.x = ClRe(c=np.array([]), r=np.array([]))
        self.gindices = np.array([], dtype=int)
        self.mask = np.array([], dtype=bool)
        self.top = np.array([], dtype=float)
        self.prev = np.array([], dtype=float)
        self.columns = np.array([], dtype=int)
        self.down_stairs = dict({'ads05': 0.5, 'ads1': 1., 'ads2': 2., 'ads3': 3.})

    def get_next(self, x=ClRe(c=np.array([], dtype=float), r=np.array([], dtype=float),
                              t=np.array([], dtype=float), s=np.array([], dtype=float), shape=np.array([], dtype=int)),
                 top=np.array([], dtype=float)):
        # прогнозирование класссификационной задачи
        prob = self.classifier.predict_proba(x.c)
        pred_mask = np.array(np.argmax(prob, axis=1), bool)
        if pred_mask[pred_mask == True].shape[0] == 0:
            return None, pred_mask,prob
        # для  1 прогнозируется следующая точка y
        y = self.regressor.predict(x.r[pred_mask]).reshape(-1)
        prev = x.r[pred_mask][:, -1]
        delta = np.abs(y - prev)
        y = prev + delta
        emask = y == prev
        y[emask] = top[pred_mask][emask]
        y_hat = y * x.s[pred_mask]
        x_hat = x.get_items(mask=pred_mask)
        r_tilde = np.hstack((x_hat.r[:, 1:], y.reshape(-1, 1)))
        x_tilde, t_tilde, shape_tilde = self.get_new(x=x_hat.c, tau=y_hat, t=x_hat.t, shape=x_hat.shape)
        return ClRe(c=x_tilde, r=r_tilde, t=t_tilde, shape=shape_tilde, s=x.s[pred_mask]), pred_mask, prob[:, 1]

    def get_counts(self, x=ClRe(c=np.array([], dtype=float), r=np.array([], dtype=float),
                                t=np.array([], dtype=float), s=np.array([], dtype=float),
                                shape=np.array([], dtype=int)),
                   top=np.array([], dtype=float), stop=10):
        self.x = x
        self.top = top
        self.gindices = self.x.indices
        self.mask = np.ones(self.x.indices.shape[0], dtype=bool)
        self.proba = np.zeros(self.x.indices.shape[0], dtype=float)
        self.p0 = np.zeros(self.x.indices.shape[0], dtype=float)
        self.dp = np.zeros(self.x.indices.shape[0], dtype=float)
        self.indices = self.gindices
        i = 1
        while (i < stop) & (self.x.indices.shape[0] > 0):
            y, pred_mask, probab = self.get_next(x=self.x, top=self.top)
            if y is None:
                return self.proba
            iout = self.indices[~pred_mask]
            iin = self.indices[pred_mask]
            self.mask = (y.r[:, -1] <= self.top[pred_mask])
            ige = self.indices[pred_mask][~self.mask]

            sub = pred_mask == True
            pred_mask[sub] = self.mask
            index = self.indices[pred_mask]
            if i == 1:
                self.p0 = probab[pred_mask]
                self.dt = probab[pred_mask]
                self.proba[index]=probab[pred_mask]

            else:
                p0 = self.p0[pred_mask]
                dt = self.dt[pred_mask]
                proba = p0 + dt * probab[pred_mask]
                dt = proba - p0
                self.proba[index] = proba
                self.dt = dt
                self.p0 = proba
            self.x = y.get_items(mask=self.mask)
            self.top = self.top[pred_mask]
            self.indices = self.indices[pred_mask]
            i = i + 1

        return self.proba

    def get_new(self, x=np.array([]), tau=np.array([]), t=np.array([]), shape=np.array([])):
        #
        # [0:'ads', 1:'ads05',2:'ads1', 3'ads2', 4'ads3',
        # 5'ivl0', 6'ivl1', 7'ivl2',8'ivl3', 9'ivl4', 10'ivl5',
        # 11'nivl0', 12'nivl1', 13'nivl2', 14'nivl3', 15'nivl4',15'nivl5',
        # 17'wmean', 18'amean', 19'percent', 20'tau', 21'interval', 22'water', 23'length']
        # t-предыстория
        # tau -новые значения
        y = x.copy()
        y[:, self.col['tau']] = tau
        q = []
        j = 0
        # print(t.shape,shape.shape)
        for i in y:
            q.append(self.set_values(i, t[j], shape[j]))
            j = j + 1
        shape = shape + 1
        return y, np.array(q), shape

    def set_values(self, x=np.array([]), t=np.array([]), shape=np.array(1)):
        tau = x[self.col['tau']]
        q = np.append(t, tau)
        n = q.shape[0]
        for k in self.down_stairs.keys():
            mask = q >= tau - self.down_stairs[k]
            x[self.col[k]] = mask[mask == True].shape[0]
        x[self.col['ads']] = n
        wm = x[self.col['wmean']]
        am = x[self.col['amean']]
        w = x[self.col['water']]
        x[self.col['wmean']] = (wm * shape + w) / (shape + 1)
        x[self.col['amean']] = (am * shape + tau) / (shape + 1)
        if n > 5:
            x[self.col['ivl5']] = x[self.col['ivl5']] + 1
        else:
            x[self.col['ivl' + str(n - 2)]] = x[self.col['ivl' + str(n - 2)]] - 1
            x[self.col['ivl' + str(n - 1)]] = x[self.col['ivl' + str(n - 1)]] + 1
        arr=x[self.col['ivl0']:self.col['ivl5']+1]
        x[self.col['nivl0']:self.col['nivl5'] + 1]=np.cumsum(arr)

        return q


def get_jd_counts(data):
    columns = ['new_id', 'L', 'D', 'interval', 'prevent', 'lost', 'length']
    frame = []
    aggdata = data.groupby('new_id')
    for i, groups in enumerate(aggdata):
        group = groups[1]
        # raw=cdata[cdata['new_id']==groups[0]]
        # print(group.shape)
        # print(group['interval'].value_counts().keys())
        for interval in group['interval'].value_counts().keys():
            subgroup = group[group['interval'] == interval]
            array = []
            array.append(groups[0])
            # array.append(groups[0])
            array.append(subgroup['L,м'].iloc[0])
            array.append(subgroup['D'].iloc[0])
            array.append(interval)
            array.append(subgroup['prevent'].sum())
            array.append(subgroup['lost'].sum())

            lenght = 0
            for j in subgroup['interseption'].value_counts().keys():
                interseption = subgroup[subgroup['interseption'] == j]
                lenght = lenght + interseption.iloc[0]['joined_length']
            array.append(lenght)
            frame.append(array)
    return pd.DataFrame(data=frame, columns=columns)
    # mask=pd.Series(index=raw.index,data=np.ones(raw.shape[0],dtype=bool))
    # mask=forward(data,raw,mask=mask,indices=subgroup.index)
    # mask=backward(data,raw,mask=mask,indices=subgroup.index


def set_fb(data, cdata):
    aggdata = data.groupby('new_id')
    for i, groups in enumerate(aggdata):
        group = groups[1]
        raw = cdata[cdata['new_id'] == groups[0]]
        # print(group.shape)
        # print(group['interval'].value_counts().keys())
        for interval in group['interval'].value_counts().keys():
            subgroup = group[group['interval'] == interval]
            mask = pd.Series(index=raw.index, data=np.ones(raw.shape[0], dtype=bool))
            mask = forward(data, raw, mask=mask, indices=subgroup.index)
            mask = backward(data, raw, mask=mask, indices=subgroup.index)


def set_reverse(data=pd.DataFrame([])):
    aggdata = data.groupby('new_id')
    for i, groups in enumerate(aggdata):
        group = groups[1]
        # raw=cdata[cdata['new_id']==groups[0]]
        # print(group.shape)
        # print(group['interval'].value_counts().keys())
        for interval in group['interval'].value_counts().keys():
            subgroup = group[group['interval'] == interval]
            indices = subgroup.index
            reverse(data, indices=indices)


def reverse(data, indices=np.array([], dtype=int)):
    group = data.loc[indices]
    j = -1
    while j >= -indices.shape[0]:
        i = indices[j]
        j = j - 1
        if data.loc[i, 'base'] == True:
            a = group.loc[i, 'lbound']
            b = group.loc[i, 'rbound']
            date = group.loc[i, 'Дата аварии']
            mask1 = (group['addres'] >= a) & (group['addres'] <= b)
            mask2 = group['Дата аварии'] < date
            # data.loc[i,'base']=True
            mask3 = mask1 & mask2
            index = mask3[mask3 == True].index
            data.loc[index, 'base'] = False


def forward(data, cdata, mask=pd.Series(data=np.array([], dtype=bool)), indices=np.array([], dtype=int)):
    for i in indices:
        if data.loc[i, 'prediction'] == 1:
            a = data.loc[i, 'lbound']
            b = data.loc[i, 'rbound']
            f, pr, pr_y, msk = get_counters(cdata, index=data.loc[i, 'oi'], a=a, b=b, mask=mask)
            mask = msk
            # print(mask[mask==False].shape)
            data.loc[i, 'prevent'] = f
    return mask


def backward(data, cdata, mask=pd.Series(data=np.array([], dtype=bool)), indices=np.array([], dtype=int)):
    for i in indices:
        if data.loc[i, 'prediction'] == 0:
            a = data.loc[i, 'lbound']
            b = data.loc[i, 'rbound']
            f, pr, pr_y, msk = get_counters(cdata, index=data.loc[i, 'oi'], a=a, b=b, mask=mask)
            mask = msk
            data.loc[i, 'lost'] = f
    return mask


def get_counters(group, index, date=3, year=2017, mask=pd.Series(data=np.array([], dtype=bool)), a=0, b=0):
    # a=group.loc[index,'lbound']
    # b=group.loc[index,'rbound']
    mask1 = (group['Адрес от начала участка (new)'] >= a) & (group['Адрес от начала участка (new)'] <= b)
    current = group.loc[index, 'Дата аварии']
    date_out = group.loc[index, 'Дата перевода в бездействие']
    current_age = group.loc[index, 'Наработка до отказа(new), лет']
    getin = group.loc[index, 'Дата ввода']
    get_out = (date_out - getin) / np.timedelta64(1, 'Y')
    mask3 = group['Дата аварии'] > current
    mask4 = group['Дата аварии'] <= current
    mask5 = group['Дата аварии'].dt.year < year
    fimask = mask1 & mask3
    future_ads = group[mask1 & mask3]
    future_ads['delta'] = (future_ads['Дата аварии'] - current) / np.timedelta64(1, 'Y')
    previous = group[mask1 & mask4]
    previous_y = group[mask1 & mask5]
    previous_count = previous.shape[0]
    previous_y_count = previous_y.shape[0]
    # data.loc[indices,'previous_y']=previous_y.shape[0]
    mask6 = future_ads['delta'] <= date
    marked = mask6[mask6 == True].index
    if get_out >= (current_age + date):
        future_ads_y = future_ads[mask & mask6]
        future = future_ads_y.shape[0]
    else:
        we_future_ads_y = future_ads[mask & mask6]
        if we_future_ads_y.shape[0] > 0:
            future = we_future_ads_y.shape[0]
        else:
            future = np.nan
    mask.loc[marked] = False
    return future, previous_count, previous_y_count, mask

