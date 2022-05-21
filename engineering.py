import numpy as np
import pandas as pd
import sets_methods as sm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import metrics

class repairs_map:
    def __init__(self):
        self.rmap=dict([])
        self.shape=0
        self.rep=np.array([])
    def fit(self,rep=np.array([])):
        self.rep=rep
        self.shape=rep.shape[0]
        if self.shape>0:
            self.get_repairs_map(self.shape-1)

    def get_repairs_map(self,k=0):
        if k == 0:
            X = self.rep[k, :].reshape(-1, 3)
            self.rmap.update({k: X})
            return X
        else:
            X = self.rep[k, :].reshape(3)
            L = self.get_repairs_map(k - 1)
            Y = sm.get_sets_residual(L, X)
            self.rmap.update({k:Y})
            return Y



def get_repairs_map(REP, k):
    if k == 0:
        X = REP[k, :].reshape(-1, 3)
        return X
    else:
        X = REP[k, :].reshape(3)
        L = get_repairs_map(REP, k - 1)
        Y = sm.get_sets_residual(L, X)
        return Y


def get_unical_repairs(data, dfield='repair_date',lfield='repair_length',afield='repair_address', scale=0,values=False):
    group = data[dfield].value_counts().keys()
    repairs = set()

    for rep in group:
        # print(rep)
        repgroup = data[data[dfield] == rep]
        for place in repgroup[afield].value_counts().keys():
            placegroup = repgroup[repgroup[afield] == place]
            for length in placegroup[lfield].value_counts().keys():
                if length >= scale:
                    repair = (rep, place, length)
                    repairs.add(repair)
    repairs = pd.DataFrame(list(repairs), columns=['Дата ремонта', 'Адрес', 'Длина']).sort_values(
        by='Дата ремонта').reset_index(drop=True)
    repairs['Дата ремонта'] = pd.to_datetime(repairs['Дата ремонта'])
    if values:
        repairs['b'] = repairs['Адрес'] + repairs['Длина']
        values = repairs[['Адрес', 'b', 'Дата ремонта']].values
        return values
    return repairs

def get_raw_repairs_v1(data, scale=0,values=False, rdfield='Дата окончания ремонта',
               rxfield='Адрес от начала участка_1',rlfield='Длина ремонтируемого участка',
               bdfield='Дата ремонта до аварии',bxfield='Адрес ремонта до аварии',blfield='Длина ремонта до аварии'):
    group1 = data[rdfield].value_counts().keys()
    group2 = data[bdfield].value_counts().keys()
    repairs = set()

    for rep in group1:
        # print(rep)
        repgroup = data[data[rdfield] == rep]
        for place in repgroup[rxfield].value_counts().keys():
            placegroup = repgroup[repgroup[rxfield] == place]
            for length in placegroup[rlfield].value_counts().keys():
                if length >= scale:
                    repair = (rep, place, length)
                    repairs.add(repair)
    for rep in group2:
        repgroup = data[data[bdfield] == rep]
        for place in repgroup[bxfield].value_counts().keys():
            placegroup = repgroup[repgroup[bxfield] == place]
            for length in placegroup[blfield].value_counts().keys():
                if length >= scale:
                    repair = (rep, place, length)
                    repairs.add(repair)

    repairs = pd.DataFrame(list(repairs), columns=['Дата ремонта', 'Адрес', 'Длина']).sort_values(by='Дата ремонта').reset_index(drop=True)
    repairs['Дата ремонта'] = pd.to_datetime(repairs['Дата ремонта'])
    if values:
        repairs['b'] = repairs['Адрес'] + repairs['Длина']
        values = repairs[['Адрес', 'b', 'Дата ремонта']].values
        return values
    return repairs

def get_raw_repairs(data, scale=0,values=False, rdfield='Дата окончания ремонта',
               rxfield='Адрес от начала участка_1',rlfield='Длина ремонтируемого участка',
               bdfield='Дата ремонта до аварии',bxfield='Адрес ремонта до аварии',blfield='Длина ремонта до аварии'):
    #group1 = data[rdfield].value_counts().keys()
    #group2 = data[bdfield].value_counts().keys()
    group=np.vstack((data[[rdfield,rxfield,rlfield]].values,data[[bdfield,bxfield,blfield]].values))
    mask=np.isnan(group[:,1].astype(np.float32))
    group=group[~mask]
    unique=np.unique(group[:,0])
    mask=np.ones(shape=group.shape[0],dtype=bool)
    repairs=[]

    for u in unique:
        index=np.where(group[mask,0]==u)[0]
        unique_place=np.unique(group[index,1])
        for up in unique_place:
            pindex=np.where(group[index,1]==up)[0]
            pindex_=index[pindex]
            unique_length=np.unique(group[pindex_,2])
            for ul in unique_length:
                lindex=np.where(group[pindex_,2]==ul)[0]
                lindex_=pindex_[lindex]
                #mask[lindex_]=False
                if ul>scale:
                    ub=up+ul
                    repair=(up,ub,u,ul)
                    repairs.append(repair)



    if values:
        #types = dict(
            #names=['a', 'b','date'],
            #formats=[ np.float, np.float,'datetime64[s]'])
        #dtype = np.dtype(types)
        #dtype = [('a', float), ('b', float), ('date', np.datetime64)]
        if len(repairs)>0:
            repairs=[x[:3] for x in repairs]
        #result=np.empty(shape=(len(repairs),3),dtype=dtype)
        #result.fill(repairs)

        result = np.array(repairs).reshape(-1,3)

        #result = np.array(repairs, dtype=[ ('date', 'np.datetime64[s]'),('a', float), ('l', float)])
        #repairs['b'] = repairs['Адрес'] + repairs['Длина']
        #values = repairs[['Адрес', 'b', 'Дата ремонта']].values
        return result
    else:
        repairs = pd.DataFrame(repairs, columns=['Адрес', 'b','Дата ремонта', 'Длина']).sort_values(
            by='Дата ремонта').reset_index(drop=True)
        repairs['Дата ремонта'] = pd.to_datetime(repairs['Дата ремонта'])

    return repairs[['Дата ремонта', 'Адрес', 'Длина']]

def get_merged_repairs(true, synthetic, epsilon=0.5,values=False):

    def split_repairs(true, synthetic, epsilon=0.5):
        #types = dict(
            #names=['a', 'b','date'],
            #formats=[ np.float, np.float,'datetime64[s]'])
        #dtype = np.dtype(types)
        #result=np.array([],dtype=dtype).reshape(-1,3)
        result = np.array([]).reshape(-1, 3)
        #result = np.array([], dtype=[('a', np.float32), ('b', np.float32), ('date', np.datetime64)]).reshape(-1, 3)
        if true.shape[0] == 0:
            return synthetic
        for synt in synthetic:
            r = true[0]
            delta = abs((synt[2] - r[2]) / np.timedelta64(1, 'Y'))
            if delta > epsilon:
                res = split_repairs(true[1:], synt.reshape(-1, 3), epsilon=epsilon)
            else:
                ispt = sm.interseption(r, synt)
                if ispt.shape[0] > 0:
                    residual = sm.get_sets_residual(synt.reshape(-1, 3), r)[:-1]
                    # print(residual)
                else:
                    residual = synt.reshape(-1, 3)
                    # print(residual)
                res = split_repairs(true[1:], residual, epsilon=epsilon)
            result = np.vstack((result, res))
        return result

    def merge_repairs(true, synthetic, epsilon=0.5):
        #types = dict(
            #names=['a', 'b','date'],
            #formats=[ np.float, np.float,'datetime64[s]'])
        #dtype = np.dtype(types)
        #result=np.array([],dtype=dtype).reshape(-1,3)
        #result = np.array([], dtype=[('a', float), ('b', float), ('date', np.datetime64)]).reshape(-1, 3)
        result = np.array([]).reshape(-1, 3)
        for s in synthetic:
            #print(s)
            res = split_repairs(true, s.reshape(-1, 3), epsilon=epsilon)
            #print(res)
            result = np.vstack((result, res))
        return result
    if values:
        merged = merge_repairs(true, synthetic, epsilon=epsilon)
        merged = np.vstack((merged, true))

        sargs=np.argsort(merged[:,2])
        return merged[sargs]

    else:
        columns = true.columns
        true['b'] = true['Адрес'] + true['Длина']
        synthetic['b'] = synthetic['Адрес'] + synthetic['Длина']
        true_reps = true[['Адрес', 'b', 'Дата ремонта']].values
        synt_reps = synthetic[['Адрес', 'b', 'Дата ремонта']].values
        merged = merge_repairs(true_reps, synt_reps, epsilon=epsilon)
        merged = np.vstack((merged, true_reps))
        df = pd.DataFrame(merged, columns=['Адрес', 'b', 'Дата ремонта'])
        df['Длина'] = df['b'] - df['Адрес']
        df = df[columns]
        df.sort_values(by='Дата ремонта', inplace=True)
        return df




def get_splited_by_repairs(data, index=np.array([],dtype=np.int32),ID='',delta=3,xfield ='Адрес от начала участка',dfield='Дата аварии',
              efield='Дата ввода',stfield='Состояние',outfield='Дата перевода в бездействие',rdfield='Дата окончания ремонта',
               rxfield='Адрес от начала участка_1',rlfield='Длина ремонтируемого участка',
               bdfield='Дата ремонта до аварии',bxfield='Адрес ремонта до аварии',blfield='Длина ремонта до аварии'):
    #group = data[data['ID простого участка'] == ID]
    group=data.loc[index]
    synthetic = get_unical_repairs(group,values=True)
    true=get_raw_repairs(group,values=True,rdfield=rdfield,rxfield=rxfield,rlfield=rlfield,bdfield=bdfield,bxfield=bxfield,blfield=blfield)
    rep=get_merged_repairs(true,synthetic,epsilon=0.5,values=True)
    #repairs['b'] = repairs['Адрес'] + repairs['Длина']
    #rep = repairs[['Адрес', 'b', 'Дата ремонта']].values
    rmap=repairs_map()
    rmap.fit(rep)
    # group['state']=True
    for i in np.arange(rep.shape[0]):
        X = rep[i]
        a = rep[i][0]
        b = rep[i][1]
        rd = rep[i][2]
        mask = (group[dfield] <= rd) & (
                    (group[xfield] <= b) & (group[xfield] >= a))
        mask1 = (group[dfield] > rd) & (
                    (group[xfield] <= b) & (group[xfield] >= a))
        indices = mask1[mask1 == True].index
        #data.loc[indexes, 'Дата ввода'] = rd
        data.loc[indices, 'a'] = a
        data.loc[indices, 'b'] = b
        data.loc[indices, 'new_id'] = str(ID) + '_' + str(i + 1)
        labels=mask[mask == True].index
        data.loc[labels, outfield] = rd
        data.loc[labels, stfield] = 'Бездействующий'
        repgroup = group[mask]
        if i > 0:
            A=rmap.rmap[i - 1]
            #A = get_repairs_map(rep, i - 1)
            T = sm.get_sets_residual(A, X.reshape(3), f=sm.interseption)[:-1]
            for t in T:
                submask = (repgroup[dfield] >= t[2]) & ((repgroup[xfield] <= t[1]) & (
                            repgroup[xfield] >= t[0]))
                mask = mask | submask
                indices1 = submask[submask == True].keys()
                dlt=(rd-t[2])/np.timedelta64(1,'Y')
                #print(dlt)
                if dlt>=delta:
                    data.loc[indices1, efield] = t[2]
                data.loc[indices1, outfield] = rd
                data.loc[indices1, stfield] = 'Бездействующий'

        #subgroup = repgroup[mask]
        #drop=mask[mask == False].index
        #group=group[~mask]

        group.drop(labels,axis=0,inplace=True)
    le = group[outfield] < group[dfield]
    group.loc[le,outfield]=np.nan
    empty = np.isnan(group[outfield])
    submask = ((group[stfield] == 'Бездействующий') | (group[stfield] == 'Демонтирован'))&empty
    indices1 = submask[submask == True].keys()
    rd = group[submask][dfield].max()
    data.loc[indices1, outfield] = rd
    if rep.shape[0]>0:
        A = rmap.rmap[rep.shape[0] - 1]
        #A = get_repairs_map(rep, rep.shape[0] - 1)
        for t in A:
            submask = (group[dfield] >t[2]) & ((group[xfield] <= t[1]) & (
                group[xfield] >= t[0])&((group[stfield]=='Бездействующий')|(group[stfield]=='Демонтирован')))
            indices1 = submask[submask == True].keys()
            rd=group[submask][dfield].max()
            dlt = (rd - t[2]) / np.timedelta64(1, 'Y')
            if dlt >= delta:
               data.loc[indices1, efield] = t[2]
            data.loc[indices1, outfield] = rd
            #data.loc[indexes1, 'Состояние'] = 'Бездействующий'
    #return data

def inscribing(data,*args,afield ='Наработка до отказа',xfield ='Адрес от начала участка',efield='Дата ввода',
               wfield ='Обводненность',lfield ='L',dfield='Дата аварии',by='ID простого участка',
               stfield='Состояние',outfield='Дата перевода в бездействие',rdfield='Дата окончания ремонта',
               rxfield='Адрес от начала участка_1',rlfield='Длина ремонтируемого участка',
               bdfield='Дата ремонта до аварии',bxfield='Адрес ремонта до аварии',blfield='Длина ремонта до аварии',
               cluster=None, repairs=None,split=None,today=np.datetime64('2021-01-29'),**kwargs):

    #Поле в data:
    #afield ='Наработка до отказа'
    #xfield ='Адрес от начала участка'
    #wfield ='Обводненность'
    #lfield ='L'
    #dfields='Дата аварии'
    #stfield='Состояние'
    #outfield='Дата перевода в бездействие'
    #rdfield='Дата окончания ремонта'
    #rxfield='Адрес от начала участка_1'
    #rlfield='Длина ремонтируемого участка'

    #bdfield='Дата ремонта до аварии'
    #bxfield='Адрес ремонта до аварии'
    #blfield='Длина ремонта до аварии'

    #cluster={'epsilon':0.1,'r':0.5,'length':3000} - словарь с параметрами алгоритма кластеризации
    #repairs={'delta':1} - словарь с параметрами алгоритма агломерации ремонтов
    #split={'delta':3} -словарь с параметрами алгоритма формирование абстракных простых участков
    #by='ID простого участка' -признак группировки простых участков
    #today - Дата выгрузки данный из БД.
    #columns = ['ID простого участка', 'D', 'L', 'S', 'Дата ввода', 'Состояние', 'Дата перевода в бездействие',
               #'Дата аварии', 'Наработка до отказа',
               #'Адрес от начала участка', 'Обводненность', 'Дата окончания ремонта',
               #'Адрес от начала участка_1', 'Длина ремонтируемого участка', 'Дата ремонта до аварии',
               #'Адрес ремонта до аварии', 'Длина ремонта до аварии']

    def fill_wtp(data):
        data[wfield].fillna(0.,inplace=True)
        mask=data[wfield]>100
        data.loc[mask,wfield]=data.loc[mask,wfield]/10.
        mask = data[wfield]< 0
        data.loc[mask, wfield] = np.abs(data.loc[mask, wfield])

    if cluster is  None:
        cluster = {'epsilon': 0.1, 'r': 0.5, 'length': 3000}
    if repairs is None:
        repairs = {'delta': 1}
    if split is None:
        split = {'delta': 3}



    data['repair_date'] = np.nan
    data['repair_length'] = np.nan
    data['repair_address'] = np.nan
    data['repair_index'] = np.nan
    data['comment'] = np.nan
    data['cluster']=np.nan

    for arg in args:
        data.loc[arg]=np.nan
    # отбрасываем записи с пустой координатой или наработкой до отказа
    mask=data[afield].isnull()|data[xfield].isnull()
    drop=mask[mask==True].index
    data.drop(drop,inplace=True)
    data.sort_values(by=afield,inplace=True)
    #data=data[~mask]
    # заполняем пропущенные значения обводненности, исправляем отрицательние или больше 100
    fill_wtp(data)
    # устанавливаем горизонт прогнозирования для действующих участков
    mask = data[stfield] == 'Действующий'
    data.loc[mask, outfield] = today

    data['a']=0
    data['b']=data[lfield]
    data['new_id'] = data[by].astype('str')

    grouped = data.groupby(by).groups
    for group in grouped:
        index=grouped[group]
        length_approach(data,index,yfield=xfield,xfield=lfield)
        wtp_approach(data,index,xfield=afield,yfield=wfield)
        set_clusters(data,index,epsilon=cluster['epsilon'],r=cluster['r'],length=cluster['length'],xfield=xfield,afield=afield,lfield=lfield)
        set_repairs_by_clustering(data,index,delta=repairs['delta'],afield=afield,xfield=xfield,dfield=dfield)
        get_splited_by_repairs(data,index,ID=group,delta=split['delta'],xfield=xfield,dfield=dfield,efield=efield,stfield=stfield,outfield=outfield)
    data['L,м']=data['b']-data['a']
    data['Адрес от начала участка (new)']=data[xfield]-data['a']
    data['Наработка до отказа(new), лет']=(data[dfield]-data[efield])/np.timedelta64(1,'Y')


def set_repairs_by_clustering(data=pd.DataFrame([]),index=np.array([],dtype=np.int32), delta=1,afield ='Наработка до отказа',xfield ='Адрес от начала участка',dfield='Дата аварии'):

    def get_cluster_shape(cluster,afield ='Наработка до отказа',xfield ='Адрес от начала участка',dfield='Дата аварии'):
        xmin = cluster[xfield].min()
        xmax = cluster[xfield].max()
        ymin = cluster[afield].min()
        ymax = cluster[afield].max()
        dmin = cluster[dfield].min()
        dmax = cluster[dfield].max()
        return (xmin, xmax), (ymin, ymax), (dmin, dmax)

    def get_cluster_interseption(cluster, data,afield ='Наработка до отказа',xfield ='Адрес от начала участка'):
        x, y, d = get_cluster_shape(cluster)
        mask = (data[xfield] >= x[0]) & (data[xfield] <= x[1]) & (
                    data[afield] > y[1])
        interseptions = data[mask]['cluster'].value_counts().keys()
        return interseptions

    pipe=data.loc[index]
    mask = pipe['cluster'] == -1
    alones = pipe[mask]
    tilde = 0
    for c in pipe[~mask]['cluster'].value_counts().keys():
        cluster = pipe[pipe['cluster'] == c]
        x, y, d = get_cluster_shape(cluster,afield=afield,xfield=xfield,dfield=dfield)
        rd = d[1]
        rl = x[1] - x[0]
        if rl < 2:
            rl = 2
        addr = x[0]
        interseption = get_cluster_interseption(cluster, pipe[~mask],afield=afield,xfield=xfield)
        comment = 0
        if len(interseption) > 0:
            for i in interseption:
                top = pipe[pipe['cluster'] == i]
                q, p, w = get_cluster_shape(top)
                if p[0] - y[1] <= delta:
                    comment = 1
                    break
        else:
            comment = 0
        alone = alones[(
                (alones[xfield] >= x[0]) & (alones[xfield] <= x[1]) & (
                alones[afield] < y[0]))]
        data.loc[cluster.index, ['repair_date', 'repair_address', 'repair_length', 'comment', 'repair_index']] = [rd,addr,rl,comment,tilde]
        if alone.shape[0] > 0:
            data.loc[alone.index, ['repair_date', 'repair_address', 'repair_length', 'comment', 'repair_index']] = [rd,addr,rl,comment,tilde]

        tilde = tilde + 1
    #return data

def set_clusters(data=pd.DataFrame([]),index=np.array([],dtype=np.int32),epsilon=0.1,r=0.5,length=3000,by="ID простого участка",top_age=30,scaler=MinMaxScaler(),xfield='Адрес от начала участка',afield='Наработка до отказа',lfield='L'):
    #data['cluster']=np.nan
    scaler=scaler
    w=np.array([1,1],dtype=np.float32)
    group = data.loc[index]
    L = group.iloc[0][lfield]
    values = group[[xfield, afield]].values
    #values_=values
    #if L < length:
        #L = length
    add = np.array([length, top_age])
    values_ = np.vstack((values, add))
    scaler.fit(values_)
    tr = scaler.transform(values)
    Epsilon = r / scaler.data_range_[1]
    w[1] = (epsilon / Epsilon) ** 2
    metric = metrics.metrics(w)
    try:
        scan = DBSCAN(min_samples=2, eps=epsilon, metric=metric.euclidian).fit(tr)
        data.loc[group.index, 'cluster'] = scan.labels_
    except(ValueError):
        data.loc[group.index, 'cluster'] = 'error'


def wtp_approach(data=pd.DataFrame([]), index=np.array([],dtype=np.int32), xfield='Наработка до отказа', yfield='Обводненность'):
    #mask - бинарная маска "Обводненность"==0 & "ID простого участка" ==ID
    #data отсортирован по xfield
    mask=data.loc[index,yfield]==0
    if (mask[mask == True].shape[0] == 0) | (mask[mask == True].shape[0] == mask.shape[0]):
        return
    i = 0
    while i < mask.shape[0]:
        if mask.iloc[i]:
            if i > 0:
                a = mask.index[i - 1]
            else:
                a = mask.index[0]

            indices = []
            while mask.iloc[i]:
                indices.append(mask.index[i])
                if i < mask.shape[0] - 1:
                    i = i + 1
                else:
                    break

            if i == mask.shape[0] - 1:
                data.loc[indices, yfield] = data.loc[a, yfield]
            else:
                b = mask.index[i]
                q = data.loc[a, yfield]
                p = data.loc[b, yfield]
                x1 = data.loc[a, xfield]
                x2 = data.loc[b, xfield]
                x=np.array([x1,x2])
                y=np.array([q,p])
                delta=abs(x2-x1)


                if q > 0:
                    if delta>0:
                        line = sm.linear_transform(x=x, y=y)
                        data.loc[indices, yfield] = data.loc[indices, xfield].apply(lambda x: line.value(x))
                    else:
                        line=sm.mean_approach(q,p)
                        data.loc[indices, yfield] = data.loc[indices, xfield].apply(lambda x: line(x))
                else:
                    data.loc[indices, yfield] = data.loc[b, yfield]
        i = i + 1


def length_approach(data=pd.DataFrame([]), index=np.array([],dtype=np.int32),xfield='L',yfield='Адрес от начала участка'):
    L=data.loc[index,xfield].iloc[0]
    amax=data.loc[index,yfield].max()
    if amax>L:
        data.loc[index,xfield]=amax