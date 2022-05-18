import numpy as np
import pandas as pd
import sets_methods as sm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import metrics

def get_repairs_map(REP, k):
    if k == 0:
        X = REP[k, :].reshape(-1, 3)
        return X
    else:
        X = REP[k, :].reshape(3)
        L = get_repairs_map(REP, k - 1)
        Y = sm.get_sets_residual(L, X)
        return Y

def get_unical_repairs(data, dfield='repair_date',lfield='repair_lenght',afield='repair_address', scale=0,values=False):
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

def get_raw_repairs(data, scale=0,values=False):
    group1 = data['Дата окончания ремонта'].value_counts().keys()
    group2 = data['Дата ремонта до аварии'].value_counts().keys()
    repairs = set()

    for rep in group1:
        # print(rep)
        repgroup = data[data['Дата окончания ремонта'] == rep]
        for place in repgroup['Адрес от начала участка.1'].value_counts().keys():
            placegroup = repgroup[repgroup['Адрес от начала участка.1'] == place]
            for length in placegroup['Длина ремонтируемого участка'].value_counts().keys():
                if length >= scale:
                    repair = (rep, place, length)
                    repairs.add(repair)
    for rep in group2:
        repgroup = data[data['Дата ремонта до аварии'] == rep]
        for place in repgroup['Адрес ремонта до аварии'].value_counts().keys():
            placegroup = repgroup[repgroup['Адрес ремонта до аварии'] == place]
            for length in placegroup['Длина ремонта до аварии'].value_counts().keys():
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

def get_merged_repairs(true, synthetic, epsilon=0.5):

    def split_repairs(true, synthetic, epsilon=0.5):
        result = np.array([], dtype=[('a', float), ('b', float), ('date', np.datetime64)]).reshape(-1, 3)
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
        result = np.array([], dtype=[('a', float), ('b', float), ('date', np.datetime64)]).reshape(-1, 3)
        for s in synthetic:
            print(s)
            res = split_repairs(true, s.reshape(-1, 3), epsilon=epsilon)
            print(res)
            result = np.vstack((result, res))
        return result
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

def get_splited_by_repairs(data, index=np.array([],dtype=np.int32),ID='',delta=3):
    #group = data[data['ID простого участка'] == ID]
    group=data.loc[index]
    synthetic = get_unical_repairs(group)
    true=get_raw_repairs(group)
    repairs=get_merged_repairs(true,synthetic,epsilon=0.5)
    repairs['b'] = repairs['Адрес'] + repairs['Длина']
    rep = repairs[['Адрес', 'b', 'Дата ремонта']].values
    # group['state']=True
    for i in np.arange(rep.shape[0]):
        X = rep[i, :]
        a = rep[i, 0]
        b = rep[i, 1]
        rd = rep[i, 2]
        mask = (group['Дата аварии'] <= rd) & (
                    (group['Адрес от начала участка'] <= b) & (group['Адрес от начала участка'] >= a))
        mask1 = (group['Дата аварии'] > rd) & (
                    (group['Адрес от начала участка'] <= b) & (group['Адрес от начала участка'] >= a))
        indices = mask1[mask1 == True].keys()
        #data.loc[indexes, 'Дата ввода'] = rd
        data.loc[indices, 'a'] = a
        data.loc[indices, 'b'] = b
        data.loc[indices, 'new_id'] = str(ID) + '_' + str(i + 1)
        data.loc[mask[mask == True].keys(), 'Дата перевода в бездействие'] = rd
        data.loc[mask[mask == True].keys(), 'Состояние'] = 'Бездействующий'
        repgroup = group[mask]
        if i > 0:
            A = get_repairs_map(rep, i - 1)
            T = sm.get_sets_residual(A, X.reshape(3), f=sm.interseption)[:-1]
            for t in T:
                submask = (repgroup['Дата аварии'] >= t[2]) & ((repgroup['Адрес от начала участка'] <= t[1]) & (
                            repgroup['Адрес от начала участка'] >= t[0]))
                mask = mask | submask
                indices1 = submask[submask == True].keys()
                dlt=(rd-t[2])/np.timedelta64(1,'Y')
                #print(dlt)
                if dlt>=delta:
                    data.loc[indices1, 'Дата ввода'] = t[2]
                data.loc[indices1, 'Дата перевода в бездействие'] = rd
                data.loc[indices1, 'Состояние'] = 'Бездействующий'

        #subgroup = repgroup[mask]
        group = group[~mask]
    le = group['Дата перевода в бездействие'] < group['Дата аварии']
    group.loc[le,'Дата перевода в бездействие']=np.nan
    empty = np.isnan(group['Дата перевода в бездействие'])
    submask = ((group['Состояние'] == 'Бездействующий') | (group['Состояние'] == 'Демонтирован'))&empty
    indices1 = submask[submask == True].keys()
    rd = group[submask]['Дата аварии'].max()
    data.loc[indices1, 'Дата перевода в бездействие'] = rd
    if repairs.shape[0]>0:
        A = get_repairs_map(rep, rep.shape[0] - 1)
        for t in A:
            submask = (group['Дата аварии'] >t[2]) & ((group['Адрес от начала участка'] <= t[1]) & (
                group['Адрес от начала участка'] >= t[0])&((group['Состояние']=='Бездействующий')|(group['Состояние']=='Демонтирован')))
            indices1 = submask[submask == True].keys()
            rd=group[submask]['Дата аварии'].max()
            dlt = (rd - t[2]) / np.timedelta64(1, 'Y')
            if dlt >= delta:
               data.loc[indices1, 'Дата ввода'] = t[2]
            data.loc[indices1, 'Дата перевода в бездействие'] = rd
            #data.loc[indexes1, 'Состояние'] = 'Бездействующий'
    return data

def inscribing(data,*args,**kwargs):
    #отбрасываем записи с пустой координатой или датой аварии

    data['repair_date'] = np.nan
    data['repair_lenght'] = np.nan
    data['repair_address'] = np.nan
    data['repair_index'] = np.nan
    data['comment'] = np.nan
    data['cluster']=np.nan
    columns=dict({'date':'Дата аварии','x':'Адрес от начала участка',
                  'L':'L','age':'Наработка до отказа','water':'Обводненность',
                  'cluster':{'epsilon':0.1,'r':0.5,'length':3000},
                  'repairs':{'delta':1},'split':{'delta':3}})
    for arg in args:
        data[arg]=np.nan

    #try:





    mask=data[afield].isnull()|data[xfield].isnull()
    data=data[~mask]
    data['a']=0
    data['b']=data[lfield]
    #by=kwargs['by']
    by=taufield
    data['new_id'] = data[by].astype('str')
    grouped = data.groupby(by).groups
    for group in grouped:
        index=grouped[group]
        length_approach(data,index,yfield=xfield,xfield=lfield)
        wtp_approach(data,index,xfield=taufield,yfield=wfield)
        set_clusters(data,index,)
        set_repairs_by_clustering(data,index)
        get_splited_by_repairs(data,index,group)
    data['L,м']=data['b']-data['a']
    data['Адрес от начала участка (new)']=data['Адрес от начала участка']-data['a']
    data['Наработка до отказа(new), лет']=(data['Дата аварии']-data['Дата ввода'])/np.timedelta64(1,'Y')





    return data

def set_repairs_by_clustering(data=pd.DataFrame([]),index=np.array([],dtype=np.int32), delta=1):

    def get_cluster_shape(cluster):
        xmin = cluster['Адрес от начала участка'].min()
        xmax = cluster['Адрес от начала участка'].max()
        ymin = cluster['Наработка до отказа'].min()
        ymax = cluster['Наработка до отказа'].max()
        dmin = cluster['Дата аварии'].min()
        dmax = cluster['Дата аварии'].max()
        return (xmin, xmax), (ymin, ymax), (dmin, dmax)

    def get_cluster_interseption(cluster, data):
        x, y, d = get_cluster_shape(cluster)
        mask = (data["Адрес от начала участка"] >= x[0]) & (data["Адрес от начала участка"] <= x[1]) & (
                    data["Наработка до отказа"] > y[1])
        interseptions = data[mask]['cluster'].value_counts().keys()
        return interseptions

    pipe=data.loc[index]
    mask = pipe['cluster'] == -1
    alones = pipe[mask]
    tilde = 0
    for c in pipe[~mask]['cluster'].value_counts().keys():
        cluster = pipe[pipe['cluster'] == c]
        x, y, d = get_cluster_shape(cluster)
        rd = d[1]
        rl = x[1] - x[0]
        if rl < 2:
            rl = 2
        addr = x[0]
        interseption = get_cluster_interseption(cluster, pipe[~mask])
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
                (alones["Адрес от начала участка"] >= x[0]) & (alones["Адрес от начала участка"] <= x[1]) & (
                alones["Наработка до отказа"] < y[0]))]
        data.loc[cluster.index, ['repair_date', 'repair_adress', 'repair_lenght', 'comment', 'repair_index']] = [rd,addr,rl,comment,tilde]
        if alone.shape[0] > 0:
            data.loc[alone.index, ['repair_date', 'repair_adress', 'repair_lenght', 'comment', 'repair_index']] = [rd,addr,rl,comment,tilde]

        tilde = tilde + 1
    return data

def set_clusters(data=pd.DataFrame([]),index=np.array([],dtype=np.int32),epsilon=0.1,r=0.5,length=3000,by="ID простого участка",top_age=30,scaler=MinMaxScaler()):
    #data['cluster']=np.nan
    scaler=scaler
    w=np.array([1,1],dtype=np.float32)
    group = data.loc[index]
    L = group.iloc[0]['L']
    values = group[['Адрес от начала участка', 'Наработка до отказа']].values
    if L < length:
        L = length
    add = np.array([length, top_age])
    values = np.vstack((values, add))
    scaler.fit(values)
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
                line = sm.linear_transform(x=x, y=y)
                if q > 0:
                    # data.loc[indices,'Обводненность']=data.loc[[a,b],'Обводненность'].mean()
                    data.loc[indices, yfield] = data.loc[indices, xfield].apply(lambda x: line.value(x))
                else:
                    data.loc[indices, yfield] = data.loc[b, yfield]
        i = i + 1

def length_approach_v1(data, xfield='L',yfield='Адрес от начала участка',ident='ID простого участка'):
    gby=data.groupby(ident)
    for i,group in enumerate(gby):
        L=group[1][xfield].iloc[0]
        amax=group[1][yfield].max()
        if amax>L:
            data.loc[group[1].index,xfield]=amax

def length_approach(data=pd.DataFrame([]), index=np.array([],dtype=np.int32),xfield='L',yfield='Адрес от начала участка'):
    L=data.loc[index,xfield].iloc[0]
    amax=data.loc[index,yfield].max()
    if amax>L:
        data.loc[index,xfield]=amax