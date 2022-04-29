import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
#% matplotlib
#inline
import pandas as pd
import seaborn as sns
from datetime import datetime
# import tensorflow_probability  as tfp
# import tensorflow as tf
import math
import pylab
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import openpyxl as xl
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, AutoMinorLocator
import matplotlib.patches as mpatches
from sklearn.preprocessing import OneHotEncoder
# from KDEpy import FFTKDE
import calendar
import warnings

warnings.filterwarnings("ignore")


def LoadIntervals(path='D:\\ml\\', file='intervals.csv',
                  short=['R', 'Наработка до отказа, лет', 'Адрес от начала участка, м', 'interval',
                         'interval_ads', 'future_ads', 'last_ads',
                         'first_ads', 'period', 'accident/time',
                         'age', 'first accident', 'last accident',
                         'to accident', 'accident_count',
                         'accident_count/L', 'Дата аварии', 'Дата ввода',
                         'D, мм', 'L. М', 'S, мм',
                         'Обводненность, %',
                         'Скорость потока, м/с'],
                  dates=['Дата ввода', 'Дата ремонта', 'Дата аварии', 'first accident', 'last accident', 'last_ads',
                         'first_ads']):
    # path='D:\\ml\\'
    Data = pd.read_csv(path + file)
    # dates=['Дата ввода','Дата аварии','first accident','last accident','last_ads','first_ads']
    for date in dates:
        Data[date] = pd.to_datetime(Data[date])
    Data['interval_age'] = (Data['Дата аварии'] - Data['first_ads']) / np.timedelta64(1, 'D')
    Data['interval_to_ads'] = (Data['Дата аварии'] - Data['last_ads']) / np.timedelta64(1, 'D')
    Data['S/D'] = Data['S'] / Data['D']
    Data['Адрес от начала участка'].fillna(-1, inplace=True)
    Data = Data[Data['Адрес от начала участка'] != -1]
    return Data


# print(Data.shape)
# Data.info()
def random_split(data, field='interval_ads'):
    xdata = data[data[field] == 1]
    ydata = data[data[field] != 1]
    xtrain, xtest = split_data_by_index(xdata)
    ytrain, ytest = split_data_by_index(ydata)
    train = xtrain.append(ytrain)
    test = xtest.append(ytest)
    sh_train_ind = np.random.permutation(train.index)
    sh_test_ind = np.random.permutation(test.index)
    shuffled_train = train.loc[sh_train_ind]
    shuffled_test = test.loc[sh_test_ind]
    return shuffled_train, shuffled_test


def split_data_by_index_v1(data, index='index', seed=42):
    indexes = pd.DataFrame(data[index].value_counts().keys(), columns=['index'])
    train_index, test_index = split_data(indexes, seed=seed)
    train = data.loc[data[index].isin(np.ravel(train_index.values))]
    test = data.loc[data[index].isin(np.ravel(test_index.values))]
    return train, test
def split_data_by_index(data, index='index', seed=42):
    indices=data[index].value_counts().keys()
    train_index, test_index = split_data(indices, seed=seed)
    train = data.loc[data[index].isin(train_index)]
    test = data.loc[data[index].isin(test_index)]
    return train, test

def split_data(data, ratio=0.2, seed=42):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))
    set_size = int(len(data) * ratio)
    test_indices = shuffled_indices[:set_size]
    train_indices = shuffled_indices[set_size:]
    return data[train_indices], data[test_indices]


def split_by_date(data, date=np.datetime64('2016-01-01'), date_column='Дата аварии'):
    shuffled_indeces = np.random.permutation(len(data))
    cdata = data.iloc[shuffled_indeces]
    if date_column == 'Дата ремонта':
        year = date.astype(object).year
        shuffled_train = cdata[cdata[date_column].dt.year != year]
        shuffled_test = cdata[cdata[date_column].dt.year == year]
    else:
        shuffled_train = cdata[cdata[date_column] < date]
        shuffled_test = cdata[cdata[date_column] >= date]
    # shuffled_test=np.random.permutation(len(test))
    # shuffled_train=train.iloc[shuffled_indices]
    # shuffled_test=test.iloc[shuffled_test]
    return shuffled_train, shuffled_test


def make_features(data, target='accident', indexes=''):
    y = data[target].values
    columns = list(data.columns)
    columns.remove(target)
    features = data[columns]
    X = features.values
    return X, y, features


def numbertodate(data, start):
    y_date = []
    for i in data:
        y_date.append(start + pd.DateOffset(days=i))
    YD = pd.to_datetime(y_date)
    return YD


def datetonumber(data, start, fields):
    for field in fields:
        data[field[1]] = (data[field[0]] - start) / np.timedelta64(1, 'D')
        data = data.drop(field[0], axis=1)
    # data['start']=(data['Дата ввода']-start)/np.timedelta64(1,'D')
    # data['accident']=(data['Дата аварии']-start)/np.timedelta64(1,'D')
    # data['last']=(data['last accident']-start)/np.timedelta64(1,'D')
    # data['first']=(data['first accident']-start)/np.timedelta64(1,'D')
    # data=data.drop('Дата ввода', axis=1)
    # data=data.drop('Дата аварии', axis=1)
    # data=data.drop('last accident', axis=1)
    # data=data.drop('first accident', axis=1)
    return data


def get_dummy(data, List):
    # List=[	'Дата аварии','Дата ввода','D, мм',	'L. М',	'S, мм',	'Материал трубы','Тип трубы',	'Завод изготовитель','Режим течения','Скорость потока, м/с']

    long = data[List]
    dummies_long = pd.get_dummies(long)
    return dummies_long


def fill_median(data, speed_median, water_median, addr=0):
    data['Скорость потока'].fillna(speed_median, inplace=True)
    data['Обводненность'].fillna(water_median, inplace=True)

    data['Адрес от начала участка (new)'].fillna(addr, inplace=True)
    # data['Обводненность, %'].fillna(water_median, inplace=True)
    return data


def robustscaler(data):
    scaler = RobustScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled


def feature_log(data, fields):
    ndata = data.copy()
    for field in fields:
        ndata[field] = np.log(data[field] + 1)
    # ndata['Скорость потока, м/с']=np.log(data['Скорость потока, м/с']+1)
    # ndata['accident_count']=np.log(data['accident_count']+1)
    # ndata['future accidents']=np.log(data['future accidents']+1)
    return ndata


def drop_columns(data, fields=[]):
    for field in fields:
        data = data.drop(field, axis=1)
    return data


def get_accident_counts(data, date=1):
    ndata = data.sort_values(by=['ID простого участка', 'Дата аварии'])
    # ndata=ndata.loc[0:5,:]
    ndata['accident_count'] = np.nan
    ndata['first accident'] = np.nan
    ndata['last accident'] = np.nan
    ndata['age'] = np.nan
    ndata['future accidents'] = np.nan
    ndata['total accidents'] = np.nan
    ID = ndata['ID простого участка'].value_counts()
    values = []
    previous = []
    # print(ID)

    for id in ID.keys():
        group = ndata[ndata['ID простого участка'] == id]
        # rg=list(range(group.shape[0]))
        total = group.shape[0]
        # print(id)
        # print(rg)
        # values.append(rg)
        k = 1
        f = group.index.shape[0]
        for pipe in group.index:
            ndata.loc[pipe, 'accident_count'] = k
            f_accident = pd.to_datetime(group.loc[pipe, 'Дата аварии'])
            future_accidents = group[group['Дата аварии'] >= f_accident]
            future_accidents['delta'] = (future_accidents['Дата аварии'] - f_accident) / np.timedelta64(1, 'Y')

            ndata.loc[pipe, 'future accidents'] = future_accidents[future_accidents['delta'] <= date].shape[0] - 1
            ndata.loc[pipe, 'total accidents'] = total

            if k == 1:
                # previous.append(group.loc[pipe,'Дата ввода'])
                first_accident = group.loc[pipe, 'Дата аварии']
                ndata.loc[pipe, 'last accident'] = group.loc[pipe, 'Дата ввода']
                accident = group.loc[pipe, 'Дата аварии']
                start = group.loc[pipe, 'Дата аварии']
                ndata.loc[pipe, 'first accident'] = first_accident
                k = k + 1
            else:
                ndata.loc[pipe, 'first accident'] = first_accident
                ndata.loc[pipe, 'last accident'] = accident

                accident = group.loc[pipe, 'Дата аварии']
                k = k + 1

    ndata['accident_count'] = ndata['accident_count'].astype(np.int64)
    ndata['future accidents'] = ndata['future accidents'].astype(np.int64)
    ndata['L. М'] = ndata['L. М'].where(ndata['L. М'] > 0, 1)
    ndata['accident_count/L'] = ndata['accident_count'].values / ndata['L. М'].values
    # ndata['accident_count/L']=ndata
    ndata['last accident'] = pd.to_datetime(ndata['last accident'])
    ndata['first accident'] = pd.to_datetime(ndata['first accident'])
    ndata['to accident'] = (ndata['Дата аварии'] - ndata['last accident']) / np.timedelta64(1, 'D')
    ndata['age'] = (ndata['last accident'] - ndata['Дата ввода']) / np.timedelta64(1, 'D')
    ndata['accident/time'] = ((ndata['Дата аварии'] - ndata['first accident']) / np.timedelta64(1, 'D')) / (
            ndata['accident_count'] + 1)
    # ndata['last accident']=ndata['to accident']
    return ndata


def get_pca(data, test=None, n_components=2):
    pca2 = PCA(n_components)
    pca2.fit(data.values)
    X_pca2 = pca2.transform(data.values)
    # X_train2,X_test2,y_train2,y_test2=train_test_split(X_pca2,y)
    # X_pca_test2=pca2.transform(X_sd_test)
    plt.figure(figsize=(25, 8))
    plt.matshow(pca2.components_, cmap='viridis', fignum=1)
    plt.yticks([0, 1], ['1 компонента', '2 компонента'])
    plt.colorbar()
    plt.xticks(range(len(data.columns)), data.columns, rotation=90, ha='left')
    plt.xlabel('Характеристика')
    plt.ylabel('Главные компоненты')
    k = 1
    for c in pca2.explained_variance_ratio_:
        print('variance_', k, '=', c)
        k = k + 1

    if test is not None:
        X_pca2_test = pca2.transform(test.values)
        return X_pca2, X_pca2_test

    return X_pca2


def fillresults(data, F, fild):
    for index in F.index:
        data.loc[index, 'Predicted'] = F.loc[index, fild]
    return data


def addoffset(date, offt):
    years = np.modf(offt)[1]
    month = np.modf(offt)[0] * 12
    months = np.modf(month)[1]
    day = np.modf(month)[0]
    #print(years,month,months,day)
    ty = date + pd.DateOffset(years=years)
    tm = ty + pd.DateOffset(months=months)
    ndays = calendar.monthrange(tm.year, tm.month)[1]
    td = tm + pd.DateOffset(days=np.modf(ndays * day)[1])
    return td


def to_date(date):
    date['Date_Predicted'] = np.nan
    for index in date.index:
        D = pd.to_datetime(date.loc[index, 'Дата ввода'])
        offset = date.loc[index, 'Predicted'].astype(np.float)
        date.loc[index, 'Date_Predicted'] = addoffset(D, offset)
    return date


def make_results(data, R, field):
    result = fillresults(data, R, field)
    mask = pd.isnull(result[field])
    result['mask'] = mask
    res = result[result['mask'] == False]
    res = to_date(res)
    return res[['Название', 'corr', 'up', 'ID простого участка', 'Месторождение', 'Состояние',
                'D, мм', 'L. М', 'S, мм', 'Материал трубы', 'Тип трубы',
                'Завод изготовитель', 'Дата ввода', 'Дата аварии', 'Date_Predicted',
                'Наработка до отказа, лет', 'Predicted', 'Адрес от начала участка, м',
                'Обводненность, %', 'Скорость потока, м/с', 'Режим течения']]


def plotfigures(data, yt, predicted, target='Наработка до отказа, лет',
                L=['accident_count', 'accident_count/L', 'D, мм', 'L. М', 'S, мм',
                   'Обводненность, %', 'Скорость потока, м/с', 'start'], title='', abs=True):
    R = data[L].copy()
    if abs:
        residual = np.abs(predicted - yt)
    else:
        residual = predicted - yt
    R['Разность'] = residual
    R[target] = yt
    R['Predicted'] = predicted
    # print(title)
    R[['Разность', target, 'Predicted']].hist(figsize=(14, 10), bins=50)
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(1, 1, 1)

    plt.title('Гистограммы прогнозируемых величин' + '. ' + title)
    n, bins, patche = plt.hist(R[target], bins=50, alpha=0.5, rwidth=0.85, label='Expected', color='blue')
    plt.hist(R['Predicted'], bins=50, alpha=0.5, rwidth=0.85, label='Predicted', color='red')
    plt.legend(loc='upper right')
    plt.title('Совмещенная гистограмма Expected-Predicted' + '. ' + title)
    plt.xlabel(target, fontsize=10)
    plt.ylabel('Частота', fontsize=10)
    majorFormatter = FormatStrFormatter('%.1f')
    minorFormatter = FormatStrFormatter('%.2f')
    minorLocator = AutoMinorLocator(n=2)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_formatter(minorFormatter)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    for label in ax.xaxis.get_ticklabels(which='major'):
        label.set_color('red')
        label.set_rotation(90)
        label.set_fontsize(12)
    for label in ax.xaxis.get_ticklabels(which='minor'):
        label.set_color('black')
        label.set_rotation(90)
        label.set_fontsize(10)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1On = True
        tick.label1.set_color('red')
        tick.label2On = False
        tick.label2.set_color('black')
        # серые деления на оси ОY слева
        tick.tick1line.set_color('black')
        tick.tick1line.set_markeredgewidth(2)
        tick.tick1line.set_markersize(15)

    for tick in ax.xaxis.get_minor_ticks():
        tick.label1On = True
        tick.label1.set_color('black')
        tick.label2On = False
        tick.label2.set_color('black')
        tick.tick1line.set_color('black')
        tick.tick1line.set_markeredgewidth(1)
        tick.tick1line.set_markersize(8)
    plt.grid(True)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.figure(figsize=(12,10))
    # plt.subplot(1,1,1)
    # plt.hist(R['Разность'], alpha=0.5, bins=50)
    # plt.title('Разность')
    g = plt.figure(figsize=(20, 8))
    ax = plt.subplot(1, 1, 1)
    n, bins, patches = plt.hist(x=residual, bins=50, color='#0504aa',
                                alpha=0.75, rwidth=0.85)
    points = []
    a = bins[0]
    for b in bins:
        if b != a:
            points.append((a + b) / 2)
            a = b
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.5)
    plt.xlabel('Ошибка', fontsize=10)
    plt.ylabel('Частота', fontsize=10)
    plt.title('Гистограмма распределения ошибок прогнозирования' + '. ' + title, fontsize=18)
    f = plt.figure(figsize=(20, 8))
    plt.xlabel('Ошибка ', fontsize=10)
    plt.ylabel('Проценты', fontsize=10)
    plt.title('Гистограмма распределения ошибок прогнозирования (%)' + '. ' + title, fontsize=18)

    ax = f.add_subplot(1, 1, 1)
    N = n * 100 / np.sum(n)
    majorFormatter = FormatStrFormatter('%.2f')
    minorFormatter = FormatStrFormatter('%.2f')
    minorLocator = AutoMinorLocator(n=4)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_formatter(minorFormatter)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.rc('xtick', direction='in')
    db = bins[1] - bins[0]
    ax.bar(points, N, db * 0.9)
    db = bins[1] - bins[0]
    step = (1 - np.mod(1, db)) / db
    ##print('step=', step)
    # print('points=', points)
    # print('n=', n)
    s = 0
    k = 0
    l = 0
    NY = []
    H = []
    pointsy = []
    a = bins[0]
    max = N[0]
    for nr in N:
        if k <= step:
            s = s + nr
            k = k + 1
            if nr > max:
                max = nr
        else:
            b = bins[l]
            pointsy.append((b + a) / 2)
            a = b
            k = 1
            NY.append(s)
            H.append(max)
            s = nr
            max = nr
        l = l + 1
        # print(pointsy)
    # print( db*step)
    rects = ax.bar(pointsy, H, db * step, alpha=0.25)
    # print('rect', len(rects))
    ax.grid(True)
    plt.rc('xtick', direction='in')
    for label in ax.xaxis.get_ticklabels(which='major'):
        label.set_color('red')
        label.set_rotation(90)
        label.set_fontsize(12)
    for label in ax.xaxis.get_ticklabels(which='minor'):
        label.set_color('black')
        label.set_rotation(90)
        label.set_fontsize(10)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1On = True
        tick.label1.set_color('red')
        tick.label2On = False
        tick.label2.set_color('black')
        # серые деления на оси ОY слева
        tick.tick1line.set_color('black')
        tick.tick1line.set_markeredgewidth(2)
        tick.tick1line.set_markersize(15)

    for tick in ax.xaxis.get_minor_ticks():
        tick.label1On = True
        tick.label1.set_color('black')
        tick.label2On = False
        tick.label2.set_color('black')
        tick.tick1line.set_color('black')
        tick.tick1line.set_markeredgewidth(1)
        tick.tick1line.set_markersize(8)

    k = 0
    for rect in rects:
        label = NY[k]
        # print(k,')',label)
        height = rect.get_height()
        ax.annotate('{0:.2f}'.format(label), xy=(rect.get_x() + rect.get_width() / 2, height))
        k = k + 1
    return R


def get_interval(teta, k, current_point, lenght, expand=True, intervals=np.array([]).reshape(-1, 2)):
    # if current_point>lenght: return None
    teta = np.abs(teta)
    k = np.abs(k)
    a = current_point - k * teta
    b = current_point + k * teta
    if (a < 0) & (b > lenght):
        a = 0
        b = lenght
    if expand:
        if (a < 0) & (b <= lenght):
            b = b - a
            a = 0
            if b > lenght:
                b = lenght
        if (a >= 0) & (b > lenght):
            a = a - (b - lenght)
            b = lenght
            if a < 0:
                a = 0
    else:
        if (a < 0) & (b <= lenght):
            a = 0
            b = b
        if (a >= 0) & (b > lenght):
            a = a
            b = lenght
    # print(a,' ',b)
    if intervals.shape[0] > 0:
        for i in np.arange(intervals.shape[0]):
            x = intervals[i, 0]
            y = intervals[i, 1]
            # print(x)
            # print(y)
            mask1 = x <= a <= y
            mask2 = x <= b <= y
            if mask1 & mask2:
                a = current_point
                b = a
                return a, b
            if mask1:
                a = y
            if mask2:
                b = x

    return a, b


def in_interval(x, a=0, b=1):
    if (x >= a) & (x <= b):
        return True
    else:
        return False


def fill_intervals_v1(group, index, k, teta):
    group = group.sort_values(by=['Дата аварии'])
    current_ad = group.loc[index, 'Дата аварии']

    # accidents=group['Дата аварии']
    # print(accidents.shape)
    # previous_accidents=group[group['Дата аварии']<current_ad]
    # future_accidents=group[group['Дата аварии']>current_ad]
    group['belong'] = True
    lenght = group.loc[index, 'L. М']
    current_point = group.loc[index, 'Адрес от начала участка, м']

    columns = list(group.columns)
    columns.append('interval')
    columns.append('interval_ads')
    columns.append('future_ads')

    columns.append('index')
    df = pd.DataFrame(index=np.arange(0, k), columns=columns)

    count = 1
    future = 0
    for ind in np.arange(0, k):
        df.loc[ind] = group.loc[index]
        df.loc[ind, 'interval'] = (ind + 1) * teta
        a, b = get_interval(teta, (ind + 1), current_point, lenght)
        # print('count=', count, ' future=',future)
        print('a=', a, ',b=', b)
        indexes = group[group['belong'] == True].index
        for i in indexes:
            other = group.loc[i, 'Адрес от начала участка, м']
            # print('point= ',other ,'index= ',i)
            if in_interval(other, a=a, b=b):
                group.loc[i, 'belong'] = False
                date = group.loc[i, 'Дата аварии']
                if date < current_ad:
                    count = count + 1
                if date > current_ad:
                    future = future + 1
                # indexes.remove(i)
                # print('belong', other)
                # print('removed', i)
        df.loc[ind, 'interval_ads'] = count
        df.loc[ind, 'future_ads'] = future
        df.loc[ind, 'index'] = index
    return df


def fill_intervals_v2(group, index, k, teta, date=1):
    group = group.sort_values(by=['Дата аварии'])
    current_ad = group.loc[index, 'Дата аварии']
    current_R = group.loc[index, 'R']
    group['r'] = group['Наработка до отказа, лет'] / group['S, мм']

    current_r = group.loc[index, 'r']
    # start=group.loc[index,'Дата ввода']
    # accidents=group['Дата аварии']
    # print(accidents.shape)
    previous_ads = group[group['Дата аварии'] < current_ad]
    future_ads = group[group['Дата аварии'] > current_ad]
    future_ads['delta'] = (future_ads['Дата аварии'] - current_ad) / np.timedelta64(1, 'Y')
    future_ads = future_ads[future_ads['delta'] <= date]

    previous_ads_r = group[group['r'] < current_r]
    future_ads_r = group[group['r'] > current_r]
    future_ads_r1 = future_ads_r[future_ads_r['r'] <= (current_R + 0.125) + (date - 1) * 0.125]
    future_ads_r2 = future_ads_r[future_ads_r['r'] <= current_r + (date / future_ads_r['S, мм'])]
    # group['belong']=True
    lenght = group.loc[index, 'L. М']
    field = 'Адрес от начала участка, м'
    current_point = group.loc[index, field]

    columns = list(group.columns)
    columns.append('interval')
    columns.append('interval_ads')
    columns.append('future_ads')
    columns.append('index')
    columns.append('last_ads')
    columns.append('first_ads')
    columns.append('interval_ads_r')
    columns.append('future_ads_r1')
    columns.append('future_ads_r')
    # columns.append('period')
    df = pd.DataFrame(index=np.arange(0, k), columns=columns)
    # print(date)

    for ind in np.arange(0, k):
        df.loc[ind] = group.loc[index]
        df.loc[ind, 'interval'] = (ind + 1) * teta
        a, b = get_interval(teta, (ind + 1), current_point, lenght)
        local_prev = previous_ads[(previous_ads[field] >= a) & (previous_ads[field] <= b)]
        local_prev_r = previous_ads_r[(previous_ads_r[field] >= a) & (previous_ads_r[field] <= b)]
        df.loc[ind, 'interval_ads'] = local_prev.shape[0] + 1
        df.loc[ind, 'interval_ads_r'] = local_prev_r.shape[0] + 1
        if local_prev.shape[0] == 0:
            df.loc[ind, 'last_ads'] = current_ad
            df.loc[ind, 'first_ads'] = current_ad
        else:
            df.loc[ind, 'last_ads'] = local_prev.loc[local_prev.iloc[-1].name, 'Дата аварии']
            df.loc[ind, 'first_ads'] = local_prev.loc[local_prev.iloc[0].name, 'Дата аварии']

        local_fut = future_ads[(future_ads[field] >= a) & (future_ads[field] <= b)]
        local_fut_r1 = future_ads_r1[(future_ads_r1[field] >= a) & (future_ads_r1[field] <= b)]
        local_fut_r2 = future_ads_r2[(future_ads_r2[field] >= a) & (future_ads_r2[field] <= b)]
        df.loc[ind, 'future_ads'] = local_fut.shape[0]
        df.loc[ind, 'future_ads_r'] = local_fut_r1.shape[0]
        df.loc[ind, 'future_ads_r1'] = local_fut_r2.shape[0]
        df.loc[ind, 'index'] = index

        # print('a=',a,',b=',b)

    # df['period']=np.nan
    df['period'] = date
    return df


def fill_intervals(group, index, k, teta, date=1, expand=True, repairs=None, intervals=[]):
    group = group.sort_values(by=['Дата аварии'])
    current_ad = group.loc[index, 'Дата аварии']
    current_year = pd.to_datetime(current_ad).year
    next_year = pd.to_datetime('01-01-' + str(current_year + date))
    beginofnext = pd.to_datetime('01-01-' + str(current_year + 1))
    current_delta = (beginofnext - current_ad) / np.timedelta64(1, 'D')
    getin = group.loc[index, 'Дата ввода']
    dateout = group.loc[index, 'Дата перевода в бездействие']
    current_R = group.loc[index, 'R']
    current_s = group.loc[index, 'S']
    left = group.loc[index, 'a']
    # right=group.loc[index,'b']
    current_age = group.loc[index, 'Наработка до отказа(new), лет']
    getout = (dateout - getin) / np.timedelta64(1, 'Y')
    getout_r = getout / current_s
    group['r'] = group['Наработка до отказа(new), лет'] / group['S']
    current_r = group.loc[index, 'r']
    columns = list(group.columns)
    columns.append('interval')
    columns.append('interval_ads')
    columns.append('int_ads_halfyear')
    columns.append('int_ads_oneyear')
    columns.append('int_ads_twoyear')
    columns.append('int_ads_threeyear')
    columns.append('future_ads')
    columns.append('index')
    columns.append('last_ads')
    columns.append('first_ads')
    columns.append('interval_ads_r')
    columns.append('future_ads_r1')
    columns.append('future_ads_r')
    columns.append('extra_lenght')
    columns.append('lbound')
    columns.append('rbound')
    columns.append('future_calendar')
    columns.append('delta_days')

    # columns.append('ads_r_indexes')
    # columns.append('ads_r1_indexes')
    # columns.append('ads_indexes')
    previous_ads = group[group['Дата аварии'] <= current_ad]
    previous_ads_r = group[group['r'] <= current_r]
    future_ads_r = group[group['r'] > current_r]
    future_ads = group[group['Дата аварии'] > current_ad]
    # print(current_ad)
    if dateout >= next_year:
        # future_ads['delta']=(future_ads['Дата аварии']-current_ad)/np.timedelta64(1,'Y')
        calendar_year = future_ads[future_ads['Дата аварии'] < next_year]
        # print('getout=',getout )
        # print('current+date=',(current_age+date) )
        # print('future_ads shape=', future_ads.shape)
    else:
        calendar_year = None
    if getout >= (current_age + date):
        future_ads['delta'] = (future_ads['Дата аварии'] - current_ad) / np.timedelta64(1, 'Y')
        future_ads = future_ads[future_ads['delta'] <= date]
        # print('getout=',getout )
        # print('current+date=',(current_age+date) )
        # print('future_ads shape=', future_ads.shape)
    else:
        future_ads = None

    if getout_r >= (current_R + date * 0.125):
        future_ads_r1 = future_ads_r[future_ads_r['r'] <= (current_R + date * 0.125)]
    else:
        future_ads_r1 = None
    if getout_r >= current_r + date / current_s:
        future_ads_r2 = future_ads_r[future_ads_r['r'] <= current_r + date * 0.125]
    else:
        future_ads_r2 = None
    # group['belong']=True
    lenght = group.loc[index, 'L,м']
    field = 'Адрес от начала участка (new)'
    current_point = group.loc[index, field]
    if not expand:
        l = int(lenght / (2 * teta))
        # k=min(k,l)
    if l == 0: return None

    # columns.append('period')
    df = pd.DataFrame(index=np.arange(0, k), columns=columns)
    dfindex = pd.DataFrame(index=np.arange(0, k),
                           columns=['ads_r_indexes', 'ads_r1_indexes', 'ads_indexes', 'calendar_indexes'])
    df[group.columns] = group.loc[index, group.columns].values.reshape(-1, group.shape[1])
    df['extra_lenght'] = 0.0
    # display(df['extra_lenght'])
    # print(df.shape,'; ',index,'; ',k)

    for ind in np.arange(0, k):
        # df.loc[ind]=group.loc[index]
        df.loc[ind, 'interval'] = (ind + 1) * teta
        a, b = get_interval(teta, (ind + 1), current_point, lenght, expand=expand, intervals=intervals)
        df.loc[ind, 'lbound'] = a
        df.loc[ind, 'rbound'] = b
        # print('repairs shape',repairs.shape)
        if repairs is not None:
            if repairs.shape[0] > 0:
                # print(repairs)
                # print(a,'',b)
                # print('current', current_point)
                extra_lenght = GetExtraLenght(repairs=repairs, current_point=current_point, date=current_ad, a=a, b=b,
                                              tilde=left)
                # print('extra_lenght=',extra_lenght)
                df.loc[ind, 'extra_lenght'] = extra_lenght
        # print(df.loc[ind,'extra_lenght'])
        local_prev = previous_ads[(previous_ads[field] >= a) & (previous_ads[field] <= b)]
        local_prev_r = previous_ads_r[(previous_ads_r[field] >= a) & (previous_ads_r[field] <= b)]
        df.loc[ind, 'interval_ads'] = local_prev.shape[0]
        df.loc[ind, 'interval_ads_r'] = local_prev_r.shape[0]
        halfy = current_ad - pd.DateOffset(months=6)
        oney = current_ad - pd.DateOffset(years=1)
        twoy = current_ad - pd.DateOffset(years=2)
        threey = current_ad - pd.DateOffset(years=3)
        local_prev_05 = local_prev[local_prev['Дата аварии'] >= halfy]
        local_prev_1 = local_prev[local_prev['Дата аварии'] >= oney]
        local_prev_2 = local_prev[local_prev['Дата аварии'] >= twoy]
        local_prev_3 = local_prev[local_prev['Дата аварии'] >= threey]
        df.loc[ind, 'int_ads_halfyear'] = local_prev_05.shape[0]
        df.loc[ind, 'int_ads_oneyear'] = local_prev_1.shape[0]
        df.loc[ind, 'int_ads_twoyear'] = local_prev_2.shape[0]
        df.loc[ind, 'int_ads_threeyear'] = local_prev_3.shape[0]
        local_prev_less = local_prev[local_prev['Дата аварии'] < current_ad]
        if local_prev_less.shape[0] == 0:
            df.loc[ind, 'last_ads'] = current_ad
            df.loc[ind, 'first_ads'] = current_ad
        else:
            df.loc[ind, 'last_ads'] = local_prev_less.loc[local_prev_less.iloc[-1].name, 'Дата аварии']
            df.loc[ind, 'first_ads'] = local_prev_less.loc[local_prev_less.iloc[0].name, 'Дата аварии']

        if future_ads is not None:
            local_fut = future_ads[(future_ads[field] >= a) & (future_ads[field] <= b)]
            df.loc[ind, 'future_ads'] = local_fut.shape[0]
            dfindex.loc[ind, 'ads_indexes'] = local_fut.index
            # print('future_ads is not none. shape=',local_fut.shape[0])
        else:
            df.loc[ind, 'future_ads'] = np.nan
        if future_ads_r1 is not None:
            local_fut_r1 = future_ads_r1[(future_ads_r1[field] >= a) & (future_ads_r1[field] <= b)]
            df.loc[ind, 'future_ads_r'] = local_fut_r1.shape[0]
            dfindex.loc[ind, 'ads_r_indexes'] = local_fut_r1.index
        else:
            df.loc[ind, 'future_ads_r'] = np.nan
        if future_ads_r2 is not None:
            local_fut_r2 = future_ads_r2[(future_ads_r2[field] >= a) & (future_ads_r2[field] <= b)]
            df.loc[ind, 'future_ads_r1'] = local_fut_r2.shape[0]
            dfindex.loc[ind, 'ads_r1_indexes'] = local_fut_r2.index
        else:
            df.loc[ind, 'future_ads_r1'] = np.nan
        if calendar_year is not None:
            local_year = calendar_year[(calendar_year[field] >= a) & (calendar_year[field] <= b)]
            df.loc[ind, 'future_calendar'] = local_year.shape[0]
            dfindex.loc[ind, 'calendar_indexes'] = local_year.index
        else:
            df.loc[ind, 'future_calendar'] = np.nan
        df.loc[ind, 'delta_days'] = current_delta
        # df.loc[ind,'future_ads']=local_fut.shape[0]
        # df.loc[ind,'future_ads_r']=local_fut_r1.shape[0]
        # df.loc[ind,'future_ads_r1']=local_fut_r2.shape[0]
        df.loc[ind, 'index'] = index

        # print('a=',a,',b=',b)

    # df['period']=np.nan
    df['period'] = date
    df = df.join(dfindex, lsuffix='_caller', rsuffix='_other')
    # display(df['extra_lenght'])
    return df


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


def get_merged_sets(x=np.array([])):
    disjoint=get_disjoint_sets(x,shape=2)
    si=np.argsort(disjoint[:,0])
    disjoint=disjoint[si]
    merged=[]
    k=0
    a=disjoint[0]
    while k<disjoint.shape[0]-1:
        b=disjoint[k+1]
        if a[1]==b[0]:
            a=merge(a,b)
        else:
            merged.append(a)
            a=b
        k+=1
    merged.append(a)
    return np.array(merged)

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


def GetSetsResidual(L, X, f=residual,shape=3):
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
        x_=GetSetsResidual(b,a,shape=shape)[:-1]
        y=get_disjoint_sets(x_,shape=shape)
        return np.vstack((a,y))
    else:
        return x

def get_horizontal_counts(data=np.array([]),interval=100,L=100):
    mask=np.ones(data.shape[0],dtype=bool)
    intervals=[]
    i=0
    while mask.shape[0]>0:
        y=data[-1]
        a=y-interval
        b=y+interval
        if a<0:
            a=0
        if b>L:
            b=L
        res=np.array([a,b])
        if i==0:
            intervals.append((0,0,0))
            i=i+1
        for ivl in intervals:
            if res.shape[0]>0:
                res=residual(res,ivl,shape=2).reshape(-1)
        if res.shape[0]>0:
            submask=(data>=res[0])&(data<=res[1])
            res=np.append(res,submask[submask==True].shape[0])
            intervals.append(res)
            data=data[~submask]
            mask=mask[~submask]
        else:
            mask[-1]=False
            data=data[mask]
            mask=mask[mask]
    return np.array(intervals[1:])


def wtp_approach(data, mask, xfield='Наработка до отказа', yfield='Обводненность'):
    #mask =data['Обводненность']==0 & data['ID простого участка']==ID
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
                x = dict({'x1': x1, 'x2': x2})
                d = dict({'d1': q, 'd2': p})
                line = linear_transform(x=x, d=d)
                if q > 0:
                    # data.loc[indices,'Обводненность']=data.loc[[a,b],'Обводненность'].mean()
                    data.loc[indices, yfield] = data.loc[indices, xfield].apply(lambda x: line.value(x))
                else:
                    data.loc[indices, yfield] = data.loc[b, yfield]
        i = i + 1
def GetRepairsMap(REP, k):
    if k == 0:
        X = REP[k, :].reshape(-1, 3)
        return X
    else:
        X = REP[k, :].reshape(3)
        L = GetRepairsMap(REP, k - 1)
        Y = GetSetsResidual(L, X)
        return Y


def GetEmptyRepairs(data, ID):
    empty = pd.DataFrame([], columns=['ID', 'Дата ремонта', 'a', 'b'])
    group = data[data['ID простого участка'] == ID]
    repairs = GetUnicalRepairs(group, scale=2)
    repairs['b'] = repairs['Адрес'] + repairs['Длина']
    rep = repairs[['Адрес', 'b', 'Дата ремонта']].values
    for i in np.arange(rep.shape[0]):
        X = rep[i, :]
        a = rep[i, 0]
        b = rep[i, 1]
        rd = rep[i, 2]
        # print('i ',i,'a ',a,'b ',b,'date', rd)
        mask1 = (group['Дата аварии'] <= rd) & (
                (group['Адрес от начала участка'] <= b) & (group['Адрес от начала участка'] >= a))
        repgroup = group[mask1]
        # print(repgroup.shape)
        mask = mask1
        mask2 = mask
        if i > 0:
            A = GetRepairsMap(rep, i - 1)
            T = GetSetsResidual(A, X.reshape(3), f=interseption)[:-1]
            # mask=group['ID простого участка']!=group['ID простого участка']
            # mask2=mask
            for t in T:
                # submask2=(repgroup['Адрес от начала участка']<=t[1])&(repgroup['Адрес от начала участка']>=t[0])
                submask = (repgroup['Дата аварии'] < t[2]) & ((repgroup['Адрес от начала участка'] <= t[1]) & (
                            repgroup['Адрес от начала участка'] >= t[0]))
                # print('i ',i,'x ',t[0],'y ',t[1],'date', t[2])
                # print(submask)
                mask = mask & ~submask
                # mask2=mask2|submask2
        subgroup = repgroup[mask]
        # subgroup1=repgroup[~mask2]
        # if (subgroup.shape[0]>0):
        # print('i ',i,'x ',a,'y ',b,'date', rd)
        # display(subgroup1[['Дата аварии' ,'Адрес от начала участка','Дата окончания ремонта',	'Адрес от начала участка.1','Длина ремонтируемого участка']])
        # display(subgroup[['Дата ввода','Дата аварии' ,'Адрес от начала участка','Дата окончания ремонта',	'Адрес от начала участка.1','Длина ремонтируемого участка']])
        # if subgro
        # print('mask ',len(mask[mask==True]))
        # print('mask2',len(mask2[mask2==False]))
        # print(mask2)
        if len(mask[mask == True]) == 0:
            subempty = pd.DataFrame([], columns=['ID', 'Дата ремонта', 'a', 'b'])
            subempty.loc[0, 'ID'] = ID
            subempty.loc[0, 'Дата ремонта'] = rd
            subempty.loc[0, 'a'] = a
            subempty.loc[0, 'b'] = b
            empty = empty.append(subempty, ignore_index=True)
        else:
            # mask3=mask|mask2
            # print(len(mask3))
            # print(group.shape)
            group = group[~mask]
    return empty


def GetSplitedByRepairs(data, ID):
    group = data[data['ID простого участка'] == ID]
    repairs = GetUnicalRepairs(group, scale=2)
    repairs['b'] = repairs['Адрес'] + repairs['Длина']
    rep = repairs[['Адрес', 'b', 'Дата ремонта']].values
    # group['state']=True
    for i in np.arange(rep.shape[0]):
        X = rep[i, :]
        a = rep[i, 0]
        b = rep[i, 1]
        rd = rep[i, 2]
        # print('i ',i,'a ',a,'b ',b,'date', rd)
        mask = (group['Дата аварии'] <= rd) & (
                (group['Адрес от начала участка'] <= b) & (group['Адрес от начала участка'] >= a))
        mask1 = (group['Дата аварии'] > rd) & (
                (group['Адрес от начала участка'] <= b) & (group['Адрес от начала участка'] >= a))
        indexes = mask1[mask1 == True].keys()
        data.loc[indexes, 'Дата ввода'] = rd
        data.loc[indexes, 'a'] = a
        data.loc[indexes, 'b'] = b
        data.loc[indexes, 'new_id'] = str(ID) + '_' + str(i + 1)
        # data.loc[mask[mask==True].keys(),'Дата перевода в бездействие']=rd
        # print(indexes)

        # repgroup.loc[mask==True,'Дата ввода']=t[2]
        data.loc[mask[mask == True].keys(), 'Дата перевода в бездействие'] = rd
        repgroup = group[mask]
        # print(repgroup.shape)
        # mask=mask1
        # mask2=mask
        if i > 0:
            A = GetRepairsMap(rep, i - 1)
            T = GetSetsResidual(A, X.reshape(3), f=interseption)[:-1]
            # mask=group['ID простого участка']!=group['ID простого участка']
            # print('group mask ',mask)
            for t in T:
                # submask2=(repgroup['Адрес от начала участка']<=t[1])&(repgroup['Адрес от начала участка']>=t[0])
                submask = (repgroup['Дата аварии'] >= t[2]) & ((repgroup['Адрес от начала участка'] <= t[1]) & (
                            repgroup['Адрес от начала участка'] >= t[0]))
                # print('i ',i,'x ',t[0],'y ',t[1],'date', t[2])
                # print('submask ',submask)

                mask = mask | submask
                # print('mask ',mask)
                # ['original_index', 'new_id', 'a', 'b',
                # 'Дата ремонта', 'Наработка до отказа(new), лет', 'R', 'L,м',
                # 'Адрес от начала участка (new)']
                # print('дата ввода',t[2], 'get out', rd, ' аварий ', len(mask[mask==True]))
                indexes1 = submask[submask == True].keys()
                # print(submask)
                data.loc[indexes1, 'Дата ввода'] = t[2]
                data.loc[indexes1, 'Дата перевода в бездействие'] = rd
                # repgroup.loc[submask==True,'Дата ввода']=t[2]
                # repgroup.loc[submask==True,'Дата перевода в бездействие']=rd
                # mask2=mask2|submask2
        subgroup = repgroup[mask]
        # subgroup1=repgroup[~mask2]
        # if (subgroup.shape[0]>0):
        # print('i ',i,'x ',a,'y ',b,'date', rd)
        # display(subgroup1[['Дата аварии' ,'Адрес от начала участка','Дата окончания ремонта',	'Адрес от начала участка.1','Длина ремонтируемого участка']])
        # display(subgroup[['Дата ввода','Дата аварии' ,'Дата перевода в бездействие',
        # 'Адрес от начала участка','original_index', 'new_id', 'a', 'b',
        # 'Дата ремонта', 'Наработка до отказа(new), лет', 'R', 'L,м',
        # 'Адрес от начала участка (new)']])
        group = group[~mask]
    return data


def GetLengthErrors(data, ID):
    group = data[data['ID простого участка'] == ID]
    empty = pd.DataFrame([], columns=['index', 'ID', 'Дата аварии', 'Адрес от начала участка', 'L'])
    length = group['L'].value_counts().keys()[0]
    maxlength = group['Адрес от начала участка'].max()

    if maxlength > length:
        mask = group['Адрес от начала участка'] > length
        errors = group[mask]
        subempty = pd.DataFrame(index=np.arange(errors.shape[0]),
                                columns=['index', 'ID', 'Дата аварии', 'Адрес от начала участка', 'L'])
        k = 0
        for l in errors.index:
            subempty.loc[k, 'ID'] = errors.loc[l, 'ID простого участка']
            subempty.loc[k, 'index'] = l
            subempty.loc[k, 'Адрес от начала участка'] = errors.loc[l, 'Адрес от начала участка']
            subempty.loc[k, 'L'] = length
            subempty.loc[k, 'Дата аварии'] = errors.loc[l, 'Дата аварии']
            k = k + 1
        empty = empty.append(subempty, ignore_index=True)
    return empty


def GetInterval(teta, k, current_point, lenght, expand=True):
    teta = np.abs(teta)
    k = np.abs(k)
    a = current_point - k * teta
    b = current_point + k * teta

    if (a < 0) & (b > lenght):
        a = 0
        b = lenght
    if expand:
        if (a < 0) & (b <= lenght):
            b = b - a
            a = 0
            if b > lenght:
                b = lenght
        if (a >= 0) & (b > lenght):
            a = a - (b - lenght)
            b = lenght
            if a < 0:
                a = 0
    else:
        if (a < 0) & (b <= lenght):
            a = 0
            b = b
        if (a >= 0) & (b > lenght):
            a = a
            b = lenght

    return a, b


def GetExtraLenght(repairs, current_point, date, a, b, tilde):
    mask = repairs[:, 2] <= date
    reps = repairs[mask]
    # print(a,' ',b)
    if reps.shape[0] == 0:
        Z = np.array([])
    else:
        Z = GetRepairsMap(reps, reps.shape[0] - 1)
        Z[:, [0, 1]] = Z[:, [0, 1]] - tilde

    x = np.array((a, b, date))
    Y = GetSetsResidual(Z, x, f=interseption)[:-1]
    # print(Z)
    # print('interception ', Y)
    return (Y[:, 1] - Y[:, 0]).sum()


def RepairsValues(repairs):
    repairs['b'] = repairs['Адрес'] + repairs['Длина']
    return repairs[['Адрес', 'b', 'Дата ремонта']].values



def split_intervals(group, ints=[100], date=[1], expand=True, repairs=None,function=fill_intervals):
    k = 0
    Data=None
    for index in group.index:

        for d in date:
            # print(repairs.shape)
            df = function(group, index, ints, d, expand, repairs=repairs)
            if df is not None:
                if k == 0:
                   Data = df.copy()
                   k = k + 1
                else:
                   Data = Data.append(df, ignore_index=True)
            # print(df.shape, 'index=',index)

    return Data

def split_separated_intervals(someid, n, teta, date=[1], split_by=np.datetime64('2016-01-01'), offset=pd.DateOffset(years=2),
                    expand=True, repairs=None):
    k = 0
    intervals = np.array([]).reshape(-1, 2)
    # offset=pd.DateOffset(years=1)
    # print(offset)
    split_by = pd.to_datetime(split_by)
    # print(split_by)
    # print(split_by-offset)
    if offset is not None:
        group = someid[(someid['Дата аварии'] < split_by) & (someid['Дата аварии'] >= split_by - offset)]
    else:
        group = someid[someid['Дата аварии'] < split_by]
    if group.shape[0] == 0: return None
    group['state'] = False
    group['color'] = True
    mask = group['color'] == True
    # print(group.shape)
    lenght = group['L,м'].value_counts().keys()[0]
    # if int(lenght/(teta*2))==0: return None
    if lenght < teta * 2: return None
    Data = None
    while mask[mask == True].shape[0] > 0:
        subgroup = group[mask]
        mdate = subgroup['Дата аварии'].max()
        ads = subgroup[subgroup['Дата аварии'] == mdate]
        i = ads.index[0]
        group.loc[i, 'state'] = True

        for d in date:
            off = int((split_by - mdate) / np.timedelta64(1, 'Y')) + d
            # print('period ',d)
            df = fill_intervals(someid, i, n, teta, off, expand, repairs=repairs, intervals=intervals)
            # print('df',df.shape)
            # print(df['new_id'].value_counts().keys()[0])
            # if df.shape[0]==0:
            # print('ID',someid['new_id'].value_counts().keys()[0], 'L ',lenght,'interval ',teta*2, 'index ',i)
            if k == 0:
                Data = df.copy()
                k = k + 1
            else:
                Data = Data.append(df, ignore_index=True)
                # k=k+1
        # print('DATA ',Data.shape)
        lbound = df.loc[0, 'lbound']
        rbound = df.loc[0, 'rbound']
        newi = np.array([lbound, rbound]).reshape(-1, 2)
        intervals = np.vstack((intervals, newi))
        submask = ((subgroup['Адрес от начала участка (new)'] >= lbound) & (
                subgroup['Адрес от начала участка (new)'] <= rbound)) & (subgroup['Дата аварии'] <= mdate)
        if len(submask[submask == True].keys()) == 0:
            print('an empty mask. ID', someid['new_id'].value_counts().keys()[0])
            return None
        group.loc[group.index.isin(submask[submask == True].keys()), 'color'] = False

        mask = group['color'] == True
    # if Data is None:
    # print('ID',someid['new_id'].value_counts().keys()[0], 'L ',lenght,'interval ',teta*2)
    return Data


def make_separated_intervals(data, n, teta, date=[1], expand=True, split_by=np.datetime64('2016-01-01'),
                             offset=pd.DateOffset(years=2), ident='ID простого участка',
                             sharedid='ID простого участка', scale=2):
    # maingroup=data[data[ident]==idt]
    # data=data[data['Состояние']=='действующий']
    ID = data[sharedid].value_counts()
    k = 0
    Data = None
    for idt in ID.keys():
        group = data[data[sharedid] == idt]
        repairs = GetUnicalRepairs(group, scale=scale)
        reps = RepairsValues(repairs)
        # group=group[group['Дата перевода в бездействие']==get_out]
        # print(group.shape)
        for subid in group[ident].value_counts().keys():
            subgroup = group[group[ident] == subid]
            # print(subgroup.shape)
            maxads = subgroup['Дата аварии'].max()
            minads = subgroup['Дата аварии'].min()
            # mask=(split_by<maxads)&(split_by>minads)
            mask = True
            if mask:
                getin = subgroup['Дата ввода'].value_counts().keys()[0]
                a = subgroup['a'].value_counts().keys()[0]
                b = subgroup['b'].value_counts().keys()[0]
                subrep = reps[np.where((reps[:, 2] > getin) & ((reps[:, 0] >= a) & (reps[:, 1] <= b)))]
                # print(subrep)
                df = split_separated_intervals(subgroup, n, teta, date, split_by, offset, expand, repairs=subrep)
                if df is not None:
                    if k == 0:
                        Data = df.copy()
                        k = k + 1
                    else:
                        Data = Data.append(df, ignore_index=True)
                # else: print('returned an empty df',subid)

    return Data


def make_intervals_v1(data, n, teta, date=[1]):
    ID = data['ID простого участка'].value_counts()
    k = 0
    for id in ID.keys():
        group = data[data['ID простого участка'] == id]
        df = split_intervals(group, n, teta, date)
        if k == 0:
            Data = df.copy()
        else:
            Data = Data.append(df, ignore_index=True)
        k = k + 1
    return Data


def join_hists(TRAIN, TEST, field):
    plt.figure(figsize=(12, 10))
    plt.title('Гистограммы прогнозируемых величин')
    n, bins, patche = plt.hist(TRAIN[field], bins=50, alpha=0.5, rwidth=0.85, label='TRAIN', color='blue')
    plt.hist(TEST[field], bins=50, alpha=0.5, rwidth=0.85, label='TEST', color='red')
    plt.legend(loc='upper right')
    plt.title('Совмещенная гистограмма TRAIN-TEST:' + field)
    plt.xlabel(field, fontsize=14)
    plt.ylabel('Частота', fontsize=14)
    majorFormatter = FormatStrFormatter('%.2f')
    minorFormatter = FormatStrFormatter('%.2f')
    minorLocator = AutoMinorLocator(n=4)


def load(path="D:\\ml\\отказы.csv"):
    data = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=True)
    data['Дата ввода'] = pd.to_datetime(data['Дата ввода'])
    data['Дата аварии'] = pd.to_datetime(data['Дата аварии'])
    return data


def GetMeans(data, field="", target='future_ads'):
    g = []
    a = []
    if field == '':
        g.append(data[target].mean())
    else:
        for age in data[field].value_counts().keys():
            T = data[data[field] == age]
            mean = T[target].mean()
            a.append(age)
            g.append(mean)
    return a, g


def GetMedians(data, field="", target='future_ads'):
    g = []
    a = []
    if field == '':
        g.append(data[target].median())
    else:
        for age in data[field].value_counts().keys():
            T = data[data[field] == age]
            mean = T[target].median()
            a.append(age)
            g.append(mean)
    return a, g


def residual_color(residual):
    color = []
    for r in residual:
        if r > 0: color.append('orange')
        if r < 0: color.append('darkblue')
        if r == 0: color.append('green')
    return color


def GetResidual(train, test, field='strates', target='future_ads'):
    mean = []
    median = []
    a = []
    if field == '':
        mean.append(train[target].mean() - test[target].mean())
        median.append(train[target].median() - test[target].median())
    else:
        for age in train[field].value_counts().keys():
            T = train[train[field] == age]
            Tt = test[test[field] == age]
            if Tt.shape[0] > 0:
                median1 = T[target].median()
                median2 = Tt[target].median()
                mean1 = T[target].mean()
                mean2 = Tt[target].mean()
                a.append(age)
            mean.append(mean1 - mean2)
            median.append(median1 - median2)
    return a, mean, median


def MakeDistributions(TRAIN, TEST, exp=False, mode='residual', target='future_ads', field='R', printing=True,
                      title='среднее значение отказов в будущем на общей статистике'):
    if exp:
        TRAIN['ads'] = np.exp(TRAIN[target]) - 1
        TEST['ads'] = np.exp(TEST[target]) - 1
    else:
        TRAIN['ads'] = TRAIN[target]
        TEST['ads'] = TEST[target]

    a, b = GetMeans(TEST, field, target='ads')
    c, d = GetMeans(TRAIN, field, target='ads')
    age, mean, median = GetResidual(TRAIN, TEST, field=field, target='ads')
    if (printing) & (field != ''):
        plt.figure(figsize=(12, 10))
        plt.title(title)
        scatter1 = plt.scatter(a, b, marker='v', c='red', s=50)
        scatter2 = plt.scatter(c, d, marker='^', c='blue', s=50)
        plt.xlabel(field)
        plt.ylabel('future accidents (mean)')
        plt.legend((scatter1, scatter2), ('TEST', 'TRAIN'), loc='upper right')
        plt.show()

        plt.figure(figsize=(12, 10))
        plt.title('Разность среднего между тренировочным и тестовым наборами')
        color = residual_color(mean)
        scatter1 = plt.scatter(age, mean, marker='o', c=color)
        plt.xlabel(field)
        plt.ylabel('mean residual')
        # plt.legend((scatter1),('TEST','TRAIN'),loc='upper right')
        plt.show()
        plt.figure(figsize=(12, 8))
        plt.title('Разность медиан  между тренировочным и тестовым наборами')
        color = residual_color(median)
        scatter2 = plt.scatter(age, median, marker='o', c=color)
        plt.xlabel(field)
        plt.ylabel('median residual')
        # plt.legend((scatter1,scatter2),('TEST','TRAIN'),loc='upper right')
        plt.show()
        plt.figure(figsize=(12, 16))
        plt.subplot(2, 1, 1)
        # plt.figure(figsize=(12,10))
        plt.title('График разности среднего значения между тренировочным и тестовым наборами')
        # color=residual_color(mean)
        zeros = [0 for x in age]
        data = np.array([age, mean])
        transposed = data.transpose()
        dsorted = transposed[np.argsort(transposed[:, 0])]
        plt.plot(dsorted[:, 0], dsorted[:, 1])
        plt.plot(dsorted[:, 0], zeros, c='black', alpha=0.8)
        plt.xlabel(field)
        plt.ylabel('mean residual')
        # plt.grid(True)
        plt.subplot(2, 1, 2)
        # plt.figure(figsize=(12,10))
        plt.title('График разности медианного значения между тренировочным и тестовым наборами')
        # color=residual_color(mean)
        data = np.array([age, median])
        transposed = data.transpose()
        dsorted = transposed[np.argsort(transposed[:, 0])]
        plt.plot(dsorted[:, 0], dsorted[:, 1])
        plt.plot(dsorted[:, 0], zeros, c='black', alpha=0.8)
        plt.xlabel(field)
        plt.ylabel('mean residual')
        # plt.grid(True)
        # plt.legend((scatter1),('TEST','TRAIN'),loc='upper right')
        plt.show()
    if mode == 'residual':
        return np.array(mean)
    if mode == 'mean':
        a, b = GetMeans(TEST, field, target='ads')
        return np.array(b)
    if mode == 'median':
        a, b = GetMedians(TEST, field, target='ads')
        return np.array(b)


def CutOff(data):
    cdata = data.copy()
    for period in data['period'].value_counts().keys():
        # print(period)
        maxdate = data[data['period'] == period]['Дата аварии'].max()  # print('maxdate ',maxdate)
        date_offset = pd.DateOffset(years=period)
        end = maxdate - date_offset
        cdata = cdata.drop(np.where((data['Дата аварии'] > end) & (data['period'] == period))[0])
    return cdata


def RoundToInt(x, comma=0.5):
    y = np.modf(x)
    a = y[0]
    b = y[1]
    if a < comma:
        a = 0
    else:
        a = 1
    return b + a


def GetSplitData(data, target='future accidents', mode='residual', field='R', date_offset=3, end=False, printing=False):
    dmax = data['Дата аварии'].max()
    dmin = data['Дата аварии'].min()
    dsplit = dmax - pd.DateOffset(years=date_offset)
    dleft = dsplit - pd.DateOffset(years=date_offset)
    dates = []
    dic = []
    d = []
    d.append(dleft)
    d.append(dsplit)
    d.append(dmax)
    # dates.append(d)
    # print('min',dmin)
    k = 1

    while dmin <= dleft:
        # print(k,') left ',dleft,', split ',dsplit, ',right ',dmax)
        TRAIN, TEST = split_by_date(data, date=dsplit)
        if end == False:
            TRAIN = TRAIN[TRAIN['Дата аварии'] >= dleft]

        # TRAIN=TRAIN[TRAIN['Дата аварии']>=dleft]
        TEST = TEST[TEST['Дата аварии'] <= dmax]
        # print('split ', dsplit)
        # print('TRAIN=',TRAIN.shape)
        # print('TEST=',TEST.shape)
        residual = MakeDistributions(TRAIN, TEST, mode=mode, target=target, field=field, printing=printing)
        mean = residual.mean()
        dic.append((mean, d))
        # print(mean)
        k = k + 1
        dmax = dsplit
        dsplit = dmax - pd.DateOffset(years=date_offset)
        dleft = dsplit - pd.DateOffset(years=date_offset)
        d = []
        d.append(dleft)
        d.append(dsplit)
        d.append(dmax)
        # dates.append(d)
    return dic


def GetMeansVars(data, target='future accidents', field='accident_count', date_offset=1):
    dmax = data['Дата аварии'].max()
    dmin = data['Дата аварии'].min()
    dleft = dmax - pd.DateOffset(years=date_offset)
    # dleft=dsplit-pd.DateOffset(years=date_offset)
    dates = []
    means = []
    var = []
    k = 1
    while dmin <= dleft:
        sdata = data[(data['Дата аварии'] >= dleft) & (data['Дата аварии'] <= dmax)]
        sdata = sdata[sdata[target] <= 10]
        sdata = sdata[sdata[field] == 2]
        dates.append(dmax)
        means.append(sdata[target].mean())
        var.append(sdata[target].var())
        dmax = dleft
        dleft = dmax - pd.DateOffset(years=date_offset)

    return dates, means, var


def GetFeatures(data, extra=pd.DataFrame(), by_date=True, dummy_target=False, date_column='Дата аварии', seed=42,
                split=np.datetime64('2016-01-01'), dummies=[],
                drop_end=False, tail=pd.DateOffset(years=1),
                split_by=pd.DateOffset(years=1), indexes='',
                drop_before=[], drop_after=[], dates=[['Дата ввода', 'start'], ['Дата аварии', 'accident'],
                                                      ['last_ads', 'last_interval'], ['first_ads', 'first_interval']],
                fields=['interval', 'interval_ads', 'interval_age', 'interval_to_ads',
                        'future_ads', 'index', 'last_ads', 'first_ads', 'period', 'R', 'future accidents',
                        'Наработка до отказа, лет', 'Адрес от начала участка, м', 'Месторождение',
                        'accident/time',
                        'age', 'first accident', 'last accident',
                        'to accident', 'accident_count',
                        'accident_count/L', 'Дата аварии', 'Дата ввода',
                        'D, мм', 'L. М', 'S, мм', 'Материал трубы',
                        'Тип трубы', 'Завод изготовитель',
                        'Обводненность, %', 'Режим течения',
                        'Скорость потока, м/с'], loglist=['future_ads_r',
                                                          'Скорость потока', 'interval_ads_r'], target='future_ads_r'):
    List = fields.copy()
    List.append(target)

    extradummy = None

    for column in drop_before:
        List.remove(column)
    data = data[List]
    if extra.shape[0] > 0:
        extra = extra[List]
        extra['sign'] = False
        data['sign'] = True
        DATA = data.append(extra)
        fulldummy = pd.get_dummies(DATA, columns=dummies)
        demodummy = fulldummy[fulldummy['sign'] == True]
        extradummy = fulldummy[fulldummy['sign'] == False]
        drop_after.append('sign')

    else:
        demodummy = pd.get_dummies(data, columns=dummies)

    # demodummy=get_dummy(data,List)

    dates_list = dates
    # print(split)
    if by_date:
        maxdate = data[date_column].max()
        # print('max '+str(maxdate))

        end = pd.to_datetime(split) + split_by
        # print('end '+str(end))
        # print('split '+str(split))
        TRAIN, TEST = split_by_date(demodummy, date=split, date_column=date_column)
        TEST = TEST[TEST[date_column] <= end]
    else:
        # TRAIN,TEST=random_split(demodummy)
        # TRAIN,TEST=split_data(demodummy,0.2)
        TRAIN, TEST = split_data_by_index(demodummy, seed=seed, index='new_id')
    end = pd.to_datetime(split) + split_by
    # print(end)

    #TEST = TEST[TEST['Наработка до отказа(new), лет'] < 30]
    if drop_end:
        TRAIN = TRAIN[TRAIN[date_column] >= pd.to_datetime(split) - tail]
    print('TEST max', TEST[date_column].max())
    print('TEST min', TEST[date_column].min())
    print('TRAIN max', TRAIN[date_column].max())
    print('TRAIN min', TRAIN[date_column].min())
    st_mindate = TRAIN['Дата ввода'].min()
    TRAIN = datetonumber(TRAIN, st_mindate, dates_list)
    TEST = datetonumber(TEST, st_mindate, dates_list)

    wmedian = TRAIN['Обводненность'].median()
    if np.isnan(wmedian): wmedian = 0
    print('Обводненность ', wmedian)
    smedian = TRAIN['Скорость потока'].median()
    if np.isnan(smedian): smedian = 0
    print('Скорость потока ', smedian)
    address = TRAIN['Адрес от начала участка (new)'].median()
    if np.isnan(address): address = 0
    TRAIN = fill_median(TRAIN, smedian, wmedian, addr=address)
    TEST = fill_median(TEST, smedian, wmedian)

    # loglist=['future_ads','accident_count',
    # 'accident_count/L','Скорость потока, м/с','interval_ads']
    TRAIN = feature_log(TRAIN, loglist)
    TEST = feature_log(TEST, loglist)
    TRAIN = drop_columns(TRAIN, drop_after)
    TEST = drop_columns(TEST, drop_after)
    if len(indexes) > 0:
        TRAIN.drop(indexes, axis=1, inplace=True)
        TEST.drop(indexes, axis=1, inplace=True)
    TRAIN = TRAIN[TRAIN[target] >= 0]
    CONTROL = TEST.loc[np.isnan(TEST[target])]
    CONTROL.drop(target, axis=1, inplace=True)
    TEST = TEST[TEST[target] >= 0]
    if dummy_target == True:
        dummy = pd.get_dummies(data[target])
        Dy = dummy.loc[dummy.index.isin(TRAIN.index)]
        Dyt = dummy.loc[dummy.index.isin(TEST.index)]
    if extradummy is not None:
        print('CONTROL max', extradummy[date_column].max())
        print('CONTROL min', extradummy[date_column].min())
        extradummy = fill_median(extradummy, smedian, wmedian)
        extradummy = feature_log(extradummy, loglist)
        extradummy = datetonumber(extradummy, st_mindate, dates_list)
        extradummy = drop_columns(extradummy, drop_after)
        drop_after.remove('sign')

        if len(indexes) > 0:
            extradummy.drop(indexes, axis=1, inplace=True)
        jEXTRA = extradummy.loc[extradummy.index[np.isnan(extradummy['classes']) == False]]
        # EXCONTROL=extradummy.loc[~extradummy.index[np.isnan(extradummy['classes'])==True]]
        eX, ey, EXTRA = make_features(jEXTRA, target, indexes=indexes)
        extradummy = extradummy.drop(target, axis=1)
        Extra = []
        Extra.append(eX)
        Extra.append(ey)
        Extra.append(EXTRA)
        Extra.append(extradummy)

    X, y, F = make_features(TRAIN, target, indexes=indexes)
    Xt, yt, Ft = make_features(TEST, target, indexes=indexes)
    train = []
    test = []
    train.append(X)
    test.append(Xt)
    train.append(y)
    test.append(yt)
    train.append(F)
    test.append(Ft)

    if extradummy is not None:
        return train, test, CONTROL, Extra

    return train, test, CONTROL


def GetSurface(data, steel='Материал трубы_сталь 20ФА', field='', period=1):
    S = data[data[steel] == 1]
    P1 = S[S['period'] == period]
    if len(field) > 0:
        P1 = P1[P1['Месторождение_' + field] == 1]
    size = P1['R'].value_counts().shape[0] * P1['interval'].value_counts().shape[0]
    # print(size)
    surface = pd.DataFrame(index=np.arange(size), columns=['R', 'interval', 'mean', 'predicted'])
    k = 0
    for i in P1['R'].value_counts().keys():
        r = P1[P1['R'] == i]
        for j in r['interval'].value_counts().keys():
            s = r[r['interval'] == j]
            targ = s['target'].mean()
            pred = s['predicted'].mean()
            surface.loc[k, 'R'] = i
            surface.loc[k, 'interval'] = j
            surface.loc[k, 'mean'] = targ
            surface.loc[k, 'predicted'] = pred
            k = k + 1
    for c in surface.columns:
        surface[c] = surface[c].astype(np.float)
    surf1 = surface.drop(np.where(np.isnan(surface['mean']))[0])
    return surf1


def PlotSurface(surface, title='', cmap='cividis'):
    fig = pylab.figure(figsize=(15, 15))
    axes = Axes3D(fig)
    surf = axes.plot_trisurf(surface['R'], surface['interval'], surface['mean'], cmap=cmap, edgecolor='none',
                             linewidth=1)
    pylab.title('Среднее значение ожидаемого числа отказов. Выборка -20% по ID.' + title)
    fig.colorbar(surf)
    axes.set_xlabel('R')
    axes.set_ylabel('interval')
    fig = pylab.figure(figsize=(15, 15))
    axes = Axes3D(fig)
    surf = axes.plot_trisurf(surface['R'], surface['interval'], surface['predicted'], cmap=cmap, edgecolor='none',
                             linewidth=1)
    pylab.title('Среднее значение прогнозируемого числа отказов. Выборка -20% по ID.' + title)
    fig.colorbar(surf)
    axes.set_xlabel('R')
    axes.set_ylabel('interval')


def GetFirsts(data):
    Ft = data[(np.exp(data['interval_ads']) - 1) == 1]
    return Ft
    # yt=Ft['target'].values


def GetForest(X, y, Xt, yt, n_estimators=100, criterion='mse', n_jobs=2, depth=None, random_state=4):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    forest = RandomForestRegressor(n_estimators=100, max_depth=depth, criterion=criterion, n_jobs=n_jobs,
                                   random_state=random_state)
    forest.fit(X_train, y_train)
    scores = []
    means = []
    stds = []
    scores.append(forest.score(X_train, y_train))
    scores.append(forest.score(X_test, y_test))
    scores.append(forest.score(Xt, yt))
    predicted = forest.predict(Xt)

    residual = np.exp(yt) - np.exp(predicted)

    means.append(residual.mean())
    stds.append(residual.std())
    return scores, means, stds, forest


def PlotKernel(data, field='accident_count', hist=False, target='future accidents', top=30, title='',
               colors=['black', 'red', 'green', 'orange', 'blue', 'brown', 'darkblue']):
    fig, ax = plt.subplots(figsize=(20, 15))
    # R=[1,2,3,5,8, 10, 15]
    R = data[field].value_counts().keys().sort_values()[0:len(colors)]
    # colors=['black','red','green','orange','blue','brown','darkblue']
    k = 0
    for i in R[0:5]:
        dist = data[(data[field] == i) & ((data[target] <= top))][target]
        dist.name = 'Отказы до: ' + str(i) + '. Mean= ' + '{:03.2f}'.format(
            dist.mean()) + '. Var= ' + '{:03.2f}'.format(dist.var())
        c = colors[k]
        # h,bins,=np.histogram(dist.values, bins=50)
        # n=np.sum(h)
        # print(n)
        # h,bins,=np.histogram(dist.values, bins=50)
        # n=np.sum(h)
        # db=bins[1]-bins[0]
        # plt.bar(bins[0:-1],h/n,width=db*0.9, color=c, alpha=0.7)
        try:
            if hist:
                h, bins, = np.histogram(dist.values, bins=50)
                n = np.sum(h)
                db = bins[1] - bins[0]
                plt.bar(bins[0:-1], h / n, width=db * 2, color=c, alpha=0.5)
                dist.plot.kde(ax=ax, legend=True, color=c)
                # plt.hist(h/n,ax=ax,bins=50,density=True, legend=True,color=c,alpha=0.7)
            else:
                dist.plot.kde(ax=ax, legend=True, color=c)
            k = k + 1
        except Exception:
            k = k + 1
            # print('Error')

    ax.set_ylabel('Probability')
    ax.set_xlabel('Количество отказов в будущем')
    # ax.grid(axis='y')
    ax.set_facecolor('white')
    majorFormatter = FormatStrFormatter('%.1f')
    minorFormatter = FormatStrFormatter('%.2f')
    minorLocator = AutoMinorLocator(n=5)

    # ax.xaxis.set_major_locator(MultipleLocator((bins[1]-bins[0])*4))
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_formatter(minorFormatter)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    for label in ax.xaxis.get_ticklabels(which='major'):
        label.set_color('black')
        # label.set_rotation(90)
        label.set_fontsize(12)
    for label in ax.xaxis.get_ticklabels(which='minor'):
        label.set_color('black')
        # label.set_rotation(90)
        label.set_fontsize(10)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1On = True
        tick.label1.set_color('black')
        tick.label2On = False
        tick.label2.set_color('black')
        # серые деления на оси ОY слева
        tick.tick1line.set_color('black')
        tick.tick1line.set_markeredgewidth(2)
        tick.tick1line.set_markersize(15)

    for tick in ax.xaxis.get_minor_ticks():
        tick.label1On = True
        tick.label1.set_color('black')
        tick.label2On = False
        tick.label2.set_color('black')
        tick.tick1line.set_color('black')
        tick.tick1line.set_markeredgewidth(1)
        tick.tick1line.set_markersize(8)
    plt.grid(True)
    plt.rc('xtick', direction='in')
    plt.title(title)


def PlotDistributions(data, hist=False, field='interval_ads_r', top=30, target='future_ads_r', shared_title='',
                      colors=[]):
    for r in data['R'].value_counts().keys().sort_values():
        rdata = data[data['R'] == r]
        for i in rdata['interval'].value_counts().keys().sort_values():
            idata = rdata[rdata['interval'] == i]
            if len(colors) > 0:
                PlotKernel(idata, hist=hist, field='interval_ads_r', colors=colors, top=top,
                           target='future_ads_r',
                           title=shared_title + 'Распределения вероятности отказов. R=' + str(r) + '; интервал ' + str(
                               i))
            else:
                PlotKernel(idata, hist=hist, field='interval_ads_r', top=top,
                           target='future_ads_r',
                           title=shared_title + 'Распределения вероятности отказов. R=' + str(r) + '; интервал ' + str(
                               i))


def PlotPoissonsComparasions(sdata1, field='interval_ads_r', target='future_ads_r', top=30, method=0.35,
                             accidents=[1, 2, 3, 6, 7]):
    for i in accidents:
        pads = sdata1[(sdata1[field] == i) & (sdata1[target] <= top)]
        print("размер выборки: ", pads.shape[0])
        lam1 = pads[target].mean()
        lam2 = pads[target].var()
        # print('lambda= ',lam)
        # print('variance= ',pads['future_ads_r'].std())
        poiss1 = np.random.poisson(lam1, pads.shape[0])
        poiss2 = np.random.poisson(lam2, pads.shape[0])
        po = pd.DataFrame(index=np.arange(poiss1.shape[0]), columns=['poisson_mean', 'poisson_var'])
        po['poisson_mean'] = poiss1
        po['poisson_var'] = poiss2
        print('lambda= ', pads[target].mean(), '; poiss_lambda=', po['poisson_mean'].mean(), '; poiss_var_lambda=',
              po['poisson_var'].mean())
        print('std= ', pads[target].var(), '; poissons_mean_var=', po['poisson_mean'].var(), '; poissons_var=',
              po['poisson_var'].var())
        fig, ax = plt.subplots(figsize=(20, 15))
        pads[target].plot.kde(ax=ax, legend=True, color='black', bw_method=method)
        po['poisson_mean'].plot.kde(ax=ax, legend=True, color='red', bw_method=method)
        po['poisson_var'].plot.kde(ax=ax, legend=True, color='blue', bw_method=method)
        plt.title('Распределения вероятности при количестве отказов в прошлом: ' + str(i - 1))
        ax.set_ylabel('Probability')
        ax.grid(axis='y')
        ax.set_facecolor('white')


def PlotFieldsMean(data, field='Месторождение', title='Распределение интенсивности отказов по месторождениям',
                   target='future_ads_r', colors=['black', 'red', 'green', 'orange', 'blue', 'brown', 'darkblue']):
    # plt.figure(figsize=(12,8))
    fig, ax = plt.subplots(figsize=(20, 15))
    k = 0
    Legs = []
    Labels = []
    plt.title(title)
    plt.xlabel('R')
    plt.ylabel('Mean')
    for f in data[field].value_counts().keys()[0:len(colors)]:
        fdata = data[data[field] == f]
        R = []
        Means = []
        for r in fdata['R'].value_counts().keys().sort_values():
            R.append(r)
            Means.append(fdata[fdata['R'] == r][target].mean())
        g, = plt.plot(R, Means, color=colors[k], marker='o')
        k = k + 1
        Legs.append(g)
        Labels.append(str(f))
    plt.legend(Legs, Labels)
    majorFormatter = FormatStrFormatter('%.2f')
    minorFormatter = FormatStrFormatter('%.2f')
    minorLocator = AutoMinorLocator(n=8)

    # ax.xaxis.set_major_locator(MultipleLocator((bins[1]-bins[0])*4))
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_formatter(minorFormatter)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    for label in ax.xaxis.get_ticklabels(which='major'):
        label.set_color('black')
        # label.set_rotation(90)
        label.set_fontsize(12)
    for label in ax.xaxis.get_ticklabels(which='minor'):
        label.set_color('black')
        label.set_rotation(90)
        label.set_fontsize(10)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1On = True
        tick.label1.set_color('black')
        tick.label2On = False
        tick.label2.set_color('black')
        # серые деления на оси ОY слева
        tick.tick1line.set_color('black')
        tick.tick1line.set_markeredgewidth(2)
        tick.tick1line.set_markersize(15)

    for tick in ax.xaxis.get_minor_ticks():
        tick.label1On = True
        tick.label1.set_color('black')
        tick.label2On = False
        tick.label2.set_color('black')
        tick.tick1line.set_color('black')
        tick.tick1line.set_markeredgewidth(1)
        tick.tick1line.set_markersize(8)
    plt.grid(True)
    plt.rc('xtick', direction='in')


def R2(y, predicted):
    r1 = y - predicted
    s1 = r1 ** 2
    D1 = np.sum(s1)
    mean = y.mean()
    r2 = y - mean
    s2 = r2 ** 2
    D2 = np.sum(s2)

    if (D1 == 0) & (D2 == 0): return 1
    if D1 == D2: return 0
    if D2 == 0: return np.infty
    r = 1 - (D1 / D2)
    return r


def display_df(df, field='Месторождение', index='index', value=1270, indexes={}, fields=[]):
    sub_df = df[df[field] == value]
    print(field + ' ' + str(value))
    for i, r in indexes.items():
        ind = sub_df[sub_df[index] == i]
        print(str(index) + ': ', str(i), 'R^2', str(r))
        return ind[fields]


def GetScore(FN, field='index', columns=['r2', 'index', 'Месторождение']):
    i = 0
    SCORES = pd.DataFrame(columns=columns)
    for f in FN['Месторождение'].value_counts().keys():
        scores = []
        Ff = FN[FN['Месторождение'] == f]
        # print(str(f),'; ',str(len(Ff['index'].value_counts().keys())))
        for ind in Ff[field].value_counts().keys():
            sind = Ff[Ff[field] == ind]

            tar = sind['target']
            pred = sind['predicted']
            r2 = R2(tar.values, pred.values)
            if r2 > -10:
                scores.append((r2, ind, f))
                i = i + 1
        if len(scores) > 0:
            SCORE = pd.DataFrame(np.array(scores), columns=columns)
            SCORES = SCORES.append(SCORE)
    SCORES.index = np.arange(SCORES.shape[0])
    # print(i)
    return SCORES


def plot_predicted(DATA=[], I=[], colors=['black', 'green', 'red', 'blue'],
                   title='Процент  совпадений прогнозов на  интервалах'):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    LEGENDS = []
    ERRORS = []
    i = 0
    for data in DATA:
        g, = plt.plot(I, data, marker='v', color=colors[i])
        LEGENDS.append(g)
        ERRORS.append('Погрешность ' + str(i))
        i = i + 1
    plt.title(title)
    majorFormatter = FormatStrFormatter('%.0f')
    minorFormatter = FormatStrFormatter('%.0f')
    minorLocator = AutoMinorLocator(n=2)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_formatter(minorFormatter)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_formatter(majorFormatter)
    plt.ylabel('%')
    plt.xlabel('интервал')
    plt.legend(LEGENDS, ERRORS)
    plt.grid(True)


def GetErrors(data):
    I = []
    ZEROS = []
    ONES = []
    TWOS = []
    THREES = []
    for f in data['interval'].value_counts().keys().sort_values():
        ff = data[data['interval'] == f]
        res = np.abs(ff['target'] - ff['predicted1'])
        zeros = res.iloc[np.where(res.values == 0)]
        ones = res.iloc[np.where(res.values == 1)]
        # print(zeros.shape)
        twos = res.iloc[np.where(res.values == 2)]
        threes = res.iloc[np.where(res.values == 3)]
        percent = (zeros.shape[0] / res.shape[0]) * 100
        I.append(f)
        ZEROS.append(percent)
        # print(str(f)+' '+str(zeros.shape[0]))
        ONES.append((ones.shape[0] / res.shape[0]) * 100)
        TWOS.append((twos.shape[0] / res.shape[0]) * 100)
        THREES.append((threes.shape[0] / res.shape[0]) * 100)
    return [ZEROS, ONES, TWOS, THREES], I


def GetErrors_v1(data, Abs=True):
    intervals = data['interval'].value_counts().keys().sort_values()
    columns = []
    columns.append('interval')
    data['error'] = data['predicted1'] - data['target']
    if Abs:
        data['error'] = np.abs(data['error'])
    errors = data['error'].value_counts().keys().sort_values()
    for e in errors:
        columns.append(str(e))
    index = np.arange(len(intervals))
    ERRORS = pd.DataFrame(index=index, columns=columns)
    for c in columns[1:]:
        ERRORS[c] = 0
    i = 0
    for f in intervals:
        ff = data[data['interval'] == f]
        ERRORS.loc[i, 'interval'] = f
        res = ff['error']
        for v in res.value_counts().keys().sort_values():
            zeros = res.iloc[np.where(res.values == v)]
            percent = (zeros.shape[0] / res.shape[0]) * 100
            ERRORS.loc[i, str(v)] = percent
        i = i + 1

    return ERRORS


def GetCounts_v1(data):
    intervals = data['interval'].value_counts().keys().sort_values()
    columns = []
    columns.append('interval')
    data['error'] = data['target']

    errors = data['error'].value_counts().keys().sort_values()
    for e in errors:
        columns.append(str(e))
    index = np.arange(len(intervals))
    ERRORS = pd.DataFrame(index=index, columns=columns)
    for c in columns[1:]:
        ERRORS[c] = 0
    i = 0
    for f in intervals:
        ff = data[data['interval'] == f]
        ERRORS.loc[i, 'interval'] = f
        res = ff['error']
        for v in res.value_counts().keys().sort_values():
            zeros = res.iloc[np.where(res.values == v)]
            percent = (zeros.shape[0] / res.shape[0]) * 100
            ERRORS.loc[i, str(v)] = percent
        i = i + 1

    return ERRORS


def plot_predicted_v1(DATA,
                      colors=['black', 'green', 'red', 'blue', 'brown', 'orange', 'yellow', 'gray', 'pink', 'indigo'],
                      title='Процент  совпадений прогнозов на  интервалах'):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    LEGENDS = []
    ERRORS = []
    i = 0
    I = DATA['interval']
    for c in DATA.columns[1:]:
        data = DATA[c].values
        g, = plt.plot(I, data, marker='v', color=colors[i])
        LEGENDS.append(g)
        ERRORS.append('Погрешность ' + str(c))
        i = i + 1
    plt.title(title)
    majorFormatter = FormatStrFormatter('%.0f')
    minorFormatter = FormatStrFormatter('%.0f')
    minorLocator = AutoMinorLocator(n=2)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_formatter(minorFormatter)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_formatter(majorFormatter)
    plt.ylabel('%')
    plt.xlabel('интервал')
    plt.legend(LEGENDS, ERRORS)
    plt.grid(True)


def plot_interval_discributions(df, cmap='viridis', title='Распределение',
                                levels=[0.5, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90], x_label='Ошибка'):
    data = df[df.columns[1:]].values
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    cf = plt.contourf(data, cmap=cmap, levels=levels)
    majorFormatter = FormatStrFormatter('%.0f')
    minorFormatter = FormatStrFormatter('%.0f')
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(majorFormatter)
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    for i in df.index:
        ylabels[i + 1] = df.loc[i, 'interval']
    k = 1
    # print(len(xlabels))
    for c in df.columns[1:]:
        xlabels[k] = c
        k = k + 1

    # majorFormatter = FormatStrFormatter('%.0f')
    # minorFormatter = FormatStrFormatter('%.0f')
    ax.set_yticklabels(ylabels)
    ax.set_xticklabels(xlabels)
    plt.ylabel('интервал')
    plt.xlabel(x_label)
    plt.colorbar(cf)
    plt.title(title)


def plot_interval_discributions_v1(Fd, cmap='viridis', cmapv='coolwarm', alpha=0.5, marker='o', title='Распределение',
                                   levels=[0.5, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90], x_label='Ошибка'):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    df = GetErrors_v1(Fd, Abs=False)
    data = df[df.columns[1:]].values
    cf = plt.contourf(data, cmap=cmap, levels=levels)
    plt.colorbar(cf)
    # cb=plt.imshow(Fd['target'].values.reshape(-1,1),cmap=cmapv)
    norm = plt.Normalize(vmin=Fd['target'].min(), vmax=Fd['target'].max())
    cmap1 = plt.cm.get_cmap(cmapv)
    k = 0
    min_error = Fd['error_float'].min()
    for i in Fd['interval'].value_counts().keys().sort_values():
        interval = Fd[Fd['interval'] == i]
        for v in interval['error'].value_counts().keys().sort_values():
            vs = interval[interval['error'] == v]
            I = pd.DataFrame(index=np.arange(vs.shape[0]), columns=['interval'])
            I['interval'] = k

            plt.scatter(vs['error_float'].values - min_error, I, marker=marker, alpha=alpha,
                        c=cmap1(norm(vs['target'].values)))

        k = k + 1

    # plt.colorbar()
    majorFormatter = FormatStrFormatter('%.0f')
    minorFormatter = FormatStrFormatter('%.0f')
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(majorFormatter)
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    for i in df.index:
        ylabels[i + 1] = df.loc[i, 'interval']
    k = 1
    # print(len(xlabels))
    for c in df.columns[1:]:
        xlabels[k] = c
        k = k + 1

    # majorFormatter = FormatStrFormatter('%.0f')
    # minorFormatter = FormatStrFormatter('%.0f')
    ax.set_yticklabels(ylabels)
    ax.set_xticklabels(xlabels)
    plt.ylabel('интервал')
    plt.xlabel(x_label)

    # cb.set_visible(False)
    # plt.colorbar(cb)
    plt.title(title)


def classes(x):
    if x == 0: return 0
    if (x >= 1) & (x <= 2): return 1
    if (x >= 3) & (x <= 5): return 2
    if (x >= 6) & (x <= 10): return 3
    if x > 10: return 4


def classes_contin(x):
    if x == 0: return 0
    if (x >= 1) & (x <= 2): return 1
    if (x > 2) & (x <= 5): return 2
    if (x > 5) & (x <= 10): return 3
    if x > 10: return 4


class Predictor():
    def __init__(self, class_weight={0: 5, 1: 1000, 2: 10, 3: 1, 4: 1}, criterion='entropy', max_depth=None):
        self.regressor = RandomForestRegressor(n_estimators=100, n_jobs=-1, criterion='mse')
        self.classifier = RandomForestClassifier(n_estimators=100, class_weight=class_weight, max_depth=max_depth,
                                                 criterion=criterion, n_jobs=-1, random_state=4)

    def fit(self, F, y):
        self.p = F['classes']
        self.G = F.copy()
        self.G.drop(['classes'], axis=1, inplace=True)
        self.Q = self.G.values
        self.X = F.values
        self.y = y
        # print(self.G.shape)
        # print(self.G.values.shape)
        # print(self.Q.shape)
        # print(self.p.shape)
        self.regressor.fit(self.X, self.y)
        self.classifier.fit(self.Q, self.p)

    def predict(self, Ft):
        # pt=Ft['classes']
        Gt = Ft.copy()
        Xt = Ft.copy()
        Gt.drop(['classes'], axis=1, inplace=True)
        Qt = Gt.values
        self.cl_predicted = self.classifier.predict(Qt)
        Xt['classes'] = self.cl_predicted
        self.predicted = self.regressor.predict(Xt.values)
        return self.predicted

    def score(self, Ft, yt):
        pt = Ft['classes']
        Gt = Ft.copy()
        Xt = Ft.copy()
        Gt.drop(['classes'], axis=1, inplace=True)
        Qt = Gt.values
        self.cl_score = self.classifier.score(Qt, pt)
        self.cl_predicted = self.classifier.predict(Qt)
        Xt['classes'] = self.cl_predicted
        self.reg_score = self.regressor.score(Xt.values, yt)


def GetUnicalRepairs(data, scale=0,values=False):
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

    repairs = pd.DataFrame(list(repairs), columns=['Дата ремонта', 'Адрес', 'Длина']).sort_values(
        by='Дата ремонта').reset_index(drop=True)
    repairs['Дата ремонта'] = pd.to_datetime(repairs['Дата ремонта'])
    if values:
        repairs['b'] = repairs['Адрес'] + repairs['Длина']
        values = repairs[['Адрес', 'b', 'Дата ремонта']].values
        return values
    return repairs
def get_unical_repairs(data, dfield='repair_date',lfield='repair_lenght',afield='repair_adress', scale=0,values=False):
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


def SplitPipes(data, date, address, length, prefix):
    before = data[data['Дата аварии'] <= date]
    ID = str(data['ID простого участка'].value_counts().keys()[0])
    a = data['a'].value_counts().keys()[0]
    b = data['b'].value_counts().keys()[0]
    # before['new_id']=before['ID простого участка']
    # cdata=data[data['Дата аварии']<=nextdate]
    # print(cdata.shape)
    cdata = data
    after1 = cdata[cdata['Адрес от начала участка'] < address]
    after2 = cdata[cdata['Адрес от начала участка'] > address + length]
    after3 = cdata[(cdata['Дата аварии'] > date) & (
            (cdata['Адрес от начала участка'] <= address + length) & (cdata['Адрес от начала участка'] >= address))]
    print(date)
    print(cdata['Дата аварии'].max())
    after1['new_id'] = ID + prefix + '_1'
    after1['a'] = a
    after1['b'] = address
    after3['new_id'] = ID + prefix + '_3'
    after3['a'] = address
    after3['b'] = address + length
    after3['Дата ввода'] = date
    after2['new_id'] = ID + prefix + '_2'
    after2['a'] = address + length
    after2['b'] = b

    return before, after1, after2, after3


def SplitPipes_v1(data, k, repairs):
    items = []
    if data.shape[0] == 0: return items
    date = repairs.loc[k, 'Дата ремонта']
    address = repairs.loc[k, 'Адрес']
    length = repairs.loc[k, 'Длина']
    before = data[data['Дата аварии'] <= date]
    # ID=str(data['ID простого участка'].value_counts().keys()[0])
    ID = str(data['new_id'].value_counts().keys()[0])
    a = data['a'].value_counts().keys()[0]
    b = data['b'].value_counts().keys()[0]
    # print('ID= ',ID, 'a=', a,'b=',b,'address=', address,' length=', length, 'k=',k)

    if (address >= a) & (address <= b):
        # before['new_id']=before['ID простого участка']
        # cdata=data[data['Дата аварии']<=nextdate]

        cdata = data
        # print('data_shape',cdata.shape)
        after1 = cdata[cdata['Адрес от начала участка'] < address]
        after2 = cdata[cdata['Адрес от начала участка'] > address + length]
        after3 = cdata[(cdata['Дата аварии'] > date) & ((cdata['Адрес от начала участка'] <= address + length) & (
                cdata['Адрес от начала участка'] >= address))]
        # print(date)
        # print(cdata['Дата аварии'].max())
        after1['new_id'] = ID + str(k + 1) + '_1'
        after1['a'] = a
        after1['b'] = address
        # print('id=',ID+str(k+1)+'_1','a=',a, ' b=',address,'af1=', after1.shape)
        after3['new_id'] = ID + str(k + 1) + '_3'
        after3['a'] = address
        after3['b'] = address + length
        # print('id=',ID+str(k+1)+'_3','a=',address, ' b=',address+length,'af3=', after3.shape)
        after3['Дата ввода'] = date
        after2['new_id'] = ID + str(k + 1) + '_2'
        after2['a'] = address + length
        after2['b'] = b
        # print('id=',ID+str(k+1)+'_2','a=',address+length, ' b=',b,'af2=', after2.shape)

        items.append(before)
        items.append(after1)
        items.append(after2)
        items.append(after3)
    else:

        items.append(data)

    return items


def SplitPipes_v2(data, k, repairs):
    items = []
    if data.shape[0] == 0: return items
    date = repairs.loc[k, 'Дата ремонта']
    address = repairs.loc[k, 'Адрес']
    length = repairs.loc[k, 'Длина']
    before = data[data['Дата аварии'] <= date]
    # ID=str(data['ID простого участка'].value_counts().keys()[0])
    ID = str(data['new_id'].value_counts().keys()[0])
    a = data['a'].value_counts().keys()[0]
    b = data['b'].value_counts().keys()[0]
    # print('ID= ',ID, 'a=', a,'b=',b,'address=', address,' length=', length, 'k=',k)

    if (address >= a) & (address <= b):
        # before['new_id']=before['ID простого участка']
        # cdata=data[data['Дата аварии']<=nextdate]

        cdata = data
        # print('data_shape',cdata.shape)
        after1 = cdata[cdata['Адрес от начала участка'] < address]
        after2 = cdata[cdata['Адрес от начала участка'] > address + length]
        after3 = cdata[(cdata['Дата аварии'] > date) & ((cdata['Адрес от начала участка'] <= address + length) & (
                cdata['Адрес от начала участка'] >= address))]
        # print(date)
        # print(cdata['Дата аварии'].max())
        after1['new_id'] = ID + str(k + 1) + '_1'
        after1['a'] = a
        after1['b'] = address
        after1['Дата ремонта'] = date
        # print('id=',ID+str(k+1)+'_1','a=',a, ' b=',address,'af1=', after1.shape)
        after3['new_id'] = ID + str(k + 1) + '_3'
        after3['a'] = address
        after3['b'] = address + length
        after3['Дата ремонта'] = date
        # print('id=',ID+str(k+1)+'_3','a=',address, ' b=',address+length,'af3=', after3.shape)
        after3['Дата ввода'] = date
        after2['new_id'] = ID + str(k + 1) + '_2'
        after2['a'] = address + length
        after2['b'] = b
        after2['Дата ремонта'] = date
        # print('id=',ID+str(k+1)+'_2','a=',address+length, ' b=',b,'af2=', after2.shape)
        # df.loc[df['Type']=='ab', 'Price'] /= 1000
        before.loc[
            (before['Адрес от начала участка'] >= address) & (before['Адрес от начала участка'] <= address + length)
            , 'Дата перевода в бездействие'] = date
        items.append(before)
        items.append(after1)
        items.append(after2)
        items.append(after3)
    else:

        items.append(data)

    return items



def MakeSplits(data, k, repairs):
    if repairs.shape[0] == 0:  return data
    items = SplitPipes_v1(data, k, repairs)
    if len(items) == 0: return items
    before = items[0]
    afters = items[1:]
    if k + 1 < repairs.shape[0]:
        # print(k+1)
        for after in afters:
            if after.shape[0] > 0:
                empty = True
                while empty & (k + 1 < repairs.shape[0]):
                    nbefore = MakeSplits(after, k + 1, repairs)
                    if len(nbefore) > 0:
                        empty = False
                    else:
                        k = k + 1
                    # else: print(k+1)

                before = before.append(nbefore)
    else:
        for after in afters:
            if after.shape[0] > 0:
                print('added', after.shape[0])
                before = before.append(after)

    return before


def MakeSplits_v1(data, k, repairs):
    if repairs.shape[0] == 0:  return data
    items = SplitPipes_v2(data, k, repairs)

    if len(items) == 4:
        before = items[0]
        afters = items[1:]
    else:
        before = pd.DataFrame(columns=items[0].columns)
        afters = []
        afters.append(items[0])

    if k + 1 == repairs.shape[0]:
        for after in afters:
            if after.shape[0] > 0:
                before = before.append(after)

    else:
        for after in afters:
            if after.shape[0] > 0:
                empty = True
                while empty & (k + 1 < repairs.shape[0]):
                    nbefore = MakeSplits_v1(after, k + 1, repairs)
                    if len(nbefore) > 0:
                        before = before.append(nbefore)
                        empty = False
                        break
                    else:
                        k = k + 1

    return before


def SplitDataByRepairs(data, scale=2):
    cdata = pd.DataFrame(columns=data.columns)
    for ID in data['ID простого участка'].value_counts().keys():
        someid = data[data['ID простого участка'] == ID].sort_values(by='Дата аварии')
        dates = ['Дата аварии', 'Дата перевода в бездействие', 'Дата окончания ремонта', 'Дата ремонта до аварии']
        for date in dates:
            someid[date] = someid[date].dt.date
            someid['a'] = 0
            someid['b'] = someid['L']
            someid['new_id'] = someid['ID простого участка']
            someid['Дата ремонта'] = np.nan
        repairs = GetUnicalRepairs(someid, scale=scale)
        Frame = MakeSplits_v1(someid, 0, repairs)
        cdata = cdata.append(Frame)
    return cdata


def dateaddfloat_v1(date, offset):
    year = np.modf(offset)[1]
    print(year)
    month = np.modf(np.modf(offset)[0] * 12)[1]
    print(month)
    print(date)
    date1 = date + np.timedelta64(int(year), 'Y') + np.timedelta64(int(month), 'M')

    print(date1)
    if date1.month == 12:
        date1 = date + np.timedelta64(int(year) + 1, 'Y') + np.timedelta64(1, 'M')
    days = int(calendar.monthrange(date1.year, date1.month + 1)[1] * np.modf(np.modf(offset)[0] * 12)[0])
    date2 = date1 + np.timedelta64(days, 'D')
    return date2


def dateaddfloat(date, offset):
    days = int(offset * 365)
    date2 = date + np.timedelta64(days, 'D')
    return date2


def GetIndexes(data, field='future_ads_r'):
    idxs = data[field].apply(lambda x: set(x))
    s = set()
    for r in idxs:
        s = s | r
    return s
def GetIndicesOfList(data,column=1,l=[]):
    if len(l)==0:return []
    mask=data[column]==l[0]
    print(l[0], ',', len(mask[mask == True].keys()))
    if len(l)>1:
        for k in l[1:]:
            x=data[column]==k
            print(k,',',len(x[x==True].keys()))
            mask=mask|x
    indices=mask[mask==True].keys()
    return indices

def to_nparray(x):
    i = x.index('[') + 1
    j = x.index(']')
    arr = x[i:j].split(',')
    if len(arr[0]) == 0:
        return np.array([])
    else:
        return np.array(arr, dtype='int')


def GetSeparatedData(file, path='D:\\ml\\', target='future_ads_r', start=2014, end=2015):
    long = ['Вид покрытия внутреннего', 'Название', 'ID простого участка',
            'Месторождение', 'D', 'L', 'S', 'Материал трубы', 'Тип трубы',
            'Завод изготовитель', 'Дата ввода', 'Состояние',
            'Дата перевода в бездействие', 'Дата аварии', 'Наработка до отказа',
            'Адрес от начала участка', 'Обводненность', 'Скорость потока',
            'Способ ликв в момент', 'ID ремонта', 'Дата окончания ремонта',
            'Адрес от начала участка.1', 'Длина ремонтируемого участка',
            'Тип ремонта', 'ID ремонта до аварии', 'Дата ремонта до аварии',
            'Адрес ремонта до аварии', 'Длина ремонта до аварии',
            'Тип ремонта до аварии', 'original_index', 'a', 'b', 'new_id',
            'Дата ремонта', 'Наработка до отказа(new), лет', 'L,м', 'R', 'r',
            'interval', 'interval_ads', 'future_ads', 'index', 'last_ads',
            'first_ads', 'interval_ads_r', 'future_ads_r1', 'future_ads_r',
            'period', 'Дата ремонта', 'extra_lenght']
    short = ['R', 'Наработка до отказа(new), лет', 'Адрес от начала участка (new)', 'original_index', 'a', 'b',
             'new_id',
             'Наработка до отказа(new), лет', 'L,м', 'R', 'r',
             'interval', 'interval_ads', 'int_ads_halfyear', 'int_ads_oneyear', 'int_ads_twoyear', 'int_ads_threeyear',
             'future_ads', 'index', 'last_ads',
             'first_ads', 'interval_ads_r', 'future_ads_r1', 'future_ads_r',
             'period', 'Дата аварии', 'Дата ввода', 'D', 'L', 'S', 'Обводненность', 'Скорость потока', 'extra_lenght']
    dates = ['Дата ввода', 'Дата аварии', 'last_ads', 'first_ads']
    # path='D:\\ml\\'
    DATA = LoadIntervals(file=file, path=path, short=short, dates=dates)
    print(DATA.shape)
    # target='future_ads_r'
    # cdata=data[(data['future_ads_r']>=0)&(data['L,м']>0)]
    ndata = DATA[DATA['L,м'] > 0]
    ndata['classes'] = ndata[target].apply(classes)
    ndata['lenght'] = ndata['rbound'] - ndata['lbound']
    ndata = ndata[ndata['lenght'] >= 0]
    # ndata['ads_classes']=ndata[target].apply(classes)
    mask = ndata['ID простого участка'].astype('str') != ndata['new_id']
    ndata['new'] = 0
    ndata.loc[mask, 'new'] = 1
    dic = dict({'future_ads_r': 'ads_r_indexes', 'future_ads_r1': 'ads_r1_indexes', 'future_ads': 'ads_indexes',
                'future_calendar': 'calendar_indexes'})
    for c in dic.keys():
        f = dic[c]
        ndata[f] = ndata[ndata[c] >= 0][f].apply(lambda x: set(to_nparray(x)))
    dictionary = dict()
    for y in ndata["Дата аварии"].dt.year.value_counts().keys().sort_values():
        # print(y)
        mask = ndata["Дата аварии"].dt.year == y
        subyear = ndata[mask]
        dictionary.update({y: subyear})
    dictionary.update({'all': ndata})
    # mask=ndata["Дата аварии"].dt.year==end
    # mask1=ndata["Дата аварии"].dt.year==start
    # extra=ndata[mask]
    # extra3=ndata[mask1]
    # dict({end:extra,start:extra3,'all':ndata})
    return dictionary


def GetMedianMeansValues(data, field='Месторождение'):
    dictionary = dict()
    for f in data[field].value_counts().keys():
        group = data[data[field] == f]
        yeardict = dict()
        for year in group['Дата аварии'].dt.year.value_counts().keys():
            y = group[group['Дата аварии'].dt.year == year]
            wmedian = y['Обводненность'].median()
            wmean = y['Обводненность'].mean()
            smedian = y['Скорость потока'].median()
            smean = y['Скорость потока'].mean()
            subdict = dict({'Обводненность': {'median': wmedian, 'mean': wmean},
                            'Скорость потока': {'median': smedian, 'mean': smean}})
            yeardict.update({year: subdict})
        dictionary.update({f: yeardict})
    return dictionary


path = 'D:\\ml\\'
#weights = np.load(path + 'class_weights.npy', allow_pickle=True).item()
regim_dict = np.load(path + 'regims.npy', allow_pickle=True).item()


class YearPredictor_v1:
    def __init__(self):
        self.predictions = None
        self.predictions_full = None
        self.mask = None
        self.Model1 = Model()
        self.Model2 = Model()
        self.Model3 = Model()

    def GetYearPredictions(self):
        if self.Model1.Predictor is None or self.Model2.Predictor is None or self.Model3.Predictor is None: return None
        self.Model1.train_predictions = self.Model1.Predictor.predict(self.Model1.features['EXTRA']['F'])
        self.Model1.train_cl_predictions = self.Model1.Predictor.cl_predicted
        self.Model1.control_predictions = self.Model1.Predictor.predict(self.Model1.features['EXTRA']['CONTROL'])
        self.Model2.train_predictions = self.Model2.Predictor.predict(self.Model2.features['EXTRA']['F'])
        self.Model2.train_cl_predictions = self.Model2.Predictor.cl_predicted
        self.Model2.control_predictions = self.Model2.Predictor.predict(self.Model2.features['EXTRA']['CONTROL'])
        self.Model3.train_predictions = self.Model3.Predictor.predict(self.Model3.features['EXTRA']['F'])
        self.Model3.train_cl_predictions = self.Model3.Predictor.cl_predicted
        self.Model3.control_predictions = self.Model3.Predictor.predict(self.Model3.features['EXTRA']['CONTROL'])
        self.Model1.JoinToFields()
        self.Model2.JoinToFields()
        self.Model3.JoinToFields()
        self.JoinPredictions()

    def JoinPredictions(self):
        if self.Model1.predictions is not None and self.Model2.predictions is not None and self.Model3.predictions is not None:
            self.predictions = self.Model1.predictions.append(self.Model2.predictions, ignore_index=True)
            self.predictions = self.predictions.append(self.Model3.predictions, ignore_index=True)
        if self.Model1.predictions_full is not None and self.Model2.predictions_full is not None:
            self.predictions_full = self.Model1.predictions_full.append(self.Model2.predictions_full, ignore_index=True)
            self.predictions_full = self.predictions_full.append(self.Model3.predictions_full, ignore_index=True)

    def GetIDsFrame(self, main, mask):
        data = main[mask]
        count = np.sum(data['interval'].value_counts().keys())
        results = pd.DataFrame(index=np.arange(count),
                               columns=['ID простого участка', 'new_id', 'interval', 'Predicted', 'Fact'])
        k = 0
        for ID in data['ID простого участка'].value_counts().keys():
            group = data[data['ID простого участка'] == ID]
            for interval in group['interval'].value_counts().keys():
                INTERVAL = group[group['interval'] == interval]
                summ = INTERVAL['predicted'].sum()
                summ1 = INTERVAL['target'].sum()
                results.loc[k, 'Predicted'] = summ
                results.loc[k, 'Fact'] = summ1
                results.loc[k, 'ID простого участка'] = ID
                results.loc[k, 'new_id'] = INTERVAL['new_id'].value_counts().keys()[0]
                results.loc[k, 'interval'] = interval
                results.loc[k, 'D'] = INTERVAL['D'].value_counts().keys()[0]
                results.loc[k, 'L,м'] = INTERVAL['L,м'].value_counts().keys()[0]
                k = k + 1
        results = results.loc[~np.isnan(results['interval'].astype('float'))]
        return results

    def GetMask(self, data):
        mask = ((data['D'] <= 219) & (data['interval'] == 150)) | ((data['D'] > 219) & (data['interval'] == 100))
        mask_x = (data['L,м'] >= (data['interval'] * 2)) & (data['L,м'] < ((data['interval'] + 10) * 2)) | mask
        self.mask = mask_x
        return self.mask


class YearPredictor:
    def __init__(self):
        self.predictions = None
        self.predictions_full = None
        self.mask = None
        self.Model1 = Model()
        self.Model2 = Model()

    def GetYearPredictions(self):
        if self.Model1.Predictor is None or self.Model2.Predictor is None: return None
        self.Model1.train_predictions = self.Model1.Predictor.predict(self.Model1.features['EXTRA']['F'])
        self.Model1.train_cl_predictions = self.Model1.Predictor.cl_predicted
        self.Model1.control_predictions = self.Model1.Predictor.predict(self.Model1.features['EXTRA']['CONTROL'])
        self.Model2.train_predictions = self.Model2.Predictor.predict(self.Model2.features['EXTRA']['F'])
        self.Model2.train_cl_predictions = self.Model2.Predictor.cl_predicted
        self.Model2.control_predictions = self.Model2.Predictor.predict(self.Model2.features['EXTRA']['CONTROL'])
        self.Model1.JoinToFields()
        self.Model2.JoinToFields()
        self.JoinPredictions()

    def JoinPredictions(self):
        if self.Model1.predictions is not None and self.Model2.predictions is not None:
            self.predictions = self.Model1.predictions.append(self.Model2.predictions, ignore_index=True)
        if self.Model1.predictions_full is not None and self.Model2.predictions_full is not None:
            self.predictions_full = self.Model1.predictions_full.append(self.Model2.predictions_full, ignore_index=True)

    def GetIDsFrame(self, main, mask):
        data = main[mask]
        count = np.sum(data['interval'].value_counts().keys())
        results = pd.DataFrame(index=np.arange(count),
                               columns=['ID простого участка', 'new_id', 'interval', 'Predicted', 'Fact'])
        k = 0
        for ID in data['ID простого участка'].value_counts().keys():
            group = data[data['ID простого участка'] == ID]
            for interval in group['interval'].value_counts().keys():
                INTERVAL = group[group['interval'] == interval]
                summ = INTERVAL['predicted'].sum()
                summ1 = INTERVAL['target'].sum()
                results.loc[k, 'Predicted'] = summ
                results.loc[k, 'Fact'] = summ1
                results.loc[k, 'ID простого участка'] = ID
                results.loc[k, 'new_id'] = INTERVAL['new_id'].value_counts().keys()[0]
                results.loc[k, 'interval'] = interval
                results.loc[k, 'D'] = INTERVAL['D'].value_counts().keys()[0]
                results.loc[k, 'L,м'] = INTERVAL['L,м'].value_counts().keys()[0]
                k = k + 1
        results = results.loc[~np.isnan(results['interval'].astype('float'))]
        return results

    def GetMask(self, data):
        mask = ((data['D'] <= 219) & (data['interval'] == 150)) | ((data['D'] > 219) & (data['interval'] == 100))
        mask_x = (data['L,м'] >= (data['interval'] * 2)) & (data['L,м'] < ((data['interval'] + 10) * 2)) | mask
        self.mask = mask_x
        return self.mask


class Model:
    def __init__(self):
        self.features = None
        self.predictions = None
        self.predictions_full = None
        self.fields = ['ID простого участка', 'Дата аварии', 'Адрес от начала участка (new)', 'interval', 'new_id', 'D',
                       'L,м', 'index', 'classes']
        self.class_weight = None
        self.train_predictions = None
        self.train_cl_predictions = None
        self.control_predictions = None
        self.Predictor = None
        self.data = None
        self.control_data = None
        self.parameters = dict({'drop_after': ['start', 'accident', 'last_interval',
                                               'first_interval', 'interval'],
                                'drop_before': ['Тип трубы', 'Завод изготовитель'],
                                'dummies': ['Месторождение', 'Материал трубы', 'Способ ликв в момент',
                                            'Вид покрытия внутреннего'],
                                'dates': [['Дата ввода', 'start'], ['Дата аварии', 'accident'],
                                          ['last_ads', 'last_interval'], ['first_ads', 'first_interval']],
                                'fields': ['Вид покрытия внутреннего',
                                           'Месторождение', 'D', 'S', 'Материал трубы',
                                           'Тип трубы', 'Завод изготовитель', 'Дата ввода',
                                           'Дата аварии', 'Обводненность', 'Скорость потока',
                                           'Способ ликв в момент', 'Наработка до отказа(new), лет', 'S/D', 'L,м', 'R',
                                           'interval', 'lenght', 'extra_lenght', 'index', 'last_ads',
                                           'first_ads', 'interval_ads',
                                           'period', 'interval_age', 'classes', 'interval_to_ads', 'int_ads_halfyear',
                                           'int_ads_oneyear', 'int_ads_twoyear', 'int_ads_threeyear',
                                           'Адрес от начала участка (new)'],
                                'loglist': [], 'split': np.datetime64('2015-01-01'), 'target': 'future_ads_r',
                                'tail': pd.DateOffset(years=4),
                                'split_by': pd.DateOffset(years=10), 'indexes': 'index', 'period': 2,
                                'date_column': 'Дата аварии', 'by_date': True, 'seed': 42, 'drop_end': False})

    def GetFeatures(self, data, control_data):
        self.data = data
        self.control_data = control_data
        self.features = GetMainFeatures(self.data, self.control_data, parameters=self.parameters)
        # return self.features

    def GetPredictor(self, class_weight=None, max_depth=None):
        self.class_weight = class_weight
        if self.features is None: return None
        self.Predictor = Predictor(class_weight=class_weight, max_depth=max_depth)
        self.Predictor.fit(self.features['Regression']['TRAIN']['F'], self.features['Regression']['TRAIN']['y'])
        self.Predictor.score(self.features['Regression']['TEST']['F'], self.features['Regression']['TEST']['y'])
        print(self.Predictor.cl_score)
        print(self.Predictor.reg_score)
        return self.Predictor

    def JoinToFields(self):
        self.predictions = self.control_data.loc[self.control_data.index.isin(self.features['EXTRA']['F'].index)][
            self.fields]
        self.predictions_full = \
        self.control_data.loc[self.control_data.index.isin(self.features['EXTRA']['CONTROL'].index)][self.fields]
        self.predictions['predicted'] = self.train_predictions
        self.predictions['cl_predicted'] = self.train_cl_predictions
        self.predictions['target'] = self.features['EXTRA']['y']
        self.predictions_full['predicted'] = self.control_predictions
        self.predictions_full['target'] = np.nan


def GetYearInalysis(sdata, tdata, cdata, year=2016, offset=1):
    data = pd.DataFrame(index=[0], columns=['Всего в ' + str(year), 'Первичных', 'Повторных', 'Повторных >20м',
                                            'Вышли по ремонту в ' + str(year), 'Предсказуемы из модели',
                                            'Введены в эксплуатацию в ' + str(year), 'Ожидаемый прогноз'])
    iyear = tdata[(tdata['Дата аварии'] >= pd.to_datetime('01-01-' + str(year))) & (
                tdata['Дата аварии'] < pd.to_datetime('01-01-' + str(year + offset)))]
    # iyear=tdata[tdata['Дата аварии'].dt.year==year]
    # iy=cdata[cdata['Дата аварии'].dt.year==year]
    iy = cdata[(cdata['Дата аварии'] >= pd.to_datetime('01-01-' + str(year))) & (
                cdata['Дата аварии'] < pd.to_datetime('01-01-' + str(year + offset)))]
    IDXES = GetIndexes(sdata['all'][sdata['all']['future_ads_r'] >= 0], 'ads_r_indexes')
    withfact = iyear[iyear.index.isin(IDXES)]
    iyear['Предсказуемы из модели с подтверждением'] = False
    iyear.loc[withfact.index, 'Предсказуемы из модели с подтверждением'] = True
    alone = []
    for ind in iy['index'].value_counts().keys():
        someind = iy[iy['index'] == ind]
        if GetAloneIndexes(someind):
            alone.append(ind)
    iyear['Первичный отказ'] = False
    iyear.loc[iyear['original_index'].isin(alone), 'Первичный отказ'] = True
    iyear['Вышли в ' + str(year) + ' по ремонту или замене'] = False
    iyear.loc[iyear[iyear['Дата перевода в бездействие'].dt.year == year].index, 'Вышли в ' + str(
        year) + ' по ремонту или замене'] = True
    iyear[['ID простого участка', 'Месторождение', 'Дата ввода',
           'Дата перевода в бездействие', 'Дата аварии', 'Наработка до отказа', 'new_id',
           'Адрес от начала участка', 'Наработка до отказа(new), лет', 'R', 'D', 'L', 'S', 'a', 'b', 'L,м',
           'Предсказуемы из модели с подтверждением',
           'Вышли в ' + str(year) + ' по ремонту или замене', 'Первичный отказ']].to_excel(
        path + 'Аварии (настроенный) ' + str(year) + ' года.xlsx')
    data.loc[0, 'Всего в ' + str(year)] = iyear.shape[0]
    data.loc[0, 'Первичных'] = iyear[iyear['Первичный отказ'] == True].shape[0]
    data.loc[0, 'Повторных'] = iyear[iyear['Первичный отказ'] == False].shape[0]
    data.loc[0, 'Повторных >20м'] = iyear[(iyear['Первичный отказ'] == False) & (iyear['L,м'] >= 20)].shape[0]
    data.loc[0, 'Вышли по ремонту в ' + str(year)] = iyear[
        (iyear['Вышли в ' + str(year) + ' по ремонту или замене'] == True) & (iyear['Первичный отказ'] == False) & (
                    iyear['L,м'] >= 20)].shape[0]
    # data.loc[0,'Предсказуемы из модели']=iyear[iyear['Предсказуемы из модели с подтверждением']==True].shape[0]
    data.loc[0, 'Предсказуемы из модели'] = len(IDXES)
    data.loc[0, 'Введены в эксплуатацию в ' + str(year)] = \
    iyear[(iyear['Первичный отказ'] == False) & (iyear['L,м'] >= 20) & (iyear['Дата ввода'].dt.year == year)].shape[0]
    data.loc[0, 'Ожидаемый прогноз'] = data.loc[0, 'Повторных >20м'] - data.loc[0, 'Вышли по ремонту в ' + str(year)] - \
                                       data.loc[0, 'Введены в эксплуатацию в ' + str(year)]
    return data


def GetYearInalysis_v1(sdata, tdata, cdata, year=2016, offset=1, target='future_ads_r', index_field="future_ads_r"):
    data = pd.DataFrame(index=[year], columns=['Всего', 'Первичных', 'Повторных', 'Повторных >20м',
                                               'Вышли по ремонту', 'Предсказуемы из модели', 'Предсказано с под-ем',
                                               'Введены в эксплуатацию', 'Ожидаемый прогноз', 'Спрогнозировано'])
    # iyear=tdata[tdata['Дата аварии'].dt.year==year]
    # iy=cdata[cdata['Дата аварии'].dt.year==year]
    iyear = tdata[(tdata['Дата аварии'] >= pd.to_datetime('01-01-' + str(year))) & (
                tdata['Дата аварии'] < pd.to_datetime('01-01-' + str(year + offset)))]
    iy = cdata[(cdata['Дата аварии'] >= pd.to_datetime('01-01-' + str(year))) &
               (cdata['Дата аварии'] < pd.to_datetime('01-01-' + str(year + offset)))]
    IDXES = GetIndexes(sdata['all'][sdata['all'][target] >= 0], field=index_field)
    withfact = iyear[iyear.index.isin(IDXES)]
    iyear['Предсказуемы из модели с подтверждением'] = False
    iyear.loc[withfact.index, 'Предсказуемы из модели с подтверждением'] = True
    alone = []
    for ind in iy['index'].value_counts().keys():
        someind = iy[iy['index'] == ind]
        if GetAloneIndexes(someind):
            alone.append(ind)
    iyear['Первичный отказ'] = False
    iyear.loc[iyear['original_index'].isin(alone), 'Первичный отказ'] = True
    iyear['Вышли в ' + str(year) + ' по ремонту или замене'] = False
    iyear.loc[iyear[(iyear['Дата перевода в бездействие'] >= pd.to_datetime('01-01-' + str(year))) &
                    (iyear['Дата перевода в бездействие'] < pd.to_datetime('01-01-' + str(year + offset)))].index,
              'Вышли в ' + str(year) + ' по ремонту или замене'] = True
    iyear[['ID простого участка', 'Месторождение', 'Дата ввода',
           'Дата перевода в бездействие', 'Дата аварии', 'Наработка до отказа', 'new_id',
           'Адрес от начала участка', 'Наработка до отказа(new), лет', 'R', 'D', 'L', 'S', 'a', 'b', 'L,м',
           'Предсказуемы из модели с подтверждением',
           'Вышли в ' + str(year) + ' по ремонту или замене', 'Первичный отказ']].to_excel(
        path + 'Аварии ' + str(year) + ' года.xlsx')
    data.loc[year, 'Всего'] = iyear.shape[0]
    data.loc[year, 'Первичных'] = iyear[iyear['Первичный отказ'] == True].shape[0]
    data.loc[year, 'Повторных'] = iyear[iyear['Первичный отказ'] == False].shape[0]
    data.loc[year, 'Повторных >20м'] = iyear[(iyear['Первичный отказ'] == False) & (iyear['L,м'] >= 20)].shape[0]
    data.loc[year, 'Вышли по ремонту'] = iyear[
        (iyear['Вышли в ' + str(year) + ' по ремонту или замене'] == True) & (iyear['Первичный отказ'] == False) & (
                    iyear['L,м'] >= 20)].shape[0]
    # data.loc[0,'Предсказуемы из модели']=iyear[iyear['Предсказуемы из модели с подтверждением']==True].shape[0]
    data.loc[year, 'Предсказуемы из модели'] = len(IDXES)
    mask = (iyear['Дата ввода'] >= pd.to_datetime('01-01-' + str(year))) & (
                iyear['Дата ввода'] < pd.to_datetime('01-01-' + str(year + offset)))
    data.loc[year, 'Введены в эксплуатацию'] = \
    iyear[(iyear['Первичный отказ'] == False) & (iyear['L,м'] >= 20) & mask].shape[0]
    data.loc[year, 'Ожидаемый прогноз'] = data.loc[year, 'Повторных >20м'] - data.loc[year, 'Вышли по ремонту'] - \
                                          data.loc[year, 'Введены в эксплуатацию']
    return data


def GetMainFeatures(data, extra=pd.DataFrame(), parameters=dict()):
    if len(parameters.keys()) == 0:
        return dict()
    feats = GetFeatures(data, extra=extra, seed=parameters['seed'], target=parameters['target'],
                        split_by=parameters['split_by'], split=parameters['split'], by_date=parameters['by_date'],
                        date_column=parameters['date_column'], loglist=parameters['loglist'],
                        indexes=parameters['indexes'], drop_end=parameters['drop_end'], dummies=parameters['dummies'],
                        tail=parameters['tail'], drop_after=parameters['drop_after'],
                        drop_before=parameters['drop_before'], fields=parameters['fields'])

    ext = None
    if len(feats) > 3:
        train = feats[0]
        test = feats[1]
        control = feats[2]
        ext = feats[3]
    else:
        train = feats[0]
        test = feats[1]
        control = feats[2]

    X = train[0]
    y = train[1]
    F = train[2]
    Xt = test[0]
    yt = test[1]
    Ft = test[2]
    p = F['classes']
    pt = Ft['classes']
    FDUM = pd.DataFrame(columns=['classes'])
    FDUM = FDUM.append(F[['classes']])
    FDUM = FDUM.append(Ft[['classes']])
    DUMMY = pd.get_dummies(FDUM['classes'])
    Dp = DUMMY.loc[DUMMY.index.isin(F.index)]
    Dpt = DUMMY.loc[DUMMY.index.isin(Ft.index)]
    G = F.copy()
    Gt = Ft.copy()
    G.drop(['classes'], axis=1, inplace=True)
    Gt.drop(['classes'], axis=1, inplace=True)
    Q = G.values
    Qt = Gt.values
    Features = dict({'Regression': {'TRAIN': {'X': train[0], 'y': train[1], 'F': train[2]},
                                    'TEST': {'X': test[0], 'y': test[1], 'F': test[2]}},
                     'Classification': {'TRAIN': {'X': Q, 'y': p,'F':G}, 'TEST': {'X': Qt, 'y': pt,'F':Gt}},
                     'DUMMY': {'TRAIN': {'y': Dp, 'X': Q}, 'TEST': {'y': Dpt, 'X': Qt}},
                     'CONTROL': control})
    if ext is not None:
        Features.update({'EXTRA': {'X': ext[0], 'y': ext[1], 'F': ext[2], 'CONTROL': ext[3]}})


    return Features


def GetBalancedWeights(p):
    bc = np.bincount(p)
    k = len(bc)
    n_samples = p.shape[0]
    weights = [n_samples / (x * k) for x in bc]
    return {k: v for k, v in zip(np.arange(k), weights)}


def GetAloneIndexes(data):
    if data['interval_ads'].prod() == 1:
        return True
    else:
        return False


def PlotIDErrors(result, interval=None, title=''):
    plt.figure(figsize=(12, 8))
    if interval is not None:
        mask = result['interval'] == interval
    else:
        mask = result['interval'] == result['interval']
        interval = 'all'
    bins = plt.hist(result[mask]['error'], bins=50)
    rang = np.round([(bins[1][x] + bins[1][x - 1]) / 2 for x in np.arange(1, len(bins[1]))], 3)
    percents = bins[0] * 100 / np.sum(bins[0])
    array = np.array([rang, bins[0], percents])
    a = min(np.abs(rang))
    instead = np.where(np.abs(array[0]) != a)
    span = array[0][instead[0]]
    eta = array[2][np.where((np.abs(array[0]) != a) & (array[0] > 0))]
    tau = array[2][np.where((np.abs(array[0]) != a) & (array[0] < 0))]
    positive = np.sum(eta)
    negative = np.sum(tau)
    for x, y, label in zip(rang, bins[0], percents):
        if y > 0:
            plt.annotate('{0:.1f}'.format(label), xy=(x, y))
    # rspan=[span[x] for x in np.where(np.array(span)>=0)[0]]
    rspan = span[np.where(span > 0)]
    lspan = span[np.where(span < 0)]
    lbins = array[1][np.where((np.abs(array[0]) != a) & (array[0] < 0))]
    rbins = array[1][np.where((np.abs(array[0]) != a) & (array[0] > 0))]
    if rspan.shape[0] * rbins.shape[0] > 0:
        plt.bar((min(rspan) + max(rspan)) / 2, max(rbins), max(rspan) - min(rspan), alpha=0.3, color='green')
        plt.annotate('{0:.1f}'.format(positive), xy=((min(rspan) + max(rspan)) / 2, max(rbins) + 1))
    if lspan.shape[0] * lbins.shape[0] > 0:
        plt.bar((min(lspan) + max(lspan)) / 2, max(lbins), max(lspan) - min(lspan), alpha=0.3, color='red')
        plt.annotate('{0:.1f}'.format(negative), xy=((min(lspan) + max(lspan)) / 2, max(lbins) + 1))
    # print(positive+negative)
    plt.xlabel('interval ' + str(interval))
    plt.ylabel('% ')
    plt.title(title)


def PlotHistogram(result, title='target', xlabel='target', bins=50):
    plt.figure(figsize=(12, 8))
    bins = plt.hist(result, bins=50)
    rang = np.round([(bins[1][x] + bins[1][x - 1]) / 2 for x in np.arange(1, len(bins[1]))], 3)
    percents = bins[0] * 100 / np.sum(bins[0])
    array = np.array([rang, bins[0], percents])
    a = min(np.abs(rang))
    instead = np.where(np.abs(array[0]) != a)
    span = array[0][instead[0]]
    eta = array[2][np.where((np.abs(array[0]) != a) & (array[0] > 0))]
    tau = array[2][np.where((np.abs(array[0]) != a) & (array[0] < 0))]
    positive = np.sum(eta)
    negative = np.sum(tau)
    for x, y, label in zip(rang, bins[0], percents):
        if y > 0:
            plt.annotate('{0:.1f}'.format(label), xy=(x, y))
    # rspan=[span[x] for x in np.where(np.array(span)>=0)[0]]
    rspan = span[np.where(span > 0)]
    lspan = span[np.where(span < 0)]
    lbins = array[1][np.where((np.abs(array[0]) != a) & (array[0] < 0))]
    rbins = array[1][np.where((np.abs(array[0]) != a) & (array[0] > 0))]
    if rspan.shape[0] * rbins.shape[0] > 0:
        plt.bar((min(rspan) + max(rspan)) / 2, max(rbins), max(rspan) - min(rspan), alpha=0.3, color='green')
        plt.annotate('{0:.1f}'.format(positive), xy=((min(rspan) + max(rspan)) / 2, max(rbins) + 1))
    if lspan.shape[0] * lbins.shape[0] > 0:
        plt.bar((min(lspan) + max(lspan)) / 2, max(lbins), max(lspan) - min(lspan), alpha=0.3, color='red')
        plt.annotate('{0:.1f}'.format(negative), xy=((min(lspan) + max(lspan)) / 2, max(lbins) + 1))
    # print(positive+negative)
    plt.xlabel(title)
    plt.ylabel('% ')
    plt.title(title)
class Fields:
    def __init__(self,path='D:\\ml\\',file='fields_dict.npy'):
        self.dictionary=np.load(path + file, allow_pickle=True).item()


def load_raw(file='Отказы общий.xlsx',path='D:\\ml\\',today=np.datetime64('2020-07-27')):
    data = pd.read_excel(path + file)
    data['original_index'] = data.index
    sd1 = data[data['Длина ремонтируемого участка'] < 2]
    sd2 = data[data['Длина ремонта до аварии'] < 2]
    data.loc[sd1.index, 'Способ ликв в момент'] = 'Установка ВГУ'
    data.loc[sd2.index, 'Способ ликв в момент'] = 'Установка ВГУ'
    data['Дата перевода в бездействие'].fillna(today, inplace=True)
    return data


def make_intervals(data, ints=[100], date=[1], expand=True, ident='ID простого участка', sharedid='ID простого участка',
                   scale=2,function=fill_intervals):
    Data=None
    # maingroup=data[data[ident]==idt]
    ID = data[sharedid].value_counts()
    k = 0
    for idt in ID.keys():
        group = data[data[sharedid] == idt]
        synthetic = get_unical_repairs(group,scale=scale)
        true = GetUnicalRepairs(group,scale=scale)
        repairs = get_merged_repairs(true, synthetic, epsilon=0.5)
        #repairs['b'] = repairs['Адрес'] + repairs['Длина']
        #rep = repairs[['Адрес', 'b', 'Дата ремонта']].values
        #repairs = GetUnicalRepairs(group, scale=scale)
        reps = RepairsValues(repairs)

        for subid in group[ident].value_counts().keys():
            try:
                subgroup = group[group[ident] == subid]
                getin = subgroup['Дата ввода'].value_counts().keys()[0]
                a = subgroup['a'].value_counts().keys()[0]
                b = subgroup['b'].value_counts().keys()[0]
                #print(a,',',b)
                subrep = reps[np.where((reps[:, 2] > getin) & ((reps[:, 0] >= a) & (reps[:, 1] <= b)))]
            # print(subrep)
                df = split_intervals(subgroup, ints, date, expand, repairs=subrep,function=function)
                if k == 0:
                    Data = df.copy()
                    #print(Data.shape)
                    k = k + 1
                else:
                    Data = Data.append(df, ignore_index=True)



            except IndexError:
                print(ident,' ',subid)


    return Data


def fill_intervals(group, index, k, teta, date=1, expand=True, repairs=None):
    group = group.sort_values(by=['Дата аварии'])
    current_ad = group.loc[index, 'Дата аварии']
    current_year = pd.to_datetime(current_ad).year
    next_year = pd.to_datetime('01-01-' + str(current_year + date))
    beginofnext = pd.to_datetime('01-01-' + str(current_year + 1))
    current_delta = (beginofnext - current_ad) / np.timedelta64(1, 'D')
    getin = group.loc[index, 'Дата ввода']
    dateout = group.loc[index, 'Дата перевода в бездействие']
    current_R = group.loc[index, 'R']
    current_s = group.loc[index, 'S']
    left = group.loc[index, 'a']
    # right=group.loc[index,'b']
    current_age = group.loc[index, 'Наработка до отказа(new), лет']
    getout = (dateout - getin) / np.timedelta64(1, 'Y')
    getout_r = getout / current_s
    group['r'] = group['Наработка до отказа(new), лет'] / group['S']
    current_r = group.loc[index, 'r']
    columns = list(group.columns)
    columns.append('interval')
    columns.append('interval_ads')
    columns.append('int_ads_halfyear')
    columns.append('int_ads_oneyear')
    columns.append('int_ads_twoyear')
    columns.append('int_ads_threeyear')
    columns.append('future_ads')
    columns.append('index')
    columns.append('last_ads')
    columns.append('first_ads')
    columns.append('interval_ads_r')
    columns.append('future_ads_r1')
    columns.append('future_ads_r')
    columns.append('future_calendar')
    columns.append('delta_days')
    columns.append('extra_lenght')
    columns.append('lbound')
    columns.append('rbound')
    previous_ads = group[group['Дата аварии'] <= current_ad]
    previous_ads_r = group[group['r'] <= current_r]
    future_ads_r = group[group['r'] > current_r]
    future_ads = group[group['Дата аварии'] > current_ad]
    if dateout >= next_year:
        calendar_year = future_ads[future_ads['Дата аварии'] < next_year]
    else:
        calendar_year = None
    if getout >= (current_age + date):
        future_ads['delta'] = (future_ads['Дата аварии'] - current_ad) / np.timedelta64(1, 'Y')
        future_ads = future_ads[future_ads['delta'] <= date]

    else:
        future_ads = None

    if getout_r >= (current_R + date * 0.125):
        future_ads_r1 = future_ads_r[future_ads_r['r'] <= (current_R + date * 0.125)]
    else:
        future_ads_r1 = None
    if getout_r >= current_r + date / current_s:
        future_ads_r2 = future_ads_r[future_ads_r['r'] <= current_r + date * 0.125]
    else:
        future_ads_r2 = None
    # group['belong']=True
    lenght = group.loc[index, 'L,м']
    field = 'Адрес от начала участка (new)'
    current_point = group.loc[index, field]
    if not expand:
        l = int(lenght / (2 * teta))
        k = min(k, l)

    # columns.append('period')
    df = pd.DataFrame(index=np.arange(0, k), columns=columns)
    df[group.columns] = group.loc[index, group.columns].values.reshape(-1, group.shape[1])
    df['extra_lenght'] = 0.0
    # display(df['extra_lenght'])
    # print(date)

    for ind in np.arange(0, k):
        # df.loc[ind]=group.loc[index]
        df.loc[ind, 'interval'] = (ind + 1) * teta
        a, b = get_interval(teta, (ind + 1), current_point, lenght, expand=expand)
        df.loc[ind, 'lbound'] = a
        df.loc[ind, 'rbound'] = b
        # print('repairs shape',repairs.shape)
        if repairs is not None:
            if repairs.shape[0] > 0:
                # print(repairs)
                # print(a,'',b)
                # print('current', current_point)
                extra_lenght = GetExtraLenght(repairs=repairs, current_point=current_point, date=current_ad, a=a, b=b,
                                              tilde=left)
                # print('extra_lenght=',extra_lenght)
                df.loc[ind, 'extra_lenght'] = extra_lenght
        # print(df.loc[ind,'extra_lenght'])
        local_prev = previous_ads[(previous_ads[field] >= a) & (previous_ads[field] <= b)]
        local_prev_r = previous_ads_r[(previous_ads_r[field] >= a) & (previous_ads_r[field] <= b)]
        df.loc[ind, 'interval_ads'] = local_prev.shape[0]
        df.loc[ind, 'interval_ads_r'] = local_prev_r.shape[0]
        halfy = current_ad - pd.DateOffset(months=6)
        oney = current_ad - pd.DateOffset(years=1)
        twoy = current_ad - pd.DateOffset(years=2)
        threey = current_ad - pd.DateOffset(years=3)
        local_prev_05 = local_prev[local_prev['Дата аварии'] >= halfy]
        local_prev_1 = local_prev[local_prev['Дата аварии'] >= oney]
        local_prev_2 = local_prev[local_prev['Дата аварии'] >= twoy]
        local_prev_3 = local_prev[local_prev['Дата аварии'] >= threey]
        df.loc[ind, 'int_ads_halfyear'] = local_prev_05.shape[0]
        df.loc[ind, 'int_ads_oneyear'] = local_prev_1.shape[0]
        df.loc[ind, 'int_ads_twoyear'] = local_prev_2.shape[0]
        df.loc[ind, 'int_ads_threeyear'] = local_prev_3.shape[0]
        local_prev_less = local_prev[local_prev['Дата аварии'] < current_ad]
        if local_prev_less.shape[0] == 0:
            df.loc[ind, 'last_ads'] = current_ad
            df.loc[ind, 'first_ads'] = current_ad
        else:
            df.loc[ind, 'last_ads'] = local_prev_less.loc[local_prev_less.iloc[-1].name, 'Дата аварии']
            df.loc[ind, 'first_ads'] = local_prev_less.loc[local_prev_less.iloc[0].name, 'Дата аварии']

        if future_ads is not None:
            local_fut = future_ads[(future_ads[field] >= a) & (future_ads[field] <= b)]
            df.loc[ind, 'future_ads'] = local_fut.shape[0]
            # print('future_ads is not none. shape=',local_fut.shape[0])
        else:
            df.loc[ind, 'future_ads'] = np.nan
        if future_ads_r1 is not None:
            local_fut_r1 = future_ads_r1[(future_ads_r1[field] >= a) & (future_ads_r1[field] <= b)]
            df.loc[ind, 'future_ads_r'] = local_fut_r1.shape[0]
        else:
            df.loc[ind, 'future_ads_r'] = np.nan
        if future_ads_r2 is not None:
            local_fut_r2 = future_ads_r2[(future_ads_r2[field] >= a) & (future_ads_r2[field] <= b)]
            df.loc[ind, 'future_ads_r1'] = local_fut_r2.shape[0]
        else:
            df.loc[ind, 'future_ads_r1'] = np.nan
        if calendar_year is not None:
            local_year = calendar_year[(calendar_year[field] >= a) & (calendar_year[field] <= b)]
            df.loc[ind, 'future_calendar'] = local_year.shape[0]
        else:
            df.loc[ind, 'future_calendar'] = np.nan
        df.loc[ind, 'delta_days'] = current_delta
        # df.loc[ind,'future_ads']=local_fut.shape[0]
        # df.loc[ind,'future_ads_r']=local_fut_r1.shape[0]
        # df.loc[ind,'future_ads_r1']=local_fut_r2.shape[0]
        df.loc[ind, 'index'] = index

        # print('a=',a,',b=',b)

    # df['period']=np.nan
    df['period'] = date
    # display(df['extra_lenght'])
    return df


def GetSplitedByRepairs(data, ID):
    group = data[data['ID простого участка'] == ID]
    repairs = GetUnicalRepairs(group, scale=2)
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
        indexes = mask1[mask1 == True].keys()
        data.loc[indexes, 'Дата ввода'] = rd
        data.loc[indexes, 'a'] = a
        data.loc[indexes, 'b'] = b
        data.loc[indexes, 'new_id'] = str(ID) + '_' + str(i + 1)
        data.loc[mask[mask == True].keys(), 'Дата перевода в бездействие'] = rd
        data.loc[mask[mask == True].keys(), 'Состояние'] = 'Бездействующий'
        repgroup = group[mask]
        if i > 0:
            A = GetRepairsMap(rep, i - 1)
            T = GetSetsResidual(A, X.reshape(3), f=interseption)[:-1]
            for t in T:
                submask = (repgroup['Дата аварии'] >= t[2]) & ((repgroup['Адрес от начала участка'] <= t[1]) & (
                            repgroup['Адрес от начала участка'] >= t[0]))
                mask = mask | submask
                indexes1 = submask[submask == True].keys()
                data.loc[indexes1, 'Дата ввода'] = t[2]
                data.loc[indexes1, 'Дата перевода в бездействие'] = rd
                data.loc[indexes1, 'Состояние'] = 'Бездействующий'

        subgroup = repgroup[mask]
        group = group[~mask]
    return data

def get_splited_by_repairs(data, ID,delta=3):
    group = data[data['ID простого участка'] == ID]
    synthetic = get_unical_repairs(group)
    true=GetUnicalRepairs(group)
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
        indexes = mask1[mask1 == True].keys()
        #data.loc[indexes, 'Дата ввода'] = rd
        data.loc[indexes, 'a'] = a
        data.loc[indexes, 'b'] = b
        data.loc[indexes, 'new_id'] = str(ID) + '_' + str(i + 1)
        data.loc[mask[mask == True].keys(), 'Дата перевода в бездействие'] = rd
        data.loc[mask[mask == True].keys(), 'Состояние'] = 'Бездействующий'
        repgroup = group[mask]
        if i > 0:
            A = GetRepairsMap(rep, i - 1)
            T = GetSetsResidual(A, X.reshape(3), f=interseption)[:-1]
            for t in T:
                submask = (repgroup['Дата аварии'] >= t[2]) & ((repgroup['Адрес от начала участка'] <= t[1]) & (
                            repgroup['Адрес от начала участка'] >= t[0]))
                mask = mask | submask
                indexes1 = submask[submask == True].keys()
                dlt=(rd-t[2])/np.timedelta64(1,'Y')
                #print(dlt)
                if dlt>=delta:
                    data.loc[indexes1, 'Дата ввода'] = t[2]
                data.loc[indexes1, 'Дата перевода в бездействие'] = rd
                data.loc[indexes1, 'Состояние'] = 'Бездействующий'

        #subgroup = repgroup[mask]
        group = group[~mask]
    le = group['Дата перевода в бездействие'] < group['Дата аварии']
    group.loc[le,'Дата перевода в бездействие']=np.nan
    empty = np.isnan(group['Дата перевода в бездействие'])
    submask = ((group['Состояние'] == 'Бездействующий') | (group['Состояние'] == 'Демонтирован'))&empty
    indexes1 = submask[submask == True].keys()
    rd = group[submask]['Дата аварии'].max()
    data.loc[indexes1, 'Дата перевода в бездействие'] = rd
    if repairs.shape[0]>0:
        A = GetRepairsMap(rep, rep.shape[0] - 1)
        for t in A:
            submask = (group['Дата аварии'] >t[2]) & ((group['Адрес от начала участка'] <= t[1]) & (
                group['Адрес от начала участка'] >= t[0])&((group['Состояние']=='Бездействующий')|(group['Состояние']=='Демонтирован')))
            indexes1 = submask[submask == True].keys()
            rd=group[submask]['Дата аварии'].max()
            dlt = (rd - t[2]) / np.timedelta64(1, 'Y')
            if dlt >= delta:
               data.loc[indexes1, 'Дата ввода'] = t[2]
            data.loc[indexes1, 'Дата перевода в бездействие'] = rd
            #data.loc[indexes1, 'Состояние'] = 'Бездействующий'
    return data
def SplitByRepairs(cdata,save_to='Отказы после ремонта_v7',path='D:\\ml\\'):
    cdata['a'] = 0
    cdata['b'] = cdata['L']
    cdata['new_id'] = cdata['ID простого участка']
    data = pd.DataFrame(columns=cdata.columns)
    for ID in cdata['ID простого участка'].value_counts().keys():
        someid = cdata[cdata['ID простого участка'] == ID]
        splited = GetSplitedByRepairs(someid, ID=ID)
        data = data.append(splited)
    data['L,м'] = data['b'] - data['a']
    data['Адрес от начала участка (new)'] = data['Адрес от начала участка'] - data['a']
    data['new_id'] = data['new_id'].astype('str')
    data['Наработка до отказа(new), лет'] = (data['Дата аварии'] - data['Дата ввода']) / np.timedelta64(1, 'Y')
    data['R'] = (np.modf((data['Наработка до отказа(new), лет'] / data['S']) / 0.125)[1]) * 0.125
    data.to_csv(path + save_to+'.csv')
    return data
def binary_fill_intervals(group, index, k, teta, date=1, expand=True, repairs=None, append=['interval','interval_ads','int_ads_halfyear','int_ads_oneyear','int_ads_twoyear',
     'int_ads_threeyear','future_ads','index','last_place','last_ads','first_ads','interval_ads_r',
     'future_ads_r1','future_ads_r','future_calendar','delta_days','extra_lenght',
     'lbound','rbound']):
    group = group.sort_values(by=['Дата аварии'])
    current_ad = group.loc[index, 'Дата аварии']
    current_year = pd.to_datetime(current_ad).year
    next_year = pd.to_datetime('01-01-' + str(current_year + date))
    beginofnext = pd.to_datetime('01-01-' + str(current_year + 1))
    current_delta = (beginofnext - current_ad) / np.timedelta64(1, 'D')
    getin = group.loc[index, 'Дата ввода']
    dateout = group.loc[index, 'Дата перевода в бездействие']
    current_R = group.loc[index, 'R']
    current_s = group.loc[index, 'S']
    halfy = current_ad - pd.DateOffset(months=6)
    oney = current_ad - pd.DateOffset(years=1)
    twoy = current_ad - pd.DateOffset(years=2)
    threey = current_ad - pd.DateOffset(years=3)
    left = group.loc[index, 'a']
    current_age = group.loc[index, 'Наработка до отказа(new), лет']
    getout = (dateout - getin) / np.timedelta64(1, 'Y')
    getout_r = getout / current_s
    group['r'] = group['Наработка до отказа(new), лет'] / group['S']
    current_r = group.loc[index, 'r']
    columns = list(group.columns)
    #columns.append(append)
    for x in append:
        columns.append(x)
    previous_ads = group[group['Дата аварии'] <= current_ad]
    previous_ads_r = group[group['r'] <= current_r]
    future_ads_r = group[group['r'] > current_r]
    future_ads = group[group['Дата аварии'] > current_ad]
    future_ads['delta'] = (future_ads['Дата аварии'] - current_ad) / np.timedelta64(1, 'Y')


    if dateout >= next_year:
        calendar_year = future_ads[future_ads['Дата аварии'] < next_year]
    else:
        calendar_year = None
        we_calendar_year=future_ads[future_ads['Дата аварии'] < next_year]


    if getout >= (current_age + date):
        future_ads_y = future_ads[future_ads['delta'] <= date]
    else:
        future_ads_y = None
        we_future_ads_y = future_ads[future_ads['delta'] <= date]

    if getout_r >= (current_R + date * 0.125):
        future_ads_r1 = future_ads_r[future_ads_r['r'] <= (current_R + date * 0.125)]
    else:
        future_ads_r1 = None
        we_future_ads_r1 = future_ads_r[future_ads_r['r'] <= (current_R + date * 0.125)]
    if getout_r >= current_r + date / current_s:
        future_ads_r2 = future_ads_r[future_ads_r['r'] <= current_r + date * 0.125]
    else:
        future_ads_r2 = None
        we_future_ads_r2 = future_ads_r[future_ads_r['r'] <= current_r + date * 0.125]
    # group['belong']=True
    lenght = group.loc[index, 'L,м']
    field = 'Адрес от начала участка (new)'
    current_point = group.loc[index, field]
    if not expand:
        l = int(lenght / (2 * teta))
        k = min(k, l)

    # columns.append('period')
    df = pd.DataFrame(index=np.arange(0, k), columns=columns)
    df[group.columns] = group.loc[index, group.columns].values.reshape(-1, group.shape[1])
    df['extra_lenght'] = 0.0
    ind=k
    while ind>0:
        a, b = get_interval(teta, (ind + 1), current_point, lenght, expand=expand)
        df.loc[ind, 'lbound'] = a
        df.loc[ind, 'rbound'] = b
        df.loc[ind, 'interval'] = (ind + 1) * teta
        df.loc[ind, 'index'] = index
        if repairs is not None:
            if repairs.shape[0] > 0:
                extra_lenght = GetExtraLenght(repairs=repairs, current_point=current_point, date=current_ad, a=a, b=b,
                                              tilde=left)
                df.loc[ind, 'extra_lenght'] = extra_lenght
        local_prev_mask = (previous_ads[field] >= a) & (previous_ads[field] <= b)
        local_prev_r_mask = (previous_ads_r[field] >= a) & (previous_ads_r[field] <= b)
        local_prev_count=local_prev_mask[local_prev_mask==True].shape[0]
        df.loc[ind, 'interval_ads'] = local_prev_count
        df.loc[ind, 'interval_ads_r'] = local_prev_r_mask[local_prev_r_mask==True].shape[0]
        local_prev = previous_ads[local_prev_mask]
        if local_prev_count>1:
            local_prev_3_mask = local_prev['Дата аварии'] >= threey
            loc3=local_prev_3_mask[local_prev_3_mask==True].shape[0]
            df.loc[ind, 'int_ads_threeyear'] = loc3
            if loc3>1:
                local_prev_2_mask = local_prev['Дата аварии'] >= twoy
                loc2=local_prev_2_mask[local_prev_2_mask==True].shape[0]
                df.loc[ind, 'int_ads_twoyear'] = loc2
                if loc2>1:
                    local_prev_1_mask = local_prev['Дата аварии'] >= oney
                    loc1=local_prev_1_mask[local_prev_1_mask==True].shape[0]
                    df.loc[ind, 'int_ads_oneyear'] = loc1
                    if loc1>1:
                        local_prev_05_mask = local_prev['Дата аварии'] >= halfy
                        loc05=local_prev_05_mask[local_prev_05_mask==True].shape[0]
                        df.loc[ind, 'int_ads_halfyear'] = loc05
                    else:
                        df.loc[ind, 'int_ads_halfyear'] = 1

                else:
                    df.loc[ind, 'int_ads_halfyear'] = 1
                    df.loc[ind, 'int_ads_oneyear'] = 1
            else:
                df.loc[ind, 'int_ads_halfyear'] = 1
                df.loc[ind, 'int_ads_oneyear'] = 1
                df.loc[ind, 'int_ads_twoyear'] = 1
        else:
            df.loc[ind, 'int_ads_halfyear'] = 1
            df.loc[ind, 'int_ads_oneyear'] = 1
            df.loc[ind, 'int_ads_twoyear'] = 1
            df.loc[ind, 'int_ads_threeyear'] = 1

        local_prev_less = local_prev[local_prev['Дата аварии'] < current_ad]
        if local_prev_less.shape[0] == 0:
            df.loc[ind, 'last_ads'] = current_ad
            df.loc[ind, 'first_ads'] = current_ad
            df.loc[ind,'last_place']=current_point
        else:
            df.loc[ind, 'last_ads'] = local_prev_less.loc[local_prev_less.iloc[-1].name, 'Дата аварии']
            df.loc[ind, 'first_ads'] = local_prev_less.loc[local_prev_less.iloc[0].name, 'Дата аварии']
            df.loc[ind, 'last_place'] = local_prev_less.loc[local_prev_less.iloc[-1].name, field]

        if future_ads_y is not None:
            local_fut = (future_ads_y[field] >= a) & (future_ads_y[field] <= b)
            locf=local_fut[local_fut==True].shape[0]
            if locf>0:
                df.loc[ind, 'future_ads'] = 1
            else:
                df.loc[ind, 'future_ads'] = 0
            # print('future_ads is not none. shape=',local_fut.shape[0])
        else:
            local_fut = (we_future_ads_y[field] >= a) & (we_future_ads_y[field] <= b)
            locf=local_fut[local_fut==True].shape[0]
            if locf>0:
                df.loc[ind, 'future_ads'] = 1
            else:
                df.loc[ind, 'future_ads'] = np.nan

        if future_ads_r1 is not None:
            local_fut_r1 = (future_ads_r1[field] >= a) & (future_ads_r1[field] <= b)
            locf_r=local_fut_r1[local_fut_r1==True].shape[0]
            if locf_r>0:
                df.loc[ind, 'future_ads_r'] = 1
            else:
                df.loc[ind, 'future_ads_r'] = 0
        else:
            local_fut_r1 = (we_future_ads_r1[field] >= a) & (we_future_ads_r1[field] <= b)
            locf_r=local_fut_r1[local_fut_r1==True].shape[0]
            if locf_r>0:
                df.loc[ind, 'future_ads_r'] =1
            else:
                df.loc[ind, 'future_ads_r'] = np.nan
        if future_ads_r2 is not None:
            local_fut_r2 = (future_ads_r2[field] >= a) & (future_ads_r2[field] <= b)
            locf_r1=local_fut_r2[local_fut_r2==True].shape[0]
            if locf_r1>0:
                df.loc[ind, 'future_ads_r1'] = 1
            else:
                df.loc[ind, 'future_ads_r1'] = 0

        else:
            local_fut_r2 = (we_future_ads_r2[field] >= a) & (we_future_ads_r2[field] <= b)
            locf_r1=local_fut_r2[local_fut_r2==True].shape[0]
            if locf_r1>0:
                df.loc[ind, 'future_ads_r1'] = 1
            else:
                df.loc[ind, 'future_ads_r1'] = np.nan

        if calendar_year is not None:
            local_year = (calendar_year[field] >= a) & (calendar_year[field] <= b)
            locf_c=local_year[local_year==True].shape[0]
            if locf_c>0:
                df.loc[ind, 'future_calendar'] = 1
            else:
                df.loc[ind, 'future_calendar'] =0
        else:
            local_year = (we_calendar_year[field] >= a) & (we_calendar_year[field] <= b)
            locf_c = local_year[local_year == True].shape[0]
            if locf_c > 0:
                df.loc[ind, 'future_calendar'] = 1
            else:
                df.loc[ind, 'future_calendar'] = np.nan

        df.loc[ind, 'delta_days'] = current_delta
        ind=ind-1

    df['period'] = date
    # display(df['extra_lenght'])
    return df
def get_tensors(data,ident='new_id',expand=False,ints=[100],date=[1],bin_count=20):
    data.sort_values(by='Наработка до отказа(new), лет',inplace=True)
    aggdata=data.groupby('new_id')
    time_range=get_time_range_v1()
    npints=np.array(ints)*2
    L=[]
    for i,group in enumerate(aggdata):
        length=group[1]['L,м'].iloc[0]
        mask=npints<=length
        n=mask[mask==True].shape[0]
        if n>0:
            for j in group[1].index:
                for d in date:
                    tensor=binary_tensor_fill(group[1],index=j,ints=ints,expand=expand,time_range=time_range,date=d)
                    if tensor is not None:
                        L.append(tensor)
                    else:
                        print('empty id',group[0])
    return np.vstack(L)



def binary_recurrent(group, index, ints=[100], date=1, expand=True, time_range=np.array([]), n=15):
    #group = group.sort_values(by=['Дата аварии'])
    current_ad = group.loc[index, 'Дата аварии']
    getin = group.loc[index, 'Дата ввода']
    dateout = group.loc[index, 'Дата перевода в бездействие']
    current_s = group.loc[index, 'S']
    current_w = group.loc[index, 'Обводненность']
    #left = group.loc[index, 'a']
    current_age = group.loc[index, 'Наработка до отказа(new), лет']
    getout = (dateout - getin) / np.timedelta64(1, 'Y')
    group['r'] = group['Наработка до отказа(new), лет'] / group['S']
    previous_ads = group[group['Дата аварии'] <= current_ad]
    future_ads = group[group['Дата аварии'] > current_ad]
    future_ads['delta'] = (future_ads['Дата аварии'] - current_ad) / np.timedelta64(1, 'Y')
    water_mean=previous_ads['Обводненность'].mean()
    age_mean = previous_ads['Наработка до отказа(new), лет'].mean()

    if getout >= (current_age + date):
        future_ads_y = future_ads[future_ads['delta'] <= date]
    else:
        future_ads_y = None
        we_future_ads_y = future_ads[future_ads['delta'] <= date]

    lenght = group.loc[index, 'L,м']
    field = 'Адрес от начала участка (new)'
    current_point = group.loc[index, field]
    k=len(ints)
    ints.sort()
    short = ints

    if not expand:
        for k,teta in enumerate(ints):
            l = int(lenght / (2 * teta))
            if l==0:
              if k==0: return None
              else:
                  short=ints[:k]
                  break

    #extra=['lbound','rbound','interval','index','future_ads','future_ads_count','period']
    #array_x=group.loc[index, columns].values
    ind=len(short)-1
    #print(ind)
    k=0
    #columns=['age','s','length','place','water','w_mean','age_mean','ivl0','ivl1','ivl2','ivl3','ivl4','ivl5',
             #'nivl0', 'nivl1', 'nivl2', 'nivl3', 'nivl4', 'nivl5','target','count','interval','index']
    while ind>=0:
        array=np.empty(24)
        #array_z=np.zeros(shape=(3,20))
        teta=short[ind]
        a, b = get_interval(teta, 1, current_point, lenght, expand=expand)
        c=0
        array[0]=current_age
        array[1] = current_s
        array[2] = b-a
        array[3] = (current_point-a)/(b-a)
        array[4] = current_w
        array[5]=water_mean
        array[6]=age_mean

        #array[32] = current_ad

        local_prev_mask = (previous_ads[field] >= a) & (previous_ads[field] <= b)
        local_prev = previous_ads[local_prev_mask]
        #print(previous_ads['Адрес от начала участка (new)'].shape,teta,lenght)
        ivls = get_horizontal_counts(previous_ads['Адрес от начала участка (new)'].values, interval=teta, L=lenght)
        res = ivls[:, 1] - ivls[:, 0]
        array[7]=res.sum() / lenght
        ivl_counts = ivls[:, 2].astype(int)
        for ii in np.arange(6):
            if ii == 5:
                mask1 = ivl_counts >= ii + 1
                mask2 = ivl_counts >= 0
            else:
                mask1 = ivl_counts == ii + 1
                mask2 = ivl_counts <= ii + 1
            array[8+ii]=mask1[mask1 == True].shape[0]
            array[14+ii] = mask2[mask2 == True].shape[0]
        array[20]=teta
        array[21]=index
        #tensor_x=get_frame(local_prev[['r',field]].values,bincount=bin_count,time_range=time_range,span=(a,b))
        #tensor=np.vstack((tensor_x,array_z))
        if future_ads_y is not None:
            local_fut = (future_ads_y[field] >= a) & (future_ads_y[field] <= b)
            locf=local_fut[local_fut==True].shape[0]
            if locf>0:
                array[22] = 1
                array[23] = locf
            else:
                array[22] = 0
                array[23] = 0
        else:
            local_fut = (we_future_ads_y[field] >= a) & (we_future_ads_y[field] <= b)
            locf=local_fut[local_fut==True].shape[0]
            if locf>0:
                array[22] = 1
                array[23] = locf
            else:
                array[22] = np.nan
                array[23] = np.nan
        ind=ind-1


        #array_y=np.append(array_x,array)
        #print(array_x.shape, array.shape,array_y.shape)
        #array_y[-1]=date
        #lst=array_y.tolist()
        #lst.append(tensor)
        #subjoined=np.array(lst)
        #array.append(current_ad)

        if k==0:
            #joined=np.array([array_y,tensor])
            #joined=np.append(array_y,tensor)
            joined=array
            k=k+1
        else:
            #joined=np.vstack((joined,np.array([array_y,tensor])))
            joined = np.vstack((joined, array))
        #print(ind, subjoined.shape,joined.shape)
    return joined

def sparsed_next(t=np.array([]),i=0,epsilon=1/12.):
    t0=t[i]
    t1=t0+epsilon
    #print(t1)
    j=i
    while j<t.shape[0]:
        if t[j]>=t1:
            return t[j]
        j=j+1
    return t1
def sparse(rs=np.array([]), epsilon=0.1):
    l = []
    # x0=rs[0]
    prev = rs[0]
    x = None
    a = prev + epsilon
    l.append(prev)
    i = 1
    while i < rs.shape[0]:
        x = rs[i]
        if x > a:
            t = rs[i]
            a = x + epsilon
            if t != prev:
                prev = t
                l.append(t)
        i = i + 1
    if (x is not None) & ~(x == np.array(l[-1])):
        l.append(x)
    return np.array(l)

def get_easy_binary_item(xdata,index=0,to_r=True, expand=False, ints=100, date=1, steps=15, epsilon=1/12.,
                   columns=['Наработка до отказа(new), лет', 'Адрес от начала участка (new)',
                            'Обводненность', 'getout','L,м', 'S','new_id','Дата аварии','index','to_out'],
                         cl_columns=['ads', 'ads05', 'ads1', 'ads2', 'ads3', 'ivl0', 'ivl1', 'ivl2', 'ivl3', 'ivl4', 'ivl5', 'nivl0', 'nivl1', 'nivl2', 'nivl3', 'nivl4', 'nivl5', 'wmean', 'amean', 'percent', 'tau', 'water', 'length']):
    def get_identity(data, date=1, a=0, b=1,index=-1,interval=100,steps=15,epsilon=1/12.):
        types=dict(names=['new_id','index','period','shape', 'Дата аварии', 'L,м', 'a', 'b', 'target', 'count','next','delta_next',
                                     'ads','ads05','ads1','ads2','ads3','ivl0','ivl1','ivl2','ivl3','ivl4','ivl5','nivl0','nivl1','nivl2','nivl3','nivl4','nivl5','wmean','amean','percent','tau','interval','water','x','s','to_out','top','length'],
                              formats=['U25',np.int32,np.int8,np.int32, 'datetime64[s]', np.float, np.float, np.float, np.float,
                                       np.float,np.float, np.float, np.float, np.float, np.float,np.float,np.float,np.float,np.float,np.float,np.float,
                                       np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float])
        columns = np.arange(steps)
        columns = [str(x) for x in columns]
        if to_r:
            tilde=1
        for c in columns:
            types['names'].append(c)
            types['formats'].append(np.float)
        dtype = np.dtype(types)
        identity = np.empty(shape=(1), dtype=dtype)
        step=dict({'ads05':0.5,'ads1':1.,'ads2':2.,'ads3':3.})
        tau=data[index,0]
        x=data[index,1]
        out=data[index,3]
        length=data[index,4]
        s=data[index,5]
        id=data[index,6]
        adate=data[index,7]
        i=data[index,8]
        to_out=data[index,9]
        identity['new_id']=id
        identity['s'] = s
        identity['to_out'] = to_out
        identity['tau'] = tau
        identity['interval'] = interval
        identity['index'] = i
        identity['period'] = date
        identity['Дата аварии'] = adate
        identity['water']=data[index, 2]
        identity['L,м'] = length
        identity['a']=a
        identity['b'] = b
        identity['x'] = x
        identity['length']=b-a
        if to_r:
            tilde=s
        else:
            tilde=1
        #print(i, index)
        #reg_columns = np.arange(steps)
        #reg_columns= [str(x) for x in reg_columns]
        mask=data[:,0]<=tau
        identity['shape'] = mask[mask==True].shape[0]
        mask1=(data[:,1]>=a)&(data[:,1]<=b)
        xmask=mask1&mask
        identity['ads']=xmask[xmask==True].shape[0]
        sparsed=sparse(data[:,0][xmask],epsilon=epsilon)[-steps:]
        for t in np.arange(1,steps+1):
            if -t>=-sparsed.shape[0]:
                identity[columns[-t]]=sparsed[-t]/tilde
            else:
                identity[columns[-t]] = 0

        for k in step.keys():
            #dlt=tau-step[k]
            substep=data[:,0]>=tau-step[k]
            smask=substep&xmask
            identity[k]=smask[smask==True].shape[0]
        ivls = get_horizontal_counts(data[:, 1][mask], interval=interval, L=length)
        res = ivls[:, 1] - ivls[:, 0]
        identity['percent']  = res.sum() / length
        w_mean = data[:, 2][mask].mean()
        a_mean = data[:, 0][mask].mean()
        identity['wmean']=w_mean
        identity['amean']=a_mean
        ivl_counts = ivls[:, 2].astype(int)
        for ii in np.arange(6):
            if ii == 5:
                mask3 = ivl_counts >= ii + 1
                mask4 = ivl_counts >= 0
            else:
                mask3 = ivl_counts == ii + 1
                mask4 = ivl_counts <= ii + 1
            identity['ivl'+str(ii)] = mask3[mask3 == True].shape[0]
            identity['nivl'+str(ii)] = mask4[mask4 == True].shape[0]
        tmask=mask1&(~mask)
        top=tau+date

        identity['top'] = top/tilde
        mask2=data[:,0]<=top
        ymask=tmask&mask2
        target=np.nan
        next=np.nan
        delta=np.nan
        identity['next']=next
        identity['delta_next']=delta
        dic={0:8./12.,1:7./12.,2:5./12.,3:4./12.,4:3./12.,5:3./12,6:2./12.,7:2./12.,}
        count = ymask[ymask == True].shape[0]
        if count>0:
            #row=np.append(tau,data[:,0][ymask])
            #toler=epsilon
            #ma=sparsed>0
            #ad=ma[ma==True].shape[0]-1
            #ad=identity['ads05'][0]
            #ad=count
            #if ad<len(dic.keys())-1:
                #toler=dic[ad]


            #next=sparsed_next(row,i=0,epsilon=toler)
            arange = np.arange(tmask.shape[0])
            inext = arange[tmask][0]
            next = data[inext, 0]
            delta = next - tau
            identity['next'] = next
            identity['delta_next'] = delta
        if top<=out:
            if count>0:
                target=1
            else:
                target=0
        else:
            if count>0:
                target=1
            else:
                target=np.nan
                count=np.nan
        identity['target']=target
        identity['count']=count
        #print(data.shape,xmask[xmask==True].shape)
        #print(data[:,0][xmask].astype(float))
        return identity,data[:,0][xmask].astype(float)
        #return identity, sparsed

    #tilde=0
    reg_columns = np.arange(steps)
    reg_columns = [str(x) for x in reg_columns]
    xdata.sort_values(by='Наработка до отказа(new), лет', inplace=True)
    group=xdata
    length = group.loc[index,'L,м']
    data=group[columns].values
    j=xdata.index.get_loc(index)
    x=data[j,1]
    L = []
    if 1*ints<=length:
        a, b = get_interval(ints, 1, x, length, expand=expand)
        bounds = np.array([a, b])
        tensor, time = get_identity(data, date=date, a=bounds[0], b=bounds[1], index=j, interval=ints, epsilon=epsilon,
                                    steps=steps)
        if tensor is not None:
            L.append((tensor, time))
            #print('appended')
        else:
            print('empty id', group[0])
    if len(L)>0:
        arr=L[0][0]
        t=L[0][1].reshape(1,-1)
    else:
        return None,None
    s=arr['s'].reshape(1,-1)
    shape=arr['shape'].reshape(1,-1)
    c=np.array(arr[cl_columns][0].tolist()).reshape(1,-1)
    r=np.array(arr[reg_columns][0].tolist()).reshape(1,-1)
    #t=
    top=arr['top']
    ret=ClRe(c=c,r=r,t=t,s=s,shape=shape)
    return ret,top

def get_easy_binary(xdata, ident='new_id', expand=False, ints=[100], date=[1], steps=15, epsilon=1/12.,
                   columns=['Наработка до отказа(new), лет', 'Адрес от начала участка (new)',
                            'Обводненность', 'getout','L,м', 'S','new_id','Дата аварии','index','to_out']):
    def get_identity(data, date=1, a=0, b=1,index=-1,interval=100,steps=15,epsilon=1/12.):
        types=dict(names=['new_id','index','period','shape', 'Дата аварии', 'L,м', 'a', 'b', 'target', 'count','next','delta_next','delta',
                                     'ads','ads05','ads1','ads2','ads3','ivl0','ivl1','ivl2','ivl3','ivl4','ivl5','nivl0','nivl1','nivl2','nivl3','nivl4','nivl5','wmean','amean','percent','tau','interval','water','x','s','to_out'],
                              formats=['U25',np.int32,np.int8,np.int32, 'datetime64[s]', np.float, np.float, np.float, np.float, np.float,
                                       np.float,np.float, np.float, np.float, np.float, np.float,np.float,np.float,np.float,np.float,np.float,np.float,
                                       np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float,np.float])
        columns = np.arange(steps)
        columns = [str(x) for x in columns]
        for c in columns:
            types['names'].append(c)
            types['formats'].append(np.float)
        dtype = np.dtype(types)
        identity = np.empty(shape=(1), dtype=dtype)
        step=dict({'ads05':0.5,'ads1':1.,'ads2':2.,'ads3':3.})
        tau=data[index,0]
        x=data[index,1]
        out=data[index,3]
        length=data[index,4]
        s=data[index,5]
        id=data[index,6]
        adate=data[index,7]
        i=data[index,8]
        to_out=data[index,9]
        identity['new_id']=id
        identity['s'] = s
        identity['to_out'] = to_out
        identity['tau'] = tau
        identity['interval'] = interval
        identity['index'] = i
        identity['period'] = date
        identity['Дата аварии'] = adate
        identity['water']=data[index, 2]
        identity['L,м'] = length
        identity['a']=a
        identity['b'] = b
        identity['x'] = x
        #print(i,index)

        mask=data[:,0]<=tau
        identity['shape'] = mask[mask==True].shape[0]
        mask1=(data[:,1]>=a)&(data[:,1]<=b)
        xmask=mask1&mask
        ads=xmask[xmask==True].shape[0]
        dt=np.nan
        prev=0
        if ads>1:
            prev=data[xmask, 0][-2]

        dt = tau - prev
        identity['delta'] = dt
        identity['ads']=ads


        sparsed=sparse(data[:,0][xmask],epsilon=epsilon)[-steps:]
        for t in np.arange(1,steps+1):
            if -t>=-sparsed.shape[0]:
                identity[columns[-t]]=sparsed[-t]
            else:
                identity[columns[-t]] = 0

        for k in step.keys():
            #dlt=tau-step[k]
            substep=data[:,0]>=tau-step[k]
            smask=substep&xmask
            identity[k]=smask[smask==True].shape[0]
        ivls = get_horizontal_counts(data[:, 1][mask], interval=interval, L=length)
        res = ivls[:, 1] - ivls[:, 0]
        identity['percent']  = res.sum() / length
        w_mean = data[:, 2][mask].mean()
        a_mean = data[:, 0][mask].mean()
        identity['wmean']=w_mean
        identity['amean']=a_mean
        ivl_counts = ivls[:, 2].astype(int)
        for ii in np.arange(6):
            if ii == 5:
                mask3 = ivl_counts >= ii + 1
                mask4 = ivl_counts >= 0
            else:
                mask3 = ivl_counts == ii + 1
                mask4 = ivl_counts <= ii + 1
            identity['ivl'+str(ii)] = mask3[mask3 == True].shape[0]
            identity['nivl'+str(ii)] = mask4[mask4 == True].shape[0]
        tmask=mask1&(~mask)
        top=tau+date
        mask2=data[:,0]<=top
        ymask=tmask&mask2
        target=np.nan
        next=np.nan
        delta=np.nan

        identity['next']=next
        identity['delta_next']=delta
        dic={0:8./12.,1:7./12.,2:5./12.,3:4./12.,4:3./12.,5:3./12,6:2./12.,7:2./12.,}
        count = ymask[ymask == True].shape[0]
        if count>0:
            #row=np.append(tau,data[:,0][ymask])
            #toler=epsilon
            #ma=sparsed>0
            #ad=ma[ma==True].shape[0]-1
            #ad=identity['ads05'][0]
            #ad=count
            #if ad<len(dic.keys())-1:
                #toler=dic[ad]


            #next=sparsed_next(row,i=0,epsilon=toler)
            arange = np.arange(tmask.shape[0])
            inext = arange[tmask][0]
            next = data[inext, 0]
            delta = next - tau
            identity['next'] = next
            identity['delta_next'] = delta



        #if count>0:


        if top<=out:
            if count>0:
                target=1
            else:
                target=0
        else:
            if count>0:
                target=1
            else:
                target=np.nan
                count=np.nan

        #top_v=min(top,out)
        #dt=top_v-tau
        #if dt>0:
            #identity['v'] = count/dt
        #else:
            #identity['v']=np.nan

        identity['target']=target
        identity['count']=count
        return identity,data[:,0][xmask].astype(float)
        #return identity, sparsed

    tilde=0
    xdata.sort_values(by='Наработка до отказа(new), лет', inplace=True)
    aggdata = xdata.groupby('new_id')
    # time_range=get_time_range_v1()
    npints = np.array(ints) * 2
    # k=len(columns)
    L = []
    for i, group in enumerate(aggdata):
        length = group[1]['L,м'].iloc[0]
        mask = npints <= length
        n = mask[mask == True].shape[0]
        data = group[1][columns].values
        #print(data.shape)
        if n > 0:
            for j in np.arange(-data.shape[0], 0):
                x = data[j, 1]
                #index=group[1].iloc[j]['index']
                for teta in ints:
                    if teta * 2 <= length:
                        for d in date:
                            a, b = get_interval(teta, 1, x, length, expand=expand)
                            bounds = np.array([a, b])
                            #print(index)
                            tensor,time = get_identity(data, date=d, a=bounds[0], b=bounds[1], index=j,interval=teta,epsilon=epsilon,steps=steps)
                            if tensor is not None:
                                L.append((tensor,time))
                            else:
                                print('empty id', group[0])
    #print(tilde)
    return np.array(L)

def get_recurrents(xdata, ident='new_id', expand=False, ints=[100], date=[1], size=15,kind='classifier',
                   columns=['Наработка до отказа(new), лет', 'Адрес от начала участка (new)',
                            'Обводненность', 'getout','L,м', 'S','new_id','Дата аварии','index']):
    if kind=='classifier':
        columns = ['Наработка до отказа(new), лет', 'Адрес от начала участка (new)',
                   'Обводненность', 'getout', 'L,м', 'S', 'new_id', 'Дата аварии', 'index']
    else:
        columns = ['Наработка до отказа(new), лет', 'Адрес от начала участка (new)',
                   'Обводненность', 'getout', 'L,м', 'S', 'new_id', 'Дата аварии', 'index','to_out']
    def to_label(x):
        if (x==0)|(x==1)|(x==2):
            return x
        else:
            return 3
    def get_identity(data, date=1, a=0, b=1,index=-1,kind='classifier'):
        dtype = np.dtype(dict(names=['new_id','index','period', 'Дата аварии', 'L,м', 'a', 'b', 'target', 'count','shape','next','delta_next'],
                              formats=['U25',np.int32,np.int8, 'datetime64[s]', np.float, np.float, np.float, np.float, np.float,np.int32,np.float, np.float]))
        identity = np.empty(shape=(1), dtype=dtype)
        tau=data[index,0]
        x=data[index,1]
        out=data[index,3]
        length=data[index,4]
        id=data[index,6]
        adate=data[index,7]
        i=data[index,8]
        identity['new_id']=id
        identity['index'] = i
        identity['period'] = date
        identity['Дата аварии'] = adate
        identity['L,м'] = length
        identity['a']=a
        identity['b'] = b
        identity['shape'] = 0
        mask=data[:,0]<=tau
        mask1=(data[:,1]>=a)&(data[:,1]<=b)
        #xmask=mask1&mask
        tmask=mask1&(~mask)
        top=tau+date
        mask2=data[:,0]<=top
        ymask=tmask&mask2
        target=np.nan
        next=np.nan
        delta=np.nan
        identity['next']=next
        identity['delta_next']=delta
        arange=np.arange(ymask.shape[0])
        count=ymask[ymask==True].shape[0]
        if count>0:
            inext=arange[ymask][0]
            #if inext<index:
                #print('error')
            next=data[inext,0]
            delta=next-tau
            identity['next'] = next
            identity['delta_next'] = delta
        if kind=='classifier':
            if top<=out:
                if count>0:
                    target=1
                else:
                    target=0
            else:
                if count>0:
                    target=1
                else:
                    target=np.nan
                    count=np.nan
        else:
            target=to_label(count)
        identity['target']=target
        identity['count']=count
        return identity

    tilde=0
    xdata.sort_values(by='Наработка до отказа(new), лет', inplace=True)
    aggdata = xdata.groupby('new_id')
    # time_range=get_time_range_v1()
    npints = np.array(ints) * 2
    # k=len(columns)
    L = []
    for i, group in enumerate(aggdata):
        length = group[1]['L,м'].iloc[0]
        mask = npints <= length
        n = mask[mask == True].shape[0]
        data = group[1][columns].values
        #print(data.shape)
        if n > 0:
            for j in np.arange(-data.shape[0], 0):
                x = data[j, 1]
                #index=group[1].iloc[j]['index']
                for teta in ints:
                    if teta * 2 <= length:
                        for d in date:
                            a, b = get_interval(teta, 1, x, length, expand=expand)
                            mask = (data[:, 1] >= a) & (data[:, 1] <= b)
                            mask[j:] = False
                            k = mask[mask == True].shape[0]+1
                            #print('k',k,'i',j)
                            if k <= size:
                                if k==size:
                                    tilde=tilde+1

                                bounds = np.array([a, b])

                                if kind=='classifier':
                                    identity = get_identity(data, date=d, a=bounds[0], b=bounds[1], index=j)
                                    tensor = recurrent(data, index=j, interval=teta, date=d, bounds=bounds, mask=mask)
                                else:
                                    identity = get_identity(data, date=d, a=bounds[0], b=bounds[1], index=j,kind=kind)
                                    tensor = reg_recurrent(data, index=j, interval=teta, date=d, bounds=bounds, mask=mask)

                                if tensor is not None:
                                    delta = size - tensor.shape[0]
                                    #print('tensor',tensor.shape[0])
                                    identity['shape'] = tensor.shape[0]
                                    if delta >= 0:
                                        if delta>0:
                                            zeros = np.zeros(shape=(delta, tensor.shape[1]))
                                            tensor = np.vstack((zeros, tensor))
                                        L.append((identity, tensor))
                                else:
                                    print('empty id', group[0])
    #print(tilde)
    return np.array(L)


def reg_recurrent(data=np.array([]), bounds=np.array([0, 1], dtype=float), index=-1, date=1, interval=100,
              mask=np.array([], dtype=bool)):
    # columns=['Наработка до отказа(new), лет', 'Адрес от начала участка (new)','Обводненность','getout','L,м','S','to_out']
    def fill(data=np.array([]), bounds=np.array([0, 1], dtype=float), index=-1, date=1, interval=100):
        array = np.empty(21)
        # dtype=np.dtype(dict(names=['new_id', 'Дата аварии','L,м','a','b','target','count'], formats=['U25','datetime64[s]',np.float,np.float,np.float,np.float,np.float]))
        # identity=np.empty(shape=(1),dtype=dtype)
        tau = data[index, 0]
        x = data[index, 1]
        w = data[index, 2]
        # i=data[index,4]
        out = data[index, 3]
        length = data[index, 4]
        s = data[index, 5]
        to_out=data[index,9]
        # id=data[index,7]
        # adate=data[index,8]
        # identity['new_id']=id
        # identity['Дата аварии'] = adate
        # identity['L,м'] = length
        a = bounds[0]
        b = bounds[1]
        # identity['a']=a
        # identity['b'] = b
        mask = data[:, 0] <= tau
        w_mean = data[:, 2][mask].mean()
        a_mean = data[:, 0][mask].mean()
        # mask1=imask
        mask1 = (data[:, 1] >= bounds[0]) & (data[:, 1] <= bounds[1])
        xmask = mask1 & mask
        tmask = mask1 & (~mask)
        top = tau + date
        mask2 = data[:, 0] <= top
        ymask = tmask & mask2
        target = np.nan
        count = ymask[ymask == True].shape[0]
        if top <= out:
            if count > 0:
                target = 1
            else:
                target = 0
        else:
            if count > 0:
                target = 1
            else:
                target = np.nan
                count = np.nan
        array[0] = tau
        array[1] = s
        array[2] = b - a
        array[3] = (x - a) / (b - a)
        array[4] = w
        array[5] = w_mean
        array[6] = a_mean
        ivls = get_horizontal_counts(data[:, 1][mask], interval=interval, L=length)
        res = ivls[:, 1] - ivls[:, 0]
        array[7] = res.sum() / length
        array[8]=to_out
        # identity['target'] = target
        # identity['count'] = count
        # array[22]=target
        # array[23]=count
        # array[20]=interval
        # array[21]=i
        ivl_counts = ivls[:, 2].astype(int)
        for ii in np.arange(6):
            if ii == 5:
                mask3 = ivl_counts >= ii + 1
                mask4 = ivl_counts >= 0
            else:
                mask3 = ivl_counts == ii + 1
                mask4 = ivl_counts <= ii + 1
            array[9 + ii] = mask3[mask3 == True].shape[0]
            array[15 + ii] = mask4[mask4 == True].shape[0]

        return array.reshape(1, -1)

    array = fill(data, bounds, index=index, interval=interval, date=date)
    # print('shape',array.shape)
    # xmask=(data[:,1]>=bounds[0])&(data[:,1]<=bounds[1])
    mask[index] = False
    while index > -data.shape[0]:
        if mask[index]:
            break
        else:
            index = index - 1
    # mask=cmask&xmask
    if mask[mask == True].shape[0] > 0:
        # index=index-1
        subarray = reg_recurrent(data, bounds, index=index, interval=interval, date=date, mask=mask)
        array = np.vstack((array, subarray))
    return array

def recurrent(data=np.array([]),bounds=np.array([0,1],dtype=float),index=-1,date=1,interval=100,mask=np.array([],dtype=bool)):
     #columns=['Наработка до отказа(new), лет', 'Адрес от начала участка (new)','Обводненность','getout','L,м','S']
    def fill(data=np.array([]),bounds=np.array([0,1],dtype=float),index=-1,date=1,interval=100):
        array = np.empty(20)
        #dtype=np.dtype(dict(names=['new_id', 'Дата аварии','L,м','a','b','target','count'], formats=['U25','datetime64[s]',np.float,np.float,np.float,np.float,np.float]))
        #identity=np.empty(shape=(1),dtype=dtype)
        tau=data[index,0]
        x=data[index,1]
        w=data[index,2]
        #i=data[index,4]
        out=data[index,3]
        length=data[index,4]
        s= data[index, 5]
        #id=data[index,7]
        #adate=data[index,8]
        #identity['new_id']=id
        #identity['Дата аварии'] = adate
        #identity['L,м'] = length
        a=bounds[0]
        b=bounds[1]
        #identity['a']=a
        #identity['b'] = b
        mask=data[:,0]<=tau
        w_mean=data[:,2][mask].mean()
        a_mean=data[:,0][mask].mean()
        #mask1=imask
        mask1=(data[:,1]>=bounds[0])&(data[:,1]<=bounds[1])
        xmask=mask1&mask
        tmask=mask1&(~mask)
        top=tau+date
        mask2=data[:,0]<=top
        ymask=tmask&mask2
        target=np.nan
        count=ymask[ymask==True].shape[0]
        if top<=out:
            if count>0:
                target=1
            else:
                target=0
        else:
            if count>0:
                target=1
            else:
                target=np.nan
                count=np.nan
        array[0]=tau
        array[1] = s
        array[2] = b-a
        array[3] = (x-a)/(b-a)
        array[4] = w
        array[5]=w_mean
        array[6]=a_mean
        ivls = get_horizontal_counts(data[:,1][mask], interval=interval, L=length)
        res = ivls[:, 1] - ivls[:, 0]
        array[7] = res.sum() / length
        #identity['target'] = target
        #identity['count'] = count
        #array[22]=target
        #array[23]=count
        #array[20]=interval
        #array[21]=i
        ivl_counts = ivls[:, 2].astype(int)
        for ii in np.arange(6):
            if ii == 5:
                mask3 = ivl_counts >= ii + 1
                mask4 = ivl_counts >= 0
            else:
                mask3 = ivl_counts == ii + 1
                mask4 = ivl_counts <= ii + 1
            array[8 + ii] = mask3[mask3 == True].shape[0]
            array[14 + ii] = mask4[mask4 == True].shape[0]

        return array.reshape(1,-1)
    array=fill(data,bounds,index=index,interval=interval,date=date)
    #print('shape',array.shape)
    #xmask=(data[:,1]>=bounds[0])&(data[:,1]<=bounds[1])
    mask[index] = False
    while index>-data.shape[0]:
        if mask[index]:
            break
        else:
            index=index-1
    #mask=cmask&xmask
    if mask[mask==True].shape[0]>0:
        #index=index-1
        subarray=recurrent(data,bounds,index=index,interval=interval,date=date,mask=mask)
        array=np.vstack((array,subarray))
    return array



def binary_tensor_fill(group, index, ints=[100], date=1, expand=True, time_range=np.array([]), bin_count=20,
                       columns=['new_id','Дата ввода','Дата аварии','Наработка до отказа(new), лет','Адрес от начала участка (new)','S','L,м']):
    #group = group.sort_values(by=['Дата аварии'])
    current_ad = group.loc[index, 'Дата аварии']
    getin = group.loc[index, 'Дата ввода']
    dateout = group.loc[index, 'Дата перевода в бездействие']
    #current_s = group.loc[index, 'S']
    #left = group.loc[index, 'a']
    current_age = group.loc[index, 'Наработка до отказа(new), лет']
    getout = (dateout - getin) / np.timedelta64(1, 'Y')
    group['r'] = group['Наработка до отказа(new), лет'] / group['S']
    previous_ads = group[group['Дата аварии'] <= current_ad]
    future_ads = group[group['Дата аварии'] > current_ad]
    future_ads['delta'] = (future_ads['Дата аварии'] - current_ad) / np.timedelta64(1, 'Y')
    water_mean=previous_ads['Обводненность'].mean()
    age_mean = previous_ads['Наработка до отказа(new), лет'].mean()

    if getout >= (current_age + date):
        future_ads_y = future_ads[future_ads['delta'] <= date]
    else:
        future_ads_y = None
        we_future_ads_y = future_ads[future_ads['delta'] <= date]

    lenght = group.loc[index, 'L,м']
    field = 'Адрес от начала участка (new)'
    current_point = group.loc[index, field]
    k=len(ints)
    ints.sort()
    short = ints

    if not expand:
        for k,teta in enumerate(ints):
            l = int(lenght / (2 * teta))
            if l==0:
              if k==0: return None
              else:
                  short=ints[:k]
                  break

    extra=['lbound','rbound','interval','index','future_ads','future_ads_count','period']
    array_x=group.loc[index, columns].values
    ind=len(short)-1
    #print(ind)
    k=0
    while ind>=0:
        array=np.empty(shape=len(extra))
        array_z=np.zeros(shape=(3,20))
        teta=short[ind]
        a, b = get_interval(teta, 1, current_point, lenght, expand=expand)
        c=0
        array[0]=a
        array[1] = b
        array[2] = teta
        array[3] = index

        local_prev_mask = (previous_ads[field] >= a) & (previous_ads[field] <= b)
        local_prev = previous_ads[local_prev_mask]
        #print(previous_ads['Адрес от начала участка (new)'].shape,teta,lenght)
        ivls = get_horizontal_counts(previous_ads['Адрес от начала участка (new)'].values, interval=teta, L=lenght)
        res = ivls[:, 1] - ivls[:, 0]
        array_z[0,0]=res.sum() / lenght
        array_z[0,1]= water_mean
        array_z[0, 2] = age_mean
        ivl_counts = ivls[:, 2].astype(int)
        for ii in np.arange(6):
            if ii == 5:
                mask1 = ivl_counts >= ii + 1
                mask2 = ivl_counts >= 0
            else:
                mask1 = ivl_counts == ii + 1
                mask2 = ivl_counts <= ii + 1
            array_z[1,ii]=mask1[mask1 == True].shape[0]
            array_z[2, ii] = mask2[mask2 == True].shape[0]
        tensor_x=get_frame(local_prev[['r',field]].values,bincount=bin_count,time_range=time_range,span=(a,b))
        tensor=np.vstack((tensor_x,array_z))
        if future_ads_y is not None:
            local_fut = (future_ads_y[field] >= a) & (future_ads_y[field] <= b)
            locf=local_fut[local_fut==True].shape[0]
            if locf>0:
                array[4] = 1
                array[5] = locf
            else:
                array[4] = 0
                array[5] = 0
        else:
            local_fut = (we_future_ads_y[field] >= a) & (we_future_ads_y[field] <= b)
            locf=local_fut[local_fut==True].shape[0]
            if locf>0:
                array[4] = 1
                array[5] = locf
            else:
                array[4] = np.nan
                array[5] = np.nan
        ind=ind-1

        array_y=np.append(array_x,array)
        #print(array_x.shape, array.shape,array_y.shape)
        array_y[-1]=date
        lst=array_y.tolist()
        lst.append(tensor)
        subjoined=np.array(lst)

        if k==0:
            #joined=np.array([array_y,tensor])
            #joined=np.append(array_y,tensor)
            joined=subjoined
            k=k+1
        else:
            #joined=np.vstack((joined,np.array([array_y,tensor])))
            joined = np.vstack((joined, subjoined))
        #print(ind, subjoined.shape,joined.shape)
    return joined
def get_time_range_v1(top=40, step=0.5,S=8):
    if step==0:
        return None
    count=np.ceil(top/step)
    bounds=[]
    size=step
    while count>0:
        bounds.append(size)
        size=size+step
        count=count-1
    return np.array(bounds)/S
def get_frame(data,bincount=20,time_range=np.array([]),span=(0.,10.)):
    t0=0
    L=[]
    for t in time_range:
        mask=(data[:,0]<=t)&(data[:,0]>t0)
        t0=t
        bins,points=np.histogram(data[mask][:,1],bins=bincount,range=span)
        L.append(bins)
    return np.array(L,dtype='float')
def binary_fill(group, index, ints=[100], date=1, expand=True, repairs=None, bin_count=15, append=['interval','interval_ads','int_ads_halfyear','int_ads_oneyear','int_ads_twoyear',
     'int_ads_threeyear','future_ads','future_ads_count','index','last_place','last_ads','first_ads','interval_ads_r',
     'future_ads_r1','future_ads_r','future_calendar','delta_days','extra_lenght',
     'lbound','rbound','place','future_ads_r1_next','future_ads_r_next','future_calendar_next','future_ads_nex']):
    group = group.sort_values(by=['Дата аварии'])
    current_ad = group.loc[index, 'Дата аварии']
    current_year = pd.to_datetime(current_ad).year
    next_year = pd.to_datetime('01-01-' + str(current_year + date))
    beginofnext = pd.to_datetime('01-01-' + str(current_year + 1))
    current_delta = (beginofnext - current_ad) / np.timedelta64(1, 'D')
    getin = group.loc[index, 'Дата ввода']
    dateout = group.loc[index, 'Дата перевода в бездействие']
    current_R = group.loc[index, 'R']
    current_s = group.loc[index, 'S']
    halfy = current_ad - pd.DateOffset(months=6)
    oney = current_ad - pd.DateOffset(years=1)
    twoy = current_ad - pd.DateOffset(years=2)
    threey = current_ad - pd.DateOffset(years=3)
    left = group.loc[index, 'a']
    current_age = group.loc[index, 'Наработка до отказа(new), лет']
    getout = (dateout - getin) / np.timedelta64(1, 'Y')
    getout_r = getout / current_s
    group['r'] = group['Наработка до отказа(new), лет'] / group['S']
    current_r = group.loc[index, 'r']
    columns = list(group.columns)
    #columns.append(append)
    for x in append:
        columns.append(x)
    for letter in ['all ','three ','two ','one ','half ']:
        for i in np.arange(bin_count):
            columns.append(letter + str(i))
    for ii in np.arange(6):
        columns.append('ivl'+str(ii+1))
        columns.append('nivl' + str(ii + 1))
    columns.append('Длина пораженных участков')
    columns.append('% пораженных участков')
    columns.append('Средняя обводненность')
    #columns.append('Средняя скорость потока')
    columns.append('Средний возраст аварий')
    previous_ads = group[group['Дата аварии'] <= current_ad]
    previous_ads_r = group[group['r'] <= current_r]
    future_ads_r = group[group['r'] > current_r]
    future_ads = group[group['Дата аварии'] > current_ad]
    future_ads['delta'] = (future_ads['Дата аварии'] - current_ad) / np.timedelta64(1, 'Y')
    water_mean=previous_ads['Обводненность'].mean()
    #speed_mean = previous_ads['Скорость потока'].mean()
    age_mean = previous_ads['Наработка до отказа(new), лет'].mean()



    if dateout >= next_year:
        calendar_year = future_ads[future_ads['Дата аварии'] < next_year]
    else:
        calendar_year = None
        we_calendar_year=future_ads[future_ads['Дата аварии'] < next_year]


    if getout >= (current_age + date):
        future_ads_y = future_ads[future_ads['delta'] <= date]
    else:
        future_ads_y = None
        we_future_ads_y = future_ads[future_ads['delta'] <= date]

    if getout_r >= (current_R + date * 0.125):
        future_ads_r1 = future_ads_r[future_ads_r['r'] <= (current_R + date * 0.125)]
    else:
        future_ads_r1 = None
        we_future_ads_r1 = future_ads_r[future_ads_r['r'] <= (current_R + date * 0.125)]
    if getout_r >= current_r + date / current_s:
        future_ads_r2 = future_ads_r[future_ads_r['r'] <= current_r + date * 0.125]
    else:
        future_ads_r2 = None
        we_future_ads_r2 = future_ads_r[future_ads_r['r'] <= current_r + date * 0.125]
    # group['belong']=True
    lenght = group.loc[index, 'L,м']
    field = 'Адрес от начала участка (new)'
    current_point = group.loc[index, field]
    k=len(ints)

    if not expand:
        ints.sort()
        short=ints
        for k,teta in enumerate(ints):
            l = int(lenght / (2 * teta))
            if l==0:
              if k==0: return None
              else:
                  short=ints[:k]
                  break



    # columns.append('period')

    df = pd.DataFrame(index=np.arange(0, len(short)), columns=columns)
    df[group.columns] = group.loc[index, group.columns].values.reshape(-1, group.shape[1])
    df['extra_lenght'] = 0.0


    ind=len(short)-1
    while ind>=0:
        teta=short[ind]
        a, b = get_interval(teta, 1, current_point, lenght, expand=expand)
        df.loc[ind, 'lbound'] = a
        df.loc[ind, 'rbound'] = b
        df.loc[ind, 'interval'] = teta
        df.loc[ind, 'place'] = current_point-a
        #print('index ',ind,' teta ',teta)
        df.loc[ind, 'index'] = index
        ivls = get_horizontal_counts(previous_ads['Адрес от начала участка (new)'].values, interval=teta, L=lenght)
        res=ivls[:,1]-ivls[:,0]
        df.loc[ind, 'Длина пораженных участков'] = res.sum()
        df.loc[ind, '% пораженных участков'] = res.sum()/lenght
        df.loc[ind, 'Средняя обводненность'] = water_mean
        #df.loc[ind, 'Средняя скорость потока'] = speed_mean
        df.loc[ind, 'Средний возраст аварий'] = age_mean
        ivl_counts=ivls[:,2].astype(int)
        for ii in np.arange(6):
            if ii==5:
                mask1=ivl_counts>=ii+1
                mask2=ivl_counts>=0
            else:
                mask1 = ivl_counts == ii + 1
                mask2 = ivl_counts <= ii + 1

            df.loc[ind, 'ivl'+str(ii+1)]=mask1[mask1==True].shape[0]
            df.loc[ind, 'nivl' + str(ii + 1)]=mask2[mask2==True].shape[0]

        if repairs is not None:
            if repairs.shape[0] > 0:
                extra_lenght = GetExtraLenght(repairs=repairs, current_point=current_point, date=current_ad, a=a, b=b,
                                              tilde=left)
                df.loc[ind, 'extra_lenght'] = extra_lenght
        local_prev_mask = (previous_ads[field] >= a) & (previous_ads[field] <= b)
        local_prev_r_mask = (previous_ads_r[field] >= a) & (previous_ads_r[field] <= b)
        local_prev_count=local_prev_mask[local_prev_mask==True].shape[0]
        df.loc[ind, 'interval_ads'] = local_prev_count
        df.loc[ind, 'interval_ads_r'] = local_prev_r_mask[local_prev_r_mask==True].shape[0]
        local_prev = previous_ads[local_prev_mask]
        bins,points=np.histogram(local_prev[field],bins=bin_count)
        bins=list(bins)
        bins.sort()
        for h,bi in enumerate(bins):
            df.loc[ind,'all '+str(h)]=bi

        local_prev_3_mask = local_prev['Дата аварии'] >= threey
        loc3=local_prev_3_mask[local_prev_3_mask==True].shape[0]
        df.loc[ind, 'int_ads_threeyear'] = loc3
        #print(loc3)
        #sdf=previous_ads[local_prev_3_mask]
        bins, points = np.histogram(local_prev[local_prev_3_mask][field], bins=bin_count)
        bins=list(bins)
        bins.sort()
        for h, bi in enumerate(bins):
            df.loc[ind, 'three ' + str(h)] = bi

        local_prev_2_mask = local_prev['Дата аварии'] >= twoy
        loc2=local_prev_2_mask[local_prev_2_mask==True].shape[0]
        df.loc[ind, 'int_ads_twoyear'] = loc2
        bins, points = np.histogram(local_prev[local_prev_2_mask][field], bins=bin_count)
        bins=list(bins)
        bins.sort()
        for h, bi in enumerate(bins):
            df.loc[ind, 'two ' + str(h)] = bi

        local_prev_1_mask = local_prev['Дата аварии'] >= oney
        loc1=local_prev_1_mask[local_prev_1_mask==True].shape[0]
        df.loc[ind, 'int_ads_oneyear'] = loc1
        bins, points = np.histogram(local_prev[local_prev_1_mask][field], bins=bin_count)
        bins=list(bins)
        bins.sort()
        for h, bi in enumerate(bins):
            df.loc[ind, 'one ' + str(h)] = bi

        local_prev_05_mask = local_prev['Дата аварии'] >= halfy
        loc05=local_prev_05_mask[local_prev_05_mask==True].shape[0]
        df.loc[ind, 'int_ads_halfyear'] = loc05
        bins, points = np.histogram(local_prev[local_prev_05_mask][field], bins=bin_count)
        bins=list(bins)
        bins.sort()
        for h, bi in enumerate(bins):
            df.loc[ind, 'half ' + str(h)] = bi


        local_prev_less = local_prev[local_prev['Дата аварии'] < current_ad]
        if local_prev_less.shape[0] == 0:
            df.loc[ind, 'last_ads'] = current_ad
            df.loc[ind, 'first_ads'] = current_ad
            df.loc[ind,'last_place']=current_point
        else:
            df.loc[ind, 'last_ads'] = local_prev_less.loc[local_prev_less.iloc[-1].name, 'Дата аварии']
            df.loc[ind, 'first_ads'] = local_prev_less.loc[local_prev_less.iloc[0].name, 'Дата аварии']
            df.loc[ind, 'last_place'] = local_prev_less.loc[local_prev_less.iloc[-1].name, field]

        if future_ads_y is not None:
            local_fut = (future_ads_y[field] >= a) & (future_ads_y[field] <= b)
            locf=local_fut[local_fut==True].shape[0]
            if locf>0:
                df.loc[ind, 'future_ads'] = 1
                df.loc[ind,'future_ads_count']=locf

                df.loc[ind,'future_ads_next']=future_ads_y[local_fut]['Дата аварии'].min()
            else:
                df.loc[ind, 'future_ads'] = 0
            # print('future_ads is not none. shape=',local_fut.shape[0])
        else:
            local_fut = (we_future_ads_y[field] >= a) & (we_future_ads_y[field] <= b)
            locf=local_fut[local_fut==True].shape[0]
            if locf>0:
                df.loc[ind, 'future_ads'] = 1
                df.loc[ind, 'future_ads_count'] = locf
                df.loc[ind, 'future_ads_next'] = we_future_ads_y[local_fut]['Дата аварии'].min()
            else:
                df.loc[ind, 'future_ads'] = np.nan

        if future_ads_r1 is not None:
            local_fut_r1 = (future_ads_r1[field] >= a) & (future_ads_r1[field] <= b)
            locf_r=local_fut_r1[local_fut_r1==True].shape[0]
            if locf_r>0:
                df.loc[ind, 'future_ads_r'] = 1
                df.loc[ind, 'future_ads_r_next'] = future_ads_r1[local_fut_r1]['Дата аварии'].min()
            else:
                df.loc[ind, 'future_ads_r'] = 0
        else:
            local_fut_r1 = (we_future_ads_r1[field] >= a) & (we_future_ads_r1[field] <= b)
            locf_r=local_fut_r1[local_fut_r1==True].shape[0]
            if locf_r>0:
                df.loc[ind, 'future_ads_r'] =1
                df.loc[ind, 'future_ads_r_next'] = we_future_ads_r1[local_fut_r1]['Дата аварии'].min()
            else:
                df.loc[ind, 'future_ads_r'] = np.nan
        if future_ads_r2 is not None:
            local_fut_r2 = (future_ads_r2[field] >= a) & (future_ads_r2[field] <= b)
            locf_r1=local_fut_r2[local_fut_r2==True].shape[0]
            if locf_r1>0:
                df.loc[ind, 'future_ads_r1'] = 1
                df.loc[ind, 'future_ads_r1_next'] = future_ads_r2[local_fut_r2]['Дата аварии'].min()
            else:
                df.loc[ind, 'future_ads_r1'] = 0

        else:
            local_fut_r2 = (we_future_ads_r2[field] >= a) & (we_future_ads_r2[field] <= b)
            locf_r1=local_fut_r2[local_fut_r2==True].shape[0]
            if locf_r1>0:
                df.loc[ind, 'future_ads_r1'] = 1
                df.loc[ind, 'future_ads_r1_next'] = we_future_ads_r2[local_fut_r2]['Дата аварии'].min()
            else:
                df.loc[ind, 'future_ads_r1'] = np.nan

        if calendar_year is not None:
            local_year = (calendar_year[field] >= a) & (calendar_year[field] <= b)
            locf_c=local_year[local_year==True].shape[0]
            if locf_c>0:
                df.loc[ind, 'future_calendar'] = 1
                df.loc[ind, 'future_calendar_next'] = calendar_year[local_year]['Дата аварии'].min()
            else:
                df.loc[ind, 'future_calendar'] =0
        else:
            local_year = (we_calendar_year[field] >= a) & (we_calendar_year[field] <= b)
            locf_c = local_year[local_year == True].shape[0]
            if locf_c > 0:
                df.loc[ind, 'future_calendar'] = 1
                df.loc[ind, 'future_calendar_next'] = we_calendar_year[local_year]['Дата аварии'].min()
            else:
                df.loc[ind, 'future_calendar'] = np.nan

        df.loc[ind, 'delta_days'] = current_delta
        ind=ind-1

    df['period'] = date
    # display(df['extra_lenght'])
    return df
def load_regims(file='regims_sorted.csv',path='D:\\ml\\',types={'ID простого участка': np.int64,'Расход': np.float64,
 'Обводненность': np.float64,"Истинная скорость жидкости":np.float64,'Скорость смеси': np.float64,'Скорость критическая': np.float64,
 'P в начале участка': np.float64,'P в конце участка': np.float64,'Коэф гидравл сопротивления': np.float64,
 'Вязкость жидкости в раб услов': np.float64,'Режим течения': 'string','Структура течения ГЖС': 'string'},dates=['Дата расчета'],engine='c',infer_datetime_format=True, dayfirst=True):
    regims = pd.read_csv(path + file, engine=engine, parse_dates=dates, dtype=types,
                         infer_datetime_format=infer_datetime_format, dayfirst=dayfirst)
    return regims


def GetNearestCP(date, sid):
    # print(date)
    mask1 = sid['Дата расчета'] >= date
    mask2 = sid['Дата расчета'] <= date

    index1 = mask1[mask1 == True]
    if len(index1) > 0:
        minimum = index1.index[0]
        ge = sid.loc[minimum]['Дата расчета']
        delta1 = (ge - date) / np.timedelta64(1, 'D')
    else:
        delta1 = np.inf

    index2 = mask2[mask2 == True]
    if len(index2) > 0:
        maximum = index2.index[-1]
        le = sid.loc[maximum]['Дата расчета']
        delta2 = (date - le) / np.timedelta64(1, 'D')
    else:
        delta2 = np.inf

    if delta1 < delta2:
        index = minimum
    else:
        index = maximum
    return index


def SetNearestCP(data, index, sid, columns=['Дата расчета', 'Расход', 'Обводненность', 'Истинная скорость жидкости',
                                            'Скорость смеси', 'Скорость критическая', 'P в начале участка',
                                            'P в конце участка', 'Коэф гидравл сопротивления',
                                            'Вязкость жидкости в раб услов', 'Режим течения',
                                            'Структура течения ГЖС']):
    date = data.loc[index]['Дата аварии']
    index1 = GetNearestCP(date, sid)
    data.loc[index, columns] = sid.loc[index1, columns]
def set_regims(data, regims):
    notinlist = []
    agg_regims = regims.groupby('ID простого участка')
    for i, group in enumerate(data.groupby('ID простого участка')):
        ID = group[0]
        someid = group[1]
        try:
            regim_ids = agg_regims.groups[ID]
            for index in someid.index:
                # print('index ',index)
                SetNearestCP(data, index, regims.loc[regim_ids])
        except KeyError:
            notinlist.append(ID)
    return notinlist


def get_square(a, b):
    c = None
    d = None
    a1 = a[0]
    a2 = a[1]
    b1 = b[0]
    b2 = b[1]
    l = np.power((a1 - b1) * (a1 - b1) + (a2 - b2) * (a2 - b2), 0.5)
    if a1 == b1:
        if a2 < b2:
            c = [a1 - l, a2]
            d = [b1 - l, b2]
        else:
            c = [a1 + l, a2]
            d = [b1 + l, b2]
    else:
        if a1 < b1:
            c = [a1, a2 + l]
            d = [b1, b2 + l]
        else:
            c = [a1, a2 - l]
            d = [b1, b2 - l]
    return c, d


def GetNewPoints(a, b):
    c = None
    d = None
    a1 = a[0]
    a2 = a[1]
    b1 = b[0]
    b2 = b[1]
    l = np.power((a1 - b1) * (a1 - b1) + (a2 - b2) * (a2 - b2), 0.5)
    ab = [b1 - a1, b2 - a2]
    c = [a1 + ab[0] / 3, a2 + ab[1] / 3]
    d = [a1 + 2 * ab[0] / 3, a2 + 2 * ab[1] / 3]
    return c, d


def fractal(points, count=1):
    # print('count ', count)
    if count == 0:
        return points
    count = count - 1
    line = []
    for i in np.arange(len(points) - 1):
        a, b = points[i:i + 2]
        x, y = GetNewPoints(a, b)
        c, d = get_square(x, y)

        line.append(a)
        line.append(x)
        line.append(c)
        line.append(d)
        line.append(y)
        line.append(b)
        # print(line)
    return fractal(line, count=count)
def fractal_new(points, count=1):
    # print('count ', count)
    if count == 0:
        return points
    count = count - 1
    line = []
    for i in np.arange(len(points) - 1):
        a, b = points[i:i + 2]
        x, y = GetNewPoints(a, b)
        c, d = GetSquare(x, y)        #print(c,' ',d)
        e=GetRectTriangle(c, d)
        line.append(a)
        line.append(x)
        line.append(c)
        line.append(e)
        line.append(d)
        line.append(y)
        line.append(b)
        # print(line)
    return fractal_new(line, count=count)


def GetRectTriangle(a, b, psi=np.pi / 2):
    ksi = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    x, y = GetOrth(ksi, a, psi=-np.pi / 2)
    return x, y
def GetSquare(a, b):
    c = GetOrth(a, b)
    d = GetOrth(b, a, psi=-np.pi / 2)
    return c, d
def GetOrth(a, b, psi=np.pi / 2):
    ab = (b[0] - a[0], b[1] - a[1])
    x = a[0] + ab[0] * np.cos(psi) - ab[1] * np.sin(psi)
    y = a[1] + ab[0] * np.sin(psi) + ab[1] * np.cos(psi)
    return x, y


def GetTriangle(a, b, psi=np.pi / 3):
    ab = (b[0] - a[0], b[1] - a[1])
    l = np.power(ab[0] * ab[0] + ab[1] * ab[1], 0.5)
    x = ab[0] * np.cos(psi) - ab[1] * np.sin(psi)
    y = ab[1] * np.cos(psi) + ab[0] * np.sin(psi)
    return x + a[0], y + a[1]


def snowflake(points, count=0, psi=np.pi / 3):
    if count == 0:
        return points
    count = count - 1
    line = []
    for i in np.arange(len(points) - 1):
        a, b = points[i:i + 2]
        # x,y=GetNewPoints(a,b)

        # c,d=GetTriangle(x,y,psi=psi)
        c, x, y = GetTrianglePoints(a, b, psi=psi)
        line.append(a)
        line.append(x)
        line.append(c)
        line.append(y)
        line.append(b)
    return snowflake(line, count, psi=psi)


def GetTrianglePoints(a, b, psi=np.pi / 3):
    ab = (b[0] - a[0], b[1] - a[1])
    l = np.power(ab[0] * ab[0] + ab[1] * ab[1], 0.5)
    t = ((b[0] + a[0]) / 2, (b[1] + a[1]) / 2)
    l_a = l / (2 + np.power(2 * (1 - np.cos(psi)), 0.5))
    l_s = l - 2 * l_a
    ta = (-(a[0] - b[0]) * l_s / (2 * l), -(a[1] - b[1]) * l_s / (2 * l))
    l_ta = np.power(ta[0] * ta[0] + ta[1] * ta[1], 0.5)
    ksi = (-ta[0] + t[0], -ta[1] + t[1])
    eta = (ta[0] * 2 + ksi[0], ta[1] * 2 + ksi[1])
    # print(ksi)
    # print(eta)
    alpha = (np.pi - psi) / 2
    x = (np.cos(alpha) * ta[0] - np.sin(alpha) * ta[1]) * l_a / l_ta + ksi[0]
    y = (np.cos(alpha) * ta[1] + np.sin(alpha) * ta[0]) * l_a / l_ta + ksi[1]
    return (x, y), ksi, eta
#gpndata=pd.read_csv(path+'Отказы после ремонта_v7.csv')
#dates=['Дата ввода','Дата аварии','Дата перевода в бездействие','Дата окончания ремонта','Дата ремонта до аварии']
#for date in dates:
    #gpndata[date]=pd.to_datetime(gpndata[date])
#strlist=['Вид покрытия внутреннего', 'Название','ID простого участка', 'Месторождение','Материал трубы',
       #'Тип трубы', 'Завод изготовитель','Состояние','Способ ликв в момент', 'ID ремонта','Тип ремонта', 'ID ремонта до аварии','Тип ремонта до аварии']
#gpndata['ID простого участка']=gpndata['ID простого участка'].astype('str')
#for s in strlist:
    #gpndata[s]=gpndata[s].astype('str')
#someid=gpndata[gpndata['new_id']=='33865']
#Data=make_intervals(someid,ints=[100,150],date=np.arange(1,2), ident='new_id', expand=False,function=binary_fill)
class quadratic_transform:
    def __init__(self,x=dict({"x1":0,'x2':0.3,'x3':1}),d=dict({"d1":2.97,'d2':1.51,'d3':0})):
        self.x1=x['x1']
        self.x2=x['x2']
        self.x3=x['x3']
        self.d1=d['d1']
        self.d2=d['d2']
        self.d3=d['d3']
        A=((self.d1-self.d2)/(self.x1-self.x2))-((self.d2-self.d3)/(self.x2-self.x3))
        B=((self.x1*self.x1-self.x2*self.x2)/(self.x1-self.x2))-((self.x2*self.x2-self.x3*self.x3)/(self.x2-self.x3))
        self.a3=A/B
        self.a2=(self.d1-self.d2)/(self.x1-self.x2)-self.a3*((self.x1*self.x1-self.x2*self.x2)/(self.x1-self.x2))
        self.a1=self.d1-self.a2*self.x1-self.a3*self.x1*self.x1
    def value(self,x):
        return self.a1+self.a2*x+self.a3*x*x

class linear_transform:
    def __init__(self, x=dict({"x1": 0.7, 'x2': 0.97}), d=dict({"d1": 0.9265102231005462, 'd2':0.811612327139037})):
        self.x1 = x['x1']
        self.x2 = x['x2']
        self.d1 = d['d1']
        self.d2 = d['d2']
        self.a2 = (self.d1 - self.d2) / (self.x1 - self.x2)
        self.a1 = (self.d1+self.d2 - self.a2*(self.x1+self.x2))*0.5

    def value(self, x):
        return self.a1 + self.a2 * x

def expand(data,n,alpha,seed=42):
    np.random.seed(seed)
    l=int(alpha*n)
    k=data.shape[0]
    expand=None
    if k==l: return data
    elif k<l:
        expand=data.copy()
        shuffled_indices = np.random.permutation(data.index)
        tau=int(l/k)
        tilde=np.fmod(l,k)
        expand=data.loc[shuffled_indices[:tilde]]
        while tau>0:
            expand=expand.append(data)
            tau=tau-1
        return expand
    else:
        shuffled_indices = np.random.permutation(data.index)
        tilde=l
        expand=data.loc[shuffled_indices[:tilde]]
        return expand
def GetOverSample(data,n,fiels='Месторождение',weights=dict()):
    Expand=None
def weighs_euclidian(x=np.array([0,1]),y=np.array([1,0]),w=np.array([1,1])):
    res=(x-y)**2
    return np.power(np.dot(res,w),0.5)
class metrics:
    def __init__(self,w=None):
        self.w=w
    def euclidian(self,x=np.array([0,1]),y=np.array([1,0]),w=None):
        res = (x - y) ** 2
        if w is not None:
            self.w=w
        elif self.w is not None:
            return np.power(np.dot(res, self.w), 0.5)
        else:
            return np.power(np.sum(res), 0.5)

def get_time_range(enter=pd.to_datetime('2016-01-01'),get_out=pd.to_datetime('2016-12-31'),S=8,step=1):
    age=(get_out-enter)/np.timedelta64(1,'Y')
    year=pd.to_datetime(enter).year
    first_year=pd.to_datetime(str(year)+'-12-31')
    tau=(first_year-enter)/np.timedelta64(1,'Y')
    #k=int((age-tau+step)/step)
    k=np.ceil((age-tau)/step)
    bound=[]
    for i in np.arange(k+1):
        bound.append(tau+step*i)
    return np.array(bound)/S
def get_random_time_range(count=10,S=8,step=1):
    tau=np.random.random()
    #k=int((age-tau+step)/step)
    k=count
    bound=[]
    for i in np.arange(k+1):
        bound.append(tau+step*i)
    return np.array(bound)/S

class pipe:
    def __init__(self):
        self.intervals=np.array([],dtype=int)
        self.empty=0
    def append(self,other):
        self.empty=self.empty+other.empty
        self.intervals=np.vstack((self.intervals,other.intervals))

def get_random_pipe(L=100,n=5, age=10,sigma=1):
    i=np.ceil(L/100)
    points_number=n*i
    points=L*np.random.random_sample(int(points_number))
    accidents=sigma * np.random.randn(points.shape[0])+age
    return np.array([points,accidents]).T


def get_clusters(L=100, sigma=(1, 30), age=10, interval=100, state=42, nmin=5, nmax=15):
    n = np.ceil(L / interval)
    rnd = np.random.RandomState(state)
    centers = rnd.randint(low=sigma[1], high=L-sigma[1], size=int(n))
    # print(centers)

    for i, c in enumerate(centers):
        teta = rnd.randint(low=nmin, high=nmax)
        # print(teta)
        X = rnd.randn(teta, 2)
        delta = 2 * rnd.randn()
        X[:, 0] = sigma[1] * X[:, 0] + c
        X[:, 1] = sigma[0] * X[:, 1] + age + delta
        empty = np.empty(X.shape[0])
        empty.fill(i)
        X = np.hstack((X, empty.reshape(-1, 1)))
        if i == 0:
            array = X
        else:
            array = np.vstack((array, X))
        mask = (array[:, 0] <= L) & (array[:, 0] >= 0)
    return array[mask]

def get_joined_matrices(data,time_range=np.array([]), lenght=100, flow=True,groupby='ID простого участка'):
    groups=data.groupby(groupby)
    pipes=pipe()
    pipes_=pipe()
    for i,group in enumerate(groups):
        s,s_=get_id_intervals(group[1], time_range=time_range, lenght=lenght, flow=flow)
        if i==0:
            pipes.intervals=s.intervals
            pipes.empty=s.empty
            pipes_.intervals = s_.intervals
            pipes_.empty = s_.empty
        else:
            pipes.append(s)
            pipes_.append(s_)
    return  pipes,pipes_

def get_joined_matrices_by_np(data,time_range=np.array([]), length=100, flow=True,groupby='new_id'):
    groups=data.groupby(groupby)
    pipes=pipe()
    pipes_=pipe()
    for i,group in enumerate(groups):
        enter,L,S=group[1].iloc[0][['Дата ввода','L','S']]
        get_out=group[1]['Дата перевода в бездействие'].max()
        try:
            s,s_=get_id_intervals_by_np(group[1][['Наработка до отказа','Адрес от начала участка' ,'Дата перевода в бездействие']].values, time_range=time_range, length=length, flow=flow,out=None,L=L,S=S, enter=enter, get_out=get_out)

            if i==0:
                pipes.intervals=s.intervals
                pipes.empty=s.empty
                pipes_.intervals = s_.intervals
                pipes_.empty = s_.empty
            else:
                pipes.append(s)
                pipes_.append(s_)
        except(ValueError):
            print('error in id '+str(group[0]))
    return  pipes,pipes_

def get_id_intervals(data, time_range=np.array([]), lenght=100, flow=True,out=None):
    List_minus = []
    List_plus = []
    if data.shape[0] == 0: return np.array(List_plus), np.array(List_minus)

    enter, get_out, L, S = data.iloc[0][['Дата ввода', 'Дата перевода в бездействие', 'L', 'S']]
    life=((get_out-enter)/np.timedelta64(1,'Y'))/S
    if time_range.shape[0] == 0:
        time_range = get_time_range(enter, get_out, S, step=1)
    tau = time_range[0]
    delta = 0
    if time_range.shape[0] >= 1:
        delta = time_range[1] - tau
    if out is None:
        out=np.ceil((life-tau)/delta)
    intervals = split(data, lenght=lenght, flow=flow)
    for i in np.arange(intervals.shape[0]):
        index, a, b = intervals[i, :]
        x = data.loc[index, 'Адрес от начала участка']
        r = data.loc[index, 'Наработка до отказа'] / S

        if delta==0:
            current=0
        else:
            current=np.ceil((r-tau)/delta)
        #current = int((r - tau + delta) / delta)
        s_plus = []
        s_minus = []
        if flow:
            minus_sub = data[(data['Адрес от начала участка'] >= x) & (data['Адрес от начала участка'] <= b)]
            plus_sub = data[(data['Адрес от начала участка'] >= a) & (data['Адрес от начала участка'] <= x)]
            s_minus.append(x)
            s_minus.append(b)
            s_plus.append(a)
            s_plus.append(x)
        else:
            minus_sub = data[(data['Адрес от начала участка'] >= b) & (data['Адрес от начала участка'] <= x)]
            plus_sub = data[(data['Адрес от начала участка'] >= x) & (data['Адрес от начала участка'] <= a)]
            s_minus.append(b)
            s_minus.append(x)
            s_plus.append(x)
            s_plus.append(a)
        for j, t in enumerate(time_range):
            if j < current:
                s_plus.append(0)
                s_minus.append(0)
            elif j>=out:
                s_plus.append(0)
                s_minus.append(0)
            else:
                min_mask = minus_sub['Наработка до отказа'] / S <= time_range[j]
                s_minus.append(min_mask[min_mask == True].shape[0])
                plus_mask = plus_sub['Наработка до отказа'] / S <= time_range[j]
                s_plus.append(plus_mask[plus_mask == True].shape[0])

        List_plus.append(s_plus)
        List_minus.append(s_minus)
        empty=np.ceil(L/lenght)
        minus=pipe()
        minus.empty=empty
        minus.intervals=np.array(List_minus, dtype=int)
        plus=pipe()
        plus.empty = empty
        plus.intervals=np.array(List_plus, dtype=int)
    #return np.array(List_plus, dtype=int), np.array(List_minus, dtype=int)
    return plus,minus
def get_id_intervals_by_np_v0(data,  time_range=np.array([]),  lenght=100, flow=True,**args):
    List_minus = []
    List_plus = []
    if data.shape[0] == 0: return np.array(List_plus), np.array(List_minus)
    L = args['L']
    S = args['S']
    if time_range.shape[0] == 0:
        enter = args['enter']
        get_out = args['get_out']

        time_range = get_time_range(enter, get_out, S, step=1)
    tau = time_range[0]
    delta=0
    if time_range.shape[0]>=1:
        delta = time_range[1]-tau

    intervals = split_by_np(data[:,1].reshape(-1,1),L=L, lenght=lenght, flow=flow)
    for i in np.arange(intervals.shape[0]):
        index, a, b = intervals[i, :]
        r=data[index][0]/S
        x = data[index][1]
        if delta==0:
            current=0
        else:
            current=np.ceil((r-tau)/delta)
        s_plus = []
        s_minus = []
        if flow:
            mask_min=(data[:,1]>=x)&(data[:,1]<=b)
            mask_plus=(data[:,1]>=a)&(data[:,1]<=x)
            s_minus.append(x)
            s_minus.append(b)
            s_plus.append(a)
            s_plus.append(x)
        else:
            mask_min=(data[:,1]>=b)&(data[:,1]<=x)
            mask_plus=(data[:,1]>=x)&(data[:,1]<=a)
            s_minus.append(b)
            s_minus.append(x)
            s_plus.append(x)
            s_plus.append(a)
        minus_sub = data[mask_min]
        plus_sub = data[mask_plus]

        for j, t in enumerate(time_range):
            if j < current:
                s_plus.append(0)
                s_minus.append(0)
            else:
                min_mask=minus_sub[:,0]/S <=time_range[j]
                plus_mask=plus_sub[:,0]/S <=time_range[j]
                s_minus.append(min_mask[min_mask == True].shape[0])
                s_plus.append(plus_mask[plus_mask == True].shape[0])

        List_plus.append(s_plus)
        List_minus.append(s_minus)
        empty=np.ceil(L/lenght)
        minus=pipe()
        minus.empty=empty
        minus.intervals=np.array(List_minus, dtype=int)
        plus=pipe()
        plus.empty = empty
        plus.intervals=np.array(List_plus, dtype=int)
    #return np.array(List_plus, dtype=int), np.array(List_minus, dtype=int)
    return plus,minus
def get_id_intervals_by_np(data,  time_range=np.array([]), step=1,  length=100., flow=True,out=None,**args):
    List_minus = []
    List_plus = []
    if data.shape[0] == 0: return np.array(List_plus), np.array(List_minus)
    L = args['L']
    S = args['S']
    enter = args['enter']
    get_out = args['get_out']

    if time_range.shape[0] == 0:
        time_range = get_time_range(enter, get_out, S, step=step)
    tau = time_range[0]
    delta=0
    if time_range.shape[0]>=1:
        delta = time_range[1]-tau

    #print(' out= ', out)
    intervals = split_by_np(data[:,1].reshape(-1,1),L=L, length=length, flow=flow)
    for i in np.arange(intervals.shape[0]):
        index, a, b = intervals[i, :]
        index=int(index)
        #print(a,b)
        r=data[index][0]/S
        x = data[index][1]
        #print(a, x, b)
        if delta==0:
            current=0
        else:
            current=np.ceil((r-tau)/delta)
        s_plus = []
        s_minus = []
        if flow:
            mask_min=(data[:,1]>=x)&(data[:,1]<=b)
            mask_plus=(data[:,1]>=a)&(data[:,1]<=x)
            s_minus.append(x)
            s_minus.append(b)
            s_plus.append(a)
            s_plus.append(x)
        else:
            mask_min=(data[:,1]>=b)&(data[:,1]<=x)
            mask_plus=(data[:,1]>=x)&(data[:,1]<=a)
            s_minus.append(b)
            s_minus.append(x)
            s_plus.append(x)
            s_plus.append(a)
        minus_sub = data[mask_min]
        plus_sub = data[mask_plus]
        minus_out=minus_sub[:,2].max()
        plus_out = plus_sub[:, 2].max()
        mlife = ((minus_out - enter) / np.timedelta64(1, 'Y')) / S
        plife = ((plus_out - enter) / np.timedelta64(1, 'Y')) / S
        if out is None:
            if delta > 0:
                mout = np.ceil((mlife - tau) / delta)
                pout=np.ceil((plife - tau) / delta)
            else:
                mout = 0
                pout=0


        for j, t in enumerate(time_range):
            if j < current:
                s_plus.append(0)
                s_minus.append(0)
            else:
                if j>mout:
                    s_minus.append(-1)
                else:
                    min_mask = minus_sub[:, 0] / S <= time_range[j]
                    s_minus.append(min_mask[min_mask == True].shape[0])

                if j>pout:
                    s_plus.append(-1)
                else:
                    plus_mask = plus_sub[:, 0] / S <= time_range[j]
                    s_plus.append(plus_mask[plus_mask == True].shape[0])





        List_plus.append(s_plus)
        List_minus.append(s_minus)
        empty=np.ceil(L/length)
        minus=pipe()
        minus.empty=empty
        minus.intervals=np.array(List_minus, dtype=int)
        plus=pipe()
        plus.empty = empty
        plus.intervals=np.array(List_plus, dtype=int)
    #return np.array(List_plus, dtype=int), np.array(List_minus, dtype=int)
    return plus,minus

def split(data, lenght=100, flow=True):
    List = []
    data['first'] = False
    data['a'] = np.nan
    data['b'] = np.nan
    L=data['L'].min()
    mask = data['first'] == False
    mask1 = (data['Адрес от начала участка'] > L) | (data['Адрес от начала участка'] < 0)
    data.loc[mask1, 'first'] = True

    while not all(data['first']):
        index = mask[mask == True].index[0]
        data.loc[index]['first'] = True
        x = data.loc[index]['Адрес от начала участка']
        if flow:
            #a,b=get_interval(teta=lenght,k=1,current_point=x,expand=False)
            a = x - lenght
            if a<0:a=0
            b = x + lenght
            if b>L: b=L
            mask1 = (data['Адрес от начала участка'] >= a) & (data['Адрес от начала участка'] <= b)
        else:
            a = x + lenght
            if a>L: a=L
            b = x - lenght
            if b < 0: b = 0
            mask1 = (data['Адрес от начала участка'] >= b) & (data['Адрес от начала участка'] <= a)
        List.append([index, a, b])

        data.loc[mask1, 'first'] = True
        mask = data['first'] == False
    return np.array(List)

def split_by_np(data=np.array([]),L=100,length=100.,flow=True):
    List = []
    data=np.hstack((data,np.zeros(shape=(data.shape[0],1))))

    #print(indices)
    mask1 = (data[:,0] > L) | (data[:,0] < 0)
    data=data[~mask1]
    indices = np.arange(data.shape[0], dtype=int)
    mask=data[:,-1]==True
    while not all(mask):
        #print(mask)
        index = indices[~mask]
        x = data[index][0, 0]
        #print(x)
        if flow:
            a = x - length
            if a<0:a=0
            b = x + length
            if b>L: b=L
            mask1=(data[:,0]>=a) & (data[:,0]<=b)
        else:
            a = x + length
            if a>L: a=L
            b = x - length
            if b < 0: b = 0
            mask1 = (data[:, 0] >= b) & (data[:, 0] <= a)
        #print(a,b)

        List.append([index[0], a, b])
        data[:,-1][mask1]=True
        mask=data[:,-1]==True
    #type=dict({'names':['index','a','b'],'formats':[np.int32,np.float,np.float]})

    #dtype=np.dtype(type)
    #array=np.empty(len(List),dtype=dtype)
    #array['index']


    return np.array(List,dtype=float)

def get_decomposition(data,number=1,state=0,stop=5,r=0):
    N_all = []
    Lambda_all = []
    try:
        indices,N,Lambda=get_vector_n(data[:,r],number=number,state=state)
        N_all.append(N)
        Lambda_all.append(Lambda)
        if r==stop:
            return N_all,Lambda_all
        r=r+1
        i=state
        #T_N=[]
        #T_Lambda=[]
        for index,n in zip(indices,N):
            N_all_,Lambda_all_=get_decomposition(data[index],number=n,stop=stop,state=i,r=r)
            N_all.append(N_all_)
            Lambda_all.append(Lambda_all_)
            i=i+1
        #N_all.append(T_N)
        #Lambda_all.append(T_Lambda)
    except(TypeError):
        print("indices ",indices)
        print("N ",N)
    return N_all, Lambda_all
def get_vector_n(c=np.array([]),number=0,state=0):
    i=state
    n=1
    N=[]
    index=[]
    N.append(number)
    if number==0:
        return np.array(index, dtype=np.int64),np.array(N),np.array([0])
    while n>0:
        indices=np.where(c==i)[0]
        n=len(np.where(c>i)[0])

        N.append(n)
        index.append(indices)
        i=i+1
    Nar=np.array(N)
    Ne=Nar[:Nar.shape[0]-1]-Nar[1:Nar.shape[0]]
    Lambda=Nar[1:Nar.shape[0]]/Nar[:Nar.shape[0]-1]
    return index,Ne,Lambda

def get_columns(data,i,A=[]):
    if (i==0):
        A.append(data[0])
        return
    if (len(data)==1):
        i=0
        get_columns(data, i, A)
        return
    #print(len(data))
    i=i-1
    for j in np.arange(len(data[0])):
        #print('j',j)
        #print('i',i)
        get_columns(data[j+1],i,A)

def get_decomposition_v1(data,number=1,state=0,stop=5,r=0):
    N_all = dict({})
    Lambda_all = dict({})
    indices,N,Lambda=get_vector_v1(data[:,r],number=number,state=state)
    N_all.update({(r,state):N})
    Lambda_all.update({(r,state):Lambda})
    if r==stop:
        return N_all,Lambda_all
    r=r+1
    i=state
    try:
       for index,n in zip(indices,N.values()):
           if n>0:
               N_all_,Lambda_all_=get_decomposition_v1(data[index],number=n,stop=stop,state=i,r=r)
               N_all.update({(r,i):N_all_})
               Lambda_all.update({(r,i):Lambda_all_})
           i=i+1
    except(AttributeError):
        print('Error')
    return N_all, Lambda_all
def get_vector_v1(c=np.array([]),number=0,state=0):
    i=state
    n=1
    N=[]
    index=[]
    N.append(number)
    if number==0:
        return np.array(index, dtype=np.int64),dict({}),dict({state:0})
    while n>0:
        indices=np.where(c==i)[0]
        n=len(np.where(c>i)[0])

        N.append(n)
        index.append(indices)
        i=i+1
    Nar=np.array(N)
    Ne=Nar[:Nar.shape[0]-1]-Nar[1:Nar.shape[0]]
    Lambda=Nar[1:Nar.shape[0]]/Nar[:Nar.shape[0]-1]
    ne=dict({})
    lamb=dict({})
    [ne.update({x[0]:x[1]}) for x in zip(np.arange(Ne.shape[0])+state,Ne)]
    [lamb.update({x[0]: x[1]}) for x in zip(np.arange(Lambda.shape[0]) + state, Lambda)]
    return index,ne,lamb
def get_columns_v1(data,i,A=[]):
    if (i==0):
        A.append(data[0])
        return
    if (len(data)==1):
        i=0
        get_columns(data, i, A)
        return
    #print(len(data))
    i=i-1
    for j in np.arange(len(data[0])):
        #print('j',j)
        #print('i',i)
        get_columns(data[j+1],i,A)
def get_ranges_v1(data):
    L=[]
    S=[]
    S.append(data)
    L.append(data[0])
    k=0
    while len(S)>0:
        k=k+1
        #print(k)
        L1=[]
        S1=[]
        for i in np.arange(len(S)):
            L1.append(get_range(S[i]))
            [S1.append(x) for x in S[i][1:]]
            #S1.append(S[i][1:])
        L.append(L1)
        S=S1
    return L
def get_ranges_v2(data):
    L=[]
    S=[]
    S.append(data)
    L.append(data[0])
    k=0
    while len(S)>0:
        k=k+1
        #print(k)
        L1=[]
        S1=[]
        for i in np.arange(len(S)):
            L1.append(get_range(S[i]))
            [S1.append(x) for x in S[i][1:]]
            #S1.append(S[i][1:])
        L.append(L1)
        S=S1
    return L
def get_range(data):
    if len(data)==1:
        return data
    else:
        L=[]
        for i,d in enumerate(data[0]):
            L.append(data[i+1][0])
    return L
def get_decomposition_v2(data,number=1,state=0,stop=5,r=0, hist=[], n_columns=10):
    #N_all = []
    #Lambda_all = []

    try:
        indices, N, Lambda = get_vector_n_v1(data[:, r], number=number, state=state)
        idx=[state]
        idx.extend(hist)
        hist=idx
        N_all = pd.DataFrame(index=(idx,),columns=np.arange(n_columns))
        Lambda_all = pd.DataFrame(index=(idx,),columns=np.arange(n_columns))
        N_all.loc[(idx,),indices.keys()]=N
        Lambda_all.loc[(idx,), indices.keys()] = Lambda

        if r == stop:
            return N_all, Lambda_all
        r = r + 1
        i = state
        # T_N=[]
        # T_Lambda=[]
        for key, n in zip(indices.keys(), N):
            index=indices[key]
            #hist_=[key]
            #hist_.extend(hist)
            #hist=hist_
            N_all_, Lambda_all_ = get_decomposition_v2(data[index], number=n, stop=stop, state=i, r=r,hist=hist, n_columns=n_columns)
            N_all=N_all.append(N_all_)
            Lambda_all=Lambda_all.append(Lambda_all_)
            i = i + 1
        # N_all.append(T_N)
        # Lambda_all.append(T_Lambda)
    except(TypeError):
        print("indices ", indices)
        print("N ", N)
    return N_all, Lambda_all
def fill_empty(N_all,Lambda_all):
    N_all.fillna(0, inplace=True)
    Lambda_all.fillna(0, inplace=True)
    new_columns={i:i+1 for i in Lambda_all.columns}
    Lambda_all.rename(columns=new_columns,inplace=True)
    def set_period(data):
        data['period'] = np.nan
        period = [1]
        i = 1
        while len(period) >= 1:
            period = [x for x in data.index if len(x) == i]
            data.loc[period, 'period'] = i
            i = i + 1
        return data
    Lambda_all=set_period(Lambda_all)
    N_all = set_period(N_all)
    return N_all,Lambda_all
def get_vector_n_v1(c=np.array([]),number=0,state=0):
    mask=c>=0
    c=c[mask]
    i=state
    n=1
    N=[]
    index=dict({})
    N.append(number)
    if number==0:
        return index,np.array(N),np.array([0])
    while n>0:
        index.update({i:np.where(c==i)[0]})
        n=len(np.where(c>i)[0])
        N.append(n)        #index.append(indices)
        i=i+1
    Nar=np.array(N,dtype=int)
    Ne=Nar[:Nar.shape[0]-1]-Nar[1:Nar.shape[0]]
    Lambda=Nar[1:Nar.shape[0]]/Nar[:Nar.shape[0]-1]
    return index,Ne,Lambda

def get_decomposition_v3(data,number=1,state=0,stop=5,r=0, hist=[], n_columns=10):
    N_all = []
    Lambda_all = []

    try:
        per=data[:, r]
        mask=per>=0
        if mask[mask==True].shape[0]>0:
            indices, N, Lambda = get_vector_n_v1(data[:, r], number=number, state=state)

            idx=[state]
            #print('idx=',len(idx))
            idx.extend(hist)
            #print('idx_ext=', len(idx))
            hist=idx
            body=np.zeros(shape=n_columns,dtype=int)
            body_l=np.zeros(shape=n_columns,dtype=float)
            ind=list(indices.keys())
            body[ind]=N
            body_l[ind] =Lambda
            N_all=[(tuple(idx),body)]
            Lambda_all = [(tuple(idx), body_l)]
            #print(len(indices), ' number =', number, 'state=', state,'idx=',len(idx))

            if r == stop:
                return N_all, Lambda_all
            r = r + 1
            i = state
            for key, n in zip(indices.keys(), N):
                index=indices[key]
                N_all_, Lambda_all_ = get_decomposition_v3(data[index], number=n, stop=stop, state=i, r=r,hist=hist, n_columns=n_columns)
                N_all.extend(N_all_)
                Lambda_all.extend(Lambda_all_)
                i = i + 1
    except(TypeError):
        print("indices ", indices)
        print("N ", N)


    return N_all, Lambda_all

def get_vector(c=np.array([]),number=0,state=0):
    mask=c>=0
    tilde=c[mask]
    i=state
    n=1
    N=[]
    index=dict({})
    N.append(number)
    if number==0:
        return index,np.array(N),np.array([0])
    while n>0:
        index.update({i:np.where(c==i)[0]})
        n=len(np.where(tilde>i)[0])
        N.append(n)        #index.append(indices)
        i=i+1
    Nar=np.array(N,dtype=int)
    Ne=Nar[:Nar.shape[0]-1]-Nar[1:Nar.shape[0]]
    return index,Ne
def get_lambda(lamb=np.array([]),number=0,state=0):
    mask=lamb>=0
    lamb=lamb[mask]
    i=state
    n=1
    N=[]
    N.append(number)
    if number==0:
        return np.array([0])
    while n>0:
        n=len(np.where(lamb>i)[0])
        N.append(n)        #index.append(indices)
        i=i+1
    La = np.array(N, dtype=int)
    Lambda=La[1:La.shape[0]]/La[:La.shape[0]-1]
    return Lambda
def to_array(function):
    def wrapper(*args,**kwargs):
        result=function(*args,**kwargs)
        try:
            result=np.array(result)
        except AttributeError:
            print('not iterable value')
        return result
    wrapper.__name__=function.__name__
    wrapper.__doc__ = function.__doc__
    return wrapper
@to_array
def get_lambda_decomposition(data,number=1,state=0,stop=5,r=0, hist=np.array([]),top=3,size=10):
    epoch = []
    try:
        per=data[:,r]
        mask=per>=0
        if mask[mask==True].shape[0]>0:
            indices, N = get_vector(data[:, r], number=number, state=state)

            for key, n in zip(indices.keys(), N):
                w = r + 1
                idx=np.array([key])
                idx=np.hstack((hist,idx))
                if size>idx.shape[0]:
                    array=np.empty((size-idx.shape[0]))
                    array.fill(-1)
                    array=np.hstack((idx,array))
                else:
                    array=idx[:size]
                hist_=idx
                index=indices[key]

                sub=np.max(data[index,w:w+top],axis=1)
                submask=sub>=0
                #if key==9:
                    #print('index',index)
                    #print('w', w)
                    #print('sub',sub)
                    #print('array', submask[submask==True].shape[0])
                    #print('data',data[index,w:w+top])
                    #print('______')
                if submask[submask==True].shape[0]>0:
                    lambd=get_lambda(sub, number=n, state=key)
                    posmask=lambd>0
                    lamb = np.cumprod(lambd[posmask]).sum()
                    shape=idx.shape[0]
                    array=np.hstack((array,n,lamb,shape))
                    epoch.append(array)
                    if r+top<=stop:
                        r_=r+1
                        epoch_= get_lambda_decomposition(data[index], number=n, stop=stop, state=key, r=r_,hist=hist_,top=top,size=size)
                        epoch.extend(epoch_)

            if r == stop:
                return epoch
    except(TypeError):
        print("indices ", indices)
        print("N ", N)
    except IndexError:
        print('index_error',r)
        return epoch
    except(ValueError):
        print("indices ", indices)
        print('w',w,w+r)

    return epoch

def get_smoothed_array(x=np.array([]),n=5,stop=-1):
    def get_smoothed(x=np.array([]), n=5, stop=-1):
        def pos(x=np.array([])):
            if x < 0:
                return 0
            else:
                return x

        d = x[0]
        delta = pos(d - n)
        x[0] = x[0] - delta
        i = 1
        while i < x.shape[0]:
            if x[i] == stop:
                break
            d = (x[i] - delta) - x[i - 1]
            delta = pos(d - n) + delta
            d = min(d, n)
            # print(i,d,delta)
            x[i] = x[i - 1] + d
            i += 1
        return x
    array=[]
    i=0
    while i<x.shape[0]:
        y=get_smoothed(x[i],n=n,stop=stop)
        array.append(y)
        i+=1
    return np.array(array)

def get_df(data,i=1):
    n = 1
    #i = 1
    frames = dict()
    period = [x for x in data if len(x[0]) == 0]
    if (len(period))>0:
        frames.update({0: data[0][1]})
    while n > 0:
        period = [x for x in data if len(x[0]) == i]
        n = len(period)
        if n == 0:
            break
        index = [x[0] for x in period]
        arrays = [x[1] for x in period]
        index = pd.MultiIndex.from_tuples(index)
        period = pd.DataFrame(data=arrays, index=index)
        frames.update({i: period})
        i = i + 1
    return frames


def get_cumprod(data):
    mask = data > 0
    cumprod = np.sum(np.cumprod(data[mask], axis=1), axis=1)
    return cumprod


def LambdaNdot(cumLambda, N,metric=metrics()):
    s = 0
    if len(N.shape)==1:
        cumlambda = cumLambda[(0,)]
        s = s + cumlambda * N[0]
        return s
    indices=np.array(cumLambda.index)

    for i in N.index:
        mask = N.loc[i] > 0
        columns = np.where(mask)[0]
        for j in columns:
            n = N.loc[i, j]
            index = [j]
            index.extend(i)
            try:
                cumlambda = cumLambda[tuple(index)]
            except(KeyError):
                trajectories=get_nearest_trajectory(np.array(tuple(index)),indices,metric=metric)
                cl=0
                for t in trajectories:
                    cl=cl+cumLambda[t]
                if trajectories.shape[0]>0:
                    cumlambda=cl/trajectories.shape[0]
                else: cumlambda=0
            s = s + cumlambda * n
    return s
def LambdaNvector(cumLambda, N,metric=metrics()):
    s = []
    if len(N.shape)==1:
        cumlambda = cumLambda[(0,)]
        s.append(((0,),N[0],cumlambda * N[0]))
        return s
    indices=np.array(cumLambda.index)

    for i in N.index:
        mask = N.loc[i] > 0
        columns = np.where(mask)[0]
        for j in columns:
            n = N.loc[i, j]
            index = [j]
            index.extend(i)
            try:
                cumlambda = cumLambda[tuple(index)]
            except(KeyError):
                trajectories = get_nearest_transfer_trajectories(np.array(tuple(index)),indices, metric=metric)
                if trajectories is None:
                    trajectories=get_nearest_trajectory(np.array(tuple(index)),indices,metric=metric)
                cl=0
                for t in trajectories:
                    cl=cl+cumLambda[t]
                if trajectories.shape[0]>0:
                    cumlambda=cl/trajectories.shape[0]
                else: cumlambda=0
            s.append((tuple(index),n, cumlambda * n))
    return s
def get_nearest_trajectory(trajectory=np.array([]), trajectories=np.array([]), metric=metrics().euclidian(),weight=None):
    distances = [metric(x, trajectory) for x in trajectories]
    minim = np.min(distances)
    mask = np.where(distances == minim)[0]
    return trajectories[mask]
def get_nearest_transfer_trajectories(trajectory=np.array([]), trajectories=np.array([]), metric=metrics().euclidian(),weight=None):
    sub=[x for x in trajectories if x[0]==trajectory[0]]
    sub=np.array(sub)
    if len(sub)==0:
        return None
    distances = [metric(x, trajectory) for x in sub]
    minim = np.min(distances)
    mask = np.where(distances == minim)[0]
    indices=[tuple(x) for x in sub[mask]]

    return  pd.MultiIndex.from_tuples(indices)
class sections:
    def __init__(self):
        self.N = [((), np.array([], dtype=int))]
        self.Lambda = [((), np.array([], dtype=float))]
        self.P= [((), np.array([], dtype=float))]
        self.Pf = pd.DataFrame([])
        self.Nf = pd.DataFrame([])
        self.Lambdaf = pd.DataFrame([])
        self.s_matrix = pipe()
        self.number=0
        self.metric=metrics().euclidian()
    def decomposition(self):
        self.number = self.s_matrix.empty+self.number
        matr = self.s_matrix.intervals[:, 2:]
        #np.save(path + 'smatrix.npy', self.s_matrix.intervals)
        self.N,self.Lambda = get_decomposition_v3(matr, number=self.number, state=0, stop=matr.shape[1] - 1,
                                                n_columns=matr.max() + 1)
        body = np.zeros(matr.max() + 1)
        body[0] = self.number
        ext = [((), body)]
        ext.extend(self.N)
        self.N = ext
        self.to_p()

    def to_p(self):
        P = []
        n = len(self.N)
        i = 0
        while i < n:
            summ = self.N[i][1].sum()
            if summ > 0:
                #P.append((self.N[i][0],self.N[i][1]))
                P.append((self.N[i][0], self.N[i][1] / summ))
            #else:
                #P.append((self.N[i][0], self.N[i][1] / summ))
            i=i+1
        self.P=tuple(P)

    def to_df(self):
        self.Nf=get_df(self.N)
        self.Lambdaf=get_df(self.Lambda)

        self.Pf=get_df(self.P)
    def predict(self, N, i,step=0,summ=True):
        if sum==True:
            p=0
            cumLambda = self.Lambdaf[i + 1]
            p=LambdaNdot(get_cumprod(cumLambda), N, metric=self.metric)
            for s in np.arange(step):
                Nt=get_next_n(N,self.Pf,step=1,start=i+s,metric=self.metric)
                cumLambda = self.Lambdaf[i + s + 2]
                p = p+LambdaNdot(get_cumprod(cumLambda), Nt[i+s+1], metric=self.metric)
                N = Nt[i+s+1]
            return p
        else:
            p=dict()
            cumLambda = self.Lambdaf[i + 1]
            p.update({(i,0):LambdaNvector(get_cumprod(cumLambda), N, metric=self.metric)})
            for s in np.arange(step):
                Nt=get_next_n(N,self.Pf,step=1,start=i+s,metric=self.metric)
                cumLambda = self.Lambdaf[i + s + 2]
                p.update({(i+s,1):LambdaNvector(get_cumprod(cumLambda), Nt[i+s+1], metric=self.metric)})
                N = Nt[i+s+1]
            return p



    def predict_v1(self, N, i,step=1):
        p=0
        cumLambda=self.Lambdaf[i+step]
        index_len=len(cumLambda.index[0])
        p=LambdaNdot(get_cumprod(cumLambda),N,metric=self.metric)
        return p
    def counter(self,summ=True):
        if summ==True:
            matrix=self.s_matrix.intervals[:,2:]
            zero=np.zeros(matrix.shape[0])
            res=[]
            for i in np.arange(matrix.shape[1]):
               a=matrix[:,i]-zero
               res.append(a.sum())
               zero=matrix[:,i]
            return np.array(res)
        else:
            counters=dict()
            for k in self.Nf.keys():
                if k>0:
                    period=self.Nf[k]
                    coun = pd.DataFrame(index=period.index)
                    cols=period.columns
                    for i in period.index:
                        coun.loc[i,0]=np.dot(period.loc[i],cols-i[0])
                    counters.update({k:coun})
            return counters





class LambdaPredictor:
    def __init__(self):
        self.data=pd.DataFrame([])
        self.time_range=np.array([])
        self.s=sections()
        self.s_=sections()
        self.lenght=100
        self.flow=True
        self.groupby = 'ID простого участка'

    def fit(self,data=pd.DataFrame([]),time_range=np.array([]),lenght=100,flow=True,groupby='ID простого участка'):
        self.lenght=lenght
        self.flow=flow
        self.groupby=groupby
        self.data=data
        self.time_range=time_range
        self.s.s_matrix,self.s_.s_matrix=get_joined_matrices(self.data,time_range=self.time_range,lenght=self.lenght,flow=self.flow,groupby=self.groupby)
        self.s.decomposition()
        self.s_.decomposition()
        self.s.to_df()
        self.s_.to_df()

class Lambda_np_Predictor:
    def __init__(self):
        self.data=pd.DataFrame([])
        self.time_range=np.array([])
        self.s=sections()
        self.s_=sections()
        self.lenght=100
        self.flow=True
        self.groupby = 'ID простого участка'

    def fit(self,data=pd.DataFrame([]),time_range=np.array([]),lenght=100,flow=True,groupby='ID простого участка'):
        self.lenght=lenght
        self.flow=flow
        self.groupby=groupby
        self.data=data
        self.time_range=time_range
        self.s.s_matrix,self.s_.s_matrix=get_joined_matrices_by_np(self.data,time_range=self.time_range,lenght=self.lenght,flow=self.flow,groupby=self.groupby)
        self.s.decomposition()
        self.s_.decomposition()
        self.s.to_df()
        self.s_.to_df()


def get_next_step(N, P, i, metric=metrics()):
    Next = []
    indices = np.array(P[i+1].index)
    if len(N.shape)==1:
        index=(0,)
        n=N[0]
        p = P[i + 1].loc[index].values
        prob = p * n
        Next.append((index,prob))
        return get_df(Next, 1)
    try:
        for k in N.index:
            for c in N.columns:
                n = N.loc[k, c]
                if n > 0:
                    index = [int(c)]
                    index.extend(k)
                    index = tuple(index)
                    try:
                        p = P[i + 1].loc[index].values
                        prob=p * n
                    except(KeyError):
                        trajectories = get_nearest_transfer_trajectories(np.array(tuple(index)), indices, metric=metric)
                        if trajectories is not None:
                            p=P[i+1].loc[trajectories].sum(axis=0)/trajectories.shape[0]
                            prob=p*n
                        else:
                            try:
                                prob=np.zeros(P[i+1].shape[1])
                                prob[index[0]]=n
                            except(IndexError):
                                prob[-1] = len(prob)
                                #print(len(N.columns))
                                #print('prob ',prob)
                                #print('index ',index)
                                #print('i ',i)

                        #print(trajectories.shape[0])
                        #print(P[i+1].loc[trajectories].sum(axis=0))
                    Next.append((index, prob))
    except(KeyError):
        return dict({i: N})
    return get_df(Next, len(index))


def get_next_n(N, P, step=1, start=1, metric=metrics()):
    if step==0: return dict({start:N})
    Next = get_next_step(N, P, i=start, metric=metric)
    if step == 1:
        return Next
    step = step - 1
    start = start + 1
    Next_ = get_next_n(Next[start], P, step=step, start=start,metric=metric)
    return Next_


def get_weibull(L=100, sigma=(1, 30),tsigma=2, age=10, interval=100, state=42, nmin=5, nmax=15, k=1.5, Lambda=1):
    # n = np.ceil(L / interval)
    rnd = np.random.RandomState(state)
    # centers = rnd.randint(low=sigma[1], high=L-sigma[1], size=int(n))
    centers = get_int_range(L, interval, random=rnd)

    for i, c in enumerate(centers):
        teta = rnd.randint(low=nmin, high=nmax)
        # print(teta)
        X = rnd.randn(teta)
        wb = Lambda * rnd.weibull(k, teta)
        delta = tsigma * rnd.randn()
        start = age + delta
        X = sigma[1] * X + c
        wr = get_wb(start, wb)
        # print(X.shape)
        # print(wr.shape)
        X = np.hstack((X.reshape(-1, 1), wr.reshape(-1, 1)))
        # print(X.shape)
        # X[:, 1] = sigma[0] * X[:, 1] + age + delta
        empty = np.empty(X.shape[0])
        empty.fill(i)
        X = np.hstack((X, empty.reshape(-1, 1)))
        # print(X.shape)
        if i == 0:
            array = X
        else:
            array = np.vstack((array, X))
        mask = (array[:, 0] <= L) & (array[:, 0] >= 0)

    return array[mask]


def get_wb(age=10, wb=np.array([])):
    wr = []
    wr.append(age)
    i = 1
    for w in wb:
        wr.append(wr[i - 1] + w)
        i = i + 1
    return np.array(wr[:-1])
def get_int_range(L,interval,random=np.random):
    n=np.ceil(L/interval)
    x=[interval*y for y in np.arange(n)]
    X=[]
    for w in np.arange(len(x))[:-1]:
        X.append(random.randint(low=x[w],high=x[w+1]))
    return np.array(X)
def trajectory_result(x,i):
    result=[]
    indices=[w[0] for w in x[(i,0)]]
    Q=[w[1] for w in x[(i,0)]]
    for ind,q in zip(indices,Q):
        s=0
        for k in x.keys():
            step=x[k]
            ads=[w[2] for w in step if w[0][-len(ind):]==ind]
            s=s+sum(ads)
        result.append((ind,q,s))
    return result
def trajectory_sum(x,i,step):
    try:
        ind=None
        result=[]
        indices=x[i].index
        for ind in indices:
            k=0
            s=0
            while k<=step:
                idx=[j for j in x[i+k].index if j[-len(ind):]==ind]
                s=s+x[i+k].loc[idx,0].sum()
                k=k+1
            result.append((ind,s))
    except(KeyError):
        if ind is None:
            return result
        else:
            result.append((ind, s))
        return result
    return result
def make_predictions_v1(model,test,lenght, time_range, step=[2,3]):
    time_range_loc=time_range
    results=dict({})
    t=0
    for ind in test['ID простого участка'].value_counts().keys():
        mask=test['ID простого участка']==ind
        pre_model=Lambda_np_Predictor()
        pre_model.fit(test[mask],time_range=time_range_loc,lenght=lenght,flow=True,groupby='ID простого участка')
        r_count=pre_model.s_.counter(summ=False)
        rcount=pre_model.s.counter(summ=False)
        ste=dict({})
        t=t+1
        try:
            for st in step:
                results_np=dict({})
                for j in np.arange(time_range_loc.shape[0]):
                    try:
                        s_=model.s_.predict(pre_model.s_.Nf[j],j,step=st,summ=False)
                        s=model.s.predict(pre_model.s.Nf[j],j,step=st,summ=False)

                        sr=trajectory_result(s,j)
                        s_r=trajectory_result(s_,j)

                        r=trajectory_sum(rcount,j+1,st)
                        r_ = trajectory_sum(r_count, j + 1, st)
                        if len(r)==0:
                            r=[(x[0],np.nan) for x in sr]
                        if len(r_)==0:
                            r_=[(x[0],np.nan) for x in s_r]
                        results_np.update({j:((r,sr),(r_,s_r))})
                        #print('Results_keys()', results_np.keys())
                    except(KeyError):
                        print('Error in ',j )
                        #print('results_keys()', results_np.keys())
                        #ste.update({st:results_np})
                ste.update({st:results_np})
        except(KeyError):
            print('error in ',j )
        results.update({ind:ste})
    return results
def make_predictions_v2(train,test,get_out, step=[2,3], free_count=0, lenght=100,div=1):
    #time_range_loc=time_range
    results=dict({})
    sub_mask=test['Дата аварии'].dt.year==get_out.year
    t=0
    for ind in test[sub_mask]['ID простого участка'].value_counts().keys():
        mask=test['ID простого участка']==ind
        mask_t=train['ID простого участка']==ind
        enter=test[mask].iloc[0]['Дата ввода']
        S=test[mask].iloc[0]['S']
        rs=1
        delta=rs/S
        time_range_loc=get_time_range(enter=enter,get_out=get_out,S=S, step=div)
        #print(time_range_loc)
        #print('loc:',time_range_loc)
        time_range_exp=time_range_loc
        for i in np.arange(max(step)+1):
            time_range_exp=np.hstack((time_range_exp,time_range_exp[-1]+delta))
        #print('expand:',time_range_exp)
        model=Lambda_np_Predictor()
        model.s.number=free_count
        model.s_.number=free_count
        model.fit(train,time_range=time_range_exp,lenght=lenght,flow=True,groupby='ID простого участка')
        model.s_.metric=metrics().euclidian
        model.s.metric=metrics().euclidian
        pre_model=Lambda_np_Predictor()
        pre_model.fit(test[mask],time_range=time_range_exp,lenght=lenght,flow=True,groupby='ID простого участка')
        r_count=pre_model.s_.counter(summ=False)
        rcount=pre_model.s.counter(summ=False)
        ste=dict({})
        t=t+1
        #print('tr_len',len(time_range_exp))
        try:
            for st in step:
                results_np=dict({})
                for j in np.arange(1,time_range_loc.shape[0]+1)[-1:]:
                    s = model.s.predict(pre_model.s.Nf[j], j, step=st, summ=False)
                    s_ = model.s_.predict(pre_model.s_.Nf[j], j, step=st, summ=False)
                    sr = trajectory_result(s, j)
                    s_r = trajectory_result(s_, j)
                    r = trajectory_sum(rcount, j + 1, st)
                    r_ = trajectory_sum(r_count, j + 1, st)
                    if len(r)==0:
                        r = [(x[0], np.nan) for x in sr]
                    if len(r_)==0:
                        r_ = [(x[0], np.nan) for x in s_r]
                    results_np.update({j: ((r, sr), (r_, s_r))})

                ste.update({st:results_np})
            results.update({ind: ste})
        except(KeyError):
            print('error in ',j )
    return results
def make_predictions_v3(train,test,get_out, step=[2,3], free_count=0, lenght=100,div=1):
    #time_range_loc=time_range
    results=dict({})
    sub_mask=test['Дата аварии'].dt.year==get_out.year+1
    t=0
    for ind in test[sub_mask]['ID простого участка'].value_counts().keys():
        mask=test['ID простого участка']==ind
        mask_t=train['ID простого участка']==ind
        enter=test[mask].iloc[0]['Дата ввода']
        S=test[mask].iloc[0]['S']
        #rs=1
        delta=div/S
        dates=dict({})


        for date in test[mask & sub_mask]['Дата аварии']:
            time_range_loc=get_time_range(enter=enter,get_out=date,S=S, step=div)
            time_range_exp=time_range_loc
            for i in np.arange(max(step)+1):
               time_range_exp=np.hstack((time_range_exp,time_range_exp[-1]+delta))
            model=Lambda_np_Predictor()
            model.s.number=free_count
            model.s_.number=free_count
            model.fit(train,time_range=time_range_exp,lenght=lenght,flow=True,groupby='ID простого участка')
            model.s_.metric=metrics().euclidian
            model.s.metric=metrics().euclidian
            pre_model=Lambda_np_Predictor()
            pre_model.fit(test[mask],time_range=time_range_exp,lenght=lenght,flow=True,groupby='ID простого участка')
            r_count=pre_model.s_.counter(summ=False)
            rcount=pre_model.s.counter(summ=False)
            ste=dict({})
            t=t+1
            try:
                for st in step:
                    results_np=dict({})
                    for j in np.arange(1,time_range_loc.shape[0]+1)[-1:]:
                        #print(j)
                        #print(date)
                        #print(pre_model.s.Nf.keys())
                        #print(model.s.Lambdaf.keys())
                        #print(model.s.Pf.keys())
                        #print(model.s.Pf[j+1].shape)
                        #print(pre_model.s.Nf[j].shape)
                        s = model.s.predict(pre_model.s.Nf[j], j, step=st, summ=False)
                        s_ = model.s_.predict(pre_model.s_.Nf[j], j, step=st, summ=False)
                        sr = trajectory_result(s, j)
                        s_r = trajectory_result(s_, j)
                        r = trajectory_sum(rcount, j + 1, st)
                        r_ = trajectory_sum(r_count, j + 1, st)
                        if len(r)==0:
                            r = [(x[0], np.nan) for x in sr]
                        if len(r_)==0:
                            r_ = [(x[0], np.nan) for x in s_r]
                        results_np.update({j: ((r, sr), (r_, s_r))})
                    ste.update({st:results_np})
            except(KeyError):
                print('error in ',j )
            dates.update({date:ste})
        results.update({ind: dates})
    return results

def get_counter_of_empty(empty,data, mask):
    IDs=empty[mask]['ID простого участка']
    free=IDs[~IDs.isin(data['ID простого участка'].value_counts().keys())]
    free_count=empty.loc[free.index,'L'].sum()
    return free_count


def merge_repairs(true, synthetic, epsilon=0.5):
    result = np.array([], dtype=[('a', float), ('b', float), ('date', np.datetime64)]).reshape(-1, 3)
    for s in synthetic:
        print(s)
        res = split_repairs(true, s.reshape(-1, 3), epsilon=epsilon)
        print(res)
        result = np.vstack((result, res))
    return result


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
            ispt = interseption(r, synt)
            if ispt.shape[0] > 0:
                residual = GetSetsResidual(synt.reshape(-1, 3), r)[:-1]
                # print(residual)
            else:
                residual = synt.reshape(-1, 3)
                # print(residual)
            res = split_repairs(true[1:], residual, epsilon=epsilon)
        result = np.vstack((result, res))
    return result


def get_merged_repairs(true, synthetic, epsilon=0.5):
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


def get_merged_features(data=pd.DataFrame([]),by='merge',group_id='new_id',size=1):
    mask=data[by]<0
    free=data[mask]
    clusters=data[~mask]
    agg=clusters.groupby(group_id)
    spared=[]
    for group in agg:
        aggc=group[1].groupby('merge')
        for cluster in aggc:
            shape=cluster[1].shape[0]
            n=size
            if shape<size:
                n=shape
            row=cluster[1].values[:n]
            spared.append(row)
    spared=np.vstack(spared)
    clustered=pd.DataFrame(spared,columns=data.columns)
    df=free.append(clustered,ignore_index=True)
    return df


def wtp_approach(data, mask, xfield='Наработка до отказа', yfield='Обводненность'):
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
                x = dict({'x1': x1, 'x2': x2})
                d = dict({'d1': q, 'd2': p})
                line = linear_transform(x=x, d=d)
                if q > 0:
                    # data.loc[indices,'Обводненность']=data.loc[[a,b],'Обводненность'].mean()
                    data.loc[indices, yfield] = data.loc[indices, xfield].apply(lambda x: line.value(x))
                else:
                    data.loc[indices, yfield] = data.loc[b, yfield]
        i = i + 1

def length_approach(data, xfield='L',yfield='Адрес от начала участка',ident='ID простого участка'):
    gby=data.groupby(ident)
    for i,group in enumerate(gby):
        L=group[1][xfield].iloc[0]
        amax=group[1][yfield].max()
        if amax>L:
            data.loc[group[1].index,xfield]=amax


def plot_id(group, cfield='cluster', save=False, path='D:\\ml\\pictures\\'):
    def get_cluster_shape(cluster):
        xmin = cluster['Адрес от начала участка'].min()
        xmax = cluster['Адрес от начала участка'].max()
        ymin = cluster['Наработка до отказа'].min()
        ymax = cluster['Наработка до отказа'].max()
        dmin = cluster['Дата аварии'].min()
        dmax = cluster['Дата аварии'].max()
        return (xmin, xmax), (ymin, ymax), (dmin, dmax)
    sid = group[['Наработка до отказа', 'Адрес от начала участка']].values
    enter = group['Дата ввода'].iloc[0]
    steel = group['Материал трубы'].iloc[0]
    field = group['Название месторождения'].iloc[0]
    ID = group['ID простого участка'].iloc[0]
    S = group['S'].iloc[0]
    repairs = get_unical_repairs(group, scale=0)
    repairs['b'] = repairs['Адрес'] + repairs['Длина']
    synt_reps = repairs[['Адрес', 'b', 'Дата ремонта']].values
    repairs['age'] = (repairs['Дата ремонта'] - enter) / np.timedelta64(1, 'Y')
    # scan=DBSCAN(min_samples=2,eps=40, metric=metric.euclidian).fit(sid)
    labels = group[cfield].values
    plt.figure(figsize=(12, 8))
    for i in repairs.index:
        age = repairs.loc[i, 'age']
        a = repairs.loc[i, 'Адрес']
        b = a + repairs.loc[i, 'Длина']
        plt.plot((a, b), (age, age), color='red', linestyle='solid')
    repairs = GetUnicalRepairs(group, scale=0)
    repairs['b'] = repairs['Адрес'] + repairs['Длина']
    true_reps = repairs[['Адрес', 'b', 'Дата ремонта']].values
    repairs['age'] = (repairs['Дата ремонта'] - enter) / np.timedelta64(1, 'Y')
    # scan=DBSCAN(min_samples=2,eps=40, metric=metric.euclidian).fit(sid)
    labels = group[cfield].values
    col_labels = ['Возр', 'Адр', 'L']
    row_labels = []
    table_vals = []

    # plt.figure(figsize=(12,8))
    for i in repairs.index:
        age = repairs.loc[i, 'age']
        a = repairs.loc[i, 'Адрес']
        b = a + repairs.loc[i, 'Длина']
        plt.plot((a, b), (age, age), color='blue', linestyle='solid')
        row_labels.append(str(i))
        table_vals.append(('{0:.2f}'.format(age), '{0:.0f}'.format(a), '{0:.1f}'.format(b - a)))

    if len(row_labels) > 0:
        the_table = plt.table(cellText=table_vals,
                              colWidths=[0.1] * 3,
                              rowLabels=row_labels,
                              colLabels=col_labels,
                              loc='center left')
    # plt.text(12,3.4,'Ремонты',size=8)

    merged = merge_repairs(true_reps, synt_reps, epsilon=0.5)
    for i in np.arange(merged.shape[0]):
        age = (merged[i, 2] - enter) / np.timedelta64(1, 'Y')
        a = merged[i, 0]
        b = merged[i, 1]
        plt.plot((a, b), (age, age), color='green', linestyle='dashed', alpha=1, linewidth=3)
    plt.title(str(field) + '; ID ' + str(ID) + '; Ввод: ' + str(enter.year) + '; ' + str(steel) + '; S=' + str(S))
    mask = labels == -1
    clustered = group[~mask]
    plt.scatter(sid[~mask][:, 1], sid[~mask][:, 0], c=labels[~mask])
    plt.scatter(sid[mask][:, 1], sid[mask][:, 0], c='red', marker='^')
    plt.xlabel("Адрес от начала участка")
    plt.ylabel('Наработка')
    for c in clustered[cfield].value_counts().keys():
        size_x, size_y, size_d = get_cluster_shape(clustered[clustered[cfield] == c])
        plt.plot((size_x[0], size_x[0]), (size_y[0], size_y[1]), color='black', linestyle='dashed')
        plt.plot((size_x[0], size_x[1]), (size_y[0], size_y[0]), color='black', linestyle='dashed')
        # plt.plot((size_x[0],size_x[1]),(size_y[1],size_y[1]),color='black')
        plt.plot((size_x[1], size_x[1]), (size_y[0], size_y[1]), color='black', linestyle='dashed')
        plt.annotate(str(c), xy=(size_x[0], size_y[1]), verticalalignment='bottom', horizontalalignment='left')
    blue_patch = mpatches.Patch(color='blue', label='Ремонты фактические')
    red_patch = mpatches.Patch(color='red', label='Ремонты синтетеческие')
    green_patch = mpatches.Patch(color='green', label='После объединения')
    plt.legend(handles=[blue_patch, red_patch, green_patch])
    if save:
        plt.savefig(path + str(ID) + '.png')
    plt.show()

def get_splitted(data,delta=3,by='ID простого участка'):
    data['a']=0
    data['b']=data['L']
    data['new_id']=data[by].astype('str')
    for ID in data['ID простого участка'].value_counts().keys():
        splited=get_splited_by_repairs(data,ID=ID, delta=delta)
    data['L,м']=data['b']-data['a']
    data['Адрес от начала участка (new)']=data['Адрес от начала участка']-data['a']
    data['Наработка до отказа(new), лет']=(data['Дата аварии']-data['Дата ввода'])/np.timedelta64(1,'Y')


class Generator:
    def __init__(self, model):
        self.model = model
        self.next = 0
        self.prev = 0
        self.top=0
        self.max=10
        self.next_values = np.array([])

    def get_value(self, x=np.array([])):
        self.prev = x[-1]

        y = self.model.predict(x.reshape(1, x.shape[0])).reshape(-1)
        #print('y',y)
        if y==self.prev:
            y=self.top

            print('same')
            #return None

        if y < self.prev:
            dt = self.prev - y
            y = self.prev + dt
        self.next = y
        #print('prev', self.prev,'next',self.next)
        self.next_values = np.append(self.next_values, self.next)

    def get_next(self, x=np.array([]), n=1):
        y = x
        self.next_values = np.array([])
        self.next = 0
        while n > 0:
            self.get_value(y)
            y = self.move(y, self.next)
            n = n - 1

    def get_count(self, x=np.array([]), top=1):
        y = x
        self.next_values = np.array([])
        self.next = 0
        self.top=top
        i = 1
        while self.next < self.top:
            self.get_value(y)
            y = self.move(y, self.next)
            if (self.next >= top)|(i>self.max):
                return i
            i = i + 1
        return i
        # n=n-1

    def move(self, x=np.array([]), y=0):
        teta = x[1:]
        teta = np.append(teta, y)
        return teta


class Generator_v1:
    def __init__(self, model):
        self.model = model
        self.x = np.array([], dtype=float)
        self.gindices = np.array([], dtype=int)
        self.count = np.array([], dtype=int)
        self.mask = np.array([], dtype=bool)
        self.top = np.array([], dtype=float)
        self.prev = np.array([], dtype=float)
        self.columns = np.array([], dtype=int)

    def get_counts(self, x=np.array([], dtype=float), top=np.array([], dtype=float)):
        self.x = x
        self.top = top
        self.gindices = np.arange(self.x.shape[0])
        self.columns = np.arange(self.x.shape[1])[1:]
        self.mask = np.ones(self.x.shape[0], dtype=bool)
        self.prev = self.x[:, -1].reshape(-1)

        self.count = np.zeros(self.x.shape[0], dtype=int)
        self.indices = self.gindices
        # self.count.np.nan
        # x=self.x
        i = 1
        while i < 10:
            # print(i)
            y = self.model.predict(self.x).reshape(-1)
            delta = np.abs(y - self.prev)
            y = self.prev + delta
            emask = y == self.prev
            # print(emask.shape)
            # print(self.top.shape)
            # print(y.shape)
            y[emask] = self.top[emask]
            self.mask = y < self.top
            # print('mask',self.mask.shape)
            # print('indices',self.indices.shape)
            index = self.indices[self.mask]
            index1 = self.indices[~self.mask]
            self.count[self.gindices[index1]] = i

            # print('index',index.shape)

            x0 = self.x[:, self.columns]
            x0 = x0[index, :]
            x1 = y[index].reshape(-1, 1)
            # print(x0.shape)
            # print(x1.shape)
            self.x = np.hstack((x0, x1))
            # self.x=y[index]
            self.top = self.top[index]
            self.prev = x1.reshape(-1)
            self.gindices = self.gindices[index]
            self.indices = np.arange(index.shape[0])
            i = i + 1
            if self.x.shape[0] == 0:
                break
        mask = self.count == 0
        self.count[mask] = i
        return self.count


class Regressor:
    def __init__(self,model=None):
        self.model=model
        self.items=[]
        self.top_borders=[]



    def forward_predict(self,data=pd.DataFrame([]), indices=np.array([], dtype=int),
                        interval=100, period=1, length=100, xfield='Адрес от начала участка (new)',
                        taufield='Наработка до отказа(new), лет'):
        def predict_and_cut(data=pd.DataFrame([]), indices=np.array([], dtype=int),
                            index=0, interval=100, period=1, length=100, xfield='Адрес от начала участка (new)',
                            taufield='Наработка до отказа(new), лет'):
            x = data.loc[index, xfield]
            tau = data.loc[index, taufield]
            a, b = get_interval(interval, 1, x, length, expand=False)
            data.loc[index, 'lbound'] = a
            data.loc[index, 'rbound'] = b

            item, top = get_easy_binary_item(data.loc[indices], date=period, index=index, ints=interval)
            if item is not None:
                self.items.append(item)
                self.top_borders.append(top)
                predict = self.model.get_counts(item, top, stop=5)
                if predict > 0.5:
                    cut = True
                else:
                    cut = False
                data.loc[index, 'predicted'] = predict
                lmask = (data.loc[indices, xfield] >= a)
                rmask = (data.loc[indices, xfield] <= b)
                xmask = lmask & rmask
                amask = (~xmask) & (~lmask)
                bmask = (~xmask) & (~rmask)
                lindex = amask[amask == True].index
                rindex = bmask[bmask == True].index
                tmask = data.loc[indices, taufield] <= tau
                mask = xmask & tmask
                ymask = xmask & (~tmask)
                yindex = ymask[ymask == True].index
                if cut:
                    data.loc[index, 'prevent'] = yindex.shape[0]
                    data.loc[yindex, 'marked'] = index
                    data.loc[yindex, 'lost'] = False
                    llength = a
                    rlength = length - b
                    data.loc[rindex, xfield] = data.loc[rindex, xfield] - b
                    data.loc[rindex, 'L,м'] = length - b
                    data.loc[lindex, 'L,м'] = a
                    return ((lindex, llength), (rindex, rlength))
                else:
                    top = xmask & (~tmask)
                    passed = top[top == True].index
                    data.loc[passed, 'lost'] = True
                    return ((indices, length),)
            else:
                return ((np.array([]), 0),)



        if indices.shape[0] > 0:
            splitted = predict_and_cut(data=data, indices=indices, index=indices[0], interval=interval,
                                       period=period, length=length, xfield=xfield, taufield=taufield)
            if len(splitted) == 1:
                self.forward_predict(data=data, indices=indices[1:], interval=interval,
                                period=period, length=length, xfield=xfield, taufield=taufield)
            else:
                for s in splitted:
                    self.forward_predict(data=data, indices=s[0], interval=interval,
                                    period=period, length=s[1], xfield=xfield, taufield=taufield)


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
class Regressor_v1:
    def __init__(self,model=None):
        self.model=model
        self.items=[]
        self.top_borders=[]



    def forward_predict(self,data=pd.DataFrame([]), indices=np.array([],dtype=int),imask=np.array([],dtype=bool),
                        interval=100, period=1, length=100, xfield='Адрес от начала участка (new)',
                        taufield='Наработка до отказа(new), лет'):
        def predict_and_cut(data=pd.DataFrame([]), indices=np.array([], dtype=int),imask=np.array([],dtype=bool),
                            index=0, interval=100, period=1, length=100, xfield='Адрес от начала участка (new)',
                            taufield='Наработка до отказа(new), лет'):
            x = data.loc[index, xfield]
            tau = data.loc[index, taufield]
            a, b = get_interval(interval, 1, x, length, expand=False)
            data.loc[index, 'lbound'] = a
            data.loc[index, 'rbound'] = b

            item, top = get_easy_binary_item(data.loc[indices], date=period, index=index, ints=interval)
            if item is not None:
                #self.items.append(item)
                #self.top_borders.append(top)
                predict = self.model.get_counts(item, top, stop=5)
                if predict > 0.5:
                    cut = True
                else:
                    cut = False
                data.loc[index, 'predicted'] = predict
                lmask = (data.loc[indices, xfield] >= a)
                rmask = (data.loc[indices, xfield] <= b)
                xmask = lmask & rmask
                amask = (~xmask) & (~lmask)
                bmask = (~xmask) & (~rmask)
                lindex = amask[amask == True].index
                rindex = bmask[bmask == True].index
                tmask = data.loc[indices, taufield] <= tau
                mask = xmask & tmask
                ymask = xmask & (~tmask)
                yindex = ymask[ymask == True].index
                if cut:
                    data.loc[index, 'prevent'] = yindex.shape[0]
                    data.loc[yindex, 'marked'] = index
                    data.loc[yindex, 'lost'] = False
                    llength = a
                    rlength = length - b
                    data.loc[rindex, xfield] = data.loc[rindex, xfield] - b
                    data.loc[rindex, 'L,м'] = length - b
                    data.loc[lindex, 'L,м'] = a
                    #print(imask[amask].shape)
                    #print(lindex.shape)
                    #print(imask[bmask].shape)
                    #print(rindex.shape)
                    #print(imask.shape)
                    return ((lindex, llength,imask[amask]), (rindex, rlength,imask[bmask]))
                else:
                    top = xmask & (~tmask)
                    passed = top[top == True].index
                    data.loc[passed, 'lost'] = True
                    #print('0indices', indices.shape)
                    #print('0mask', imask.shape)
                    return ((indices, length,imask),)
            else:
                return ((np.array([],dtype=int), 0,np.array([],dtype=bool)),(np.array([],dtype=int), 0,np.array([],dtype=bool)))

        #print('indices',indices.shape)
        #print('mask',imask.shape)



        if indices[imask].shape[0] > 0:
            index=indices[imask][0]
            splitted = predict_and_cut(data=data, indices=indices, index=index,imask=imask, interval=interval,
                                       period=period, length=length, xfield=xfield, taufield=taufield)
            if len(splitted) == 1:
                #print(splitted)
                #print('imask', imask)
                k=indices.get_loc(index)
                #print(k)
                imask[k]=False
                #print('ind', indices.shape)
                #print('ma', imask.shape)
                #print('imask',imask)
                #print('returned',splitted[0][2])
                self.forward_predict(data=data, indices=indices,imask=imask, interval=interval,
                                period=period, length=length, xfield=xfield, taufield=taufield)
            else:
                for s in splitted:
                    #print('i',s[0].shape)
                    #print('m',s[2].shape)
                    self.forward_predict(data=data, indices=s[0],imask=s[2], interval=interval,
                                    period=period, length=s[1], xfield=xfield, taufield=taufield)

class Generator_v2:
    def __init__(self, classifier, regressor, col=dict({})):
        self.classifier = classifier
        self.regressor = regressor
        self.col = col
        self.x = ClRe(c=np.array([]), r=np.array([]))
        # self.x=np.array([],dtype=float)
        self.gindices = np.array([], dtype=int)
        #self.count = np.array([], dtype=int)
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
        # self.columns=np.arange(self.x.r.shape[1])[1:]
        self.mask = np.ones(self.x.indices.shape[0], dtype=bool)
        # self.prev=self.x.r[:,-1].reshape(-1)
        #self.count = np.zeros(self.x.indices.shape[0], dtype=int)
        self.proba = np.zeros(self.x.indices.shape[0], dtype=float)
        self.p0 = np.zeros(self.x.indices.shape[0], dtype=float)
        self.dp = np.zeros(self.x.indices.shape[0], dtype=float)
        self.indices = self.gindices
        # self.count.np.nan
        # x=self.x
        i = 1
        while (i < stop) & (self.x.indices.shape[0] > 0):
            y, pred_mask, probab = self.get_next(x=self.x, top=self.top)
            if y is None:
                return self.proba
            iout = self.indices[~pred_mask]
            iin = self.indices[pred_mask]
            #self.count[iout] = i - 1
            self.mask = (y.r[:, -1] <= self.top[pred_mask])
            ige = self.indices[pred_mask][~self.mask]
            #self.count[ige] = i
            sub = pred_mask == True
            pred_mask[sub] = self.mask
            index = self.indices[pred_mask]
            if i == 1:
                self.p0 = probab[pred_mask]
                self.dt = probab[pred_mask]
                self.proba[index]=probab[pred_mask]
                #self.proba[iin] = probab[iin]
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
            # print(i)
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
            #x[self.col['nivl5']] = x[self.col['nivl5']] + 1
        else:
            x[self.col['ivl' + str(n - 2)]] = x[self.col['ivl' + str(n - 2)]] - 1
            x[self.col['ivl' + str(n - 1)]] = x[self.col['ivl' + str(n - 1)]] + 1

            #x[self.col['nivl' + str(n - 2)]] = x[self.col['nivl' + str(n - 2)]] - 1
            #j = n - 1
            #while j <= 5:
                #x[self.col['nivl' + str(j)]] = x[self.col['nivl' + str(j)]] + 1
                #j = j + 1
        arr=x[self.col['ivl0']:self.col['ivl5']+1]
        #print(arr)
        x[self.col['nivl0']:self.col['nivl5'] + 1]=np.cumsum(arr)

        return q
#marr=np.array([[0,1,1,3,3,4],[1,2,3,3,4,4],[2,2,3,4,4,5],[1,1,2,2,3,4],[3,4,4,5,5,5]])
#marr=np.array([[0,1,1],[1,2,3],[2,2,3],[1,1,2],[3,4,4]])
#marr=np.array([[0,0,1,1,2,2],[0,0,0,0,1,2],[0,0,0,1,1,2],[0,0,0,2,2,3],[0,0,0,0,0,3]])
#N_all,Lambda_all=get_decomposition_v2(marr,number=10,state=0,stop=marr.shape[1]-1)
#L=get_ranges_v1(N_all)
#try:
    #i=0
    #while i>=0:
        #dat=[]
        #get_columns(N_all,i,dat)
        #print(dat)
       # i=i+1
#except (IndexError): print(i)
#dat=[]
#get_columns(N_all,2,dat)
#s=pd.read_csv(path+'my_s.csv').values
#N_all,Lambda_all=get_decomposition_v2(s[:,3:],k=10,i=2,stop=3)
#marr=np.array([[1,1,2],[0,0,1],[0,1,1],[0,2,2],[0,0,0]])
#N_all,Lambda_all=get_decomposition(marr,number=10,state=0,stop=marr.shape[1]-1)
#print(N_all)


#a=np.array([1,1,0,0])
#b=np.array([[0,0,0,0],[1,1,1,0]])
#metric=metrics(w=np.ones(a.shape[0]))
#distances=get_nearest_trajectory(a,b,metric=metric)
#print(distances)
#nf=np.load(path+'Nf.npy',allow_pickle=True)[()]
#pf=np.load(path+'Pf.npy',allow_pickle=True)[()]
#i=3
#step=2
#Next=get_next_n(nf[i],pf,step=step,start=i)
#display(Next[i+step])
#display(nf[i+step])
#pre_model=Lambda_np_Predictor()
#t=pre_model.s.number
#dates=['Дата ввода','Дата аварии','Дата перевода в бездействие','Дата окончания ремонта','Дата ремонта до аварии']
#group=pd.read_csv(path+'sg.csv',parse_dates=dates,infer_datetime_format=True, dayfirst=True)
#tr=np.load(path+'tr.npy')
#model=Lambda_np_Predictor()
#lenght=100
#model.fit(group,time_range=tr,lenght=lenght,flow=True,groupby='ID простого участка')
#model.s_.metric=metrics().euclidian
#model.s.metric=metrics().euclidian
#res=make_predictions_v1(model,group,time_range=tr,lenght=100,step=[0])

#file='repaired_v2.csv'
#dates=['Дата ввода','Дата аварии','Дата перевода в бездействие','Дата окончания ремонта','Дата ремонта до аварии']
#cdata=pd.read_csv(path+file,parse_dates=dates,infer_datetime_format=True, dayfirst=True)
#mask=np.isnan(cdata['Дата перевода в бездействие'])
#print(mask[mask==True].shape)
#IDs=cdata[~mask]['new_id'].value_counts().keys()
#=cdata[cdata['new_id'].isin(IDs)]
#cdata['Адрес от начала участка (new)']=cdata['Адрес от начала участка']-cdata['a']
#cdata=cdata[['Вид покрытия внутреннего', 'Название',
       #'Месторождение', 'D', 'S', 'Материал трубы', 'Тип трубы',
       #'Завод изготовитель', 'Дата ввода', 'Состояние',
       #'Дата перевода в бездействие', 'Дата аварии',
       #'Обводненность', 'Скорость потока',
       #'Способ ликв в момент', 'ID ремонта', 'Дата окончания ремонта',
       #'Адрес от начала участка.1', 'Длина ремонтируемого участка',
       #'Тип ремонта', 'ID ремонта до аварии', 'Дата ремонта до аварии',
       #'Адрес ремонта до аварии', 'Длина ремонта до аварии',
       #'Тип ремонта до аварии', 'original_index', 'a', 'b', 'new_id',
       #'cluster', 'repair_date', 'repair_lenght', 'comment', 'repair_adress',
       #'Название месторождения', 'repair_index', 'check_out', 'old_cluster',
       #'L,м', 'Наработка до отказа(new), лет',
       #'Адрес от начала участка (new)']]
#cdata.rename(columns=dict({'new_id':'ID простого участка','Наработка до отказа(new), лет':'Наработка до отказа','Адрес от начала участка (new)':'Адрес от начала участка','L,м':'L'}), inplace=True)
#mask1=cdata['Дата аварии']>cdata['Дата ввода']
#sdata=cdata[mask1]
#fdata=sdata[sdata['Адрес от начала участка']<=sdata['L']]
#mask=np.isnan(fdata['Дата перевода в бездействие'])
#fdata=fdata[~mask]
#all_pipes=pd.read_csv(path+'active_pipes.csv')
#params=dict({1270:('сталь 10',)})
#results=dict()
#for k in params.keys():
    #mask1=(fdata['Месторождение']==k)&(fdata['Состояние']=="Действующий")
    #some_id=fdata[mask1].iloc[0]["ID простого участка"]
    #f=all_pipes[all_pipes["ID простого участка"]==int(some_id)].iloc[0]['Месторождение']
    #steel=dict()
    #for s in params[k]:
        #mask=(all_pipes['Месторождение']==f)&((all_pipes['Материал трубы']==s))
        #count=get_counter_of_empty(all_pipes,fdata,mask)
        #mask2=(fdata['Месторождение']==k)&(fdata['Материал трубы']==s)
        #print('f:',k,' m:',s,'shape ',mask2[mask2==True].shape)
        #lenghts=dict()
        #short=fdata[mask2]
        #print('месторождение ',k, ', сталь ',s, ', длина=',count)
        #train,test=ml.split_data_by_index(fdata[mask2],index='ID простого участка')
        #get_out=pd.to_datetime('2015-12-31')
        #train_x=short['Дата аварии']<=get_out
        #test_x=short['Дата аварии']>get_out
        #test_id=test_x["ID простого участка"].value_counts().keys()
        #test=short[test_x]
        #test=short
        #train=short[short['Дата аварии']<=get_out]
        #print(test.shape)
        #print(train.shape)
        #ct=test['Наработка до отказа'].max()
        #int(ct)
        #time_range_loc=ml.get_random_time_range(count=ct)
        #print('train',train.shape)
        #print('test',test.shape)
        #mask=test['ID простого участка']==50041617
        #print('test_',test[mask].shape)
        #for l in [100]:
            #model=ml.Lambda_np_Predictor()
            #lenght=l
            #free_count=np.ceil(count/lenght)
            #model.s.number=free_count
            #model.s_.number=free_count
            #model.fit(train,time_range=time_range_loc,lenght=lenght,flow=True,groupby='ID простого участка')
            #model.s_.metric=ml.metrics().euclidian
            #model.s.metric=ml.metrics().euclidian
            #mask=test["ID простого участка"]=='50000195'
            #print(mask[mask==True].shape)
            #mask=test["ID простого участка"]==test["ID простого участка"].value_counts().keys()[0]
            #res=make_predictions_v2(train,test[mask],get_out=get_out,lenght=l,step=[2], free_count=free_count,div=0.5)
            #lenghts.update({l:res})
        #steel.update({s:lenghts})
    #results.update({k:steel})
#np.save(path+'vector_predictions_split_by_R_v7.npy',results)
#results=dict()
#for k in ['все месторождения']:
    #steel=dict()
    #for s in ['все марки']:
        #mask=(all_pipes['Месторождение']==all_pipes['Месторождение'])
        #count=get_counter_of_empty(all_pipes,fdata,mask)
        #lenghts=dict()
        #short=fdata
        #get_out=pd.to_datetime('2015-12-31')
        #test=short[short['ID простого участка']==str(50043101)]
        #train=short[short['Дата аварии']<=get_out]
        #for l in [100]:
            #lenght=l
            #free_count=np.ceil(count/lenght)
            #res=make_predictions_v2(train,test,get_out=get_out,lenght=l,step=[2], free_count=free_count,div=0.25)
            #lenghts.update({l:res})
        #steel.update({s:lenghts})
    #results.update({k:steel})
#np.save(path+'vector_predictions_split_by_R_v5.npy',results)'''
#s=np.load(path+'s_matrix_v1.npy')
#train_data=get_lambda_decomposition(s,number=28372,stop=s.shape[1]-3,size=30)