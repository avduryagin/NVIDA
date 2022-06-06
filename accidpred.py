import engineering as en
import generator as gn
import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn


def predict(json,*args,**kwargs):
    class predictor:
        def __init__(self, *args, **kwargs):
            self.data = pd.DataFrame([])
            self.feat = en.features()
            self.gen = gn.Generator()
            # self.columns=["ID простого участка","Адрес от начала участка","Наработка до отказа","interval","predicted","time_series","probab"]
            self.columns = ['id_simple_sector', 'locate_simple_sector', 'worl_avar_first',
                            'interval', 'predicted', 'time_series', 'probab', 'lbound', 'rbound']
            self.results = pd.DataFrame([], columns=self.columns)

        def fit(self, data, *args, **kwargs):
            self.data = data
            en.inscribing(self.data, *args, **kwargs)
            self.feat.fit(self.data, *args, **kwargs)

        def predict(self):
            self.predicted = self.gen.predict(self.feat.ClRe, self.feat.horizon)
            self.probab = np.cumsum(np.cumprod(self.gen.p.T, axis=1), axis=1)
            self.time_series = np.multiply(self.gen.r.T, self.feat.s.reshape(-1, 1))

        def fill(self):
            for i in np.arange(self.feat.data.shape[0]):
                self.results.loc[i, 'time_series'] = self.time_series[i]
                self.results.loc[i, 'probab'] = self.probab[i]
            self.results.loc[:, 'predicted'] = self.predicted
            self.results.loc[:, 'interval'] = self.feat.data['interval'].reshape(-1).astype(np.int32)
            index = self.feat.data['index'].reshape(-1)
            self.results.loc[:, self.columns[0]] = self.data.loc[index, "ID простого участка"].values.astype(np.int32)
            self.results.loc[:, self.columns[1:3]] = self.data.loc[
                index, ["Адрес от начала участка", "Наработка до отказа"]].values
            delta = self.data.loc[index, 'a'].values
            self.results.loc[:, ['lbound', 'rbound']] = np.add(
                rfn.structured_to_unstructured(self.feat.data[['a', 'b']]).reshape(-1, 2), delta.reshape(-1, 1))
            self.json = self.results.to_json(orient='records')

    dtype = {'id_simple_sector': np.int32, 'd': np.float32, 'l': np.float32, 's': np.float32,
              'date_input': np.datetime64, 'status': object, 'status_date_bezd': np.datetime64,
              'date_avar': np.datetime64, 'worl_avar_first': np.float32, 'locate_simple_sector': np.float32,
              'sw': np.float32, 'date_end_remont': np.datetime64, 'locate_simple_sector_1': np.float32,
              'l_remont': np.float32, 'date_rem_before_avar': np.datetime64, 'locate_remont_avar': np.float32,
              'l_remont_before_avar': np.float32}
    to_rename = dict(
        {"id_simple_sector": 'ID простого участка', "d": 'D', "l": 'L', "s": 'S', "date_input": 'Дата ввода',
         "status": 'Состояние',
         "status_date_bezd": 'Дата перевода в бездействие', "date_avar": 'Дата аварии',
         "worl_avar_first": 'Наработка до отказа',
         "locate_simple_sector": 'Адрес от начала участка', "sw": 'Обводненность',
         "date_end_remont": 'Дата окончания ремонта',
         "locate_simple_sector_1": 'Адрес от начала участка_1', "l_remont": 'Длина ремонтируемого участка',
         "date_rem_before_avar": 'Дата ремонта до аварии', "locate_remont_avar": 'Адрес ремонта до аварии',
         "l_remont_before_avar": 'Длина ремонта до аварии'})


    data=pd.read_json(json,orient='split',dtype=dtype)
    data.rename(columns=to_rename,inplace=True)
    model=predictor()
    model.fit(data,mode='bw')
    model.predict()
    model.fill()
    return model.json











