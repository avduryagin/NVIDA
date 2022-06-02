import requests
import lxml.html
import os
import time


class hh_request():
    def __init__(self):
        self.headers={'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.80 Safari/537.36',
         'authority': 'spb.hh.ru','method': 'POST','path': '/account/login?backurl=%2F','accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
         'accept-encoding': 'gzip, deflate, br','content-type': 'application/x-www-form-urlencoded'}
        self.values={'username': '79633440605','password': 'nabukh0d0n0s0r','backUrl': 'https://spb.hh.ru/','remember': 'yes','action': 'Войти','_xsrf': 'xsrf'}
        self.url='https://spb.hh.ru/account/login?backurl=%2F'
        self.token=None
        self.session=None
        self.cv_id={'math':'b6f94738ff042595420039ed1f697336795442','project_manager':'25ab77c5ff033a6eba0039ed1f6752534d5541'}
        self.step=14401
    def auth(self):
        self.session=requests.Session()
        g=self.session.get(self.url,headers=self.headers)
        dic=self.parser(g.headers)
        #xsrf=g.headers['Set-Cookie']
        #xsrf_spl=xsrf.split(' ')
        #xsrf_token=xsrf_spl[17][6:-1]
        try:
            xsrf_token=dic['_xsrf']
        except KeyError:
            xsrf_token=''
        self.values['_xsrf']=xsrf_token
        self.token=xsrf_token
        req=self.session.post(self.url,data=self.values,headers=self.headers)
        return req
    def sv_updater(self,cv_id=None):
        values={'resume':cv_id,'undirectable':''}
        values['undirectable']=True
        headers={'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.80 Safari/537.36',
         'authority': 'spb.hh.ru','method': 'POST','path': '/applicant/resumes/touch',
         'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
         'accept-encoding': 'gzip, deflate, br','content-type': 'application/x-www-form-urlencoded','x-xsrftoken': self.token}
        url='https://spb.hh.ru/applicant/resumes/touch'
        update=self.session.request('POST',url,data=values,headers=headers)
        return update.status_code
    def update(self):
        self.auth()
        do=True
        while do:
            for c_val in self.cv_id.values():
                code = self.sv_updater(c_val)
                print(time.time(), ' code: ', code, ' cv: ', c_val, 'token=', self.token, end=" ")
                print(' ')
            time.sleep(self.step)


    def parser(self,header):
        def split(s, markers, dic):
            if len(markers) == 0:
                if len(s) > 1:
                    dic.update({s[0].strip(): s[1].strip()})
                else:
                    return s


            else:
                marker = markers[0]
                if type(s) is list:
                    for s_ in s:
                        splited = s_.split(marker)
                        sp = split(splited, markers[1:], dic)
                else:
                    splited = s.split(marker)
                    sp = split(splited, markers[1:], dic)
                return sp

        marker = [';', ',', '=']
        parsed = dict({})
        dic = dict(header)
        for key in dic.keys():
            s = split(dic[key], marker, parsed)
            if s is not None:
                parsed.update({key: s})

        return parsed
    
#hh=hh_request()
#hh.auth()