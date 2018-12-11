import pickle
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy
import json
import re
import cherrypy

# đối tượng trả về
class ResponsePack(object):
    def __init__(self):
        self.operant=''
        self.param=''

# nạp mô hình command-classifier
with open('command-classifier', 'rb') as trained_model:
    model = pickle.load(trained_model)

# nạp dữ liệu để xây dựng bộ chuyển TF-IDF
df = pandas.read_csv("fruitg.csv")
Xpre = df['SENTENCE']

# xây dựng bộ chuyển đổi dữ liệu thành TF-IDF
tfidf_converter = TfidfVectorizer(max_features=150)
tfidf_converter.fit_transform(Xpre)

# hàm xử lý request POST
@cherrypy.expose
class CommandClassify(object):
    def POST(self, data=''):
        res = ResponsePack()
        cherrypy.response.headers['Access-Control-Allow-Origin'] = '*'
        
        # tìm cụm các số
        res.param = re.findall(r"\d{4,16}", data)

        # chuyển 'data' thành dạng biểu diễn TF-IDF
        datamat = tfidf_converter.transform([data])

        # dự đoán xác suất chính xác của mô hình với biểu diễn 'data'
        proba = model.predict_proba(datamat)

        # nếu xác suất chính xác hơn 50%, coi đó là phân loại lệnh
        if numpy.max(proba) >= 0.5:
            operant = model.predict(datamat)
            res.operant = operant[0]

        return json.dumps(res.__dict__)

if __name__ == '__main__':
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')],
        }
    }
    cherrypy.quickstart(CommandClassify(), '/', conf)