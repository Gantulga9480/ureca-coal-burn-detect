import csv
from datetime import datetime
import numpy as np

def load_csv(file):
    date = []
    temp = []
    humd = []
    annt = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            d, t, h, a = row
            d = datetime.fromisoformat(d)
            date.append(d)
            temp.append(float(t))
            humd.append(float(h))
            annt.append(round(float(a)))
    return (date, temp, humd, annt)

def preprocess(file):
    date, temperature, humidity, annotation = load_csv(file)

    time_for_inference = []
    for d in date:
        d = d.hour + d.minute / 60
        time_for_inference.append(d)

    dn = np.array(time_for_inference) / 24
    tn = np.array(temperature)
    tn /= 50
    hn = np.array(humidity)
    hn /= 100

    dn_list = dn.tolist()
    tn_list = tn.tolist()
    hn_list = hn.tolist()

    return dn_list, tn_list, hn_list, annotation, date