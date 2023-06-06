import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D
import skfuzzy
import sys
import joblib

class Wangmendel():

    def __init__(self, train, test, no_mf):
        self.train = train
        self.test = test
        self.no_mf = no_mf

    def generate_mfs(self):
        self.qpa_range = np.linspace(min(self.train["qpa"]), max(self.train["qpa"]), self.no_mf)
        self.pulse_range = np.linspace(min(self.train["pulse"]), max(self.train["pulse"]), self.no_mf)
        self.resp_range = np.linspace(min(self.train["resp"]), max(self.train["resp"]), self.no_mf)
        self.gravity_range = np.linspace(min(self.train["gravity"]), max(self.train["gravity"]), self.no_mf)
        qpa_dif = self.qpa_range[1] - self.qpa_range[0]
        pulse_dif = self.pulse_range[1] - self.pulse_range[0]
        resp_dif = self.resp_range[1] - self.resp_range[0]
        gravity_dif = self.gravity_range[1] - self.gravity_range[0]
        self.qpa_range = np.linspace(min(self.train["qpa"])-qpa_dif, max(self.train["qpa"])+qpa_dif, self.no_mf + 2)
        self.pulse_range = np.linspace(min(self.train["pulse"])-pulse_dif, max(self.train["pulse"])+pulse_dif, self.no_mf + 2)
        self.resp_range = np.linspace(min(self.train["resp"])-resp_dif, max(self.train["resp"])+resp_dif, self.no_mf + 2)
        self.gravity_range = np.linspace(min(self.train["gravity"])-gravity_dif, max(self.train["gravity"])+gravity_dif, self.no_mf + 2)
        self.qpa_mf = []
        self.pulse_mf = []
        self.resp_mf = []
        self.gravity_mf = []
        self.center = []
        for i in range(len(self.gravity_range) - 3):
            self.qpa_mf.append(skfuzzy.trapmf(self.qpa_range, [self.qpa_range[i],self.qpa_range[i+1],self.qpa_range[i+2],self.qpa_range[i+3]]))
            self.pulse_mf.append(skfuzzy.trapmf(self.pulse_range, [self.pulse_range[i],self.pulse_range[i+1],self.pulse_range[i+2],self.pulse_range[i+3]]))
            self.resp_mf.append(skfuzzy.trapmf(self.resp_range, [self.resp_range[i],self.resp_range[i+1],self.resp_range[i+2],self.resp_range[i+3]]))
            self.gravity_mf.append(skfuzzy.trapmf(self.gravity_range, [self.gravity_range[i],self.gravity_range[i+1],self.gravity_range[i+2],self.gravity_range[i+3]]))
            self.center.append( (self.gravity_range[i+1]+self.gravity_range[i+2])/2)

        # for i in self.qpa_mf:
        #     plt.plot(self.qpa_range,i,color='coral')
        # plt.xlabel("qpa")
        # plt.ylabel("mfs of qpa")
        # plt.show()

        # for i in self.pulse_mf:
        #     plt.plot(self.pulse_range,i,color='gold')
        # plt.xlabel("pulse")
        # plt.ylabel("mfs of pulse")
        # plt.show()

        # for i in self.resp_mf:
        #     plt.plot(self.resp_range,i,color='navy')
        # plt.xlabel("resp")
        # plt.ylabel("mfs of resp")
        # plt.show()

        # for i in self.gravity_mf:
        #     plt.plot(self.gravity_range,i,color='green')
        # plt.xlabel("gravity")
        # plt.ylabel("mfs of gravity")
        # plt.show()
    
    def generate_rules(self):
        rules = []
        for i in range(len(self.train["qpa"])):
            list_qpa, list_pulse, list_resp, list_gravity = [], [], [], []
            for j in range(len(self.gravity_mf)):
                list_qpa.append(skfuzzy.interp_membership(self.qpa_range, self.qpa_mf[j], self.train["qpa"][i]))
                list_pulse.append(skfuzzy.interp_membership(self.pulse_range, self.pulse_mf[j], self.train["pulse"][i]))
                list_resp.append(skfuzzy.interp_membership(self.resp_range, self.resp_mf[j], self.train["resp"][i]))
                list_gravity.append(skfuzzy.interp_membership(self.gravity_range, self.gravity_mf[j], self.train["gravity"][i]))
            
            qpa_max = np.argmax(list_qpa)
            pulse_max = np.argmax(list_pulse)
            resp_max = np.argmax(list_resp)
            gravity_max = np.argmax(list_gravity)

            if list_qpa[qpa_max] == 0:
                print("qpa: ", self.train["qpa"][i])
            
            if list_pulse[pulse_max] == 0:
                print("pulse: ", self.train["pulse"][i])

            if list_resp[resp_max] == 0:
                print("resp: ", self.train["resp"][i])

            if list_gravity[gravity_max] == 0:
                print("gravity: ", self.train["gravity"][i])

            degree = list_qpa[qpa_max] * list_pulse[pulse_max] * list_resp[resp_max] * list_gravity[gravity_max]
            rules.append([qpa_max, pulse_max, resp_max, gravity_max, degree])
        
        # print("Primitive Rules:")
        # print("We have ", len(rules), " rules")

        rules.sort(key= lambda rules:rules[0], reverse=True)
        rules.sort(key= lambda rules:rules[1], reverse=True)
        rules.sort(key= lambda rules:rules[2], reverse=True)
        rules.sort(key= lambda rules:rules[3], reverse=True)
        rules.sort(key= lambda rules:rules[4], reverse=True)

        no_dup_rules = []
        alternative = []
        alternative.append(rules[0][0:-1])
        no_dup_rules.append(rules[0])
        for rule in rules[1:]:
            if [rule[0],rule[1],rule[2],rule[3]] not in alternative:
                alternative.append(rule[0:-1])
                no_dup_rules.append(rule)
        
        self.final_rules = []
        self.final_rules.append(no_dup_rules[0])
        alternative = []
        alternative.append(no_dup_rules[0][0:-2])
        for rule in no_dup_rules[1:]:
            if [rule[0],rule[1],rule[2]] not in alternative:
                self.final_rules.append(rule)
                alternative.append(rule[0:-2])

        # print('final rules: ')
        # print('we have',len(self.final_rules),'non conflict and non duplicate rule' )
        # K=pd.DataFrame(self.final_rules)
        # print(K)
    
    def get_results(self, train_or_test):
        if(train_or_test == "train"):
            data = self.train
        elif(train_or_test == "test"):
            data = self.test

        rules = np.array(self.final_rules, dtype=int)

        Y_predict = []

        for i in range(len(data["gravity"])):

            list_qpa, list_pulse, list_resp = [], [], []

            for j in range(len(self.gravity_mf)):
                list_qpa.append(skfuzzy.interp_membership(self.qpa_range, self.qpa_mf[j], data["qpa"][i]))
                list_pulse.append(skfuzzy.interp_membership(self.pulse_range, self.pulse_mf[j], data["pulse"][i]))
                list_resp.append(skfuzzy.interp_membership(self.resp_range, self.resp_mf[j], data["resp"][i]))
            
            var1, var2 = 0, 0
            for k in range(len(rules)):
                var1 += list_qpa[rules[k][0]] * list_pulse[rules[k][1]] * list_resp[rules[k][2]] * self.center[rules[k][3]]
                var2 += list_qpa[rules[k][0]] * list_pulse[rules[k][1]] * list_resp[rules[k][2]]
            if var2 == 0:
                Y_predict.append(0)
            else:
                Y_predict.append(var1/var2)
        
        mse = sum(((data["gravity"] - Y_predict) ** 2) / (2*len(data["gravity"])) )

        csv_file_path = f'wang_mendel_{train_or_test}.csv' 

        fieldnames = ["real numeric gravity", "real class gravity", "predicted numeric gravity", "predicted class gravity"]

        acc = 0

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(data["gravity"])):
                predicted_class = 1 if Y_predict[i] < 25 else ( 2 if Y_predict[i] < 50 else (3 if Y_predict[i] < 75 else 4))
                if predicted_class == data["class"][i]:
                    acc += 1               
                info = {"real numeric gravity" : data["gravity"][i], 
                        "real class gravity" : data["class"][i],
                        "predicted numeric gravity" : Y_predict[i],
                        "predicted class gravity" : predicted_class}
                writer.writerow(info)
        
        print(f"Accuracy for {train_or_test} is {acc/len(data['gravity']) * 100} %")

        # plt.figure()
        # plt.plot(range(100),data["gravity"][0:100],color= 'green')
        # plt.plot(range(100) , Y_predict[0:100],color='black')
        # plt.show()    
        # print(f'the {train_or_test} error is :',mse)


#Load data

train_or_test = sys.argv[1]

if(train_or_test == "train"):

    no_mf = int(sys.argv[2])

    data = np.loadtxt('treino_sinais_vitais_com_label.txt', 
                    delimiter=',', 
                    usecols=[3, 4, 5, 6, 7])

    np.random.shuffle(data)
    train = data[0:560]
    test = data[560:800]

    qpa_train = train[:,0]
    pulse_train = train[:,1]
    resp_train = train[:,2]
    gravity_train = train[:,3]
    class_train = train[:,4]

    train = {"qpa" : qpa_train, "pulse" : pulse_train, "resp": resp_train, "gravity": gravity_train, "class": class_train}

    qpa_test = test[:,0]
    pulse_test = test[:,1]
    resp_test = test[:,2]
    gravity_test = test[:,3]
    class_test = test[:,4]

    test = {"qpa" : qpa_test, "pulse" : pulse_test, "resp": resp_test, "gravity": gravity_test, "class": class_test}

    wm = Wangmendel(train, test, no_mf)

    wm.generate_mfs()
    wm.generate_rules()
    wm.get_results("train")
    wm.get_results("test")

    joblib.dump(wm, 'wang_mendel_model.pkl')

elif(train_or_test == "test"):

    file_name = sys.argv[2]

    data = np.loadtxt(file_name, 
                    delimiter=',', 
                    usecols=[3, 4, 5, 6, 7])
    
    qpa_test = data[:,0]
    pulse_test = data[:,1]
    resp_test = data[:,2]
    gravity_test = data[:,3]
    class_test = data[:,4]

    test = {"qpa" : qpa_test, "pulse" : pulse_test, "resp": resp_test, "gravity": gravity_test, "class": class_test}

    wm = joblib.load('wang_mendel_model.pkl')

    wm.test = test

    wm.get_results("test")
