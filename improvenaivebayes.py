import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd
import csv
import prince
from sklearn.naive_bayes import ComplementNB,CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OrdinalEncoder



def cramers_v(data):
    chi2, _, _, _ = chi2_contingency(data)    
    n = np.sum(data.to_numpy())  # total number of observations?
    rows, cols = data.shape
    v = (chi2/ (n*(min(rows, cols) - 1)))**0.5
    return v

def get_data():
    with open("Discretized_HeartAttack.csv") as f:
        reader = csv.reader(f)
        features = next(reader)

        split = dict()
        for i in range(len(features)):
            split[i] = []
        for row in reader:
            for i, r in enumerate(row):
                split[i] += [r]
    j = [i for i in range(len(features)-1)]
    data = pd.DataFrame(split)

    vs = []
    v_max = 0
    tables = [[] for i in range(len(features)-1)]
    maxes =  [0 for i in range(len(features)-1)]
    for i in range(0,len(features)-1):
        #contingency_tables += [pd.crosstab(data[i], data[len(features)-1])]
        #why is there like an extra loop here???? 
        #looping over it 22*22 times instead of just 22 times???
        for index in range(0,len(features)-1):
            if (index != i):
                p = pd.crosstab(data[i],data[index])
                v = cramers_v(p)
                tables[i] += [v]
                if v > maxes[i]:
                    maxes[i] = v
            else:
                tables[i] += [-1]
    return features, tables, maxes, data

def complement_naive_bayes(data):
    X,y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cnb = ComplementNB() #better with imbalanced datasets we have an 8:2 ratio
    cnb.fit(X_train, y_train)

    y_pred = cnb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cnb.classes_, yticklabels=cnb.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def full_mca(data):
    X,y = data
    mca = prince.MCA(n_components=2,random_state=42)
    transformed = mca.fit_transform(pd.DataFrame(X))
    X_train, X_test, y_train, y_test = train_test_split(transformed, y, test_size=0.2, random_state=42)
    cnb = ComplementNB() #better with imbalanced datasets we have an 8:2 ratio
    cnb.fit(X_train, y_train)

    y_pred = cnb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cnb.classes_, yticklabels=cnb.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def hierarchical_mca(data,features, tables,maxes):

    # print(tables)
    mca_set =[[i] for i in range(len(features)-1)]
    naive_set = []
    for i in range(len(tables)):
        # naive_mi = True
        for index, val in enumerate(tables[i]):
            normalize = val/maxes[i]
            # if normalize == 1:
            #     normalize = (maxes[i] - 0.009)/maxes[i]
            if normalize > 0:
                print(f"{features[index]} attribute's correlation with the {features[i]} attribute: {normalize}") #1 means dependent, 0 means independent, 
                if normalize>0.5:
                    mca_set[i] += [index]#features[index]]
        # if (naive_mi):
        #were gonna make an assumption. none of the attributes are independent to one another. they all have some kind of dependence. use these algorithms to quanitfy that dependence then normalize to find which ones are truely correlated to each other the most.
        #     mca_set += [features[i]]
        #we need mroe than one feature for naive bayes. if we do mca on the whole thing well end up with one single feature
        print()


    # mca_set = []
    # for index, v in enumerate(vs):
    #     noramlize = v/v_max
    #     print(f"{features[index]} attribute's correlation with the class: {noramlize}") #1 means dependent, 0 means independent, 
    #     if noramlize < 0.15: #below .15 is independent enough
    #         naive_set += [index]#straight to naive bayes
    #     else:
    #         mca_set += [index]# do mca first

    # naive_data = data[naive_set].copy()

    #get rid of copies:
    no_copies = []
    for subset in mca_set:
        if sorted(subset) not in no_copies:
            no_copies += [subset]

    transformed = []
    for index, subset in enumerate(no_copies):
        # no_copies[index] = data[subset].copy()
        mca = prince.MCA(n_components=2,random_state=42)
        t = mca.fit_transform(data[subset].copy())
        # scaler = StandardScaler()
        # transformed_scaled = scaler.fit_transform(t)
        transformed += [t]

    X_mca = np.hstack(transformed)
    mca = prince.MCA(n_components=2,random_state=42)
    global_mca = mca.fit_transform(pd.DataFrame(X_mca))

    #running mca this many times introduces the high possibility of generating a ot of noise
    y = data[len(features)-1]

    #no negative values for ComplementNB
    scaler = MinMaxScaler()
    X = scaler.fit_transform(global_mca)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cnb = ComplementNB() #better with imbalanced datasets we have an 8:2 ratio
    cnb.fit(X_train, y_train)

    y_pred = cnb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cnb.classes_, yticklabels=cnb.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # mca.fit(mca_data)
    # print()
    # print(mca_data)
    # input()
    # print(naive_data.to_numpy())
    # input()
    # print(data[len(features)-1].copy().to_numpy())
    # input()
    # cnb.fit(naive_data.to_numpy(), data[len(features)-1].copy().to_numpy())
