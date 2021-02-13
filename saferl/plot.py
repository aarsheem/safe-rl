import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
    f = open(filename)
    l = f.readline()
    theta = []
    while l:
        s = l.split(",")
        theta.append([])
        for x in s:
            theta[-1].append(float(x))
        l = f.readline()
    return np.array(theta)

def read_performance(filename):
    f = open(filename)
    l = f.readline()
    per = []
    while l:
        s = l.split(":")
        if s[0] == "Estimated Lower Bound":
            per.append(float(s[1]))
        l = f.readline()
    return np.array(per)

def write_results(filename, results):
    for idx, r in enumerate(results):
        f = open(filename+str(idx+1)+".csv", "w+")
        for i, x in enumerate(r):
            if i:
                f.write(",")
            f.write(str(x))
        f.close()
    

filename = "data/multiple"
theta = []
per = []
X = []
P = []
for i in range(0, 20):
    f = filename + str(i+1)
    theta.append(read_file(f+".csv"))
    per.append(read_performance(f+".txt"))
    T = []
    for j in range(len(theta[i])):
        t = theta[i][j]
        p = per[i][j]
        T.append([p, t])
    T.sort(reverse=True, key=lambda x: x[0])
    for j in range(5):
        X.append(T[j][1])
        P.append(T[j][0])

X = np.array(X)
write_results("results/", X)
for i in range(X.shape[1]):
    y = X[:,i]
    x = np.arange(X.shape[0])
    plt.plot(x, y, label="dim="+str(i))
plt.plot(x, P, label="performance")
plt.legend()
plt.show()
