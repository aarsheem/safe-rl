from systemrl.environments.gridworld import Gridworld
from systemrl.environments.cartpole import Cartpole
from systemrl.environments.mountaincar import Mountaincar
from systemrl.policies.linear_softmax import LinearSoftmax
from saferl.evaluate import Evaluate
import numpy as np
import matplotlib.pyplot as plt


def write_data(env, episodes, numFeatures, k, theta, filename, feature):
    numActions = env.numActions()
    dataFile = open(filename+".csv", "w+")

    dataFile.write(str(numFeatures)+"\n")
    dataFile.write(str(numActions)+"\n")
    dataFile.write(str(k)+"\n")
    for idx, i in enumerate(theta.flatten()):
        if idx:
            dataFile.write(",")
        dataFile.write(str(i))
    dataFile.write("\n"+str(episodes)+"\n")

    linearSoftmax = LinearSoftmax(numFeatures, numActions, k)
    linearSoftmax.parameters = theta
    #keep in mind this value
    maxSteps = 20
    evaluate = Evaluate(env, linearSoftmax, maxSteps, feature)
    returns = 0
    for n in range(episodes):
        states, actions, rewards = evaluate.episode()
        for i in range(len(states)):
            normState = states[i]
            action = actions[i]
            reward = rewards[i]
            if i:
                dataFile.write(",")
            for s in normState:
                dataFile.write(str(s)+",")
            dataFile.write(str(action)+","+str(reward))
        dataFile.write("\n")
        #gamma is one
        returns += np.sum(rewards)
    print("mean returns: ",returns/episodes)
    dataFile.close()

def read_data(filename, disp=False):
    dataFile = open(filename+".csv")
    numFeatures = int(dataFile.readline())
    numActions = int(dataFile.readline())
    k = int(dataFile.readline())
    #dealt extra comma
    theta = [float(i) for i in dataFile.readline().split(",")]
    episodes = int(dataFile.readline())
    assert(len(theta) == numActions * (k+1)**numFeatures)
    H = []
    for episode in range(episodes):
        H.append([])
        #state, action, reward
        H[-1].append([])
        H[-1].append([])
        H[-1].append([])
        #extra comma
        h = dataFile.readline().split(",")
        i = 0
        while i < len(h):
            state = []
            for _ in range(numFeatures):
                state.append(float(h[i]))
                i += 1
            H[episode][0].append(state)
            H[episode][1].append(int(h[i]))
            i += 1
            H[episode][2].append(float(h[i]))
            i += 1
    if disp:
        for i in range(episodes):
            print("states: ", H[i][0])
            print("actions: ", H[i][1])
            print("rewards: ", H[i][2])
    return numFeatures, numActions, k, theta, H

def read_theta(filename):
    dataFile = open(filename+".txt")
    theta = []
    line = dataFile.readline()
    while line:
        theta.append([float(i) for i in line.split()])
        line = dataFile.readline()
    return np.array(theta)

def main(k=1, feature=-1):
    #env = Gridworld()
    env = Cartpole()
    #env = Mountaincar()
    episodes = 50000
    #-1 corresponds to clean data
    #otherwise represents the noisy feature index
    if feature == -1:
        fStr = "all"
        features = env.numFeatures()
    else:
        fStr = str(feature)
        features = 1
    actions = env.numActions()
    filename = "data/cartpole_theta_"+str(k)+"_"+fStr
    theta = read_theta(filename)#np.zeros(((k+1)**features, actions))#read_theta(filename)
    filename = "data/cartpole_good_"+str(k)+"_"+fStr+"_"+str(episodes)
    write_data(env, episodes, features, k, theta, filename, feature)
    read_data(filename)

if __name__ == "__main__":
    if 1:
        K = [1, 2]
        feature = [-1, 0, 1, 2, 3]
        for k in K:
            for f in feature:
                main(k, f)
    else:
        numFeatures, numActions, k, theta, D = read_data("data/data")
        print("maxlen: ",np.max([len(h[0]) for h in D]))
        print("for one episode:")
        print("states: ",D[0][0])
        print("actions: ",D[0][1])
        print("rewards: ",D[0][2])
        #print("maxstate: ",np.max([np.max(h[0]) for h in D]))
        #print("minstate: ",np.min([np.min(h[0]) for h in D]))
        #print("maxreward: ",np.max([np.max(h[2]) for h in D]))
        #print("minreward: ",np.min([np.min(h[2]) for h in D]))

        from collections import defaultdict
        from mpl_toolkits.mplot3d import Axes3D
        returns = []
        states = [[],[]]
        rewards = [[],[]]
        freq = [defaultdict(lambda: 0), defaultdict(lambda: 0)]
        actions = [0, 0]
        count = defaultdict(lambda: 0)
        resolution = 50
        for h in D:
            g = 0
            for i in range(len(h[0])):
                s = h[0][i][0]
                a = h[1][i]
                r = h[2][i]
                s = np.rint(s*resolution)
                r = np.rint((r/20+0.5)*resolution)
                freq[a][(s,r)] += 1
                count[s] += 1

        def f_action_0(X, Y):
            Z = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    s = X[i][j]
                    r = Y[i][j]
                    s = np.rint(s*resolution)
                    r = np.rint((r/20+0.5)*resolution)
                    Z[i][j] = freq[0][(s, r)]
                    if Z[i][j]:
                        Z[i][j] /= count[s]
            return Z

        def f_action_1(X, Y):
            Z = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    s = X[i][j]
                    r = Y[i][j]
                    s = np.rint(s*resolution)
                    r = np.rint((r/20+0.5)*resolution)
                    Z[i][j] = freq[1][(s, r)]
                    if Z[i][j]:
                        Z[i][j] /= count[s]
            return Z

        X = np.linspace(-0.1, 1.1, 200)
        Y = np.linspace(-11, 11, 200)
        X, Y = np.meshgrid(X, Y)
        Z0 = f_action_0(X, Y)
        Z1 = f_action_1(X, Y)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z0, cmap='viridis')
        ax.set_xlabel('state')
        ax.set_ylabel('reward')
        ax.set_zlabel('p(reward|state,action=0)')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z1, cmap='viridis')
        ax.set_xlabel('state')
        ax.set_ylabel('reward')
        ax.set_zlabel('p(reward|state,action=1)')
        plt.show()



