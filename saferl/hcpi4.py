import numpy as np
import random
from saferl.data_gen import read_data, write_data
from saferl.evaluate import Evaluate
from systemrl.environments.cartpole import Cartpole
from systemrl.policies.linear_softmax import LinearSoftmax
import matplotlib.pyplot as plt
from scipy import stats
import cma

class HCPI():
    def __init__(self, numFeatures, numActions, k, theta, D, outFile):
        self.numFeatures = numFeatures
        self.numActions = numActions
        self.thetaB = theta
        #every solution sees different parts of data
        random.shuffle(D)
        self.D = D
        self.linearSoftmax = LinearSoftmax(numFeatures, numActions, k)
        self.currPerformance = self.performance_from_history()
        print("Default Performance: ", self.currPerformance, file=outFile)
        partition = 70 #percentage
        partition = int(partition / 100 * len(D))
        self.Dc = D[:partition]
        self.Ds = D[partition:]
        self.init_piB()
        self.outFile = outFile
        self.thetas = []

    def init_piB(self):
        self.piB_Dc = []
        self.piB_Ds = []
        for H in self.Dc:
            self.piB_Dc.append(self.piE(self.thetaB, H))
        for H in self.Ds:
            self.piB_Ds.append(self.piE(self.thetaB, H))

    def performance_from_history(self):
        #gamma is one
        returns = []
        for h in self.D:
            returns.append(np.sum(h[2]))
        return np.mean(returns)

    def piE(self, theta, H):
        self.linearSoftmax.parameters = np.array(theta)
        pr_s = self.linearSoftmax.getActionGivenStatesProbabilities(H[0])
        piE = pr_s[np.arange(pr_s.shape[0]),H[1]]
        return piE

    def pdis_H(self, thetaE, piB, H):
        piE = self.piE(thetaE, H)
        return np.sum(np.cumprod(piE/piB) * H[2])

    def pdis_D(self, thetaE, mode=None):
        PDIS = []
        if mode is None:
            D = self.Dc
            piB = self.piB_Dc
        else:
            D = self.Ds
            piB = self.piB_Ds
        for idx, H in enumerate(D):
            PDIS.append(self.pdis_H(thetaE, piB[idx], H))
        return PDIS

    def safety_test(self, thetaE, delta):
        result = False
        n = len(self.Ds)
        PDIS = self.pdis_D(thetaE, "safety test")
        mean = np.mean(PDIS)
        std = np.std(PDIS, ddof=1)
        t = stats.t.ppf(1-delta, n-1) 
        estimate = mean - t*std/np.sqrt(n)
        print("Safety Test:", file=self.outFile)
        print("mean: ",mean, file=self.outFile)
        print("std: ",std, file=self.outFile)
        print("Estimated Lower Bound: ", estimate, file=self.outFile)
        if estimate > self.currPerformance:
            print("Pass", file=self.outFile)
            result = True
        else:
            print("Fail", file=self.outFile)
        return estimate, result

    def optimize(self, thetaE, delta):
        n = len(self.Ds)
        PDIS = self.pdis_D(thetaE)
        mean = np.mean(PDIS)
        #reg term
        reg = 0.01 * np.square(thetaE).sum()
        return -mean + reg
        std = np.std(PDIS, ddof=1)
        t = stats.t.ppf(1-delta, n-1) 
        estimate = mean - 2*std/np.sqrt(n)*t
        if estimate > self.currPerformance:
            #should be large
            return -mean + 10000
        return -mean

    def candidate_selection(self, stdInit, delta, maxIter=5):
        es = cma.CMAEvolutionStrategy(self.thetaB, stdInit,{'tolfun':1e-5})
        currIter = 0
        while (not es.stop()) and currIter < maxIter:
            currIter += 1
            solutions = es.ask()
            evaluations = []
            for sol in solutions:
                evaluations.append(self.optimize(sol, delta))
            es.tell(solutions, evaluations)
            print("theta: ",es.mean, file=self.outFile)
            self.safety_test(es.mean, delta)
            self.thetas.append(es.mean)
            es.disp()
        es.result_pretty()
        return np.array(self.thetas)

def get_multiple_candidates(maxIter, delta, stdInit, numFeatures, numActions, k, thetaB, D, count):
    filename = "data/multiple" + str(count)
    outFile = open(filename+".txt", "w+")
    hcpi = HCPI(numFeatures, numActions, k, thetaB, D, outFile)
    thetaE = hcpi.candidate_selection(stdInit, delta, maxIter)
    thetaFilename = filename+".csv"
    np.savetxt(thetaFilename, thetaE, delimiter=',') 
    outFile.close()


def main(maxIter, delta, stdInit):
    from multiprocessing import Process as p
    filename = "data/data"
    numFeatures, numActions, k, thetaB, D = read_data(filename)

    numCandidates = 20
    process = []
    for count in range(numCandidates):
        process.append(p(target=get_multiple_candidates, 
            args=(maxIter, delta, stdInit, numFeatures, numActions, k, thetaB, D, count+1)))
        process[-1].start()

    for proc in process:
        proc.join()
        

if __name__ == "__main__":
    maxIter = 100
    delta = 0.01
    stdInit = 1
    main(maxIter, delta, stdInit)
