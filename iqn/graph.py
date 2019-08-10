import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
overview = plt.figure()
rewardsGraph = overview.add_subplot(1, 1, 1)

resultsA = np.loadtxt("./test-results/3x-64.txt",delimiter=",")
resultsB = np.loadtxt("./test-results/lr2e-4.txt",delimiter=",")
resultsC = np.loadtxt("./test-results/3x-128.txt",delimiter=",")
resultsD = np.loadtxt("./test-results/lim-100.txt",delimiter=",")
resultsE = np.loadtxt("./test-results/128-20-atoms.txt",delimiter=",")
resultsF = np.loadtxt("./test-results/lr-3-20-atoms.txt",delimiter=",")
resultsG = np.loadtxt("./test-results/lr-3-20-atoms-2.txt",delimiter=",")
resultsH = np.loadtxt("./test-results/3x-128-2.txt",delimiter=",")
resultsI = np.loadtxt("./test-results/lr-1-8-vmax200-12-atom.txt",delimiter=",")
resultsJ = np.loadtxt("./test-results/post-optimization-updates.txt",delimiter=",")
resultsK = np.loadtxt("./test-results/10x-run.txt",delimiter=",")
for i in resultsA:
    rewardsGraph.plot(i, color='red', alpha=0.25)

for i in resultsB:
    rewardsGraph.plot(i, color='red', alpha=0.25)

for i in resultsC:
    rewardsGraph.plot(i, color='red', alpha=0.25)

for i in resultsD:
    rewardsGraph.plot(i, color='red', alpha=0.25)

for i in resultsE:
    rewardsGraph.plot(i, color='red', alpha=0.25)

for i in resultsF:
    rewardsGraph.plot(i, color='red', alpha=0.25)

for i in resultsG:
    rewardsGraph.plot(i, color='red', alpha=0.25)

for i in resultsH:
    rewardsGraph.plot(i, color='red', alpha=0.25)

for i in resultsI:
    rewardsGraph.plot(i, color='red', alpha=0.25)

for i in resultsJ:
    rewardsGraph.plot(i, color='blue')

# for i in [resultsK]:
#     rewardsGraph.plot(i, color='blue')

overview.canvas.draw()
plt.pause(20)
