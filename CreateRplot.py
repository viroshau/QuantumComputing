import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math as math
#sns.set()

def CostLandscapeHeatMap(meshGamma,meshBeta,Z,z_min,z_max,title):
    fig = plt.figure()
    c = plt.pcolormesh(meshGamma,meshBeta,Z,vmin = (z_min),vmax = (z_max),cmap = 'coolwarm')
    plt.ylabel(r'$\beta_p$')
    plt.xlabel(r'$\gamma_p$')
    plt.xlim(0,np.pi)
    plt.ylim(0,np.pi/2)
    plt.title(title)
    plt.colorbar(c,format = '%.1f')
    plt.show()  

def CostLandscapeplotter(meshGamma,meshBeta,Z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(meshGamma,meshBeta,Z,cmap='coolwarm')
    ax.set_ylabel(r'$\beta_{10}$')
    ax.set_xlabel(r'$\gamma_{10}$')
    ax.set_zlabel(r'C($\gamma_{10},\beta_{10}$)')
    plt.show()  

numP = 10
 
fig,ax = plt.subplots(nrows = 2,ncols = 2, sharex=True)

""" 
for i in ([6,8,10,12,14]):
    optimalcosts = np.load(f'Parameterfix{i}NodeGraph.npy')
    bestresult = np.floor(optimalcosts[-1])
    ax[0,0].plot(range(1,numP+1),optimalcosts/bestresult,label = i,c = colors[i])

    optimalcosts = np.load(f'RandomInitializations{i}NodeGraph.npy')
    optimalcosts = np.average(optimalcosts,axis = 1)
    ax[1,0].plot(range(1,numP+1),optimalcosts/bestresult,label = i,c= colors[i])
"""
for i in ([6,8,10,12]):
    optimalcosts = np.load(f'Parameterfix{i}NodeGraphTesting.npy')
    bestresult = np.floor(optimalcosts[-1])
    print(optimalcosts)
    ax[0,0].plot(range(1,numP+1),optimalcosts/bestresult,label = i)

    optimalcosts = np.load(f'RandomInitializations{i}NodeGraphTesting.npy')
    optimalcosts = np.average(optimalcosts,axis = 1)
    ax[1,0].plot(range(1,numP+1),optimalcosts/bestresult,label = i)

optimalcosts = np.load(f'Parameterfix{14}NodeGraph.npy')
bestresult = np.floor(optimalcosts[-1])
ax[0,0].plot(range(1,numP+1),optimalcosts/bestresult,label = 14)
optimalcosts = np.load(f'RandomInitializations{14}NodeGraph.npy')
optimalcosts = np.average(optimalcosts,axis = 1)
ax[1,0].plot(range(1,numP+1),optimalcosts/bestresult,label = 14)
ax[0,0].legend()
ax[1,0].legend()
ax[0,0].grid()
ax[1,0].grid()
plt.xticks(range(1,11))

ax[1,0].set_xlabel('Depth p')
ax[0,0].set_ylabel('Approximation ratio r')
ax[1,0].set_ylabel('Average approximation ratio r')
ax[0,0].set_title('u3R graphs with parameterfixing')
ax[1,0].set_title('u3R graphs with random initialization')

for i in ([6,7,8,9,10]):
    optimalcosts = np.load(f'Parameterfix{i}RandomNodeGraph.npy')
    bestresult = np.floor(optimalcosts[-1])
    ax[0,1].plot(range(1,numP+1),optimalcosts/bestresult,label = i)

    optimalcosts = np.load(f'RandomInitializations{i}RandomNodeGraph.npy')
    optimalcosts = np.average(optimalcosts,axis = 1)
    ax[1,1].plot(range(1,numP+1),optimalcosts/bestresult,label = i)

ax[0,1].legend()
ax[1,1].legend()
ax[0,1].grid()
ax[1,1].grid()
plt.xticks(range(1,11))

ax[1,1].set_xlabel('Depth p')
ax[0,1].set_ylabel('Approximation ratio r')
ax[1,1].set_ylabel('Average approximation ratio r')
ax[0,1].set_title('$G(n,0.5)$ graphs with parameterfixing')
ax[1,1].set_title('$G(n,0.5)$ graphs with random initialization')
plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.95, wspace=0.135, hspace=0.125)

plt.show()

gammas = np.arange(0,np.pi + 0.1,0.05)
betas = np.arange(0,np.pi/2 + 0.1,0.05)
meshGamma,meshBeta = np.meshgrid(gammas, betas)

Z = np.load(f'CostLandscape{0},n=8,p = 16, gridsize0.05.npy')
for i in range(1,50):
    Z += np.load(f'CostLandscape{i},n=8,p = 16, gridsize0.05.npy')

#CostLandscapeplotter(meshGamma,meshBeta,Z/50)

gammas = np.arange(0,np.pi + 0.1,0.01)
betas = np.arange(0,np.pi/2 + 0.1,0.01)
meshGamma,meshBeta = np.meshgrid(gammas, betas)
Z = np.load(f'CostLandscape,n=8,p=1,0.01.npy')

#CostLandscapeplotter(meshGamma,meshBeta,Z)


"""
numP = 10

fig,ax = plt.subplots(3,4,sharex = True)
gammaMatrix = np.zeros((numP,numP))
betaMatrix = np.zeros((numP,numP))

for j in range(4):
    optimal_parameters = np.load(f'OptimalParameters{2*(j+1) + 4}NodeGraphTesting.npy')
    for i in range(0,numP):
        gammaMatrix[i,:i+1] = optimal_parameters[i,:i+1]
        betaMatrix[i,:i+1] = optimal_parameters[i,i+1:2*(i+1)]

    gammaMatrix = np.where(gammaMatrix == 0, np.nan, gammaMatrix)
    betaMatrix = np.where(betaMatrix == 0, np.nan, betaMatrix)

    #Create optimal parameter plot similar to Zhou
    ax[0,j].plot(range(1,numP+1),gammaMatrix[-1]/np.pi,marker = '.',ls  = '--',label = r'$\gamma_i$')
    ax[0,j].plot(range(1,numP+1),betaMatrix[-1]/np.pi,marker = '.',ls = '--',label = r'$\beta_i$')
    for g in range(numP):
        ax[1,j].plot(range(1,numP+1), gammaMatrix[:,g]/np.pi,marker = '.',ls = '-.',label = f'{g+1}')
        ax[2,j].plot(range(1,numP+1), betaMatrix[:,g]/np.pi,marker = '.',ls = '-.',label = f'{g+1}')

    ax[0,j].grid()
    ax[1,j].grid()
    ax[2,j].grid()
    ax[2,j].set_xlabel('Depth p')

    ax[1,j].set_ylim(0,1)
    ax[2,j].set_ylim(0,0.5)
ax[0,0].set_ylabel('Angles')
ax[1,0].set_ylabel(r'$\gamma_i/\pi$')
ax[2,0].set_ylabel(r'$\beta_i/\pi$')

ax[0,0].set_title(f'u3R graph with 6 nodes')
ax[0,1].set_title(f'u3R graph with 8 nodes')
ax[0,2].set_title(f'u3R graph with 10 nodes')
ax[0,3].set_title(f'u3R graph with 12 nodes')

handles, labels = ax[2,3].get_legend_handles_labels()
ax[1,3].legend(handles, labels, loc='upper right', ncol = 5,prop={"size":7})
ax[2,3].legend(handles, labels, loc='upper right', ncol = 5,prop={"size":7})

handles, labels = ax[0,3].get_legend_handles_labels()
ax[0,3].legend(handles, labels, loc='right', ncol = 1,prop = {"size":7})
plt.subplots_adjust(left = 0.05,right = 0.99,bottom = 0.05,top = 0.95,wspace = 0.2,hspace = 0.075)
plt.show()






classicalresult = 7 #This was the result from the graph considered when these plots were generated
gammas = np.arange(0,np.pi + 0.1,0.01)
betas = np.arange(0,np.pi/2 + 0.1,0.01)
meshGamma,meshBeta = np.meshgrid(gammas, betas)

Z1 = np.load('CostLandscape,n=6,p=1,0.01.npy')
Z2 = np.load('ParamfixMAXIMUMCostLandscape,n=6,p=2,0.01.npy')
Z3 = np.load('ParamfixMINIMUMCostLandscape,n=6,p=2,0.01.npy')

#CostLandscapeHeatMap(meshGamma,meshBeta,Z1,np.min(Z1),np.max(Z1),r'p = 1')
#CostLandscapeHeatMap(meshGamma,meshBeta,Z2,np.min(Z2),np.max(Z2),r'p = 2')
#CostLandscapeHeatMap(meshGamma,meshBeta,Z3,np.min(Z3),np.max(Z3),r'p = 2')
CostLandscapeplotter(meshGamma,meshBeta,Z1)
CostLandscapeplotter(meshGamma,meshBeta,Z2)
CostLandscapeplotter(meshGamma,meshBeta,Z3)
#CostLandscapeHeatMap(meshGamma,meshBeta,Z1,0,-1*classicalresult,1,6)
#CostLandscapeHeatMap(meshGamma,meshBeta,Z2,0,-1*classicalresult,1,6)
#CostLandscapeHeatMap(meshGamma,meshBeta,Z3,0,-1*classicalresult,1,6)
"""

"""
Nodes8 = np.zeros((10,8))
for i in range(10):
    f = np.load(f'8NodeCosts{i}.npy')
    Nodes8[i] = f

Nodes10 = np.zeros((16,8))
for i in range(16):
    Nodes10[i] = np.load(f'10NodeCosts{i}.npy')

Nodes12 = np.zeros((14,8))
for i in range(14):
    Nodes12[i] = np.load(f'12NodeCosts{i}.npy')

plt.figure()
for i in range(len(Nodes12)):
    plt.plot(range(1,9),Nodes12[i])
    plt.show()

plt.figure()
plt.plot(range(1,9),np.average(Nodes8,axis = 0),label = '8Nodes',marker = 'o',linestyle = '--')
plt.plot(range(1,9),np.average(Nodes10,axis = 0),label = '10Nodes',marker = 'o',linestyle = '--')
plt.plot(range(1,9),np.average(Nodes12,axis = 0),label = '12Nodes',marker = 'o',linestyle = '--')
plt.legend()
plt.title('Average QAOA performance on 3-regular graphs')
plt.xlabel('Ansatz Depth p')
plt.ylabel('Approximation ratio r')
plt.show()

Random10Node = np.zeros((11,16))
for i in range(11):
    Random10Node[i] = np.load(f'10RandomNodeCosts{i}.npy')

plt.figure()
for i in range(11):
    plt.plot(range(1,17),Random10Node[i],marker = '+',linestyle=':',c = 'grey',alpha = 0.5,label = 'Graph Instance')
plt.plot(range(1,17),np.average(Random10Node,axis = 0),marker = 'o',linestyle='-',c = 'blue',label = 'Average')
plt.plot(range(1,17),0.8785*np.ones(len(range(1,17))),color = 'red',linestyle = '--',label = 'Goemans-Williamson Algorithm')
plt.xlabel('Ansatz Depth p')
plt.ylabel('Approximation ratio r')
plt.title('QAOA Performance on 10 node Erdős–Rényi graphs')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()

Random11Node = np.zeros((11,16))
for i in range(11):
    Random11Node[i] = np.load(f'11RandomNodeCosts{i}.npy')

plt.figure()
for i in range(11):
    plt.plot(range(1,17),Random11Node[i],marker = '+',linestyle=':',c = 'grey',alpha = 0.5,label = 'Graph Instance')
plt.plot(range(1,17),np.average(Random11Node,axis = 0),marker = 'o',linestyle='-',c = 'blue',label = 'Average')
plt.plot(range(1,17),0.8785*np.ones(len(range(1,17))),color = 'red',linestyle = '--',label = 'Goemans-Williamson Algorithm')
plt.xlabel('Ansatz Depth p')
plt.ylabel('Approximation ratio r')
plt.title('QAOA Performance on 11 node Erdős–Rényi graphs')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()
"""

