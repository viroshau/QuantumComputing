import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math as math
from matplotlib.ticker import FormatStrFormatter


def LinearInterpolation1DVector(parameters):
    p = len(parameters)//2
    #results = LinearInterpolationInit(np.reshape(parameters,(2,p)))
    gammas = np.interp(np.linspace(1,p,p+1,endpoint = True),range(1,p+1),parameters[0:p])
    betas = np.interp(np.linspace(1,p,p+1,endpoint = True),range(1,p+1),parameters[p:])
    results = np.concatenate((gammas,betas))
    return results

numP = 10
nodes = 10

"""
for j in range(10):
    fig,ax = plt.subplots(nrows = 2,ncols= 2)
    optimalcosts = np.load(f'INTERPParameterfix{nodes}NodeGraphTesting{j}.npy')
    optimal_parameters  = np.load(f'INTERPOptimalParameters{nodes}NodeGraphTesting{j}.npy')

    bestresult = np.floor(np.min(optimalcosts))
    ax[0,0].plot(range(1,numP+1),optimalcosts/bestresult)
    ax[0,0].set_xlabel('p')
    ax[0,0].set_ylabel('Approximation ratio r')
    ax[0,0].set_title('Performance of INTERP')

    gammaMatrix = np.zeros((numP,numP))
    betaMatrix = np.zeros((numP,numP))

    for i in range(0,numP):
        gammaMatrix[i,:i+1] = optimal_parameters[i,:i+1]
        betaMatrix[i,:i+1] = optimal_parameters[i,i+1:2*(i+1)]

    ax[0,1].plot(range(1,numP+1),gammaMatrix[-1]/np.pi,marker = '.',ls  = '--',label = r'$\gamma_i$')
    ax[0,1].plot(range(1,numP+1),betaMatrix[-1]/np.pi,marker = '.',ls = '--',label = r'$\beta_i$')
    ax[0,1].set_xlabel('i')
    ax[0,1].set_ylabel('Angles/$\pi$')
    ax[0,1].set_title(r'($\gamma_i^*,\beta_i^*$) at p = 10')

    uppertriagindices = np.triu_indices(numP,k = 1)
    gammaMatrix[uppertriagindices] = np.nan
    betaMatrix[uppertriagindices] = np.nan

    for i in range(numP):
        ax[1,0].scatter(range(1,numP+1), gammaMatrix[:,i]/np.pi,marker = 'o',label = f'{i+1}')
        ax[1,1].scatter(range(1,numP+1), betaMatrix[:,i]/np.pi,marker = 'o',label = f'{i+1}')
    ax[1,0].set_ylabel(r'$\gamma_i/\pi$')
    ax[1,1].set_ylabel(r'$\beta_i/\pi$')
    ax[1,0].set_xlabel('p')
    ax[1,1].set_xlabel('p')
    results = np.zeros_like(optimal_parameters)
    results[0,:2] = optimal_parameters[0,:2]
    for i in range(1,numP):
        results[i,:2*(i+1)] = LinearInterpolation1DVector(optimal_parameters[i-1,:2*i])

    gammaMatrix = np.zeros((numP,numP))
    betaMatrix = np.zeros((numP,numP))

    for i in range(0,numP):
        gammaMatrix[i,:i+1] = results[i,:i+1]
        betaMatrix[i,:i+1] = results[i,i+1:2*(i+1)]
    gammaMatrix[uppertriagindices] = np.nan
    betaMatrix[uppertriagindices] = np.nan

    for i in range(numP):
        ax[1,0].plot(range(1,numP+1), gammaMatrix[:,i]/np.pi,marker = 'o',label = f'{i+1}',ls = '--',alpha = 0.2)
        ax[1,1].plot(range(1,numP+1), betaMatrix[:,i]/np.pi,marker = 'o',label = f'{i+1}',ls = '--',alpha = 0.2)
plt.show()
"""

""" 
nodes = 10
totalcosts = np.load(f'INTERPTotalSimsCosts{nodes}NodeGraph.npy')
parameters = np.load(f'INTERPTotalSimsOptimalParameters{nodes}NodeGraph.npy')

totalcosts = np.load(f'INTERPTotalSimsCosts{nodes}NodeGraphFewerAttempts1.npy')
parameters = np.load(f'INTERPTotalSimsOptimalParameters{nodes}NodeGraphFewerAttempts1.npy')

for k in range(25):
    for j in range(1):
        fig,ax = plt.subplots(nrows = 2,ncols= 2)
        optimalcosts = totalcosts[k,j]
        optimal_parameters  = parameters[k,j]

        bestresult = np.floor(np.min(optimalcosts))
        ax[0,0].plot(range(1,numP+1),optimalcosts/bestresult,marker = 'o',ls = '--',label = f'{nodes} Node graph')
        ax[0,0].set_xlabel('$p$')
        ax[0,0].set_ylabel('$r$')
        ax[0,0].set_title('Approximation ratio $r$')
        ax[0,0].legend()

        gammaMatrix = np.zeros((numP,numP))
        betaMatrix = np.zeros((numP,numP))

        for i in range(0,numP):
            gammaMatrix[i,:i+1] = optimal_parameters[i,:i+1]
            betaMatrix[i,:i+1] = optimal_parameters[i,i+1:2*(i+1)]

        ax[0,1].plot(range(1,numP+1),gammaMatrix[-1]/np.pi,marker = 'o',ls  = '--',label = r'$\gamma_i$')
        ax[0,1].plot(range(1,numP+1),betaMatrix[-1]/np.pi,marker = 'o',ls = '--',label = r'$\beta_i$')
        ax[0,1].set_xlabel('$i$')
        ax[0,1].set_ylabel('Angles/$\pi$')
        ax[0,1].set_title(r'($\gamma_i^*,\beta_i^*$) at $p$ = 10')
        ax[0,1].legend()

        uppertriagindices = np.triu_indices(numP,k = 1)
        gammaMatrix[uppertriagindices] = np.nan
        betaMatrix[uppertriagindices] = np.nan

        for i in range(numP):
            ax[1,0].plot(range(1,numP+1), gammaMatrix[:,i]/np.pi,marker = 'o',ls = '--',label = f'{i+1}')
            ax[1,1].plot(range(1,numP+1), betaMatrix[:,i]/np.pi,marker = 'o',ls = '--',label = f'{i+1}')
        ax[1,0].set_ylabel(r'$\gamma_i/\pi$')
        ax[1,1].set_ylabel(r'$\beta_i/\pi$')
        ax[1,0].set_xlabel('$p$')
        ax[1,1].set_xlabel('$p$')
        handles, labels = ax[1,0].get_legend_handles_labels()
        ax[1,0].legend(handles, labels, loc='lower left', ncol = 5,prop={"size":7})
        ax[1,1].legend(handles, labels, loc='lower left', ncol = 5,prop={"size":7})
        ax[1,0].set_title('Optimal $\gamma_i^*$ at each depth $p$')
        ax[1,1].set_title(r'Optimal $\beta_i^*$ at each depth $p$')

        ax[1,0].set_xticks(range(1,11))
        ax[1,1].set_xticks(range(1,11))
        ax[0,0].set_xticks(range(1,11))
        ax[0,1].set_xticks(range(1,11))

        ax[1,0].set_ylim(0,1)
        ax[1,1].set_ylim(0,0.5)

        ax[1,0].grid()
        ax[1,1].grid()
        ax[0,0].grid()
        ax[0,1].grid()

        results = np.zeros_like(optimal_parameters)
        results[0,:2] = optimal_parameters[0,:2]
        for i in range(1,numP):
            results[i,:2*(i+1)] = LinearInterpolation1DVector(optimal_parameters[i-1,:2*i])

        gammaMatrix = np.zeros((numP,numP))
        betaMatrix = np.zeros((numP,numP))

        for i in range(0,numP):
            gammaMatrix[i,:i+1] = results[i,:i+1]
            betaMatrix[i,:i+1] = results[i,i+1:2*(i+1)]
        gammaMatrix[uppertriagindices] = np.nan
        betaMatrix[uppertriagindices] = np.nan

        #for i in range(numP):
        #    ax[1,0].plot(range(1,numP+1), gammaMatrix[:,i]/np.pi,marker = 'o',label = f'{i+1}',ls = '--',alpha = 0.2)
        #    ax[1,1].plot(range(1,numP+1), betaMatrix[:,i]/np.pi,marker = 'o',label = f'{i+1}',ls = '--',alpha = 0.2)
        plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.95, wspace=0.135, hspace=0.225)
        ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.show()
"""

random = False

colors = [0,0,0,0,0,0,'tab:blue',0,'tab:orange',0,'tab:green',0,'tab:red',0,'tab:purple']
colors = [0,0,0,0,0,0,'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink']

fig,ax = plt.subplots(ncols = 2)
for j in [6,7,8,9,10,12]:
    totalcosts = np.load(f'FinalINTERPRandomCosts{j}NodeGraph.npy')
    parameters = np.load(f'FinalINTERPRandomParameters{j}NodeGraph.npy')
    print(np.shape(totalcosts))
    bestresult = np.floor(totalcosts[:,-1])
    totalcosts = totalcosts/bestresult[:,None]

    variances = np.std(totalcosts,axis =0)
    variancesParameters = np.std(parameters,axis = 0)

    y1 = np.min(totalcosts,axis = 0)
    y2 = np.max(totalcosts,axis = 0)

    optimalcosts = np.average(totalcosts,axis = 0)
    optimal_parameters  = np.average(parameters,axis = 0)

    ax[0].plot(range(1,numP+1),optimalcosts,marker = 'o',ls = '--',label = f'{j} Node graph',c = colors[j])
    #ax[0].fill_between(range(1,numP+1), y1, y2,color = colors[j],alpha = 0.1)
    ax[0].set_xlabel('$p$')
    ax[0].set_ylabel('$r$')
    ax[0].set_title('Approximation ratio $r$')
    ax[0].legend(fontsize = 'small')

    gammaMatrix = np.zeros((numP,numP))
    betaMatrix = np.zeros((numP,numP))

    gammaVarianceMatrix = np.zeros((numP,numP))
    betaVarianceMatrix = np.zeros((numP,numP))
    for i in range(0,numP):
        gammaMatrix[i,:i+1] = optimal_parameters[i,:i+1]
        betaMatrix[i,:i+1] = optimal_parameters[i,i+1:2*(i+1)]
        gammaVarianceMatrix[i,:i+1] = variancesParameters[i,:i+1]
        betaVarianceMatrix[i,:i+1] = variancesParameters[i,i+1:2*(i+1)]

    ax[1].plot(range(1,numP+1),gammaMatrix[-1]/np.pi,marker = 'o',ls  = '-',label = r'$\gamma_i$,'+f' {j} Node graph',c = colors[j])
    ax[1].plot(range(1,numP+1),betaMatrix[-1]/np.pi,marker = 'd',ls = ':',label = r'$\beta_i$,'+f' {j} Node graph',c = colors[j])
    ax[1].fill_between(range(1,numP+1), gammaMatrix[-1]/np.pi + gammaVarianceMatrix[-1]/np.pi, gammaMatrix[-1]/np.pi - gammaVarianceMatrix[-1]/np.pi,color = colors[j],alpha = 0.1)
    ax[1].fill_between(range(1,numP+1), betaMatrix[-1]/np.pi + betaVarianceMatrix[-1]/np.pi, betaMatrix[-1]/np.pi - betaVarianceMatrix[-1]/np.pi,color = colors[j],alpha = 0.1)
    ax[1].set_xlabel('$i$')
    ax[1].set_ylabel('Angles/$\pi$')
    ax[1].set_title(r'($\gamma_i^*,\beta_i^*$) at $p$ = 10')
    ax[1].legend(fontsize = 'small')

    ax[0].set_xticks(range(1,11))
    ax[1].set_xticks(range(1,11))
    ax[0].grid()
    ax[1].grid()

    fig.suptitle('Average performance of INTERP on u3R graphs')
    plt.subplots_adjust(left=0.05, right=0.975, wspace=0.135, hspace=0.225)

plt.show()



