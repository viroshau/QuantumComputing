import pennylane as qml
import networkx as nx
from pennylane.qnode import qnode
from scipy.optimize import minimize
from pennylane import numpy as np
import matplotlib.pyplot as plt
import cvxgraphalgs as cvxgr
import time
import tqdm as tqdm

def binary_representation(array):
    textarray = []
    for i in array:
        textarray.append(str(np.binary_repr(array[i], width=len(graph.nodes())) ))
    return textarray

def create_graph(degree,nodes,edges = [],plotting = True):
    if np.shape(edges) == (0,):
        #If weights is set to None, create a random regular graph of given degree
        graph = nx.generators.random_regular_graph(degree,nodes)
        nx.set_edge_attributes(graph, values = 1, name = 'weight')
        edges = [e for e in graph.edges.data("weight", default=1)]
        if plotting == True:
            pos=nx.circular_layout(graph)
            nx.draw(graph, pos, with_labels=True, font_weight='bold')
            edge_weight = nx.get_edge_attributes(graph,'weight')
            nx.draw_networkx_edge_labels(graph, pos, edge_labels = edge_weight)
            plt.show()
        return edges,graph
    else:
        #In this case, you have to input a list of [(node1,node2,weight)] to create the graph
        graph = nx.Graph()
        for i,j,weight in edges:
            graph.add_edge(i,j,weight = weight)
        if plotting == True:
            pos=nx.circular_layout(graph)
            nx.draw(graph, pos, with_labels=True, font_weight='bold')
            edge_weight = nx.get_edge_attributes(graph,'weight')
            nx.draw_networkx_edge_labels(graph, pos, edge_labels = edge_weight)
            plt.show()
        return edges, graph

def ProblemUnitary(edge1, edge2, t, weight):
    qml.CNOT(wires=[edge1, edge2])
    qml.RZ(-t * weight, wires=edge2)
    qml.CNOT(wires=[edge1, edge2])

def MixerUnitary(edge1, t):
    qml.RX(t,wires = edge1)
    """
    qml.Hadamard(wires=edge1)
    qml.RZ(2 * t, wires=edge1)
    qml.Hadamard(wires=edge1)
    """

def OneLayer(gamma, beta, graph):
    for i, j in graph.edges():
        ProblemUnitary(i, j, gamma, 1)
    for i in graph.nodes():
        MixerUnitary(i, beta)

def LinearInterpolationInit(parameters):
    """Generates the initial parameters used for optimization at depth p+1

    Args:
        parameters ([type]): [A matrix of size (2,p) where first row is gamma and second is beta]
    """
    p = len(parameters[0])
    NewParameters = np.zeros((2,p+1))
    for i in range(p+1):
        if i == 0:
            NewParameters[0,i] = ((p-i)/p)*parameters[0,i]
            NewParameters[1,i] = ((p-i)/p)*parameters[1,i]
        elif i == p:
            NewParameters[0,i] = ((i)/p)*parameters[0,i-1]
            NewParameters[1,i] = ((i)/p)*parameters[1,i-1]
        else:
            NewParameters[0,i] = ((i)/p)*parameters[0,i-1] + ((p-i)/p)*parameters[0,i]
            NewParameters[1,i] = ((i)/p)*parameters[1,i-1] + ((p-i)/p)*parameters[1,i]
        return NewParameters

def CreateProblemHamiltonian(edges):
    #H_C = 0.5*SUM(Z_i*Z_j - I)*w_ij is a costfunction where the global minima is the correct solution
    """Creates the Hamiltonian using the list of (edge1,edge2,weight)

    Args:
        edges ([type]): [1D list with elements (edge1,edge2,weight)]

    Returns:
        [type]: [Returns a qml.Hamiltonian object encoding the hamiltonian]
    """
    coeff = []
    observables = []
    tot_weight = 0
    for i,j,w in edges:
        coeff.append(0.5*w)
        observables.append(qml.PauliZ(i)@qml.PauliZ(j))
        tot_weight +=w
    coeff.append(-0.5*tot_weight)
    observables.append(qml.Identity(0))
    return qml.Hamiltonian(coeff,observables)

def CostLandscapeplotter(meshGamma,meshBeta,Z, p = 0):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(meshGamma,meshBeta,Z,cmap='coolwarm')
    ax.set_ylabel(r'$\beta_p$')
    ax.set_xlabel(r'$\gamma_p$')
    ax.set_zlabel('C')
    ax.set_title(f'Depth p = {p}')
    #ax.scatter(0.58821361, 0.36506835,np.min(Z),s = 150,marker = 'D',c = 'darkorange')
    #ax.scatter(0.64220579,1.14829102,np.max(Z),s = 150,marker = 'D',c = 'seagreen')
    plt.show()  

def CostLandscapeHeatMap(meshGamma,meshBeta,Z,z_min,z_max,p):
    fig = plt.figure()
    c = plt.pcolormesh(meshGamma,meshBeta,Z,vmin = z_min,vmax = z_max,cmap = 'coolwarm')
    plt.ylabel(r'$\beta_p$')
    plt.xlabel(r'$\gamma_p$')
    plt.xlim(0,np.pi)
    plt.ylim(0,np.pi/2)
    plt.title(f'Cost-Landscape for a {len(graph.nodes())} nodes u{3}R graph, p = {p}')
    plt.colorbar(c)
    plt.show()  

def bestClassicalHeuristicResult(graph):
    cut_value_classical = 0
    for i in range(10):
        cut_value1,partition = nx.algorithms.approximation.maxcut.one_exchange(graph)
        cut_value2,partition = nx.algorithms.approximation.maxcut.one_exchange(graph)
        recovered = cvxgr.algorithms.goemans_williamson_weighted(graph)
        cut_value3 = recovered.evaluate_cut_size(graph)
        cut_value_classical = max(cut_value_classical,cut_value1,cut_value2,cut_value3)
    print(f'The best classical cost is: {cut_value_classical}')
    return cut_value_classical

@np.vectorize
def CalculateCostlandscape(gamma,beta):
    global z_min,z_max,cost_h
    params = np.array([gamma,beta])
    cost =  parameterfixedcircuitONLYPARAMSVERSION(params,cost_h)
    if cost < z_min:
        z_min = cost
    if cost > z_max:
        z_max  = cost
    return cost

def CostLandscapeProcedureFull(gammabound,betabound,discretization,p):
    gammas = np.arange(*gammabound,discretization)
    betas = np.arange(*betabound,discretization)

    meshGamma,meshBeta = np.meshgrid(gammas, betas)

    start_time = time.time()
    Z = CalculateCostlandscape(meshGamma,meshBeta)
    end_time = time.time()
    print(f"The execution time of Meshgrid-calcs is: {end_time-start_time}")
    #CostLandscapeplotter(meshGamma,meshBeta,Z,p)
    return Z

def Simulation(numP,numRandomAttempts,gammabound,betabound,cost_h):
    costs = np.zeros((numP,numRandomAttempts))
    optimal_parameters = np.zeros((numP,2*numP))
    fixedpar = np.array([],requires_grad = False)

    for j in range(numP):
        print(f'Status of whole simulation: {j+1}/{numP}')
        #This is the p-level circuit
        bestcost = 0
        bestPLayerParameter = np.zeros(2*(j+1))
        bounds = tuple([gammabound for i in range(len(bestPLayerParameter)//2)] + [betabound for i in range(len(bestPLayerParameter)//2)])
        
        for i in range(numRandomAttempts):
            fixedgamma = np.append(fixedpar[:len(fixedpar)//2],np.random.uniform(*gammabound))
            fixedbeta = np.append(fixedpar[len(fixedpar)//2:],np.random.uniform(*betabound))
            params = np.concatenate((fixedgamma,fixedbeta),requires_grad = True)

            optimizer = minimize(circuit, params, args = (cost_h), method='L-BFGS-B', bounds = bounds, jac = qml.grad(circuit, argnum=0))
            #optimizer = minimize(parameterfixedcircuit, changableparams, args = (fixedpar,), method='NELDER-MEAD', bounds = bounds)
            if optimizer.fun < bestcost:
                bestcost = optimizer.fun
                bestPLayerParameter = optimizer.x

            costs[j,i] = optimizer.fun
            print(f'Round{i}/{numRandomAttempts}',end = '\r')

        fixedpar = bestPLayerParameter
        optimal_parameters[j,:2*(j+1)] = fixedpar
        print(f'BestParameter = {fixedpar}')
        print(f'Current best r = {-1*bestcost/classicalresult}')
        print(f'Current Average r = {-1*np.average(costs[j])/classicalresult}')
    return costs, optimal_parameters

def SimulationZhouMethod(numP,numRandomAttempts,gammabound,betabound,cost_h):
    costs = np.zeros(numP)
    optimal_parameters = np.zeros((numP,2*numP))
    """ 
    params = np.array([],requires_grad = False)
    cost1layer = 0
    for i in range(numRandomAttempts):
        bounds = [gammabound,betabound]
        randomparam = np.concatenate((np.random.uniform(*gammabound,size = 1),np.random.uniform(*betabound,size = 1)))
        optimizer = minimize(circuit, randomparam, args = (cost_h), method='L-BFGS-B', bounds = bounds, jac = qml.grad(circuit, argnum=0))
        if optimizer.fun < cost1layer:
            cost1layer = optimizer.fun 
            params = optimizer.x
    costs[0] = cost1layer
    optimal_parameters[0,:2] = params
    """
    params = np.array([0.2*np.pi,0.12*np.pi],requires_grad = True)
    bounds = [gammabound,betabound]
    optimizer = minimize(circuit, params, args = (cost_h), method='L-BFGS-B', bounds = bounds, jac = qml.grad(circuit, argnum=0))
    params = optimizer.x 
    costs[0] = optimizer.fun
    optimal_parameters[0,:2] = optimizer.x
    for j in range(1,numP):
        #print(f'Status of whole simulation: {j+1}/{numP}')
        #This is the p-level circuit
        params = LinearInterpolation1DVector(params) #interpolates parameters to initial value at layer j+1
        bounds = tuple([gammabound for i in range(len(params)//2)] + [betabound for i in range(len(params)//2)])
    
        optimizer = minimize(circuit, params, args = (cost_h), method='L-BFGS-B', bounds = bounds, jac = qml.grad(circuit, argnum=0))
        
        params = optimizer.x

        costs[j] = optimizer.fun
        optimal_parameters[j,:2*(j+1)] = params

        #print(f'Current best r = {-1*optimizer.fun/classicalresult}')
    return costs, optimal_parameters

def SimulationZhouMethodFourier(numP,numRandomAttempts,gammabound,betabound,cost_h):
    costs = np.zeros(numP)
    optimal_parameters = np.zeros((numP,2*numP))
    params = np.array([],requires_grad = False)

    cost1layer = 0
    for i in range(numRandomAttempts):
        randomparam = np.concatenate((np.random.uniform(*gammabound,size = 1),np.random.uniform(*betabound,size = 1)))
        optimizer = minimize(FOURIERCost, randomparam, args = (cost_h), method='BFGS')
        if optimizer.fun < cost1layer:
            cost1layer = optimizer.fun 
            params = optimizer.x

    costs[0] = cost1layer
    optimal_parameters[0,:2] = FromUVtoQAOAaParameter(params)
    for j in range(1,numP):
        print(f'Status of whole simulation: {j+1}/{numP}')
        #This is the p-level circuit
        uvector = np.append(params[:len(params)//2],0)
        vvector = np.append(params[len(params)//2:],0)
        params = np.concatenate((uvector,vvector))
        optimizer = minimize(FOURIERCost, params, args = (cost_h), method='BFGS')
        
        params = optimizer.x

        costs[j] = optimizer.fun
        optimal_parameters[j,:2*(j+1)] = FromUVtoQAOAaParameter(params)

        print(f'BestParameter = {optimal_parameters[j,:2*(j+1)]}')
        print(f'Current best r = {-1*optimizer.fun/classicalresult}')
    return costs, optimal_parameters
    
def AverageSimulation(numP,numRandomAttempts,gammabound,betabound,cost_h):
    costs = np.zeros((numP,numRandomAttempts))

    for j in range(numP):
        print(f'Status of whole simulation: {j+1}/{numP}')
        #This is the p-level circuit
        bounds = tuple([gammabound for i in range(j+1)] + [betabound for i in range(j+1)])
        
        for i in range(numRandomAttempts):
            randomgamma = np.random.uniform(*gammabound,size = j+1)
            randombeta = np.random.uniform(*betabound,size = j+1)
            params = np.concatenate((randomgamma,randombeta),requires_grad = True)

            optimizer = minimize(circuit, params, args = (cost_h), method='L-BFGS-B', bounds = bounds, jac = qml.grad(circuit, argnum=0))
            costs[j,i] = optimizer.fun

            print(f'Round{i}/{numRandomAttempts}',end = '\r')
        print(f'Average r = {-1*np.average(costs[j])/classicalresult}')
    
    return costs

def LinearInterpolationInit(parameters):
    """Generates the initial parameters used for optimization at depth p+1

    Args:
        parameters ([type]): [A matrix of size (2,p) where first row is gamma and second is beta]
    """
    p = len(parameters[0])
    NewParameters = np.zeros((2,p+1))
    for i in range(p+1):
        if i == 0:
            NewParameters[0,i] = ((p-i)/p)*parameters[0,i]
            NewParameters[1,i] = ((p-i)/p)*parameters[1,i]
        elif i == p:
            NewParameters[0,i] = ((i)/p)*parameters[0,i-1]
            NewParameters[1,i] = ((i)/p)*parameters[1,i-1]
        else:
            NewParameters[0,i] = ((i)/p)*parameters[0,i-1] + ((p-i)/p)*parameters[0,i]
            NewParameters[1,i] = ((i)/p)*parameters[1,i-1] + ((p-i)/p)*parameters[1,i]
    return NewParameters

def LinearInterpolation1DVector(parameters):
    p = len(parameters)//2
    #results = LinearInterpolationInit(np.reshape(parameters,(2,p)))
    gammas = np.interp(np.linspace(1,p,p+1,endpoint = True),range(1,p+1),parameters[0:p])
    betas = np.interp(np.linspace(1,p,p+1,endpoint = True),range(1,p+1),parameters[p:])
    results = np.concatenate((gammas,betas))
    return results
#Create graph instance

def FromUVtoQAOAaParameter(parameters):
    p = len(parameters)//2
    gammas = np.zeros(p)
    betas  = np.zeros(p)
    for i in range(p):
        for k in range(p):
            gammas[i]+= parameters[k]*np.sin((k+0.5)*(i+0.5)*(np.pi/p))
            betas[i] += parameters[p+k]*np.cos((k+0.5)*(i+0.5)*(np.pi/p))
    return np.concatenate((gammas,betas))

z_min = 500
z_max = -500
numRandomAttempts = 20
numP = 10

nodes = 10
renyuigraph = False

if renyuigraph:
    betabound = (0,np.pi/2)
    gammabound = (0,2*np.pi)
    graph = nx.fast_gnp_random_graph(nodes,0.5,seed = 123)
else:
    betabound = (0,np.pi/2)
    gammabound = (0,np.pi)
    graph = nx.generators.random_regular_graph(3,nodes,seed = 123)

nqubits = len(graph.nodes())
classicalresult = bestClassicalHeuristicResult(graph)
cost_h, mixer_h = qml.qaoa.maxcut(graph)

"""
edges = np.array(graph.edges(),requires_grad=False)
edgestuple = np.zeros((len(edges),3))
for i in range(len(edges)):
    edgestuple[i] = (edges[i,0],edges[i,1],1)
edges,graph = create_graph(0,0,edgestuple,plotting = False)
"""

dev = qml.device('default.qubit', wires=nqubits)
@qml.qnode(dev)
def circuit(x,cost_h):
    for i in range(nqubits):
        qml.Hadamard(wires = i)
    
    p = len(x)//2
    for i in range(p):
        for j,k in graph.edges():
            qml.CNOT(wires = [j,k])
            qml.RZ(-x[:p][i], wires = k)
            qml.CNOT(wires = [j,k])

        #for j,k,w in edges:
            #qml.CNOT(wires = [int(j.item()),int(k.item())])
            #qml.RZ(-x[:p][i], wires = int(k.item()))
            #qml.CNOT(wires = [int(j.item()),int(k.item())])

        for j in range(nqubits):
            qml.RX(2*x[p:][i],wires = j)

    return qml.expval(cost_h.item())

@qml.qnode(dev)
def parameterfixedcircuit(params,x,cost_h):
    for i in range(nqubits):
        qml.Hadamard(wires = i)
    
    p = len(x)//2
    for i in range(p):
        for j,k in graph.edges():
            qml.CNOT(wires = [j,k])
            qml.RZ(-x[:p][i], wires = k)
            qml.CNOT(wires = [j,k])
    
        for j in range(nqubits):
            qml.RX(2*x[p:][i],wires = j)

    p = len(params)//2
    for i in range(p):
        for j,k in graph.edges():
            qml.CNOT(wires = [j,k])
            qml.RZ(-params[:p][i], wires = k)
            qml.CNOT(wires = [j,k])
    
        for j in range(nqubits):
            qml.RX(2*params[p:][i],wires = j)
    
    return qml.expval(cost_h.item())

globalx = np.array([],requires_grad = False)
@qml.qnode(dev)
def parameterfixedcircuitONLYPARAMSVERSION(params,cost_h):
    for i in range(nqubits):
        qml.Hadamard(wires = i)
    
    p = len(globalx)//2
    for i in range(p):
        for j,k in graph.edges():
            qml.CNOT(wires = [j,k])
            qml.RZ(-globalx[:p][i], wires = k)
            qml.CNOT(wires = [j,k])
    
        for j in range(nqubits):
            qml.RX(2*globalx[p:][i],wires = j)

    p = len(params)//2
    for i in range(p):
        for j,k in graph.edges():
            qml.CNOT(wires = [j,k])
            qml.RZ(-params[:p][i], wires = k)
            qml.CNOT(wires = [j,k])
    
        for j in range(nqubits):
            qml.RX(2*params[p:][i],wires = j)
    
    return qml.expval(cost_h.item())

def FOURIERCost(x,cost_h):
    parameters = FromUVtoQAOAaParameter(x)
    return circuit(parameters,cost_h)

""" 
globalx = np.concatenate((np.random.uniform(*gammabound,size = 9),np.random.uniform(*betabound,size = 9)))
Z = CostLandscapeProcedureFull(gammabound,betabound,0.1,10)
for i in range(25):
    globalx = np.concatenate((np.random.uniform(*gammabound,size = 9),np.random.uniform(*betabound,size = 9)))
    Z += CostLandscapeProcedureFull(gammabound,betabound,0.1,10)

np.save('AverageZ.npy',Z)
""" 
Z = np.load('AverageZ.npy')
gammas = np.arange(*gammabound,0.1)
betas = np.arange(*betabound,0.1)
meshGamma,meshBeta = np.meshgrid(gammas, betas)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(meshGamma,meshBeta,Z/26,cmap='coolwarm')
ax.set_ylabel(r'$\beta_p$')
ax.set_xlabel(r'$\gamma_p$')
ax.set_zlabel('C')
ax.set_title(f'Depth p = {10}')
plt.show()
stop

"""
#SimulationZhouMethodFourier(numP,numRandomAttempts,gammabound,betabound,cost_h)
graphssize = 10
attemptspergraph = 1
Totalcosts = np.zeros((graphssize,attemptspergraph,numP))
Totaloptimalparameters = np.zeros((graphssize,attemptspergraph,numP,2*numP))
for i in tqdm.tqdm(range(graphssize)):
    #graph = nx.generators.random_regular_graph(3,nodes)
    graph = nx.fast_gnp_random_graph(nodes,0.5)
    classicalresult = bestClassicalHeuristicResult(graph)
    cost_h, mixer_h = qml.qaoa.maxcut(graph)
    for j in (range(attemptspergraph)):
        optimalcosts,optimal_parameters = SimulationZhouMethod(numP,numRandomAttempts,gammabound,betabound,cost_h)
        Totalcosts[i,j] = optimalcosts
        Totaloptimalparameters[i,j] = optimal_parameters

np.save(f'FinalINTERPRandomCosts6NodeGraph.npy',np.squeeze(Totalcosts))
np.save(f'FinalINTERPRandomParameters6NodeGraph.npy',np.squeeze(Totaloptimalparameters))

np.save(f'INTERPTotalSimsCosts{nqubits}NodeRandomGraphFewerAttempts1.npy',Totalcosts)
np.save(f'INTERPTotalSimsOptimalParameters{nqubits}NodeRandomGraphFewerAttempts1.npy',Totaloptimalparameters)
#globalx = np.array([3.14159265,3.14159265,3.14159265,3.14159265,3.14159265,3.14159265,3.14159265,3.14159265,3.14159265,3.14159265,0.,0.,0.,0.,0.,0.,1.57079633,1.57079633,0.,1.57079633])
"""

""" 
optimalcosts,optimal_parameters = Simulation(numP,numRandomAttempts,gammabound,betabound,cost_h)
averagecosts = AverageSimulation(numP,numRandomAttempts,gammabound,betabound,cost_h)

np.save(f'Parameterfix{nqubits}NodeGraphTestingMAXIMIZATION.npy',np.min(optimalcosts,axis = 1))
np.save(f'OptimalParameters{nqubits}NodeGraphTestingMAXIMIZATION.npy',optimal_parameters)
np.save(f'RandomInitializations{nqubits}NodeGraphTesting.npy',averagecosts)
"""
#optimalcosts,optimal_parameters = SimulationZhouMethod(numP,numRandomAttempts,gammabound,betabound,cost_h)

#optimalcosts = np.load(f'INTERPParameterfix{6}NodeGraphTesting{3}.npy')
#optimal_parameters = np.load(f'INTERPOptimalParameters{6}NodeGraphTesting{3}.npy')
gammaMatrix = np.zeros((numP,numP))
betaMatrix = np.zeros((numP,numP))

for i in range(0,numP):
    gammaMatrix[i,:i+1] = optimal_parameters[i,:i+1]
    betaMatrix[i,:i+1] = optimal_parameters[i,i+1:2*(i+1)]

plt.figure()
plt.plot(range(1,numP+1),gammaMatrix[-1]/np.pi,marker = '.',ls  = '--',label = r'$\gamma_i$')
plt.plot(range(1,numP+1),betaMatrix[-1]/np.pi,marker = '.',ls = '--',label = r'$\beta_i$')
plt.show()

gammaMatrix = np.where(gammaMatrix == 0, np.nan, gammaMatrix)
betaMatrix = np.where(betaMatrix == 0, np.nan, betaMatrix)

fig, ax = plt.subplots(2, sharex='col')
for i in range(numP):
    ax[0].plot(range(1,numP+1), gammaMatrix[:,i],marker = 'o',label = f'{i+1}')
    ax[1].plot(range(1,numP+1), betaMatrix[:,i],marker = 'o',label = f'{i+1}')
ax[1].set_xlabel('i')
ax[0].set_ylabel(r'$\gamma_i$')
ax[1].set_ylabel(r'$\beta_i$')
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center',ncol = numP)
plt.show()

plt.figure()
plt.plot(range(1,numP+1),-1*optimalcosts/classicalresult,label = 'Parameterfix')
plt.legend()
plt.show()

"""
p = 16
params = np.zeros(2*p)

optimalparameters = np.zeros(2*p)
bestcost = 0
start_time = time.time()
for i in range(20):
    params[:p] = np.random.uniform(-np.pi/2,np.pi/2,size = p,requires_grad = True)
    params[p:] = np.random.uniform(-np.pi/4,np.pi/4,size = p,requires_grad = True)

    bounds1 = [(-np.pi/2,np.pi/2) for i in range(p)]
    bounds2 = [(-np.pi/4,np.pi/4) for i in range(p)]
    bounds = bounds1 + bounds2
    bounds = tuple(bounds)

    optimizer = minimize(circuit, params, method='L-BFGS-B', jac = qml.grad(circuit, argnum=0),bounds = bounds)

    if optimizer.fun < bestcost:
        bestcost = optimizer.fun
        optimalparameters = optimizer.x
    print(f'Round {i}/{10}', end = '\r')
end_time = time.time()


print(f"Training took: {end_time-start_time}  seconds")   
print(f'Performance r = {-1*bestcost/classicalresult}')
print(f'{optimalparameters[:p]}')
print(f'{optimalparameters[p:]}')

plt.figure()
plt.plot(range(1,p+1),optimalparameters[:p]/np.pi,label = r'\gamma')
plt.plot(range(1,p+1),optimalparameters[p:]/np.pi,label = r'\beta')
plt.legend()
plt.show()
"""