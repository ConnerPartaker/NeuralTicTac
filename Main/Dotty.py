import random
import numpy as np
import matplotlib.pyplot as plt
import Main.NumpyNN as NN



SIZE = [2, 4, 1]    #For this mode, must begin with 2, end with 1
ITER = 500            #Number of iterations per test check and user update
TESTS= 50           #Number of cases to test against. The larger, the more accurate, but longer
WKSTS= 500
#Must return [0] or [1]
def func(array): return [0 if sum(np.abs(array))>.5 else 1]
    


def getinp(x):
    return [round(2*random.random()-1, 4) for _ in range(x)]

def test(size = SIZE, iters = ITER, testnum = TESTS, f = func):
    
    nn = NN.NeuralNet(size, ['ReLu']*(len(size)-2))
        
    #Present them in a graph
    testsX, testsY, testsZ = [], [], []
    scttrX, scttrY = [[], []], [[], []]
    contour = [None]
    ax = plt.figure().add_axes((0,0,1,1))
    
    
    #Initializes all tests. Note that after a sort the i's are not nescissarily points used
    #In sum, it instead initializes a random spaced mesh with which contour works easier
    for _ in range(testnum):
        i = getinp(2)
        testsX.append(i[0])
        testsY.append(i[1])
        testsZ.append(0)
    
    testsX += [-1,-1, 1, 1]
    testsY += [-1, 1,-1, 1]
    testsZ += [ 0, 0, 0, 0]
        
    for i in range(0, 50):
        if testnum > 50:
            ind = random.randint(0, testnum-1)
        else:
            ind = i
            
        x, y = testsX[ind], testsY[ind]
        conc = f([x, y])[0]
        scttrX[conc].append(x)
        scttrY[conc].append(y)
    
    #Update the currently shown plot of the neural state
    def updatePlot():
        if contour[0]:
            for tp in contour[0].collections:
                tp.remove()
                
        contour[0] = ax.tricontourf(testsX, testsY, testsZ, 14, cmap="RdBu_r")
        
        ax.scatter(scttrX[0], scttrY[0], c='r', linewidths = .01)
        ax.scatter(scttrX[1], scttrY[1], c='b', linewidths = .01)
        
        plt.draw()
        plt.show(block = False)
        plt.pause(.1)
    
    #Function to test whether the nn has solved the function
    def solved():
        
        done = True

        for i in range(len(testsX)):
            x, y = testsX[i], testsY[i]
            conc = nn.feedforward([x, y])[0]
            testsZ[i] = conc
            done &= (1 if testsZ[i]>.5 else 0) == f([x, y])[0]
                
        updatePlot()
        return done
    

    rounds = 0
    while not solved(): 
        rounds += 1
        nn.adapt(iters, [(inp, f(inp)) for inp in [getinp(size[0]) for _ in range(WKSTS)]])
    
    print(rounds*iters, 'Generations to solve')
    input("Press Enter to continue")
        
        
if __name__ == "__main__":
    
    #pr = cProfile.Profile()
    
    #pr.enable()
    test()
    #pr.disable()
    
    #ps = pstats.Stats(pr).sort_stats('time')
    #ps.print_stats()