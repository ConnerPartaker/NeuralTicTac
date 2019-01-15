import cProfile, pstats, random
import numpy as np
import Main.NumpyNN as NN



SIZE = [2, 3, 1]
ITER = 1
def func(array): return [array[0] ^ array[1]]



def getinp(x):
    return np.array([random.choice([0, 1]) for _ in range(x)])

def test(size, iters, func):
    
    nn = NN.NeuralNet(size)
    
    tests = []
    for _ in range(100):
        inp = getinp(size[0])
        tests.append((inp, func(inp)))
    
    #Function to test whether the nn has solved the function
    def solved():
        num = 0
        for i in tests:
            conc = list(map(round, nn.feedforward(i[0])))
            if conc == i[1]:
                num += 1
        
        pct = round(num*1.0/len(tests), 3)
        print (pct, ' completed')
        return(True if pct > .95 else False)

    rounds = 0
    while not solved(): 
        rounds += 1
        nn.adapt(iters, [(inp, func(inp)) for inp in [getinp(size[0]) for _ in range(2000)]])
    
    print(rounds*iters, 'Generations to solve')
    


if __name__ == "__main__":
    
    #pr = cProfile.Profile()
    
    #pr.enable()
    test(SIZE, ITER, func)
    #pr.disable()
    
    #ps = pstats.Stats(pr).sort_stats('time')
    #ps.print_stats()