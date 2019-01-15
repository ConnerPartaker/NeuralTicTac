'''
Created on Oct 26, 2018

@author: Conne
'''
import math, random
import Main.NEAT as NE


def func(z):
    return z[0] ^ z[1]

jimbert = NE.Genus(SP = 300, 
                   ST = 2.5, 
                   MUT = (.004, .002, .002, .004),
                   IS = .025, 
                   SG = 15)
gen = 0
while True:
    
    gen += 1
    fArray = []
    
    for species in jimbert.species:
        
        fArray.append([])
        for genome in species.genomes:
            
            fitness = 1
            
            for i in range(4):
                ipt = [i//2, i%2]
                if genome.eval(ipt) == [func(ipt)]:
                    fitness += 1
         
            fArray[-1].append(fitness)
            if fitness >= 5:
                print(str(gen) + ' generations to solve by')
                print(genome.string())
                exit(0)
    
    if gen % 5 == 0:
        if gen < 750:
            print("{} generations, at {}, {}".format(gen, jimbert.maxfit, sorted([int(s.fitness) for s in jimbert.species], reverse=True)))
        else:
            print("750+ generations")
            exit(0)
                    
    jimbert.nextGen(fArray)
    
    
    
    
'''
---The Fridge---
QUICKEST IN XOR SOLVING
66 generations to solve by
Genome of
    node 0 of deg 0
    node 1 of deg 0
    node 2 of deg 0
    node 4 of deg 1
    node 5 of deg 1
    node 3 of deg inf
    gene 3 from 0 to 3 weight -0.17835329309491224, en:True
    gene 5 from 3 to 4 weight 1.9493796939358168, en:True
    gene 6 from 1 to 4 weight -1.5242216039913565, en:True
    gene 7 from 3 to 5 weight -0.1685195616489336, en:True
    gene 9 from 4 to 3 weight -1.1389852590114233, en:True
    gene 10 from 2 to 5 weight 0.5297728629370875, en:True

SMALLEST IN XOR SOLVING
179 generations to solve by
Genome of
    node 0 of deg 0
    node 1 of deg 0
    node 2 of deg 0
    node 4 of deg 1
    node 3 of deg inf
    gene 1 from 1 to 3 weight 1.156288949646051, en:True
    gene 2 from 0 to 3 weight 1.3907615801319992, en:True
    gene 3 from 2 to 3 weight -0.7325421222957251, en:True
    gene 4 from 0 to 4 weight -1.796024013619067, en:True
    gene 5 from 3 to 4 weight -1.7150024378598052, en:True
    gene 7 from 4 to 3 weight 1.1977063703432267, en:True
    gene 8 from 1 to 4 weight 0.9005407508351251, en:True
'''