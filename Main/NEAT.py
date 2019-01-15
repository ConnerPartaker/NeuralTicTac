'''
Created on Oct 22, 2018

@author: Conner Partaker
Reference biologist: Seth Belcher


test
    find all fitnesses
        regular fitness for genomes
            must be calculated according to task
        adjusted fitness for genomes
            f' = f/N  N=num genomes in species (not really but good estimate)
        fitness for species
            fs = sum(f')
trim
    based on fitness, SOTF trim N lowest genomes per species, based proportionally on fitness
        N = a/(fs)
reproduce
    Do the reproduction thing, breed together what's left of the fittest
mutate new offspring
    mutate edit connections
        perturb weights, enable/disable connections
    mutate add connection
        connect previously unconnected nodes
        assign innovation number (look through gene pool for repeat)
        if new add to gene pool
    mutate add node
        disable connection, add node between, connect
        first gets weight 1, second gets old weight
        Again, search gene pool for these genes, assign new, etc.
speciate
    Loop genomes. If S from ancestor < S_t, add.
        S = <E/N, D/N, Wave>*c N #genes from larger (can be 1 if N<20)
'''
import math, random
from functools import total_ordering

INPUTS = 2
OUTPUTS = 1


def sigmoid(x): return 2/(1 + math.exp(-5*x)) - 1


      
class Gene:
    
    def __init__(self, ends, weight = None, enabled = True, innov = None):
        self.ends  = ends
        self.weight= weight if weight else random.random()*4-2
        self.on    = enabled
        self.innov = innov
    
    def __eq__(self, other):
        return self.ends == other.ends
    
    def __repr__(self):
        ends = self.ends if type(self.ends[0]) is int else (self.ends[0].num, self.ends[1].num)
        return 'gene {} from {} to {} weight {}, en:{}'.format(self.innov, ends[0], ends[1], self.weight, self.on)
    
    def deepCopy(self):
        ends = self.ends if type(self.ends[0]) is int else (self.ends[0].num, self.ends[1].num)  
        return Gene(ends, self.weight, self.on, self.innov)



@total_ordering
class Node:
    
    def __init__(self, num, deg):
        self.num  = num
        self.deg  = deg
        self.into = []
        self.val  = 0.0
    
    def __eq__(self, other):
        return self.num == other.num
    
    def __lt__(self, other):
        return (self.deg, self.num) < (other.deg, other.num)
    
    def __repr__(self):
        return 'node {} of deg {}'.format(self.num, self.deg)
    
    def deepCopy(self):
        return Node(self.num, self.deg)
         
         
         
BASE = [Node(i, 0 if i <= INPUTS else math.inf) for i in range(OUTPUTS + INPUTS + 1)] 
   
class Genome:
    
    def __init__(self, CG = [], nodes = BASE):
        self.CG    = [g.deepCopy() for g in CG]
        self.nodes = [n.deepCopy() for n in nodes]
        
        for g in self.CG: self.set(g)
        
        self.fitness  = 0
        self.afitness = 0
    
    def string(self):
        out = 'Genome of'
        for i in self.nodes + self.CG:
            out += '\n\t' + str(i)
            
        return out
    
    def deepCopy(self):
        return Genome(self.CG, self.nodes)
    
    
    def set(self, g):
        
        ends = [None, None]
        
        for n in self.nodes:
            
            if type(ends[0]) == type(None) and g.ends[0] == n.num:
                ends[0] = n
            if type(ends[1]) == type(None) and g.ends[1] == n.num:
                ends[1] = n
                n.into.append(g)
            if type(ends[0]) != type(None) and type(ends[1]) != type(None): break
        
        g.ends = tuple(ends)
        
        
    def setFit(self, fitness, num):
        
        self.fitness = fitness
        self.afitness= fitness/num
        
        return self.afitness
    
    
    def eval(self, inputs):
        
        self.nodes.sort()
        
        #Assign input values to sensor nodes, and evaluate the network
        for i in zip(self.nodes, [1] + inputs): i[0].val = i[1]
        
        for n in self.nodes[len(inputs) + 1:]:
            n.val = sum([g.weight*g.on*g.ends[0].val for g in n.into])
        
        return [n.val > 0 for n in self.nodes[-OUTPUTS:]]
    


class Species:
    
    def __init__(self, parent = None, init = None):
        self.genomes  = [parent] if not init else [Genome() for _ in range(init)]
        self.fitness  = 0
        self.maxfit   = 0
        self.stag     = 0
    
    def string(self):
        out = 'Species of'
        for idx, i in enumerate(self.genomes):
            out += '\n{}: {}'.format(idx + 1, i.string())
        out = out.replace('\n', '\n\t')

        return out
    
        
    def calcFit(self, fitness):
        #Loop through all genomes, pass fitness, and sum up their afitness in self.fitness
        for f, g in zip(fitness, self.genomes): 
            
            g.setFit(f, len(self.genomes))
        
        #Push the species fitness, and see if it's the top fitness recorded
        self.fitness = sum(fitness)/len(self.genomes)
        
        if self.fitness > self.maxfit:
                self.maxfit = self.fitness
                self.stag = -1
                
        #Update stagnancy. If stag should be 0, it will already be -1, making this still true
        self.stag += 1
            
               
        
class Genus:
    
    def __init__(self, SP = 300, 
                       TA = .4, 
                       ST = 2, 
                       MUT = (.03, .05, .02, .04), 
                       CVR = .75, 
                       C = (1, 1, .4), 
                       W = (.9, .25, .1), 
                       IO = .75, 
                       IS = .02, 
                       SG = 15):
        
        self.POPULATION = SP
        self.TRIMALPHA = TA
        self.STHRESH = ST
        self.MUT = MUT
        self.CROSSOVER = CVR
        self.C = C
        self.W = W
        self.INHERITON = IO
        self.INTERSPECIES = IS
        self.STAG = SG
        
        self.genes   = []
        self.gen     = 0
        self.species = [Species(init = SP)]
        self.innov   = 0
        self.maxfit  = 0
    
    def string(self):
        out = 'Genus of'
        for idx, i in enumerate(self.species):
            out += '\n{}: {}'.format(idx + 1, i.string())
        out = out.replace('\n', '\n\t')

        return out
    
    
    
    def addGene(self, g):

        for gene in self.genes:
            if g == gene:
                g.innov = gene.innov
                break
        else:
            self.innov += 1
            g.innov = self.innov
            self.genes.append(g.deepCopy())
            


    def delta(self, p1, p2):
        #delta W, disjoint, excess, max size
        out, com = [0, 0, 0, max(len(p1.CG), len(p2.CG))], 0
        ptr1, ptr2 = 0, 0
        
        p1.CG.sort(key = lambda g : g.innov)
        p2.CG.sort(key = lambda g : g.innov)
        
        while(ptr1 < len(p1.CG) and ptr2 < len(p2.CG)):
            
            g1, g2 = p1.CG[ptr1], p2.CG[ptr2]
            
            if g1.innov <= g2.innov: ptr1 += 1 
            if g1.innov >= g2.innov: ptr2 += 1
            if g1.innov == g2.innov:
                out[0] += abs(g1.weight - g1.weight)
                com += 1
            else:
                out[1] += 1
        
        if com != 0: out[0] /= com
        out[2]  = max(len(p1.CG) - ptr1, len(p2.CG) - ptr2)
        
        return (self.C[0]*out[1] + self.C[1]*out[2])*1.0/max(out[3] - 20, 1) + self.C[2]*out[0]
        
    
    
    def mutate(self, genome):
        
        #Mutate add gene
        if random.random() < self.MUT[0]:
            
            #Choose end nodes. Only stipulation is the out node can't be an input. All else fair game
            n1, n2 = random.sample(genome.nodes, 2)
            while n1.deg == math.inf: n1 = random.choice(genome.nodes)
            while n2.deg == 0       : n2 = random.choice(genome.nodes)
            
            #Create the gene. Again, anything's fair game, including backwards and recurrent genes
            g = Gene((n1.num, n2.num))
            self.addGene(g)
            
            #As long as the gene isn't already in the genome by happenstance, add it to the genome
            if g.innov not in [gene.innov for gene in genome.CG]:
                genome.CG.append(g)
                genome.set(g)
        
        #Mutate add node
        if len(genome.CG) != 0 and random.random() < self.MUT[1]:
            
            #Choose gene to split, get the new node number too. Deactivate gene
            g, num = random.choice(genome.CG), len(genome.nodes)
            g.on = False
            
            #Create new genes. First should have weight 1, second should have old weight
            #New genes should be in old genes direction. New node should have degree 1 more than the least around it
            g1, g2 = Gene((g.ends[0].num, num), weight = 1), Gene((g.ends[1].num, num), weight = g.weight)
            n = Node(num, min([n.deg for n in g.ends]) + 1)
            
            #Give the innovation nums, add node and genes to genome (cannot be repeats because new node num)
            self.addGene(g1)
            self.addGene(g2)
            
            genome.CG.append(g1)
            genome.CG.append(g2)
            genome.nodes.append(n)
            genome.set(g1)
            genome.set(g2)
        
        #Mutate enable/disable connection
        if len(genome.CG) != 0:
            
            #Chance to turn on a randomly selected gene (given it's off)
            g = random.choice(genome.CG)
            if not g.on and random.random() < self.MUT[2]:
                g.on = True
                
            #Chance to turn off a randomly selected gene (given it's on)
            g = random.choice(genome.CG)
            if g.on and random.random() < self.MUT[3]:
                g.on = False
        
        #Mutate perturb weights
        if random.random() < self.W[0]:
            #For each gene, chance to select a complete new value. Otherwise do a tiny move based on W[2]
            for g in genome.CG:
                if random.random() < self.W[1]:
                    g.weight = random.random()*4-2
                else:
                    g.weight += (random.random()*2-1) * self.W[2]
            
        return genome
        
        
        
    def crossover(self, p1, p2):
        #Make sure parents are in the correct order. Lower fitness disjoint genes are discarded anyway, so just clone p1
        if p1.fitness < p2.fitness: p1, p2 = p2, p1
        
        child = p1.deepCopy()
        
        #Loop over genes. If the gene is in p2, check enablance, and randomly choose which weight to take on
        for g in child.CG:
            if g in p2.CG:
                g2 = p2.CG[p2.CG.index(g)]
                if g.on ^ g2.on:
                    g.on = random.random() < self.INHERITON
                if g2.on and random.random() < .5:
                    g.weight = p2.CG[p2.CG.index(g)].weight
        
        return self.mutate(child)

        
                    
    def reproduce(self):
        
        nextGen = []
        nextmax = self.maxfit
        
        #remove any stagnant species, unless it has the maximum fitness of the population
        for s in self.species[:]:
            if s.maxfit >= self.maxfit:
                nextmax = max(s.maxfit, nextmax)
            elif s.stag >= self.STAG:
                self.species.remove(s)
        
        self.maxfit = nextmax
        totFit = sum([s.fitness for s in self.species])
        
        #remove any weak species that will have 0 children. Otherwise cull the bottom alpha%
        for s in self.species[:]:
            if self.POPULATION * s.fitness < totFit:
                self.species.remove(s)
            else:
                s.genomes.sort(key = lambda g : g.fitness)
                s.genomes[:int(self.TRIMALPHA * len(s.genomes))] = []
                
                
        totFit = sum([s.fitness for s in self.species])
            
        #reproduce from the surviving species
        for s in self.species:
            
            #Get breed num, but make n-1 offspring; the last one is the untouched top genome of the species
            num = int(self.POPULATION * s.fitness / totFit)
            
            for _ in range(num - 1):
                
                if random.random() < self.CROSSOVER:
                    #Choose a second species, then choose parents. Parents may be the same, since species may have 1 genome
                    s2 = s
                    if random.random() < self.INTERSPECIES:
                        s2 = random.choice(self.species)
            
                    p1 = random.choice(s.genomes)
                    p2 = random.choice(s2.genomes)
                    
                    nextGen.append(self.crossover(p1, p2))
                
                else:
                    #Choose a random genome to mutate and send on to the next gen
                    nextGen.append(self.mutate(random.choice(s.genomes)))
                    
        
        #Fill in the rest with random babies
        for i in range(self.POPULATION - len(nextGen)):
            nextGen.append(self.mutate(random.choice(random.choice(self.species).genomes)))
            
        #Clean out the species, leaving only the top genome of the population
        for s in self.species: s.genomes = [s.genomes[0]]
        return nextGen

    
    
    def nextGen(self, fitness):

        #Trickle down fitnesses of everything, remove a species if it's stagnant unless it's got record fitness
        for f, s in zip(fitness, self.species[:]): 
            s.calcFit(f)
        
        #Speciate and add children
        for genome in self.reproduce():
            for species in self.species:
                if self.delta(species.genomes[0], genome) < self.STHRESH:
                    species.genomes.append(genome)
                    break
            #If no matching species is found, create one around this genome
            else:
                self.species.append(Species(genome))
        
        #Finally, any old species now uninhabited should be removed
        for s in self.species[:]:
            if len(s.genomes) == 0:
                self.species.remove(s)
                
                
                
                
                