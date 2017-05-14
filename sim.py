#!/usr/bin/python3

import os
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


############
# MAIN
############

# File System Setup
mainpath= "./newtesting/"
subfolder = "funny/"
fullpath = mainpath+subfolder
os.makedirs(fullpath, exist_ok=True)


# Basic Parameters

windowLength = 2016
simLength = windowLength*40
nLic = 230
factor = 2
cap = int(windowLength/nLic) * factor


# Make Arrays for hashPower
hashPower = np.empty(nLic)

# Define Function for hash power distribution
def myFunny(x):
    #return np.exp(-x/48)*(1+0.5*np.cos(100/(x+1)))
    return np.exp(-x/48)

# Generate HashPower distribution
for i in range(0, nLic):
    hashPower[i] = myFunny(i)
hashPower = np.sort(hashPower) # from small to large
cumulPower = np.cumsum(hashPower)
hashPower = hashPower / cumulPower[-1] # normalize hashingPower
cumulPower = cumulPower / cumulPower[-1] # to total HashPower = 1.0

# This cumulPower array gives every Miner a certain interval inside of [0,1]
# that 'belongs' to that miner in the sense that if we generate a random
# number in [0,1), it will fall into that interval with a probability 
# equal to that miner's contribution to the total hashrate. 

# Plot Distribution for reference
plt.bar(range(1, nLic+1),hashPower)
#plt.axis([0, nLic+1, 0, 2])
plt.savefig(mainpath + subfolder + "figure_0.png", dpi=600)
plt.clf()

# Helper function
def FirstLarger(l, n):
    i = 0
    while l[i]<=n:
        i+=1
    return i
    
# Algorithms for Adding Coins.
# First basic Cap
def AddCoin_BasicCap(i, coinsinwindow, totalcoins, blocktimes, whoMinedBlock, difficulty):
    # Reset the Cap at start of windowLength
    if i%windowLength==0:
        coinsinwindow.fill(0)
        if i == 0:
            difficulty[0] = 1.0
        else:
            difficulty[0] *= np.average(blocktimes[(i-windowLength):i])
        # add difficulty adjustment using average of blocktime in last window
    
    # Get remaining network hash power
    remainingPower = 0;
    for j in range(0,nLic):
        if coinsinwindow[j] < cap: # all Miner's that reached Cap drop out
            remainingPower+=hashPower[j]
            
    # Blocktime rises proportional to 1/rem. hash rate (half the network hashrate -> double the blocktime)
    blocktimes[i] = 1.0/remainingPower / difficulty[0]
    
    # pick Miner with probability equal to his contribution to network hash rate
    ran = np.random.random()
    pos = FirstLarger(cumulPower,ran)
    
    # if that Miner already has reached cap, just choose another one.
    # (blocktime rise is already accounted for)
    while coinsinwindow[pos]>=cap:
        ran = np.random.random()
        pos = FirstLarger(cumulPower,ran)
        
    # Give that miner the coin
    totalcoins[pos] += 1
    coinsinwindow[pos] +=1
    # note who got the coin
    whoMinedBlock[i] = pos;
        
# now floating Cap    
def AddCoin_FloatingCap(i, coinsinwindow, totalcoins, blocktimes, whoMinedBlock, difficulty):
    # Determine from where to start counting coins
    if i <= windowLength:
        start = 0
    else:
        start = i - windowLength
    
    if i%windowLength==0:
        if i == 0:
            difficulty[0] = 1.0
        else:
            difficulty[0] *= np.average(blocktimes[i-windowLength:i])
        
    # # Count coins for each miner
    # coinsinwindow.fill(0)
    # for k in range(start, i):
    #     miner = int(whoMinedBlock[k])
    #     coinsinwindow[miner] += 1
    
    # Get remaining network hash power
    remainingPower = 0;
    for j in range(0,nLic):
        if coinsinwindow[j] < cap: # all Miner's that reached Cap drop out
            remainingPower+=hashPower[j]
    
    # Blocktime rises proportional to 1/rem. hash rate (half the network hashrate -> double the blocktime)
    blocktimes[i] = 1.0/remainingPower / difficulty[0]
    
    # pick Miner with probability equal to his contribution to network hash rate
    ran = np.random.random()
    pos = FirstLarger(cumulPower,ran)
    
    # if that Miner already has reached cap, just choose another one.
    # (blocktime rise is already accounted for)
    while coinsinwindow[pos]>=cap:
        ran = np.random.random()
        pos = FirstLarger(cumulPower,ran)
        
    # Give that miner the coin
    totalcoins[pos] += 1
    
    coinsinwindow[pos] += 1
    
    # note who got the coin
    whoMinedBlock[i] = pos;
    
    # move window along
    if i>=windowLength:
        miner = int(whoMinedBlock[i-windowLength])
        coinsinwindow[miner] -= 1
    
# now Bucket System 
def AddCoin_Buckets(i, buckets, totalcoins, blocktimes, whoMinedBlock, difficulty):
    # Determine from where to start counting coins
    
    if i%windowLength==0:
        if i == 0:
            difficulty[0] = 1.0
        else:
            difficulty[0] *= np.average(blocktimes[i-windowLength:i])
        
    # Get remaining network hash power
    remainingPower = 0;
    for j in range(0,nLic):
        if buckets[j] > 0: # all Miner's that reached Cap drop out
            remainingPower+=hashPower[j]
    
    # Blocktime rises proportional to 1/rem. hash rate (half the network hashrate -> double the blocktime)
    blocktimes[i] = 1.0/remainingPower / difficulty[0]
    
    # pick Miner with probability equal to his contribution to network hash rate
    ran = np.random.random()
    pos = FirstLarger(cumulPower,ran)
    
    # if that Miner already has reached cap, just choose another one.
    # (blocktime rise is already accounted for)
    while buckets[pos]<=0:
        ran = np.random.random()
        pos = FirstLarger(cumulPower,ran)
        
    # Give that miner the coin
    totalcoins[pos] += 1
    
    buckets[pos] -= nLic
    
    # note who got the coin
    whoMinedBlock[i] = pos;
    
    # refill all buckets 
    buckets += 1



# Now start simulating

# First Basic Cap
# Make Arrays for Miners
coinsinwindow = np.zeros(nLic)
totalcoins = np.zeros(nLic)

# Make Arrays for blockchain
blocktimes = np.empty(simLength)
whoMinedBlock = np.empty(simLength)

difficulty=np.array([1.0],dtype=float)
for i in range(0,simLength):
    if i%10000==0:
        print(i)
    AddCoin_BasicCap(i, coinsinwindow, totalcoins, blocktimes, whoMinedBlock, difficulty)


# Plot results
# Coins in last window
plt.bar(np.arange(1, nLic+1, 1), coinsinwindow)
#plt.axis([0, nLicenses+1, 0, cap + 2])
plt.savefig(fullpath+"plot_coinsinwindow_basic.png", dpi=600)
plt.yscale('log')
plt.savefig(fullpath+"logplot_coinsinwindow_basic.png", dpi=600)
plt.clf()
plt.yscale('linear')

# Total coins 
plt.bar(np.arange(1, nLic+1, 1), totalcoins)
#plt.axis([0, nLicenses+1, 0, capRange])
plt.savefig(fullpath+"plot_totalcoins_basic.png", dpi=600)
plt.yscale('log')
plt.savefig(fullpath+"logplot_totalcoins_basic.png", dpi=600)
plt.clf()
plt.yscale('linear')

# Blocktimes
plt.plot(np.arange(1, simLength+1, 1),blocktimes)
#plt.axis([0, simLength, 0, bTime1Range])
plt.savefig(fullpath+"plot_blocktimes_basic.png", dpi=600)
plt.yscale('log')
plt.savefig(fullpath+"logplot_blocktimes_basic.png", dpi=600)
plt.clf()
plt.yscale('linear')




# Now Floating Cap
# Make Arrays for Miners
coinsinwindow = np.zeros(nLic)
totalcoins = np.zeros(nLic)

# Make Arrays for blockchain
blocktimes = np.empty(simLength, dtype=float)
whoMinedBlock = np.zeros(simLength)

difficulty=np.array([1.0],dtype=float)
for i in range(0,simLength):
    if i%10000==0:
        print(i)
    AddCoin_FloatingCap(i, coinsinwindow, totalcoins, blocktimes, whoMinedBlock, difficulty)

# Plot results
# Coins in last window
plt.bar(np.arange(1, nLic+1, 1), coinsinwindow)
#plt.axis([0, nLicenses+1, 0, cap + 2])
plt.savefig(fullpath+"plot_coinsinwindow_floating.png", dpi=600)
plt.yscale('log')
plt.savefig(fullpath+"logplot_coinsinwindow_floating.png", dpi=600)
plt.clf()
plt.yscale('linear')

# Total coins 
plt.bar(np.arange(1, nLic+1, 1), totalcoins)
#plt.axis([0, nLicenses+1, 0, capRange])
plt.savefig(fullpath+"plot_totalcoins_floating.png", dpi=600)
plt.yscale('log')
plt.savefig(fullpath+"logplot_totalcoins_floating.png", dpi=600)
plt.clf()
plt.yscale('linear')

# Blocktimes
plt.plot(np.arange(1, simLength+1, 1),blocktimes)
#plt.axis([0, simLength, 0, bTime1Range])
plt.savefig(fullpath+"plot_blocktimes_floating.png", dpi=600)
plt.yscale('log')
plt.savefig(fullpath+"logplot_blocktimes_floating.png", dpi=600)
plt.clf()
plt.yscale('linear')


# Now Bucket System
# Make Arrays for Miners
buckets = np.zeros(nLic)
buckets += 1
totalcoins = np.zeros(nLic)

# Make Arrays for blockchain
blocktimes = np.empty(simLength, dtype=float)
whoMinedBlock = np.zeros(simLength)

difficulty=np.array([1.0],dtype=float)
for i in range(0,simLength):
    if i%10000==0:
        print(i)
    AddCoin_FloatingCap(i, buckets, totalcoins, blocktimes, whoMinedBlock, difficulty)

# Plot results
# Coins in last window
plt.bar(np.arange(1, nLic+1, 1), buckets)
#plt.axis([0, nLicenses+1, 0, cap + 2])
plt.savefig(fullpath+"plot_coinsinwindow_buckets.png", dpi=600)
plt.yscale('log')
plt.savefig(fullpath+"logplot_coinsinwindow_buckets.png", dpi=600)
plt.clf()
plt.yscale('linear')

# Total coins 
plt.bar(np.arange(1, nLic+1, 1), totalcoins)
#plt.axis([0, nLicenses+1, 0, capRange])
plt.savefig(fullpath+"plot_totalcoins_buckets.png", dpi=600)
plt.yscale('log')
plt.savefig(fullpath+"logplot_totalcoins_buckets.png", dpi=600)
plt.clf()
plt.yscale('linear')

# Blocktimes
plt.plot(np.arange(1, simLength+1, 1),blocktimes)
#plt.axis([0, simLength, 0, bTime1Range])
plt.savefig(fullpath+"plot_blocktimes_buckets.png", dpi=600)
plt.yscale('log')
plt.savefig(fullpath+"logplot_blocktimes_buckets.png", dpi=600)
plt.clf()
plt.yscale('linear')
