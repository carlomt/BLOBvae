from __future__ import division, print_function
import ROOT
import numpy as np
import root_numpy

inputfile = ROOT.TFile("out.root","READ")

nTrainingSet = 900
nTestSet = 100

tree = f.Get("tree")

x_extra_training = np.zeros(nTrainingSet,15)
x_extra_test = np.zeros(nTestSet,15)
for i, event in enumerate(tree):
    j = 0
    if i < nTrainingSet: 
        x_extra = x_extra_training
    else:
        x_extra = x_extra_test

    x_extra[i,j] = tree.evt
    j +=1
    x_extra[i,j] = tree.b
    j +=1
    x_extra[i,j] = tree.nfrg
    j +=1
    for ifrag in range(0,tree.nfrg):
        x_extra[i,j] = tree.Af[ifrag]
        j +=1
        x_extra[i,j] = tree.zf[ifrag]
        j +=1
        x_extra[i,j] = tree.Pzf[ifrag]
        j +=1
        x_extra[i,j] = tree.Eecc[ifrag]
        j +=1

np.save("x_extra_test.npy",x_extra_test)
np.save("x_extra_training.npy",x_extra_training)


x_train = np.zeros([nTrainingSet,128,128,128,2])
x_test  = np.zeros([nTestSet,128,128,128,2])

for i in range(0,nTrainingSet+nTestSet):
    hX = inputfile.Get("hX1;"+str(i+1))
    hP = inputfile.Get("hP1;"+str(i+1))
    x = root_numpy.hist2array(hX)
    p = root_numpy.hist2array(hP)
    x /= x.max()
    p /= p.max()
    if i < nTrainingSet:
        x_train[i,:,:,:,0] = x
        x_train[i,:,:,:,1] = p
    else:
        x_test[i-nTrainingSet,:,:,:,0] = x
        x_test[i-nTrainingSet,:,:,:,1] = p
inputfile.Close()


