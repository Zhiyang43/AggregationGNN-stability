# Zhiyang Wang, zhiyangw@seas.upenn.edu
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu

# Test the movie recommendation dataset on several architectures.

# When it runs, it produces the following output:
#   - It trains the specified models and saves the best and the last model
#       parameters of each realization on a directory named 'savedModels'.
#   - It saves a pickle file with the torch random state and the numpy random
#       state for reproducibility.
#   - It saves a text file 'hyperparameters.txt' containing the specific
#       (hyper)parameters that control the run, together with the main (scalar)
#       results obtained.
#   - If desired, logs in tensorboardX the training loss and evaluation measure
#       both of the training set and the validation set. These tensorboardX logs
#       are saved in a logsTB directory.
#   - If desired, saves the vector variables of each realization (training and
#       validation loss and evaluation measure, respectively); this is saved
#       in pickle format. These variables are saved in a trainVars directory.
#   - If desired, plots the training and validation loss and evaluation
#       performance for each of the models, together with the training loss and
#       validation evaluation performance for all models. The summarizing
#       variables used to construct the plots are also saved in pickle format. 
#       These plots (and variables) are in a figs directory.

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:
import os
import numpy as np
import matplotlib
from torch.functional import norm
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import Utils.graphTools as graphTools
import Utils.dataTools
import Utils.graphML as gml
import Modules.architectures as archit
import Modules.model as model
import Modules.train as train
import Modules.loss as loss

#\\\ Separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

graphType = 'user' # Graph type: 'user'-based or 'movie'-based
labelID = [1] # Which node to focus on (either a list or the str 'all')
# When 'movie': [1]: Toy Story, [50]: Star Wars, [258]: Contact,
# [100]: Fargo, [180]: Return of the Jedi, [294]: Liar, liar
if labelID == 'all':
    labelIDstr = 'all'
elif len(labelID) == 1:
    labelIDstr = '%03d' % labelID[0]
else:
    labelIDstr = ['%03d_' % i for i in labelID]
    labelIDstr = "".join(labelIDstr)
    labelIDstr = labelIDstr[0:-1]
thisFilename = 'movieGNN' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run
dataDir = os.path.join('datasets','movielens')

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + graphType + '-' + labelIDstr + '-' + today
# Create directory 
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Create the file where all the (hyper)parameters and results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#\\\ Save seeds for reproducibility
#   PyTorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
randomStates.append({})
randomStates[1]['module'] = 'torch'
randomStates[1]['state'] = torchState
randomStates[1]['seed'] = torchSeed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)

########
# DATA #
########

useGPU = False # If true, and GPU is available, use it.
ratioTrain = 0.9 # Ratio of training samples
ratioValid = 0.1 # Ratio of validation samples (out of the total training
# samples)
# Final split is:
#   nValidation = round(ratioValid * ratioTrain * nTotal)
#   nTrain = round((1 - ratioValid) * ratioTrain * nTotal)
#   nTest = nTotal - nTrain - nValidation
nPerturbationRealizations = 10

nDataSplits = 10 # Number of data realizations
# Obs.: The built graph depends on the split between training, validation and
# testing. Therefore, we will run several of these splits and average across
# them, to obtain some result that is more robust to this split.
    
# Given that we build the graph from a training split selected at random, it
# could happen that it is disconnected, or directed, or what not. In other 
# words, we might want to force (by removing nodes) some useful characteristics
# on the graph
keepIsolatedNodes = False # If True keeps isolated nodes
forceUndirected = True # If True forces the graph to be undirected
forceConnected = True # If True returns the largest connected component of the
    # graph as the main graph
kNN = 50 # Number of nearest neighbors
multiplier = 1
maxDataPoints = None # None to consider all data points
nPerturbationSize = 10 # Number of perturbation sizes
minPerturbationSize = 1e-3 # Minimimum perturbation size
maxPerturbationSize = 0.5 # Maximum perturbation size
epsilon = np.logspace(np.log10(minPerturbationSize),
                      np.log10(maxPerturbationSize),
                      num = nPerturbationSize)


#\\\ Save values:
writeVarValues(varsFile,
               {'labelID': labelID,
                'graphType': graphType,
                'ratioTrain': ratioTrain,
                'ratioValid': ratioValid,
                'nDataSplits': nDataSplits,
                'keepIsolatedNodes': keepIsolatedNodes,
                'forceUndirected': forceUndirected,
                'forceConnected': forceConnected,
                'kNN': kNN,
                'maxDataPoints': maxDataPoints,
                'useGPU': useGPU,
                'multiplier': multiplier})

############
# TRAINING #
############

#\\\ Individual model training options
trainer = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.001 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.SmoothL1Loss
minRatings = 0 # Discard samples (rows and columns) with less than minRatings 
    # ratings
interpolateRatings = False # Interpolate ratings with nearest-neighbors rule
    # before feeding them into the GNN

#\\\ Overall training options
nEpochs = 40 # Number of epochs
batchSize = 10 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 5 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'trainer': trainer,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

#################
# ARCHITECTURES #
#################

# Just four architecture one- and two-layered Selection and Local GNN. The main
# difference is that the Local GNN is entirely local (i.e. the output is given
# by a linear combination of the features at a single node, instead of a final
# MLP layer combining the features at all nodes).
    
# Select desired architectures
doGraphonGNN = False
doGraphonGNNRegularSampling = False
doGraphonGNNRegularIntegration = False
doGraphonGNNIrregularSampling = False
doGraphonGNNIrregularIntegration = False
integration = False
irregular = False
normalize = True
# num_selectednodes = [472, 236]
doSelectionGNN = False
doLocalGNN = False
doAggregationGNN = True
doCoarsening = False

do1Layer = False
do2Layers = True

writeVarValues(varsFile,
               {'integration': integration,
               'irregular': irregular,
               'normalize': normalize})

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. Do not forget to add the name of the architecture
# to modelList.

# If the hyperparameter dictionary is called 'hParams' + name, then it can be
# picked up immediately later on, and there's no need to recode anything after
# the section 'Setup' (except for setting the number of nodes in the 'N' 
# variable after it has been coded).

# The name of the keys in the hyperparameter dictionary have to be the same
# as the names of the variables in the architecture call, because they will
# be called by unpacking the dictionary.

modelList = []

#\\\\\\\\\\\\\\\\\\\
#\\\ GRAPHON GNN \\\
#\\\\\\\\\\\\\\\\\\\

if doGraphonGNN:

    hParamsGraphonGNN = {} # Hyperparameters (hParams) for the GNN

    hParamsGraphonGNN['name'] = 'GraphonGNN' # Name of the architecture
    hParamsGraphonGNN['archit'] = archit.GraphonGNN
    
    #\\\ Architecture parameters
    hParamsGraphonGNN['dimNodeSignals'] = [1, 32, 8] # Features per layer
    hParamsGraphonGNN['nFilterTaps'] = [5, 5] # Number of filter taps
    hParamsGraphonGNN['bias'] = True # Decide whether to include a bias term
    hParamsGraphonGNN['nonlinearity'] = nn.ReLU # Selected nonlinearity
    hParamsGraphonGNN['nSelectedNodes'] = num_selectednodes
    hParamsGraphonGNN['dimLayersMLP'] = [1] # Dimension of the fully
                                            # connected layers after the GCN layers
    hParamsGraphonGNN['GSOs'] = None
  


    #\\\ Save Values:
    writeVarValues(varsFile, hParamsGraphonGNN)
    modelList += [hParamsGraphonGNN['name']]

if doGraphonGNNRegularIntegration:

    hParamsGraphonGNNRegularIntegration = {} # Hyperparameters (hParams) for the GNN

    hParamsGraphonGNNRegularIntegration['name'] = 'GraphonGNNRegularIntegration' # Name of the architecture
    hParamsGraphonGNNRegularIntegration['archit'] = archit.GraphonGNN
    
    #\\\ Architecture parameters
    hParamsGraphonGNNRegularIntegration['dimNodeSignals'] = [1, 32, 8] # Features per layer
    hParamsGraphonGNNRegularIntegration['nFilterTaps'] = [5, 5] # Number of filter taps
    hParamsGraphonGNNRegularIntegration['bias'] = True # Decide whether to include a bias term
    hParamsGraphonGNNRegularIntegration['nonlinearity'] = nn.ReLU # Selected nonlinearity
    hParamsGraphonGNNRegularIntegration['nSelectedNodes'] = num_selectednodes 
    hParamsGraphonGNNRegularIntegration['dimLayersMLP'] = [1] # Dimension of the fully
                                            # connected layers after the GCN layers
    hParamsGraphonGNNRegularIntegration['GSOs'] = None
  


    #\\\ Save Values:
    writeVarValues(varsFile, hParamsGraphonGNNRegularIntegration)
    modelList += [hParamsGraphonGNNRegularIntegration['name']]

if doGraphonGNNRegularSampling:

    hParamsGraphonGNNRegularSampling = {} # Hyperparameters (hParams) for the GNN

    hParamsGraphonGNNRegularSampling['name'] = 'GraphonGNNRegularSampling' # Name of the architecture
    hParamsGraphonGNNRegularSampling['archit'] = archit.GraphonGNN
    
    #\\\ Architecture parameters
    hParamsGraphonGNNRegularSampling['dimNodeSignals'] = [1, 32, 8] # Features per layer
    hParamsGraphonGNNRegularSampling['nFilterTaps'] = [5, 5] # Number of filter taps
    hParamsGraphonGNNRegularSampling['bias'] = True # Decide whether to include a bias term
    hParamsGraphonGNNRegularSampling['nonlinearity'] = nn.ReLU # Selected nonlinearity
    hParamsGraphonGNNRegularSampling['nSelectedNodes'] = num_selectednodes 
    hParamsGraphonGNNRegularSampling['dimLayersMLP'] = [1] # Dimension of the fully
                                            # connected layers after the GCN layers
    hParamsGraphonGNNRegularSampling['GSOs'] = None
  


    #\\\ Save Values:
    writeVarValues(varsFile, hParamsGraphonGNNRegularSampling)
    modelList += [hParamsGraphonGNNRegularSampling['name']]

if doGraphonGNNIrregularSampling:

    hParamsGraphonGNNIrregularSampling = {} # Hyperparameters (hParams) for the GNN

    hParamsGraphonGNNIrregularSampling['name'] = 'GraphonGNNIrregularSampling' # Name of the architecture
    hParamsGraphonGNNIrregularSampling['archit'] = archit.GraphonGNN
    
    #\\\ Architecture parameters
    hParamsGraphonGNNIrregularSampling['dimNodeSignals'] = [1, 32, 8] # Features per layer
    hParamsGraphonGNNIrregularSampling['nFilterTaps'] = [5, 5] # Number of filter taps
    hParamsGraphonGNNIrregularSampling['bias'] = True # Decide whether to include a bias term
    hParamsGraphonGNNIrregularSampling['nonlinearity'] = nn.ReLU # Selected nonlinearity
    hParamsGraphonGNNIrregularSampling['nSelectedNodes'] = num_selectednodes 
    hParamsGraphonGNNIrregularSampling['dimLayersMLP'] = [1] # Dimension of the fully
                                            # connected layers after the GCN layers
    hParamsGraphonGNNIrregularSampling['GSOs'] = None
  


    #\\\ Save Values:
    writeVarValues(varsFile, hParamsGraphonGNNIrregularSampling)
    modelList += [hParamsGraphonGNNIrregularSampling['name']]

if doGraphonGNNIrregularIntegration:

    hParamsGraphonGNNIrregularIntegration = {} # Hyperparameters (hParams) for the GNN

    hParamsGraphonGNNIrregularIntegration['name'] = 'GraphonGNNIrregularIntegration' # Name of the architecture
    hParamsGraphonGNNIrregularIntegration['archit'] = archit.GraphonGNN
    
    #\\\ Architecture parameters
    hParamsGraphonGNNIrregularIntegration['dimNodeSignals'] = [1, 32, 8] # Features per layer
    hParamsGraphonGNNIrregularIntegration['nFilterTaps'] = [5, 5] # Number of filter taps
    hParamsGraphonGNNIrregularIntegration['bias'] = True # Decide whether to include a bias term
    hParamsGraphonGNNIrregularIntegration['nonlinearity'] = nn.ReLU # Selected nonlinearity
    hParamsGraphonGNNIrregularIntegration['nSelectedNodes'] = num_selectednodes 
    hParamsGraphonGNNIrregularIntegration['dimLayersMLP'] = [1] # Dimension of the fully
                                            # connected layers after the GCN layers
    hParamsGraphonGNNIrregularIntegration['GSOs'] = None
  


    #\\\ Save Values:
    writeVarValues(varsFile, hParamsGraphonGNNIrregularIntegration)
    modelList += [hParamsGraphonGNNIrregularIntegration['name']]



#\\\\\\\\\\\\\\\\\\\\\\\
#\\\ AGGREGATION GNN \\\
#\\\\\\\\\\\\\\\\\\\\\\\

if doAggregationGNN:

    #\\\ Basic parameters for all the Local GNN architectures
    
    hParamsAggGNNWithPel= {} # Hyperparameters (hParams) for the Local GNN (LclGNN)
    
    hParamsAggGNNWithPel['name'] = 'AggGNNWithPel'
    # Chosen architecture
    hParamsAggGNNWithPel['archit'] = archit.AggregationGNN
    
    # Graph convolutional parameters
    hParamsAggGNNWithPel['dimFeatures'] = [1, 32, 8] # Features per layer
    hParamsAggGNNWithPel['nFilterTaps'] = [5, 5] # Number of filter taps per layer
    hParamsAggGNNWithPel['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    hParamsAggGNNWithPel['nonlinearity'] = nn.ReLU # Selected nonlinearity
    # Pooling
    hParamsAggGNNWithPel['poolingFunction'] = torch.nn.MaxPool1d # Summarizing function
    hParamsAggGNNWithPel['poolingSize'] = [1, 1] # poolingSize-hop neighborhood that
    hParamsAggGNNWithPel['nSelectedNodes'] = None
        # is affected by the summary
    # Readout layer: local linear combination of features
    hParamsAggGNNWithPel['dimLayersMLP'] = [1] # Dimension of the fully connected layers
        # after the GCN layers (map); this fully connected layer is applied only
        # at each node, without any further exchanges nor considering all nodes
        # at once, making the architecture entirely local.
    # Graph structure
    hParamsAggGNNWithPel['GSO'] = None # To be determined later on, based on data
    hParamsAggGNNWithPel['order'] = None
    
    #\\\ Save Values:
    # writeVarValues(varsFile, hParamsAggGNNWithPel)
    # modelList += [hParamsAggGNNWithPel['name']]

    #\\\\\\\\\\\\
    #\\\ MODEL: Aggregation GNN with all Layers with no penalty loss
    #\\\\\\\\\\\\


    hParamsAggGNNNoPelFull = deepcopy(hParamsAggGNNWithPel)

    hParamsAggGNNNoPelFull['name'] = 'AggGNNNoPelFull' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsAggGNNNoPelFull)
    modelList += [hParamsAggGNNNoPelFull['name']]


    #    \\\\\\\\\\\\
    # \\\ MODEL: Aggregation GNN with all Layers with no penalty loss
    # \\\\\\\\\\\\


    hParamsAggGNNNoPelN2 = deepcopy(hParamsAggGNNWithPel)

    hParamsAggGNNNoPelN2['name'] = 'AggGNNNoPelN2' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsAggGNNNoPelN2)
    modelList += [hParamsAggGNNNoPelN2['name']]


    #    \\\\\\\\\\\\
    # \\\ MODEL: Aggregation GNN with all Layers with no penalty loss
    # \\\\\\\\\\\\


    hParamsAggGNNNoPelN4 = deepcopy(hParamsAggGNNWithPel)

    hParamsAggGNNNoPelN4['name'] = 'AggGNNNoPelN4' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsAggGNNNoPelN4)
    modelList += [hParamsAggGNNNoPelN4['name']]




    # hParamsAggGNNAggN4 = deepcopy(hParamsAggGNNWithPel)

    # hParamsAggGNNAggN4['name'] = 'AggGNNAggN4' # Name of the architecture
   

    # # #\\\ Save Values:
    # writeVarValues(varsFile, hParamsAggGNNAggN4)
    # modelList += [hParamsAggGNNAggN4['name']]


       #\\\\\\\\\\\\
    #\\\ MODEL: Aggregation GNN with all Layers with no penalty loss
    #\\\\\\\\\\\\


    hParamsAggGNNNoPel5 = deepcopy(hParamsAggGNNWithPel)

    hParamsAggGNNNoPel5['name'] = 'AggGNNNoPel5' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsAggGNNNoPel5)
    modelList += [hParamsAggGNNNoPel5['name']]


      #\\\\\\\\\\\\
    #\\\ MODEL: Aggregation GNN with all Layers with no penalty loss
    #\\\\\\\\\\\\


    # hParamsAggGNNAggN2 = deepcopy(hParamsAggGNNWithPel)

    # hParamsAggGNNAggN2['name'] = 'AggGNNAggN2' # Name of the architecture
   

    # #\\\ Save Values:
    # writeVarValues(varsFile, hParamsAggGNNAggN2)
    # modelList += [hParamsAggGNNAggN2['name']]






    # hParamsAggGNNAgg5 = deepcopy(hParamsAggGNNWithPel)

    # hParamsAggGNNAgg5['name'] = 'AggGNNAgg5' # Name of the architecture
   

    # # #\\\ Save Values:
    # writeVarValues(varsFile, hParamsAggGNNAgg5)
    # modelList += [hParamsAggGNNAgg5['name']]



#\\\\\\\\\\\\\\\\\\\\\
#\\\ SELECTION GNN \\\
#\\\\\\\\\\\\\\\\\\\\\

if doSelectionGNN:

    #\\\ Basic parameters for all the Selection GNN architectures
    
    hParamsSelGNN = {} # Hyperparameters (hParams) for the Selection GNN (SelGNN)
    
    hParamsSelGNN['name'] = 'SelGNN'
    # Chosen architecture
    hParamsSelGNN['archit'] = archit.SelectionGNN
    
    # Graph convolutional parameters
    hParamsSelGNN['dimNodeSignals'] = [1, 32, 8] # Features per layer
    hParamsSelGNN['nFilterTaps'] = [5, 5] # Number of filter taps per layer
    hParamsSelGNN['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    hParamsSelGNN['nonlinearity'] = nn.ReLU # Selected nonlinearity
    # Pooling
    hParamsSelGNN['poolingFunction'] = gml.NoPool # Summarizing function
    hParamsSelGNN['nSelectedNodes'] = None # To be determined later on
    hParamsSelGNN['poolingSize'] = [2, 2] # poolingSize-hop neighborhood that
        # is affected by the summary
    # Full MLP readout layer (this layer breaks the locality of the solution)
    hParamsSelGNN['dimLayersMLP'] = [1] # Dimension of the fully connected
        # layers after the GCN layers, we just need to output a single scalar
    # Graph structure
    hParamsSelGNN['GSO'] = None # To be determined later on, based on data
    hParamsSelGNN['order'] = None

#\\\\\\\\\\\\
#\\\ MODEL 1: Selection GNN with 1 less layer
#\\\\\\\\\\\\

#    hParamsSelGNN1Ly = deepcopy(hParamsSelGNN)
#
#    hParamsSelGNN1Ly['name'] += '1Ly' # Name of the architecture
#    
#    hParamsSelGNN1Ly['dimNodeSignals'] = hParamsSelGNN['dimNodeSignals'][0:-1]
#    hParamsSelGNN1Ly['nFilterTaps'] = hParamsSelGNN['nFilterTaps'][0:-1]
#    hParamsSelGNN1Ly['poolingSize'] = hParamsSelGNN['poolingSize'][0:-1]
#
#    #\\\ Save Values:
#    writeVarValues(varsFile, hParamsSelGNN1Ly)
#    modelList += [hParamsSelGNN1Ly['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 2: Selection GNN with all Layers
#\\\\\\\\\\\\

    hParamsSelGNN2LyWithPel = deepcopy(hParamsSelGNN)

    hParamsSelGNN2LyWithPel['name'] += '2LyWithPel' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsSelGNN2LyWithPel)
    modelList += [hParamsSelGNN2LyWithPel['name']]

#\\\\\\\\\\\\
#\\\ MODEL 3: Selection GNN with all Layers with no penalty loss
#\\\\\\\\\\\\


    hParamsSelGNN2LyNoPel = deepcopy(hParamsSelGNN)

    hParamsSelGNN2LyNoPel['name'] += '2LyNoPel' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsSelGNN2LyNoPel)
    modelList += [hParamsSelGNN2LyNoPel['name']]

    
#\\\\\\\\\\\\\\\\\
#\\\ LOCAL GNN \\\
#\\\\\\\\\\\\\\\\\

if doLocalGNN:

    #\\\ Basic parameters for all the Local GNN architectures
    
    hParamsLclGNN = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)
    
    hParamsLclGNN['name'] = 'LclGNN'
    # Chosen architecture
    hParamsLclGNN['archit'] = archit.LocalGNN
    
    # Graph convolutional parameters
    hParamsLclGNN['dimNodeSignals'] = [1, 32, 8] # Features per layer
    hParamsLclGNN['nFilterTaps'] = [5, 5] # Number of filter taps per layer
    hParamsLclGNN['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    hParamsLclGNN['nonlinearity'] = nn.ReLU # Selected nonlinearity
    # Pooling
    hParamsLclGNN['poolingFunction'] = gml.MaxPoolLocal # Summarizing function
    hParamsLclGNN['nSelectedNodes'] = None # To be determined later on
    hParamsLclGNN['poolingSize'] = [10, 10] # poolingSize-hop neighborhood that
        # is affected by the summary
    # Readout layer: local linear combination of features
    hParamsLclGNN['dimReadout'] = [1] # Dimension of the fully connected layers
        # after the GCN layers (map); this fully connected layer is applied only
        # at each node, without any further exchanges nor considering all nodes
        # at once, making the architecture entirely local.
    # Graph structure
    hParamsLclGNN['GSO'] = None # To be determined later on, based on data
    hParamsLclGNN['order'] = 'Degree' 

#\\\\\\\\\\\\
#\\\ MODEL 3: Local GNN with 1 less layer
#\\\\\\\\\\\\

#    hParamsLclGNN1Ly = deepcopy(hParamsLclGNN)
#
#    hParamsLclGNN1Ly['name'] += '1Ly' # Name of the architecture
#    
#    hParamsLclGNN1Ly['dimNodeSignals'] = hParamsLclGNN['dimNodeSignals'][0:-1]
#    hParamsLclGNN1Ly['nFilterTaps'] = hParamsLclGNN['nFilterTaps'][0:-1]
#    hParamsLclGNN1Ly['poolingSize'] = hParamsLclGNN['poolingSize'][0:-1]
#
#    #\\\ Save Values:
#    writeVarValues(varsFile, hParamsLclGNN1Ly)
#    modelList += [hParamsLclGNN1Ly['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 4: Local GNN with all Layers
#\\\\\\\\\\\\

    hParamsLclGNN2Ly = deepcopy(hParamsLclGNN)

    hParamsLclGNN2Ly['name'] += '2Ly' # Name of the architecture

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsLclGNN2Ly)
    modelList += [hParamsLclGNN2Ly['name']]


#\\\\\\\\\\\\\\\\\\
#\\\ COARSENING \\\
#\\\\\\\\\\\\\\\\\\

if doCoarsening:

    #\\\ Basic parameters for all the Selection GNN architectures
    
    hParamsCoarsening = {} # Hyperparameters (hParams) for the Selection GNN (Coarsening)
    
    hParamsCoarsening['name'] = 'Coarsening'
    # Chosen architecture
    hParamsCoarsening['archit'] = archit.SelectionGNN
    
    # Graph convolutional parameters
    hParamsCoarsening['dimNodeSignals'] = [1, 32, 8] # Features per layer
    hParamsCoarsening['nFilterTaps'] = [5, 5] # Number of filter taps per layer
    hParamsCoarsening['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    hParamsCoarsening['nonlinearity'] = nn.ReLU # Selected nonlinearity
    # Pooling
    hParamsCoarsening['poolingFunction'] =  nn.MaxPool1d # Summarizing function
    hParamsCoarsening['nSelectedNodes'] = None # To be determined later on
    hParamsCoarsening['poolingSize'] = [10, 10] # poolingSize-hop neighborhood that
        # is affected by the summary
    # Full MLP readout layer (this layer breaks the locality of the solution)
    hParamsCoarsening['dimLayersMLP'] = [1] # Dimension of the fully connected
        # layers after the GCN layers, we just need to output a single scalar
    # Graph structure
    hParamsCoarsening['GSO'] = None # To be determined later on, based on data
    hParamsCoarsening['order'] = None
    hParamsCoarsening['coarsening'] = True
    
    #\\\ Save Values:
    writeVarValues(varsFile, hParamsCoarsening)
    modelList += [hParamsCoarsening['name']]

###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 0 # After how many training steps, print the partial results
#   0 means to never print partial results while training
xAxisMultiplierTrain = 100 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 20 # How many validation steps in between those shown,
    # same as above.
figSize = 6 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers

#\\\ Save values:
writeVarValues(varsFile,
               {'doPrint': doPrint,
                'doLogging': doLogging,
                'doSaveVars': doSaveVars,
                'doFigs': doFigs,
                'saveDir': saveDir,
                'printInterval': printInterval,
                'figSize': figSize,
                'lineWidth': lineWidth,
                'markerShape': markerShape,
                'markerSize': markerSize})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ Determine processing unit:
if useGPU and torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.empty_cache()
else:
    device = 'cpu'
# Notify:
if doPrint:
    print("Device selected: %s" % device)

#\\\ Logging options
if doLogging:
    # If logging is on, load the tensorboard visualizer and initialize it
    from Utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
accBest = {} # Accuracy for the best model
accLast = {} # Accuracy for the last model
ILconstantBest = {}
phiHatTestBest = {} 
accDiffBest = {} # Accuracy difference for each perturbation with the best model
accDiffLast = {} # Accuracy difference for each perturbation with the last model
GNNdiffBest = {}

for thisModel in modelList: # Create an element for each split realization,
    accBest[thisModel] = [None] * nDataSplits
    accLast[thisModel] = [None] * nDataSplits
    ILconstantBest[thisModel] = [None] * nDataSplits
    accDiffBest[thisModel] = [None] * nDataSplits
    accDiffLast[thisModel] = [None] * nDataSplits
    GNNdiffBest[thisModel] = [None] * nDataSplits
    phiHatTestBest[thisModel] = [None] * nDataSplits

    for split in range(nDataSplits):
        # Across perturbation size
        accDiffBest[thisModel][split] = [None] * nPerturbationSize
        accDiffLast[thisModel][split] = [None] * nPerturbationSize
        GNNdiffBest[thisModel][split] = [None] * nPerturbationSize


####################
# TRAINING OPTIONS #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of these options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

trainingOptions = {}

if doLogging:
    trainingOptions['logger'] = logger
if doSaveVars:
    trainingOptions['saveDir'] = saveDir
if doPrint:
    trainingOptions['printInterval'] = printInterval
if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
trainingOptions['validationInterval'] = validationInterval

#%%##################################################################
#                                                                   #
#                    DATA SPLIT REALIZATION                         #
#                                                                   #
#####################################################################

# Start generating a new data split for each of the number of data splits that
# we previously specified

for split in range(nDataSplits):

    #%%##################################################################
    #                                                                   #
    #                    DATA HANDLING                                  #
    #                                                                   #
    #####################################################################

    ############
    # DATASETS #
    ############
    
    if doPrint:
        print("Loading data", end = '')
        if nDataSplits > 1:
            print(" for split %d" % (split+1), end = '')
        print("...", end = ' ', flush = True)

    #   Load the data, which will give a specific split
    data = Utils.dataTools.MovieLens(graphType, labelID, ratioTrain, ratioValid,
                                     dataDir, keepIsolatedNodes,
                                     forceUndirected, forceConnected, kNN,
                                     maxDataPoints = maxDataPoints,
                                     minRatings = minRatings,
                                     interpolate = interpolateRatings)
    
    if doPrint:
        print("OK")

    #########
    # GRAPH #
    #########
    
    if doPrint:
        print("Setting up the graph...", end = ' ', flush = True)

    # Create graph
    adjacencyMatrix = data.getGraph()
    G = graphTools.Graph('adjacency', adjacencyMatrix.shape[0],
                         {'adjacencyMatrix': adjacencyMatrix})
    G.computeGFT() # Compute the GFT of the stored GSO

    # And re-update the number of nodes for changes in the graph (due to
    # enforced connectedness, for instance)
    nNodes = G.N
    num_selectednodes = [nNodes, nNodes]

    # Once data is completely formatted and in appropriate fashion, change its
    # type to torch
    data.astype(torch.float64)
    # data.to(device)
    # And the corresponding feature dimension that we will need to use
    data.expandDims()
    
    if doPrint:
        print("OK")

    #%%##################################################################
    #                                                                   #
    #                    MODELS INITIALIZATION                          #
    #                                                                   #
    #####################################################################

    # This is the dictionary where we store the models (in a model.Model
    # class, that is then passed to training).
    modelsGNN = {}

    # If a new model is to be created, it should be called for here.
    
    if doPrint:
        print("Model initialization...", flush = True)
        
    for thisModel in modelList:
        
        # Get the corresponding parameter dictionary
        hParamsDict = deepcopy(eval('hParams' + thisModel))
        
        # Now, this dictionary has all the hyperparameters that we need to pass
        # to the architecture, but it also has the 'name' and 'archit' that
        # we do not need to pass them. So we are going to get them out of
        # the dictionary
        thisName = hParamsDict.pop('name')
        callArchit = hParamsDict.pop('archit')
        
        # If more than one graph or data realization is going to be carried out,
        # we are going to store all of thos models separately, so that any of
        # them can be brought back and studied in detail.
        if nDataSplits > 1:
            thisName += 'G%02d' % split
            
        if doPrint:
            print("\tInitializing %s..." % thisName,
                  end = ' ',flush = True)
            
        ##############
        # PARAMETERS #
        ##############

        #\\\ Optimizer options
        #   (If different from the default ones, change here.)
        thisTrainer = trainer
        thisLearningRate = learningRate
        thisBeta1 = beta1
        thisBeta2 = beta2

        #\\\ Ordering
        S, order = graphTools.permIdentity(G.S/np.max(np.real(G.E)))
        # Do not forget to add the GSO to the input parameters of the archit
        if "GraphonGNNRegularSampling" not in thisName and "GraphonGNNRegularIntegration" not in thisName and "GraphonGNNIrregularSampling" not in thisName and "GraphonGNNIrregularIntegration" not in thisName:
            hParamsDict['GSO'] = S
            hParamsDict['nSelectedNodes'] = num_selectednodes
            print(thisName)
            if "AggGNNAggN2" in thisName or "AggGNNNoPelN2" in thisName:
                print("yes1")
                hParamsDict['maxN'] = int( nNodes / 4 )
            
            if "AggGNNAggN4"  in thisName or "AggGNNNoPelN4" in thisName:
                print("yes2")
                hParamsDict['maxN'] = int( nNodes / 6 )

            if "AggGNNAgg5" in thisName or "AggGNNNoPel5" in thisName:
                print("yes3")
                hParamsDict['maxN'] = int( nNodes / 12 )

            if "AggGNN" not in thisName:
                print("yes4")
                hParamsDict['nSelectedNodes'] = num_selectednodes
        elif "GraphonGNNRegularSampling" in thisName: # Only for GraphonGNN
            GSOs = []
            GSOs.append(S)
            n0 = G.N
            S0 = S
            Ns = []
            for n in hParamsDict['nSelectedNodes']:
                #n = S0.shape[1]
                r = np.floor(n0/(n-1))
                idx = list(np.arange(0,n0,r).astype(int))
                Sn = S0[idx,:]
                Sn = Sn[:,idx]
                GSOs.append(Sn)
                Ns.append(Sn.shape[1])                   
                n0 = n
                S0 = Sn
            hParamsDict['GSOs'] = GSOs           
            hParamsDict['nSelectedNodes'] = Ns
            

        elif "GraphonGNNIrregularSampling" in thisName:
            GSOs = []
            GSOs.append(S)
            n0 = G.N
            S0 = S
            Ns = []
            for n in hParamsDict['nSelectedNodes']:
                r = np.floor(n0/(n-1))
                idx_1 = list(np.arange(0,n0,r).astype(int))
                idx = np.sort( np.random.choice( n0, len(idx_1), replace=False ) )
                Sn = S0[idx,:]
                Sn = Sn[:,idx]
                GSOs.append(Sn)
                Ns.append(Sn.shape[1])
                    
                n0 = n
                S0 = Sn
            hParamsDict['GSOs'] = GSOs           
            hParamsDict['nSelectedNodes'] = Ns

                

        elif "GraphonGNNRegularIntegration" in thisName:
            GSOs = []
            GSOs.append(S)
            n0 = G.N
            S0 = S
            Ns = []
            for n in hParamsDict['nSelectedNodes']:
                r = np.floor(n0/(n-1))
                idx = list(np.arange(0,n0,r).astype(int))
                Sn = np.zeros([len(idx), len(idx)])
               
                for i in range(len(idx)):
                    for j in range(i, len(idx)):
                        if j == len(idx)-1 and i == len(idx)-1:
                            temp_ave = np.mean( S0[idx[i]: n0-1, idx[j]: n0-1] )

                        elif i == len(idx)-1:
                            temp_ave = np.mean( S0[idx[i]: n0-1, idx[j]: idx[j+1]] )
                            
                        elif j == len(idx)-1:
                            temp_ave = np.mean( S0[idx[i]: idx[i+1], idx[j]: n0-1] )
                            
                        else:
                            temp_ave = np.mean( S0[idx[i]: idx[i+1], idx[j]: idx[j+1]] )
                        Sn[i,j] = temp_ave
                        Sn[j,i] = Sn[i, j]

                GSOs.append(Sn)
                Ns.append(Sn.shape[1])
                    # print(Sn.shape[1])
                n0 = n
                S0 = Sn
            hParamsDict['GSOs'] = GSOs           
            hParamsDict['nSelectedNodes'] = Ns

        elif "GraphonGNNIrregularIntegration" in thisName:
            GSOs = []
            GSOs.append(S)
            n0 = G.N
            S0 = S
            Ns = []
            for n in hParamsDict['nSelectedNodes']:
                r = np.floor(n0/(n-1))
                idx_1 = list(np.arange(0,n0,r).astype(int))
                idx = np.sort( np.random.choice( n0, len(idx_1), replace=False ) )
                Sn = np.zeros([len(idx), len(idx)])

                for i in range(len(idx)):
                    for j in range(i, len(idx)):
                        if j == len(idx)-1 and i == len(idx)-1:
                            temp_ave = np.mean( S0[idx[i]: n0-1, idx[j]: n0-1] )

                        elif i == len(idx)-1:
                            temp_ave = np.mean( S0[idx[i]: n0-1, idx[j]: idx[j+1]] )
                            
                        elif j == len(idx)-1:
                            temp_ave = np.mean( S0[idx[i]: idx[i+1], idx[j]: n0-1] )
                            
                        else:
                            temp_ave = np.mean( S0[idx[i]: idx[i+1], idx[j]: idx[j+1]] )
                        Sn[i,j] = temp_ave
                        Sn[j,i] = Sn[i, j]

                GSOs.append(Sn)
                Ns.append(Sn.shape[1])
                    # print(Sn.shape[1])
                n0 = n
                S0 = Sn
            hParamsDict['GSOs'] = GSOs          
            hParamsDict['nSelectedNodes'] = Ns


        thisSelected =   hParamsDict['nSelectedNodes'] 
        ################
        # ARCHITECTURE #
        ################

        thisArchit = callArchit(**hParamsDict)
        thisArchit.to(device)
        
        #############
        # OPTIMIZER #
        #############

        if thisTrainer == 'ADAM':
            thisOptim = optim.Adam(thisArchit.parameters(),
                                   lr = learningRate,
                                   betas = (beta1, beta2))
        elif thisTrainer == 'SGD':
            thisOptim = optim.SGD(thisArchit.parameters(),
                                  lr = learningRate)
        elif thisTrainer == 'RMSprop':
            thisOptim = optim.RMSprop(thisArchit.parameters(),
                                      lr = learningRate, alpha = beta1)

        ########
        # LOSS #
        ########
        if "SelGNN2LyNoPel" in thisName or "AggGNNNoPel" in thisName:
            # thisLossFunction = loss.adaptExtraDimensionLoss(lossFunction)
            thisLossFunction = lossFunction()
        else:
            thisLossFunction = loss.integralLipschitzLoss(lossFunction, multiplier)
            S = torch.tensor(S)
            thisLossFunction.addGraph('GSO', S)
            thisLossFunction.addArchit(thisArchit)
            thisLossFunction.astype(torch.float64)

        thisLossFunction.to(device)


        #########
        # MODEL #
        #########

        modelCreated = model.Model(thisArchit,
                                   thisLossFunction,
                                   thisOptim,
                                   thisName, saveDir, order)

        modelsGNN[thisName] = modelCreated

        writeVarValues(varsFile,
                       {'name': thisName,
                        'thisTrainer': thisTrainer,
                        'thisLearningRate': thisLearningRate,
                        'thisBeta1': thisBeta1,
                        'thisBeta2': thisBeta2,
                        'Selected Nodes': thisSelected})

        if doPrint:
            print("OK")
            
    if doPrint:
        print("Model initialization... COMPLETE")

    #%%##################################################################
    #                                                                   #
    #                    TRAINING                                       #
    #                                                                   #
    #####################################################################


    ############
    # TRAINING #
    ############

    # On top of the rest of the training options, we pass the identification
    # of this specific data split realization.

    if nDataSplits > 1:
        trainingOptions['graphNo'] = split

    # This is the function that trains the models detailed in the dictionary
    # modelsGNN using the data data, with the specified training options.
    train.MultipleModels(modelsGNN, data,
                         nEpochs = nEpochs, batchSize = batchSize,
                         **trainingOptions)

    #%%##################################################################
    #                                                                   #
    #                    EVALUATION                                     #
    #                                                                   #
    #####################################################################

    # Now that the model has been trained, we evaluate them on the test
    # samples.

    # We have two versions of each model to evaluate: the one obtained
    # at the best result of the validation step, and the last trained model.

    ########
    # DATA #
    ########

    xTest, yTest = data.getSamples('test')
    xTest = xTest.to(device)
    yTest = yTest.to(device)

    ##############
    # BEST MODEL #
    ##############

    if doPrint:
        print("Total testing RMSE (Best):", flush = True)

    for key in modelsGNN.keys():
        xTestOrdered = xTest[:,:,modelsGNN[key].order]

        with torch.no_grad():
            # Process the samples
            if 'singleNodeForward' in dir(modelsGNN[key].archit):
                if 'getLabelID' in dir(data):
                    targetIDs = data.getLabelID('test')
                    yHatTest = modelsGNN[key].archit\
                                            .singleNodeForward(xTest, targetIDs)
            else:
                yHatTest, phiHatTest = \
                            modelsGNN[key].archit(xTestOrdered)
            # yHatTest is of shape
            #   testSize x numberOfClasses
            # We compute the accuracy
            thisAccBest = data.evaluate(yHatTest, yTest)
            if "NoPel" not in key:
                ILconstant = modelsGNN[key].loss.computeILconstant()
            else:
                ILconstant = 0
            print(ILconstant)
            

        # if doPrint:
        #     print("%s: %6.4f %s" % (key, thisAccBest, phiHatTest.shape), flush = True)

        # Save value
        writeVarValues(varsFile,
                   {'accBest%s' % key: thisAccBest,
                   'ILconstant%s' % key: ILconstant})

        # Now check which is the model being trained
        for thisModel in modelList:
            # If the name in the modelList is contained in the name with
            # the key, then that's the model, and save it
            # For example, if 'SelGNNDeg' is in thisModelList, then the
            # correct key will read something like 'SelGNNDegG01' so
            # that's the one to save.
            if thisModel in key:
                accBest[thisModel][split] = thisAccBest.item()
                if "NoPel" not in key:
                    ILconstantBest[thisModel][split] = [ILconstant.item()]
                else:
                    ILconstantBest[thisModel][split] = [0]
                
                phiHatTestBest[thisModel][split] = phiHatTest.cpu().numpy()
                
            # This is so that we can later compute a total accuracy with
            # the corresponding error.
        
    for thisModel in modelList:
        print("%s: %s" % (thisModel,  phiHatTestBest[thisModel][split].shape), flush = True)

        

    ##############
    # LAST MODEL #
    ##############

    # And repeat for the last model

    if doPrint:
        print("Total testing RMSE (Last):", flush = True)

    # Update order and adapt dimensions
    for key in modelsGNN.keys():
        modelsGNN[key].load(label = 'Last')

        with torch.no_grad():
            # Process the samples
            if 'singleNodeForward' in dir(modelsGNN[key].archit):
                if 'getLabelID' in dir(data):
                    targetIDs = data.getLabelID('test')
                    yHatTest = modelsGNN[key].archit\
                                     .singleNodeForward(xTest, targetIDs)
            else:
                yHatTest,_ = modelsGNN[key].archit(xTest)
            # yHatTest is of shape
            #   testSize x numberOfClasses
            # We compute the accuracy
            thisAccLast = data.evaluate(yHatTest, yTest)

        if doPrint:
            print("%s: %6.4f" % (key, thisAccLast), flush = True)

        # Save values:
        writeVarValues(varsFile,
                   {'accLast%s' % key: thisAccLast})
        # And repeat for the last model:
        for thisModel in modelList:
            if thisModel in key:
                accLast[thisModel][split] = thisAccLast.item()

       
    
  

    #%%##################################################################
    #                                                                   #
    #                    PERTURBATION                                   #
    #                                                                   #
    #####################################################################

    # Now that the models have been trained, we want to see how these models
    # fare for different levels of perturbations. For each perturbation
    # size, we run nPerturbationRealizations to account for the randomness
    # of the perturbation.

    for size in range(nPerturbationSize):

        # Now, we need to initialize the results we will get for each
        # perturbation realization, so
        for thisModel in modelList:
            accDiffBest[thisModel][split][size] = []
            accDiffLast[thisModel][split][size] = []
            GNNdiffBest[thisModel][split][size] = []

        # Select the corresponding value of perturbation
        thisEps = epsilon[size]
        
        if doPrint:
            print("Perturbation Size: %.4f. Running" % thisEps,
                  end = '', flush = True)
            

        # For each value of epsilon, thisEps, we run nPerturbation
        # realizations to account for the randomness of the perturbation
        for perturbation in range(nPerturbationRealizations):

            # Create the perturbed matrix E such that ||E|| < epsilon and
            # ||E/m_N - I|| < epsilon
            E = np.random.uniform(-thisEps,
                                  -(1-thisEps) * thisEps,
                                  nNodes)
            E = np.diag(E)
            # And now that we have the new graph, compute the accuracy, on
            # the new graph, for each model
            for key in modelsGNN.keys():

                #########
                # GRAPH #
                #########

                # Build the perturbed graph
                #   Get the order (this will also be useful for xTest)
                thisOrder = modelsGNN[key].order
                #   Build the corresponding S (with the corresponding order)
                #   (Not great notation, but note that G.E stands for the
                #   eigenvalues of the graph, while E stands for the error
                #   matrix. I know, but it's just that 'lambda' is too long)
                if len(G.S.shape) == 3:
                    S = G.S[:,thisOrder,:][:,:,thisOrder] / np.max(np.real(G.E))
                elif len(G.S.shape) == 2:
                    S = G.S[thisOrder,:][:,thisOrder] / np.max(np.real(G.E))
                #   Get the perturbed graph
                Shat = S + E.conj().T.dot(S) + S.dot(E)
                #   Move it to torch
                Shat = torch.tensor(Shat).to(device)

                # The key contains the number of graph and data realization.
                # However, all the data is indexed by the name of the model,
                # so we need to get rid of this numbering to be able to
                # adequately save the data in the corresponding key.
                for m in modelList:
                    if m in key:
                        thisModel = m

                ##############
                # BEST MODEL #
                ##############

                # Load the Best parameters
                modelsGNN[key].load(label = 'Best')
                # Get the data with the corresponding order (doesn't matter
                # for a one-layer with no pooling)
                xTestOrdered = xTest[:,:,thisOrder]

                with torch.no_grad():
                    # Change the GSO
                    if "SelGNN2Ly" in key:
                        modelsGNN[key].archit.changeGSO(Shat)
                    # Process the samples
                        yHatTestPerturb, phiHatTestPerturb = \
                            modelsGNN[key].archit.splitForward(xTestOrdered)
                    elif "AggGNN" in key:
                        yHatTestPerturb, phiHatTestPerturb = modelsGNN[key].archit.changeS_forward(Shat, xTestOrdered)
                    # yHatTest is of shape
                    #   testSize x numberOfClasses
                    # phiHatTest is of shape
                    #   testSize x numberOfFeatures x numberOfNodes
                    # where numberOfFeatures are the features in the last
                    # layer of the GNN.
                    # We compute the accuracy
                    thisAccBestPerturb = data.evaluate(yHatTestPerturb,
                                                       yTest)

                # Now, compute the differences that we will later average
                #   Convert this accuracy to numpy
                thisAccBestPerturb = thisAccBestPerturb.cpu().numpy()
                #   Best accuracy for this model
                thisAccBest = np.array(accBest[thisModel][split])
                #   Compute the difference
                thisAccDiffBest = np.abs(thisAccBestPerturb - thisAccBest)
                #   And normalize
                thisAccDiffBest = thisAccDiffBest / thisAccBest

                # And also for the output of the GNN
                #   Convert to numpy
                phiHatTestPerturb = phiHatTestPerturb.cpu().numpy()
                #   Get the test output
                thisPhiHatTest = phiHatTestBest[thisModel][split]
                #   Compute norm 2
                thisGNNdiffBest = np.linalg.norm(phiHatTestPerturb - \
                                                 thisPhiHatTest,
                                                 ord = 2, axis = 2)
                #   This has shape sampleSize x numberOfFeatures, and it gives
                #   me a scalar value for each feature. Then, to actually
                #   compute the norm, we need to square this number and add for
                #   all features, which is nothing more than the norm 2 across
                #   the feature dimension
                thisGNNdiffBest = np.linalg.norm(thisGNNdiffBest,
                                                 ord = 2, axis = 1)
                #   Average over the test set
                thisGNNdiffBest = np.mean(thisGNNdiffBest)
               

                
                accDiffBest[thisModel][split][size] += [thisAccDiffBest]
                GNNdiffBest[thisModel][split][size] += [thisGNNdiffBest]

            

                ##############
                # LAST MODEL #
                ##############

                # Load the last parameters
                modelsGNN[key].load(label = 'Last')
                # Get the test data
                xTestOrdered = xTest[:,:,thisOrder]

                with torch.no_grad():
                    if "SelGNN2Ly" in key:
                        # Change the GSO
                        modelsGNN[key].archit.changeGSO(Shat)
                        # Process the samples
                        yHatTestPerturb, phiHatTestPerturb = \
                            modelsGNN[key].archit.splitForward(xTestOrdered)
                    elif "AggGNN" in key:
                        yHatTestPerturb,_ = modelsGNN[key].archit.changeS_forward(Shat, xTestOrdered)
                    # yHatTest is of shape
                    #   testSize x numberOfClasses
                    # phiHatTest is of shape
                    #   testSize x numberOfFeatures x numberOfNodes
                    # We compute the accuracy
                    thisAccLastPerturb = data.evaluate(yHatTestPerturb,
                                                       yTest)

                # Now, compute the differences that we will later average
                #   Convert this accuracy to numpy
                thisAccLastPerturb = thisAccLastPerturb.cpu().numpy()
                #   Last accuracy for this model
                thisAccLast=np.array(accLast[thisModel][split])
                #   Compute the difference
                thisAccDiffLast = np.abs(thisAccLastPerturb - thisAccLast)
                #   And normalize
                thisAccDiffLast = thisAccDiffLast / thisAccLast

                accDiffLast[thisModel][split][size] += [thisAccDiffLast]

                
                
            if doPrint:
                print(".", end = '', flush = True)
                
        if doPrint:
            print(" OK")

        # Now that we have computed the accuracy difference and the GNN
        # output difference, we can store the average for each perturbation
        # size (and replace the information of each realization of each
        # size)
        for thisModel in modelList:
            # Convert them to np.array
            accDiffBest[thisModel][split][size] = \
                                   np.array(accDiffBest[thisModel][split][size])
            accDiffLast[thisModel][split][size] = \
                                   np.array(accDiffLast[thisModel][split][size])
            
            #   This is still a dictionary of lists, but now the last list
            #   has been converted to a numpy array.
            # Average
            #   To avoid keeping new variable names and the likes, we will
            #   replace the np.array (which contains the difference at each
            #   perturbation realization), by its average
            accDiffBest[thisModel][split][size] = \
                                    np.mean(accDiffBest[thisModel][split][size])
            #   By averaging the numpy array we just created, we get a
            #   single scalar, which is the mean accuracy difference over
            #   all perturbations.
            #   Repeat for all variables concerned
            accDiffLast[thisModel][split][size] = \
                                    np.mean(accDiffLast[thisModel][split][size])
            GNNdiffBest[thisModel][split][size] = \
                                    np.mean(GNNdiffBest[thisModel][split][size])
            



############################
# FINAL EVALUATION RESULTS #
############################

# Now that we have computed the accuracy of all runs, we can obtain a final
# result (mean and standard deviation)

meanAccBest = {} # Mean across data splits
meanAccLast = {} # Mean across data splits
meanILconstantBest = {} # Mean IL constant (best model) averaged across graphs
stdDevILconstantBest = {} # Dev IL constant (best model) averaged across graphs
stdDevAccBest = {} # Standard deviation across data splits
stdDevAccLast = {} # Standard deviation across data splits
meanAccDiffBest = {} # Mean across graphs (after having averaged across data
    # realizations)
meanAccDiffLast = {} # Mean across graphs
stdDevAccDiffBest = {} # Mean across graphs (after having averaged across data
    # realizations)
stdDevAccDiffLast = {} # Mean across graphs

meanGNNdiffBest = {}
stdDevGNNdiffBest = {}

if doPrint:
    print("\nFinal evaluations (%02d data splits)" % (nDataSplits))

for thisModel in modelList:
    # Convert the lists into a nDataSplits vector
    accBest[thisModel] = np.array(accBest[thisModel])
    accLast[thisModel] = np.array(accLast[thisModel])
    ILconstantBest[thisModel] = np.array(ILconstantBest[thisModel])
    accDiffBest[thisModel] = np.array(accDiffBest[thisModel])
    accDiffLast[thisModel] = np.array(accDiffLast[thisModel])
    GNNdiffBest[thisModel] = np.array(GNNdiffBest[thisModel])

    # And now compute the statistics (across graphs)
    meanAccBest[thisModel] = np.mean(accBest[thisModel])
    meanAccLast[thisModel] = np.mean(accLast[thisModel])
    meanILconstantBest[thisModel]=np.median(ILconstantBest[thisModel], axis = 0)
    stdDevILconstantBest[thisModel]= np.std(ILconstantBest[thisModel], axis = 0)
    stdDevAccBest[thisModel] = np.std(accBest[thisModel])
    stdDevAccLast[thisModel] = np.std(accLast[thisModel])
    meanAccDiffBest[thisModel] = np.mean(accDiffBest[thisModel], axis = 0)
    meanAccDiffLast[thisModel] = np.mean(accDiffLast[thisModel], axis = 0)
    stdDevAccDiffBest[thisModel] = np.std(accDiffBest[thisModel], axis = 0)
    stdDevAccDiffLast[thisModel] = np.std(accDiffLast[thisModel], axis = 0)
    meanGNNdiffBest[thisModel] = np.mean(GNNdiffBest[thisModel], axis = 0)
    stdDevGNNdiffBest[thisModel] = np.std(GNNdiffBest[thisModel], axis = 0)


    # And print it:
    if doPrint:
        print("\t%s: %6.4f (+-%6.4f) [Best] %6.4f (+-%6.4f) [Last]" % (
                thisModel,
                meanAccBest[thisModel],
                stdDevAccBest[thisModel],
                meanAccLast[thisModel],
                stdDevAccLast[thisModel]))
        print("C = %6.4f (+- %6.4f)" % (meanILconstantBest[thisModel],
                                        stdDevILconstantBest[thisModel]),
              end = '\n\n')
            
        print("\t%s: %s (+-%s) [Best] %s (+-%s) [Last]" % (
                thisModel,
                meanAccDiffBest[thisModel],
                stdDevAccDiffBest[thisModel],
                meanAccDiffLast[thisModel],
                stdDevAccDiffLast[thisModel]))

    # Save values
    writeVarValues(varsFile,
               {'meanAccBest%s' % thisModel: meanAccBest[thisModel],
                'stdDevAccBest%s' % thisModel: stdDevAccBest[thisModel],
                'meanAccLast%s' % thisModel: meanAccLast[thisModel],
                'stdDevAccLast%s' % thisModel : stdDevAccLast[thisModel],
                'meanILconstantBest%s' % thisModel: \
                                               meanILconstantBest[thisModel],
                'stdDevILconstantBest%s' % thisModel: \
                                               stdDevILconstantBest[thisModel]})

#%%##################################################################
#                                                                   #
#                    PLOT                                           #
#                                                                   #
#####################################################################

# Finally, we might want to plot several quantities of interest

if doFigs and doSaveVars:

    ###################
    # DATA PROCESSING #
    ###################

    # Again, we have training and validation metrics (loss and accuracy
    # -evaluation-) for many runs, so we need to carefully load them and compute
    # the relevant statistics from these realizations.

    #\\\ SAVE SPACE:
    # Create the variables to save all the realizations. This is, again, a
    # dictionary, where each key represents a model, and each model is a list
    # for each data split.
    # Each data split, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    lossTrain = {}
    evalTrain = {}
    lossValid = {}
    evalValid = {}
    # Initialize the splits dimension
    for thisModel in modelList:
        lossTrain[thisModel] = [None] * nDataSplits
        evalTrain[thisModel] = [None] * nDataSplits
        lossValid[thisModel] = [None] * nDataSplits
        evalValid[thisModel] = [None] * nDataSplits

    #\\\ FIGURES DIRECTORY:
    saveDirFigs = os.path.join(saveDir,'figs')
    # If it doesn't exist, create it.
    if not os.path.exists(saveDirFigs):
        os.makedirs(saveDirFigs)

    #\\\ LOAD DATA:
    # Path where the saved training variables should be
    pathToTrainVars = os.path.join(saveDir,'trainVars')
    # Get all the training files:
    allTrainFiles = next(os.walk(pathToTrainVars))[2]
    # Go over each of them (this can't be empty since we are also checking for
    # doSaveVars to be true, what guarantees that the variables have been
    # saved.)
    for file in allTrainFiles:
        # Check that it is a pickle file
        if '.pkl' in file:
            # Open the file
            with open(os.path.join(pathToTrainVars,file),'rb') as fileTrainVars:
                # Load it
                thisVarsDict = pickle.load(fileTrainVars)
                # store them
                nBatches = thisVarsDict['nBatches']
                thisLossTrain = thisVarsDict['lossTrain']
                thisEvalTrain = thisVarsDict['evalTrain']
                thisLossValid = thisVarsDict['lossValid']
                thisEvalValid = thisVarsDict['evalValid']
                # This graph is, actually, the data split dimension
                if 'graphNo' in thisVarsDict.keys():
                    thisG = thisVarsDict['graphNo']
                else:
                    thisG = 0
                # And add them to the corresponding variables
                for key in thisLossTrain.keys():
                # This part matches each data realization (matched through
                # the graphNo key) with each specific model.
                    for thisModel in modelList:
                        if thisModel in key:
                            lossTrain[thisModel][thisG] = thisLossTrain[key]
                            evalTrain[thisModel][thisG] = thisEvalTrain[key]
                            lossValid[thisModel][thisG] = thisLossValid[key]
                            evalValid[thisModel][thisG] = thisEvalValid[key]
    # Now that we have collected all the results, we have that each of the four
    # variables (lossTrain, evalTrain, lossValid, evalValid) has a list for
    # each key in the dictionary. This list goes through the data split.
    # Each split realization is actually an np.array.

    #\\\ COMPUTE STATISTICS:
    # The first thing to do is to transform those into a matrix with all the
    # realizations, so create the variables to save that.
    meanLossTrain = {}
    meanEvalTrain = {}
    meanLossValid = {}
    meanEvalValid = {}
    stdDevLossTrain = {}
    stdDevEvalTrain = {}
    stdDevLossValid = {}
    stdDevEvalValid = {}
    # Initialize the variables
    for thisModel in modelList:
        # Transform into np.array
        lossTrain[thisModel] = np.array(lossTrain[thisModel])
        evalTrain[thisModel] = np.array(evalTrain[thisModel])
        lossValid[thisModel] = np.array(lossValid[thisModel])
        evalValid[thisModel] = np.array(evalValid[thisModel])
        # Each of one of these variables should be of shape
        # nDataSplits x numberOfTrainingSteps
        # And compute the statistics
        meanLossTrain[thisModel] = np.mean(lossTrain[thisModel], axis = 0)
        meanEvalTrain[thisModel] = np.mean(evalTrain[thisModel], axis = 0)
        meanLossValid[thisModel] = np.mean(lossValid[thisModel], axis = 0)
        meanEvalValid[thisModel] = np.mean(evalValid[thisModel], axis = 0)
        stdDevLossTrain[thisModel] = np.std(lossTrain[thisModel], axis = 0)
        stdDevEvalTrain[thisModel] = np.std(evalTrain[thisModel], axis = 0)
        stdDevLossValid[thisModel] = np.std(lossValid[thisModel], axis = 0)
        stdDevEvalValid[thisModel] = np.std(evalValid[thisModel], axis = 0)


    ####################
    # SAVE FIGURE DATA #
    ####################

    # And finally, we can plot. But before, let's save the variables mean and
    # stdDev so, if we don't like the plot, we can re-open them, and re-plot
    # them, a piacere.
    #   Pickle, first:
    varsPickle = {}

    varsPickle['nEpochs'] = nEpochs
    varsPickle['nBatches'] = nBatches
    varsPickle['meanLossTrain'] = meanLossTrain
    varsPickle['stdDevLossTrain'] = stdDevLossTrain
    varsPickle['meanEvalTrain'] = meanEvalTrain
    varsPickle['stdDevEvalTrain'] = stdDevEvalTrain
    varsPickle['meanLossValid'] = meanLossValid
    varsPickle['stdDevLossValid'] = stdDevLossValid
    varsPickle['meanEvalValid'] = meanEvalValid
    varsPickle['stdDevEvalValid'] = stdDevEvalValid
    with open(os.path.join(saveDirFigs,'figVars.pkl'), 'wb') as figVarsFile:
        pickle.dump(varsPickle, figVarsFile)
        
    ########
    # PLOT #x
    ########

    # Accuracy Difference (Best)
    fracErrorBar = 3 # Division of the standard deviation
    
    plotAccDiffBest = plt.figure(figsize=(1.5*figSize, 1*figSize))
    for thisModel in modelList:
        plt.errorbar(epsilon,
                     meanAccDiffBest[thisModel],
                     yerr = stdDevAccDiffBest[thisModel]/fracErrorBar,
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    ## Plotting the polynomials



    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.xscale('log')
    plt.ylabel(r'RMSE Difference',fontsize=16)
    plt.xlabel(r'$\varepsilon$',fontsize=16)
    plt.legend(modelList)
    plotAccDiffBest.savefig(os.path.join(saveDirFigs,'accDiffBest.pdf'),
                    bbox_inches = 'tight')

    # Accuracy Difference (Last)
    plotAccDiffLast = plt.figure(figsize=(1.5*figSize, 1*figSize))
    for thisModel in modelList:
        plt.errorbar(epsilon,
                     meanAccDiffLast[thisModel],
                     yerr = stdDevAccDiffLast[thisModel]/fracErrorBar,
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.xscale('log')
    plt.ylabel(r'RMSE Difference')
    plt.xlabel(r'$\varepsilon$')
    plt.legend(modelList)
    plotAccDiffLast.savefig(os.path.join(saveDirFigs,'accDiffLast.pdf'),
                    bbox_inches = 'tight')

    

    plotGNNdiffBest = plt.figure(figsize=(1*figSize, 0.618*figSize))
    for thisModel in modelList:
        plt.errorbar(epsilon,
                     meanGNNdiffBest[thisModel],
                     yerr = stdDevGNNdiffBest[thisModel]/fracErrorBar,
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    SMALL_SIZE = 16
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # ax = plt.gca() #you first need to get the axis handle
    # ratio = 0.6180
    # x_left, x_right = ax.get_xlim()
    # y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$\|\Phi(\mathbf{S},\mathbf{x}) ' +\
                                       '- \Phi(\hat{\mathbf{S}},\mathbf{x})\|$',fontsize=16)
    plt.xlabel(r'$\varepsilon$',fontsize=16)
    plt.legend([   'AggGNNNoPelN', 'AggGNNNoPelN2'  , 'AggGNNNoPelN4' , 'AggGNNNoPelN8'])
    # plt.grid(visible = 1)
    plotGNNdiffBest.savefig(os.path.join(saveDirFigs,'GNNdiffBest.pdf'),
                    bbox_inches = 'tight')






    # Compute the x-axis
    # xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
    # print(xTrain)
    # xValid = np.arange(0, nEpochs * nBatches, \
    #                       validationInterval*xAxisMultiplierValid)

    # # If we do not want to plot all the elements (to avoid overcrowded plots)
    # # we need to recompute the x axis and take those elements corresponding
    # # to the training steps we want to plot
    # if xAxisMultiplierTrain > 1:
    #     # Actual selected samples
    #     selectSamplesTrain = xTrain
        
    #     # Go and fetch tem
    #     for thisModel in modelList:
    #         meanLossTrain[thisModel] = meanLossTrain[thisModel]\
    #                                                 [selectSamplesTrain]
    #         stdDevLossTrain[thisModel] = stdDevLossTrain[thisModel]\
    #                                                     [selectSamplesTrain]
    #         meanEvalTrain[thisModel] = meanEvalTrain[thisModel]\
    #                                                 [selectSamplesTrain]
    #         stdDevEvalTrain[thisModel] = stdDevEvalTrain[thisModel]\
    #                                                     [selectSamplesTrain]
    # print(meanEvalTrain['GraphonGNN'])  


    # # And same for the validation, if necessary.
    # if xAxisMultiplierValid > 1:
    #     selectSamplesValid = np.arange(0, len(meanLossValid[thisModel]), \
    #                                    xAxisMultiplierValid)
    #     for thisModel in modelList:
    #         meanLossValid[thisModel] = meanLossValid[thisModel]\
    #                                                 [selectSamplesValid]
    #         stdDevLossValid[thisModel] = stdDevLossValid[thisModel]\
    #                                                     [selectSamplesValid]
    #         meanEvalValid[thisModel] = meanEvalValid[thisModel]\
    #                                                 [selectSamplesValid]
    #         stdDevEvalValid[thisModel] = stdDevEvalValid[thisModel]\
    #                                                     [selectSamplesValid]

    # #\\\ LOSS (Training and validation) for EACH MODEL
    # for key in meanLossTrain.keys():
    #     lossFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
    #     plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
    #                  color = '#01256E', linewidth = lineWidth,
    #                  marker = markerShape, markersize = markerSize)
    #     plt.errorbar(xValid, meanLossValid[key], yerr = stdDevLossValid[key],
    #                  color = '#95001A', linewidth = lineWidth,
    #                  marker = markerShape, markersize = markerSize)
    #     plt.ylabel(r'Loss')
    #     plt.xlabel(r'Training steps')
    #     plt.legend([r'Training', r'Validation'])
    #     plt.title(r'%s' % key)
    #     lossFig.savefig(os.path.join(saveDirFigs,'loss%s.pdf' % key),
    #                     bbox_inches = 'tight')

    # #\\\ RMSE (Training and validation) for EACH MODEL
    # for key in meanEvalTrain.keys():
    #     accFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
    #     plt.errorbar(xTrain, meanEvalTrain[key], yerr = stdDevEvalTrain[key],
    #                  color = '#01256E', linewidth = lineWidth,
    #                  marker = markerShape, markersize = markerSize)
    #     plt.errorbar(xValid, meanEvalValid[key], yerr = stdDevEvalValid[key],
    #                  color = '#95001A', linewidth = lineWidth,
    #                  marker = markerShape, markersize = markerSize)
    #     plt.ylabel(r'RMSE')
    #     plt.xlabel(r'Training steps')
    #     plt.legend([r'Training', r'Validation'])
    #     plt.title(r'%s' % key)
    #     accFig.savefig(os.path.join(saveDirFigs,'eval%s.pdf' % key),
    #                     bbox_inches = 'tight')

    # # LOSS (training) for ALL MODELS
    # allLossTrain = plt.figure(figsize=(1.61*figSize, 1*figSize))
    # for key in meanLossTrain.keys():
    #     plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
    #                  linewidth = lineWidth,
    #                  marker = markerShape, markersize = markerSize)
    # plt.ylabel(r'Loss')
    # plt.xlabel(r'Training steps')
    # plt.legend(list(meanLossTrain.keys()))
    # allLossTrain.savefig(os.path.join(saveDirFigs,'allLossTrain.pdf'),
    #                 bbox_inches = 'tight')

    # # RMSE (validation) for ALL MODELS
    # allEvalValid = plt.figure(figsize=(1.61*figSize, 1*figSize))
    # for key in meanEvalValid.keys():
    #     plt.errorbar(xValid, meanEvalValid[key], yerr = stdDevEvalValid[key],
    #                  linewidth = lineWidth,
    #                  marker = markerShape, markersize = markerSize)
    # plt.ylabel(r'RMSE')
    # plt.xlabel(r'Training steps')
    # plt.legend(list(meanEvalValid.keys()))
    # allEvalValid.savefig(os.path.join(saveDirFigs,'allEvalValid.pdf'),
    #                 bbox_inches = 'tight')
