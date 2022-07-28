# 2019/07/22~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
loss.py Loss functions

adaptExtraDimensionLoss: wrapper that handles extra dimensions
"""

import torch.nn as nn
import numpy as np
import torch

import Utils.graphTools as graphTools

class integralLipschitzLoss(nn.modules.loss._Loss):
    
    def __init__(self, lossFunction, multiplier):
        # Initialize parent class
        super().__init__()
        # Initialize loss function (has to be a valid loss function in the
        # torch.nn module)
        self.loss = lossFunction() # (We just use the forward of this function)
        # Save the multiplier value
        self.multiplier = multiplier
        # Attribute to store the eigenvalues
        self.eigenvalues = None
        self.eigenvaluesPowers = None
        self.newEigenvalues = None # None means there are no eigenvalues
        #   stored, True means that the graph has changed, and therefore are
        #   new eigenvalues, False means that the eigenvalues have already
        #   been used to compute the eigenvaluesPowers and need not be computed
        #   again. Essentially, adding a new graph sets this variable to True,
        #   and adding an architecture for the first time (thus, computing the
        #   eigenvaluesPowers) sets it to False.
        # Attribute to store the architecture
        self.archit = None
        # Penalty dictionary
        self.penaltyList = ['ILpenalty']
        self.nPenalties = len(self.penaltyList)
        
    def addGraph(self, graphType, *args):
        # The information we actually need are a range of eigenvalues, so, to
        # not be constantly computing the eigendecomposition, we would give the
        # chance of inputing any of the following three options:
        #   'GSO': Give the GSO so that the eigenvalue will be computed
        #   'eigenvalues': Give the range of eigenvalues directly
        #   'interval': Give maximum and minimum intervals
        assert graphType == 'GSO' or graphType == 'eigenvalues' \
                or graphType == 'interval'
        # Let's start with the first case of inputing a GSO
        if graphType == 'GSO':
            assert len(args) == 1
            S = args[0]
            # S could be a E x N x N or an N x N
            assert len(S.shape) == 2 or len(S.shape) == 3
            # Ideally, the GSO S would be a numpy array, but it could also be a
            # torch.tensor in situations where the S changes dynamically, so, 
            # if it is a torch.tensor, convert it into a numpy
            if 'torch' in repr(S.dtype):
                device = S.device
                S = S.data.cpu().numpy()
                useTorch = True
            else:
                useTorch = False
            #   If the GSO is being learned, somehow, this operation destroys 
            #   any possibility of taking gradients with respect to S, so that
            #   the penalty is _only_ on the filter taps, and not on the
            #   eigenvalues.
            #   In any case, we wouldn't be able to backpropagate on the
            #   eigenvalues, anyway.
            # Now, check whether we have two or three dimensions
            if len(S.shape) == 2:
                assert S.shape[0] == S.shape[1] # Check it is a square matrix
                N = S.shape[0] # Number of nodes
                E = 1 # number of edge features
                S = S.reshape([1, N, N]) # 1 x N x N
            else:
                assert S.shape[1] == S.shape[2] # Check it is a collection of
                    # square matrices
                N = S.shape[1] # Number of nodes
                E = S.shape[0] # Number of edge features
            # Compute the GFT
            #   Store the eigenvalues in an E x N matrix
            eigenvalues = np.zeros([E,N])
            #   Compute the GFT for each edge feature
            for e in range(E):
                eigenvalueMatrix, _ = graphTools.computeGFT(S[e], order = 'no')
                eigenvalues[e] = np.diag(eigenvalueMatrix)
            # Check if we need to return them to torch
            if useTorch:
                eigenvalues = torch.tensor(eigenvalues).to(device)
                
        # Next, an input list of eigenvalues is given
        elif graphType == 'eigenvalues':
            assert len(args) == 1
            eigenvalues = args[0]
            # Check appropriate shape
            assert len(eigenvalues.shape) == 1 or len(eigenvalues.shape) == 2
            # If it has only one edge feature, add it explicitly.
            if len(eigenvalues.shape) == 1:
                N = eigenvalues.shape[0]
                eigenvalues = eigenvalues.reshape([1, N])
                
        # Finally, it could just be a min and max eigenvalue, and we fill in
        # with the numbers in between
        elif graphType == 'interval':
            # Here, the arguments need to be at least two (min, max), and at
            # most three (min, max, number of samples)
            assert len(args) == 2 or len(args) == 3
            eigenvalueMin = args[0]
            eigenvalueMax = args[1]
            assert eigenvalueMin < eigenvalueMax # This is a strict inequality!
                # Otherwise, we don't have any interval to create
            # Check if they have to be in torch or can be in numpy
            if 'torch' in repr(eigenvalueMin.dtype) \
                    and 'torch' in repr(eigenvalueMax.dtype):
                useTorch = True
                device = eigenvalueMin.device
                eigenvalueMin = eigenvalueMin.data.cpu().numpy()
                eigenvalueMax = eigenvalueMin.data.cpu().numpy()
            # Check if we have a pre-specified number of 
            if len(args) == 3:
                nSamples = int(args[3]) # Force it to be an int
                assert nSamples > 0 # There has to be some sample to create
            else:
                # By default, we create 10 samples
                nSamples = 10
            # Create the eigenvalues
            eigenvalues = np.linspace(eigenvalueMin, eigenvalueMax,
                                      num = nSamples)
            # Check if we need to return them to torch
            if useTorch:
                eigenvalues = torch.tensor(eigenvalues).to(device)

        # And store it in internal attributes
        self.eigenvalues = eigenvalues
        # Add a flag to keep track if eigenvalues have been updated
        # (i.e. whenever a graph gets update; this flag will indicate that the
        # eigenvalues powers need to be recalculated, when adding a new 
        # architecture)
        self.newEigenvalues = True
            
    def astype(self, dataType):
        
        if self.eigenvalues is not None:
            if 'torch' in repr(dataType): # If the target data type is torch
                self.eigenvalues = torch.tensor(self.eigenvalues).type(dataType)
            else: # If it is not torch (and it thus should be numpy)
                self.eigenvalues = dataType(self.eigenvalues)
                
    def to(self, device):
        
        # Only works if they are torch
        if 'torch' in repr(self.eigenvalues.dtype):
            self.eigenvalues = self.eigenvalues.to(device)
            
    def addArchit(self, archit):
        # Check that the architecture and the eigenvalues have the same number
        # of edge features
        assert self.eigenvalues is not None \
                and self.eigenvalues.shape[0] == archit.E
        # Now we need to add the architecture
        self.archit = archit
        #   Note that this is a pointer, so it is the _same_ architecture, so
        #   we have access to the same parameters, which is what we were looking
        #   for.
        # Now, let's compute the eigenvalue powers. To avoid unnecessary 
        # computations (recall that for each training round, we update the 
        # architecture, but the GSO remains the same), we compute the powers
        # of the eigenvalues only when they haven't been computed yet, or when
        # the architecture changed (most significantly, we have more filter taps
        # so we need more powers)
        if self.newEigenvalues is True:
            self.computeEigenvaluesPowers()
            self.newEigenvalues = False # Once we have computed the powers for
                # the new eigenvalues, we can just set this variable to false.
                # It will be back to True whenever we add a new graph, and 
                # therefore, new eigenvalues.
        elif max(self.archit.K) > self.eigenvaluesPowers.shape[1]:
            self.computeEigenvaluesPowers()
        
    def computeEigenvaluesPowers(self):
        
        assert self.archit is not None
        
        # Let's start by preparing the eigenvalues.
        # The eigenvalues are a matrix of shape E x nSamples.
        # And we need to multiply this, elementwise, with a matrix of filter 
        # taps that would be of shape
        #   F x E x K x G
        # where
        #   F: output features
        #   E: edge features
        #   K: filter taps
        #   G: input features
        # While we don't care about F and G (the same eigenvalues have to be
        # multiplied, irrespective of the F and G dimensions), we do care about
        # the E, K and nSamples dimension. So we will make the eigenvalues a
        # E x K x nSamples tensor.
        # So, first, let's get the value of K
        K = max(self.archit.K)
        E = self.eigenvalues.shape[0]
        nSamples = self.eigenvalues.shape[1]
        # First goes a zero (that multiplies the first filter h_{0}), then goes
        # h_{1} that multiplies all ones, and then goes all the powers, because
        # the penalty is lambda * h'(lambda) = lambda * (h1 + h2 lambda + ...)
        eigenvaluesPowers = torch.zeros([E, 1, nSamples],
                                 device = self.eigenvalues.device)
        eigenvaluesPowers = torch.cat((eigenvaluesPowers,
                                       torch.ones([E, 1, nSamples],
                                                  device = \
                                                      self.eigenvalues.device)),
                                      dim = 1) # E x 2 x nSamples
        # Now we need to multiply the eigenvalues with themselves K-1 times
        thisEigenvalues = torch.ones([E, nSamples],
                                     device = self.eigenvalues.device)
        for k in range(1,K-1):
            thisEigenvalues = thisEigenvalues * self.eigenvalues
            eigenvaluesPowers = torch.cat((eigenvaluesPowers,
                                     thisEigenvalues.reshape([E, 1, nSamples])),
                                    dim = 1) # E x k x nSamples
        # Now, eigenvalues is supposed to be of shape E x K x nSamples
        self.eigenvaluesPowers = eigenvaluesPowers
        
    def computeILconstant(self):
        
        assert self.archit is not None
        #   This also checks that we have the eigenvalues and the 
        #   eigenvaluesPowers since they were computed through the addArchit()
        #   method.
        
        E = self.eigenvalues.shape[0]
        nSamples = self.eigenvalues.shape[1]
        
        # Let's move onto each parameter
        l = 0 # Layer counter
        ILconstant = torch.empty(0).to(self.eigenvalues.device) 
        agg = 0
            # Initial value for the IL penalty
        # For each parameter,
        for param in self.archit.parameters():
            # print(len(param.shape))
            # Check it has dimension 4 (it is the filter taps)
            if len(param.shape) == 4 or len(param.shape)==3:
                
                if len(param.shape)==3:
                    agg = 1
                    assert param.shape[0] == self.archit.F[l+1] # F
                    assert param.shape[2] == self.archit.K[l]   # K
                    assert param.shape[1] == self.archit.F[l]   # G
                    Fl = param.shape[0]
                    Kl = param.shape[2]
                    Gl = param.shape[1]
                    E = 1
                else:
                # Check if the dimensions coincide, the param has to have
                # Fl x E x Kl x Gl
                # where Fl, Gl and Kl change with each layer l, but E is fixed
                    assert param.shape[0] == self.archit.F[l+1] # F
                    assert param.shape[1] == E                  # E
                    assert param.shape[2] == self.archit.K[l]   # K
                    assert param.shape[3] == self.archit.F[l]   # G
                    Fl = param.shape[0]
                    Kl = param.shape[2]
                    Gl = param.shape[3]
                # We need to multiply it with the eigenvalues that have shape
                # E x K x nSamples
                # So we reshape it to be
                param = param.reshape([Fl, Gl, E, Kl, 1])
                # Repeat it the length of the eigenvalues
                param = param.repeat([1, 1, 1, 1, nSamples])
                # Elementwise multiplication with the eigenvalues to get the
                # integral Lipscthiz condition
                hPrime = param * self.eigenvaluesPowers[:, 0:Kl, :]
                #   h'(lambda): Derivative of h
                #   Fl x Gl x E x K x nSamples
                # And we need to add over the K dimension to actually compute
                # the derivative
                hPrime = torch.sum(hPrime, dim = 3)
                #   Fl x Gl x E x nSamples
                # Now, we need to multiply h'(lambda) with lambda.
                # The eigenvalues have shape E x nSamples
                # hPrime has shape Fl x Gl x E x nSamples
                # So we need to reshape the eigenvalues
                thisILconstant = self.eigenvalues.reshape([1,1,E,nSamples]) \
                                                                        * hPrime + hPrime

                                                                        
                # Apply torch.abs and torch.max over the nSamples dimension
                # (second argument are the positions of the maximum, which we 
                # don't care about).
                thisILconstant, _ = torch.max(torch.abs(thisILconstant), dim=3)
                #   Fl x Gl x E
                # This torch.max does not have a second argument because it is
                # the maximum of all the numbers
                thisILconstant = torch.max(thisILconstant)
                # Add the constant to the list
              
                ILconstant = torch.cat((ILconstant,thisILconstant.unsqueeze(0)))
                
                # And increase the number of layers
                l = l + 1
        if agg:
            # print(ILconstant[0])
            finalILconstant = ILconstant[0].clone().detach().requires_grad_(True)
        else:
            finalILconstant = torch.max(ILconstant, dim = 0).values

        
        # After we computed the IL constant for each layer, we pick the
        # maximum and go with that
        return finalILconstant
        
        
    def forward(self, *args):
        # args have to be the exact same arguments that would go into the 
        # torch.nn loss function that was initialized in this wrapper
        mainLoss = self.loss(*args) # Output of selected loss function
        
        # Compute the value of the constant so far
        ILpenalty = self.computeILconstant()
        
        # And add it to the penalty dictionary for output
        penalty = {}
        penalty[self.penaltyList[0]] = ILpenalty
                
        # Now that we have computed the integral Lipschitz penalty, we can add
        # it to the main loss (weighed by the multiplier), and returned
        return mainLoss + self.multiplier * ILpenalty, mainLoss, penalty
        
    def extra_repr(self):
        for p in range(self.nPenalties):
            reprString = '(penalty%02d): %s\n' % (p+1, self.penaltyList[p])
            
        return reprString

class adaptExtraDimensionLoss(nn.modules.loss._Loss):
    """
    adaptExtraDimensionLoss: wrapper that handles extra dimensions
    
    Some loss functions take vectors as inputs while others take scalars; if we
    input a one-dimensional vector instead of a scalar, although virtually the
    same, the loss function could complain.
    
    The output of the GNNs is, by default, a vector. And sometimes we want it
    to still be a vector (i.e. crossEntropyLoss where we output a one-hot 
    vector) and sometimes we want it to be treated as a scalar (i.e. MSELoss).
    Since we still have a single training function to train multiple models, we
    do not know whether we will have a scalar or a vector. So this wrapper
    adapts the input to the loss function seamlessly.
    
    Eventually, more loss functions could be added to the code below to better
    handle their dimensions.
    
    Initialization:
        
        Input:
            lossFunction (torch.nn loss function): desired loss function
            arguments: arguments required to initialize the loss function
            >> Obs.: The loss function gets initialized as well
            
    Forward:
        Input:
            estimate (torch.tensor): output of the GNN
            target (torch.tensor): target representation
    """
    
    # When we want to compare scalars, we will have a B x 1 output of the GNN,
    # since the number of features is always there. However, most of the scalar
    # comparative functions take just a B vector, so we have an extra 1 dim
    # that raises a warning. This container will simply get rid of it.
    
    # This allows to change loss from crossEntropy (class based, expecting 
    # B x C input) to MSE or SmoothL1Loss (expecting B input)
    
    def __init__(self, lossFunction, *args):
        # The second argument is optional and it is if there are any extra 
        # arguments with which we want to initialize the loss
        
        super().__init__()
        
        if len(args) > 0:
            self.loss = lossFunction(*args) # Initialize loss function
        else:
            self.loss = lossFunction()
        
    def forward(self, estimate, target):
        
        # What we're doing here is checking what kind of loss it is and
        # what kind of reshape we have to do on the estimate
        
        if 'CrossEntropyLoss' in repr(self.loss):
            # This is supposed to be a one-hot vector batchSize x nClasses
            assert len(estimate.shape) == 2
        elif 'SmoothL1Loss' in repr(self.loss) \
                    or 'MSELoss' in repr(self.loss) \
                    or 'L1Loss' in repr(self.loss):
            # In this case, the estimate has to be a batchSize tensor, so if
            # it has two dimensions, the second dimension has to be 1
            if len(estimate.shape) == 2:
                assert estimate.shape[1] == 1
                estimate = estimate.squeeze(1)
            assert len(estimate.shape) == 1
        
        return self.loss(estimate, target)