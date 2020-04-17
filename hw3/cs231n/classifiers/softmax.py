from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    D, C = W.shape
    N, _ = X.shape

    for i in range(N) : 
      l = np.dot(X[i], W)
      exp = np.exp(l)
      numerator = np.sum(exp)

      for c in range(C) :
        p = exp[c] / numerator
        if c == y[i] :
          dW[: , c] += (p - 1) * X[i]
        else :
          dW[: , c] += p * X[i]

      loss += -np.log(exp[y[i]] / numerator)  
    
    loss /= N
    loss +=  reg * np.sum(W * W)
    dW /= N
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    D, C = W.shape
    N, _ = X.shape

    l = np.dot(X, W)
    exp = np.exp(l)
    ans = exp[np.arange(N), y]
    numerator = np.sum(exp, axis = 1)
    loss = -np.log(ans / numerator)
    loss = np.sum(loss) / N + reg * np.sum(W*W)

    
    p = exp / numerator[:, np.newaxis]
    p[np.arange(N), y] -= 1

    dW += np.dot(X.T, p)
    dW  = dW / N + reg * W




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
