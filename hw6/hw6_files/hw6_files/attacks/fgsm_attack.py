import numpy as np
import sys
import tensorflow.compat.v1 as tf
tf.experimental.output_all_intermediates(True)

class FGSMAttack(object):
    def __init__(self, model, epsilon, loss_func):
        # Arguements
        self.epsilon = epsilon # maximum perturbation
        self.loss_func = loss_func # type of loss function (xent for Cross-Entropy loss, cw for Carlini-Wagner loss)
        
        # Networks
        self.x_input = model.x_input # a placeholder for image
        self.y_input = model.y_input # a placeholder for label
        self.logits = model.logits # unnormalized logits
        
        ####################################################################
        # TODO: Implement Cross-Entropy loss and Carlini-Wagner loss.      #
        # For Carlini-Wagner loss, set the confidence of hinge loss to 50. #
        # Be careful about the sign of loss.                               #
        ####################################################################
        
        # If loss function is Cross-Entropy loss
        if self.loss_func == 'xent':
            self.losses = None
        # If loss function is Carlini-Wagner loss
        elif self.loss_func == 'cw':
            correct_logits = None
            max_logits = None
            self.losses = None
        else:
            print('Loss function must be xent or cw')
            sys.exit()
        
        ###########################################################################################################
        # TODO: Define a tensor whose value represents the gradient of loss function with respect to input image. #
        ###########################################################################################################
 
        self.gradients = None
    
    def perturb(self, image, label, sess):
        ################################################################################################
        # TODO: Given an image and the corresponding label, generate an adversarial image using FGSM.  # 
        # Please note that the pixel values of an adversarial image must be in a valid range [0, 255]. #
        ################################################################################################
        lower = None
        upper = None
         
        grads = None
        adv_image = None
        adv_image = np.clip(adv_image, lower, upper)
        
        return adv_image
      
