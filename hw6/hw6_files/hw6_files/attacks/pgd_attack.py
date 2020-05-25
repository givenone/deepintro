import numpy as np
import tensorflow.compat.v1 as tf

from attacks.fgsm_attack import FGSMAttack

class PGDAttack(object):
    def __init__(self, model, epsilon, step_size, num_steps, loss_func):
        # Arguements
        self.epsilon = epsilon # maximum perturbation
        self.step_size = step_size # step size per iteration
        self.num_steps = num_steps # the number of iterations
        self.loss_func = loss_func # type of loss function (xent for Cross-Entropy loss, cw for Carlini-Wagner loss)
        
        # Networks
        self.x_input = model.x_input # a placeholder for image
        self.y_input = model.y_input # a placeholder for label
        
        ##########################################
        # TODO: Create an instance of FGSMAttack #
        ##########################################
        
        self.fgsm_attack = None
    
    def perturb(self, image, label, sess):
        ################################################################################################
        # TODO: Given an image and the corresponding label, generate an adversarial image using PGD.   # 
        # Please note that the pixel values of an adversarial image must be in a valid range [0, 255]. #
        ################################################################################################
        
        lower = None
        upper = None
        adv_image = np.copy(image)
        
        for step in range(None):
            adv_image = None
            adv_image = np.clip(adv_image, lower, upper)
        
        return adv_image

