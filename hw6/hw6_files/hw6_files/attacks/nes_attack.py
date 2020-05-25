import numpy as np
import tensorflow.compat.v1 as tf
import sys

class NESAttack(object):
    def __init__(self, model, epsilon, step_size, num_steps, loss_func):
        # Arguements
        self.epsilon = epsilon # maximum perturbation
        self.step_size = step_size # step size per iteration
        self.num_steps = num_steps # the number of iterations
        self.loss_func = loss_func # type of loss function (xent for Cross-Entropy loss, cw for Carlini-Wagner loss)
        
        # Networks
        self.x_input = model.x_input # a placeholder for image
        self.y_input = model.y_input # a placeholder for label
        self.logits = model.logits # unnormalized logits

        # NES
        self.nes_batch_size = 50 # number of vectors to sample for NES estimation
        self.sigma = 0.25 # noise scale variable for NES estimation

        ##################################################################################
        # TODO: Implement Cross-Entropy loss and Carlini-Wagner loss.                    #
        # For Carlini-Wagner loss, set the confidence of hinge loss to 50.               #
        # Be careful about the sign of the loss.                                         #
        # You may use the same implementation from FGSM!                                 #
        ##################################################################################
        
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

    def grad_est(self, image, label, sess):
        ###########################################################################################################
        # TODO: Estimate the pixelwise gradient of the image regarding to the loss function using NES technique.  #
        # Below structure is given for guidance, but it's free to ignore it and build on your own.                #
        #                                                                                                         #
        # noise_pos: (self.nes_batch_size) number of image-size random vectors sampled from standard normal dist. #
        # noise: full noise concatenating noise_pos and -noise_pos                                                #
        # image_batch: image tiled and added with sigma * noise                                                   #
        # label_batch: label tiled to (image_batch) size                                                          #
        # grad_est: resulting gradient estimation                                                                 #
        ###########################################################################################################

        n, h, w, c = image.shape
        noise_pos = None
        noise = None
        image_batch = None
        label_batch = None
        
        losses = sess.run(self.losses,
                          feed_dict={self.x_input: image_batch,
                                     self.y_input: label_batch})
        
        grad_est = None

        return grad_est
    
    def perturb(self, image, label, sess):
        ###################################################################################################
        # TODO: Given an image and the corresponding label, generate an adversarial image using black-box #
        # PGD attack with NES gradient estimation.                                                        #
        # Please note that the pixel values of an adversarial image must be in a valid range [0, 255].    #
        ###################################################################################################
        
        lower = None
        upper = None
        adv_image = np.copy(image)
        
        for step in range(None):
            grads = None
            adv_image = None
            adv_image = np.clip(adv_image, lower, upper)
        
        return adv_image
        
