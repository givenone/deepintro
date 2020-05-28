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
        
        C = tf.reduce_max(self.y_input)
        sh = tf.shape(self.x_input)
        # If loss function is Cross-Entropy loss
        if self.loss_func == 'xent':
            self.losses = tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(self.y_input, 10), logits = self.logits)
        # If loss function is Carlini-Wagner loss
        elif self.loss_func == 'cw':
            correct_logits = tf.gather_nd(self.logits, 
            tf.stack( [tf.range(sh[0]), self.y_input], axis = 1)
            )

            copy = tf.identity(self.logits)

            copy = tf.tensor_scatter_nd_update(copy,
            tf.stack( [tf.range(sh[0]), self.y_input] , axis = 1),
            tf.zeros(sh[0]) )

            max_logits = tf.reduce_max(copy, axis = 1) 

            #print(self.logits, max_logits, correct_logits)
            self.losses = tf.maximum(max_logits - correct_logits, -50)
        else:
            print('Loss function must be xent or cw')
            sys.exit()
        
        ###########################################################################################################
        # TODO: Define a tensor whose value represents the gradient of loss function with respect to input image. #
        ###########################################################################################################
 
        self.gradients =  tf.gradients(self.losses, [self.x_input])
    
    def perturb(self, image, label, sess):
        ################################################################################################
        # TODO: Given an image and the corresponding label, generate an adversarial image using FGSM.  # 
        # Please note thatW the pixel values of an adversarial image must be in a valid range [0, 255]. #
        ################################################################################################
        lower = np.maximum(image - self.epsilon, 0)
        upper = np.minimum(image + self.epsilon, 255)
         
        grads = sess.run(self.gradients, feed_dict={self.x_input: image, self.y_input : label})[0]

        adv_image = image + self.epsilon * np.sign(grads)
        adv_image = np.clip(adv_image, lower, upper)
        
        return adv_image
      
