"""
Project code+scripts for 8DC00 course.
"""

import numpy as np
import cad_util as util
import matplotlib.pyplot as plt
import registration as reg
import cad
import scipy
from IPython.display import display, clear_output
import scipy.io


def nuclei_measurement():

    fn = '../data/nuclei_data.mat'
    mat = scipy.io.loadmat(fn)
    test_images = mat["test_images"] # shape (24, 24, 3, 20730)
    test_y = mat["test_y"] # shape (20730, 1)
    training_images = mat["training_images"] # shape (24, 24, 3, 21910)
    training_y = mat["training_y"] # shape (21910, 1)

    montage_n = 300
    sort_ix = np.argsort(training_y, axis=0)
    sort_ix_low = sort_ix[:montage_n] # get the 300 smallest
    sort_ix_high = sort_ix[-montage_n:] #Get the 300 largest

    # visualize the 300 smallest and the 300 largest nuclei
    X_small = training_images[:,:,:,sort_ix_low.ravel()]
    X_large = training_images[:,:,:,sort_ix_high.ravel()]
    fig = plt.figure(figsize=(16,8))
    ax1  = fig.add_subplot(121)
    ax2  = fig.add_subplot(122)
    util.montageRGB(X_small, ax1)
    ax1.set_title('300 smallest nuclei')
    util.montageRGB(X_large, ax2)
    ax2.set_title('300 largest nuclei')

    # dataset preparation
    imageSize = training_images.shape

    # every pixel is a feature so the number of features is:
    # height x width x color channels
    numFeatures = imageSize[0]*imageSize[1]*imageSize[2]
    training_x = training_images.reshape(numFeatures, imageSize[3]).T.astype(float)
    test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

    ## training linear regression model
    #---------------------------------------------------------------------#
    # TODO: Implement training of a linear regression model for measuring
    # the area of nuclei in microscopy images. Then, use the trained model
    # to predict the areas of the nuclei in the test dataset.

    training_x_ones= util.addones(training_x)

    Theta,E_train=reg.ls_solve(training_x_ones,training_y)

    predicted_y = util.addones(test_x).dot(Theta)


    ## quadratic regression
    training_x_ones_quad = util.addones(np.concatenate((training_x,training_x**2),axis=1))

    Theta_quad , E_train = reg.ls_solve(training_x_ones_quad, training_y)

    predicted_y_quad = util.addones(np.concatenate((test_x,test_x**2),axis=1)).dot(Theta_quad)

    # Error

    E = np.transpose(predicted_y - test_y).dot(predicted_y - test_y)/len(test_y)
    E_quad = np.transpose(predicted_y_quad - test_y).dot(predicted_y_quad - test_y)/len(test_y)
    print(E)
    print(E_quad)
    #---------------------------------------------------------------------#

    # visualize the results
    fig2 = plt.figure(figsize=(16,8))
    ax1  = fig2.add_subplot(221)
    line1, = ax1.plot(predicted_y, test_y, ".g", markersize=3)
    ax1.grid()
    ax1.set_xlabel('Area')
    ax1.set_ylabel('Predicted Area')
    ax1.set_title('Training with full sample')

    ax2 = fig2.add_subplot(222)
    line1, = ax2.plot(predicted_y_quad, test_y, ".g", markersize=3)
    ax2.grid()
    ax2.set_xlabel('Area')
    ax2.set_ylabel('Predicted Area')
    ax2.set_title('Training with full sample quadratic')

    #training with smaller number of training samples
    #---------------------------------------------------------------------#
    # Train a model with reduced dataset size (e.g. every fourth
    # training sample).

    N=4;
    ix = np.random.randint(len(training_x), size=round(len(training_x)/N))
    d_training_x = training_x[ix, :]
    d_training_y = training_y[ix, :]

    d_training_x_ones = util.addones(d_training_x)

    d_Theta, d_E_train = reg.ls_solve(d_training_x_ones, d_training_y)

    d_predicted_y = util.addones(test_x).dot(d_Theta)

    # quadratic regression with downsampled data
    d_training_x_ones_quad = util.addones(np.concatenate((d_training_x, d_training_x**2),axis=1))

    d_Theta_quad, E_train = reg.ls_solve(d_training_x_ones_quad, d_training_y)

    d_predicted_y_quad = util.addones(np.concatenate((test_x, test_x ** 2),axis=1)).dot(d_Theta_quad)

    # Error

    d_E = np.transpose(d_predicted_y - test_y).dot(d_predicted_y - test_y) / len(test_y)
    d_E_quad = np.transpose(d_predicted_y_quad - test_y).dot(d_predicted_y_quad - test_y) / len(test_y)
    print(d_E)
    print(d_E_quad)

    #---------------------------------------------------------------------#

    # visualize the results
    ax3  = fig2.add_subplot(223)
    line2, = ax3.plot(d_predicted_y, test_y, ".g", markersize=3)
    ax3.grid()
    ax3.set_xlabel('Area')
    ax3.set_ylabel('Predicted Area')
    ax3.set_title('Training with smaller sample')

    ax4  = fig2.add_subplot(224)
    line2, = ax4.plot(d_predicted_y_quad, test_y, ".g", markersize=3)
    ax4.grid()
    ax4.set_xlabel('Area')
    ax4.set_ylabel('Predicted Area')
    ax4.set_title('Training with smaller sample with quadratic regression')




def nuclei_classification():
    ## dataset preparation
    fn = '../data/nuclei_data_classification.mat'
    mat = scipy.io.loadmat(fn)

    test_images = mat["test_images"] # (24, 24, 3, 20730)
    test_y = mat["test_y"] # (20730, 1)
    training_images = mat["training_images"] # (24, 24, 3, 14607)
    training_y = mat["training_y"] # (14607, 1)
    validation_images = mat["training_images"] # (24, 24, 3, 14607)
    validation_y = mat["training_y"] # (14607, 1)

    ## dataset preparation
    imageSize = training_images.shape
    # every pixel is a feature so the number of features is:
    # height x width x color channels
    numFeatures = imageSize[0]*imageSize[1]*imageSize[2]
    training_x = training_images.reshape(numFeatures, training_images.shape[3]).T.astype(float)
    validation_x = validation_images.reshape(numFeatures, validation_images.shape[3]).T.astype(float)
    test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

    # the training will progress much better if we
    # normalize the features
    meanTrain = np.mean(training_x, axis=0).reshape(1,-1)
    stdTrain = np.std(training_x, axis=0).reshape(1,-1)

    training_x = training_x - np.tile(meanTrain, (training_x.shape[0], 1))
    training_x = training_x / np.tile(stdTrain, (training_x.shape[0], 1))

    validation_x = validation_x - np.tile(meanTrain, (validation_x.shape[0], 1))
    validation_x = validation_x / np.tile(stdTrain, (validation_x.shape[0], 1))

    test_x = test_x - np.tile(meanTrain, (test_x.shape[0], 1))
    test_x = test_x / np.tile(stdTrain, (test_x.shape[0], 1))

    ## training linear regression model
    #-------------------------------------------------------------------#
    # TODO: Select values for the learning rate (mu), batch size
    # (batch_size) and number of iterations (num_iterations), as well as
    # initial values for the model parameters (Theta) that will result in
    # fast training of an accurate model for this classification problem.
    mu = 0.0001

    batch_size = 500

    Theta = np.zeros((training_x.shape[1]+1,1))#0.2*np.random.rand(training_x.shape[1]+1, 1)

    num_iterations = 50
    #-------------------------------------------------------------------#

    xx = np.arange(num_iterations)
    loss = np.empty(*xx.shape)
    loss[:] = np.nan
    validation_loss = np.empty(*xx.shape)
    validation_loss[:] = np.nan
    g = np.empty(*xx.shape)
    g[:] = np.nan

    fig = plt.figure(figsize=(8,8))
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (average per sample)')
    ax2.set_title('mu = '+str(mu))
    h1, = ax2.plot(xx, loss, linewidth=2) #'Color', [0.0 0.2 0.6],
    h2, = ax2.plot(xx, validation_loss, linewidth=2) #'Color', [0.8 0.2 0.8],
    ax2.set_ylim(0, 0.7)
    ax2.set_xlim(0, num_iterations)
    ax2.grid()

    text_str2 = 'iter.: {}, loss: {:.3f}, val. loss: {:.3f}'.format(0, 0, 0)
    txt2 = ax2.text(0.3, 0.95, text_str2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)

    for k in np.arange(num_iterations):
        # pick a batch at random
        idx = np.random.randint(training_x.shape[0], size=batch_size)

        training_x_ones = util.addones(training_x[idx,:])
        validation_x_ones = util.addones(validation_x)

        # the loss function for this particular batch
        loss_fun = lambda Theta: cad.lr_nll(training_x_ones, training_y[idx], Theta)

        # gradient descent
        # instead of the numerical gradient, we compute the gradient with
        # the analytical expression, which is much faster
        Theta_new = Theta - mu*cad.lr_agrad(training_x_ones, training_y[idx], Theta).T

        loss[k] = loss_fun(Theta_new)/batch_size
        validation_loss[k] = cad.lr_nll(validation_x_ones, validation_y, Theta_new)/validation_x.shape[0]

        # visualize the training
        h1.set_ydata(loss)
        h2.set_ydata(validation_loss)
        text_str2 = 'iter.: {}, loss: {:.3f}, val. loss={:.3f} '.format(k, loss[k], validation_loss[k])
        txt2.set_text(text_str2)

        Theta = None
        Theta = np.array(Theta_new)
        Theta_new = None
        tmp = None

        display(fig)
        clear_output(wait = True)
        plt.pause(.005)
