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
    sort_ix_high = sort_ix[-montage_n:] # Get the 300 largest

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

    training_x_ones = util.addones(training_x)

    Theta, E_train = reg.ls_solve(training_x_ones, training_y)

    predicted_y = util.addones(test_x).dot(Theta)


    ## quadratic regression
    training_x_ones_quad = util.addones(np.concatenate((training_x,np.square(training_x)),axis=1))

    Theta_quad , E_train_quad = reg.ls_solve(training_x_ones_quad, training_y)

    predicted_y_quad = util.addones(np.concatenate((test_x,np.square(test_x)),axis=1)).dot(Theta_quad)

    # Error

    E = np.transpose(predicted_y - test_y).dot(predicted_y - test_y)/len(predicted_y)
    E_quad = np.transpose(predicted_y_quad - test_y).dot(predicted_y_quad - test_y)/len(predicted_y_quad)

    #---------------------------------------------------------------------#

    # visualize the results
    fig2 = plt.figure(figsize=(16,8))
    ax1 = fig2.add_subplot(221)
    data1, = ax1.plot(predicted_y, test_y, ".g", markersize=1)# , alpha=0.5)
    data1.set_label("Data")
    ax1.grid()
    ax1.set_xlabel('Area')
    ax1.set_ylabel('Predicted Area')
    ax1.set_title('1st order linear regression')
    error_str_1 = "Error (n-norm) = " + str(round(E.item(0),2))
    txt_1 = ax1.text(0.7, 0.08, error_str_1, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax1.transAxes)

    ax2 = fig2.add_subplot(222)
    data2, = ax2.plot(predicted_y_quad, test_y, ".g", markersize=1)# , alpha=0.5)
    data2.set_label("Data")
    ax2.grid()
    ax2.set_xlabel('Area')
    ax2.set_ylabel('Predicted Area')
    ax2.set_title('2nd order linear regression')
    error_str_2 = "Error (n-norm) = " + str(round(E_quad.item(0),2))
    txt_2 = ax2.text(0.7, 0.08, error_str_2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)

    #training with smaller number of training samples
    #---------------------------------------------------------------------#
    # Train a model with reduced dataset size (e.g. every fourth
    # training sample).

    N=4;
    ix = np.random.randint(len(training_x), size=round(len(training_x)/N))  # Random indices-list with length set/4
    d_training_x = training_x[ix, :]
    d_training_y = training_y[ix, :]

    d_training_x_ones = util.addones(d_training_x)

    d_Theta, d_E_train = reg.ls_solve(d_training_x_ones, d_training_y)

    d_predicted_y = util.addones(test_x).dot(d_Theta)

    # quadratic regression with down-sampled data
    d_training_x_ones_quad = util.addones(np.concatenate((d_training_x, d_training_x**2),axis=1))

    d_Theta_quad, E_train = reg.ls_solve(d_training_x_ones_quad, d_training_y)

    d_predicted_y_quad = util.addones(np.concatenate((test_x, test_x ** 2),axis=1)).dot(d_Theta_quad)

    # Error

    d_E = np.transpose(d_predicted_y - test_y).dot(d_predicted_y - test_y) / len(d_predicted_y)
    d_E_quad = np.transpose(d_predicted_y_quad - test_y).dot(d_predicted_y_quad - test_y) / len(d_predicted_y_quad)

    #---------------------------------------------------------------------#

    # visualize the results
    ax3  = fig2.add_subplot(223)
    data3, = ax3.plot(d_predicted_y, test_y, ".g", markersize=1)# , alpha=0.5)
    data3.set_label("Data")
    ax3.grid()
    ax3.set_xlabel('Area')
    ax3.set_ylabel('Predicted Area')
    ax3.set_title('Downsampled 1st order linear regression')
    error_str_3 = "Error (n-norm) = " + str(round(d_E.item(0), 2))
    txt_3 = ax3.text(0.7, 0.08, error_str_3, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax3.transAxes)

    ax4  = fig2.add_subplot(224)
    data4, = ax4.plot(d_predicted_y_quad, test_y, ".g", markersize=1)# , alpha=0.5)
    data4.set_label("Data")
    ax4.grid()
    ax4.set_xlabel('Area')
    ax4.set_ylabel('Predicted Area')
    ax4.set_title('Downsampled 2nd order linear regression')
    error_str_4 = "Error (n-norm) = " + str(round(d_E_quad.item(0), 2))
    txt_4 = ax4.text(0.7, 0.08, error_str_4, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax4.transAxes)

    # Extend plots with y=x line and legend for readability purposes
    lineplotx = [-500, 500]
    lineploty = [-500, 500]
    for ax in [ax1,ax2,ax3,ax4]:

        # Hold current axis sizes (so lines don't screw up scatter auto-scale)
        xmin, xmax, ymin, ymax = ax.axis()
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

        #Plot y = x line
        line, = ax.plot(lineplotx, lineploty, 'r--')
        line.set_label("y=x (illustrative)")

        ax.legend(loc="upper left", prop={'size': 12})

    plt.tight_layout()


def nuclei_classification(batchsize=80,muinitial=0.0004, numiterations=50, order=1,downsample_factor=1):
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

    # Setup datasets
    training_x = training_images.reshape(numFeatures, training_images.shape[3]).T.astype(float)
    validation_x = validation_images.reshape(numFeatures, validation_images.shape[3]).T.astype(float)
    test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

    # Append datasets with n-order features if necessary
    if order == 1:
        pass
    else:
        for i in range(int(order)):
            training_x = np.concatenate((training_x, training_x ** (i+1)), axis=1)
            validation_x = np.concatenate((validation_x, validation_x ** (i + 1)), axis=1)
            test_x = np.concatenate((test_x, test_x ** (i + 1)), axis=1)

    # Setup downsampling indices
    if downsample_factor == 1:
        pass
    else:
        ix = np.random.randint(len(training_x), size=round(len(training_x)*downsample_factor))
        training_x = training_x[ix, :]

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
    # Select values for the learning rate (mu), batch size
    # (batch_size) and number of iterations (num_iterations), as well as
    # initial values for the model parameters (Theta) that will result in
    # fast training of an accurate model for this classification problem.

    # After iterative prototyping the following batch size (50) proved to be the best
    batch_size = batchsize

    # Initiate learning curve with random non-zero theta
    Theta = 0.0000002*np.random.rand(training_x.shape[1]+1, 1)

    # 50 iterations proved to be more than enough, while yielding decent computing times
    num_iterations = numiterations

    # Define function for variable mu for better learning performance
    mu = muinitial
    fun_mu = lambda k: mu * np.exp(-5 * k / num_iterations)

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
    ax2.set_title("Logistic regression - learning curve")
    h1, = ax2.plot(xx, loss, linewidth=2) #'Color', [0.0 0.2 0.6],
    h2, = ax2.plot(xx, validation_loss, linewidth=2) #'Color', [0.8 0.2 0.8],
    h1.set_label("Training data")
    h2.set_label("Validation data")

    ax2.set_ylim(0.4, 1.0)
    ax2.set_xlim(0, num_iterations)
    ax2.legend(loc='lower left')
    ax2.grid()

    # Added text box for readability
    info_str = 'mu = ' + str(mu) + " * np.exp(-5 * k / " + str(num_iterations) + ")\nbatch size = " + str(batch_size)
    if downsample_factor == 1:
        height = 0.85
    else:
        info_str = info_str + "\ndownsample factor = " + str(downsample_factor)
        height = 0.82
    txt_info = ax2.text(0.55, height, info_str, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)

    text_str2 = 'iter.: {}, loss: {:.3f}, val. loss: {:.3f}'.format(0, 0, 0)
    txt2 = ax2.text(0.55, 0.95, text_str2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)

    STOP = False

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

        Theta_new = Theta - fun_mu(k)*cad.lr_agrad(training_x_ones, training_y[idx], Theta).T

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

        # Stop condition --> validation loss function has been pretty much flat for 10 iterations in a row.
        n = 10
        if k > n+1:
            gradient = [0] * n
            for i in range(n):
                gradient[i] = np.abs(validation_loss[k-1-i]-validation_loss[k-i])

            if np.sum(gradient) < 0.005:
                STOP = True
                ax2.plot(k, validation_loss[k], 'x', color='red', markersize=12)
                ax2.axvline(x=k,color='red')
                display(fig)

        if STOP:
            break

    predicted_labels = (util.addones(test_x).dot(Theta) > 0.5).astype(int)

    num_trues = 0
    for record in range(len(predicted_labels)):
        if predicted_labels[record] == test_y[record]:
            num_trues = num_trues + 1

    accuracy = num_trues / len(predicted_labels)

    print("ACCURACY = ", round(accuracy, 5))

def optimise_nuclei_classification(numiterations=50, order=1):

    print("========== RUN optimise_nuclei_classification =========\n")
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

    # Setup datasets
    training_x = training_images.reshape(numFeatures, training_images.shape[3]).T.astype(float)
    validation_x = validation_images.reshape(numFeatures, validation_images.shape[3]).T.astype(float)
    test_x = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

    # Append datasets with n-order features if necessary
    if order == 1:
        pass
    else:
        for i in range(int(order)):
            training_x = np.concatenate((training_x, training_x ** (i+1)), axis=1)
            validation_x = np.concatenate((validation_x, validation_x ** (i + 1)), axis=1)
            test_x = np.concatenate((test_x, test_x ** (i + 1)), axis=1)

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

    print("Dataset prepared. \nStart regression training...\n")

    ## training linear regression model

    # Initiate learning curve with random non-zero theta
    Theta = 0.0000002*np.random.rand(training_x.shape[1]+1, 1)

    # 50 iterations proved to be more than enough, while yielding decent computing times (will not be optimised over)
    num_iterations = numiterations

    # Set some sample initial mu's
    mu = [0.0002, 0.0003, 0.0004]

    # Set some sample batch sizes
    batch_size = [10, 20, 30, 50, 80]

    print(len(batch_size)*len(mu), " iterables detected.\n\n")
    val_loss_matrix = np.zeros((len(batch_size), len(mu)))

    percentage = 0
    for i in range(len(batch_size)):
        for j in range(len(mu)):
            fun_mu = lambda k: mu[j] * np.exp(-5 * k / num_iterations)

            xx = np.arange(num_iterations)
            loss = np.empty(*xx.shape)
            loss[:] = np.nan
            validation_loss = np.empty(*xx.shape)
            validation_loss[:] = np.nan
            g = np.empty(*xx.shape)
            g[:] = np.nan

            STOP = False

            for k in np.arange(num_iterations):
                # pick a batch at random
                idx = np.random.randint(training_x.shape[0], size=batch_size[i])

                training_x_ones = util.addones(training_x[idx,:])
                validation_x_ones = util.addones(validation_x)

                # the loss function for this particular batch
                loss_fun = lambda Theta: cad.lr_nll(training_x_ones, training_y[idx], Theta)

                # gradient descent
                # instead of the numerical gradient, we compute the gradient with
                # the analytical expression, which is much faster

                Theta_new = Theta - fun_mu(k)*cad.lr_agrad(training_x_ones, training_y[idx], Theta).T

                loss[k] = loss_fun(Theta_new)/batch_size[i]
                validation_loss[k] = cad.lr_nll(validation_x_ones, validation_y, Theta_new)/validation_x.shape[0]

                Theta = None
                Theta = np.array(Theta_new)
                Theta_new = None
                tmp = None

                # Stop condition --> validation loss function has been pretty much flat for 10 iterations in a row.
                n = 10
                if k > n+1:
                    gradient = [0] * n
                    for z in range(n):
                        gradient[z] = np.abs(validation_loss[k-1-z]-validation_loss[k-z])

                    if np.sum(gradient) < 0.005:
                        STOP = True

                if STOP:
                    break
            # Store validation loss in matrix
            val_loss_matrix[i, j] = validation_loss[k]
        percentage = percentage + 100/len(batch_size)
        print(round(percentage, 1), " percent completed...")

    (i_min, j_min) = tuple(np.where(val_loss_matrix == np.min(val_loss_matrix)))
    best_batch_size = batch_size[int(i_min)]
    best_mu = mu[int(j_min)]

    print("\n\nCOMPLETED!")
    print("Optimal batch size = ", best_batch_size)
    print("Optimal initial mu = ", best_mu)
