##############################
###### Import libraries ######
##############################
import cv2  #for image loading+user electrode location selection, install 'opencv-python'
#for ICA + time plots + heatmap + spectral analysis
from scipy.stats import kurtosis
from picard import picard #*need to cite*,  install "python-picard"
from numpy.linalg import inv
from scipy.interpolate import griddata
import matplotlib.image as mpimg
from matplotlib.pyplot import cm #colormaps
import matplotlib.ticker as ticker
from scipy import signal
import math
from matplotlib.widgets import Slider #for horizontal scrolling in time plot
#generic libraries
import numpy as np
import matplotlib.pyplot as plt

####################################################
###### Global electrode coordinate variables #######
####################################################
# NOTE: if you prefer, for each mask you can obtain the electrode locations only once, save the coordinates as an array and use them directly for future analysis.
# for example, for the pilot picture the x,y coordinates are approximately:
# x_coor=[419, 412, 365, 192, 313, 300, 224, 316, 134, 292, 383, 408, 303, 194, 132, 202]
# y_coor=[705, 623, 838, 827, 773, 691, 613, 596, 523, 551, 527, 424, 257, 251, 220, 167]

# electrode coordinates (empty if need to be obtained)
x_coor = []
y_coor = []



#########################
###### Load image #######
#########################
def image_load(image_path):
    # load the image, write the path where the image is saved (if there is no image uncomment these two lines)
    global img
    img = cv2.imread(image_path, 1)  # for electrode location selection
    image = mpimg.imread(image_path)  # for heatmap

    # image dimensions
    height = img.shape[0]
    width = img.shape[1]

    return image, height, width


#######################################
###### select electrode location ######
#######################################

def click_event(event, x, y, flags, params):
    global x_coor
    global y_coor
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        x_coor.append(x)
        y_coor.append(y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '(' + str(x) + ',' +
                    str(y) + ')', (x, y), font,
                    0.5, (255, 0, 0), 2)
        cv2.imshow('image', img)

    return x_coor, y_coor


def get_location():
    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()


############################################
###### Helper functions for time plot ######
############################################

# for unique annotations in time plot
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ncol = math.ceil(len(unique) / 25)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=ncol, *zip(*unique))


#########################################################
###### ICA: heatmap + time plot + Spectral Density ######
#########################################################

# function for heatmap only
def ica_heatmap(image_path,number_of_channels, W, K, order):
    image, height, width = image_load(image_path)  # load image

    if not x_coor:  # if manually inserted coordinates, there is not need to get the electrode locations (otherwise select electrodes from image)
        get_location()

    # calculations for the heatmap
    inverse = np.absolute(inv(np.matmul(W, K)))

    grid_y, grid_x = np.mgrid[1:height + 1, 1:width + 1]

    points = np.column_stack((x_coor, y_coor))

    f_interpolate = []
    for i in range(number_of_channels):
        f_interpolate.append(griddata(points, inverse[:, i], (grid_x, grid_y), method='linear'))

    # plot heatmap
    fig, axs = plt.subplots(2, int(number_of_channels / 2), figsize=(16, 8))
    axs = axs.ravel()
    # plt.imshow(image)
    for i in range(number_of_channels):
        axs[i].imshow(image)
        axs[i].pcolormesh(f_interpolate[order[i]], cmap='jet', alpha=0.5)
        axs[i].set_title("ICA Source %d" % (i + 1))
        axs[i].axis('off')

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()


# function for time plot only
def ica_time_plot(annotations_bool, hscroll_bool, model, T_plots,set_annotations,  set_times, sampling_rate):
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.set_title('ICA sources')

    # display the x-axis in minutes:seconds
    ax.set_xlabel('time [min:sec]')
    time_func = (lambda a: divmod(int(a / sampling_rate), 60))
    ticks = ticker.FuncFormatter(lambda x, pos: (str(time_func(x)[0]).zfill(2) + ':' + str(time_func(x)[1]).zfill(2)))
    ax.xaxis.set_major_formatter(ticks)

    if (annotations_bool == True):
        # obtain unique annotations and assign them unique colors
        unique_annotations, reverse_indexes = np.unique(set_annotations, return_inverse=True)
        unique_colors = cm.rainbow(np.linspace(0, 1, unique_annotations.shape[0]))

        # add annotations
        for time, annotation, color in zip(set_times, unique_annotations[reverse_indexes],
                                           unique_colors[reverse_indexes]):
            ax.axvline(x=time * sampling_rate, linestyle='--', label=annotation, color=color, linewidth=1.75)

        legend_without_duplicate_labels(ax)

    # calculate the offset
    max_model = np.max(model, axis=1)
    max_model = np.roll(max_model, 1)
    offsets = max_model - np.min(model, axis=1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)

    # plot
    ax.plot((model[:, :T_plots] + offsets[:, np.newaxis]).T, color='blue')

    # display the source number on the y-axis
    y_values = (model[:, :T_plots] + offsets[:, np.newaxis]).mean(axis=1)
    source_label_func = lambda x: "ICA source %d" % (x + 1)
    y_label = np.flip(np.array([source_label_func(source) for source in range(model.shape[0])]))
    plt.yticks(y_values, y_label)

    fig.tight_layout()

    if (hscroll_bool == True):
        # for horiztonal scrolling in time plot
        def update(val):
            pos = slider_position.val
            min_value = np.min((model[0, :T_plots] + offsets[0, np.newaxis]))
            max_value = np.max((model[-1, :T_plots] + offsets[-1, np.newaxis]))
            ax.axis([pos, pos + 60 * sampling_rate, math.floor(min_value), math.ceil(max_value)])
            time = divmod(int(pos / sampling_rate), 60)
            min, sec = time[0], time[1]
            slider_position.valtext.set_text('{:02d}:{:02d}'.format(min, sec))

            fig.canvas.draw_idle()

        # Set the axis and slider position in the plot
        axis_position = plt.axes([0.2, 0, 0.65, 0.03], facecolor='white')

        slider_position = Slider(axis_position, 'Time', 0, T_plots)

        # update function called using on_changed() function
        slider_position.on_changed(update)

    else:
        slider_position = []

    fig.canvas.draw()
    fig.canvas.flush_events()

    return slider_position


# ICA analysis includes the ICA algorithm, time plots, calling the heatmap function, and spectral density plots
def ica(sigbufs, sampling_rate, heatmap_bool=False, image_path=None, hscroll_bool=False, annotations_bool=False, set_annotations=None,  set_times=None):

    n=sigbufs.shape[0]
    size_limit=sigbufs.shape[1]

    K, W, Y = picard(sigbufs, n_components=n, ortho=True, max_iter=200)  # ICA algorithm
    # note: if the algorithm doesn't converge you can increase the 'max_iter' value

    order = np.argsort(kurtosis(Y[:, :size_limit], axis=1))
    model = Y[order]

    # time plot of the ICA signal sources
    keep_sliders = ica_time_plot(annotations_bool, hscroll_bool, model, size_limit, set_annotations,  set_times,sampling_rate)

    order = order[::-1]

    # condition for plotting the heatmap
    if (heatmap_bool == True):
        ica_heatmap(image_path, n, W, K, order)

    # spectral density (Welch's method)

    # smoothed
    window = signal.get_window('hamming', 500)
    f_smooth, Pxx_den_smooth = signal.welch(Y, sampling_rate, window=window, noverlap=250, nperseg=500, nfft=512,
                                            scaling='density')

    # all
    len_window = int(Y.shape[1] / 4)
    window = signal.get_window('hamming', len_window)
    f, Pxx_den = signal.welch(Y, sampling_rate, window=window, nperseg=len_window,
                              nfft=2 ** math.ceil(math.log2(abs(len_window))), scaling='density')

    # plot spectral density
    fig, axs = plt.subplots(2, int(sigbufs.shape[0] / 2), figsize=(16, 8))
    axs = axs.ravel()
    for i in range(16):
        axs[i].plot(f, Pxx_den[order[i], :], color='black')
        axs[i].plot(f_smooth, Pxx_den_smooth[order[i], :], color='blue', linewidth=2)
        axs[i].set_ylim([0, 0.025])
        axs[i].set_xlim([0, 500])
        axs[i].set_xlabel('frequency [Hz]')
        axs[i].set_ylabel('PSD [V**2/Hz]')
        if (i != 0 and i != int(sigbufs.shape[0] / 2)):
            axs[i].get_yaxis().set_visible(False)
        if (i < int(sigbufs.shape[0] / 2)):
            axs[i].get_xaxis().set_visible(False)

    fig.tight_layout()
    plt.show()





