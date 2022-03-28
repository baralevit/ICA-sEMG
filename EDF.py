##############################
###### Import libraries ######
##############################
from ica_analysis import ica #ICA analysis script (contains time plot, heatmap and spectral density plots
import pyedflib #to convert EDF file to a numpy array, install 'pyEDFlib'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



##################################
###### User required inputs ######
##################################

#edf file path
edf_path=r"C:/Users/User/EDFfile.edf"

#sampling rate for the EMG signals
sampling_rate = 250  # needed for the time plot and spectral analysis

#number of sets to split the data into
num_sets=1

# specify the options you want
heatmap_bool = True  # to display heatmap on timage
# if heatmap_bool is set to True you need to specify the image path
if heatmap_bool==True:
    image_path= r"C:/Users/User/picture.jpg"

hscroll_bool = False  # enables horizontal scroll in the time plot


annotations_bool = False  # to display annotations in time plot
# if annotations_bool is set to True you need to specify the annotations signifing the start and end of the recording
# unless specified otherwise the default is 'Recording Started' and 'Ended by DAU'
start_annotations = 'Recording Started'
end_annotations = 'Ended by DAU'

############################
###### Load EDF file #######
############################

f = pyedflib.EdfReader(edf_path)

########################################
###### Split the file (if needed) ######
########################################

n = f.signals_in_file  # number of channels/signals
signal_labels = f.getSignalLabels()  # label of the channels/signals

# size_limit is the size (number of samples) of the EDF file
size_limit = int(f.getNSamples()[0] / num_sets)

set_seconds = size_limit / sampling_rate  # time of the set, in seconds

#####################################################################
###### Obtain annotations, if annotations_bool is set to True #######
#####################################################################

if (annotations_bool == True):
    annotations = f.readAnnotations()
    # retrieve desired annotations and their times, based on a string condition
    start_index, end_index = int(np.where(annotations[2] == start_annotations)[0]) + 1, int(
        np.where(annotations[2] == end_annotations)[0])
    relevant_annotations, relevant_times = annotations[2][start_index:end_index], annotations[0][start_index:end_index]

for set in range(num_sets):

    ##########################################
    ###### convert EDF to a numpy array ######
    ##########################################
    sigbufs = np.zeros((n, size_limit))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i, start=set * size_limit, n=size_limit)

    #######################################################################
    ###### obtain the annotations and their times for the curent set ######
    #######################################################################

    if (annotations_bool == True):
        condition = (relevant_times <= set_seconds * (set + 1)) & (relevant_times >= set * set_seconds)
        set_times = relevant_times[condition] - set_seconds * set
        set_indexes = np.where(condition)[0]
        set_annotations = relevant_annotations[set_indexes]

    # run ICA to display time plot, spectral analysis, and heatmap (optional)

    # if heatmap_bool==True and annotations_bool==True

    if (heatmap_bool == True and annotations_bool == True):

        ica(sigbufs, sampling_rate, heatmap_bool, image_path, hscroll_bool, annotations_bool, set_annotations,
            set_times)


    # if heatmap_bool==False and annotations_bool==True

    elif (heatmap_bool == False and annotations_bool == True):

        ica(sigbufs, sampling_rate, heatmap_bool, hscroll_bool=hscroll_bool, annotations_bool=annotations_bool,
            set_annotations=set_annotations,

            set_times=set_times)

    # if heatmap_bool==True and annotations_bool==False
    elif (heatmap_bool == True and annotations_bool == False):
        ica(sigbufs, sampling_rate, heatmap_bool, image_path, hscroll_bool, annotations_bool)

    # if heatmap_bool==False and annotations_bool==False

    elif (heatmap_bool == False and annotations_bool == False):

        ica(sigbufs, sampling_rate, heatmap_bool, hscroll_bool=hscroll_bool, annotations_bool=annotations_bool)

    ####################################
    ###### convert to a dataframe ######
    ####################################
    # df_signals = pd.DataFrame(sigbufs.T, columns=signal_labels)  # create dataframe for signals
    # df_annotations=pd.DataFrame(np.stack((set_annotations,set_times),axis=1), columns=["Annotation", "Time [sec]"])  # create dataframe for annotations+annotation times

    #########################
    ###### save as csv ######
    #########################
    # to save a dataset as a csv uncomment the following lines

    # if set==#choose the dataset(s) you want saved as csv#:
    # df_signals.to_csv("Signal dataset %s.csv" % set, index=False)
    # df_annotations.to_csv("Annotations dataset %s.csv" % set, index=False)

    plt.close('all')
