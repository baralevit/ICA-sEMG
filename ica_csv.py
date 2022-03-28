##############################
###### Import libraries ######
##############################
from ica_analysis import ica #ICA analysis script (contains time plot, heatmap and spectral density plots)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



##################################
###### User required inputs ######
##################################

#csv file path
csv_path="C:/Users/User/Filtered Signals.csv"

#sampling rate for the EMG signals
sampling_rate = 250  # needed for the time plot and spectral analysis

# specify the options you want
heatmap_bool = True  # to display heatmap on timage
# if heatmap_bool is set to True you need to specify the image path
if heatmap_bool==True:
    image_path= r"C:/Users/User/Heatmap Picture.jpg"

hscroll_bool = True  # enables horizontal scroll in the time plot


annotations_bool = True  # to display annotations in time plot
# if annotations_bool is set to True you need to specify the annotations signifing the start and end of the recording
# unless specified otherwise the default is 'Recording Started' and 'Ended by DAU'
start_annotations = 'Recording Started'
end_annotations = 'Ended by DAU'
#annotations file path
annotations_path="C:/Users/User/Annotations.csv"


############################
###### Load CSV file #######
############################

df=pd.read_csv(csv_path)

###########################
###### Get file info ######
###########################

signal_labels = list(df)  # label of the channels/signals

set_seconds = df.shape[0] / sampling_rate  # time of the set, in seconds

#####################################################################
###### Obtain annotations, if annotations_bool is set to True #######
#####################################################################

if (annotations_bool == True):

    annotations = pd.read_csv(annotations_path).to_numpy()
    set_annotations, set_times = annotations[:,0], annotations[:,1]

else:
    set_annotations, set_times=[],[]

#########################################
###### convert df to a numpy array ######
#########################################
sigbufs = df.to_numpy().T


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


plt.close('all')
