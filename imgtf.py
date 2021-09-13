import datetime as dt
from sigmf import SigMFFile, sigmffile
import os, sys
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from PIL import Image 
import cmath

#f = open("log_imgTF.txt", "w")

path = "/root/spawc21_wideband_dataset/train"
filenames = [fn for fn in os.listdir(path) if fn.endswith("data")]
file_count = len(filenames)
count = 0

switcher_id_mod = {
    0:"NOISE",
    1:"PSK2",
    2:"QAM64",
    3:"OFDM",
    4:"QAM16",
    5:"QAM256",
    6:"PSK4",
    7:"FSK2",
    8:"AM_DSB",
    9:"AM_SSB",
    10:"OOK",
    11:"FM",
    12:"GMSK",
    13:"FSK4",
    14:"PSK8"
}

switcher_mod_id = {
    "NOISE":0,
    "PSK2":1,
    "QAM64":2,
    "OFDM":3,
    "QAM16":4,
    "QAM256":5,
    "PSK4":6,
    "FSK2":7,
    "AM_DSB":8,
    "AM_SSB":9,
    "OOK":10,
    "FM":11,
    "GMSK":12,
    "FSK4":13,
    "PSK8":14
}

switcher_id_color = {
    0:[0, 0, 0],
    1:[231, 76, 60],
    2:[155, 89, 182],
    3:[142, 68, 173],
    4:[41, 128, 185],
    5:[52, 152, 219],
    6:[26, 188, 156],
    7:[34, 153, 84],
    8:[29, 131, 72],
    9:[241, 196, 15],
    10:[243, 156, 18],
    11:[202, 111, 30],
    12:[186, 74, 0],
    13:[49, 118, 247],
    14:[236, 8, 21]
}

######################
# Important parameters
RESOLUTION_FREQ = 2048 # Size of the FFT
OVERLAP_FACTOR = 2 # Define the sample step forward at each FFT iteration 
OVERLAP = int(RESOLUTION_FREQ / OVERLAP_FACTOR)
SUB_IMAGE_COUNT = 10000 # Number of sub image to create, define the number of sample per sub image

for filename in ["west-wideband-modrec-ex59-tmpl15-20.04.sigmf-data"]: #in filenames: 

    print("File %s / %s : %s" % (count, file_count, filename))

    #########################################
    # General information and data extraction
    signal = sigmffile.fromfile(os.path.join(path, filename))
    file_sample_count = signal.sample_count
    SAMPLE_PER_SUB_IMAGE = int(file_sample_count / SUB_IMAGE_COUNT)
    samples = signal.read_samples(0,file_sample_count)
    annotations = signal.get_annotations()
    file_id = filename[filename.find("-ex")+3:filename.find("-tmp")]

    print("Building Time/Freq image - Freq res : %s, overlap : %s, len sub image : %s\n" % (RESOLUTION_FREQ, OVERLAP, SAMPLE_PER_SUB_IMAGE))

    #################################
    # FFT with sliding window on data
    freq, time, Zxx = sig.stft(samples, nperseg = RESOLUTION_FREQ, noverlap = RESOLUTION_FREQ - OVERLAP, return_onesided = False)
    max_real = np.max(np.real(Zxx))
    max_imag = np.max(np.imag(Zxx))
    max_power_db = np.max(np.power(np.power(np.real(Zxx),2) + np.power(np.imag(Zxx),2), 0.5))
    
    # Normalize freq axis [0, etc, 0.499, -0.5, etc, -0.0009] to [-0.5, etc, -0.00097, 0, etc, 0.4999]
    Zxx = np.concatenate([Zxx[int(RESOLUTION_FREQ/2):RESOLUTION_FREQ],Zxx[0:int(RESOLUTION_FREQ/2)]])
    freq = np.concatenate([freq[int(RESOLUTION_FREQ/2):RESOLUTION_FREQ],freq[0:int(RESOLUTION_FREQ/2)]])

    print("Data extraction done - len Zxx : %s / %s" % (len(Zxx), len(Zxx[0])))

    #################################################
    # Annotations extraction and label image creation 
    Zxx_label = np.zeros((RESOLUTION_FREQ, len(time), 3)).astype(np.uint8)
    for annotation in annotations:
        # Extract
        modulation = annotation[SigMFFile.DESCRIPTION_KEY]
        sample_start = annotation[SigMFFile.START_INDEX_KEY]
        sample_count = annotation[SigMFFile.LENGTH_INDEX_KEY]
        freq_min = annotation[SigMFFile.FLO_KEY]
        freq_max = annotation[SigMFFile.FHI_KEY]

        # Normalize
        sample_start_res = int(sample_start / OVERLAP)
        sample_count_res = int(np.ceil(sample_count / OVERLAP))
        freq_min_res = int((freq_min / 0.5 * (RESOLUTION_FREQ/2)) + (RESOLUTION_FREQ/2))
        freq_max_res = int((freq_max / 0.5 * (RESOLUTION_FREQ/2)) + (RESOLUTION_FREQ/2))
    
        print("Mod %s : samp start %s # %s - samp count %s # %s - freq min %s # %s - freq max %s # %s" % (
            switcher_mod_id.get(modulation.upper()),
            sample_start, sample_start_res,
            sample_count, sample_count_res,
            freq_min, freq_min_res,
            freq_max, freq_max_res
        ))

        # Update label image
        for i in range(freq_min_res, freq_max_res):
            for y in range(sample_start_res , min(sample_start_res + sample_count_res, len(time))):
                Zxx_label[i][y] = [np.uint8(item) for item in switcher_id_color.get(switcher_mod_id.get(modulation.upper()))]
                
       
    print("Complete label image build")
    
    ###############################################################################
    # Build each sub image and label of size RESOLUTION_FREQ * SAMPLE_PER_SUB_IMAGE
    for i in range(0, len(time), SAMPLE_PER_SUB_IMAGE): 
        sub_Zxx_label = []
        sub_Zxx = []
        signal_label = False
        
        for y in range(0, RESOLUTION_FREQ):
            
            sub_samples = Zxx[y][i:(min(i + SAMPLE_PER_SUB_IMAGE, len(time)))]

            # Compute
            real = np.real(sub_samples)
            imag = np.imag(sub_samples)
            power_db = np.power(np.power(np.real(sub_samples),2) + np.power(np.imag(sub_samples),2), 0.5)

            # Normalize
            real = ((real / max_real)*256).astype(np.uint8)
            imag = ((imag / max_imag)*256).astype(np.uint8)
            power_db = ((power_db / max_power_db)*256).astype(np.uint8)

            # Create image/label on all samples in a sub image for one frequency
            sub_Zxx.append(np.dstack((real, imag, power_db))[0])
            sub_Zxx_label.append(Zxx_label[y][i:i+len(sub_samples)])

        # Image with TF signal
        im = Image.fromarray(np.asarray(sub_Zxx), "RGB")
        im.save('TF_%s_%s_RES_%s_%s_file_%s.png' % (i,min( i + SAMPLE_PER_SUB_IMAGE, len(time)), RESOLUTION_FREQ, SAMPLE_PER_SUB_IMAGE, file_id))

        # Image with label 
        im_label = Image.fromarray(np.asarray(sub_Zxx_label), "RGB")
        im_label.save('LABEL_TF_%s_%s_RES_%s_%s_file_%s.png' % (i, min(i + SAMPLE_PER_SUB_IMAGE, len(time)), RESOLUTION_FREQ, SAMPLE_PER_SUB_IMAGE, file_id))

        print("Sub image TF %s - Data : %s - Label %s" % (i, im, im_label))

    print("Plot done")
    count += 1






