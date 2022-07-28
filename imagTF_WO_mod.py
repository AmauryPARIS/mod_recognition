# Imported from Jupiter Notebook
import datetime as dt
from sigmf import SigMFFile, sigmffile
import os, sys, collections
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from PIL import Image 
import cmath
import tensorflow as tf

# tf.app.flags.DEFINE_enum('label_format', 'png', ['png'],
#                          'Segmentation label format.')

################################################################################################
#                                       IMPORTANT PATH
################################################################################################
path_local = "/root/spawc21_wideband_dataset/train" 
path_veterine = "/app/data/Modulation_recognition/spawc21_wideband_dataset/train"
path_save_png_veterine = "/app/data/Modulation_recognition/spawc21_wideband_dataset/PNG_"
path_save_tf_examples_train = "/app/data/Modulation_recognition/spawc21_wideband_dataset/tfrecords/train"
path_save_tf_examples_val = "/app/data/Modulation_recognition/spawc21_wideband_dataset/tfrecords/val"
filenames = [fn for fn in os.listdir(path_veterine) if fn.endswith("data")]
file_count = len(filenames)
count = 0
path = path_veterine


################################################################################################
#                                       ID SWITCH
################################################################################################
switcher_id_mod = {
    0:"NOISE",
    1:"SIG"
}

switcher_mod_id = {
    "NOISE":0,
    "PSK2":1,
    "QAM64":1,
    "OFDM":1,
    "QAM16":1,
    "QAM256":1,
    "PSK4":1,
    "FSK2":1,
    "AM_DSB":1,
    "AM_SSB":1,
    "OOK":1,
    "FM":1,
    "GMSK":1,
    "FSK4":1,
    "PSK8":1
}

switcher_id_color = {
    0:[0, 0, 0],
    1:[250, 250, 250]
}


################################################################################################
#                                       Various functions
################################################################################################
def _int64_list_feature(values):
  """Returns a TF-Feature of int64_list.
  Args:
    values: A scalar or an iterable of integer values.
  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.
  Args:
    values: A string.
  Returns:
    A TF-Feature.
  """
  if isinstance(values, str):
    values = values.encode()

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(values):
  """Returns a TF-Feature of bytes.
  Args:
    values: A string.
  Returns:
    A TF-Feature.
  """

  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[values.numpy()]))

def image_to_tfexample(image_data, filename, height, width, seg_data):
  """Converts one image/segmentation pair to tf example.

  Args:
    image_data: string of image data.
    filename: image filename.
    height: image height.
    width: image width.
    seg_data: string of semantic segmentation data.

  Returns:
    tf example of one image/segmentation pair.
  """
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _float_list_feature(image_data),
      'image/filename': _bytes_list_feature(filename),
      'image/format': _bytes_list_feature('raw'),
      'image/channels': _int64_list_feature(3),
      'image/height': _int64_list_feature(height),
      'image/width': _int64_list_feature(width),
      'image/segmentation/class/encoded': (
          _float_list_feature(seg_data)),
      'image/segmentation/class/format': _bytes_list_feature('raw'),
  }))

"""
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_list_feature(image_data),
      'image/filename': _bytes_list_feature(filename),
      'image/format': _bytes_list_feature(
          _IMAGE_FORMAT_MAP[FLAGS.image_format]),
      'image/height': _int64_list_feature(height),
      'image/width': _int64_list_feature(width),
      'image/channels': _int64_list_feature(3),
      'image/segmentation/class/encoded': (
          _bytes_list_feature(seg_data)),
      'image/segmentation/class/format': _bytes_list_feature(
          FLAGS.label_format),
  }))
"""


################################################################################################
#                                       Parameters
################################################################################################
RESOLUTION_FREQ = 2048 # Size of the FFT
OVERLAP_FACTOR = 2 # Define the sample step forward at each FFT iteration 
OVERLAP = int(RESOLUTION_FREQ / OVERLAP_FACTOR)
# SUB_IMAGE_COUNT = 10000 # Number of sub image to create, define the number of sample per sub image
SAMPLE_PER_SUB_IMAGE = 600 #int(file_sample_count / SUB_IMAGE_COUNT)
CREATE_PNG = True # Create PNG to vizualize TF image and label
PANOPTIC_LABEL_DIVISOR = 20 # Could be any integer > 1 (or more like > num_classes)
LIST_VAL_FILES = [10, 24, 28, 32, 35, 39, 43, 55, 58, 59, 63, 68, 69, 72, 78, 83, 88, 101, 104, 107, 118, 128] # TO BE UPDATED IF WORKING WITH SUBSET OF DATASET

################################################################################################
#                                       Image computation
################################################################################################
for filename in ["west-wideband-modrec-ex59-tmpl15-20.04.sigmf-meta"]: #filenames: 

    print("File %s / %s : %s" % (count, file_count, filename))

    #########################################
    # General information and data extraction
    signal = sigmffile.fromfile(os.path.join(path, filename))
    file_sample_count = signal.sample_count
    samples = signal.read_samples(0,file_sample_count)
    annotations = signal.get_annotations()
    file_id = filename[filename.find("-ex")+3:filename.find("-tmp")]

    print("Building Time/Freq image - Freq res : %s, overlap : %s, len sub image : %s\n" % (RESOLUTION_FREQ, OVERLAP, SAMPLE_PER_SUB_IMAGE))

    ###############
    # Set TF writer
    if int(file_id) in LIST_VAL_FILES:
        output_filename = os.path.join(
            path_save_tf_examples_val,
            '%s_%s#%s#%s_val.tfrecord' % (file_id, RESOLUTION_FREQ, OVERLAP_FACTOR, SAMPLE_PER_SUB_IMAGE))
        print("Val TF records")
    else:
        output_filename = os.path.join(
            path_save_tf_examples_train,
            '%s_%s#%s#%s_train.tfrecord' % (file_id, RESOLUTION_FREQ, OVERLAP_FACTOR, SAMPLE_PER_SUB_IMAGE))
        print("Train TF records")

    #################################
    # FFT with sliding window on data
    freq, time, Zxx = sig.stft(samples, nperseg=RESOLUTION_FREQ, noverlap=RESOLUTION_FREQ - OVERLAP, return_onesided=False)

    # Normalize freq axis [0, etc, 0.499, -0.5, etc, -0.0009] to [-0.5, etc, -0.00097, 0, etc, 0.4999]
    Zxx = np.concatenate([Zxx[int(RESOLUTION_FREQ/2):RESOLUTION_FREQ], Zxx[0:int(RESOLUTION_FREQ/2)]])
    freq = np.concatenate([freq[int(RESOLUTION_FREQ/2):RESOLUTION_FREQ], freq[0:int(RESOLUTION_FREQ/2)]])

    # Compute
    real = np.real(Zxx)
    imag = np.imag(Zxx)
    power = np.sqrt(np.real(Zxx)**2 + np.imag(Zxx)**2)
    real = (real/power + 1) * 127.5
    imag = (imag/power + 1) * 127.5
    power_db = 10*np.log10(power)
    power_db = (np.maximum(power_db, -80) + 80)*(255/80)
    
    if CREATE_PNG:
        # Normalize to [0:256] UINT8 - ONLY FOR PNG FORMAT
#         max_real = np.max(real)
#         max_imag = np.max(imag)
#         max_power_db = np.max(power_db)
#         real_uint8 = ((real + 1) * 127).astype(np.uint8)
#         imag_uint8 = ((imag + 1) * 127).astype(np.uint8)
#         power_db_uint8 = (power_db * 256).astype(np.uint8)

        # recursively create directories if necessary
        if not os.path.exists(os.path.join(path_save_png_veterine, file_id)):
            os.makedirs(os.path.join(path_save_png_veterine, file_id))  

    print("Data extraction done - len Zxx : %s / %s" % (len(Zxx), len(Zxx[0])))

    #################################################
    # Annotations extraction and label image creation 
    Zxx_label_colored = np.zeros((RESOLUTION_FREQ, len(time), 3)).astype(np.uint8)
    Zxx_label = np.zeros((RESOLUTION_FREQ, len(time))).astype(np.uint8)

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

        mod_id = switcher_mod_id.get(modulation.upper())

        print("Mod %s : samp start %s # %s - samp count %s # %s - freq min %s # %s - freq max %s # %s" % (
            mod_id,
            sample_start, sample_start_res,
            sample_count, sample_count_res,
            freq_min, freq_min_res,
            freq_max, freq_max_res
        ))

        # Update label image
        Zxx_label[freq_min_res:freq_max_res,sample_start_res:min(sample_start_res + sample_count_res, len(time))] = np.uint8(mod_id)
        Zxx_label_colored[freq_min_res:freq_max_res,sample_start_res:min(sample_start_res + sample_count_res, len(time))] = [np.uint8(item) for item in switcher_id_color.get(mod_id)]
#         for i in range(freq_min_res, freq_max_res):
#             for y in range(sample_start_res, min(sample_start_res + sample_count_res, len(time))):
#                 Zxx_label[i][y] = np.uint8(mod_id)
#                 Zxx_label_colored[i][y] = [np.uint8(item) for item in switcher_id_color.get(mod_id)]

    print("Complete label image build")

    ###############################################################################
    # Build each sub image and label of size RESOLUTION_FREQ * SAMPLE_PER_SUB_IMAGE

    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
        for i in range(0, len(time), SAMPLE_PER_SUB_IMAGE):

            sub_Zxx_label_colored = [] # Samples for PNG label
            sub_Zxx_label = [] # Sample for TF label
#             sub_Zxx_uint8 = [] # Sample for PNG of signal
            sub_Zxx = [] # Sample for TF signal

            # Always create images of the same size, allowing for overlap between the last two
            if i+SAMPLE_PER_SUB_IMAGE <= len(time):
                imin = i
                imax = i + SAMPLE_PER_SUB_IMAGE
            else:
                imin = len(time) - SAMPLE_PER_SUB_IMAGE
                imax = len(time)

            sub_Zxx = np.dstack((real[:, imin:imax], imag[:, imin:imax], power_db[:, imin:imax]))
            sub_Zxx_label = Zxx_label[:, imin:imax]

            if CREATE_PNG:
#                 sub_Zxx_uint8 = np.dstack((real_uint8[:,imin:imax], imag_uint8[:,imin:imax], power_db_uint8[:,imin:imax]))
                sub_Zxx_label_colored = Zxx_label_colored[:,imin:imax]

            image_name = 'TF_%s_%s_RES_%s_%s_file_%s.png' % (i, min(i + SAMPLE_PER_SUB_IMAGE, len(time)), RESOLUTION_FREQ, SAMPLE_PER_SUB_IMAGE, file_id)

            if CREATE_PNG:
                # Image with TF signal
#                 im = Image.fromarray(np.asarray(sub_Zxx_uint8), "RGB")
#                 im.save(os.path.join(path_save_png_veterine, file_id, image_name))
                im = Image.fromarray((np.array(sub_Zxx)).astype(dtype=np.uint8))                
                im.save(os.path.join(path_save_png_veterine, file_id, image_name))

                # Image with label
                im_label = Image.fromarray(np.asarray(sub_Zxx_label_colored), "RGB")
                im_label.save(os.path.join(path_save_png_veterine, file_id, "LABEL_%s" % (image_name)))
            
            # Go from semantic labelling to panoptic
            sub_Zxx_label = tf.constant(np.array(sub_Zxx_label),tf.int32) * PANOPTIC_LABEL_DIVISOR
            # TF Example
            example = image_to_tfexample(tf.io.serialize_tensor(np.array(sub_Zxx)), image_name, RESOLUTION_FREQ, SAMPLE_PER_SUB_IMAGE, tf.io.serialize_tensor(sub_Zxx_label))
            tfrecord_writer.write(example.SerializeToString())
            if CREATE_PNG:
                print("Sub image TF %s - Data : %s - Label %s" % (i, im, im_label))

        print("Plot done")
        count += 1
        
