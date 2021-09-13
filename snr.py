import datetime as dt
from sigmf import SigMFFile, sigmffile
import os, sys
import numpy as np

f = open("log_snr.txt", "w")
metadata = []
path = "/root/spawc21_wideband_dataset/train"
filenames = [fn for fn in os.listdir(path) if fn.endswith("data")]
file_count = len(filenames)
count = 0

switcher_id_mod = {
    0:"PSK8",
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
    14:"NOISE"
}

switcher_mod_id = {
    "PSK8":0,
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
    "NOISE":14
}

print("# Get metadata\n")

for filename in filenames:
    # print(filename)
    # filename =  os.path.splitext(os.path.join(path, filename))[0]
    # print(filename)
    print("File %s / %s : %s" % (count, file_count, filename))
    f.write("File %s / %s : %s\n" % (count, file_count, filename))

    #filename = '/root/spawc21_wideband_dataset/train/west-wideband-modrec-ex1-tmpl2-20.04.sigmf-meta' # extension is optional

    signal = sigmffile.fromfile(os.path.join(path, filename))
    file_sample_count = signal.sample_count
    samples = signal.read_samples(0,file_sample_count)
    annotations = signal.get_annotations()
    file_id = filename[-26:-11]
    signal = 0

    mod_power_dict = {14 : [0]}

    signal_mod_map = []
    for i in range(0, file_sample_count):
        signal_mod_map.append([14])
    signal_energy = np.zeros(file_sample_count)

    for annotation in annotations:
        signal += 1
        modulation = annotation[SigMFFile.DESCRIPTION_KEY]
        sample_start = annotation[SigMFFile.START_INDEX_KEY]
        sample_count = annotation[SigMFFile.LENGTH_INDEX_KEY]

        mod_power_dict[switcher_mod_id.get(modulation.upper())] = []

        for i in range(sample_start, sample_start + sample_count):
            if signal_mod_map[i] == [14]:
                signal_mod_map[i] = [switcher_mod_id.get(modulation.upper())]
            else:
                signal_mod_map[i].append(switcher_mod_id.get(modulation.upper()))

    
    signal_energy = np.power(np.power(np.real(samples),2) + np.power(np.imag(samples),2), 0.5)

    

    for i in range(0,file_sample_count-1):
        for mod in signal_mod_map[i]:
            mod_power_dict[mod].append(signal_energy[i])


    if len(mod_power_dict[14]) != 0:
        mean_noise = np.mean(mod_power_dict[14])
        dB_noise = 10*np.log10(mean_noise)
        min_val_noise = np.min(mod_power_dict[14])
        max_val_noise = np.max(mod_power_dict[14])
        variance_noise = np.var(mod_power_dict[14])
        ecarttype_noise = np.std(mod_power_dict[14])

        print("%s,NOISE,%s,%s,%s,%s,%s,%s" % (file_id, mean_noise, dB_noise, min_val_noise, max_val_noise, variance_noise, ecarttype_noise))
        f.write("%s,NOISE,%s,%s,%s,%s,%s,%s\n" % (file_id, mean_noise, dB_noise, min_val_noise, max_val_noise, variance_noise, ecarttype_noise))
 
    for mod in mod_power_dict.keys():
        if len(mod_power_dict[mod]) != 0 and mod != 14:
            mean = np.mean(mod_power_dict[mod])
            dB = 10*np.log10(mean)
            min_val = np.min(mod_power_dict[mod])
            max_val = np.max(mod_power_dict[mod])
            variance = np.var(mod_power_dict[mod])
            ecarttype = np.std(mod_power_dict[mod])

            print("%s,%s,%s,%s,%s,%s,%s,%s,%s" % (file_id, switcher_id_mod.get(mod), mean, dB, min_val, max_val, variance, ecarttype, dB - dB_noise ))
            f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (file_id, switcher_id_mod.get(mod), mean, dB, min_val, max_val, variance, ecarttype, dB - dB_noise ))




 
    count += 1 


f.close()