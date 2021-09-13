import datetime as dt
from sigmf import SigMFFile, sigmffile
import os
import numpy as np

metadata = []
path = "/root/spawc21_wideband_dataset/train"
filenames = [fn for fn in os.listdir(path) if fn.endswith("meta")]
file_count = len(filenames)
count = 0

f = open("log_bw_duration.txt", "w")

print("# Get metadata")

for filename in filenames:
    print("File %s / %s : %s" % (count, file_count, filename))

    #filename = '/root/spawc21_wideband_dataset/train/west-wideband-modrec-ex1-tmpl2-20.04.sigmf-meta' # extension is optional
    signal = sigmffile.fromfile(os.path.join(path, filename))
    annotations = signal.get_annotations()
    file_id = filename[-26:-11]
    signal = 0

    for annotation in annotations:
        signal += 1
        modulation = annotation[SigMFFile.DESCRIPTION_KEY]
        freq_min = annotation[SigMFFile.FLO_KEY]
        freq_max = annotation[SigMFFile.FHI_KEY]
        sample_start = annotation[SigMFFile.START_INDEX_KEY]
        sample_count = annotation[SigMFFile.LENGTH_INDEX_KEY]
        metadata.append({"file_id" : file_id, "signal" : signal, "mod" : modulation, "freq_min" : freq_min, "freq_max" : freq_max, "sample_start" : sample_start, "sample_count" : sample_count})
    
    count += 1

print(str(metadata))
f.write(str(metadata))

print("\n \n")
print("####################")
print("# Analyse metadata #")
print("####################")
print("\n")

f.write("\n\n# Analyse metadata #\n\n")


##################################################
print("Modulations BW")
f.write("\n\n# Modulations BW #\n\n")
modulations_bw = {}
for signal in metadata:
    bw = signal["freq_max"] - signal["freq_min"]
    if signal["mod"] in modulations_bw:
        modulations_bw[signal["mod"]]["min"] = np.minimum(modulations_bw[signal["mod"]]["min"], bw)
        modulations_bw[signal["mod"]]["max"] = np.maximum(modulations_bw[signal["mod"]]["max"], bw)
    else:
        modulations_bw[signal["mod"]] = {"min" : bw, "max" : bw}

for mod in modulations_bw:
    list_print = []
    for key in sorted(modulations_bw[mod].keys()):
        list_print.append("%s : %s" % (key, modulations_bw[mod][key]))
    print("%s : %s" % (mod,list_print))
    f.write("%s : %s\n" % (mod,list_print))




##################################################
print("Modulations sample count")
f.write("\n\n# Modulations sample count #\n\n")
modulations_sample_count = {}
for signal in metadata:
    sample_count = signal["sample_count"]
    if signal["mod"] in modulations_sample_count:
        modulations_sample_count[signal["mod"]]["min"] = np.minimum(modulations_sample_count[signal["mod"]]["min"], sample_count)
        modulations_sample_count[signal["mod"]]["max"] = np.maximum(modulations_sample_count[signal["mod"]]["max"], sample_count)
    else:
        modulations_sample_count[signal["mod"]] = {"min" : sample_count, "max" : sample_count}

for mod in modulations_sample_count:
    list_print = []
    for key in sorted(modulations_sample_count[mod].keys()):
        list_print.append("%s : %s" % (key, modulations_sample_count[mod][key]))
    print("%s : %s" % (mod,list_print))
    f.write("%s : %s\n" % (mod,list_print))
print("\n")


##################################################
print("Modulations sample count")
f.write("\n\n# plot BW / duration #\n\n")
for signal in metadata:
    print("%s,%s,%s,%s" % (signal["mod"],signal["freq_max"] - signal["freq_min"], signal["sample_count"], signal["file_id"]))
    f.write("%s,%s,%s,%s\n" % (signal["mod"],signal["freq_max"] - signal["freq_min"], signal["sample_count"], signal["file_id"]))
    
print("\n")



f.close()