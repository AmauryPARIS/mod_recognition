import datetime as dt
from sigmf import SigMFFile, sigmffile
import os

metadata = []
path = "/root/spawc21_wideband_dataset/train"
filenames = [fn for fn in os.listdir(path) if fn.endswith("meta")]
file_count = len(filenames)
count = 0

print("# Get metadata")

for filename in filenames[:2]:
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

print(metadata)

print("\n \n")
print("####################")
print("# Analyse metadata #")
print("####################")
print("\n")

##################################################
print("Modulations")
modulations = {}
for signal in metadata:
    if signal["mod"] in modulations:
        modulations[signal["mod"]] += 1
    else:
        modulations[signal["mod"]] = 1
print(modulations)
print("\n")

##################################################
print("Modulations count")
modulation_count = len(modulations)
print(modulation_count)
print("\n")

##################################################
print("Modulations BW")
modulations_bw = {}
for signal in metadata:
    bw = round(signal["freq_max"] - signal["freq_min"], 1)
    if signal["mod"] in modulations_bw:
        if bw in modulations_bw[signal["mod"]]:
            modulations_bw[signal["mod"]][bw] += 1
        else:
            modulations_bw[signal["mod"]][bw] = 1
    else:
        modulations_bw[signal["mod"]] = {bw : 1}

for mod in modulations_bw:
    list_print = []
    for key in sorted(modulations_bw[mod].keys()):
        list_print.append("%s : %s" % (key, modulations_bw[mod][key]))
    print("%s : %s" % (mod,list_print))
print("\n")

##################################################
print("Modulations sample count")
modulations_sample_count = {}
for signal in metadata:
    round_sample_count = round(signal["sample_count"], -4)
    if signal["mod"] in modulations_sample_count:
        if round_sample_count in modulations_sample_count[signal["mod"]]:
            modulations_sample_count[signal["mod"]][round_sample_count] += 1
        else:
            modulations_sample_count[signal["mod"]][round_sample_count] = 1
    else:
        modulations_sample_count[signal["mod"]] = {round_sample_count : 1}

for mod in modulations_sample_count:
    list_print = []
    for key in sorted(modulations_sample_count[mod].keys()):
        list_print.append("%s : %s" % (key, modulations_sample_count[mod][key]))
    print("%s : %s" % (mod,list_print))
print("\n")

##################################################
print("Modulations sample start")
modulations_sample_start = {}
for signal in metadata:
    round_sample_start = round(signal["sample_start"], -7)
    if signal["mod"] in modulations_sample_start:
        if round_sample_start in modulations_sample_start[signal["mod"]]:
            modulations_sample_start[signal["mod"]][round_sample_start] += 1
        else:
            modulations_sample_start[signal["mod"]][round_sample_start] = 1
    else:
        modulations_sample_start[signal["mod"]] = {round_sample_start : 1}

for mod in modulations_sample_start:
    list_print = []
    for key in sorted(modulations_sample_start[mod].keys()):
        list_print.append("%s : %s" % (key, modulations_sample_start[mod][key]))
    print("%s : %s" % (mod,list_print))
print("\n")

##################################################
print("Modulations freq down")
modulations_freq_down = {}
for signal in metadata:
    round_freq_down = round(signal["freq_min"], 1)
    if signal["mod"] in modulations_freq_down:
        if round_freq_down in modulations_freq_down[signal["mod"]]:
            modulations_freq_down[signal["mod"]][round_freq_down] += 1
        else:
            modulations_freq_down[signal["mod"]][round_freq_down] = 1
    else:
        modulations_freq_down[signal["mod"]] = {round_freq_down : 1}

for mod in modulations_freq_down:
    list_print = []
    for key in sorted(modulations_freq_down[mod].keys()):
        list_print.append("%s : %s" % (key, modulations_freq_down[mod][key]))
    print("%s : %s" % (mod,list_print))
print("\n")

##################################################
print("Signal per files")
signals_per_files = {}
for signal in metadata:
    if signal["file_id"] in signals_per_files:
        signals_per_files[signal["file_id"]] += 1
    else:
        signals_per_files[signal["file_id"]] = 1
print(signals_per_files)
print("\n")

##################################################
print("Overlapping")
overlapping_signal_bw_time = []
overlapping_signal_bw_time_count= 0
overlapping_signal_time = []
overlapping_signal_time_count = 0
metadata_compute = metadata
maxi = 0
for signal in metadata_compute:
    overlap_count_bw_time = 0
    overlap_bw_time = []
    overlap_count_time = 0
    overlap_time = []
    f_down = signal["freq_min"]
    f_up = signal["freq_max"]
    t_down = signal["sample_start"]
    t_up = t_down + signal["sample_count"]

    


    for other_signal in metadata_compute:
        if signal["file_id"] == other_signal["file_id"] and signal != other_signal:
            f_overlap = max(f_down, other_signal["freq_min"]) < min(f_up, other_signal["freq_max"])
            t_overlap = max(t_down, other_signal["sample_start"]) < min(t_up, other_signal["sample_start"] + other_signal["sample_count"])

            if f_overlap and t_overlap:
                overlap_count_bw_time += 1
                overlap_bw_time.append(other_signal["mod"])
                overlapping_signal_bw_time_count += 1
                
            
            if t_overlap:
                overlap_count_time += 1
                overlap_time.append(other_signal["mod"])
                overlapping_signal_time_count += 1

    if overlap_count_time > maxi:
        maxi = overlap_count_time
    if overlap_count_bw_time != 0:
        overlapping_signal_bw_time.append({"file_id" : signal["file_id"], "mod" : signal["mod"], "other_signal_count" : overlapping_signal_bw_time, "other_signal" : overlap_bw_time})
    if overlap_count_time != 0:
        overlapping_signal_time.append({"file_id" : signal["file_id"], "mod" : signal["mod"], "other_signal_count" : overlap_count_time,"other_signal" : overlap_time})

            
signal_count = len(metadata)
print("Shared BW + Shared time - Count = %s/%s" % (overlapping_signal_bw_time_count, signal_count))
print(overlapping_signal_bw_time)
print("\n")

print("Diff BW + Shared time - Count = %s/%s - Max = %s" % (overlapping_signal_time_count, signal_count, maxi))
print(overlapping_signal_time)
print("\n")


