# Modulation Recognition

Link to challenge [here](https://eval.ai/web/challenges/challenge-page/1057/overview)

## Files 
- parse_meta.py : retrieve general information within the project SigMF meta data
- bw_duration.py : MIN/MAX values for BW or sample count
- snr.py : compute power information on each modulation/file (linear and log power, variance and SNR)
- imgtf.py : create the Freq/Time image of a signal and the label image
- imgtf_wo_mod.py : equivalent of imagtf but without any modulation distinction - only noise and signal 
- open_imag_and_label *TF_image*: open simultaneously the *TF_image* and its label equivalent using feh



## Instructions
`
cd /app/mod_reco/deeplab2
python3 trainer/train.py --config_file="/app/mod_reco/deeplab2/configs/Wireless/resnet50_os32_semseg.textproto" --mode=train --model_dir="/app/data/Modulation_recognition/models/" --num_gpus=2 >> /app/data/Modulation_recognition/log006.log `

`%cd /app/mod_reco/deeplab2
!python3 trainer/train.py --config_file="/app/mod_reco/deeplab2/configs/Wireless/resnet50_os32_semseg.textproto" --mode=eval --model_dir="/app/data/Modulation_recognition/models/" --num_gpus=1 >> /app/data/Modulation_recognition/log006_eval.log`
