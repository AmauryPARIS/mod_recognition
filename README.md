# Modulation Recognition

Link to challenge [here](https://eval.ai/web/challenges/challenge-page/1057/overview)

## Files 
- parse_meta.py : retrieve general information within the project SigMF meta data
- bw_duration.py : MIN/MAX values for BW or sample count
- snr.py : compute power information on each modulation/file (linear and log power, variance and SNR)
- imgtf.py : create the Freq/Time image of a signal and the label image
- open_imag_and_label *TF_image*: open simultaneously the *TF_image* and its label equivalent using feh