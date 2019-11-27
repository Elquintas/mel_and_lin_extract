import numpy as np

preemphasize = True
emphasis = 0.97
sample_rate = 8000
rescale = True
rescaling_max = 0.999

nfft = 512
win_size = 200    # typically 0.05 * sampling_rate
hop_size = 50	  # typically 0.25 * win_size


num_mels =70
fmin = 55
fmax = 4000     #7600 if sample rate is higher

symmetric_mels = True
max_abs_value = 4.


magnitude_power = 2.
min_level_db = -80
ref_level_db = 20
signal_normalization = True
