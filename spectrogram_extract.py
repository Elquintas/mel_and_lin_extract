import librosa
import librosa.filters
import numpy as np
import matplotlib.pyplot as plt
import os
import hparams
from scipy import signal
from scipy.io import wavfile
import tqdm

def preemphasis(wav):
	if hparams.preemphasize:
		return signal.lfilter([1,-hparams.emphasis],[1],wav)
	return wav

def _stft(y):
	return librosa.stft(y=y, n_fft=hparams.nfft, hop_length=hparams.hop_size, 
				win_length=hparams.win_size, pad_mode='constant')

def _amp_to_db(y):
	min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
	return 20*np.log10(np.maximum(min_level, y))

_mel_basis = None
def _linear_to_mel(spectrogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	return np.dot(_mel_basis,spectrogram)

def _build_mel_basis():
	assert hparams.fmax <= hparams.sample_rate // 2
	return librosa.filters.mel(hparams.sample_rate, hparams.nfft, n_mels=hparams.num_mels,
								fmin=hparams.fmin, fmax=hparams.fmax)

def _normalize(S):
	if hparams.symmetric_mels:
		return np.clip((2 * hparams.max_abs_value)*((S-hparams.min_level_db)/(-hparams.min_level_db))
									-hparams.max_abs_value, -hparams.max_abs_value, hparams.max_abs_value)
	else: 
		return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) 
												/ (-hparams.min_level_db)), 0, hparams.max_abs_value)


def melspectrogram(wav):
	D = _stft(wav)
	S = _amp_to_db(_linear_to_mel(np.abs(D)**hparams.magnitude_power))-hparams.ref_level_db

	if hparams.signal_normalization:
		return _normalize(S)
	return S
	
def linspectrogram(wav):
	D = _stft(wav)
	S = _amp_to_db(np.abs(D)**hparams.magnitude_power)-hparams.ref_level_db

	if hparams.signal_normalization:
		return _normalize(S)
	return S	

if __name__=="__main__":
	
	path = os.getcwd()
	wav_dir = os.path.join(path,'wavs')
	mel_dir = os.path.join(path,'mels')
	lin_dir = os.path.join(path,'linear')
	
	num_files = len([f for f in os.listdir(wav_dir)if os.path.isfile(os.path.join(wav_dir, f))])
	print('Total Number of files in input directory: {}'.format(num_files))
	
	bar = tqdm.tqdm(total=num_files,desc='Feature Extraction')
	for filename in os.listdir(wav_dir):
		if filename.endswith('.wav'):
			wav, fr = librosa.load(os.path.join(wav_dir, filename))
			preemph_wav = preemphasis(wav)
			
		
			if hparams.rescale:
				wav = wav / np.abs(wav).max() * hparams.rescaling_max
				preemph_wav = preemph_wav / np.abs(preemph_wav).max() * hparams.rescaling_max
			
				#Rescale Confirmation
				if (wav > 1.).any() or (-1 > wav).any():
					raise RuntimeError('wav has invalid value: {}'.format(wav.path))
				if (preemph_wav > 1.).any() or (-1 > preemph_wav).any():
					raise RuntimeError('wav has invalid value: {}'.format(wav.path))
			
			mel_spectrogram = melspectrogram(preemph_wav).astype(np.float32)
			mel_frames = mel_spectrogram.shape[1]
			#print(np.shape(mel_spectrogram))
		
			linear_spectrogram = linspectrogram(preemph_wav).astype(np.float32)
			lin_frames = linear_spectrogram.shape[1]
			#print(np.shape(linear_spectrogram))
		
			timesteps = mel_frames * hparams.hop_size
		
			#print("Timesteps: {}".format(timesteps))
					
			#plt.imshow(mel_spectrogram,  aspect='auto');
			#plt.gca().invert_yaxis()
			#plt.ylim(0, np.shape(mel_spectrogram)[0])
			#plt.colorbar(shrink=0.65, orientation='horizontal')
			#plt.tight_layout()
			#plt.show()
		
			filename = os.path.splitext(filename)[0]
			#print(filename)
			mel_filename = 'mel-{}.npy'.format(filename)
			linear_filename = 'lin-{}.npy'.format(filename)
			
			np.save(os.path.join(mel_dir, mel_filename),mel_spectrogram.T, allow_pickle=False)
			np.save(os.path.join(lin_dir, linear_filename),linear_spectrogram.T, allow_pickle=False)
			
			bar.update(1)
					
		else:
			continue
	
		
		
		
		








