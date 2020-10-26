# Audio Identifier
Straightforward audio identification algorithm that uses spectral constellation maps to represent audio files. Based on A.L. Wang’s algorithm developed in 2003 for Shazam, although with some key differences. The fingerprints are derived from the peaks in the STFT (spectrogram) of the audio, which are compressed in the form of hashes.
 
mean average precision measures (MAP):

Pop music: 76% precision

Classical: 32% precision

Jazz: 40% precision

