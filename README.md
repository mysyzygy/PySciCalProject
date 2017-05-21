# PySciCalProject

Create a script that filters .wav files into 31 bands and measures the crest factor (dynamic range) of each band, 
analyzes the results with machine learning, and accurately predicts the genre based on the dynamic range measurements.

Tasks - 

1. Create a class (spectral_loudness.audio.bandpass) that divides a stereo wav file into n frequency bands with an FIR filterbank.
Create a second class (spectral_loudness.audio.loudness) that measures each band's crest factor, i.e (peak values / rms), and returns an
array of crest factor measurements for each fft window.

2. Create a Spectrum Analyzer, along with another animated graph overlay showing the dynamic range of each 
frequency band in real time.

3. Using the dynamic range tools from step 1, use machine learning to analyze different genres of music 
and create a musical genre predictor.  


DEPENDENCIES:
- python 3.4.2
- numpy, scipy, matplotlib, pylab, sounddevice, pandas, mpl_toolkits

TO RUN:

$ python spectral_loudness.py -i '/path/to/file.wav'

MACHINE LEARNING:

PySciCal_Project_machine_learning.py was created with a Jupyter notebook

NOTES:

spectral loudness currently only support 48kHz, 16-bit stereo wav files.

FUNCTIONAL HIERARCHY:

- spectral_loudness.audio.engine.Engine() - class that runs audio playback, animation and filter/loudness processing
- spectral_loudness.audio.bandpass.Bandpass() - filters audio with n FIR bandpass filters
- spectral_loudness.audio.loudness.Loudness() - measures loudness (lufs) and true peak for each frequency buffer
- spectral_loudness.plotting.animate.Animation() - runs animation during playback
- spectral_loudness.plotting.animate.Histogram() - generates peak and loudness histograms for animation
- spectral_loudness.plotting.animate.BarGraph() - generates average peak and loudness bar graph displayed and the end of playback.

note: all other files were used for testing or reference and should be ignored.