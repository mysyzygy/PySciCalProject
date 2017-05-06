# PySciCalProject

Create a script that filters .wav files into 31 bands and measures the crest factor (dynamic range) of each band, 
analyzes the results with machine learning, and accurately predicts the genre based on the dynamic range measurements.

Tasks - 

1. Create a class that divides a stereo wav file into 31 frequency bands (3 per octave) with an FFT. 
Create a second class that measures each band's crest factor, i.e (peak values / rms), and returns an 
array of crest factor measurements for each fft window. 

2. Create a Spectrum Analyzer, along with another animated graph overlay showing the dynamic range of each 
frequency band in real time.

3. Using the dynamic range tools from step 1, use machine learning to analyze different genres of music 
and create a musical genre predictor.  
