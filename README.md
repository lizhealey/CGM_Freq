<!-- #region -->

# CGM-Freq: A Python Library for Frequency Domain Analysis of Continuous Glucose Monitoring Data
Elizabeth Healey


## General
The library has 4 primary modules 
    ** Filtering. This module takes in the CGM signal and applies a low-pass filter based on a desired cutoff frequency. It returns a filtered signal and also plots the filter response and the filtered signal.  
    ** Feature Generation. This module has two parts. The first part computes frequency domain features based on the FFT and PSD. The second part computes basic time domain features. 
    ** Visualization. This module allows for visualization of the FFT and PSD of the signal and also plots the time series signal.
    ** Spectrogram. This module computes the spectrogram and returns the output of the spectrogram as well as plots the spectrogram along with the time series signal.

## Installation
 * Clone the source of this library: `git clone https://github.com/lizhealey/cgmfreq.git`
 * Install dependencies: `pip install -r ./requirements.txt `

## An example
See tutorial.ipynb for example usage
<!-- #endregion -->

```python
If you found this useful, please cite!

@article{}
```
