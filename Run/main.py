'''
# Imports
import math
import numpy as np
import os
import warnings
import json
from numpy import shape, mean
from scipy.io import wavfile
from scipy.signal import resample, find_peaks, hilbert

warnings.filterwarnings("ignore")  # Ignore warning messages to avoid unnecessary output

# Define input/output directories
input_dir = "C:/Users/rlessard/Desktop/runThis"  # Path containing .wav files
output_dir = "C:/Users/rlessard/Desktop/SoundscapeCodeDesktop/DataOutput/10_minutes/runThis.json"  # Path to store .mat/.json file

# Global variables
file_dir = os.listdir(input_dir)
num_bits = 16  # Determines the range of possible amplitude values that can be represented (2^16 different levels)
RS = -178.3  # -175.2  # BE SURE TO CHANGE FOR EACH HYDROPHONE, sensitivity is based on hydrophone, not recorder
peak_volts = 2  # Peak to peak voltage range
arti = 1  # make 1 if calibration tone present

# Analysis options
timewin = 60  # length of time window in seconds for analysis bins
fft_win = 1  # length of fft window in minutes
avtime = 0.1  # Averaging time duration in seconds for autocorrelation measurements
flow = 50  # Low frequency cutoff
fhigh = 300  # High frequency cutoff

def frequency_resample(fs, x):
    """
    Adjust the sample rate of the audio data by downsampling or upsampling as needed.

    Args:
    fs: Original sample rate of the audio data.
    x: Audio signal data.

    Returns:
    fs: Updated sample rate after resampling.
    x: Resampled audio data.
    """
    if fs == 576000:
        x = resample(x, len(x) // 4)
        fs = fs // 4
    elif fs == 288000:
        x = resample(x, len(x) // 2)
        fs = fs // 2
    elif fs == 16000:
        x = resample(x, len(x) * 9)
        fs = fs * 9
    elif fs == 512000:
        roundNum = 4
        x = resample(x, len(x) // roundNum)
        fs = fs / 3.5555555555555555555

    return fs, x


def minute_padding(num_timewin, pts_per_timewin, p_filt):
    """
    Pads the filtered signal to ensure that the length matches full minute time windows.

    Args:
    num_timewin: Total number of time windows for the signal.
    pts_per_timewin: Number of points in each time window.
    p_filt: Filtered audio signal.

    Returns:
    timechunk_matrix: Reshaped signal into a 2D matrix, where each column is a time window.
    p_filt_padded: Padded filtered signal to match the required length.
    """

    padding_length = num_timewin * pts_per_timewin - len(p_filt)  # Calculate padding length required to fill all time windows
    p_filt_padded = np.concatenate((p_filt, np.zeros(padding_length)))  # Pad the filtered signal with zeros
    timechunk_matrix = p_filt_padded.reshape(pts_per_timewin, num_timewin)  # Reshape padded signal into matrix with time window dimensions
    return timechunk_matrix, p_filt_padded


def calculate_impulsivity(impulsivity, tcm_rearrange):
    """
    Calculate impulsivity using kurtosis of the rearranged time chunks.

    Args:
    impulsivity: List to store impulsivity values.
    tcm_rearrange: Rearranged time chunk matrix.

    Returns:
    impulsivity: Updated list with calculated impulsivity values.
    """
    kmat = kurtosis_reilly(np.array(tcm_rearrange))
    impulsivity.extend(kmat)
    return impulsivity


def calculate_spl(rms_matrix, tcm_rearrange, SPLrms, SPLpk):
    """
        Calculate SPL (Sound Pressure Level) in both RMS and Peak formats.

        Args:
        rms_matrix: RMS values of the time chunks.
        tcm_rearrange: Rearranged time chunk matrix.
        SPLrms: List to store SPL RMS values.
        SPLpk: List to store SPL Peak values.

        Returns:
        SPLrms: Updated list with SPL RMS values.
        SPLpk: Updated list with SPL Peak values.
        """
    SPLrmshold = 20 * np.log10(rms_matrix)
    SPLpkhold = np.max(20 * np.log10(np.abs(tcm_rearrange)), axis=0)
    SPLrms.extend(SPLrmshold)
    SPLpk.extend(SPLpkhold)
    return SPLrms, SPLpk


def calculate_autocorrelation(autocorr, acorr):
    """
    Calculate autocorrelation for periodicity analysis.

    Args:
    autocorr: Current autocorrelation matrix.
    acorr: New autocorrelation values to append.

    Returns:
    autocorr: Updated autocorrelation matrix.
    """
    if autocorr is None:
        autocorr = acorr
    else:
        autocorr = np.column_stack((autocorr, acorr))
    return autocorr


def f_WAV_frankenfunction_reilly(num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh):
    """
    Main function to process WAV files and compute various soundscape metrics.

    Args:
    num_bits: bit rate of hydrophone
    peak_volts: voltage of the recorder, peak to peak
    file_dir: the directory of files intended for processing
    RS: hydrophone sensitivity
    timewin: size of time windows in seconds for calculation of soundscape metrics
    avtime: averaging time duration in seconds for autocorrelation measurements
    fft_win: size of time window in minutes over which fft is performed
    arti: enter 1 if there are artifacts such as calibration tones at the beginnings of recordings
    flow: lower frequency cutoff
    fhigh: upper frequency cutoff

    Returns:
    SPLrms: matrix of root mean square sound pressure level, each column is a sound file, each row is 1 min of data
    SPLpk: matrix of peak SPL (the highest SPL in a sample), each column is a sound file, each row is 1 min of data
    impulsivity: uses kurtosis function to measure impulsive sounds
    peakcount: a count of the # of times that the autocorrelation threshold is exceeded
    autocorr: a matrix of autocorrelation calculations, each column is 1 min of data, each row is the autocorrelation
              value calculated at .1 sec time
    dissim: a matrix of the amount of uniformity between each 1 min of data compared to the following minute of data,
            should have n-1 rows where n is the number of minutes of data
    """

    num_files = len(file_dir)  # Number of .wav files in input directory
    p = []  # Placeholder for audio data
    pout = []  # Placeholder for filtered audio data
    SPLrms = []  # Root mean square (RMS) sound pressure level (SPL) for each time window
    SPLpk = []  # Peak SPL for each time window
    impulsivity = []  # Placeholder for impulsivity metric
    peakcount = []  # Placeholder for periodicity metric (peak count)
    autocorr = None  # Placeholder for autocorrelation results
    dissim = []  # Placeholder for dissimilarity metric

    # Loop through each WAV file and process it
    for ii in range(num_files):
        print(f"\n{ii + 1} out of {num_files}")  # lists ii as a variable to tell you every time it completes a loop
        filename = os.path.join(input_dir, file_dir[ii])  # Get full path of current file
        rs = (10 ** (RS / 20))  # Convert hydrophone sensitivity from dB to a linear scale
        max_count = 2 ** num_bits  # Calculate maximum count based on bit depth of audio data
        conv_factor = peak_volts / max_count  # Calculate conversion factor for voltage

        # Determine sample rate and audio data from .wav file
        fs, x = wavfile.read(filename)

        # Downsample if sample rate is too high
        fs, x = frequency_resample(fs, x)

        if num_bits == 24:
            x = x >> 8  # Bit shift; accounts for audioread casting of 24 bit to 32 bit (zeroes behind)

        v = x.astype(float) * conv_factor

        p = v / rs  # Voltage to pressure
        if arti == 1:
            p = p[int(6 * fs) - 1:]  # trims first 4 sec of recording to remove calibration tone
        pout.extend(p)  # Make this so the new p gets added at end of original p

        p_filt = dylan_bpfilt(pout, 1 / fs, flow, fhigh)
        pout = []
        pts_per_timewin = int(timewin * fs)  # Number of samples per time window - set window * 576 kHz sample rate

        num_timewin = math.floor(len(p_filt) / pts_per_timewin) + 1  # Number of time windows contained in the sound file

        # Pad the signal
        timechunk_matrix, p_filt_padded = minute_padding(num_timewin, pts_per_timewin, p_filt)

        # Create matrices with correct dimensions
        tcm_rearrange = create_2d_array_by_columns(p_filt_padded, pts_per_timewin, num_timewin)
        rms_matrix = rms_reilly(timechunk_matrix, 0)

        SPLrms, SPLpk = calculate_spl(rms_matrix, tcm_rearrange, SPLrms, SPLpk)

        # Compute impulsivity (kurtosis) for the rearranged time windows
        calculate_impulsivity(impulsivity, tcm_rearrange)

        # Periodicity
        pkcount, acorr = f_solo_per_GM2(p_filt_padded, fs, timewin, avtime)
        peakcount.extend(pkcount)

        # Autocorrelation
        autocorr = calculate_autocorrelation(autocorr, acorr)

        # Calculate dissimilarity between adjacent time windows (D-index)
        Dfin = f_solo_dissim_GM1(pts_per_timewin, num_timewin, fft_win, fs, tcm_rearrange)
        dissim.extend(Dfin)

    # Reshape matrices (Convert from vector to matrix)
    dissim = np.reshape(dissim, (num_files, int(len(dissim) / num_files)))
    impulsivity = np.reshape(impulsivity, (num_files, int(len(impulsivity) / num_files)))
    peakcount = np.reshape(peakcount, (num_files, int(len(peakcount) / num_files)))
    SPLpk = np.reshape(SPLpk, (num_files, int(len(SPLpk) / num_files)))
    SPLrms = np.reshape(SPLrms, (num_files, int(len(SPLrms) / num_files)))

    # Round autocorrelation values to avoid floating-point precision issues
    autocorr = np.round(autocorr, 15)

    # Change dimensions for mxn to nxm
    SPLrms = reshape_vertical(SPLrms)
    SPLpk = reshape_vertical(SPLpk)
    impulsivity = reshape_vertical(impulsivity)
    peakcount = reshape_vertical(peakcount)
    dissim = reshape_vertical(dissim)

    # Write the results to a JSON file
    write_to_json(output_dir, SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim)

    return SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim


def create_2d_array_by_columns(input_array, row, col):
    """
    Creates a 2D array by filling data column-wise.

    Args:
    input_array: Input data to fill the array.
    row: Number of rows.
    col: Number of columns.

    Returns:
    result: 2D array with data filled column-wise.
    """
    result = np.zeros((row, col))
    for c in range(col):
        for r in range(row):
            result[r][c] = input_array[c * row + r]
    return result


def column_max_SPL(timechunk_matrix):
    """
    Get the number of columns (assuming all rows have the same length)
    Returns the peak in each column
    """
    num_columns = len(timechunk_matrix[0])

    SPLpkhold = []
    for col in range(num_columns):
        column_values = [abs(row[col]) for row in timechunk_matrix if row[col] != 0]
        if column_values:
            max_value = max(column_values)
            spl = 20 * math.log10(max_value)
            SPLpkhold.append(spl)
        else:
            SPLpkhold.append(float('-inf'))  # Used to represent undefined SPL value

    return SPLpkhold


def kurtosis_reilly(x, flag=1, dim=None):
    """
    Calculates kurtosis of the input data.

    Args:
    x: Input data for kurtosis calculation.

    Returns:
    k: Kurtosis values.
    """
    flag = 1

    # Determine the dimension if not provided
    if flag not in (0, 1) and flag is not None:
        raise ValueError("Bad flag value: flag should be 0 or 1.")

    if dim is None:
        # Handle the special case where x is empty.
        if np.array_equal(x, np.array([])):
            print("x is empty")
            return np.nan

        # Determine the dimension along which np.nanmean will work.
        dim = next((i for i, s in enumerate(x.shape) if s != 1), None)

        if dim is None:
            dim = 0

    x0 = x - np.nanmean(x, axis=dim, keepdims=True)

    s2 = nanmean(x0 ** 2, axis=dim)  # biased variance estimator
    m4 = np.nanmean(np.power(x0, 4), axis=0)

    k = m4 / np.square(s2)  # Determine element-wise square

    # Bias correction
    if flag == 0:
        n = np.sum(~np.isnan(x), axis=dim)
        n[n < 4] = np.nan  # bias correction not defined for n < 4
        k = ((n + 1) * k - 3 * (n - 1)) * (n - 1) / ((n - 2) * (n - 3)) + 3

    return k


def nanmean(arr, axis=None):
    """
    Convert the input to a numpy array if it isn't already
    """
    arr = np.array(arr)

    # Create a mask to ignore NaN, 0, and empty values
    mask = ~np.isnan(arr) & (arr != 0) & (arr != "")

    # Apply the mask and calculate the mean
    masked_arr = np.where(mask, arr, np.nan)
    return np.nanmean(masked_arr, axis=axis)


def rms_reilly(x, dim=None):
    ''''''
    Rewritten from MATLAB
    For vectors, RMS(X) is the root mean square value in X.
    For matrices, RMS(X) is a row vector containing the RMS value from each column.
    Y = RMS(X,DIM) operates along the dimension DIM.
    ''''''

    global vertical_averages_sqrt
    if np.isrealobj(x):
        if dim is not None:
            sqmexx = np.square(x)
            [sqmexx_a, sqmexx_b] = shape(sqmexx)

            sqmexx = sqmexx.reshape(-1)

            reshaped_sqmexx = sqmexx.reshape((sqmexx_b, sqmexx_a))
            transposed_sqmexx = reshaped_sqmexx.T

            vertical_averages = mean(transposed_sqmexx, axis=0)
            vertical_averages_sqrt = np.sqrt(vertical_averages)

    return vertical_averages_sqrt


def bpfilt_initial_values(ts, samint):
    """
    Define initial values for dylan_bpfilt
    """
    npts = len(ts)
    reclen = npts * samint
    spec = np.fft.fft(ts, npts)
    aspec = np.abs(spec)
    pspec = np.angle(spec)
    freq = np.fft.fftshift(np.arange(-npts / 2, npts / 2) / reclen)

    return npts, spec, aspec, pspec, freq


def dylan_bpfilt(ts, samint, flow, fhigh):
    """
    Performs bandpass filtering.

    Args:
    data: Input audio data.
    fs_inv: Inverse of the sample rate.
    flow: Lower frequency bound for filtering.
    fhigh: Upper frequency bound for filtering.

    Returns:
    Filtered data.
    """
    npts, spec, aspec, pspec, freq = bpfilt_initial_values(ts, samint)

    if fhigh == 0:
        fhigh = 1 / (2 * samint)

    ifr = np.where((np.abs(freq) >= flow) & (np.abs(freq) <= fhigh))[0]  # Calculate metrics between flow and fhigh
    rspec = np.zeros_like(spec)
    ispec = np.zeros_like(spec)

    rspec[ifr] = aspec[ifr] * np.cos(pspec[ifr])
    ispec[ifr] = aspec[ifr] * np.sin(pspec[ifr])
    filtspec2 = rspec + 1j * ispec

    tsfilt = np.real(np.fft.ifft(filtspec2, npts))

    return tsfilt


def f_solo_per_GM2(p_filt, fs, timewin, avtime):
    """
    Function for calculating peak count and autocorrelation.

    Args:
    data: Input audio data.
    fs: Sample rate of the audio data.
    timewin: Time window size for analysis.
    avtime: Averaging time for autocorrelation.

    Returns:
    Peak count and autocorrelation values.
    """
    p_av = []
    avwin = int(fs * avtime)
    sampwin = int(fs * timewin)
    ntwin = len(p_filt) // sampwin  # Number of minutes
    p_filt = p_filt[:sampwin * ntwin]

    p_filt = distribute_array(p_filt, ntwin)

    p_filt = p_filt ** 2

    numavwin = p_filt.shape[0] // avwin

    for jj in range(ntwin):
        avwinmatrix = distribute_array_2d(p_filt[:, jj], numavwin, avwin)

        p_avi = np.mean(avwinmatrix, axis=0)
        p_av.append(p_avi)

    p_av = np.transpose(p_av)
    p_avtot = np.array(p_av)

    shape0, shape1 = np.shape(p_avtot)

    max_lag = int(shape0 * 0.7)
    acorr = np.zeros((max_lag + 1, shape1))
    pkcount = np.zeros(shape1)

    for zz in range(shape1):
        P, _ = correl_5(p_avtot[:, zz], p_avtot[:, zz], max_lag, 0)
        acorr[:, zz] = P
        pks, _ = find_peaks(acorr[:, zz], prominence=0.5)
        pkcount[zz] = len(pks)

    return pkcount, acorr


def distribute_array(arr_1d, dim):
    """
    Calculate the number of elements per column.
    Create a matrix, where the number of columns is equal to number of minutes in recording.
    Appends vectors to create 2D matrix
    """
    elements_per_column = len(arr_1d) // dim

    arr_2d = np.empty((elements_per_column, dim))

    for i in range(dim):
        start_index = i * elements_per_column
        end_index = (i + 1) * elements_per_column
        arr_2d[:, i] = arr_1d[start_index:end_index]

    return arr_2d


def distribute_array_2d(arr_1d, num_columns, num_rows=None):
    """
    If num_rows is not specified, calculate it based on the array length and num_columns.
    """
    if num_rows is None:
        num_rows = len(arr_1d) // num_columns

    # Ensure the input array has enough elements
    if len(arr_1d) < num_rows * num_columns:
        raise ValueError("Input array is too small for the specified dimensions")

    # Create an empty 2D array with appropriate dimensions
    arr_2d = np.empty((num_rows, num_columns))

    # Iterate through the 1D array and place elements in the 2D array
    for i in range(num_columns):
        start_index = i * num_rows
        end_index = (i + 1) * num_rows
        arr_2d[:, i] = arr_1d[start_index:end_index]

    return arr_2d


def f_solo_dissim_GM1(pts_per_timewin, num_timewin, fft_win, fs, tcm_rearrange):
    """
    Function for calculating dissimilarity.

    Args:
    pts_per_timewin: Points per time window.
    num_timewin: Number of time windows.
    fft_win: FFT window size.
    fs: Sample rate of the audio data.
    tcm_rearrange: Rearranged time chunk matrix.
    Returns:
    Dissimilarity values.
    """
    tcm_rearrange = np.array(tcm_rearrange)  # Uses numpy to perform mathematical operations

    pts_per_fft = int(fft_win * fs)  # Calc size fft window
    numfftwin = int(np.floor(pts_per_timewin / pts_per_fft))  # Number of fft windows

    D = []

    for kk in range(num_timewin - 1):
        analytic1 = hilbert(tcm_rearrange[:, kk], axis=-1)
        analytic2 = hilbert(tcm_rearrange[:, kk + 1])

        at1 = abs(analytic1) / np.sum(abs(analytic1))
        at2 = abs(analytic2) / np.sum(abs(analytic2))

        Dt = np.sum(abs(at1 - at2)) / 2

        s3a = tcm_rearrange[:, kk]
        s3a = s3a[:int(pts_per_fft * numfftwin)]
        s3a = create_2d_array_by_columns(s3a, pts_per_fft, numfftwin)

        s3a = np.array(s3a)
        ga = np.abs(np.fft.fft(s3a, axis=0)) / s3a.shape[0]

        sfa = np.mean(ga, axis=1)
        Sfa = abs(sfa) / np.sum(abs(sfa))

        s3b = tcm_rearrange[:, kk + 1]
        s3b = s3b[:int(pts_per_fft * numfftwin)]
        s3b = np.array(s3b)
        s3b = create_2d_array_by_columns(s3b, pts_per_fft, numfftwin)

        s3b = np.array(s3b)
        gb = np.abs(np.fft.fft(s3b, axis=0)) / s3b.shape[0]
        sfb = np.mean(gb, axis=1)
        Sfb = abs(sfb) / np.sum(abs(sfb))

        Df = np.sum(abs(Sfb - Sfa)) / 2
        Di = Dt * Df

        D.append(Di)

    Dfin = np.array(D)
    return Dfin


def correl_5(ts1, ts2, lags, offset):
    ''''''
    Used to calculate autocorrelation
    ''''''
    P = np.zeros(lags + 1)
    nlags = np.arange(0, lags + 1)

    for i in range(lags + 1):
        ng = 1
        sx = 2
        sy = 3
        sxx = 4
        syy = 5
        sxy = 6

        for k in range(len(ts1) - (i + offset)):
            x = ts1[k]
            y = ts2[k + (i + offset)]
            if not np.isnan(x) and not np.isnan(y):
                sx += x
                sy += y
                sxx += x * x
                syy += y * y
                sxy += x * y
                ng += 1

        covar1 = (sxy / ng) - ((sx / ng) * (sy / ng))
        denom1 = np.sqrt((sxx / ng) - (sx / ng) ** 2)
        denom2 = np.sqrt((syy / ng) - (sy / ng) ** 2)
        P[i] = covar1 / (denom1 * denom2)

    return P, nlags


def reshape_vertical(matrix):
    """
    Converts matrix dimensions from mxn to nxm
    """
    matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return matrix


def write_to_json(output_dir, SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim):
    """
    Write calculated metrics to JSON file

    Args:
    output_dir: Path to save the output JSON file.
    SPLrms: Matrix of SPLrms values.
    SPLpk: Matrix of SPLpk values.
    impulsivity: Matrix of impulsivity values.
    peakcount: Matrix of peak count values.
    autocorr: Matrix of autocorrelation values.
    dissim: Matrix of dissimilarity values.
    """
    # Create a dictionary to hold all the results
    result_data = {
        'SPLrms': SPLrms,
        'SPLpk': SPLpk,
        'impulsivity': impulsivity,
        'peakcount': peakcount,
        'autocorr': autocorr.tolist(),
        'dissim': dissim
    }

    # Write the dictionary to a JSON file
    with open(output_dir, 'w') as f:
        json.dump(result_data, f, indent=4)



if __name__ == '__main__':
    SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim = f_WAV_frankenfunction_reilly(
        num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh)
'''
import librosa
from numpy import size, shape

'''
import math
import numpy as np
import os
import warnings
import json
from numpy import shape, mean, size
from numpy.array_api import zeros
from scipy.io import wavfile
from scipy.signal import resample, find_peaks, hilbert

warnings.filterwarnings("ignore")  # Ignore warning messages to avoid unnecessary output

# Define input/output directories
input_dir = "C:/Users/rlessard/Desktop/runThisInput/AMAR"  # Path containing .wav files
output_dir = "C:/Users/rlessard/Desktop/runThisOutput/runThisPython.json"  # Path to store .mat/.json file


def frequency_resample(fs, x):
    """
    Adjust the sample rate of the audio data by downsampling or upsampling as needed.

    Args:
    fs: Original sample rate of the audio data.
    x: Audio signal data.

    Returns:
    fs: Updated sample rate after resampling.
    x: Resampled audio data.
    """
    if fs == 576000:
        x = resample(x, len(x) // 4)
        fs = fs // 4
    elif fs == 288000:
        x = resample(x, len(x) // 2)
        fs = fs // 2
    elif fs == 16000:
        x = resample(x, len(x) * 9)
        fs = fs * 9
    elif fs == 512000:
        roundNum = 4
        x = resample(x, len(x) // roundNum)
        fs = fs / 3.5555555555555555555

    return fs, x


def minute_padding(num_timewin, pts_per_timewin, p_filt):
    """
    Pads the filtered signal to ensure that the length matches full minute time windows.

    Args:
    num_timewin: Total number of time windows for the signal.
    pts_per_timewin: Number of points in each time window.
    p_filt: Filtered audio signal.

    Returns:
    timechunk_matrix: Reshaped signal into a 2D matrix, where each column is a time window.
    p_filt_padded: Padded filtered signal to match the required length.
    """

    padding_length = num_timewin * pts_per_timewin - len(p_filt)  # Calculate padding length required to fill all time windows
    p_filt_padded = np.concatenate((p_filt, np.zeros(padding_length)))  # Pad the filtered signal with zeros
    timechunk_matrix = p_filt_padded.reshape(pts_per_timewin, num_timewin)  # Reshape padded signal into matrix with time window dimensions
    return timechunk_matrix, p_filt_padded


def calculate_impulsivity(impulsivity, tcm_rearrange):
    """
    Calculate impulsivity using kurtosis of the rearranged time chunks.

    Args:
    impulsivity: List to store impulsivity values.
    tcm_rearrange: Rearranged time chunk matrix.

    Returns:
    impulsivity: Updated list with calculated impulsivity values.
    """
    kmat = kurtosis_reilly(np.array(tcm_rearrange))
    impulsivity.extend(kmat)
    return impulsivity


def calculate_spl(rms_matrix, tcm_rearrange, SPLrms, SPLpk):
    """
        Calculate SPL (Sound Pressure Level) in both RMS and Peak formats.

        Args:
        rms_matrix: RMS values of the time chunks.
        tcm_rearrange: Rearranged time chunk matrix.
        SPLrms: List to store SPL RMS values.
        SPLpk: List to store SPL Peak values.

        Returns:
        SPLrms: Updated list with SPL RMS values.
        SPLpk: Updated list with SPL Peak values.
        """
    SPLrmshold = 20 * np.log10(rms_matrix)
    SPLpkhold = np.max(20 * np.log10(np.abs(tcm_rearrange)), axis=0)
    SPLrms.extend(SPLrmshold)
    SPLpk.extend(SPLpkhold)
    return SPLrms, SPLpk


def calculate_autocorrelation(autocorr, acorr):
    """
    Calculate autocorrelation for periodicity analysis.

    Args:
    autocorr: Current autocorrelation matrix.
    acorr: New autocorrelation values to append.

    Returns:
    autocorr: Updated autocorrelation matrix.
    """
    if autocorr is None:
        autocorr = acorr
    else:
        autocorr = np.column_stack((autocorr, acorr))
    return autocorr


def f_WAV_frankenfunction_reilly(num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh):

    # print("num_bits: ", num_bits, "peak_volts: ", peak_volts, "file_dir: ", file_dir, "RS: ", RS, "timewin: ", timewin, "avtime: ", avtime, "fft_win: ", fft_win, "arti: ", arti, "flow: ", flow, "fhigh: ", fhigh)

    """
    Main function to process WAV files and compute various soundscape metrics.

    Args:
    num_bits: Bit rate of the audio data.
    peak_volts: Peak voltage of the recorder.
    file_dir: Directory of files to process.
    RS: Hydrophone sensitivity in dB.
    timewin: Size of time windows for analysis (in seconds).
    avtime: Averaging time for autocorrelation.
    fft_win: FFT window size for analysis.
    arti: Indicates if the recording contains calibration tones (1 for true).
    flow: Lower frequency cutoff for bandpass filtering.
    fhigh: Upper frequency cutoff for bandpass filtering.

    Returns:
    SPLrms: Root mean square sound pressure level matrix.
    SPLpk: Peak sound pressure level matrix.
    impulsivity: Impulsivity measurements based on kurtosis.
    peakcount: Count of peaks in autocorrelation function.
    autocorr: Autocorrelation matrix for periodicity analysis.
    dissim: Dissimilarity matrix comparing adjacent time windows.
    """

    num_files = len(file_dir)  # Number of .wav files in input directory
    p = []  # Placeholder for audio data
    pout = []  # Placeholder for filtered audio data
    SPLrms = []  # Root mean square (RMS) sound pressure level (SPL) for each time window
    SPLpk = []  # Peak SPL for each time window
    impulsivity = []  # Placeholder for impulsivity metric
    peakcount = []  # Placeholder for periodicity metric (peak count)
    autocorr = None  # Placeholder for autocorrelation results
    dissim = []  # Placeholder for dissimilarity metric

    # Loop through each WAV file and process it
    for ii in range(num_files):
        filename = os.path.join(input_dir, file_dir[ii])  # Get full path of current file
        rs = (10 ** (RS / 20))  # Convert hydrophone sensitivity from dB to a linear scale
        max_count = 2 ** num_bits  # Calculate maximum count based on bit depth of audio data
        conv_factor = peak_volts / max_count  # Calculate conversion factor for voltage

        # Determine sample rate and audio data from .wav file
        fs, x = wavfile.read(filename)
        # print(fs)
        # print(x)

        # Downsample if sample rate is too high
        # fs, x = frequency_resample(fs, x)
        # print(fs)
        # print(x)

        if num_bits == 24:
            x = x >> 8  # Bit shift; accounts for audioread casting of 24 bit to 32 bit (zeroes behind)
        # print(num_bits)

        v = x.astype(float) * conv_factor

        v_zero = zeros(9 * len(v))
        j = 0
        for i in range(size(v_zero)):
            if i % 9 == 0:
                v_zero[i] = v[j]
                j += 1

        p = v / rs  # Voltage to pressure

        p_zero = zeros(9 * len(p))
        j = 0
        for i in range(size(p_zero)):
            if i % 9 == 0:
                p_zero[i] = p[j]
                j += 1

        print("p_zero", p_zero[:20])
        print("p_zero", p_zero[-20:])
        print("p_zero.shape", p_zero.shape)

        if arti == 1:
            # REPLACE P WITH P_ZERO
            p_zero = p_zero[6 * fs - 1:]  # trims first 4 sec of recording to remove calibration tone

        # pout = pout.append(p_zero)
        # print("pout size before extend:", size(pout))
        pout.extend(p_zero)  # Make this so the new p gets added at end of original p
        # print("pout size after extend: " + str(size(pout)))
        # print("pout[:10]: ",pout[:10])
        # print("pout[-10]: ", pout[-10:])

        # print(pout[0])
        # print(1 / fs)
        # print(flow)
        # print(fhigh)
        p_filt = dylan_bpfilt(pout, 1 / fs, flow, fhigh)
        pout = []
        pts_per_timewin = int(timewin * fs)  # Number of samples per time window - set window * 576 kHz sample rate

        num_timewin = math.floor(len(p_filt) / pts_per_timewin) + 1  # Number of time windows contained in the sound file

        # Pad the signal
        timechunk_matrix, p_filt_padded = minute_padding(num_timewin, pts_per_timewin, p_filt)

        # Create matrices with correct dimensions
        tcm_rearrange = create_2d_array_by_columns(p_filt_padded, pts_per_timewin, num_timewin)
        # print("timechunk_matrix: " + str(timechunk_matrix))
        rms_matrix = rms_reilly(timechunk_matrix, 0)
        # print("rms_matrix: " + str(rms_matrix))

        SPLrms, SPLpk = calculate_spl(rms_matrix, tcm_rearrange, SPLrms, SPLpk)

        # Compute impulsivity (kurtosis) for the rearranged time windows
        calculate_impulsivity(impulsivity, tcm_rearrange)

        # Periodicity
        pkcount, acorr = f_solo_per_GM2(p_filt_padded, fs, timewin, avtime)
        peakcount.extend(pkcount)

        # Autocorrelation
        autocorr = calculate_autocorrelation(autocorr, acorr)

        # Calculate dissimilarity between adjacent time windows (D-index)
        Dfin = f_solo_dissim_GM1(pts_per_timewin, num_timewin, fft_win, fs, tcm_rearrange)
        dissim.extend(Dfin)

    # Reshape matrices (Convert from vector to matrix)
    dissim = np.reshape(dissim, (num_files, int(len(dissim) / num_files)))
    impulsivity = np.reshape(impulsivity, (num_files, int(len(impulsivity) / num_files)))
    peakcount = np.reshape(peakcount, (num_files, int(len(peakcount) / num_files)))
    SPLpk = np.reshape(SPLpk, (num_files, int(len(SPLpk) / num_files)))
    SPLrms = np.reshape(SPLrms, (num_files, int(len(SPLrms) / num_files)))

    # Round autocorrelation values to avoid floating-point precision issues
    autocorr = np.round(autocorr, 15)

    return SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim


def create_2d_array_by_columns(input_array, row, col):
    """
    Creates a 2D array by filling data column-wise.

    Args:
    input_array: Input data to fill the array.
    row: Number of rows.
    col: Number of columns.

    Returns:
    result: 2D array with data filled column-wise.
    """
    result = np.zeros((row, col))
    for c in range(col):
        for r in range(row):
            result[r][c] = input_array[c * row + r]
    return result


def column_max_SPL(timechunk_matrix):
    """
    Get the number of columns (assuming all rows have the same length)
    Returns the peak in each column
    """
    num_columns = len(timechunk_matrix[0])

    SPLpkhold = []
    for col in range(num_columns):
        column_values = [abs(row[col]) for row in timechunk_matrix if row[col] != 0]
        if column_values:
            max_value = max(column_values)
            spl = 20 * math.log10(max_value)
            SPLpkhold.append(spl)
        else:
            SPLpkhold.append(float('-inf'))  # Used to represent undefined SPL value

    return SPLpkhold


def kurtosis_reilly(x, flag=1, dim=None):
    """
    Calculates kurtosis of the input data.

    Args:
    x: Input data for kurtosis calculation.

    Returns:
    k: Kurtosis values.
    """
    flag = 1

    # Determine the dimension if not provided
    if flag not in (0, 1) and flag is not None:
        raise ValueError("Bad flag value: flag should be 0 or 1.")

    if dim is None:
        # Handle the special case where x is empty.
        if np.array_equal(x, np.array([])):
            print("x is empty")
            return np.nan

        # Determine the dimension along which np.nanmean will work.
        dim = next((i for i, s in enumerate(x.shape) if s != 1), None)

        if dim is None:
            dim = 0

    x0 = x - np.nanmean(x, axis=dim, keepdims=True)

    s2 = nanmean(x0 ** 2, axis=dim)  # biased variance estimator
    m4 = np.nanmean(np.power(x0, 4), axis=0)

    k = m4 / np.square(s2)  # Determine element-wise square

    # Bias correction
    if flag == 0:
        n = np.sum(~np.isnan(x), axis=dim)
        n[n < 4] = np.nan  # bias correction not defined for n < 4
        k = ((n + 1) * k - 3 * (n - 1)) * (n - 1) / ((n - 2) * (n - 3)) + 3

    return k


def nanmean(arr, axis=None):
    """
    Convert the input to a numpy array if it isn't already
    """
    arr = np.array(arr)

    # Create a mask to ignore NaN, 0, and empty values
    mask = ~np.isnan(arr) & (arr != 0) & (arr != "")

    # Apply the mask and calculate the mean
    masked_arr = np.where(mask, arr, np.nan)
    return np.nanmean(masked_arr, axis=axis)


def rms_reilly(x, dim=None):
    '''
'''
    Rewritten from MATLAB
    For vectors, RMS(X) is the root mean square value in X.
    For matrices, RMS(X) is a row vector containing the RMS value from each column.
    Y = RMS(X,DIM) operates along the dimension DIM.
    '''
'''

    global vertical_averages_sqrt
    if np.isrealobj(x):
        if dim is not None:
            sqmexx = np.square(x)
            [sqmexx_a, sqmexx_b] = shape(sqmexx)

            sqmexx = sqmexx.reshape(-1)

            reshaped_sqmexx = sqmexx.reshape((sqmexx_b, sqmexx_a))
            transposed_sqmexx = reshaped_sqmexx.T

            vertical_averages = mean(transposed_sqmexx, axis=0)
            vertical_averages_sqrt = np.sqrt(vertical_averages)

    return vertical_averages_sqrt


def bpfilt_initial_values(ts, samint):
    """
    Define initial values for dylan_bpfilt
    """
    npts = len(ts)
    reclen = npts * samint
    spec = np.fft.fft(ts, npts)
    aspec = np.abs(spec)
    pspec = np.angle(spec)
    freq = np.fft.fftshift(np.arange(-npts / 2, npts / 2) / reclen)

    return npts, spec, aspec, pspec, freq


def dylan_bpfilt(ts, samint, flow, fhigh):
    """
    Performs bandpass filtering.

    Args:
    data: Input audio data.
    fs_inv: Inverse of the sample rate.
    flow: Lower frequency bound for filtering.
    fhigh: Upper frequency bound for filtering.

    Returns:
    Filtered data.
    """
    npts, spec, aspec, pspec, freq = bpfilt_initial_values(ts, samint)

    if fhigh == 0:
        fhigh = 1 / (2 * samint)

    ifr = np.where((np.abs(freq) >= flow) & (np.abs(freq) <= fhigh))[0]  # Calculate metrics between flow and fhigh
    rspec = np.zeros_like(spec)
    ispec = np.zeros_like(spec)

    rspec[ifr] = aspec[ifr] * np.cos(pspec[ifr])
    ispec[ifr] = aspec[ifr] * np.sin(pspec[ifr])
    filtspec2 = rspec + 1j * ispec

    tsfilt = np.real(np.fft.ifft(filtspec2, npts))

    return tsfilt


def f_solo_per_GM2(p_filt, fs, timewin, avtime):
    """
    Function for calculating peak count and autocorrelation.

    Args:
    data: Input audio data.
    fs: Sample rate of the audio data.
    timewin: Time window size for analysis.
    avtime: Averaging time for autocorrelation.

    Returns:
    Peak count and autocorrelation values.
    """
    p_av = []
    avwin = int(fs * avtime)
    sampwin = int(fs * timewin)
    ntwin = len(p_filt) // sampwin  # Number of minutes
    p_filt = p_filt[:sampwin * ntwin]

    p_filt = distribute_array(p_filt, ntwin)

    p_filt = p_filt ** 2

    numavwin = p_filt.shape[0] // avwin

    for jj in range(ntwin):
        avwinmatrix = distribute_array_2d(p_filt[:, jj], numavwin, avwin)

        p_avi = np.mean(avwinmatrix, axis=0)
        p_av.append(p_avi)

    p_av = np.transpose(p_av)
    p_avtot = np.array(p_av)

    shape0, shape1 = np.shape(p_avtot)

    max_lag = int(shape0 * 0.7)
    acorr = np.zeros((max_lag + 1, shape1))
    pkcount = np.zeros(shape1)

    for zz in range(shape1):
        P, _ = correl_5(p_avtot[:, zz], p_avtot[:, zz], max_lag, 0)
        acorr[:, zz] = P
        pks, _ = find_peaks(acorr[:, zz], prominence=0.5)
        pkcount[zz] = len(pks)

    return pkcount, acorr


def distribute_array(arr_1d, dim):
    """
    Calculate the number of elements per column.
    Create a matrix, where the number of columns is equal to number of minutes in recording.
    Appends vectors to create 2D matrix
    """
    elements_per_column = len(arr_1d) // dim

    arr_2d = np.empty((elements_per_column, dim))

    for i in range(dim):
        start_index = i * elements_per_column
        end_index = (i + 1) * elements_per_column
        arr_2d[:, i] = arr_1d[start_index:end_index]

    return arr_2d


def distribute_array_2d(arr_1d, num_columns, num_rows=None):
    """
    If num_rows is not specified, calculate it based on the array length and num_columns.
    """
    if num_rows is None:
        num_rows = len(arr_1d) // num_columns

    # Ensure the input array has enough elements
    if len(arr_1d) < num_rows * num_columns:
        raise ValueError("Input array is too small for the specified dimensions")

    # Create an empty 2D array with appropriate dimensions
    arr_2d = np.empty((num_rows, num_columns))

    # Iterate through the 1D array and place elements in the 2D array
    for i in range(num_columns):
        start_index = i * num_rows
        end_index = (i + 1) * num_rows
        arr_2d[:, i] = arr_1d[start_index:end_index]

    return arr_2d


def f_solo_dissim_GM1(pts_per_timewin, num_timewin, fft_win, fs, tcm_rearrange):
    """
    Function for calculating dissimilarity.

    Args:
    pts_per_timewin: Points per time window.
    num_timewin: Number of time windows.
    fft_win: FFT window size.
    fs: Sample rate of the audio data.
    tcm_rearrange: Rearranged time chunk matrix.
    Returns:
    Dissimilarity values.
    """
    tcm_rearrange = np.array(tcm_rearrange)  # Uses numpy to perform mathematical operations

    pts_per_fft = int(fft_win * fs)  # Calc size fft window
    numfftwin = int(np.floor(pts_per_timewin / pts_per_fft))  # Number of fft windows

    D = []

    for kk in range(num_timewin - 1):
        analytic1 = hilbert(tcm_rearrange[:, kk], axis=-1)
        analytic2 = hilbert(tcm_rearrange[:, kk + 1])

        at1 = abs(analytic1) / np.sum(abs(analytic1))
        at2 = abs(analytic2) / np.sum(abs(analytic2))

        Dt = np.sum(abs(at1 - at2)) / 2

        s3a = tcm_rearrange[:, kk]
        s3a = s3a[:int(pts_per_fft * numfftwin)]
        s3a = create_2d_array_by_columns(s3a, pts_per_fft, numfftwin)

        s3a = np.array(s3a)
        ga = np.abs(np.fft.fft(s3a, axis=0)) / s3a.shape[0]

        sfa = np.mean(ga, axis=1)
        Sfa = abs(sfa) / np.sum(abs(sfa))

        s3b = tcm_rearrange[:, kk + 1]
        s3b = s3b[:int(pts_per_fft * numfftwin)]
        s3b = np.array(s3b)
        s3b = create_2d_array_by_columns(s3b, pts_per_fft, numfftwin)

        s3b = np.array(s3b)
        gb = np.abs(np.fft.fft(s3b, axis=0)) / s3b.shape[0]
        sfb = np.mean(gb, axis=1)
        Sfb = abs(sfb) / np.sum(abs(sfb))

        Df = np.sum(abs(Sfb - Sfa)) / 2
        Di = Dt * Df

        D.append(Di)

    Dfin = np.array(D)
    return Dfin


def correl_5(ts1, ts2, lags, offset):
    '''
'''
    Used to calculate autocorrelation
    '''
'''
    P = np.zeros(lags + 1)
    nlags = np.arange(0, lags + 1)

    for i in range(lags + 1):
        ng = 1
        sx = 2
        sy = 3
        sxx = 4
        syy = 5
        sxy = 6

        for k in range(len(ts1) - (i + offset)):
            x = ts1[k]
            y = ts2[k + (i + offset)]
            if not np.isnan(x) and not np.isnan(y):
                sx += x
                sy += y
                sxx += x * x
                syy += y * y
                sxy += x * y
                ng += 1

        covar1 = (sxy / ng) - ((sx / ng) * (sy / ng))
        denom1 = np.sqrt((sxx / ng) - (sx / ng) ** 2)
        denom2 = np.sqrt((syy / ng) - (sy / ng) ** 2)
        P[i] = covar1 / (denom1 * denom2)

    return P, nlags


def reshape_vertical(matrix):
    """
    Converts matrix dimensions from mxn to nxm
    """
    matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return matrix


def write_to_json(output_dir, SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim):
    """
    Write calculated metrics to JSON file

    Args:
    output_dir: Path to save the output JSON file.
    SPLrms: Matrix of SPLrms values.
    SPLpk: Matrix of SPLpk values.
    impulsivity: Matrix of impulsivity values.
    peakcount: Matrix of peak count values.
    autocorr: Matrix of autocorrelation values.
    dissim: Matrix of dissimilarity values.
    """
    # Create a dictionary to hold all the results
    result_data = {
        'SPLrms': SPLrms,
        'SPLpk': SPLpk,
        'impulsivity': impulsivity,
        'peakcount': peakcount,
        'autocorr': autocorr.tolist(),
        'dissim': dissim
    }

    # Write the dictionary to a JSON file
    with open(output_dir, 'w') as f:
        json.dump(result_data, f, indent=4)


file_dir = os.listdir(input_dir)
num_bits = 16
RS = -178.3  # BE SURE TO CHANGE FOR EACH HYDROPHONE
# Sensitivity is based on hydrophone, not recorder
peak_volts = 2
arti = 1  # Make 1 if calibration tone present

# Analysis options
timewin = 60  # Length of time window in seconds for analysis bins
fft_win = 1  # Length of fft window in minutes
avtime = 0.1
flow = 50  # Low frequency
fhigh = 300  # High frequency

if __name__ == '__main__':
    SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim = f_WAV_frankenfunction_reilly(
        num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh)

    # Change dimensions for mxn to nxm
    SPLrms = reshape_vertical(SPLrms)
    SPLpk = reshape_vertical(SPLpk)
    impulsivity = reshape_vertical(impulsivity)
    peakcount = reshape_vertical(peakcount)
    dissim = reshape_vertical(dissim)

    # Write the results to a JSON file
    write_to_json(output_dir, SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim)
'''

'''
import math
import numpy as np
import os
import warnings
import json
from scipy.io import wavfile
from scipy.signal import find_peaks, hilbert

warnings.filterwarnings("ignore")  # Ignore warning messages to avoid unnecessary output

# Define input/output directories
input_dir = "C:/Users/rlessard/Desktop/runThisInput/ble_two"  # Path containing .wav files
output_dir = "C:/Users/rlessard/Desktop/runThisOutput/runThisPython.json"  # Path to store .mat file


def downsample(x, N, phase=0):
    """
    Downsample input signal.

    Parameters:
        x : numpy.ndarray
            The input signal (1D or 2D).
        N : int
            The downsample factor. Keep every N-th sample.
        phase : int, optional
            The sample offset (default is 0).

    Returns:
        numpy.ndarray
            The downsampled signal.

    Examples:
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = downsample(x, 3)  # Downsample by 3
        y_with_phase = downsample(x, 3, 2)  # Downsample by 3 with phase offset of 2
    """

    # Validate inputs
    if not isinstance(x, np.ndarray):
        raise ValueError("Input signal 'x' must be a numpy ndarray.")

    if not isinstance(N, int) or N <= 0:
        raise ValueError("Downsample factor 'N' must be a positive integer.")

    if not isinstance(phase, int) or phase < 0 or phase >= N:
        raise ValueError(f"Phase offset 'phase' must be an integer in the range [0, {N - 1}].")

    # Downsample signal
    if x.ndim == 1:
        y = x[phase::N]  # Downsample 1D signal
    elif x.ndim == 2:
        y = x[phase::N, :]  # Downsample along rows for 2D signal
    else:
        raise ValueError("Input signal 'x' must be 1D or 2D.")

    return y


def upsample(x, N, phase=0):
    """
    Upsample input signal.

    Parameters:
        x : numpy.ndarray
            The input signal (1D or 2D numpy array).
        N : int
            The upsample factor. Insert N-1 zeros between input samples.
        phase : int, optional
            The sample offset (default is 0).

    Returns:
        numpy.ndarray
            The upsampled signal.

    Examples:
        x = np.array([1, 2, 3, 4])
        y = upsample(x, 3)  # Upsample by 3
        y_with_phase = upsample(x, 3, 2)  # Upsample by 3 with phase offset of 2
    """

    # Validate inputs
    if not isinstance(x, np.ndarray):
        raise ValueError("Input signal 'x' must be a numpy ndarray.")

    if not isinstance(N, int) or N <= 0:
        raise ValueError("Upsample factor 'N' must be a positive integer.")

    if not isinstance(phase, int) or phase < 0 or phase >= N:
        raise ValueError(f"Phase offset 'phase' must be an integer in the range [0, {N - 1}].")

    # Save original size of x (possibly N-D)
    size_x = x.shape

    # Total elements in x
    n_elements = x.size

    # Convert to a column vector (1D array)
    x_col = x.reshape(n_elements)

    # Create an array for the upsampled signal
    y_col = np.zeros(n_elements * N, dtype=x.dtype)

    # Perform the upsampling
    y_col[phase::N] = x_col

    # Update the dimensions to reflect upsampling
    new_size = list(size_x)
    new_size[0] = new_size[0] * N  # Update the first dimension if 1D signal

    # Restore N-D shape
    y = y_col.reshape(new_size)

    return y

def frequency_resample(fs, x):
    """
    Adjust audio data sample rate by down-sampling or up-sampling

    Args:
    fs: Original audio data sample rate
    x: Audio signal data

    Returns:
    fs: Updated sample rate after resampling
    x: Resampled audio data
    """

    if fs == 576000:
        x = downsample(x, 4)  # downsample by taking every 4th element
        fs = fs / 4
    elif fs == 288000:
        x = downsample(x, 2)  # downsample by taking every 2nd element
        fs = fs / 2
    elif fs == 16000:
        x = upsample(x, 9)  # upsample by repeating each element 9 times
        fs = fs * 9
    elif fs == 8000:
        x = upsample(x, 18)  # upsample by repeating each element 18 times
        fs = fs * 18
    elif fs == 512000:
        x = downsample(x, 4)  # downsample by taking every 4th element
        fs = fs / 3.5555555555555555555

    return fs, x


def minute_padding(num_timewin, pts_per_timewin, p_filt):
    """
    Pads filtered signal to ensure that length matches full minute time windows

    Args:
    num_timewin: Total number of time windows for signal
    pts_per_timewin: Number of parts in each time window
    p_filt: Filtered audio signal

    Returns:
    timechunk_matrix: Reshaped signal into a 2D matrix, where each column is a time window
    p_filt_padded: Padded filtered signal to match required length
    """

    padding_length = num_timewin * pts_per_timewin - len(p_filt)  # Calculate padding length required to fill all time windows
    p_filt_padded = np.concatenate((p_filt, np.zeros(padding_length)))  # Pad filtered signal with zeros
    timechunk_matrix = p_filt_padded.reshape(pts_per_timewin, num_timewin)  # Reshape padded signal into matrix with time window dimensions
    return timechunk_matrix, p_filt_padded


def calculate_impulsivity(impulsivity, tcm_rearrange):
    """
    Calculate impulsivity using kurtosis of rearranged time chunks

    Args:
    impulsivity: List to store impulsivity values
    tcm_rearrange: Rearranged time chunk matrix

    Returns:
    impulsivity: Updated list with calculated impulsivity values
    """

    kmat = kurtosis_reilly(np.array(tcm_rearrange))
    impulsivity.extend(kmat)
    return impulsivity


def calculate_spl(rms_matrix, tcm_rearrange, SPLrms, SPLpk):
    """
        Calculate SPL (Sound Pressure Level) in both RMS and Peak formats.

        Args:
        rms_matrix: RMS values of the time chunks.
        tcm_rearrange: Rearranged time chunk matrix.
        SPLrms: List to store SPL RMS values.
        SPLpk: List to store SPL Peak values.

        Returns:
        SPLrms: Updated list with SPL RMS values.
        SPLpk: Updated list with SPL Peak values.
        """

    SPLrmshold = 20 * np.log10(rms_matrix)
    SPLpkhold = 20 * np.log10(np.max(np.abs(tcm_rearrange), axis=0))
    # SPLpkhold = np.max(20 * np.log10(np.abs(tcm_rearrange)), axis=0)
    SPLrms.extend(SPLrmshold)
    SPLpk.extend(SPLpkhold)
    return SPLrms, SPLpk


def calculate_autocorrelation(autocorr, acorr):
    """
    Calculate autocorrelation for periodicity analysis.

    Args:
    autocorr: Current autocorrelation matrix.
    acorr: New autocorrelation values to append.

    Returns:
    autocorr: Updated autocorrelation matrix.
    """
    if autocorr is None:
        autocorr = acorr
    else:
        autocorr = np.column_stack((autocorr, acorr))
    return autocorr

def replace_with_zeros(x, num):
    """
    Leave 1/num values as original
    Replace all other values with 0
    """
    mask = np.arange(len(x)) % num == 0
    x = np.where(mask, x, 0)

    return x


def f_WAV_frankenfunction_reilly(num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh):
    """
    Main function to process WAV files and compute various soundscape metrics.

    Args:
    num_bits: Bit rate of the audio data.
    peak_volts: Peak voltage of the recorder.
    file_dir: Directory of files to process.
    RS: Hydrophone sensitivity in dB.
    timewin: Size of time windows for analysis (in seconds).
    avtime: Averaging time for autocorrelation.
    fft_win: FFT window size for analysis.
    arti: Indicates if the recording contains calibration tones (1 for true).
    flow: Lower frequency cutoff for bandpass filtering.
    fhigh: Upper frequency cutoff for bandpass filtering.

    Returns:
    SPLrms: Root mean square sound pressure level matrix.
    SPLpk: Peak sound pressure level matrix.
    impulsivity: Impulsivity measurements based on kurtosis.
    peakcount: Count of peaks in autocorrelation function.
    autocorr: Autocorrelation matrix for periodicity analysis.
    dissim: Dissimilarity matrix comparing adjacent time windows.
    """

    num_files = len(file_dir)  # Number of .wav files in input directory
    file_lengths = []  # Ensure that files of unequal lengths have appropriate output dimensions
    # print("num_files:", num_files)
    p = []  # Placeholder for audio data
    pout = []  # Placeholder for filtered audio data
    SPLrms = []  # Root-mean-square (RMS) sound pressure level (SPL) for each time window
    SPLpk = []  # Peak SPL for each time window
    impulsivity = []  # Placeholder for impulsivity metric
    peakcount = []  # Placeholder for periodicity metric (peak count)
    autocorr_list = None  # Placeholder for autocorrelation results
    dissim = []  # Placeholder for dissimilarity metric

    # Loop through each WAV file and process it
    for ii in range(num_files):
        # File and signal preparation
        filename = os.path.join(input_dir, file_dir[ii])
        rs = (10 ** (RS / 20))
        max_count = 2 ** num_bits
        conv_factor = peak_volts / max_count

        # Read and preprocess audio file
        fs, x = wavfile.read(filename)
        fs, x = frequency_resample(fs, x)
        x = replace_with_zeros(x, 9)

        if num_bits == 24:
            x = x >> 8

        v = x.astype(float) * conv_factor
        p = v / rs

        if arti == 1:
            p = p[6 * fs - 1:]

        # Bandpass filtering
        p_filt = dylan_bpfilt(p, 1 / fs, flow, fhigh)

        # Time window and file length calculation
        pts_per_timewin = int(timewin * fs)
        num_timewin = math.floor(len(p_filt) / pts_per_timewin) + 1
        file_length_minutes = math.floor(len(p_filt) / (fs * 60))
        file_lengths.append(file_length_minutes)

        # Signal padding
        timechunk_matrix, p_filt_padded = minute_padding(num_timewin, pts_per_timewin, p_filt)
        tcm_rearrange = create_2d_array_by_columns(p_filt_padded, pts_per_timewin, num_timewin)
        rms_matrix = rms_reilly(timechunk_matrix, 0)

        # Sound Pressure Level calculations
        SPLrms_current, SPLpk_current = calculate_spl(rms_matrix, tcm_rearrange, [], [])

        # Truncate to actual file length
        max_file_length = max(file_lengths)
        SPLrms_padded = (SPLrms_current[:file_length_minutes] +
                         [0] * (max_file_length - len(SPLrms_current[:file_length_minutes])))
        SPLpk_padded = (SPLpk_current[:file_length_minutes] +
                        [0] * (max_file_length - len(SPLpk_current[:file_length_minutes])))

        SPLrms.append(SPLrms_padded)
        SPLpk.append(SPLpk_padded)

        # Impulsivity (Kurtosis) calculation
        impulsivity_current = kurtosis_reilly(tcm_rearrange)[:file_length_minutes]
        impulsivity_padded = (list(impulsivity_current) +
                              [0] * (max_file_length - len(impulsivity_current)))
        impulsivity.append(impulsivity_padded)

        # Periodicity analysis
        pkcount, acorr = f_solo_per_GM2(p_filt_padded, fs, timewin, avtime)
        peakcount_current = pkcount[:file_length_minutes]
        peakcount_padded = (list(peakcount_current) +
                            [0] * (max_file_length - len(peakcount_current)))
        peakcount.append(peakcount_padded)

        # Autocorrelation
        if acorr is not None:
            # Truncate to file length and pad with zeros
            acorr_current = acorr[:, :file_length_minutes]
            acorr_padded = np.pad(acorr_current,
                                  ((0, 0), (0, max_file_length - acorr_current.shape[1])),
                                  mode='constant')
            autocorr_list = [autocorr_list, acorr_padded]
            # autocorr_list.append(acorr_padded)

        # Dissimilarity calculation
        Dfin = f_solo_dissim_GM1(pts_per_timewin, num_timewin, fft_win, fs, tcm_rearrange)
        dissim_current = Dfin[:file_length_minutes - 1]
        dissim_padded = (list(dissim_current) +
                         [0] * (max_file_length - 1 - len(dissim_current)))
        dissim.append(dissim_padded)

        # Combine autocorrelation results
    if autocorr_list:
        autocorr = autocorr_list
        # autocorr = np.stack(autocorr_list, axis=2)
    else:
        autocorr = None

    return SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim


def create_2d_array_by_columns(input_array, row, col):
    """
    Creates a 2D array by filling data column-wise.

    Args:
    input_array: Input data to fill the array.
    row: Number of rows.
    col: Number of columns.

    Returns:
    result: 2D array with data filled column-wise.
    """
    result = np.zeros((row, col))
    for c in range(col):
        for r in range(row):
            result[r][c] = input_array[c * row + r]
    return result


def column_max_SPL(timechunk_matrix):
    """
    Get the number of columns (assuming all rows have the same length)
    Returns the peak in each column
    """
    num_columns = len(timechunk_matrix[0])

    SPLpkhold = []
    for col in range(num_columns):
        column_values = [abs(row[col]) for row in timechunk_matrix if row[col] != 0]
        if column_values:
            max_value = max(column_values)
            spl = 20 * math.log10(max_value)
            SPLpkhold.append(spl)
        else:
            SPLpkhold.append(float('-inf'))  # Used to represent undefined SPL value

    return SPLpkhold


def kurtosis_reilly(x, flag=1, dim=None):
    """
    Calculates kurtosis of the input data.

    Args:
    x: Input data for kurtosis calculation.

    Returns:
    k: Kurtosis values.
    """
    flag = 1

    # Determine the dimension if not provided
    if flag not in (0, 1) and flag is not None:
        raise ValueError("Bad flag value: flag should be 0 or 1.")

    if dim is None:
        # Handle the special case where x is empty.
        if np.array_equal(x, np.array([])):
            print("x is empty")
            return np.nan

        # Determine the dimension along which np.nanmean will work.
        dim = next((i for i, s in enumerate(x.shape) if s != 1), None)

        if dim is None:
            dim = 0

    x0 = x - np.nanmean(x, axis=dim, keepdims=True)

    s2 = nanmean(x0 ** 2, axis=dim)  # biased variance estimator
    m4 = np.nanmean(np.power(x0, 4), axis=0)

    k = m4 / np.square(s2)  # Determine element-wise square

    # Bias correction
    if flag == 0:
        n = np.sum(~np.isnan(x), axis=dim)
        n[n < 4] = np.nan  # bias correction not defined for n < 4
        k = ((n + 1) * k - 3 * (n - 1)) * (n - 1) / ((n - 2) * (n - 3)) + 3

    return k


def nanmean(arr, axis=None):
    """
    Convert the input to a numpy array if it isn't already
    """
    arr = np.array(arr)

    # Create a mask to ignore NaN, 0, and empty values
    mask = ~np.isnan(arr) & (arr != 0) & (arr != "")

    # Apply the mask and calculate the mean
    masked_arr = np.where(mask, arr, np.nan)
    return np.nanmean(masked_arr, axis=axis)


def rms_reilly(x, dim=None):
    ''''''
    Rewritten from MATLAB
    For vectors, RMS(X) is the root mean square value in X.
    For matrices, RMS(X) is a row vector containing the RMS value from each column.
    Y = RMS(X,DIM) operates along the dimension DIM.
    ''''''

    global vertical_averages_sqrt
    if np.isrealobj(x):
        if dim is not None:
            sqmexx = np.square(x)
            [sqmexx_a, sqmexx_b] = np.shape(sqmexx)

            sqmexx = sqmexx.reshape(-1)

            reshaped_sqmexx = sqmexx.reshape((sqmexx_b, sqmexx_a))
            transposed_sqmexx = reshaped_sqmexx.T

            vertical_averages = np.mean(transposed_sqmexx, axis=0)
            vertical_averages_sqrt = np.sqrt(vertical_averages)

    return vertical_averages_sqrt


def bpfilt_initial_values(ts, samint):
    """
    Define initial values for dylan_bpfilt
    """
    npts = len(ts)
    reclen = npts * samint
    spec = np.fft.fft(ts, npts)
    aspec = np.abs(spec)
    pspec = np.angle(spec)
    freq = np.fft.fftshift(np.arange(-npts / 2, npts / 2) / reclen)

    return npts, spec, aspec, pspec, freq


def dylan_bpfilt(ts, samint, flow, fhigh):
    """
    Performs bandpass filtering.

    Args:
    data: Input audio data.
    fs_inv: Inverse of the sample rate.
    flow: Lower frequency bound for filtering.
    fhigh: Upper frequency bound for filtering.

    Returns:
    Filtered data.
    """

    npts, spec, aspec, pspec, freq = bpfilt_initial_values(ts, samint)

    if fhigh == 0:
        fhigh = 1 / (2 * samint)

    ifr = np.where((np.abs(freq) >= flow) & (np.abs(freq) <= fhigh))[0]  # Calculate metrics between flow and fhigh
    rspec = np.zeros_like(spec)
    ispec = np.zeros_like(spec)

    rspec[ifr] = aspec[ifr] * np.cos(pspec[ifr])
    ispec[ifr] = aspec[ifr] * np.sin(pspec[ifr])
    filtspec2 = rspec + 1j * ispec

    tsfilt = np.real(np.fft.ifft(filtspec2, npts))

    return tsfilt


def f_solo_per_GM2(p_filt, fs, timewin, avtime):
    """
    Function for calculating peak count and autocorrelation.

    Args:
    data: Input audio data.
    fs: Sample rate of the audio data.
    timewin: Time window size for analysis.
    avtime: Averaging time for autocorrelation.

    Returns:
    Peak count and autocorrelation values.
    """
    p_av = []
    avwin = int(fs * avtime)
    sampwin = int(fs * timewin)
    ntwin = len(p_filt) // sampwin  # Number of minutes
    p_filt = p_filt[:sampwin * ntwin]

    p_filt = distribute_array(p_filt, ntwin)

    p_filt = p_filt ** 2

    numavwin = p_filt.shape[0] // avwin

    for jj in range(ntwin):
        avwinmatrix = distribute_array_2d(p_filt[:, jj], numavwin, avwin)

        p_avi = np.mean(avwinmatrix, axis=0)
        p_av.append(p_avi)

    p_av = np.transpose(p_av)
    p_avtot = np.array(p_av)

    shape0, shape1 = np.shape(p_avtot)

    max_lag = int(shape0 * 0.7)
    acorr = np.zeros((max_lag + 1, shape1))
    pkcount = np.zeros(shape1)

    for zz in range(shape1):
        P, _ = correl_5(p_avtot[:, zz], p_avtot[:, zz], max_lag, 0)
        acorr[:, zz] = P
        pks, _ = find_peaks(acorr[:, zz], prominence=0.5)
        pkcount[zz] = len(pks)

    return pkcount, acorr


def distribute_array(arr_1d, dim):
    """
    Calculate the number of elements per column.
    Create a matrix, where the number of columns is equal to number of minutes in recording.
    Appends vectors to create 2D matrix
    """
    elements_per_column = len(arr_1d) // dim

    arr_2d = np.empty((elements_per_column, dim))

    for i in range(dim):
        start_index = i * elements_per_column
        end_index = (i + 1) * elements_per_column
        arr_2d[:, i] = arr_1d[start_index:end_index]

    return arr_2d


def distribute_array_2d(arr_1d, num_columns, num_rows=None):
    """
    If num_rows is not specified, calculate it based on the array length and num_columns.
    """
    if num_rows is None:
        num_rows = len(arr_1d) // num_columns

    # Ensure the input array has enough elements
    if len(arr_1d) < num_rows * num_columns:
        raise ValueError("Input array is too small for the specified dimensions")

    # Create an empty 2D array with appropriate dimensions
    arr_2d = np.empty((num_rows, num_columns))

    # Iterate through the 1D array and place elements in the 2D array
    for i in range(num_columns):
        start_index = i * num_rows
        end_index = (i + 1) * num_rows
        arr_2d[:, i] = arr_1d[start_index:end_index]

    return arr_2d


def f_solo_dissim_GM1(pts_per_timewin, num_timewin, fft_win, fs, tcm_rearrange):
    """
    Function for calculating dissimilarity.

    Args:
    pts_per_timewin: Points per time window.
    num_timewin: Number of time windows.
    fft_win: FFT window size.
    fs: Sample rate of the audio data.
    tcm_rearrange: Rearranged time chunk matrix.
    Returns:
    Dissimilarity values.
    """
    tcm_rearrange = np.array(tcm_rearrange)  # Uses numpy to perform mathematical operations

    pts_per_fft = int(fft_win * fs)  # Calc size fft window
    numfftwin = int(np.floor(pts_per_timewin / pts_per_fft))  # Number of fft windows

    D = []

    for kk in range(num_timewin - 1):
        analytic1 = hilbert(tcm_rearrange[:, kk], axis=-1)
        analytic2 = hilbert(tcm_rearrange[:, kk + 1])

        at1 = abs(analytic1) / np.sum(abs(analytic1))
        at2 = abs(analytic2) / np.sum(abs(analytic2))

        Dt = np.sum(abs(at1 - at2)) / 2

        s3a = tcm_rearrange[:, kk]
        s3a = s3a[:int(pts_per_fft * numfftwin)]
        s3a = create_2d_array_by_columns(s3a, pts_per_fft, numfftwin)

        s3a = np.array(s3a)
        ga = np.abs(np.fft.fft(s3a, axis=0)) / s3a.shape[0]

        sfa = np.mean(ga, axis=1)
        Sfa = abs(sfa) / np.sum(abs(sfa))

        s3b = tcm_rearrange[:, kk + 1]
        s3b = s3b[:int(pts_per_fft * numfftwin)]
        s3b = np.array(s3b)
        s3b = create_2d_array_by_columns(s3b, pts_per_fft, numfftwin)

        s3b = np.array(s3b)
        gb = np.abs(np.fft.fft(s3b, axis=0)) / s3b.shape[0]
        sfb = np.mean(gb, axis=1)
        Sfb = abs(sfb) / np.sum(abs(sfb))

        Df = np.sum(abs(Sfb - Sfa)) / 2
        Di = Dt * Df

        D.append(Di)

    Dfin = np.array(D)
    return Dfin


def correl_5(ts1, ts2, lags, offset):
    ''''''
    Used to calculate autocorrelation
    ''''''
    P = np.zeros(lags + 1)
    nlags = np.arange(0, lags + 1)

    for i in range(lags + 1):
        ng = 1
        sx = 2
        sy = 3
        sxx = 4
        syy = 5
        sxy = 6

        for k in range(len(ts1) - (i + offset)):
            x = ts1[k]
            y = ts2[k + (i + offset)]
            if not np.isnan(x) and not np.isnan(y):
                sx += x
                sy += y
                sxx += x * x
                syy += y * y
                sxy += x * y
                ng += 1

        covar1 = (sxy / ng) - ((sx / ng) * (sy / ng))
        denom1 = np.sqrt((sxx / ng) - (sx / ng) ** 2)
        denom2 = np.sqrt((syy / ng) - (sy / ng) ** 2)
        P[i] = covar1 / (denom1 * denom2)

    return P, nlags


# def reshape_vertical(matrix):
#     """
#     Converts matrix dimensions from mxn to nxm
#     """
#     matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
#     return matrix


def pad_to_uniform_length(data, pad_value=0):
    """
    Pads each row in data to ensure all rows have equal length.

    Args:
        data: List of lists or arrays with varying lengths.
        pad_value: Value to pad shorter rows with.

    Returns:
        A 2D NumPy array with uniform row lengths, or an empty array if no valid data.
    """
    if not data:
        return np.array([])  # Return an empty array if data is empty

    # Flatten any nested structures and filter out invalid entries
    flattened_data = []
    for item in data:
        if item is None:
            continue
        if isinstance(item, list):
            for sub_item in item:
                if isinstance(sub_item, np.ndarray):
                    flattened_data.append(sub_item)
        elif isinstance(item, np.ndarray):
            flattened_data.append(item)

    if not flattened_data:  # Handle case where all entries are invalid
        return np.array([])

    # Ensure all rows are lists and calculate max length
    max_length = max(row.shape[1] if row.ndim > 1 else len(row) for row in flattened_data)
    padded_data = [
        np.pad(row, ((0, 0), (0, max_length - row.shape[1])), constant_values=pad_value) if row.ndim > 1 else
        np.pad(row, (0, max_length - len(row)), constant_values=pad_value)
        for row in flattened_data
    ]
    return np.array(padded_data)


def reshape_vertical(matrix):
    """
    Reshape a 2D matrix from m x n to n x m
    """
    return matrix.T  # Assuming reshape_vertical transposes the array


def write_to_json(output_dir, SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim):
    """
    Write calculated metrics to JSON file
    """
    result_data = {
        "SPLrms": SPLrms.tolist() if isinstance(SPLrms, np.ndarray) else SPLrms,
        "SPLpk": SPLpk.tolist() if isinstance(SPLpk, np.ndarray) else SPLpk,
        "Impulsivity": impulsivity.tolist() if isinstance(impulsivity, np.ndarray) else impulsivity,
        "PeakCount": peakcount.tolist() if isinstance(peakcount, np.ndarray) else peakcount,
        "Autocorr": autocorr.tolist() if isinstance(autocorr, np.ndarray) else autocorr,
        "Dissim": dissim.tolist() if isinstance(dissim, np.ndarray) else dissim,
    }

    with open(output_dir, 'w') as out_file:
        json.dump(result_data, out_file, indent=4)


if __name__ == '__main__':
    file_dir = os.listdir(input_dir)
    num_bits = 16
    RS = -178.3  # Hydrophone sensitivity, change as needed
    peak_volts = 2
    arti = 1  # Calibration tone present
    # Analysis options
    timewin = 60  # Analysis window in seconds
    fft_win = 1  # FFT window in minutes
    avtime = 0.1
    flow = 50  # Low frequency cutoff
    fhigh = 300  # High frequency cutoff

    # Call your main processing function
    SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim = f_WAV_frankenfunction_reilly(
        num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh
    )

    # Reshape and pad matrices
    SPLrms = reshape_vertical(pad_to_uniform_length(SPLrms))
    SPLpk = reshape_vertical(pad_to_uniform_length(SPLpk))
    impulsivity = reshape_vertical(pad_to_uniform_length(impulsivity))
    peakcount = reshape_vertical(pad_to_uniform_length(peakcount))

    # print(f"autocorr before filtering: {autocorr}")

    if autocorr:
        # Remove invalid entries and pad/reshape
        autocorr_filtered = [item for item in autocorr if item is not None]
        autocorr = reshape_vertical(pad_to_uniform_length(autocorr_filtered))
    else:
        autocorr = np.array([])  # Handle empty autocorr gracefully

    # print(f"autocorr after processing: {autocorr.shape}")

    dissim = reshape_vertical(pad_to_uniform_length(dissim))

    # Write to JSON
    write_to_json(output_dir, SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim)
'''

import math
import numpy as np
import os
import warnings
import json
from scipy.io import wavfile
from scipy.signal import find_peaks, hilbert
import soundfile as sf

warnings.filterwarnings("ignore")  # Ignore warning messages to avoid unnecessary output


def downsample(x, N, phase=0):
    """
    Downsample input signal.

    Parameters:
        x : numpy.ndarray
            The input signal (1D or 2D).
        N : int
            The downsample factor. Keep every N-th sample.
        phase : int, optional
            The sample offset (default is 0).

    Returns:
        numpy.ndarray
            The downsampled signal.

    Examples:
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = downsample(x, 3)  # Downsample by 3
        y_with_phase = downsample(x, 3, 2)  # Downsample by 3 with phase offset of 2
    """

    # Validate inputs
    if not isinstance(x, np.ndarray):
        raise ValueError("Input signal 'x' must be a numpy ndarray.")

    if not isinstance(N, int) or N <= 0:
        raise ValueError("Downsample factor 'N' must be a positive integer.")

    if not isinstance(phase, int) or phase < 0 or phase >= N:
        raise ValueError(f"Phase offset 'phase' must be an integer in the range [0, {N - 1}].")

    # Downsample signal
    if x.ndim == 1:
        y = x[phase::N]  # Downsample 1D signal
    elif x.ndim == 2:
        y = x[phase::N, :]  # Downsample along rows for 2D signal
    else:
        raise ValueError("Input signal 'x' must be 1D or 2D.")

    return y


def upsample(x, N, phase=0):
    """
    Upsample input signal.

    Parameters:
        x : numpy.ndarray
            The input signal (1D or 2D numpy array).
        N : int
            The upsample factor. Insert N-1 zeros between input samples.
        phase : int, optional
            The sample offset (default is 0).

    Returns:
        numpy.ndarray
            The upsampled signal.

    Examples:
        x = np.array([1, 2, 3, 4])
        y = upsample(x, 3)  # Upsample by 3
        y_with_phase = upsample(x, 3, 2)  # Upsample by 3 with phase offset of 2
    """

    # Validate inputs
    if not isinstance(x, np.ndarray):
        raise ValueError("Input signal 'x' must be a numpy ndarray.")

    if not isinstance(N, int) or N <= 0:
        raise ValueError("Upsample factor 'N' must be a positive integer.")

    if not isinstance(phase, int) or phase < 0 or phase >= N:
        raise ValueError(f"Phase offset 'phase' must be an integer in the range [0, {N - 1}].")

    # Save original size of x (possibly N-D)
    size_x = x.shape

    # Total elements in x
    n_elements = x.size

    # Convert to a column vector (1D array)
    x_col = x.reshape(n_elements)

    # Create an array for the upsampled signal
    y_col = np.zeros(n_elements * N, dtype=x.dtype)

    # Perform the upsampling
    y_col[phase::N] = x_col

    # Update the dimensions to reflect upsampling
    new_size = list(size_x)
    new_size[0] = new_size[0] * N  # Update the first dimension if 1D signal

    # Restore N-D shape
    y = y_col.reshape(new_size)

    return y

def frequency_resample(fs, x):
    """
    Adjust audio data sample rate by down-sampling or up-sampling

    Args:
    fs: Original audio data sample rate
    x: Audio signal data

    Returns:
    fs: Updated sample rate after resampling
    x: Resampled audio data
    """

    if fs == 576000:
        x = downsample(x, 4)  # downsample by taking every 4th element
        fs = fs / 4
    elif fs == 288000:
        x = downsample(x, 2)  # downsample by taking every 2nd element
        fs = fs / 2
    elif fs == 16000:
        x = upsample(x, 9)  # upsample by repeating each element 9 times
        fs = fs * 9
    elif fs == 8000:
        x = upsample(x, 18)  # upsample by repeating each element 18 times
        fs = fs * 18
    elif fs == 512000:
        x = downsample(x, 4)  # downsample by taking every 4th element
        fs = fs / 3.5555555555555555555

    print("fs:", fs)
    print("x:", x)

    return fs, x


def minute_padding(num_timewin, pts_per_timewin, p_filt):
    """
    Pads filtered signal to ensure that length matches full minute time windows

    Args:
    num_timewin: Total number of time windows for signal
    pts_per_timewin: Number of parts in each time window
    p_filt: Filtered audio signal

    Returns:
    timechunk_matrix: Reshaped signal into a 2D matrix, where each column is a time window
    p_filt_padded: Padded filtered signal to match required length
    """

    padding_length = num_timewin * pts_per_timewin - len(p_filt)  # Calculate padding length required to fill all time windows
    p_filt_padded = np.concatenate((p_filt, np.zeros(padding_length)))  # Pad filtered signal with zeros
    timechunk_matrix = p_filt_padded.reshape(pts_per_timewin, num_timewin)  # Reshape padded signal into matrix with time window dimensions
    return timechunk_matrix, p_filt_padded


def calculate_impulsivity(impulsivity, tcm_rearrange):
    """
    Calculate impulsivity using kurtosis of rearranged time chunks

    Args:
    impulsivity: List to store impulsivity values
    tcm_rearrange: Rearranged time chunk matrix

    Returns:
    impulsivity: Updated list with calculated impulsivity values
    """

    kmat = kurtosis_reilly(np.array(tcm_rearrange))
    impulsivity.extend(kmat)
    return impulsivity


def calculate_spl(rms_matrix, tcm_rearrange, SPLrms, SPLpk):
    """
        Calculate SPL (Sound Pressure Level) in both RMS and Peak formats.

        Args:
        rms_matrix: RMS values of the time chunks.
        tcm_rearrange: Rearranged time chunk matrix.
        SPLrms: List to store SPL RMS values.
        SPLpk: List to store SPL Peak values.

        Returns:
        SPLrms: Updated list with SPL RMS values.
        SPLpk: Updated list with SPL Peak values.
        """

    SPLrmshold = 20 * np.log10(rms_matrix)
    SPLpkhold = 20 * np.log10(np.max(np.abs(tcm_rearrange), axis=0))
    # SPLpkhold = np.max(20 * np.log10(np.abs(tcm_rearrange)), axis=0)
    SPLrms.extend(SPLrmshold)
    SPLpk.extend(SPLpkhold)
    return SPLrms, SPLpk


def calculate_autocorrelation(autocorr, acorr):
    """
    Calculate autocorrelation for periodicity analysis.

    Args:
    autocorr: Current autocorrelation matrix.
    acorr: New autocorrelation values to append.

    Returns:
    autocorr: Updated autocorrelation matrix.
    """
    if autocorr is None:
        autocorr = acorr
    else:
        autocorr = np.column_stack((autocorr, acorr))
    return autocorr

def replace_with_zeros(x, num):
    """
    Leave 1/num values as original
    Replace all other values with 0
    """
    mask = np.arange(len(x)) % num == 0
    x = np.where(mask, x, 0)

    return x


def f_WAV_frankenfunction_reilly(num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh, output_directory):
    """
    Main function to process WAV files and compute various soundscape metrics.

    Args:
    num_bits: Bit rate of the audio data.
    peak_volts: Peak voltage of the recorder.
    file_dir: Directory of files to process.
    RS: Hydrophone sensitivity in dB.
    timewin: Size of time windows for analysis (in seconds).
    avtime: Averaging time for autocorrelation.
    fft_win: FFT window size for analysis.
    arti: Indicates if the recording contains calibration tones (1 for true).
    flow: Lower frequency cutoff for bandpass filtering.
    fhigh: Upper frequency cutoff for bandpass filtering.

    Returns:
    SPLrms: Root mean square sound pressure level matrix.
    SPLpk: Peak sound pressure level matrix.
    impulsivity: Impulsivity measurements based on kurtosis.
    peakcount: Count of peaks in autocorrelation function.
    autocorr: Autocorrelation matrix for periodicity analysis.
    dissim: Dissimilarity matrix comparing adjacent time windows.
    """

    num_files = len(file_dir)  # Number of .wav files in input directory
    # print("num_files:", num_files)
    p = []  # Placeholder for audio data
    pout = []  # Placeholder for filtered audio data
    SPLrms = []  # Root-mean-square (RMS) sound pressure level (SPL) for each time window
    SPLpk = []  # Peak SPL for each time window
    impulsivity = []  # Placeholder for impulsivity metric
    peakcount = []  # Placeholder for periodicity metric (peak count)
    autocorr = None  # Placeholder for autocorrelation results
    dissim = []  # Placeholder for dissimilarity metric

    # Loop through each WAV file and process it
    for ii in range(num_files):
        filename = os.path.join(output_directory, file_dir[ii])  # Get full path of current file
        rs = (10 ** (RS / 20))  # Convert hydrophone sensitivity from dB to a linear scale
        max_count = 2 ** num_bits  # Calculate maximum count based on bit depth of audio data
        conv_factor = peak_volts / max_count  # Calculate conversion factor for voltage

        # Determine sample rate and audio data from .wav file
        fs, x = wavfile.read(filename)

        # Downsample if sample rate is too high
        fs, x = frequency_resample(fs, x)
        x = replace_with_zeros(x, 9)

        if num_bits == 24:
            x = x >> 8  # Bit shift; accounts for audioread casting of 24 bit to 32 bit (zeroes behind)

        v = x.astype(float) * conv_factor

        p = v / rs  # Voltage to pressure
        if arti == 1:
            p = p[6 * fs - 1:]  # trims first 4 sec of recording to remove calibration tone
        pout.extend(p)  # Make this so the new p gets added at end of original p

        p_filt = dylan_bpfilt(pout, 1 / fs, flow, fhigh)
        pout = []
        pts_per_timewin = int(timewin * fs)  # Number of samples per time window - set window * 576 kHz sample rate

        num_timewin = math.floor(len(p_filt) / pts_per_timewin) + 1  # Number of time windows contained in the sound file

        # Pad the signal
        timechunk_matrix, p_filt_padded = minute_padding(num_timewin, pts_per_timewin, p_filt)

        # Create matrices with correct dimensions
        tcm_rearrange = create_2d_array_by_columns(p_filt_padded, pts_per_timewin, num_timewin)
        rms_matrix = rms_reilly(timechunk_matrix, 0)

        SPLrms, SPLpk = calculate_spl(rms_matrix, tcm_rearrange, SPLrms, SPLpk)

        # Compute impulsivity (kurtosis) for the rearranged time windows
        calculate_impulsivity(impulsivity, tcm_rearrange)

        # Periodicity
        pkcount, acorr = f_solo_per_GM2(p_filt_padded, fs, timewin, avtime)
        peakcount.extend(pkcount)

        # Autocorrelation
        autocorr = calculate_autocorrelation(autocorr, acorr)

        # Calculate dissimilarity between adjacent time windows (D-index)
        Dfin = f_solo_dissim_GM1(pts_per_timewin, num_timewin, fft_win, fs, tcm_rearrange)
        dissim.extend(Dfin)

    # Reshape matrices (Convert from vector to matrix)
    dissim = np.reshape(dissim, (num_files, int(len(dissim) / num_files)))
    impulsivity = np.reshape(impulsivity, (num_files, int(len(impulsivity) / num_files)))
    peakcount = np.reshape(peakcount, (num_files, int(len(peakcount) / num_files)))
    SPLpk = np.reshape(SPLpk, (num_files, int(len(SPLpk) / num_files)))
    SPLrms = np.reshape(SPLrms, (num_files, int(len(SPLrms) / num_files)))

    # Round autocorrelation values to avoid floating-point precision issues
    autocorr = np.round(autocorr, 15)

    return SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim


def create_2d_array_by_columns(input_array, row, col):
    """
    Creates a 2D array by filling data column-wise.

    Args:
    input_array: Input data to fill the array.
    row: Number of rows.
    col: Number of columns.

    Returns:
    result: 2D array with data filled column-wise.
    """
    result = np.zeros((row, col))
    for c in range(col):
        for r in range(row):
            result[r][c] = input_array[c * row + r]
    return result


def column_max_SPL(timechunk_matrix):
    """
    Get the number of columns (assuming all rows have the same length)
    Returns the peak in each column
    """
    num_columns = len(timechunk_matrix[0])

    SPLpkhold = []
    for col in range(num_columns):
        column_values = [abs(row[col]) for row in timechunk_matrix if row[col] != 0]
        if column_values:
            max_value = max(column_values)
            spl = 20 * math.log10(max_value)
            SPLpkhold.append(spl)
        else:
            SPLpkhold.append(float('-inf'))  # Used to represent undefined SPL value

    return SPLpkhold


def kurtosis_reilly(x, flag=1, dim=None):
    """
    Calculates kurtosis of the input data.

    Args:
    x: Input data for kurtosis calculation.

    Returns:
    k: Kurtosis values.
    """
    flag = 1

    # Determine the dimension if not provided
    if flag not in (0, 1) and flag is not None:
        raise ValueError("Bad flag value: flag should be 0 or 1.")

    if dim is None:
        # Handle the special case where x is empty.
        if np.array_equal(x, np.array([])):
            print("x is empty")
            return np.nan

        # Determine the dimension along which np.nanmean will work.
        dim = next((i for i, s in enumerate(x.shape) if s != 1), None)

        if dim is None:
            dim = 0

    x0 = x - np.nanmean(x, axis=dim, keepdims=True)

    s2 = nanmean(x0 ** 2, axis=dim)  # biased variance estimator
    m4 = np.nanmean(np.power(x0, 4), axis=0)

    k = m4 / np.square(s2)  # Determine element-wise square

    # Bias correction
    if flag == 0:
        n = np.sum(~np.isnan(x), axis=dim)
        n[n < 4] = np.nan  # bias correction not defined for n < 4
        k = ((n + 1) * k - 3 * (n - 1)) * (n - 1) / ((n - 2) * (n - 3)) + 3

    return k


def nanmean(arr, axis=None):
    """
    Convert the input to a numpy array if it isn't already
    """
    arr = np.array(arr)

    # Create a mask to ignore NaN, 0, and empty values
    mask = ~np.isnan(arr) & (arr != 0) & (arr != "")

    # Apply the mask and calculate the mean
    masked_arr = np.where(mask, arr, np.nan)
    return np.nanmean(masked_arr, axis=axis)


def rms_reilly(x, dim=None):
    '''
    Rewritten from MATLAB
    For vectors, RMS(X) is the root mean square value in X.
    For matrices, RMS(X) is a row vector containing the RMS value from each column.
    Y = RMS(X,DIM) operates along the dimension DIM.
    '''

    global vertical_averages_sqrt
    if np.isrealobj(x):
        if dim is not None:
            sqmexx = np.square(x)
            [sqmexx_a, sqmexx_b] = np.shape(sqmexx)

            sqmexx = sqmexx.reshape(-1)

            reshaped_sqmexx = sqmexx.reshape((sqmexx_b, sqmexx_a))
            transposed_sqmexx = reshaped_sqmexx.T

            vertical_averages = np.mean(transposed_sqmexx, axis=0)
            vertical_averages_sqrt = np.sqrt(vertical_averages)

    return vertical_averages_sqrt


def bpfilt_initial_values(ts, samint):
    """
    Define initial values for dylan_bpfilt
    """
    npts = len(ts)
    reclen = npts * samint
    spec = np.fft.fft(ts, npts)
    aspec = np.abs(spec)
    pspec = np.angle(spec)
    freq = np.fft.fftshift(np.arange(-npts / 2, npts / 2) / reclen)

    return npts, spec, aspec, pspec, freq


def dylan_bpfilt(ts, samint, flow, fhigh):
    """
    Performs bandpass filtering.

    Args:
    data: Input audio data.
    fs_inv: Inverse of the sample rate.
    flow: Lower frequency bound for filtering.
    fhigh: Upper frequency bound for filtering.

    Returns:
    Filtered data.
    """
    npts, spec, aspec, pspec, freq = bpfilt_initial_values(ts, samint)

    if fhigh == 0:
        fhigh = 1 / (2 * samint)

    ifr = np.where((np.abs(freq) >= flow) & (np.abs(freq) <= fhigh))[0]  # Calculate metrics between flow and fhigh
    rspec = np.zeros_like(spec)
    ispec = np.zeros_like(spec)

    rspec[ifr] = aspec[ifr] * np.cos(pspec[ifr])
    ispec[ifr] = aspec[ifr] * np.sin(pspec[ifr])
    filtspec2 = rspec + 1j * ispec

    tsfilt = np.real(np.fft.ifft(filtspec2, npts))

    return tsfilt


def f_solo_per_GM2(p_filt, fs, timewin, avtime):
    """
    Function for calculating peak count and autocorrelation.

    Args:
    data: Input audio data.
    fs: Sample rate of the audio data.
    timewin: Time window size for analysis.
    avtime: Averaging time for autocorrelation.

    Returns:
    Peak count and autocorrelation values.
    """
    p_av = []
    avwin = int(fs * avtime)
    sampwin = int(fs * timewin)
    ntwin = len(p_filt) // sampwin  # Number of minutes
    p_filt = p_filt[:sampwin * ntwin]

    p_filt = distribute_array(p_filt, ntwin)

    p_filt = p_filt ** 2

    numavwin = p_filt.shape[0] // avwin

    for jj in range(ntwin):
        avwinmatrix = distribute_array_2d(p_filt[:, jj], numavwin, avwin)

        p_avi = np.mean(avwinmatrix, axis=0)
        p_av.append(p_avi)

    p_av = np.transpose(p_av)
    p_avtot = np.array(p_av)

    shape0, shape1 = np.shape(p_avtot)

    max_lag = int(shape0 * 0.7)
    acorr = np.zeros((max_lag + 1, shape1))
    pkcount = np.zeros(shape1)

    for zz in range(shape1):
        P, _ = correl_5(p_avtot[:, zz], p_avtot[:, zz], max_lag, 0)
        acorr[:, zz] = P
        pks, _ = find_peaks(acorr[:, zz], prominence=0.5)
        pkcount[zz] = len(pks)

    return pkcount, acorr


def distribute_array(arr_1d, dim):
    """
    Calculate the number of elements per column.
    Create a matrix, where the number of columns is equal to number of minutes in recording.
    Appends vectors to create 2D matrix
    """
    elements_per_column = len(arr_1d) // dim

    arr_2d = np.empty((elements_per_column, dim))

    for i in range(dim):
        start_index = i * elements_per_column
        end_index = (i + 1) * elements_per_column
        arr_2d[:, i] = arr_1d[start_index:end_index]

    return arr_2d


def distribute_array_2d(arr_1d, num_columns, num_rows=None):
    """
    If num_rows is not specified, calculate it based on the array length and num_columns.
    """
    if num_rows is None:
        num_rows = len(arr_1d) // num_columns

    # Ensure the input array has enough elements
    if len(arr_1d) < num_rows * num_columns:
        raise ValueError("Input array is too small for the specified dimensions")

    # Create an empty 2D array with appropriate dimensions
    arr_2d = np.empty((num_rows, num_columns))

    # Iterate through the 1D array and place elements in the 2D array
    for i in range(num_columns):
        start_index = i * num_rows
        end_index = (i + 1) * num_rows
        arr_2d[:, i] = arr_1d[start_index:end_index]

    return arr_2d


def f_solo_dissim_GM1(pts_per_timewin, num_timewin, fft_win, fs, tcm_rearrange):
    """
    Function for calculating dissimilarity.

    Args:
    pts_per_timewin: Points per time window.
    num_timewin: Number of time windows.
    fft_win: FFT window size.
    fs: Sample rate of the audio data.
    tcm_rearrange: Rearranged time chunk matrix.
    Returns:
    Dissimilarity values.
    """
    tcm_rearrange = np.array(tcm_rearrange)  # Uses numpy to perform mathematical operations

    pts_per_fft = int(fft_win * fs)  # Calc size fft window
    numfftwin = int(np.floor(pts_per_timewin / pts_per_fft))  # Number of fft windows

    D = []

    for kk in range(num_timewin - 1):
        analytic1 = hilbert(tcm_rearrange[:, kk], axis=-1)
        analytic2 = hilbert(tcm_rearrange[:, kk + 1])

        at1 = abs(analytic1) / np.sum(abs(analytic1))
        at2 = abs(analytic2) / np.sum(abs(analytic2))

        Dt = np.sum(abs(at1 - at2)) / 2

        s3a = tcm_rearrange[:, kk]
        s3a = s3a[:int(pts_per_fft * numfftwin)]
        s3a = create_2d_array_by_columns(s3a, pts_per_fft, numfftwin)

        s3a = np.array(s3a)
        ga = np.abs(np.fft.fft(s3a, axis=0)) / s3a.shape[0]

        sfa = np.mean(ga, axis=1)
        Sfa = abs(sfa) / np.sum(abs(sfa))

        s3b = tcm_rearrange[:, kk + 1]
        s3b = s3b[:int(pts_per_fft * numfftwin)]
        s3b = np.array(s3b)
        s3b = create_2d_array_by_columns(s3b, pts_per_fft, numfftwin)

        s3b = np.array(s3b)
        gb = np.abs(np.fft.fft(s3b, axis=0)) / s3b.shape[0]
        sfb = np.mean(gb, axis=1)
        Sfb = abs(sfb) / np.sum(abs(sfb))

        Df = np.sum(abs(Sfb - Sfa)) / 2
        Di = Dt * Df

        D.append(Di)

    Dfin = np.array(D)
    return Dfin


def correl_5(ts1, ts2, lags, offset):
    '''
    Used to calculate autocorrelation
    '''
    P = np.zeros(lags + 1)
    nlags = np.arange(0, lags + 1)

    for i in range(lags + 1):
        ng = 1
        sx = 2
        sy = 3
        sxx = 4
        syy = 5
        sxy = 6

        for k in range(len(ts1) - (i + offset)):
            x = ts1[k]
            y = ts2[k + (i + offset)]
            if not np.isnan(x) and not np.isnan(y):
                sx += x
                sy += y
                sxx += x * x
                syy += y * y
                sxy += x * y
                ng += 1

        covar1 = (sxy / ng) - ((sx / ng) * (sy / ng))
        denom1 = np.sqrt((sxx / ng) - (sx / ng) ** 2)
        denom2 = np.sqrt((syy / ng) - (sy / ng) ** 2)
        P[i] = covar1 / (denom1 * denom2)

    return P, nlags


def reshape_vertical(matrix):
    """
    Converts matrix dimensions from mxn to nxm
    """
    matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return matrix


def get_audio_lengths(directory):
    """
    Retrieve the lengths (in seconds) of all sound files in a directory.

    Args:
        directory (str): Path to the directory containing sound files.

    Returns:
        dict: A dictionary where keys are file names and values are their lengths in seconds.
    """
    audio_lengths = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip if not a file
        if not os.path.isfile(file_path):
            continue

        # Try to load the audio file
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=audio_data, sr=sample_rate) / 60  # Round up for full minutes
            audio_lengths[filename] = duration
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    lengths_arr = list(audio_lengths.values())

    return lengths_arr


def reorganize_matrix(matrix):
    """
    Reorganize a matrix to create a new structure with specified alignment.

    Args:
        matrix (list): Original 2D matrix.

    Returns:
        list: Reorganized 2D matrix.
    """
    # Convert the input to a NumPy array for easier manipulation
    matrix = np.array(matrix)

    # Step 1: Split the original matrix into two columns
    column1 = matrix[:, 0]  # First column
    column2 = matrix[:, 1]  # Second column

    # Step 2: Prepare the reorganized matrix structure
    reorganized = []

    # Alternate elements between column1 and column2
    for i in range(len(column1)):
        if i < len(column2):
            reorganized.append([column1[i], column2[i]])
        else:
            reorganized.append([column1[i], 0])  # Fill missing values with 0

    # Remaining elements in column2
    for i in range(len(column1), len(column2)):
        reorganized.append([0, column2[i]])

    return reorganized


def write_to_json(output_dir, SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim):
    """
    Write calculated metrics to JSON file

    Args:
    output_dir: Path to save the output JSON file.
    SPLrms: Matrix of SPLrms values.
    SPLpk: Matrix of SPLpk values.
    impulsivity: Matrix of impulsivity values.
    peakcount: Matrix of peak count values.
    autocorr: Matrix of autocorrelation values.
    dissim: Matrix of dissimilarity values.
    """
    # Create a dictionary to hold all the results
    result_data = {
        'SPLrms': SPLrms,
        'SPLpk': SPLpk,
        'impulsivity': impulsivity,
        'peakcount': peakcount,
        'autocorr': autocorr.tolist(),
        'dissim': dissim
    }

    # Write the dictionary to a JSON file
    with open(output_dir, 'w') as f:
        json.dump(result_data, f, indent=4)


def create_multiple_audio_files(input_directory, output_directory, base_name, target_duration_seconds,
                                target_sample_rate=8000):
    combined_audio = []
    total_duration = 0.0
    file_counter = 1

    for file_name in sorted(os.listdir(input_directory)):
        if file_name.endswith('.wav'):
            input_file = os.path.join(input_directory, file_name)

            # Read audio file with original sample rate
            audio_data, original_sample_rate = librosa.load(input_file, sr=None)

            # Resample to target sample rate if needed
            if original_sample_rate != target_sample_rate:
                audio_data = librosa.resample(audio_data,
                                              orig_sr=original_sample_rate,
                                              target_sr=target_sample_rate)

            duration_seconds = len(audio_data) / target_sample_rate

            combined_audio.append(audio_data)
            total_duration += duration_seconds

            while total_duration >= target_duration_seconds:
                output_audio = np.concatenate(combined_audio)
                max_samples = int(target_duration_seconds * target_sample_rate)
                output_audio_chunk = output_audio[:max_samples]

                output_file = os.path.join(output_directory, f"{base_name}_part{file_counter}.wav")
                sf.write(output_file, output_audio_chunk, target_sample_rate)

                combined_audio = [output_audio[max_samples:]]
                total_duration -= target_duration_seconds
                file_counter += 1

    if combined_audio and total_duration > 0:
        remaining_audio = np.concatenate(combined_audio)
        output_file = os.path.join(output_directory, f"{base_name}_part{file_counter}.wav")
        sf.write(output_file, remaining_audio, target_sample_rate)

def process_directory_of_wav_files(input_directory, output_directory, target_duration_seconds):
    os.makedirs(output_directory, exist_ok=True)
    base_name = os.path.basename(input_directory)
    create_multiple_audio_files(input_directory, output_directory, base_name, target_duration_seconds)


if __name__ == '__main__':
    # Define input/output directories
    input_dir = "C:/Users/rlessard/Desktop/runThisInput/AMAR"  # Path containing .wav files
    output_dir = "C:/Users/rlessard/Desktop/runThisOutput/runThisPython.json"  # Path to store .mat file
    # even_len_dir = "C:/Users/rlessard/Desktop/runThisInput/even_lengths"  # Path with even-length recordings

    # Get sound file lengths to split into even lengths
    lengths = get_audio_lengths(input_dir)
    avg_length = sum(lengths) / len(lengths) * 60  # Convert to seconds

    input_directory = input_dir
    target_duration_seconds = avg_length
    # process_directory_of_wav_files(input_directory, even_len_dir, target_duration_seconds)

    file_dir = os.listdir(input_dir)
    # file_dir = os.listdir(even_len_dir)
    num_bits = 16
    RS = -178.3  # BE SURE TO CHANGE FOR EACH HYDROPHONE
    # Sensitivity is based on hydrophone, not recorder
    peak_volts = 2
    arti = 1  # Make 1 if calibration tone present

    # Analysis options
    timewin = 60  # Length of time window in seconds for analysis bins
    fft_win = 1  # Length of fft window in minutes
    avtime = 0.1
    flow = 50  # Low frequency
    fhigh = 300  # High frequency

    SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim = f_WAV_frankenfunction_reilly(num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh, input_dir)
    # SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim = f_WAV_frankenfunction_reilly(num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh, even_len_dir)

    # Change dimensions for mxn to nxm
    SPLrms = reshape_vertical(SPLrms)
    SPLpk = reshape_vertical(SPLpk)
    impulsivity = reshape_vertical(impulsivity)
    peakcount = reshape_vertical(peakcount)
    dissim = reshape_vertical(dissim)

    # Write the results to a JSON file
    write_to_json(output_dir, SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim)
