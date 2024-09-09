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
input_dir = "C:/Users/rlessard/Desktop/5593 organized/240406165958_240413225912/output_10min"  # Path containing .wav files
output_dir = "C:/Users/rlessard/Desktop/SoundscapeCodeDesktop/DataOutput/10_minutes/output_python.json"  # Path to store .mat file


def frequency_resample(fs, x):
    """
    Adjust the sample rate of the audio data (downsample if necessary).

    Args:
    fs: Original sample rate of the audio data.
    x: Audio signal data.

    Returns:
    Updated sample rate and audio data after downsampling or upsampling.
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
    tcm_rearrange = np.array(tcm_rearrange)
    kmat = kurtosis_reilly(tcm_rearrange)
    impulsivity.extend(kmat)
    return impulsivity


def calculate_spl(rms_matrix, tcm_rearrange, SPLrms, SPLpk):
    SPLrmshold = 20 * np.log10(rms_matrix)  # Log transforms the rms pressure
    SPLpkhold = np.max(20 * np.log10(np.abs(tcm_rearrange)), axis=0)

    SPLrms.extend(SPLrmshold)  # Logarithmic conversion of RMS values to SPL
    SPLpk.extend(SPLpkhold)  # Find peak SPL for each time window
    return SPLrms, SPLpk


def calculate_autocorrelation(autocorr, acorr):
    # Append autocorrelation results for each file
    if autocorr is None:
        autocorr = acorr
    else:
        autocorr = np.column_stack((autocorr, acorr))
    return autocorr


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

        # Downsample if sample rate is too high
        fs, x = frequency_resample(fs, x)

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

        autocorr = calculate_autocorrelation(autocorr, acorr)

        # CCalculate dissimilarity between adjacent time windows (D-index)
        Dfin = f_solo_dissim_GM1(pts_per_timewin, num_timewin, fft_win, fs, tcm_rearrange)
        dissim.extend(Dfin)

    # Reshape matrices (One row per recording)
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
    Create matrix with data being inserted by column, rather than by row.
    """
    rows = row
    cols = col

    # Initialize the 2D array with zeros
    result = [[0 for _ in range(cols)] for _ in range(rows)]

    # Fill the 2D array by columns
    for col in range(cols):
        for row in range(rows):
            index = col * rows + row
            result[row][col] = input_array[index]

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
    Rewritten from MATLAB
    K = KURTOSIS(X) returns the sample kurtosis of the values in X.
    For a vector input, K is the fourth central moment of X, divided by fourth power of its standard deviation.
    For a matrix input, K is a row vector containing the sample kurtosis of each column of X.
    For N-D arrays, KURTOSIS operates along the first non-singleton dimension

    KURTOSIS(X,0) adjusts the kurtosis for bias.
    KURTOSIS(X,1) is the same as KURTOSIS(X), and does not adjust the bias.
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
    Create a bypass filter.
    Uses maximum and minimum frequencies for metric calculations.
    """
    npts, spec, aspec, pspec, freq = bpfilt_initial_values(ts, samint)

    if fhigh == 0:
        fhigh = 1 / (2 * samint)

    ifr = np.where((np.abs(freq) >= flow) & (np.abs(freq) <= fhigh))[0]  # Calculate metrics between flow and fhigh
    filtspec2 = np.zeros_like(spec, dtype=complex)
    rspec = np.zeros_like(spec)
    ispec = np.zeros_like(spec)

    rspec[ifr] = aspec[ifr] * np.cos(pspec[ifr])
    ispec[ifr] = aspec[ifr] * np.sin(pspec[ifr])
    filtspec2 = rspec + 1j * ispec

    tsfilt = np.real(np.fft.ifft(filtspec2, npts))

    return tsfilt


def f_solo_per_GM2(p_filt, fs, timewin, avtime):
    '''
    Calculates peakcount and autocorrelation.
    '''
    p_av = []
    p_avtot = []
    avwin = int(fs * avtime)
    sampwin = int(fs * timewin)
    ntwin = len(p_filt) // sampwin  # Number of minutes
    p_filt = p_filt[:sampwin * ntwin]

    p_filt = distribute_array(p_filt, ntwin)

    p_filt = p_filt ** 2

    numavwin = p_filt.shape[0] // avwin

    p_av = []

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
    '''
    Calculates dissimilarity
    '''
    tcm_rearrange = np.array(tcm_rearrange)  # Uses numpy to perform mathematical operations

    pts_per_fft = int(fft_win * fs)  # Calc size fft window
    numfftwin = int(np.floor(pts_per_timewin / pts_per_fft))  # Number of fft windows

    Dfin = []
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
