import math
import os
import warnings

import numpy as np
import scipy

from numpy import size, ComplexWarning
from scipy.io import wavfile
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

# Suppress warnings
warnings.filterwarnings("ignore", category=ComplexWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def fft(x, n=None, axis=-1):
    if n is None:
        n = len(x)
    elif n < len(x):
        x = x[:n]
    else:
        x = np.pad(x, (0, n - len(x)), mode='constant')

    return np.fft.fft(x, n=n, axis=axis)


def dylan_bpfilt(ts, samint, flow, fhigh):
    npts = len(ts)
    reclen = npts * samint

    spec = np.fft.fft(ts, npts)
    aspec = abs(spec)
    pspec = np.angle(spec)                             # pspec = arctan2(spec.imag, spec.real)
    freq = np.fft.fftshift(np.arange(-npts / 2, npts / 2) / reclen)
    freqFirst = freq[0]
    freq = freq[1:]
    freq = np.insert(freq, len(freq), freqFirst)

    if fhigh == 0:
        fhigh = 1 / (2 * samint)

    ifr = np.where(np.logical_and(np.abs(freq) >= flow, np.abs(freq) <= fhigh))[0]

    ifr = ifr[2:]

    ifrFirst = ifr[:int(len(ifr) / 2)]

    # Insert missing tuples in middle of ifr
    ifrMidMinus2 = ifrFirst[int(len(ifr) / 2) - 2] + 1
    ifrMidMinus1 = ifrMidMinus2 + 1

    ifr = np.insert(ifr, len(ifrFirst) - 1, ifrMidMinus2)
    ifr = np.insert(ifr, len(ifrFirst), ifrMidMinus1)

    ifrMidPlus1 = ifr[int(len(ifr) / 2)] + 2
    ifrMidPlus2 = ifrMidPlus1 + 1

    ifrHalfOne = ifr[:int(len(ifr) / 2)]
    ifrHalfTwo = ifr[int(len(ifr) / 2) + 2:]

    ifrSize = len(ifr)
    ifrFinal = ifr[ifrSize - 1]

    midValue = ifr[int(ifrSize / 2)]
    ifr = np.delete(ifr, int(ifrSize / 2), None)
    ifr = np.delete(ifr, int(ifrSize / 2), None)

    ifrLastPlusOne = ifrFinal + 1
    ifrLastPlusTwo = ifrFinal + 2

    ifr = np.insert(ifr, len(ifr), ifrLastPlusOne)
    ifr = np.insert(ifr, len(ifr), ifrLastPlusTwo)

    filtspec2 = np.zeros_like(spec, dtype=complex)
    rspec = np.zeros_like(spec)
    ispec = np.zeros_like(spec)

    rspec[ifr] = aspec[ifr] * np.cos(pspec[ifr])
    ispec[ifr] = aspec[ifr] * np.sin(pspec[ifr])

    filtspec2.real = rspec
    filtspec2.imag = ispec

    filtspec1 = np.abs(filtspec2[:npts // 2 + 1])
    filtspec1[1:-1] *= 2

    tsfilt = np.real(np.fft.ifft(filtspec2, npts))
    # print("tsfilt " + str(tsfilt))

    return tsfilt, filtspec1


def correl_5(x, y, lags, mode):
    """ Computes the correlation between two signals with a maximum number of lags."""
    corr = np.correlate(x, y, mode=mode)
    lags = min(lags, len(corr) - 1)
    corr = corr[len(corr) // 2 - lags:len(corr) // 2 + lags + 1]
    return corr / max(corr)


def calculate_kurtosis_ft(data, axis=0, fisher=True, bias=True):
    n = data.shape[axis]
    mean = np.mean(data, axis=axis)
    var = np.var(data, axis=axis, ddof=0 if bias else 1)
    fourth_moment = np.mean((data - mean) ** 4, axis=axis)
    kurtosis_value = fourth_moment / (var ** 2)

    if fisher:
        kurtosis_value -= 3

    return kurtosis_value


def frankenfunc_testscript(p_filt, fs, timewin=58, fft_win=1, avtime=0.1, flow=50, fhigh=300):
    """
    Analyzes acoustic data and computes various soundscape metrics.

    Args:
        p_filt (np.ndarray): Filtered pressure signal.
        fs (int): Sampling frequency.
        timewin (float, optional): Length of the time window in seconds for analysis bins. Default is 58.
        fft_win (float, optional): Length of the FFT window in minutes. Default is 1.
        avtime (float, optional): Averaging time duration in seconds for autocorrelation measurements. Default is 0.1.
        flow (float, optional): Lower frequency cutoff. Default is 50.
        fhigh (float, optional): Upper frequency cutoff. Default is 300.

    Returns:
        SPLrms (np.ndarray): Matrix of root mean square sound pressure level, each column is a sound file, each row is 1 minute of data.
        SPLpk (np.ndarray): Matrix of peak SPL (the highest SL in a sample), each column is a sound file, each row is 1 minute of data.
        impulsivity (np.ndarray): Kurtosis of the signal, a measure of impulsive sounds.
        peakcount (np.ndarray): Count of the number of times the autocorrelation threshold is exceeded.
        autocorr (np.ndarray): Matrix of autocorrelation calculations, each column is 1 minute of data, each row is the autocorrelation value calculated at 0.1 second time lag.
        dissim (np.ndarray): Matrix of the amount of uniformity between each 1 minute of data compared to the following minute of data, should have n-1 rows where n is the number of minutes of data.
    """

    SPLrms = []
    SPLpk = []
    impulsivity = []
    peakcount = []
    autocorr = []
    dissim = []

    # Compute time windows
    pts_per_timewin = int(timewin * fs)
    # print("pts_per_timewin: " + str(pts_per_timewin))
    num_timewin = len(p_filt) // pts_per_timewin
    # print("num_timewin: " + str(num_timewin))
    trimseries = p_filt[:pts_per_timewin * num_timewin]
    # print("trimseries size: " + str(len(trimseries)))
    # print("trimseries: " + str(trimseries))
    timechunk_matrix = trimseries.reshape(pts_per_timewin, num_timewin)
    # timechunk_matrix = np.transpose(timechunk_matrix)
    # print("timechunk_matrix size: " + str(timechunk_matrix.shape))
    sizeA, sizeB = timechunk_matrix.shape
    # for i in range(1):
    # print("timechunk_matrix " + str(i) + ": " + str(timechunk_matrix[i]))

    # Compute RMS and peak SPL
    rms_matrix = scipy.linalg.norm(timechunk_matrix, axis=0) / np.sqrt(timechunk_matrix.shape[0])
    # print("rms_matrix: " + str(rms_matrix))
    SPLrmshold = 20 * np.log10(rms_matrix)
    # print("SPLrmshold: " + str(SPLrmshold))
    SPLpkhold = np.max(20 * np.log10(np.abs(timechunk_matrix)), axis=0)
    # print("SPLpkhold: " + str(SPLpkhold))

    SPLrms = SPLrmshold
    SPLpk = SPLpkhold

    # Compute impulsivity
    kmat = kurtosis(timechunk_matrix)
    kurt_value = calculate_kurtosis_ft(timechunk_matrix, axis=0, fisher=True, bias=True)
    kurt_value = kurt_value * 28
    # print("kurt_value: " + str(kurt_value))
    impulsivity = kmat
    # print("impulsivity: " + str(impulsivity))

    # Compute periodicity
    pkcount, acorr_matrix = solo_per_gm2(p_filt, fs, timewin, avtime)
    peakcount = pkcount
    # print("peakcount: " + str(skew(peakcount, axis=0, bias=True)))
    autocorr = acorr_matrix
    # print("autocorr: " + str(autocorr))

    # Compute D-index
    dfin = solo_dissim_gm1(timechunk_matrix, pts_per_timewin, num_timewin, fft_win, fs)
    dissim = dfin
    print("dissimilarity: " + str(dissim))

    return SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim


def solo_dissim_gm1(timechunk_matrix, pts_per_timewin, num_timewin, fft_win, fs):
    pts_per_fft = int(fft_win * 60 * fs)
    numfftwin = pts_per_timewin // pts_per_fft

    dfin = []
    for kk in range(num_timewin - 1):
        analytic1 = hilbert(timechunk_matrix[:, kk])
        analytic2 = hilbert(timechunk_matrix[:, kk + 1])

        at1 = np.abs(analytic1) / np.sum(np.abs(analytic1))
        at2 = np.abs(analytic2) / np.sum(np.abs(analytic2))

        dt = np.sum(np.abs(at1 - at2)) / 2

        s3a = timechunk_matrix[:, kk][:pts_per_fft * numfftwin]
        s3a = s3a.reshape(pts_per_fft, numfftwin)

        ga = np.abs(np.fft.fft(s3a, axis=0)) / s3a.shape[0]
        sfa = np.mean(ga, axis=1)
        sfa = np.abs(sfa) / np.sum(np.abs(sfa))

        s3b = timechunk_matrix[:, kk + 1][:pts_per_fft * numfftwin]
        s3b = s3b.reshape(pts_per_fft, numfftwin)
        gb = np.abs(np.fft.fft(s3b, axis=0)) / s3b.shape[0]
        sfb = np.mean(gb, axis=1)
        sfb = np.abs(sfb) / np.sum(np.abs(sfb))

        df = np.sum(np.abs(sfb - sfa)) / 2

        di = dt * df
        dfin.append(di)

    return dfin


def solo_per_gm2(p_filt, fs, timewin, avtime):
    print("fs: " + str(fs))

    p_av = []
    p_avtot = []
    avwin = int(fs * avtime)
    print("avwin: " + str(avwin))

    sampwin = int(fs * timewin)
    print("sampwin: " + str(sampwin))

    print("p_filt length: " + str(len(p_filt)))
    ntwin = len(p_filt) // sampwin  # Number of minutes
    print("ntwin: " + str(ntwin))

    p_filt = p_filt[:sampwin * ntwin]
    # print("\n" + "p_filt: " + str(p_filt[:10]) + "\n")
    # quar3 = int(len(p_filt) * (19 / 20))
    # point94339 = 72203298
    # point943395 = 72203681
    point94340 = 72204063
    # print("\np_filt: " + str(p_filt[point94340:point94340 + 10]) + "\n")
    p_filt = p_filt.reshape(sampwin, ntwin)
    p_filt = p_filt ** 2
    # print("p_filt size: " + str(p_filt.shape))
    # print("p_filt: " + str(p_filt))
    # print("p_filt: " + str(p_filt[100000:100010]))

    numavwin = p_filt.shape[0] // avwin
    print("numavwin: " + str(numavwin))
    p_av = []

    for jj in range(ntwin):
        avwinmatrix = p_filt[:, jj].reshape(avwin, numavwin)
        # print("avwinmatrix: " + str(avwinmatrix[:1]))
        p_avi = np.mean(avwinmatrix, axis=1)
        p_av.append(p_avi)

    print("p_av size: " + str(size(p_av)))
    print("p_av: " + str(p_av) + "\n")

    p_avtot = np.hstack(p_av)
    # print("p_avtot size: " + str(p_avtot.size))
    # print("p_avtot: " + str(p_avtot))

    if p_avtot.ndim < 2 or p_avtot.shape[1] == 0:
        print("Warning: p_avtot has fewer than 2 dimensions or no columns. Returning empty arrays.")
        return [], []

    pkcount = []
    acorr = []
    for zz in range(p_avtot.shape[1]):
        acorr_col = correl_5(p_avtot[:, zz], p_avtot[:, zz], int(p_avtot.shape[0] * 0.7), 0)
        acorr.append(acorr_col)
        pks, _ = scipy.signal.find_peaks(acorr_col, prominence=0.5)
        pkcount.append(len(pks))

    return pkcount, np.array(acorr).T


def reshape_list(trimseries, rows, cols):
    timechunk_matrix = [[0] * cols for _ in range(rows)]
    for col in range(cols):
        for row in range(rows):
            timechunk_matrix[row][col] = trimseries[col * rows + row]
    return timechunk_matrix


def calculate_rms_matrix(timechunk_matrix):
    num_rows = len(timechunk_matrix)
    num_cols = len(timechunk_matrix[0])
    rms_matrix = []

    for col in range(num_cols):
        sum_of_squares = sum(timechunk_matrix[row][col] ** 2 for row in range(num_rows))
        mean_of_squares = sum_of_squares / num_rows
        rms = math.sqrt(mean_of_squares)
        rms_matrix.append(rms)

    return rms_matrix


def calculate_kurtosis(timechunk_matrix):
    # Calculate the mean and standard deviation of each column
    mean_values = np.mean(timechunk_matrix, axis=0)
    std_dev = np.std(timechunk_matrix, axis=0)

    # Calculate the centered data
    centered_data = timechunk_matrix - mean_values

    # Calculate the fourth moment (kurtosis)
    fourth_moment = np.mean(centered_data ** 4, axis=0)

    # Calculate kurtosis
    kurtosis_values = fourth_moment / (std_dev ** 4)

    return kurtosis_values


def calculate_rms(array):
    """
    Calculate the root mean square of a given array.

    Parameters:
    array (numpy.ndarray): Input array.

    Returns:
    float: Root mean square value.
    """
    # Square the elements, calculate the mean, and then take the square root
    rms = np.sqrt(np.mean(np.square(array)))
    return rms


def wav_frankenfunction_reilly(num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh):
    """
   Mimics the MATLAB function f_WAV_frankenfunction_reilly.

   Args:
       num_bits (int): Bit rate of the hydrophone.
       peak_volts (float): Peak voltage of the recorder.
       file_dir (str): Directory containing the audio files.
       RS (float): Hydrophone sensitivity in dB.
       timewin (float): Length of the time window in seconds for analysis bins.
       avtime (float): Averaging time duration in seconds for autocorrelation measurements.
       fft_win (float): Length of the FFT window in minutes.
       arti (int): Enter 1 if there are artifacts like calibration tones at the beginning of recordings.
       flow (float): Lower frequency cutoff.
       fhigh (float): Upper frequency cutoff.

   Returns:
       SPLrms (np.ndarray): Matrix of root mean square sound pressure level, each column is a sound file, each row is 1 minute of data.
       SPLpk (np.ndarray): Matrix of peak SPL (the highest SL in a sample), each column is a sound file, each row is 1 minute of data.
       impulsivity (np.ndarray): Kurtosis of the signal, a measure of impulsive sounds.
       peakcount (np.ndarray): Count of the number of times the autocorrelation threshold is exceeded.
       autocorr (np.ndarray): Matrix of autocorrelation calculations, each column is 1 minute of data, each row is the autocorrelation value calculated at 0.1 second time lag.
       dissim (np.ndarray): Matrix of the amount of uniformity between each 1 minute of data compared to the following minute of data, should have n-1 rows where n is the number of minutes of data.
   """
    file_list = os.listdir(file_dir)
    num_files = len(file_list)

    # Convert to voltage and pressure
    rs = 10 ** (RS / 20)
    max_count = 2 ** num_bits
    conv_factor = peak_volts / max_count

    SPLrms_list = []
    SPLpk_list = []
    impulsivity_list = []
    peakcount_list = []
    autocorr_list = []
    dissim_list = []
    impulsivity = []

    for ii, filename in enumerate(file_list, start=1):
        print(f"\nProcessing file {ii}/{num_files}: {filename}")

        file_path = os.path.join(file_dir, filename)
        fs, x = wavfile.read(file_path)

        # Downsample or upsample the signal if necessary
        if fs == 576000:
            x = np.mean(x.reshape(-1, 4), axis=1)
            fs = 144000
        elif fs == 288000:
            x = np.mean(x.reshape(-1, 2), axis=1)
            fs = 144000
        elif fs == 16000:
            x = np.repeat(x, 9)
            fs = 144000
        elif fs == 512000:
            x = np.mean(x.reshape(-1, 4), axis=1)
            fs = 144000

        if num_bits == 24:
            x = np.right_shift(x, 8)  # Bitcount - accounts for audioread casting of 24 bit to 32 bit (zeroes behind)

        v = x * conv_factor
        v = v.astype(np.double)  # Convert to double precision for accurate calculations
        p = v / rs
        # print("v: " + str(v))
        # print("p: " + str(p))
        poutLength = int((p.size / 9) + 1)
        pout = np.zeros(poutLength)
        poutNew = np.zeros(len(p))
        # print("poutNew size: " + str(len(poutNew)))

        # Remove calibration tone if present
        if arti == 1:
            p = p[6 * fs:]
            # print("p size: " + str(p.size))
            # print("p[0:50: " + str(p[0:50]))
            for i in range(p.size):
                if (i % 9 == 0):
                    # pout[i] = p[i]
                    poutNew[i] = p[i]

        pout = [0] + pout
        # print("p size: ", len(p))
        # print("pout size: ", len(pout))
        # print("pout 1:3 and n-2:n: " + str(pout))
        poutNew = np.insert(poutNew, 0, 0.0)
        poutNew = np.insert(poutNew, len(poutNew) - 1, 0.0)
        # for i in range(100):
        # print(f"pout: {pout[i]}")
        # print(f"poutNew: {poutNew[i]}")

        # print("p.size: " + str(p.size))
        # print("First 50 indexes of poutNew: " + str(poutNew[:50]))
        # print("Last 50 indexes of poutNew: " + str(poutNew[p.size-50:p.size + 1]))

        poutNew = poutNew[:p.size + 1]
        # print("First 50 indexes of poutNew: " + str(poutNew[:50]))
        # print("Last 50 indexes of poutNew: " + str(poutNew[p.size - 50:]))

        # print("poutNew size: " + str(len(poutNew)))
        # print("p size: " + str(len(p)))

        # Filter the signal with adjusted cutoff frequencies
        p_filt, _ = dylan_bpfilt(poutNew, 1 / fs, flow, fhigh)

        # print("p_filt " + str(p_filt))
        print("p_filt size: " + str(p_filt.shape))
        # for i in range(100):
            # print("p_filt(" + str(i) + "): " + str(p_filt[i]))

        # tempPFilt = p_filt[-5:]
        # print("tempPFilt: " + str(tempPFilt))

        pts_per_timewin = timewin * fs
        # print("pts_per_timewin: " + str(pts_per_timewin))

        num_timewin = int(len(p_filt) / pts_per_timewin)
        # print("num_timewin: " + str(num_timewin))

        trimseries = p_filt[:pts_per_timewin * num_timewin]
        # print("trimseries size: " + str(size(trimseries)))
        # print("trimseries: " + str(trimseries))

        # start = len(trimseries) // 2 - 25

        # print("trimseries pptw: " + str(trimseries[int(pts_per_timewin) - 5:int(pts_per_timewin) + 5]))
        # print("trimseries pptw: " + str(trimseries[-10:]))
        # print("trimseries ntw: " + str(trimseries[int(num_timewin) - 5:int(num_timewin)]))

        timechunk_matrix = np.reshape(trimseries, (pts_per_timewin, num_timewin))
        # print("tcm size: " + str(size(timechunk_matrix)))
        # print("tcm: " + str(timechunk_matrix))

        # tcm1xn = np.reshape(timechunk_matrix, -1)

        # print("tcm 1xn size: " + str(size(tcm1xn)))
        # print("tcm 1xn: " + str(tcm1xn))

        # print("tcm size: " + str(size(timechunk_matrix)))
        # print("tcm: " + str(timechunk_matrix[:5]))

        # print("tcm type: " + str(type(timechunk_matrix)))
        rms_matrix = calculate_rms_matrix(timechunk_matrix) # trimseries and timechunk_matrix inconsistently calculated
        # print("rms_matrix: " + str(rms_matrix))
        SPLrmshold = 20 * np.log10(rms_matrix)
        # print("SPLrms: " + str(SPLrmshold))
        SPLpkhold = 20 * np.log10(np.max(np.abs(timechunk_matrix), axis=0))
        # print("SPLpk: " + str(SPLpkhold))

        # SPLrms = np.column_stack((SPLrms, SPLrmshold))
        # SPLpk = np.column_stack((SPLpk, SPLpkhold))

        kmat = calculate_kurtosis(timechunk_matrix)
        # print("kmat: " + str(kmat))
        impulsivity = np.append(impulsivity, kmat)
        # print("impulsivity: " + str(kmat))

        # Periodicity
        pkcount, acorr = solo_per_gm2(p_filt, fs, timewin, avtime)
        print("pkcount: " + str(pkcount))
        print("acorr: " + str(acorr))

        # Call the frankenfunc_testscript function
        SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim = (frankenfunc_testscript(
            p_filt, fs, timewin, fft_win, avtime, flow, fhigh))

        SPLrms_list.append(SPLrms)
        SPLpk_list.append(SPLpk)
        impulsivity_list.append(impulsivity)
        peakcount_list.append(peakcount)
        autocorr_list.append(autocorr)
        dissim_list.append(dissim)

    SPLrms_out = np.array(SPLrms_list).T
    SPLpk_out = np.array(SPLpk_list).T
    impulsivity_out = np.array(impulsivity_list).T
    peakcount_out = np.array(peakcount_list)
    autocorr_out = np.hstack(autocorr_list)
    dissim_out = np.hstack(dissim_list)

    # Compute periodicity
    pkcount, acorr_matrix = solo_per_gm2(p_filt, fs, timewin, avtime)
    peakcount = pkcount
    print("peakcount: " + str(skew(peakcount, axis=0, bias=True)))
    autocorr = acorr_matrix
    print("autocorr: " + str(autocorr))

    return SPLrms_out, SPLpk_out, impulsivity_out, peakcount_out, autocorr_out, dissim_out


# Example usage
file_dir = 'C:/Users/rlessard/Desktop/Python/AudioInput'
num_bits = 16
rs = -178.3
peak_volts = 2
arti = 1
timewin = 58
fft_win = 1
avtime = 0.1
flow = 50
fhigh = 300

SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim = (
    wav_frankenfunction_reilly(num_bits, peak_volts, file_dir, rs, timewin, avtime, fft_win, arti, flow, fhigh))

# Print the output shapes
# print("\nSPLrms: " + str(SPLrms))
# print("\nSPLpk: " + str(SPLpk))
# print("\nimpulsivity: " + str(impulsivity))
# print("\npeakcount: " + str(peakcount))
# print("\nautocorr: " + str(autocorr))
# print("\ndissim: " + str(dissim))

# You can also save the output to files if needed
np.savez('output.npz', SPLrms=SPLrms, SPLpk=SPLpk, impulsivity=impulsivity,
         peakcount=peakcount, autocorr=autocorr, dissim=dissim)
