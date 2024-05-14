import os
import numpy as np
import scipy
from numpy import size
from scipy.io import wavfile
from scipy.signal import hilbert, butter, sosfilt
from findpeaks import findpeaks
from scipy.stats import kurtosis
from scipy.fft import fftfreq, fftshift
from scipy import arctan2


def fft(x, n = None, axis=-1):
    if n is None:
        n = len(x)
    elif n < len(x):
        x = x[:n]
    else:
        x = np.pad(x, (0, n - len(x)), mode='constant')

    return np.fft.fft(x, n=n, axis=axis)


def dylan_bpfilt(ts, samint, flow, fhigh):
    npts = len(ts)
    # print("npts: " + str(npts))
    reclen = npts * samint
    # print("reclen: " + str(reclen))

    spec = fft(ts, npts)
    aspec = abs(spec)
    pspec = arctan2(spec.imag, spec.real)
    freq = np.fft.fftshift(np.arange(-npts / 2, npts / 2) / reclen)
    freqFirst = freq[0]
    freq = freq[1:]
    freq = np.insert(freq, len(freq), freqFirst)

    # print("freq size: " + str(len(freq)))
    # print("freq: " + str(freq))


    # print("Mid freq -3: " + str(freq[int(len(freq) / 2) - 3]))
    # print("Mid freq -2: " + str(freq[int(len(freq) / 2) - 2]))
    # print("Mid freq -1: " + str(freq[int(len(freq) / 2) - 1]))
    # print("Mid freq 0: " + str(freq[int(len(freq) / 2)]))
    # print("Mid freq +1: " + str(freq[int(len(freq) / 2) + 1]))
    # print("Mid freq +2: " + str(freq[int(len(freq) / 2) + 2]))
    # print("Mid freq +3: " + str(freq[int(len(freq) / 2) + 3]))

    if fhigh == 0:
        fhigh = 1 / (2 * samint)

    ifr = np.where(np.logical_and(np.abs(freq) >= flow, np.abs(freq) <= fhigh))[0]
    # print("ifr size: ", len(ifr))

    # for i in range(10):
    # print("ifr " + str(i) + ": " + str(ifr[i]))

    # print("ifr: " + str(ifr[:10]))

    ifr = ifr[2:]

    # print("ifr size: ", len(ifr))

    ifrFirst = ifr[:int(len(ifr) / 2)]
    # print("ifr first: " + str(ifrFirst))

    # Insert missing tuples in middle of ifr
    ifrMidMinus2 = ifrFirst[int(len(ifr) / 2) - 2] + 1
    ifrMidMinus1 = ifrMidMinus2 + 1

    ifr = np.insert(ifr, len(ifrFirst) - 1, ifrMidMinus2)
    ifr = np.insert(ifr, len(ifrFirst), ifrMidMinus1)

    ifrMidPlus1 = ifr[int(len(ifr) / 2)] + 2
    ifrMidPlus2 = ifrMidPlus1 + 1
    # print("ifr thirdIndex: " + str(ifrMidPlus1))
    # print("ifr fourthIndex: " + str(ifrMidPlus2))

    ifrHalfOne = ifr[:int(len(ifr) / 2)]
    ifrHalfTwo = ifr[int(len(ifr) / 2) + 2:]

    ifrSize = len(ifr)
    print("ifrSize: " + str(ifrSize))
    ifrFinal = ifr[ifrSize - 1]
    print("ifrFinal: " + str(ifrFinal))

    print("Middle values: " + str(ifr[int(ifrSize / 2) - 2: int(ifrSize / 2) + 2]))

    midValue = ifr[int(ifrSize / 2)]
    print("MidValue: " + str(midValue))
    ifr = np.delete(ifr, int(ifrSize / 2), None)
    ifr = np.delete(ifr, int(ifrSize / 2), None)
    print("Middle values: " + str(ifr[int(ifrSize / 2) - 2: int(ifrSize / 2) + 2]))

    ifrLastPlusOne = ifrFinal + 1
    print("ifrLastPlusOne: " + str(ifrLastPlusOne))
    ifrLastPlusTwo = ifrFinal + 2
    print("ifrLastPlusTwo: " + str(ifrLastPlusTwo))

    ifr = np.insert(ifr, len(ifr), ifrLastPlusOne)
    ifr = np.insert(ifr, len(ifr), ifrLastPlusTwo)

    # print("ifr First 100: ", ifr[:100])
    # print("ifr Final 100: ", ifr[-100:])
    # print("ifrHalfTwo", ifrHalfTwo)

    # print("ifr first half: " + str(ifr[:int(len(ifr) / 2) + 2]))

    # print("ifr middle: " + str(ifr[int(len(ifr) / 2) - 4:int(len(ifr) / 2) + 4]))

    # ifrSecond = ifr[int(len(ifr) / 2):]
    # print("ifr second: " + str(ifrSecond))

    # for j in range(int(len(ifr) / 2) - 5, int(len(ifr) / 2) + 5):
    # print("ifr " + str(j) + ": " + str(ifr[j]))

    # print("ifr: " + str(ifr[len(ifr) - 10:]))

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

    # print("dylan_bpfilt done")
    return tsfilt, filtspec1


def correl_5(x, y, lags, mode):
    """
    Computes
    the
    correlation
    between
    two
    signals
    with a maximum number of lags."""
    corr = np.correlate(x, y, mode=mode)
    lags = min(lags, len(corr) - 1)
    corr = corr[len(corr) // 2 - lags:len(corr) // 2 + lags + 1]
    return corr / max(corr)


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
    num_timewin = len(p_filt) // pts_per_timewin
    trimseries = p_filt[:pts_per_timewin * num_timewin]
    timechunk_matrix = trimseries.reshape(pts_per_timewin, num_timewin)

    # Compute RMS and peak SPL
    rms_matrix = np.sqrt(np.mean(timechunk_matrix ** 2, axis=0))
    SPLrmshold = 20 * np.log10(rms_matrix)
    SPLpkhold = np.max(20 * np.log10(np.abs(timechunk_matrix)), axis=0)

    SPLrms = SPLrmshold
    SPLpk = SPLpkhold

    # Compute impulsivity
    kmat = np.apply_along_axis(lambda x: kurtosis(x, bias=False), 0, timechunk_matrix)
    impulsivity = kmat

    # Compute periodicity
    pkcount, acorr_matrix = solo_per_gm2(p_filt, fs, timewin, avtime)
    peakcount = pkcount
    autocorr = acorr_matrix

    # Compute D-index
    dfin = solo_dissim_gm1(timechunk_matrix, pts_per_timewin, num_timewin, fft_win, fs)
    dissim = dfin

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
    avwin = int(fs * avtime)
    sampwin = int(fs * timewin)
    ntwin = len(p_filt) // sampwin  # Number of minutes

    p_filt = p_filt[:sampwin * ntwin]
    p_filt = p_filt.reshape(sampwin, ntwin)
    p_filt = p_filt ** 2
    numavwin = p_filt.shape[0] // avwin
    p_av = []

    for jj in range(ntwin):
        avwinmatrix = p_filt[:, jj].reshape(avwin, numavwin)
        p_avi = np.mean(avwinmatrix, axis=1)
        p_av.append(p_avi)

    p_avtot = np.hstack(p_av)

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
        # # print(f"pout: {pout[i]}")
        # print(f"poutNew: {poutNew[i]}")

        # print("p.size: " + str(p.size))
        # print("First 50 indexes of poutNew: " + str(poutNew[:50]))
        # print("Last 50 indexes of poutNew: " + str(poutNew[p.size-50:p.size + 1]))

        poutNew = poutNew[:p.size+1]
        # print("First 50 indexes of poutNew: " + str(poutNew[:50]))
        # print("Last 50 indexes of poutNew: " + str(poutNew[p.size - 50:]))


        print("poutNew size: " + str(len(poutNew)))
        print("p size: " + str(len(p)))

        # Filter the signal with adjusted cutoff frequencies
        p_filt, _ = dylan_bpfilt(poutNew, 1 / fs, flow, fhigh)
        # print("p_filt size: " + str(len(p_filt)))
        # print("p_filt[:10 = ", p_filt[:10])

        # Call the frankenfunc_testscript function
        SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim = frankenfunc_testscript(p_filt, fs, timewin, fft_win,
                                                                                         avtime, flow, fhigh)


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

SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim = wav_frankenfunction_reilly(num_bits, peak_volts, file_dir, rs,
                                                                                     timewin, avtime, fft_win, arti,
                                                                                     flow, fhigh)

# Print the output shapes
print('SPLrms shape:', SPLrms.shape)
print("SPLrms: " + str(SPLrms))
print('SPLpk shape:', SPLpk.shape)
print("SPLpk: " + str(SPLpk))
print('impulsivity shape:', impulsivity.shape)
print("impulsivity: " + str(impulsivity))
print('peakcount shape:', peakcount.shape)
print("peakcount: " + str(peakcount))
print('autocorr shape:', autocorr.shape)
print("autocorr: " + str(autocorr))
print('dissim shape:', dissim.shape)
print("dissim: " + str(dissim))

# You can also save the output to files if needed
np.savez('output.npz', SPLrms=SPLrms, SPLpk=SPLpk, impulsivity=impulsivity,
         peakcount=peakcount, autocorr=autocorr, dissim=dissim)