import math
from array import array
from statistics import median

import numpy as np
import os

import pandas as pd
import scipy
from numpy import size, zeros, transpose, shape, mean
from pandas._testing import to_array
from scipy.io import wavfile
from scipy.signal import resample
from scipy.stats import kurtosis, moment
import time
from scipy.signal import find_peaks
from scipy.signal import hilbert
from numpy.fft import fft


def f_WAV_frankenfunction_reilly(num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh):
    num_files = len(file_dir)
    p = []  # Empty variable
    pout = []
    SPLrms = []  # Empty variable for SPLrms allows func to build from 0
    SPLpk = []
    impulsivity = []
    peakcount = []
    autocorr = None
    dissim = []

    start_time = time.time()
    for ii in range(num_files):  # ii is an index value - completes 1 loop for every file until all files are analyzed
        print(f"\n{ii + 1} out of {num_files}")  # lists ii as a variable to tell you every time it completes a loop
        filename = os.path.join(r'C:\Users\rlessard\Desktop\5593 organized\240406165958_240413225912\output_10min',
                                file_dir[ii])
        rs = (10 ** (RS / 20))
        max_count = 2 ** num_bits
        conv_factor = peak_volts / max_count
        fs, x = wavfile.read(filename)
        # print("fs: " + str(fs))
        # print("x: " + str(x))
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

        if num_bits == 24:
            x = x >> 8  # Bit shift

        v = x.astype(float) * conv_factor

        # for i in range(len(v)): # Used for shorter sound files (<= 9 minutes)
        # if i % 9 != 0:
        # v[i] = 0

        # print("v size: " + str(len(v)))
        # print("v: " + str(v[:10]))
        p = v / rs
        # print("p size: " + str(len(p)))
        # print("p: " + str(p[:5]))
        if arti == 1:
            p = p[6 * fs - 1:]  # Different indexing than MATLAB (Must include " - 1")
            # print("p 1: " + str(p[:5]))
        pout.extend(p)
        # pout.insert(0, 0) # Used for shorter sound files (<= 9 minutes)
        # pout.insert(0, pout[0])
        # print("pout size: " + str(len(pout)))
        # print("pout: " + str(pout[:10]))
        middle = len(pout) // 2
        # print("pout middle: " + str(pout[middle - 5:middle + 5]))
        # print("pout last: " + str(pout[-5:]))

        p = []
        p_filt = dylan_bpfilt(pout, 1 / fs, flow, fhigh)
        # print("p_filt size: " + str(len(p_filt)))
        # print("p_filt: " + str(p_filt))
        pout = []
        pts_per_timewin = int(timewin * fs)  # Number of samples per time window - set window * 576 kHz sample rate
        # print("pts_per_timewin ", pts_per_timewin)

        num_timewin = math.floor(len(p_filt) / pts_per_timewin) + 1
        # print("num_timewin ", num_timewin)

        # Pad the signal to accomodate the extra time window
        padding_length = num_timewin * pts_per_timewin - len(p_filt)
        # print("padding_length ", padding_length)
        p_filt_padded = np.concatenate((p_filt, np.zeros(padding_length)))
        # print("p_filt_padded ", p_filt_padded)

        # num_timewin = len(p_filt) // pts_per_timewin + 1  # Number of time windows contained in the sound file
        # print("num_timewin ", num_timewin)
        # trimseries = p_filt[
        # :(pts_per_timewin * num_timewin)]  # Trims the time series to make it fit into matrix columns
        # print("trimseries", trimseries)

        timechunk_matrix = p_filt_padded.reshape(pts_per_timewin, num_timewin)

        # print("p_filt_padded: " + str(p_filt_padded))
        # print("pts_per_timewin: " + str(pts_per_timewin))
        # print("num_timewin: " + str(num_timewin))

        # print("timechunk_matrix shape ", timechunk_matrix.shape)
        # print("timechunk_matrix: " + str(timechunk_matrix))

        # timechunk_matrix = trimseries.reshape((pts_per_timewin, num_timewin))  # Shapes trimmed series into sample per
        # time chunk rows x number of time windows
        # in file columns

        [tcmSizeA, tcmSizeB] = timechunk_matrix.shape
        # print("timechunk_matrix size: " + str(tcmSizeA) + " " + str(tcmSizeB))
        # print("timechunk_matrix: " + str(timechunk_matrix))

        # rms_matrix = np.sqrt(np.mean(np.square(timechunk_matrix), axis=0))    # Calculates the rms pressure of the matrix
        # rms_matrix = rms_by_row(timechunk_matrix)
        # timechunk_matrix = transpose(timechunk_matrix)
        # rms_matrix = scipy.linalg.norm(timechunk_matrix, axis=0) / np.sqrt(timechunk_matrix.shape[0])
        # rms_matrix = calculate_rms_matrix(timechunk_matrix)
        # rms_matrix = custom_rms(timechunk_matrix)
        # rms_matrix = np.sqrt(np.mean(timechunk_matrix ** 2, axis=0))
        # rms_matrix = np.sqrt(np.mean(np.square(timechunk_matrix), axis=0))
        # rms_matrix = rmsValue(timechunk_matrix, len(timechunk_matrix))
        # rms_matrix = calculate_rms(timechunk_matrix)
        # rms_matrix = np.sqrt(np.mean(np.square(timechunk_matrix), axis=0))
        # rms_matrix = nanrms(timechunk_matrix)
        # rms_matrix = rms(timechunk_matrix)
        # mean_square = moment(timechunk_matrix, moment=2, axis=1, nan_policy='omit')
        # rms_matrix = np.sqrt(mean_square + np.mean(timechunk_matrix ** 2, axis=1))

        # rms_matrix = transpose(rms_matrix)
        rms_matrix = rms_reilly(timechunk_matrix, 0)
        # print("rms_matrix shape ", rms_matrix.shape)
        # print("rms_matrix: " + str(rms_matrix))
        # rms_matrix = scipy.linalg.norm(timechunk_matrix, axis=1)

        SPLrmshold = 20 * np.log10(rms_matrix)  # Log transforms the rms pressure
        # print("SPLrmshold size: " + str(SPLrmshold.shape))
        # print("SPLrmshold: " + str(SPLrmshold))

        # SPLpkhold = np.max(20 * np.log10(np.abs(timechunk_matrix)), axis=0)  # Identifies the peak in rms pressure
        # flattened = [item for sublist in timechunk_matrix for item in sublist]
        # SPLpkhold = max(20 * math.log10(abs(x)) for x in flattened if x != 0)

        # print("tcm size: " + str(timechunk_matrix.shape))
        # print("tcm: " + str(timechunk_matrix))

        SPLpkhold = column_max_SPL(timechunk_matrix)

        # print("SPLpkhold: " + str(SPLpkhold))

        # print(np.abs(timechunk_matrix))

        tcm_abs = abs(timechunk_matrix)
        # print("tcm abs shape: " + str(tcm_abs.shape))
        # print("tcm_abs: " + str(tcm_abs))
        l10_tcm = np.log10(tcm_abs)
        # print("l10 tcm shape: " + str(l10_tcm.shape))
        # print("l10_tcm: " + str(l10_tcm))
        l10_tcm_20 = 20 * l10_tcm
        # print("l10_20 tcm shape: " + str(l10_tcm_20.shape))
        # print("l10_tcm_20: " + str(l10_tcm_20))

        max_values_per_column = np.max(l10_tcm_20, axis=0)
        # print("max_values_per_column: " + str(max_values_per_column))

        SPLpkhold = np.max(20 * np.log10(np.abs(timechunk_matrix)), axis=0)
        # SPLpkhold = np.max(l10_tcm_20, axis=0)
        # print("SPLpkhold shape: " + str(SPLpkhold.shape))
        # print("SPLpkhold: " + str(SPLpkhold) + "\n")
        # SPLpkhold = find_greatest_in_columns(l10_tcm_20)
        # SPLpkhold = calculate_splpkhold(timechunk_matrix)
        # tposeSPL = transpose(l10_tcm_20)
        # SPLpkhold = find_greatest_in_columns(tposeSPL)

        # Step 1: Calculate the absolute value
        # abs_timechunk_matrix = np.abs(timechunk_matrix)
        # print("Absolute values:\n", abs_timechunk_matrix)

        # Step 2: Compute the logarithm base 10
        # log_timechunk_matrix = np.log10(abs_timechunk_matrix)
        # print("Logarithm base 10 of absolute values:\n", log_timechunk_matrix)

        # Step 3: Multiply the logarithm by 20
        # scaled_log_timechunk_matrix = 20 * log_timechunk_matrix
        # print("Scaled logarithm values:\n", scaled_log_timechunk_matrix)

        # Create a list of all values in index 0
        # zero_list = []
        # [height, width] = scaled_log_timechunk_matrix.shape
        # for i in range(height):
        # zero_list.append(scaled_log_timechunk_matrix[i][0])
        # print("Max in column " + str(0) + ": " + str(max(zero_list)))

        # zero_list = []
        # for i in range(height):
        # zero_list.append(scaled_log_timechunk_matrix[i][1])
        # print("Max in column " + str(1) + ": " + str(max(zero_list)))

        # print("Scaled logarithm values:\n", scaled_log_timechunk_matrix[:5])
        # print("Scaled logarithm values:\n", scaled_log_timechunk_matrix[-5:])

        # [dim1, dim2] = scaled_log_timechunk_matrix.shape
        # max_vals = []

        # for i in range(dim2):
        # max_vals.append(scaled_log_timechunk_matrix[:, 0].max())

        # print("max_vals: " + str(max_vals))
        # print("First: " + str(scaled_log_timechunk_matrix[0][0]))
        # print("First: " + str(scaled_log_timechunk_matrix[-1][0]))

        # Step 4: Find the maximum value in each column
        # SPLpkhold = np.max(scaled_log_timechunk_matrix, axis=0)
        # SPLpkhold = np.max((20 * np.log10(np.abs(timechunk_matrix))), axis=0)
        # print("SPLpkhold:", SPLpkhold)

        # SPLpkhold = np.max(20 * np.log10(abs(timechunk_matrix)), axis=0)
        # print("SPLpkhold size: " + str(shape(SPLpkhold)))
        # print("SPLpkhold: " + str(SPLpkhold))
        SPLrms.extend(SPLrmshold)  # This var SPLrms is the outputted rms matrix
        # print("SPLrms: " + str(SPLrms))
        SPLpk.extend(SPLpkhold)  # Generates the pk matrix
        # print("SPLpk: " + str(SPLpk))
        # This means the function will give two output matrices [SPLrms SPLpk]
        # So you need to call them both when you use the function

        # Impulsivity
        print("timechunk_matrix: " + str(timechunk_matrix))

        kmat = kurtosis_reilly(timechunk_matrix, tcmSizeB=tcmSizeB)
        # kmat = kurtosis_ignore_zeros(timechunk_matrix)
        # kmat = kurtosis(timechunk_matrix)

        print("kmat: " + str(kmat))
        impulsivity.extend(kmat)
        print("impulsivity: " + str(impulsivity))


        # print("\np_filt_padded: " + str(p_filt_padded[-10:]))
        # print("fs: " + str(fs))
        # print("timewin: " + str(timewin))
        # print("avtime: " + str(avtime))

        # Periodicity
        pkcount, acorr = f_solo_per_GM2(p_filt_padded, fs, timewin, avtime)
        # print("pkcount: " + str(pkcount))
        # print("acorr: " + str(acorr))

        peakcount.extend(pkcount)
        # print("pkcount: " + str(peakcount))

        if autocorr is None:
            autocorr = acorr
        else:
            autocorr = np.column_stack((autocorr, acorr))

        acmax = np.max(autocorr, axis=0)
        # print("acmax: " + str(acmax))
        acmin = np.min(autocorr, axis=0)
        # print("acmin: " + str(acmin))
        acmean = np.mean(autocorr, axis=0)
        # print("acmean: " + str(acmean))
        acmedian = np.median(autocorr, axis=0)
        # print("acmedian: " + str(acmedian))

        # D-index
        Dfin = f_solo_dissim_GM1(timechunk_matrix, pts_per_timewin, num_timewin, fft_win, fs)
        # print("Dfin: " + str(Dfin))
        dissim.extend(Dfin)
        # print("dissim: " + str(dissim))

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")
    return SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim


def column_max_SPL(timechunk_matrix):
    # Get the number of columns (assuming all rows have the same length)
    num_columns = len(timechunk_matrix[0])

    SPLpkhold = []
    for col in range(num_columns):
        column_values = [abs(row[col]) for row in timechunk_matrix if row[col] != 0]
        if column_values:
            max_value = max(column_values)
            spl = 20 * math.log10(max_value)
            SPLpkhold.append(spl)
        else:
            SPLpkhold.append(float('-inf'))  # Or any other value to represent undefined SPL

    return SPLpkhold


def kurtosis_ignore_zeros(timechunk_matrix):
    # Initialize an array to store kurtosis values for each column
    kurtosis_values = np.zeros(timechunk_matrix.shape[1])

    # Loop through each column to calculate kurtosis
    for i in range(timechunk_matrix.shape[1]):
        column_data = timechunk_matrix[:, i]

        # Ignore zero values
        non_zero_data = column_data[column_data != 0]

        if len(non_zero_data) > 0:
            mean_val = np.mean(non_zero_data)  # Mean of the non-zero data
            s2 = np.sum((non_zero_data - mean_val) ** 2) / len(non_zero_data)  # Variance
            s4 = np.sum((non_zero_data - mean_val) ** 4) / len(non_zero_data)  # Fourth moment

            # Calculate kurtosis
            kurtosis_values[i] = s4 / (s2 ** 2)
        else:
            # If all values are zero, kurtosis is undefined, so we set it to NaN or some other value
            kurtosis_values[i] = np.nan

    return kurtosis_values


def kurtosis_reilly(x, flag=1, dim=None, tcmSizeB=0):
    # print("x: " + str(x))
    # print("flag: " + str(flag))
    # print("nargin < 2 || isempty(flag)")
    flag = 1
    # print("flag = 1")

    # Determine the dimension if not provided
    if flag not in (0, 1) and flag is not None:
        raise ValueError("Bad flag value: flag should be 0 or 1.")

    if dim is None:
        # print("nargin < 3 or isempty(dim)")

        # Handle the special case where x is empty.
        if np.array_equal(x, np.array([])):
            print("x is empty")
            return np.nan

        # Determine the dimension along which np.nanmean will work.
        dim = next((i for i, s in enumerate(x.shape) if s != 1), None)
        # print("dim: " + str(dim))

        if dim is None:
            dim = 0
            # print("dim = 0")

    # Center x, compute its fourth and second moments
    x0 = x - np.nanmean(x, axis=dim, keepdims=True)
    # x0_offset = subtract_ten(x0, tcmSizeB)
    # first_elem = x0[0][0]
    # print("first_elem: " + str(first_elem - tcmSizeB))

    [x0_dim1, x0_dim2] = x0.shape
    # print("x0 shape: " + str(x0_dim1) + ", " + str(x0_dim2))

    # x0 = x0_offset
    # print("x0: " + str(x0[:2]))

    # print("n(x,dim): " + str(x - np.nanmean(x, axis=dim, keepdims=True)))
    # print("x0: " + str(x0[:2]))
    # print("x0: " + str(x0[-2:]))
    # [x0_dim1, x0_dim2] = x0.shape
    # print("x0 shape: " + str(x0_dim1) + ", " + str(x0_dim2))
    s2 = nanmean(x0**2, axis=dim)  # biased variance estimator
    print("s2: " + str(s2))
    m4 = np.nanmean(np.power(x0, 4), axis=0)
    # print("m4: " + str(m4[:2]))

    # k = kurtosis(x0, axis=dim)
    # print("kurtosis: " + str(k))

    k = m4 / np.square(s2)
    # print("k: " + str(k[:2]))

    # Bias correction
    if flag == 0:
        n = np.sum(~np.isnan(x), axis=dim)
        n[n < 4] = np.nan  # bias correction not defined for n < 4
        k = ((n + 1) * k - 3 * (n - 1)) * (n - 1) / ((n - 2) * (n - 3)) + 3

    return k


def nanmean(arr, axis=None):
    # Convert the input to a NumPy array if it isn't one already
    arr = np.array(arr)

    # Create a mask to ignore NaN, 0, and empty values
    mask = ~np.isnan(arr) & (arr != 0) & (arr != "")

    # Apply the mask and calculate the mean
    masked_arr = np.where(mask, arr, np.nan)
    return np.nanmean(masked_arr, axis=axis)


def subtract_ten(matrix, tcmSizeA):
    return [[element - tcmSizeA for element in row] for row in matrix]


def calculate_splpkhold(timechunk_matrix):
    # Calculate the absolute values of the elements in the matrix
    abs_matrix = np.abs(timechunk_matrix)

    # Compute 20 * log10 of the absolute values
    log_matrix = 20 * np.log10(abs_matrix)

    # Find the maximum value in the matrix
    splpkhold = np.max(log_matrix)

    return splpkhold


def find_greatest_in_columns(matrix):
    # matrix = transpose(matrix)
    # Ensure the matrix is a 2D numpy array
    matrix = np.atleast_2d(matrix)

    # Initialize an empty list to hold the maximum values of each column
    max_list = []

    # Determine dimension sizes
    [mat_dim1, mat_dim2] = matrix.shape

    # Iterate over each column index
    for col in range(mat_dim1):
        # Append the maximum value in the current column to max_list
        max_list.append(np.max(matrix[col, :]))

    print("max_list size: " + str(shape(max_list)))
    print("max_list: " + str(max_list[:15]))
    print("max_list: " + str(max_list[-15:]))
    return max_list


def custom_ln(x, iterations=100):
    print("Start custom ln")
    """
    Approximate the natural logarithm of x using the Newton-Raphson method.
    """
    if x <= 0:
        raise ValueError("Input must be greater than zero")
    guess = x
    for _ in range(iterations):
        guess = guess - (np.exp(guess) - x) / np.exp(guess)
    return guess


def custom_log10(x):
    print("Start custom log10")
    """
    Compute the base-10 logarithm of x using custom_ln.
    """
    ln10 = custom_ln(10)
    return custom_ln(x) / ln10


def compute_SPLrmshold(rms_matrix):
    print("Start compute SPLrmshold")
    """
    Compute the SPLrmshold for the given RMS matrix.
    """
    SPLrmshold = np.array([20 * custom_log10(x) for x in rms_matrix.flatten()]).reshape(rms_matrix.shape)
    return SPLrmshold


def rms_reilly(x, dim=None):
    global vertical_averages_sqrt
    if np.isrealobj(x):
        # print("x[:5]" + str(x[:5]))
        # print("x size: " + str(shape(x)))
        # print("a")
        if dim is not None:
            # print("b")

            sqmexx = np.square(x)
            # [shape_a, shape_b] = np.shape(sqmexx)
            # print("sqmexx square size: " + str(shape_a) + " " + str(shape_b))
            # print("sqmexx square: ")

            # print("\naverage of 0: " + str(mean(sqmexx[0, :])))

            # print("np mean: " + str(np.sqrt(np.mean(sqmexx, axis=1))))

            [sqmexx_a, sqmexx_b] = shape(sqmexx)
            # sqmexx = mean(sqmexx)

            # sqmexx = np.arange(sqmexx_a * sqmexx_b).reshape(sqmexx_a, sqmexx_b)
            # split_arrays = [sqmexx[i, :] for i in range(sqmexx.shape[1])]
            # for i in range(len(split_arrays)):
            # print("\nsplit_arrays " + str(i) + ": " + str(split_arrays[i]))

            sqmexx = sqmexx.reshape(-1)
            # print("sqmexx shape: " + str(sqmexx.shape))
            # print("sqmexx: " + str(sqmexx))

            # array_2d = np.reshape(sqmexx, (shape_a, shape_b))

            # mat2d = np.zeros((shape_a, shape_b))
            # print("mat2d shape: " + str(mat2d.shape))

            reshaped_sqmexx = sqmexx.reshape((sqmexx_b, sqmexx_a))
            transposed_sqmexx = reshaped_sqmexx.T
            # print("transposed_sqmexx shape: " + str(transposed_sqmexx.shape))
            # print("transposed_sqmexx 0:2: " + str(transposed_sqmexx[0:2]) + "\n")
            # print("transposed_sqmexx -2 " + str(transposed_sqmexx[-2]))
            # print("transposed_sqmexx -1: " + str(transposed_sqmexx[-1]) + "\n")
            vertical_averages = mean(transposed_sqmexx, axis=0)
            # print("Averages: " + str(vertical_averages))
            vertical_averages_sqrt = np.sqrt(vertical_averages)
            # print("Vertical averages sqrt: " + str(vertical_averages_sqrt))

            # Create an array of the first element in each of the 5568000 rows
            # vert_array = []
            # for i in range(sqmexx_a):
            # print(mat2d[0, i])
            # vert_array.append(mat2d[0, i])
            # print("Column 1 average: " + str(np.mean(vert_array)))

            # column_averages = mean_reilly(mat2d, 1)
            # print("Averages: " + str(column_averages))
            # print("Square roots: " + str(np.sqrt(column_averages)))
            # Print the results
            # for i, avg in enumerate(column_averages):
            # print("Average of column " + str(i) + ": " + str(math.sqrt(avg)))

            # print("mat2d shape: " + str(mat2d.shape))
            # print("mat2d: " + str(mat2d))

            # print("new sqmexx shape: " + str(sqmexx.shape))
            # print("new sqmexx: " + str(sqmexx))

            # sqmexx = mean(split_arrays)
            # sqmexx = [np.mean(arraysq) for arraysq in split_arrays]
            # sqmexx_list = []
            # for i in range(len(sqmexx)):
            # sqmexx_list.append(math.sqrt(sqmexx[i]))

            # averages = [mean(arr) for arr in split_arrays]
            # for i, avg in enumerate(averages):
            # print(f"Average of array {i + 1}: {avg}")
            # print(f"Square root of array {i + 1}: {np.sqrt(avg)}\n")

            # sqmexx = transpose(sqmexx)
            # for i in range(10):
            # print(str(i) + " " + str(sqmexx[i, :]))

            # print(str(shape_a / 2 - 2) + " " + str(sqmexx[0, :]))
            # print(str(shape_a / 2 - 1) + " " + str(sqmexx[1, :]))
            # print(str(shape_a / 2) + " " + str(sqmexx[2, :]))
            # print(str(shape_a / 2 + 1) + " " + str(sqmexx[0, :]))
            # print(str(shape_a / 2 + 2) + " " + str(sqmexx[1, :]) + "\n")

            # print(str(shape_a - 2) + " " + str(sqmexx[-3, :]))
            # print(str(shape_a - 1) + " " + str(sqmexx[-2, :]))
            # print(str(shape_a) + " " + str(sqmexx[-1, :]))

            # column_averages = np.mean(sqmexx, axis=0)
            # for i in range(len(column_averages)):
            # print("column_averages " + str(i + 1) + ": " + str(math.sqrt(column_averages[i])))

            # print("sqmexx_list: " + str(sqmexx_list))
            # print("sqmexx mean: " + str(sqmexx))

            # print("return: " + str(np.sqrt(np.mean(x * x))))
            # df = pd.DataFrame(x, index=None)
            # rows = len(df.axes[0])
            # cols = len(df.axes[1])

            # print("np square x: " + str(np.square(x)))

            # rms_mat = []
            # [sizeA, sizeB] = shape(x)
            # for i in range(int(sizeA)):
            # rms_mat.append(np.sqrt(np.mean(x * x)))
            # rms_mat.append(np.sqrt(np.mean(x * x)))
            # print("rms_mat[" + str(i) + "]: " + str(x[i]))
            # rms_mat = transpose(rms_mat)
            # print("rms_mat size: " + str(shape(rms_mat)))
            # print("rms_reilly return: " + str(rms_mat[:3][:10]))
            # return rms_mat
            # return np.sqrt(np.mean(x * x))
        # else:
        # print("c")
        # return np.sqrt(np.mean(x * x, axis=dim))
    # else:
    # print("d")
    # if dim is None:
    # print("e")
    # return np.sqrt(np.mean(np.real(x) * np.real(x) + np.imag(x) * np.imag(x)))
    # else:
    # print("f")
    # return np.sqrt(np.mean(np.real(x) * np.real(x) + np.imag(x) * np.imag(x), axis=dim))
    # print("Returns " + str(vertical_averages_sqrt))
    return vertical_averages_sqrt


def mean_reilly(x, dim=None, flag='default', nanflag='includenan'):
    # Convert x to numpy array if it's not already
    x = np.asarray(x)

    # Handle 'all' dimension
    if dim == 'all':
        x = x.flatten()
        dim = 0

    # If dim is not specified, find the first non-singleton dimension
    if dim is None:
        dim = next((i for i, s in enumerate(x.shape) if s != 1), 0)

    # Handle different flag types
    if flag == 'default':
        if x.dtype in [np.float32, np.float64]:
            dtype = x.dtype
        else:
            dtype = np.float64
    elif flag == 'double':
        dtype = np.float64
    elif flag == 'native':
        dtype = x.dtype
    else:
        raise ValueError("Invalid flag")

    # Handle NaN values
    if nanflag == 'omitnan':
        return np.nanmean(x, axis=dim, dtype=dtype)
    elif nanflag == 'includenan':
        return np.mean(x, axis=dim, dtype=dtype)
    else:
        raise ValueError("Invalid nanflag")


def convert_2d_to_1d_arrays(array_2d):
    """
    Converts a 2D array into several 1D arrays.

    Parameters:
    array_2d (list of list of int/float): The 2D array to convert.

    Returns:
    list of list of int/float: A list containing the 1D arrays.
    """
    return [list(row) for row in array_2d]


def rms(x, dim=None):
    x = np.asarray(x)  # Convert input to numpy array if it's not already

    if dim is None:
        # If no dimension is specified, calculate RMS over the entire array
        return np.sqrt(np.mean(np.abs(x) ** 2))
    else:
        # Calculate RMS along the specified dimension
        return np.sqrt(np.mean(np.abs(x) ** 2, axis=dim))


def nanrms(x):
    """
        Calculate the Root Mean Square (RMS) of a signal.

        :param x: Array-like input signal
        :return: RMS value
        """
    # Convert input to numpy array if it's not already
    x = np.array(x)

    # Get the number of samples
    N = len(x)

    # Calculate the RMS
    x_RMS = np.sqrt((1 / N) * np.sum(np.abs(x) ** 2))

    return x_RMS


def calculate_rms(values):
    """
    Calculate the Root Mean Square of a list of values.

    :param values: List or array of numeric values
    :return: Root Mean Square value
    """
    # Convert input to numpy array if it's not already
    values = np.array(values)

    # Square all values
    squared = np.square(values)

    # Calculate the mean of squared values
    mean_squared = np.mean(squared)

    # Take the square root of the mean
    rms = np.sqrt(mean_squared)

    return rms


def rmsValue(arr, n):
    square = 0
    mean = 0.0
    root = 0.0

    # Calculate square
    for i in range(0, n):
        square += (arr[i] ** 2)

    # Calculate Mean
    mean = (square / (float)(n))

    # Calculate Root
    root = math.sqrt(mean)

    return root


def custom_rms(x):
    # Convert input to numpy array if it's not already
    array_2d = np.array(x)

    # Transpose
    array_2d = transpose(array_2d)

    # Get the number of rows
    rows = array_2d.shape[0]

    # Initialize the output array
    rms_values = np.zeros(rows)

    # Calculate RMS for each row
    for i in range(rows):
        # Extract the current row
        row = array_2d[i, :]

        # Square all elements in the row
        squared = np.square(row)

        # Calculate the mean of squared values
        mean_squared = np.mean(squared)

        # Take the square root and store the result
        rms_values[i] = np.sqrt(mean_squared)

    return rms_values


def calculate_rms_matrix(timechunk_matrix):
    num_rows = len(timechunk_matrix[0])
    num_cols = len(timechunk_matrix)
    rms_matrix = []

    for row in range(num_rows):
        sum_of_squares = sum(timechunk_matrix[col][row] ** 2 for col in range(num_cols))
        mean_of_squares = sum_of_squares / num_cols
        rms = math.sqrt(mean_of_squares)
        rms_matrix.append(rms)

    return rms_matrix


def rms_by_row(array_2d):
    # Calculate RMS for each of the 10 arrays (columns)
    rms_values = np.sqrt(np.mean(array_2d ** 2, axis=0))
    return rms_values


def dylan_bpfilt(ts, samint, flow, fhigh):
    npts = len(ts)
    reclen = npts * samint
    spec = np.fft.fft(ts, npts)
    aspec = np.abs(spec)
    pspec = np.angle(spec)
    freq = np.fft.fftshift(np.arange(-npts / 2, npts / 2) / reclen)

    if fhigh == 0:
        fhigh = 1 / (2 * samint)

    ifr = np.where((np.abs(freq) >= flow) & (np.abs(freq) <= fhigh))[0]
    filtspec2 = np.zeros_like(spec, dtype=complex)
    rspec = np.zeros_like(spec)
    ispec = np.zeros_like(spec)

    rspec[ifr] = aspec[ifr] * np.cos(pspec[ifr])
    ispec[ifr] = aspec[ifr] * np.sin(pspec[ifr])
    filtspec2 = rspec + 1j * ispec

    tsfilt = np.real(np.fft.ifft(filtspec2, npts))

    return tsfilt


def f_solo_per_GM2(p_filt, fs, timewin, avtime):
    print("fs is " + str(fs))
    p_av = []
    p_avtot = []
    avwin = int(fs * avtime)
    print("avwin is " + str(avwin))
    sampwin = int(fs * timewin)
    print("sampwin is " + str(sampwin))
    print("Length of p_filt " + str(len(p_filt)))
    ntwin = len(p_filt) // sampwin  # Number of minutes
    print("ntwin is " + str(ntwin))
    p_filt = p_filt[:sampwin * ntwin]
    # print("p_filt is " + str(p_filt[57504005-5:57504005-4]))

    # p_filt = p_filt.reshape(sampwin, ntwin)
    # p_filt = np.reshape(p_filt, (sampwin, ntwin))

    p_filt = distribute_array(p_filt, ntwin)

    # print("p_filt shape: " + str(p_filt.shape))
    # print("p_filt: " + str(p_filt))
    # p_filt = arr_2d

    # print("p_filt shape: " + str(size(p_filt)))
    # print("p_filt: " + str(p_filt))

    # print("p_filt shape is " + str(p_filt.shape))
    # print("p_filt is " + str(p_filt[:, 0]))
    # print("p_filt is " + str(p_filt[1]))
    # print("p_filt is " + str(p_filt[2]))
    # print("p_filt is " + str(p_filt[9]))
    # print("p_filt is " + str(p_filt[:, 10]))
    # print("p_filt is " + str(p_filt))

    p_filt = p_filt ** 2
    # print("\np_filt shape is " + str(p_filt.shape))
    # print("p_filt is " + str(p_filt))
    # print("p_filt: " + str(p_filt[0][:3]))
    # print("p_filt 10 x 10: " + str(p_filt[:10, :10]))
    # print("p_filt: " + str(p_filt[0:5750402, 0:5]))

    numavwin = p_filt.shape[0] // avwin
    # print("numavwin is " + str(numavwin))

    p_av = []
    # p_filt = transpose(p_filt)
    print("\n")

    for jj in range(ntwin):
        # avwinmatrix = reshape_matrix(p_filt, avwin, numavwin)
        avwinmatrix = distribute_array_2d(p_filt[:, jj], numavwin, avwin)
        # avwinmatrix = p_filt[:, jj].reshape(avwin, numavwin)
        # avwinmatrix = np.reshape(p_filt[:, jj], (numavwin, avwin))
        # avwinmatrix = [p_filt[i:i + avwin, jj] for i in range(0, len(p_filt), numavwin)]
        # avwinmatrix = custom_reshape(p_filt, [numavwin, avwin])
        # p_filt_column = np.array([row[jj] for row in p_filt])
        # avwinmatrix = custom_reshape(p_filt_column, [avwin, numavwin])
        # p_filt_column = p_filt[:, jj]
        # avwinmatrix = p_filt_column.reshape(numavwin, avwin)


        # avwinmatrix = custom_reshape(p_filt_column, [avwin, numavwin])

        # for idx, chunk in enumerate(avwinmatrix):
            # print(f"avwinmatrix[{idx}] is {chunk}")

        # print("avwinmatrix shape is " + str(avwinmatrix.shape))
        # print("avwinmatrix is " + str(avwinmatrix[jj]))
        # print("avwinmatrix is " + str(avwinmatrix[1]))
        # print("avwinmatrix is " + str(avwinmatrix[2]))
        # print("avwinmatrix is " + str(avwinmatrix[-1]))
        # print("avwinmatrix is " + str(avwinmatrix))
        p_avi = np.mean(avwinmatrix, axis=0)
        # print("p_avi: " + str(p_avi) + "\n")

        p_av.append(p_avi)
        # print("p_av: " + str(p_avi))

    p_av_dim1, p_av_dim2 = np.shape(p_av)
    # p_av = custom_reshape(p_av, (p_av_dim1, p_av_dim2))


    p_av = np.transpose(p_av)
    p_avtot = np.array(p_av)
    # print("p_av size: " + str(shape(p_av)))
    # print("p_av: " + str(p_av))
    # print("p_av: " + str(p_av[0]))
    # print("p_av: " + str(p_av[-1]))

    # p_avtot = p_avtot.append(p_av)
    # p_avtot = np.hstack((p_avtot, p_av))
    # p_avtot = p_avtot + [p_av]
    # p_avtot.extend(p_av)

    # p_avtot = np.array(p_av).T
    # print("p_avtot is " + str(p_avtot))

    shape0, shape1 = np.shape(p_avtot)

    max_lag = int(shape0 * 0.7)
    acorr = np.zeros((max_lag + 1, shape1))
    pkcount = np.zeros(shape1)

    for zz in range(shape1):
        P, _ = correl_5(p_avtot[:, zz], p_avtot[:, zz], max_lag, 0)
        acorr[:, zz] = P
        # print("acorr: " + str(acorr[:, zz]))
        pks, _ = find_peaks(acorr[:, zz], prominence=0.5)
        pkcount[zz] = len(pks)

    # print("pkcount: " + str(pkcount))
    # ("acorr: " + str(acorr))
    return pkcount, acorr


def transpose_same_dimensions(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    # Create a new matrix with the same dimensions
    result = [[0 for _ in range(cols)] for _ in range(rows)]

    # Perform the transpose operation
    for i in range(rows):
        for j in range(cols):
            result[i][j] = matrix[j][i]

    return result


def custom_reshape(array, new_shape):
    # Assuming array is already flat (1D)
    result = []
    for i in range(0, len(array), new_shape[1]):
        result.append(array[i:i + new_shape[1]])
    return result

def custom_mean(matrix):
    return [sum(col) / len(col) for col in zip(*matrix)]


def custom_reshape(array, new_shape):
    flat = [item for sublist in array for item in sublist]
    result = []
    for i in range(0, len(flat), new_shape[1]):
        result.append(flat[i:i + new_shape[1]])
    return result


def reshape_matrix(original_matrix, avwin, numavwin):
    # Reshape the matrix
    reshaped = original_matrix.reshape(9600, 600, 10)

    # Transpose and reshape to get the final shape
    final_matrix = reshaped.transpose(0, 2, 1).reshape(9600, 600)

    return final_matrix


def distribute_array(arr_1d, dim):
    # Calculate the number of elements per column (10% of the total)
    elements_per_column = len(arr_1d) // dim

    # Create an empty 2D array with 10 columns
    arr_2d = np.empty((elements_per_column, dim))

    # Iterate through the 1D array and place elements in the 2D array
    for i in range(dim):
        start_index = i * elements_per_column
        end_index = (i + 1) * elements_per_column
        arr_2d[:, i] = arr_1d[start_index:end_index]

    return arr_2d


def distribute_array_2d(arr_1d, num_columns, num_rows = None):
    # If num_rows is not specified, calculate it based on the array length and num_columns
    if num_rows is None:
        num_rows = len(arr_1d) // num_columns

    # Ensure the input array has enough elements
    if len(arr_1d) < num_rows * num_columns:
        raise ValueError("Input array is too small for the specified dimensions")

    # Create an empty 2D array with the specified dimensions
    arr_2d = np.empty((num_rows, num_columns))

    # Calculate the number of elements per column
    elements_per_column = num_rows

    # Iterate through the 1D array and place elements in the 2D array
    for i in range(num_columns):
        start_index = i * elements_per_column
        end_index = (i + 1) * elements_per_column
        arr_2d[:, i] = arr_1d[start_index:end_index]

    return arr_2d


def remove_zeros(matrix):
    # Filter out zero values from each row
    filtered_matrix = [[value for value in row if value != 0] for row in matrix]

    # Remove empty rows
    filtered_matrix = [row for row in filtered_matrix if row]

    return filtered_matrix


def f_solo_dissim_GM1(timechunk_matrix, pts_per_timewin, num_timewin, fft_win, fs):
    # print("timechunk_matrix: " + str(timechunk_matrix))
    # print("pts_per_timewin: " + str(pts_per_timewin))
    # print("num_timewin: " + str(num_timewin))
    # print("fft_win: " + str(fft_win))
    # print("fs: " + str(fs))

    pts_per_fft = int(fft_win * fs)  # Calc size fft window
    numfftwin = int(np.floor(pts_per_timewin / pts_per_fft))  # Number of fft windows

    Dfin = []
    D = []

    for kk in range(num_timewin - 1):
        analytic1 = hilbert(timechunk_matrix[:, kk])
        analytic2 = hilbert(timechunk_matrix[:, kk + 1])

        at1 = abs(analytic1) / np.sum(abs(analytic1))
        at2 = abs(analytic2) / np.sum(abs(analytic2))

        Dt = np.sum(abs(at1 - at2)) / 2

        s3a = timechunk_matrix[:, kk]
        s3a = s3a[:int(pts_per_fft * numfftwin)]
        s3a = s3a.reshape((pts_per_fft, numfftwin))
        ga = abs(fft(s3a, axis=0)) / s3a.shape[0]
        sfa = np.mean(ga, axis=1)
        Sfa = abs(sfa) / np.sum(abs(sfa))

        s3b = timechunk_matrix[:, kk + 1]
        s3b = s3b[:int(pts_per_fft * numfftwin)]
        s3b = s3b.reshape(pts_per_fft, numfftwin)
        gb = abs(fft(s3b)) / s3b.shape[0]
        sfb = np.mean(gb, axis=1)
        Sfb = abs(sfb) / np.sum(abs(sfb))

        Df = np.sum(abs(Sfb - Sfa)) / 2

        Di = Dt * Df

        D.append(Di)

    Dfin = np.array(D)
    return Dfin


def correl_5(ts1, ts2, lags, offset):
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


file_dir = os.listdir(r'C:\Users\rlessard\Desktop\5593 organized\240406165958_240413225912\output_10min')
num_bits = 16
RS = -178.3  # BE SURE TO CHANGE FOR EACH HYDROPHONE
# Sensitivity is based on hydrophone, not recorder
peak_volts = 2
arti = 1  # Make 1 if calibration tone present

# Analysis options
timewin = 60  # Length of time window in seconds for analysis bins
fft_win = 1  # Length of fft window in minutes
avtime = 0.1
flow = 50
fhigh = 300

SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim = f_WAV_frankenfunction_reilly(num_bits, peak_volts, file_dir,
                                                                                       RS, timewin, avtime, fft_win,
                                                                                       arti, flow, fhigh)
# scipy.io.savemat(r'C:\Users\rlessard\Desktop\SoundscapeCodeDesktop\DataOutput\240525225624_240531105601.mat', {'SPLrms': SPLrms, 'SPLpk': SPLpk, 'impulsivity': impulsivity, 'peakcount': peakcount, 'autocorr': autocorr, 'dissim': dissim})

print("\nSPLrms:\n" + str(SPLrms))
print("\nSPLpk:\n" + str(SPLpk))
print("\nImpulsivity:\n" + str(impulsivity))
print("\nPeakCount:\n" + str(peakcount))
print("\nAutocorr:\n" + str(autocorr))
print("\nDissim:\n" + str(dissim))
