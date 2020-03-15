import numpy as np

from library import QamSignal
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
from scipy.integrate import trapz
from scipy.signal import welch

from library.receiver_dsp  import cd_compensation
from library.dsp import equalizer
from library.dsp import rotate_overall_phase
from library.dsp import superscalar
from library.channel import NonlinearFiber
from resampy import resample
from library.dsp import MatchedFilter
from library.dsp import sync_correlation
from library.myplot import scatterplot
from scipy.signal import correlate


def calc_auto_acf(cpr):
    cprx = cpr[1, cpr[1, :] != 0]
    cpry = cpr[0, cpr[0, :] != 0]
    cpr = np.array([cprx, cpry])
    xpol_acf = np.abs(correlate(cpr[0, :], cpr[0, :]))
    xpol_acf = xpol_acf[len(cpr[0, :]) - 1:] / len(cpr[0, :])

    ypol_acf = np.abs(correlate(cpr[1, :], cpr[1, :]))
    ypol_acf = ypol_acf[len(cpr[1, :]) - 1:] / len(cpr[1, :])
    return xpol_acf / np.max(xpol_acf), ypol_acf / np.max(ypol_acf)


class ADC(object):
    def __call__(self, signal, sps_in_fiber, sps, inplace=True):
        # from resampy import resample
        import copy
        tempx = resample(signal[0], sps_in_fiber, sps, filter='kaiser_fast')
        tempy = resample(signal[1], sps_in_fiber, sps, filter='kaiser_fast')
        new_sample = np.array([tempx, tempy])
        signal.sps_in_fiber = sps
        if inplace:
            signal.ds_in_fiber = new_sample
        else:
            signal = copy.deepcopy(signal)
            signal.ds_in_fiber = new_sample
        return signal


import joblib


class Receiver(object):

    def __init__(self, kind):
        self.hxx = None
        self.hyy = None
        self.hyx = None
        self.hxy = None
        self.spectrum = None
        self.centroid = None
        self.bandwidth = None
        self.correlation = None
        self.span_setting = None
        self.error_xpol = None
        self.error_ypol = None
        self.xpol_snr_db = None
        self.ypol_snr_db = None
        self.kind = None
        self.span_setting = np.zeros((1, 15))
        self.correlation = None

    def find_bandwidth(self, freqs, pxx):

        pxx = 10 * np.log10(pxx / np.max(pxx))
        max_amplitude = np.max(pxx)
        down_3db = max_amplitude - 3
        index1 = None
        index2 = None
        for index, amp in enumerate(pxx):
            if np.abs(down_3db - amp) < 0.1 and index1 is None:
                index1 = index
        index2 = -index1

        freq1 = freqs[index1]
        freq2 = freqs[index2]
        bw = np.abs(freq2 - freq1)

        return bw

    def calc_centroid(self, freqs, pxx):

        fenzi = trapz(freqs * pxx, freqs)
        fenmu = trapz(pxx, freqs)
        return fenzi / fenmu

    def dsp_process(self, signal_name):
        signal = joblib.load(signal_name)[1]
        link_config_ith = int(signal_name.split('/')[-1].split('_')[0])
        config = joblib.load('./dataconfigv1_5wss')

        span_setting = config[link_config_ith][1:]
        self.span_setting = span_setting
        fiber_for_cdc = NonlinearFiber(alpha=0.2, D=16.7, gamma=1.3, length=np.sum(self.span_setting) * 80,
                                       step_length=20 / 1000)
        signal = cd_compensation(signal, fiber_for_cdc, True)
        # resample
        signal = ADC()(signal, signal.sps_in_fiber, signal.sps)
        # matched filter
        matched_filter = MatchedFilter(roll_off=0.2, sps=signal.sps_in_fiber)
        signal = matched_filter(signal)
        import matplotlib.pyplot as plt
        [freqs, pxx] = welch(signal[0, :], fs=signal.fs_in_fiber, detrend=False, nfft=1024,
                             return_onesided=False)
        self.centroid = self.calc_centroid(freqs, pxx)
        self.spectrum = np.fft.fftshift(pxx)
        self.bandwidth = self.find_bandwidth(freqs, pxx)
        # rotate phase
        signal[:] = rotate_overall_phase(signal[:], signal.symbol, sps=signal.sps)
        # LMS
        normal_factor = np.mean((np.abs(signal[:])) ** 2, axis=1, keepdims=True)
        signal[:] = signal[:] / np.sqrt(normal_factor)

        # res, error_xpol, error_ypol, wxx, wxy, wyy, wyx = \
        #     equalizer(ntaps=35, signal=signal[:], mu=np.array([0.0001/4, 0.0001/4]), training_sequence=signal.symbol, sps=2,
        #         nloop=3, training_time=3,constl=np.unique(signal.symbol[0]))
        #
        res, (wxx, wxy, wyx, wyy, error_xpol, error_ypol) = equalizer(signal, 2, 35, 0.0001 / 4, 3, 'lms',
                                                                      training_time=3, train_symbol=signal.symbol)

        symbol_after_equalizaton = np.zeros_like(signal.symbol)
        symbol_after_equalizaton[:, :len(res[0])] = res
        self.hxx = wxx
        self.hxy = wxy
        self.hyy = wyy
        self.hyx = wyx
        self.error_xpol = error_xpol
        self.error_ypol = error_ypol

        # sync
        # symbol = symbol_after_equalizaton
        symbol = sync_correlation(signal.symbol, symbol_after_equalizaton, 1)
        # cpe
        mask = symbol[:] != 0
        symbol = symbol[mask]
        symbol.shape = 2, -1
        train_symbol = signal.symbol
        train_symbol = train_symbol[mask]
        train_symbol.shape = 2, -1

        symbol_x, train_x = superscalar(symbol[0], train_symbol[0], 200, 8, np.unique(signal.symbol[0]), g=0.1 / 2)
        symbol_y, train_y = superscalar(symbol[1], train_symbol[1], 200, 8, np.unique(signal.symbol[0]), g=0.1 / 2)
        self.correlation = calc_auto_acf(np.array([np.atleast_2d(symbol_x)[0], np.atleast_2d(symbol_y)[0]]))[0][1:15]

        symbol_x = np.atleast_2d(symbol_x)
        # scatterplot(symbol_x)

        train_x = np.atleast_2d(train_x)
        symbol_y = np.atleast_2d(symbol_y)
        train_y = np.atleast_2d(train_y)
        symbol = np.array([symbol_x[0], symbol_y[0]])
        signal.symbol = np.array([train_x[0], train_y[0]])
        # calc snr
        noise = symbol[:, 1024:-1024] - signal.symbol[:, 1024:-1024]
        noise = np.mean(np.abs(noise) ** 2, axis=1, keepdims=True)
        xpol_snr = (1 - noise[0]) / noise[0]
        ypol_snr = (1 - noise[1]) / noise[1]
        xpol_snr_db = 10 * np.log10(xpol_snr)
        ypol_snr_db = 10 * np.log10(ypol_snr)
        self.xpol_snr_db = xpol_snr_db
        self.ypol_snr_db = ypol_snr_db


def processing_files(dirname, kind, save_name):
    import os
    names = os.listdir(dirname)

    feature = []
    import tqdm
    BaseDir = dirname + '/'

    for name in tqdm.tqdm(names):
        # name = names[13]
        receiver = Receiver(kind)
        anomaly_value = float(name.split('_')[-1])

        try:
            receiver.dsp_process(BaseDir + name)
            # receiver.dsp_process(name)
            signal_name = BaseDir + name
            #
            link_config_ith = int(signal_name.split('/')[-1].split('_')[0])
            # link_config_ith = 0
            with open(f'{save_name}', 'a+') as f:
                f.write(name)
                f.write(' ')
                f.write(str(link_config_ith))
                f.write(' span number: ')
                f.write(str(np.sum(receiver.span_setting)))
                f.write(':')
                f.write(str(receiver.xpol_snr_db))
                f.write('\n')
            if receiver.xpol_snr_db < 0:
                continue
        except Exception as e:
            print(e)
            continue
        ith = int(name.split('_')[2])
        receiver.hxx = np.atleast_2d(receiver.hxx)[0]
        hxx = np.abs(np.fft.fftshift(np.fft.fft(receiver.hxx)))
        receiver.spectrum = np.atleast_2d(receiver.spectrum)

        feature.append(np.hstack(
            [anomaly_value / 1e9, kind, receiver.spectrum[0], hxx, np.atleast_2d(receiver.correlation)[0],
             receiver.centroid, receiver.bandwidth, receiver.span_setting, ith]))

    header = ['anomaly_value', 'anomaly_kind']
    header.extend([f'spectrum{i}' for i in range(receiver.spectrum[0].shape[0])])
    header.extend([f'hxx_{i}' for i in range(hxx.shape[0])])
    header.extend(f'corr_{i}' for i in range(np.atleast_2d(receiver.correlation).shape[1]))
    header.append('centroid')
    header.append('bw')
    header.extend([f'span_setting{i}' for i in range(len(receiver.span_setting))])
    header.append('anomaly_location')
    import pandas as pd
    array = pd.DataFrame(np.array(feature), columns=header)
    array.to_csv(f'../5_{kind}_for_mlgeneralization.csv', columns=header, index=None)
    return array


if __name__ == '__main__':
    array1 = processing_files(r'i:/paper/data_generalization/5wss_FSNonlinear', 0,
                              save_name='../5wssfs_generalization.csv')
    array2 = processing_files(r'i:/paper/data_generalizaiton/5wss_FTNonlinear', 1,
                              save_name='../5wssft_generalization.csv')
    pass