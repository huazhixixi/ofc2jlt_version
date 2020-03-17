import numpy as np

from library import QamSignal
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#print(sys.path)
from scipy.integrate import trapz
from scipy.signal import welch
from scipy.signal import correlate

from library.receiver_dsp  import cd_compensation
from library.receiver_dsp import LMS
from library.receiver_dsp import Superscalar
from library.channel import NonlinearFiber
from resampy import resample
from library.receiver_dsp import matched_filter
from library.receiver_dsp import syncsignal_tx2rx

import joblib

def calc_auto_acf(cpr):
    # cprx = cpr[1, cpr[1, :] != 0]
    # cpry = cpr[0, cpr[0, :] != 0]
    # cpr = np.array([cprx, cpry])
    xpol_acf = np.abs(correlate(cpr[0, :], cpr[0, :]))
    xpol_acf = xpol_acf[len(cpr[0, :]) - 1:] / len(cpr[0, :])

    ypol_acf = np.abs(correlate(cpr[1, :], cpr[1, :]))
    ypol_acf = ypol_acf[len(cpr[1, :]) - 1:] / len(cpr[1, :])
    return xpol_acf / np.max(xpol_acf), ypol_acf / np.max(ypol_acf)


class Receiver(object):

    def __init__(self):
        self.equalizer = None
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
        signal = QamSignal.load(signal_name)
        link_config_ith = int(signal_name.split('/')[-1].split('_')[0])
        config = joblib.load('./dataconfigv1_5wss')

        span_setting = config[link_config_ith][1:]
        self.span_setting = span_setting
        fiber_for_cdc = NonlinearFiber(alpha=0.2, D=16.7, gamma=1.3, length=np.sum(self.span_setting) * 80,
                                       step_length=20 / 1000,slope=0,accuracy='single',reference_wavelength=1550)
        signal = cd_compensation(fiber_for_cdc,signal,signal.fs_in_fiber)
        # resample
        import resampy
        signal.samples = resampy.resample(signal.samples,signal.sps_in_fiber,signal.sps)
        # matched filter
        signal = matched_filter(signal,0.2)
        import matplotlib.pyplot as plt
        [freqs, pxx] = welch(signal[0, :], fs=signal.fs, detrend=False, nfft=1024,
                             return_onesided=False)
        self.centroid = self.calc_centroid(freqs, pxx)
        self.spectrum = np.fft.fftshift(pxx)
        self.bandwidth = self.find_bandwidth(freqs, pxx)
        # rotate phase
        phase = np.angle(np.mean(signal.samples[:,::signal.sps]/signal.symbol,axis=-1,keepdims=True))
        signal[:] = signal[:] * np.exp(-1j*phase)
        # LMS
        signal.inplace_normalise()
        
        self.equalizer = LMS(ntaps=35,loops=3,train_iter=3,lr_train=0.001/4)
        
        signal = self.equalizer.equalize(signal)
        
        signal.symbol = syncsignal_tx2rx(signal.samples,signal.symbol)
        signal.symbol = signal.symbol[:,:signal.samples.shape[1]]
        # sync
        # symbol = symbol_after_equalizaton
        # cpe
        cpe = Superscalar(200,0.1/2,20,0,8)
        signal = cpe.prop(signal)
        # symbol_x, train_x = superscalar(symbol[0], train_symbol[0], 200, 8, np.unique(signal.symbol[0]), g=0.1 / 2)
        # symbol_y, train_y = superscalar(symbol[1], train_symbol[1], 200, 8, np.unique(signal.symbol[0]), g=0.1 / 2)
        self.correlation = calc_auto_acf(signal[:])[0][1:15]
        noise = signal.samples[:, 1024:-1024] - signal.symbol[:, 1024:-1024]
        noise = np.mean(np.abs(noise) ** 2, axis=1, keepdims=True)
        xpol_snr = (1 - noise[0]) / noise[0]
        ypol_snr = (1 - noise[1]) / noise[1]
        xpol_snr_db = 10 * np.log10(xpol_snr)
        ypol_snr_db = 10 * np.log10(ypol_snr)
        self.xpol_snr_db = xpol_snr_db
        self.ypol_snr_db = ypol_snr_db


def processing_files(dirname,names, kind, log_name,savedir):
    import os

    feature = []
    import tqdm
    BaseDir = dirname

    for name in tqdm.tqdm(names):
        # name = names[13]
        receiver = Receiver()
        anomaly_value = float(name.split('_')[-2])

        try:
            receiver.dsp_process(BaseDir + name)
            # receiver.dsp_process(name)
            signal_name = BaseDir + name
            #
            link_config_ith = int(signal_name.split('/')[-1].split('_')[0])
            # link_config_ith = 0
            with open(f'{log_name}', 'a+') as f:
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
        ith = int(name.split('_')[-1])
        receiver.hxx = np.atleast_2d(receiver.equalizer.wxx)[0]
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
    array.to_csv(savedir, columns=header, index=None)
    return array


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor,wait

    basedir = '../datatest_guding/5wss/ft/'
    name = os.listdir(basedir)
    name_0 = name[:250]
    name_1 = name[250:500]
    name_2 = name[500:750]
    name_3 = name[750:1000]
    res = []
    log_name = 'log'
    csv_save_dir = '../datatest_guding/5wss/'
    with ProcessPoolExecutor(4) as executor:
        res.append(executor.submit(processing_files, basedir,name_0, 'ft', 'log_process580', csv_save_dir+'5wssft_processe80.csv'))
        res.append(executor.submit(processing_files, basedir,name_1, 'ft', 'log_process581', csv_save_dir+'5wssft_processe81.csv'))

        res.append(executor.submit(processing_files, basedir,name_2, 'ft', 'log_process582', csv_save_dir+'5wssft_processe82.csv'))

        res.append(executor.submit(processing_files, basedir,name_3, 'ft', 'log_process583', csv_save_dir+'5wssft_processe83.csv'))

        wait(res)

    # re.dsp_process('1_5wss_fs_10739319156.76123_3')
    # array1 = processing_files(r'i:/paper/data_generalization/5wss_FSNonlinear', 0,
    #                           save_name='../5wssfs_generalization.csv')
    # array2 = processing_files(r'i:/paper/data_generalizaiton/5wss_FTNonlinear', 1,
    #                           save_name='../5wssft_generalization.csv')
    # pass