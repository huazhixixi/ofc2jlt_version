import numpy as np
from library import QamSignal, Laser, WSS


def generate_signal(roll_off: float, signal_power: float) -> QamSignal:
    signal = QamSignal(16, 35e9, 2, 4, 65536, 2)
    signal.prepare(roll_off)
    laser = Laser(None, False, 193.1e12, signal_power)
    laser.prop(signal)
    return signal


def prop(signal: QamSignal, ins) -> QamSignal:
    visualize(ins)

    for i in ins:
        i.prop(signal)

    return signal


from typing import List
from library import NonlinearFiber,WSS
from library.optics import Edfa


def visualize(ins: List):
    for index, ele in enumerate(ins):
        if isinstance(ele, NonlinearFiber):
            if index == len(ins) - 1:
                print('Fiber', end='')
            else:
                print('Fiber', end='->')
        if isinstance(ele, Edfa):
            if index == len(ins) - 1:
                print('edfa', end='')
            else:
                print("EDFA", end='->')
        if isinstance(ele, WSS):
            if index == len(ins) - 1:
                print('WSS', '')
            else:
                print('wss', end='->')
    print('\n')


def generate_anomaly_wss(kind: str, failure_value: float, number: float, soft_failure_location: float) -> [WSS]:
    res = []
    for _ in range(number):
        if _ ==  soft_failure_location:
            if kind.lower() == 'fs':
                wss = WSS(failure_value, 50e9, 8.8e9)
            if kind.lower() == 'ft':
                wss = WSS(0, failure_value, 8.8e9)
        else:
            wss = WSS(0, 50e9, 8.8e9)
        res.append(wss)
    return res


class CoherentReceiver:

    def prop(self, signal, span, roll_off):
        from library import cd_compensation, matched_filter, LMS
        from library.receiver_dsp import syncsignal_tx2rx
        import resampy
        signal = cd_compensation(span, signal, signal.fs_in_fiber)
        signal.samples = resampy.resample(signal.samples, signal.sps_in_fiber, signal.sps)
        signal = matched_filter(signal, roll_off)

        angle = np.angle(np.mean(signal[:, ::2] / signal.symbol, axis=-1, keepdims=True))
        signal.samples = signal.samples * np.exp(-1j * angle)
        signal.inplace_normalise()
        lms = LMS(321, 3, 3, 0.001)
        signal = lms.equalize(signal)
        signal.symbol = syncsignal_tx2rx(signal.samples, signal.symbol)[:, :signal.shape[1]]

        noise = signal.samples - signal.symbol
        power = np.mean(np.abs(noise) ** 2, axis=-1)
        power = np.sum(power)
        snr = (2 - power) / power
        snr = 10 * np.log10(snr)

        print(snr)
        self.equalizer = lms
        return signal


def remind(file):
    from pygame import mixer
    import time

    mixer.init()
    mixer.music.load(file)
    mixer.music.play()
    time.sleep(5)
    mixer.music.stop()
