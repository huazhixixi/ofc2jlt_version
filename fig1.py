import numpy as np
from funtions import generate_anomaly_wss, generate_signal, prop, CoherentReceiver, remind

from library import NonlinearFiber, ConstantGainEdfa, QamSignal

# for ith in range(5):
#     spans = [NonlinearFiber(0.2,16.7,80,1550,0,'single') for _ in range(5)]
#     wsses = generate_anomaly_wss('ft',28e9,5,ith)
#     edfas = [ConstantGainEdfa(16,5,True) for _ in range(5)]
#
#
#     ins = []
#
#     for span,edfa,wss in zip(spans,edfas,wsses):
#         ins.append(span)
#         ins.append(edfa)
#         ins.append(wss)
#
#     signal = generate_signal(0.02,0)
#     signal = prop(signal,ins)
#     signal.save(f'data/ft_16_{ith}')
# remind('water.mp3')

res = []
for i in range(5):
    spans = [NonlinearFiber(0.2,16.7,80,1550,0,'single') for _ in range(5)]
    re = CoherentReceiver()
    signal = QamSignal.load(f'data/ft_16_{i}')
    #
    re.prop(signal,spans,0.02)

    res.append(re.equalizer.freq_response()[0][0])
remind('water.mp3')
import joblib
joblib.dump(res,'data/lms_equalizer3')