from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from funtions import generate_signal, prop
from funtions import generate_anomaly_wss
from library.channel import LinearFiber



import joblib
import numpy as np

from library import NonlinearFiber, ConstantGainEdfa

import time
def simulation(failure_kind,link_configuration,is_ase,save_dir,ith_link_base):
    try:


        for iththth,link in enumerate(link_configuration):
            total_number = link[0]
            if failure_kind.lower()=='fs':
                failure_value = (5*np.random.rand()+12)*1e9
            elif failure_kind.lower()=='ft':
                failure_value = (10 * np.random.rand() + 25) * 1e9
            else:
                raise ValueError
            # soft_failure_location = np.random.randint(0,total_number)
            link = link[1:]
            for soft_failure_location in range(total_number):


                    spans = []
                    wsses = generate_anomaly_wss(failure_kind,failure_value,total_number,soft_failure_location)
                    for index,span_number in enumerate(link):
                        temp = [LinearFiber(alpha=0.2, D=16.7, length=80, reference_wavelength=1550, slope=0),ConstantGainEdfa(16,5,is_ase=is_ase)]*span_number


                        spans.extend(temp)
                        spans.append(wsses[index])

                    signal = generate_signal(0.2,0)
                    signal = prop(signal,spans)
                    signal.save(save_dir+f'/{ith_link_base+iththth}_{total_number}wss_{failure_kind}_{failure_value}_{soft_failure_location}')

    except Exception as e:
        print(e)

if __name__ == '__main__':
    import joblib
    dataconfig = joblib.load('dataconfigv1_8wss')
    process1 = dataconfig[:40]
    process2 = dataconfig[40:80]
    process3 = dataconfig[80:120]
    process4 = dataconfig[120:160]
    process5 = dataconfig[160:200]

    res = []
    failure_kind = 'fs'
    # simulation(failure_kind, process1, True, '../datanew/5wss/fs/p1/', 0)

    with ProcessPoolExecutor(5) as executor:

        res.append(executor.submit(simulation,failure_kind,process1,True,'../datatest_guding/8wss/fs/',0))
        res.append(executor.submit(simulation,failure_kind,process2,True,'../datatest_guding/8wss/fs/',40))

        res.append(executor.submit(simulation,failure_kind,process3,True,'../datatest_guding/8wss/fs/',80))

        res.append(executor.submit(simulation,failure_kind,process4,True,'../datatest_guding/8wss/fs/',120))
        res.append(executor.submit(simulation,failure_kind,process5,True,'../datatest_guding/8wss/fs/',160))

        wait(res)




