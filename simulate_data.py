from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from funtions import generate_signal, prop
from funtions import generate_anomaly_wss



import joblib
import numpy as np

from library import NonlinearFiber, ConstantGainEdfa

import time
def simulation(failure_kind,link_configuration,is_ase,save_dir):
    try:


        for link in link_configuration:
            total_number = link[0]
            if failure_kind.lower()=='fs':
                failure_value = (12*np.random.rand()+5)*1e9
            elif failure_kind.lower()=='ft':
                failure_value = (10 * np.random.rand() + 25) * 1e9
            else:
                raise ValueError
            soft_failure_location = np.random.randint(0,total_number)
            link = link[1:]
            spans = []
            wsses = generate_anomaly_wss(failure_kind,failure_value,total_number,soft_failure_location)
            for index,span_number in enumerate(link):
                temp = [NonlinearFiber(alpha=0.2, D=16.7, gamma=1.3, length=80, reference_wavelength=1550, slope=0,
                               accuracy='single'),ConstantGainEdfa(16,5,is_ase=is_ase)]*span_number


                spans.extend(temp)
                spans.append(wsses[index])

            signal = generate_signal(0.2,0)
            signal = prop(signal,spans)

            joblib.dump([link,signal,spans,wsses],save_dir+f'/{total_number}wss_{failure_kind}_{failure_value}_{soft_failure_location}')
    except Exception as e:
        print(e)

if __name__ == '__main__':
    import joblib
    dataconfig = joblib.load('dataconfigv1_5wss')
    process1 = dataconfig[:250]
    process2 = dataconfig[250:500]
    process3 = dataconfig[500:750]
    process4 = dataconfig[750:1000]

    res = []
    print('xixi')
    with ProcessPoolExecutor(4) as executor:

        res.append(executor.submit(simulation,'fs',process1,True,'../datanew/5wss/fs/p1/'))
        res.append(executor.submit(simulation,'fs',process2,True,'../datanew/5wss/fs/p2/'))

        res.append(executor.submit(simulation,'fs',process3,True,'../datanew/5wss/fs/p3/'))

        res.append(executor.submit(simulation,'fs',process4,True,'../datanew/5wss/fs/p4/'))

        wait(res)


    #simulation('ft',process1,True,'../datanew/5wss/fs/p1/')


