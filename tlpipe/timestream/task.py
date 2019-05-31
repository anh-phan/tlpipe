import numpy as np
import h5py
import timestream_task
from tlpipe.container.timestream import Timestream
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.utils.path_util import input_path


class Task(timestream_task.TimestreamTask):

    params_init = {
                    'task_param': 'param_val',
                  }

    prefix = 't_'

    def process(self, ts):

        print('isinstance(ts, RawTimestream): {}'.format(isinstance(ts, RawTimestream))) 
        print('isinstance(ts, Timestream): {}'.format(isinstance(ts, Timestream))) 

        print('ts.local_vis.shape: {}'.format(ts.local_vis.shape))

        ts.redistribute('baseline')

        print('ts[\'local_hour\']: {}'.format(ts['local_hour']))
        print('ts[\'local_hour\'].local_data: {}'.format(ts['local_hour'].local_data))

        print('ts.local_vis.shape: {}'.format(ts.local_vis.shape))
        print('ts.local_vis_mask.shape: {}'.format(ts.local_vis_mask.shape))

        print(ts.local_vis[1800,255,0,0])
        print(ts.local_vis[1800,255,1,0])
        print(ts.local_vis[1800,255,2,0])
        print(ts.local_vis[1800,255,3,0])

        print('ts[\'blorder\'].local_data: {}'.format(ts['blorder'].local_data))

        #print('ts.freq[:]: {}'.format(ts.freq[:]))
        # nf = len(freq)

        # # shold check freq, pol and feed here, omit it now...

        # for fi in range(nf):
        #     for pi in [pol.index('xx'), pol.index('yy')]:
        #         pi_ = gain_pd[pol[pi]]
        #         for bi, (fd1, fd2) in enumerate(ts['blorder'].local_data):
        #             g1 = gain[fi, pi_, feedno.index(fd1)]
        #             g2 = gain[fi, pi_, feedno.index(fd2)]
        #             if np.isfinite(g1) and np.isfinite(g2):
        #                 ts.local_vis[:, fi, pi, bi] /= (g1 * np.conj(g2))
        #             else:
        #                 # mask the un-calibrated vis
        #                 ts.local_vis_mask[:, fi, pi, bi] = True

        return super(Task, self).process(ts)