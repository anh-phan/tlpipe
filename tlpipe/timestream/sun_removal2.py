import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import h5py
import timestream_task
# from tlpipe.container.timestream import Timestream
from tlpipe.container.raw_timestream import RawTimestream
from scipy.interpolate import interp1d, Rbf
from numpy import linalg as LA
import scipy.signal as signal
import numpy.ma as ma
import logging

# Set the module logger.
logger = logging.getLogger(__name__)


class SunRemoval2(timestream_task.TimestreamTask):

    params_init = {
        'filter_order': 2.0,  # 'none', linear', 'nearest' or 'rbf'
        'cutoff_freq' : 0.01,
        'filter_threshold': 60,
        'bad_feeds': []
    }

    prefix = 'sr_'

    def process(self, rt):

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        filter_order = self.params['filter_order']
        cutoff_freq = self.params['cutoff_freq']
        thres = self.params['filter_threshold']
        bad_feeds = self.params['bad_feeds']
        
        #########################################
        bl_length = len(rt['blorder'].local_data)
        bl = np.empty([bl_length,3],'int')
        
        # first col is the index (0-527), 2nd and 3rd are the baselines
        for i in xrange(bl_length):
            bl[i] = [i,rt['blorder'].local_data[i][0],rt['blorder'].local_data[i][1]]
        
        sort_index = np.lexsort((bl[:,2], bl[:,1]))  # sort according to 1st and 2nd columns
        bl = bl[sort_index]

        #########################################

        num_bl = len(bl)
        num_feed = len(np.unique(bl[:, 1]))
        print('bad_feeds: {}'.format(bad_feeds))

        eig_values = np.zeros((rt.local_vis.shape[0], rt.local_vis.shape[1], num_feed), dtype=complex)
        eig_vectors = np.zeros((rt.local_vis.shape[0], rt.local_vis.shape[1], num_feed, num_feed), dtype=complex) 

        eig_values_ma = ma.zeros(eig_values.shape,dtype=complex)

        logger.info("Task %s: Processing eigenvalues and eigenvectors..." % self.__class__.__name__)

        time_mask = np.zeros((rt.local_vis.shape[0], rt.local_vis.shape[1]), dtype = bool)
        for fi in xrange(rt.local_vis.shape[1]):
            for ti in xrange(rt.local_vis.shape[0]):
                if np.all(rt.local_vis_mask[ti, fi, :]):
                    time_mask[ti, fi] = True
                    continue 
                vis_matrix = np.zeros((num_feed, num_feed), dtype=complex)
                for b in xrange(num_bl):
                    r = bl[b][1]
                    c = bl[b][2]
                    if r in bad_feeds or c in bad_feeds:
                        continue
                    else:
                        vis_value = rt.local_vis[ti, fi, bl[b][0]]
                        if r == c:  # autocorrelation
                            # initialize the autocorrelation to 0.0
                            vis_matrix[int(r)-1, int(c)-1] = 0.0
                        else:
                            vis_matrix[int(r)-1, int(c)-1] = vis_value
                            vis_matrix[int(c)-1, int(r)-1] = np.conjugate(vis_value)

                for bf in bad_feeds:
                    vis_matrix[bf-1,bf-1] = 1
        
                # Now fill in the correct value for the autocorrelation
                for i in xrange(0, num_feed):
                    if (i+1) in bad_feeds:
                        continue
                    N = 0
                    N_cross = 0
                    for j in xrange(0, num_feed):
                        if (j+1) in bad_feeds:
                            continue
                        for k in xrange(0, num_feed):
                            if (k+1) in bad_feeds:
                                continue
                            if ((i != j) and (i != k) and (j != k)):
                                if(np.abs(vis_matrix[j, k]) == 0):
                                    vis_matrix[i, i] += 0.0001
                                    N += 1
                                else:
                                    try:
                                        vis_matrix[i,i] += vis_matrix[i,j]*vis_matrix[i,k]/vis_matrix[j,k]
                                        N += 1
                                        if i%2 == 1 and (j != (i-1)) and (k != (i-1)) and np.abs(vis_matrix[j,k])>0.0001:
                                            vis_matrix[i-1,i] += vis_matrix[i-1,k]*vis_matrix[j,i]/vis_matrix[j,k]
                                            vis_matrix[i,i-1] += np.conjugate(vis_matrix[i-1,k]*vis_matrix[j,i]/vis_matrix[j,k])
                                            N_cross += 1
                                    except ZeroDivisionError:
                                        vis_matrix[i, i] += 0.0001
                                        #sN += 1
                                

                    vis_matrix[i, i] = np.abs(vis_matrix[i, i])/N
                    if i%2 == 1:
                        vis_matrix[i-1,i] = vis_matrix[i-1,i]/N_cross
                        vis_matrix[i,i-1] = vis_matrix[i,i-1]/N_cross

                for bf in bad_feeds:
                    vis_matrix[bf-1,bf-1] = 1

                eig_values[ti, fi, :], eig_vectors[ti, fi, :] = LA.eigh(vis_matrix)

            # Masking
            for n in xrange(0,num_feed):
                eig_values_ma[:, fi, n] = ma.masked_array(eig_values[:, fi, n], mask=time_mask, dtype=complex)

        del eig_values
        eig_values = eig_values_ma
        eig_values.dump('eigenvalues_all')
        eig_vectors.dump('eigenvectors_all')

        logger.info("Task %s: Filtering largest eigenvalue." % self.__class__.__name__)

        ########### Filtering ###########################
        # for fi in xrange(rt.local_vis.shape[1]):
        #     max_e = eig_values[:,fi,-1]  # last (largest eigenvalues)
        #     B, A = signal.butter(filter_order, cutoff_freq, output='ba')
        #     # Apply the filter
        #     max_f = signal.filtfilt(B,A, max_e)
        #     residuals = max_e - max_f

        #     var = np.var(residuals)
        #     med = np.median(max_f)

        #     for ti in xrange(rt.local_vis.shape[0]):
        #         if max_e[ti] > med + thres*var:
        #             eig_values[ti,fi,-1] = 0
        #################################################

        logger.info("Task %s: Reconstructing the visibilities." % self.__class__.__name__)

        for fi in xrange(rt.local_vis.shape[1]):
            for ti in xrange(rt.local_vis.shape[0]):
                if np.all(rt.local_vis_mask[ti, fi, :]):
                    continue 
                vis_matrix = np.matmul(eig_vectors[ti, fi, :],np.matmul(np.diag(eig_values[ti, fi, :]),eig_vectors[ti, fi, :].conj().T))
                # flatten
                for b in bl:
                    row = b[1] - 1
                    col = b[2] - 1
                    rt.local_vis[ti, fi, b[0]] = vis_matrix[row, col]
        
        del eig_values
        del eig_vectors

        return super(SunRemoval2, self).process(rt)


