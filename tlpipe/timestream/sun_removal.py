import numpy as np
import h5py
import timestream_task
# from tlpipe.container.timestream import Timestream
from tlpipe.container.raw_timestream import RawTimestream
from scipy.interpolate import interp1d, Rbf
from numpy import linalg as LA
import scipy.signal as signal


class SunRemoval(timestream_task.TimestreamTask):

    params_init = {
        'filter_order': 2.0,  # 'none', linear', 'nearest' or 'rbf'
        'cutoff_freq' : 0.01,
        'filter_threshold': 60
    }

    prefix = 'sr_'

    def process(self, rt):

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        filter_order = self.params['filter_order']
        cutoff_freq = self.params['cutoff_freq']
        thres = self.params['filter_threshold']
        
        #########################################
        bl_length = len(rt['blorder'].local_data)
	    # create an empty array of size bl_length rows x 3 cols
        bl = np.empty([bl_length,3],'int')
        
        # populate the array: first col is the index (0-527), 2nd and 3rd are the baselines
        for i in xrange(bl_length):
            bl[i] = [i,rt['blorder'].local_data[i][0],rt['blorder'].local_data[i][1]]
	    sort_index = np.lexsort((bl[:,2], bl[:,1]))  # sort according to 1st and 2nd columns
	    bl = bl[sort_index]
        #########################################
        
        num_bl = len(bl)
        num_feed = len(np.unique(bl[:, 1]))

        eig_values_x = np.zeros((rt.local_vis.shape[0], rt.local_vis.shape[1], num_feed/2), dtype=complex)
        eig_values_y = np.zeros((rt.local_vis.shape[0], rt.local_vis.shape[1], num_feed/2), dtype=complex) 
        eig_vectors_x = np.zeros((rt.local_vis.shape[0], rt.local_vis.shape[1], num_feed/2, num_feed/2), dtype=complex) 
        eig_vectors_y = np.zeros((rt.local_vis.shape[0], rt.local_vis.shape[1], num_feed/2, num_feed/2), dtype=complex) 

        for fi in xrange(rt.local_vis.shape[1]):
            for ti in xrange(rt.local_vis.shape[0]):
                if np.any(rt.local_vis_mask[ti, fi, :]):
                    continue 
                vis_matrix_x = np.zeros((num_feed/2, num_feed/2), dtype=complex)  # 16 by 16 matrix
                vis_matrix_y = np.zeros((num_feed/2, num_feed/2), dtype=complex)  # 16 by 16 matrix

                for b in xrange(num_bl):
                    r = bl[b][1]
                    c = bl[b][2]
                    if r % 2 == 1 and c % 2 == 1:
                        vis_value = rt.local_vis[ti, fi, bl[b][0]]
                        if r == c:  # autocorrelation
                            # initialize the autocorrelation to 0.0
                            vis_matrix_x[int(r/2), int(c/2)] = 0.0
                        else:
                            vis_matrix_x[int(r/2), int(c/2)] = vis_value
                            vis_matrix_x[int(c/2), int(r/2)] = np.conjugate(vis_value)
                    if r % 2 == 0 and c % 2 == 0:
                        vis_value = rt.local_vis[ti, fi,bl[b][0]]
                        if r == c:  #autocorrelation
                            vis_matrix_y[int(r/2)-1, int(c/2)-1] = 0.0 # initialize the autocorrelation to 0.0
                        else:
                            vis_matrix_y[int(r/2)-1, int(c/2)-1] = vis_value
                            vis_matrix_y[int(c/2)-1, int(r/2)-1] = np.conjugate(vis_value)
        
                # Now fill in the correct value for the autocorrelation
                for i in xrange(0, num_feed/2):
                    N = 0
                    for j in xrange(0, num_feed/2):
                        for k in xrange(0, num_feed/2):
                            if ((i != j) and (i != k) and (j != k)):
                                if(np.abs(vis_matrix_x[j, k]) == 0):
                                    vis_matrix_x[i, i] += 0.0001
                                else:
                                    try:
                                        vis_matrix_x[i,i] += vis_matrix_x[i,j]*vis_matrix_x[i,k]/vis_matrix_x[j,k]
                                    except ZeroDivisionError:
                                        vis_matrix_x[i, i] += 0.0001
                                if(np.abs(vis_matrix_y[j, k]) == 0):
                                    vis_matrix_y[i, i] += 0.0001
                                else:
                                    try:
                                        vis_matrix_y[i,i] += vis_matrix_y[i,j]*vis_matrix_y[i,k]/vis_matrix_y[j,k]
                                    except ZeroDivisionError:
                                        vis_matrix_y[i, i] += 0.0001
                                N += 1

                    vis_matrix_x[i, i] = np.abs(vis_matrix_x[i, i])/N
                    vis_matrix_y[i, i] = np.abs(vis_matrix_y[i, i])/N

                eig_values_x[ti, fi, :], eig_vectors_x[ti, fi, :] = LA.eigh(vis_matrix_x)
                eig_values_y[ti, fi, :], eig_vectors_y[ti, fi, :] = LA.eigh(vis_matrix_y)

        ########### Filtering #############
        for fi in xrange(rt.local_vis.shape[1]):
            max_x = eig_values_x[:,fi,-1]  # last (largest eigenvalues)
            max_y = eig_values_y[:,fi,-1]  
            B, A = signal.butter(filter_order, cutoff_freq, output='ba')
            # Apply the filter
            max_xf = signal.filtfilt(B,A, max_x)
            max_yf = signal.filtfilt(B,A, max_y)
            residuals_x = max_x - max_xf
            residuals_y = max_y - max_yf

            var_x = np.var(residuals_x)
            med_x = np.median(max_xf)
            var_y = np.var(residuals_y)
            med_y = np.median(max_yf)

            for ti in xrange(rt.local_vis.shape[0]):
                if max_x[ti] > med_x + thres*var_x:
                    eig_values_x[i,fi,-1] = 0
                if max_y[ti] > med_y + thres*var_y:
                    eig_values_y[i,fi,-1] = 0
        ###################################

        for fi in xrange(rt.local_vis.shape[1]):
            for ti in xrange(rt.local_vis.shape[0]):
                if np.any(rt.local_vis_mask[ti, fi, :]):
                    continue 
                vis_matrix_x = np.matmul(eig_vectors_x[ti, fi, :],np.matmul(np.diag(eig_values_x[ti, fi, :]),eig_vectors_x[ti, fi, :].conj().T))
                vis_matrix_y = np.matmul(eig_vectors_y[ti, fi, :],np.matmul(np.diag(eig_values_y[ti, fi, :]),eig_vectors_y[ti, fi, :].conj().T))
                vis_matrix = np.zeros((num_feed,num_feed),dtype=complex)  # 32 by 32 matrix
                vis_matrix = vis_matrix * np.nan
                for r in range(num_feed):
                    for c in range(num_feed):
                        if r <= c and r % 2 == 0 and c % 2 == 0:
                            vis_matrix[r,c] = vis_matrix_x[int(r/2), int(c/2)]
                        elif r <= c and r % 2 == 1 and c % 2 == 1:
                            vis_matrix[r,c] = vis_matrix_y[int(r/2), int(c/2)]
                # flatten
                for b in bl:
                    row = b[1] - 1
                    col = b[2] - 1
                    if not np.isnan(vis_matrix[row, col]):
                        rt.local_vis[ti, fi, b[0]] = vis_matrix[row, col]
        
        del eig_values_x
        del eig_values_y
        del eig_vectors_x
        del eig_vectors_y
        del vis_matrix_x
        del vis_matrix_y
        del vis_matrix

        return super(SunRemoval, self).process(rt)


