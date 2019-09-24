import numpy as np
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


class SunRemoval(timestream_task.TimestreamTask):

    params_init = {
        'filter_order': 2.0,  # 'none', linear', 'nearest' or 'rbf'
        'cutoff_freq' : 0.01,
        'filter_threshold': 60
    }

    prefix = 'sr_'

    def process(self, rt):

        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        filter_order = self.params['filter_order']
        cutoff_freq = self.params['cutoff_freq']
        thres = self.params['filter_threshold']
        
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

        eig_values_x = np.zeros((rt.local_vis.shape[0], rt.local_vis.shape[1], num_feed/2), dtype=complex)
        eig_values_y = np.zeros((rt.local_vis.shape[0], rt.local_vis.shape[1], num_feed/2), dtype=complex) 
        eig_vectors_x = np.zeros((rt.local_vis.shape[0], rt.local_vis.shape[1], num_feed/2, num_feed/2), dtype=complex) 
        eig_vectors_y = np.zeros((rt.local_vis.shape[0], rt.local_vis.shape[1], num_feed/2, num_feed/2), dtype=complex) 

        eig_values_xma = ma.zeros(eig_values_x.shape,dtype=complex)
        eig_values_yma = ma.zeros(eig_values_y.shape,dtype=complex)

        logger.info("Task %s: Processing eigenvalues and eigenvectors..." % self.__class__.__name__)

        time_mask = np.zeros((rt.local_vis.shape[0], rt.local_vis.shape[1]), dtype = bool)
        for fi in xrange(rt.local_vis.shape[1]):
            for ti in xrange(rt.local_vis.shape[0]):
                if np.all(rt.local_vis_mask[ti, fi, :]):
                    time_mask[ti, fi] = True
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

            # Masking
            for n in xrange(0,num_feed/2):
                eig_values_xma[:, fi, n] = ma.masked_array(eig_values_x[:, fi, n], mask=time_mask)
                eig_values_yma[:, fi, n] = ma.masked_array(eig_values_y[:, fi, n], mask=time_mask)

        del eig_values_x
        del eig_values_y
        eig_values_x = eig_values_xma
        eig_values_y = eig_values_yma

        logger.info("Task %s: Filtering largest eigenvalue." % self.__class__.__name__)

        tangent = []
        for ti in xrange(eig_vectors_x.shape[0]):
            l = []
            for i in xrange(eig_vectors_x.shape[2]-1):
                temp = 0
                for j in xrange(i+1,eig_vectors_x.shape[2]):
                    temp += np.power(np.abs(eig_vectors_x[ti,0,j,-1]),2)
                if temp < 1e-15:
                    t = np.nan
                else:
                    t = np.abs(eig_vectors_x[ti,0,j,-1])/np.sqrt(temp)
                l.append(t)
                del t, temp
            tangent.append(l)
            del l
        tangent = np.array(tangent,dtype=float)

        tangent_filled = []
        for i in xrange(tangent.shape[1]):
            temp = np.copy(tangent[:,i])
            nans, x = nan_helper(temp)
            temp[nans] = np.interp(x(nans), x(~nans), temp[~nans])
            tangent_filled.append(temp)
            del temp
        tangent_filled = np.array(tangent_filled,dtype=float).T

        del tangent
        tangent = tangent_filled
        del tangent_filled

        ### Filtering ###
        tangent_filtered = []
        for i in xrange(tangent.shape[1]):
            B, A = signal.butter(filter_order, cutoff_freq, output='ba')
            temp = signal.filtfilt(B,A, tangent)
            tangent_filtered.append(temp)
            del temp
        tangent_filtered = np.array(tangent_filtered,dtype=float).T

        del tangent
        tangent = tangent_filtered
        del tangent_filtered

        eig_vector_abs_x = []
        lim = eig_values_x.shape[2]
        for ti in xrange(eig_vectors_x.shape[0]):
            x = np.zeros((lim,),dtype=float)
            for i in xrange(lim):
                if i != lim - 1:
                    x[i] = np.sin(np.arctan(tangent[ti,i]))
                else:
                    x[i] = 1.0
                for j in xrange(i):
                    x[i] = x[i]*np.cos(np.arctan(tangent[ti,j]))
            eig_vector_abs_x.append(x)
        del x
        eig_vector_abs_x = np.array(eig_vector_abs_x,dtype=float)  # this is the np.abs of the largest eigenvector

        eig_vector_max_x = []
        for i in xrange(lim):
            eig_vector_max_x.append(np.multiply(eig_vector_abs_x[:,i],np.cos(np.angle(eig_vectors_x[:,0,i,-1]))) 
                                + 1j* np.multiply(eig_vector_abs_x[:,i],np.sin(np.angle(eig_vectors_x[:,0,i,-1]))))
        eig_vector_max_x = np.array(eig_vector_abs_x,dtype=complex).T
        
        del eig_vector_abs_x, lim

        vis_sun_x = []
        for ti in xrange(eig_vectors_x.shape[0]):
            temp = eig_values_x[ti,0,-1]*np.outer(eig_vectors_x[ti, 0, :, -1].conj().T,eig_vectors_x[ti, 0, :, -1])
            vis_sun_x.append(temp)
            del temp
        vis_sun_x = np.array(vis_sun_x,dtype=complex)
        vis_sun_x = vis_sun_x.conj()

        logger.info("Task %s: Reconstructing the visibilities." % self.__class__.__name__)

        for fi in xrange(rt.local_vis.shape[1]):
            for ti in xrange(rt.local_vis.shape[0]):
                if np.all(rt.local_vis_mask[ti, fi, :]):
                    continue 
                vis_matrix_x = np.matmul(eig_vectors_x[ti, fi, :],np.matmul(np.diag(eig_values_x[ti, fi, :]),eig_vectors_x[ti, fi, :].conj().T))
                vis_matrix_x = vis_matrix_x - vis_sun_x[ti]
                vis_matrix_y = np.matmul(eig_vectors_y[ti, fi, :],np.matmul(np.diag(eig_values_y[ti, fi, :]),eig_vectors_y[ti, fi, :].conj().T))
                vis_matrix = np.zeros((num_feed,num_feed),dtype=complex)
                vis_toggle = np.zeros((num_feed,num_feed),dtype=bool)
                for r in range(num_feed):
                    for c in range(num_feed):
                        if r <= c and r % 2 == 0 and c % 2 == 0:
                            vis_matrix[r,c] = vis_matrix_x[int(r/2), int(c/2)]
                            vis_toggle[r,c] = True
                        elif r <= c and r % 2 == 1 and c % 2 == 1:
                            vis_matrix[r,c] = vis_matrix_y[int(r/2), int(c/2)]
                            vis_toggle[r,c] = True
                # flatten
                for b in bl:
                    row = b[1] - 1
                    col = b[2] - 1
                    if vis_toggle[row, col]:
                        rt.local_vis[ti, fi, b[0]] = vis_matrix[row, col]
        
        del eig_values_x
        del eig_values_y
        del eig_vectors_x
        del eig_vectors_y

        return super(SunRemoval, self).process(rt)


