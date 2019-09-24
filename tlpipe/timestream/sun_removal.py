import numpy as np
import h5py
import ephem
import aipy as a
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
        'filter_threshold': 60,
        'span' : 600, # minutes
        'remove_mean' : False
    }

    prefix = 'sr_'

    def process(self, rt):

        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        filter_order = self.params['filter_order']
        cutoff_freq = self.params['cutoff_freq']
        thres = self.params['filter_threshold']
        span = self.params['span']
        remove_mean = self.params['remove_mean']

        #########################################
        bl_length = len(rt['blorder'].local_data)
        bl = np.empty([bl_length,3],'int')
        
        # first col is the index (0-527), 2nd and 3rd are the baselines
        for i in xrange(bl_length):
            bl[i] = [i,rt['blorder'].local_data[i][0],rt['blorder'].local_data[i][1]]
        
        sort_index = np.lexsort((bl[:,2], bl[:,1]))  # sort according to 1st and 2nd columns
        bl = bl[sort_index]
        #########################################

        nt = rt.local_vis.shape[0] # number of time points of this process

        if nt > 0:

            srclist, cutoff, catalogs = a.scripting.parse_srcs('Sun', 'misc')
            cat = a.src.get_catalog(srclist, cutoff, catalogs)
            s = cat.values()[0] # the Sun

            # get transit time of calibrator
            # array
            aa = rt.array
            local_juldate = rt['jul_date'].local_data
            aa.set_jultime(local_juldate[0]) # the first obs time point of this process

            mask_inds = []

            # previous transit
            prev_transit = aa.previous_transit(s)
            prev_transit_start = a.phs.ephem2juldate(prev_transit - 0.5 * span * ephem.minute) # Julian date
            prev_transit_end = a.phs.ephem2juldate(prev_transit + 0.5 * span * ephem.minute) # Julian date
            prev_transit_start_ind = np.searchsorted(local_juldate, prev_transit_start, side='left')
            prev_transit_end_ind = np.searchsorted(local_juldate, prev_transit_end, side='right')
            if prev_transit_end_ind > 0:
                mask_inds.append((prev_transit_start_ind, prev_transit_end_ind))

            # next transit
            next_transit = aa.next_transit(s)
            next_transit_start = a.phs.ephem2juldate(next_transit - 0.5 * span * ephem.minute) # Julian date
            next_transit_end = a.phs.ephem2juldate(next_transit + 0.5 * span * ephem.minute) # Julian date
            next_transit_start_ind = np.searchsorted(local_juldate, next_transit_start, side='left')
            next_transit_end_ind = np.searchsorted(local_juldate, next_transit_end, side='right')
            if next_transit_start_ind < nt:
                mask_inds.append((next_transit_start_ind, next_transit_end_ind))

            # then all next transit if data is long enough
            while (next_transit_end_ind < nt):
                aa.set_jultime(next_transit_end)
                next_transit = aa.next_transit(s)
                next_transit_start = a.phs.ephem2juldate(next_transit - 0.5 * span * ephem.minute) # Julian date
                next_transit_end = a.phs.ephem2juldate(next_transit + 0.5 * span * ephem.minute) # Julian date
                next_transit_start_ind = np.searchsorted(local_juldate, next_transit_start, side='left')
                next_transit_end_ind = np.searchsorted(local_juldate, next_transit_end, side='right')
                if next_transit_start_ind < nt:
                    mask_inds.append((next_transit_start_ind, next_transit_end_ind))

        rt.create_dataset('sun_removal',data=np.array(mask_inds))
        rt['sun_removal'].attrs['span (minutes)'] = float(span)
        rt['sun_removal'].attrs['dimname'] = 'begin, end sun_removal'
        rt['sun_removal'].attrs['unit'] = 'index in file'

        logger.info("[(Sunrise,Sunset)]: %s" %str(mask_inds))

        if remove_mean:
            logger.info("Task %s: Removing the mean..." % self.__class__.__name__)
            for si, ei in mask_inds:
                #print('si: {}, ei: {}'.format(si,ei))
                for fi in xrange(rt.local_vis.shape[1]):
                    vis = ma.array(rt.local_vis[:,fi,:], mask=rt.local_vis_mask[:,fi,:])
                    me1 = ma.mean(ma.array(rt.local_vis[:si,fi,:], mask=rt.local_vis_mask[:si,fi,:]),axis=0)
                    me2 = ma.mean(ma.array(rt.local_vis[ei:,fi,:], mask=rt.local_vis_mask[ei:,fi,:]),axis=0)
                    med = ma.mean([me1, me2],axis=0)
                    del me1, me2
                    vis = vis - med
                    for ti in xrange(vis.shape[0]):
                        if np.all(rt.local_vis_mask[ti, fi, :]):
                            continue
                        else:
                            rt.local_vis[ti,fi,:] = vis[ti,:]
                    del med, vis

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
                for i in xrange(0,num_feed/2):
                    if eig_vectors_x[ti, fi, 0, i] < 0:
                        eig_vectors_x[ti, fi, :, i] = -1.0 * eig_vectors_x[ti, fi, :, i]
                    if eig_vectors_y[ti, fi, 0, i] < 0:
                        eig_vectors_y[ti, fi, :, i] = -1.0 * eig_vectors_y[ti, fi, :, i]

            # Masking
            for n in xrange(0,num_feed/2):
                eig_values_xma[:, fi, n] = ma.masked_array(eig_values_x[:, fi, n], mask=time_mask)
                eig_values_yma[:, fi, n] = ma.masked_array(eig_values_y[:, fi, n], mask=time_mask)

        del eig_values_x
        del eig_values_y
        eig_values_x = eig_values_xma
        eig_values_y = eig_values_yma

        #freq = rt.attrs['history'][2653:2656].split(',')[0]
        #eig_values_x[:,0,-1].dump('eva_x_' + freq)
        #eig_values_y[:,0,-1].dump('eva_y_' + freq)
        #eig_vectors_x[:,0,:,-1].dump('eve_x_' + freq)
        #eig_vectors_y[:,0,:,-1].dump('eve_y_' + freq)

        logger.info("Task %s: Filtering largest eigenvalue." % self.__class__.__name__)

        tangent_x = []
        tangent_y = []
        for ti in xrange(eig_vectors_x.shape[0]):
            l_x = []
            l_y = []
            for i in xrange(eig_vectors_x.shape[2]-1):
                temp_x = 0
                temp_y = 0
                for j in xrange(i+1,eig_vectors_x.shape[2]):
                    temp_x += np.power(np.abs(eig_vectors_x[ti,0,j,-1]),2)
                    temp_y += np.power(np.abs(eig_vectors_y[ti,0,j,-1]),2)
                if temp_x < 1e-15:
                    t_x = np.nan
                else:
                    t_x = np.abs(eig_vectors_x[ti,0,j,-1])/np.sqrt(temp_x)
                l_x.append(t_x)
                if temp_y < 1e-15:
                    t_y = np.nan
                else:
                    t_y = np.abs(eig_vectors_y[ti,0,j,-1])/np.sqrt(temp_y)
                l_y.append(t_y)
                del t_x, temp_x, t_y, temp_y
            tangent_x.append(l_x)
            tangent_y.append(l_y)
            del l_x, l_y
        tangent_x = np.array(tangent_x,dtype=float)
        tangent_y = np.array(tangent_y,dtype=float)

        tangent_x_filled = []
        tangent_y_filled = []
        for i in xrange(tangent_x.shape[1]):
            temp_x = np.copy(tangent_x[:,i])
            temp_y = np.copy(tangent_y[:,i])
            nans_x, x = nan_helper(temp_x)
            nans_y, y = nan_helper(temp_y)
            temp_x[nans_x] = np.interp(x(nans_x), x(~nans_x), temp_x[~nans_x])
            temp_y[nans_y] = np.interp(y(nans_y), y(~nans_y), temp_y[~nans_y])
            tangent_x_filled.append(temp_x)
            tangent_y_filled.append(temp_y)
            del temp_x, nans_x, x, temp_y, nans_y, y
        tangent_x_filled = np.array(tangent_x_filled,dtype=float).T
        tangent_y_filled = np.array(tangent_y_filled,dtype=float).T

        del tangent_x, tangent_y
        tangent_x = tangent_x_filled
        tangent_y = tangent_y_filled
        del tangent_x_filled, tangent_y_filled

        ### Filtering ###
        tangent_x_filtered = []
        tangent_y_filtered = []
        for i in xrange(tangent_x.shape[1]):
            B, A = signal.butter(filter_order, cutoff_freq, output='ba')
            temp_x = signal.filtfilt(B,A, tangent_x[:,i])
            temp_y = signal.filtfilt(B,A, tangent_y[:,i])
            tangent_x_filtered.append(temp_x)
            tangent_y_filtered.append(temp_y)
            del temp_x, temp_y
        tangent_x_filtered = np.array(tangent_x_filtered,dtype=float).T
        tangent_y_filtered = np.array(tangent_y_filtered,dtype=float).T

        del tangent_x, tangent_y
        tangent_x = tangent_x_filtered
        tangent_y = tangent_y_filtered
        del tangent_x_filtered, tangent_y_filtered

        eig_vector_abs_x = []
        eig_vector_abs_y = []
        lim = eig_values_x.shape[2]
        for ti in xrange(eig_vectors_x.shape[0]):
            x = np.zeros((lim,),dtype=float)
            y = np.zeros((lim,),dtype=float)
            for i in xrange(lim):
                if i != lim - 1:
                    x[i] = np.sin(np.arctan(tangent_x[ti,i]))
                    y[i] = np.sin(np.arctan(tangent_y[ti,i]))
                else:
                    x[i] = 1.0
                    y[i] = 1.0
                for j in xrange(i):
                    x[i] = x[i]*np.cos(np.arctan(tangent_x[ti,j]))
                    y[i] = x[i]*np.cos(np.arctan(tangent_y[ti,j]))
            eig_vector_abs_x.append(x)
            eig_vector_abs_y.append(y)
        del x, y
        eig_vector_abs_x = np.array(eig_vector_abs_x,dtype=float)  # this is the np.abs of the largest eigenvector
        eig_vector_abs_y = np.array(eig_vector_abs_y,dtype=float)

        eig_vector_max_x = []
        eig_vector_max_y = []
        for i in xrange(lim):
            eig_vector_max_x.append(np.multiply(eig_vector_abs_x[:,i],np.cos(np.angle(eig_vectors_x[:,0,i,-1]))) 
                                + 1j* np.multiply(eig_vector_abs_x[:,i],np.sin(np.angle(eig_vectors_x[:,0,i,-1]))))
            eig_vector_max_y.append(np.multiply(eig_vector_abs_y[:,i],np.cos(np.angle(eig_vectors_y[:,0,i,-1]))) 
                                + 1j* np.multiply(eig_vector_abs_y[:,i],np.sin(np.angle(eig_vectors_y[:,0,i,-1]))))
        eig_vector_max_x = np.array(eig_vector_abs_x,dtype=complex).T
        eig_vector_max_y = np.array(eig_vector_abs_y,dtype=complex).T
        
        del eig_vector_abs_x, eig_vector_abs_y, lim

        logger.info("Task %s: Reconstructing the visibilities." % self.__class__.__name__)

        vis_sun_x = []
        vis_sun_y = []
        for ti in xrange(eig_vectors_x.shape[0]):
            temp_x = eig_values_x[ti,0,-1]*np.outer(eig_vectors_x[ti, 0, :, -1].conj().T,eig_vectors_x[ti, 0, :, -1])
            temp_y = eig_values_y[ti,0,-1]*np.outer(eig_vectors_y[ti, 0, :, -1].conj().T,eig_vectors_y[ti, 0, :, -1])
            vis_sun_x.append(temp_x)
            vis_sun_y.append(temp_y)
            del temp_x, temp_y
        vis_sun_x = np.array(vis_sun_x,dtype=complex)
        vis_sun_x = vis_sun_x.conj()
        vis_sun_y= np.array(vis_sun_y,dtype=complex)
        vis_sun_y = vis_sun_y.conj()

        for fi in xrange(rt.local_vis.shape[1]):
            #for ti in xrange(rt.local_vis.shape[0]):
            for si, ei in mask_inds:
                print('si: {}, ei: {}'.format(si, ei))
                for ti in xrange(si,ei):
                    if np.all(rt.local_vis_mask[ti, fi, :]):
                        continue 
                    vis_matrix_x = np.matmul(eig_vectors_x[ti, fi, :],np.matmul(np.diag(eig_values_x[ti, fi, :]),eig_vectors_x[ti, fi, :].conj().T))
                    vis_matrix_x = vis_matrix_x - vis_sun_x[ti]
                    vis_matrix_y = np.matmul(eig_vectors_y[ti, fi, :],np.matmul(np.diag(eig_values_y[ti, fi, :]),eig_vectors_y[ti, fi, :].conj().T))
                    vis_matrix_y = vis_matrix_y - vis_sun_y[ti]
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


