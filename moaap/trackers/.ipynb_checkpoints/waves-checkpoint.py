import numpy as np
from scipy import fftpack, ndimage, signal
from moaap.utils.constants import g, a, beta, pi, NA
from moaap.utils.data_proc import tukey_latitude_mask, temporal_tukey_window, interpolate_temporal
from moaap.utils.segmentation import watershed_3d_overlap_parallel, analyze_watershed_history
from moaap.utils.object_props import clean_up_objects, BreakupObjects, ConnectLon_on_timestep
import gc
import sys

def track_tropwaves_tb(tb,
                   Lat,
                   connectLon,
                   dT,
                   Gridspacing,
                   er_th = 0.05,  
                   mrg_th = 0.05, 
                   igw_th = 0.2,  
                   kel_th = 0.1,  
                   eig0_th = 0.1, 
                   breakup = 'watershed',
                   analyze_twave_history = False
                   ):
    """
    Identifies and tracks tropical waves using wavenumber-frequency filtering 
    (Wheeler & Kiladis method) applied to precipitation data.

    Parameters
    ----------
    pr : np.ndarray
        Precipitation data.
    Lat : np.ndarray
        Latitude grid.
    dT : int
        Time step (hours).
    er_th, mrg_th, igw_th, kel_th, eig0_th : float
        Amplitude thresholds for identifying Equatorial Rossby, Mixed Rossby Gravity, 
        Inertia Gravity, Kelvin, and Eastward Inertia Gravity waves respectively.
    breakup : str
        Method to handle merging objects ('breakup' or 'watershed').
    analyze_twave_history : bool, optional
        If True, computes watershed merge/split history.

    Returns
    -------
    mrg_objects, igw_objects, kelvin_objects, eig0_objects, er_objects : np.ndarray
        Labeled object arrays for each wave type.
    """

    tb = np.asarray(tb, dtype=np.float32)

    ew_mintime = 32
    
    # use Turkey (hals cos) function to tamper region
    lat_mask = tukey_latitude_mask(Lat, lat_start=17.0, lat_stop=25.0)
    tb_eq = tb.copy()
    tb_eq[tb_eq > 350] = np.nan
    tb_eq[tb_eq < 150] = np.nan

    # compute anomalies
    tb_eq = tb_eq - np.nanmean(tb_eq, axis=(1,2), keepdims=True)
    tb_eq = tb_eq * lat_mask[None,:]
    tb_eq[np.isnan(tb_eq)] = 0
    
    # pad the Tb to avoid boundary effects
    # temporal turkey tapping:
    nt = tb_eq.shape[0]
    win = temporal_tukey_window(nt, alpha=0.2)
    tb_eq = tb_eq * win[:, None, None]
    pad_size = int(tb_eq.shape[0] * 0.2)
    tb_eq = np.pad(tb_eq, ((pad_size,pad_size),(0,0),(0,0)), mode='reflect')
     
    tb_eq = interpolate_temporal(tb_eq)
    tropical_waves = KFfilter(tb_eq,
                     int(24/dT))

    wave_names = ['ER','MRG','IGW','Kelvin','Eig0']

    print('        track tropical waves')
    rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
    for wa in range(5):
        print('            work on ' + wave_names[wa])
        if wa == 0:
            amplitude = KFfilter.erfilter(tropical_waves, fmin=None, fmax=None, kmin=-10, kmax=-1, hmin=0, hmax=90, n=1) # had to set hmin from 8 to 0
            wave = amplitude[pad_size:-pad_size] < er_th
            threshold = er_th
        if wa == 1:
            amplitude = KFfilter.mrgfilter(tropical_waves)
            wave = amplitude[pad_size:-pad_size] < mrg_th
            threshold = mrg_th
        elif wa == 2:
            amplitude = KFfilter.igfilter(tropical_waves)
            wave = amplitude[pad_size:-pad_size] < igw_th
            threshold = igw_th
        elif wa == 3:
            amplitude = KFfilter.kelvinfilter(tropical_waves)
            wave = amplitude[pad_size:-pad_size] < kel_th
            threshold = kel_th
        elif wa == 4:
            amplitude = KFfilter.eig0filter(tropical_waves)
            wave = amplitude[pad_size:-pad_size] < eig0_th
            threshold = eig0_th

        amplitude = amplitude[pad_size:-pad_size]

        if breakup == 'breakup':
            print('                break up long tropical waves that have many elements')
            wave_objects, object_split = BreakupObjects(wave_objects,
                                    int(ew_mintime/dT),
                                    dT)
        elif breakup == 'watershed':
            min_dist=int((1000 * 10**3)/Gridspacing)
            wave_amp = amplitude
            wave_objects = watershed_3d_overlap_parallel(
                    wave_amp *-1,
                    np.abs(threshold),
                    np.abs(threshold),
                    min_dist,
                    dT,
                    mintime = ew_mintime,
                    )
            
        if connectLon == 1:
            print('                connect waves objects over date line')
            wave_objects = ConnectLon_on_timestep(wave_objects)

        wave_objects, _ = clean_up_objects(wave_objects,
                          dT,
                          min_tsteps=int(ew_mintime/dT))

        if wa == 0:
            er_objects = wave_objects.copy()
        if wa == 1:
            mrg_objects = wave_objects.copy()
        if wa == 2:
            igw_objects = wave_objects.copy()
        if wa == 3:
            kelvin_objects = wave_objects.copy()
        if wa == 4:
            eig0_objects = wave_objects.copy()

        if analyze_twave_history:
            min_dist=int((1000 * 10**3)/Gridspacing)
            print(f"    Minimum distance between {wave_names[wa]} maxima for watershed analysis: {min_dist} grid cells")
            union_array, events, histories = analyze_watershed_history(
                wave_objects, min_dist, wave_names[wa].lower()
            )

            union_array_clean = {int(k): int(v) for k, v in union_array.items()}
            events_clean = [
            {
                'type': e['type'],
                'time': int(e['time']),
                'from_label': int(e['from_label']),
                'to_label': int(e['to_label']),
                'distance': float(e['distance'])
            }
            for e in events
            ]
            histories_clean = {int(root): [int(label) for label in labels] for root, labels in histories.items()}

            print(f"    Printing union array: {dict(list(union_array_clean.items()))}")
            print(f"    Printing events: {events_clean}")
            print(f"    Printing histories: {dict(list(histories_clean.items()))}")


        del wave
        del wave_objects
        gc.collect()
    return mrg_objects, igw_objects, kelvin_objects, eig0_objects, er_objects

class KFfilter:
    """class for wavenumber-frequency filtering for WK99 and WKH00"""
    def __init__(self, datain, spd, tim_taper=0.1):
        """Arguments:
        
       'datain'    -- the data to be filtered. dimension must be (time, lat, lon)

       'spd'       -- samples per day

       'tim_taper' -- tapering ratio by cos. applay tapering first and last tim_taper%
                      samples. default is cos20 tapering

                      """
        ntim, nlat, nlon = datain.shape

        #remove dominal trend
        data = signal.detrend(datain, axis=0)

        #tapering
        if tim_taper == 'hann':
            window = signal.hann(ntim)
            data = data * window[:,NA,NA]
        elif tim_taper > 0:
        #taper by cos tapering same dtype as input array
            tp = int(ntim*tim_taper)
            window = np.ones(ntim, dtype=datain.dtype)
            x = np.arange(tp)
            window[:tp] = 0.5*(1.0-np.cos(x*pi/tp))
            window[-tp:] = 0.5*(1.0-np.cos(x[::-1]*pi/tp))
            data = data * window[:,NA,NA]

        #FFT
        self.fftdata = fftpack.fft2(data, axes=(0,2))

        #Note
        # fft is defined by exp(-ikx), so to adjust exp(ikx) multipried minus         
        wavenumber = -fftpack.fftfreq(nlon)*nlon
        frequency = fftpack.fftfreq(ntim, d=1./float(spd))
        knum, freq = np.meshgrid(wavenumber, frequency)

        #make f<0 domain same as f>0 domain
        #CAUTION: wave definition is exp(i(k*x-omega*t)) but FFT definition exp(-ikx)
        #so cahnge sign
        knum[freq<0] = -knum[freq<0]
        freq = np.abs(freq)
        self.knum = knum
        self.freq = freq

        self.wavenumber = wavenumber
        self.frequency = frequency

    def decompose_antisymm(self):
        """
        decompose attribute data to sym and antisym component.

        Parameters
        ----------
        None
        """
        fftdata = self.fftdata
        nf, nlat, nk = fftdata.shape
        symm = 0.5*(fftdata[:,:nlat/2+1,:] + fftdata[:,nlat:nlat/2-1:-1,:])  
        anti = 0.5*(fftdata[:,:nlat/2,:] - fftdata[:,nlat:nlat/2:-1,:]) 
        
        self.fftdata = np.concatenate([anti, symm],axis=1)

    def kfmask(self, fmin=None, fmax=None, kmin=None, kmax=None):
        """return wavenumber-frequency mask for wavefilter method

        Arguments:

           'fmin/fmax' --

           'kmin/kmax' --

        Returns:
              'mask' -- 2D boolean array (wavenumber, frequency).domain to be filterd
        """
        nf, nlat, nk = self.fftdata.shape
        knum = self.knum
        freq = self.freq

        #wavenumber cut-off
        mask = np.zeros((nf,nk), dtype=bool)
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        return mask

    def wavefilter(self, mask):
        """apply wavenumber-frequency filtering by original mask.
        
        Arguments:
        
           'mask' -- 2D boolean array (wavenumber, frequency).domain to be filterd
                     is False (True member to be zero)
        Returns:
              'filterd' -- filtered data in the original data space
        """
        wavenumber = self.wavenumber
        frequency = self.frequency
        fftdata = self.fftdata.copy()
        nf, nlat, nk = fftdata.shape

        if (nf, nk) != mask.shape:
            print( "mask array size is incorrect.")
            sys.exit()

        mask = np.repeat(mask[:,NA,:], nlat, axis=1)    
        fftdata[mask] = 0.0

        #inverse FFT
        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    #filter
    def kelvinfilter(self, fmin=0.05, fmax=0.4, kmin=None, kmax=14, hmin=8, hmax=90):
        """kelvin wave filter

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number

           'hmin/hmax' --equivalent depth
        
        Returns:
              'filterd' -- filtered data in the original data space
        """
        
        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = np.zeros((nf,nk), dtype=bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = np.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / np.sqrt(beta*c) #adusting day^-1 to s^-1
            k     = knum/a * np.sqrt(c/beta)         #adusting ^2pia to ^m
            mask = mask | (omega - k <0)
        if hmax != None:
            c = np.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / np.sqrt(beta*c) #adusting day^-1 to s^-1
            k     = knum/a * np.sqrt(c/beta)         #adusting ^2pia to ^m
            mask = mask | (omega - k >0)

        mask = np.repeat(mask[:,NA,:], nlat, axis=1)
        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def erfilter(self, fmin=None, fmax=None, kmin=-10, kmax=-1, hmin=8, hmax=90, n=1):
        """equatorial wave filter

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number

           'hmin/hmax' -- equivalent depth

           'n'         -- meridional mode number
        
        Returns:
            'filterd' -- filtered data in the original data space
        """

        if n <=0 or n%1 !=0:
            print("n must be n>=1 integer")
            sys.exit()

        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = np.zeros((nf,nk), dtype=bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = np.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / np.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * np.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega*(k**2 + (2*n+1)) + k  < 0)
        if hmax != None:
            c = np.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / np.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * np.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega*(k**2 + (2*n+1)) + k  > 0)
        mask = np.repeat(mask[:,NA,:], nlat, axis=1)
        
        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def igfilter(self, fmin=None, fmax=None, kmin=-15, kmax=-1, hmin=12, hmax=90, n=1):
        """n>=1 inertio gravirt wave filter. default is n=1 WIG.

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number. negative is westward, positive is
                          eastward

           'hmin/hmax' -- equivalent depth

           'n'         -- meridional mode number
        
        Returns:
            'filterd' -- filtered data in the original data space
        """
        if n <=0 or n%1 !=0:
            print("n must be n>=1 integer. for n=0 EIG you must use eig0filter method.")
            sys.exit()

        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = np.zeros((nf,nk), dtype=bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = np.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / np.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * np.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k**2 - (2*n+1)  < 0)
        if hmax != None:
            c = np.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / np.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * np.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k**2 - (2*n+1)  > 0)
        mask = np.repeat(mask[:,NA,:], nlat, axis=1)
        fftdata[mask] = 0.0
        
        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def eig0filter(self, fmin=None, fmax=0.55, kmin=0, kmax=15, hmin=12, hmax=50):
        """n>=0 eastward inertio gravirt wave filter.

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number. negative is westward, positive is
                          eastward

           'hmin/hmax' -- equivalent depth

        Returns:
            'filterd' -- filtered data in the original data space
        """
        if kmin < 0:
            print("kmin must be positive. if k < 0, this mode is MRG")
            sys.exit()

        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = np.zeros((nf,nk), dtype=bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = np.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / np.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * np.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k*omega - 1 < 0)
        if hmax != None:
            c = np.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / np.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * np.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k*omega - 1 > 0)
        mask = np.repeat(mask[:,NA,:], nlat, axis=1)
        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def mrgfilter(self, fmin=None, fmax=None, kmin=-10, kmax=-1, hmin=8, hmax=90):
        """mixed Rossby gravity wave

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number. negative is westward, positive is
                          eastward

           'hmin/hmax' -- equivalent depth

        Returns:
            'filterd' -- filtered data in the original data space
        """
        if kmax > 0:
            print("kmax must be negative. if k > 0, this mode is the same as n=0 EIG")
            sys.exit()
            
        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape

        # filtering ############################################################
        mask = np.zeros((nf,nk), dtype=bool)
        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        if hmin != None:
            c = np.sqrt(g*hmin)
            omega = 2.*pi*freq/24./3600. / np.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * np.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k*omega - 1 < 0)
        if hmax != None:
            c = np.sqrt(g*hmax)
            omega = 2.*pi*freq/24./3600. / np.sqrt(beta*c)           #adusting day^-1 to s^-1
            k     = knum/a * np.sqrt(c/beta)                         #adusting ^2pia to ^m               
            mask = mask | (omega**2 - k*omega - 1 > 0)
        mask = np.repeat(mask[:,NA,:], nlat, axis=1)
        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real

    def tdfilter(self, fmin=None, fmax=None, kmin=-20, kmax=-6):
        """KTH05 TD-type filter.

        Arguments:

           'fmin/fmax' -- unit is cycle per day

           'kmin/kmax' -- zonal wave number. negative is westward, positive is
                          eastward
        
        Returns:
            'filterd' -- filtered data in the original data space
        """
        fftdata = self.fftdata.copy()
        knum = self.knum
        freq = self.freq
        nf, nlat, nk = fftdata.shape
        mask = np.zeros((nf,nk), dtype=bool)

        #wavenumber cut-off
        if kmin != None:
            mask = mask | (knum < kmin) 
        if kmax != None:
            mask = mask | (kmax < knum)

        #frequency cutoff
        if fmin != None:
            mask = mask | (freq < fmin)
        if fmax != None:
            mask = mask | (fmax < freq)

        #dispersion filter
        mask = mask | (84*freq+knum-22 > 0) | (210*freq+2.5*knum-13 < 0)                                                                                         
        mask = np.repeat(mask[:,NA,:], nlat, axis=1)

        fftdata[mask] = 0.0

        filterd = fftpack.ifft2(fftdata, axes=(0,2))
        return filterd.real


