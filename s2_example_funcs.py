#!/usr/bin/env python

import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import ephem

import prosail
import gp_emulator
from eoldas_ng import ObservationOperatorTimeSeriesGP

__author__ = "J Gomez-Dans"
__version__ = "1.0 (09.03.2015)"
__email__ = "j.gomez-dans@ucl.ac.uk"

class ObservationOperatorNew ( ObservationOperatorTimeSeriesGP ):
    """
    A practical class that changes how we "fish out" the emulators from 
    ``self.emulators``. All we need to do is to add a new ``time_step`` 
    method to the class.
    """
    def __init__ ( self, state_grid, state, observations, mask, emulators, bu, \
            band_pass=None, bw=None ):
        ObservationOperatorTimeSeriesGP.__init__ ( self, state_grid, state, observations, \
                                                  mask, emulators, bu, band_pass, bw )
        self.sel_emu_keys = []
        keys = np.array(emulators.keys())
        
        s2_emu_locs = [np.argmin(np.sum(( keys - mask[i, (3,2)])**2,axis=1)) \
                                 for i in xrange(mask.shape[0]) ]
        self.sel_emu_keys = keys[s2_emu_locs]
        print len(self.sel_emu_keys), observations.shape
    
    def time_step ( self, this_loc ):
        k = tuple(self.sel_emu_keys[this_loc])
        this_obs = self.observations[ :, this_loc]
        return self.emulators[k], this_obs, \
            [self.band_pass, self.bw]
            

def plot_config ():
    """Update the MPL configuration"""
    config_json='''{
            "lines.linewidth": 2.0,
            "axes.edgecolor": "#bcbcbc",
            "patch.linewidth": 0.5,
            "legend.fancybox": true,
            "axes.color_cycle": [
                "#FC8D62",
                "#66C2A5",
                "#8DA0CB",
                "#E78AC3",
                "#A6D854",
                "#FFD92F",
                "#E5C494",
                "#B3B3B3"
            ],
            "axes.facecolor": "w",
            "axes.labelsize": "large",
            "axes.grid": false,
            "patch.edgecolor": "#eeeeee",
            "axes.titlesize": "x-large",
            "svg.embed_char_paths": "path",
            "xtick.direction" : "out",
            "ytick.direction" : "out",
            "xtick.color": "#262626",
            "ytick.color": "#262626",
            "axes.edgecolor": "#262626",
            "axes.labelcolor": "#262626",
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12
            
    }
    '''
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['xtick.minor.size'] = 10
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['ytick.minor.size'] = 10
    plt.rcParams['ytick.minor.width'] = 0.5
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']

    s = json.loads ( config_json )
    plt.rcParams.update(s)
    plt.rcParams["axes.formatter.limits"] = [-4,4]
    

def pretty_axes( ax ):
    """This function takes an axis object ``ax``, and makes it purrty.
    Namely, it removes top and left axis & puts the ticks at the
    bottom and the left"""

    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(True)  
    ax.spines["right"].set_visible(False)              
    ax.spines["left"].set_visible(True)  

    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    loc = plt.MaxNLocator( 6 )
    ax.yaxis.set_major_locator( loc )
    

    ax.tick_params(axis="both", which="both", bottom="on", top="off",  
            labelbottom="on", left="on", right="off", labelleft="on")  

    
def create_observations2 ( state, parameter_grid, latitude, longitude, the_time="10:30",\
        b_min = np.array( [ 620., 841, 459, 545, 1230, 1628, 2105] ), \
        b_max = np.array( [ 670., 876, 479, 565, 1250, 1652, 2155] ), \
        every=5, prop=0.4, WINDOW=3, noise_scalar=1, vza_var=7.):
    """This function creates the observations for  a given temporal evolution
    of parameters, loation, and bands. By default, we take only MODIS bands. 
    The function does a number of other things:
    1.- Calculate missing observations due to simulated cloud
    2.- Add noise
    TODO: There's a problem  with pyephem, gives silly solar altitudes!!!"""
    wv = np.arange ( 400, 2501 )
    n_bands = b_min.shape[0]
    band_pass = np.zeros(( n_bands,2101), dtype=np.bool)
    bw = np.zeros( n_bands )
    bh = np.zeros( n_bands )
    for i in xrange( n_bands ):
        band_pass[i,:] = np.logical_and ( wv >= b_min[i], \
                wv <= b_max[i] )
        bw[i] = b_max[i] - b_min[i]
        bh[i] = ( b_max[i] + b_min[i] )/2.
    import ephem
    o = ephem.Observer()
    o.lat, o.long, o.date = latitude, longitude, "2011/1/1 %s" % the_time
    dd = o.date

    t = np.arange ( np.random.randint(1,every), 366 )
    obs_doys = np.array ( [ i for i in t if i % every == 0 ] )
    

    #prop = 0.4
    #WINDOW = 3
    weightings = np.repeat(1.0, WINDOW) / WINDOW
    
    xx = np.convolve(np.random.rand(len(t)*100),weightings,'valid')[WINDOW:WINDOW+len(t)]

    maxx = sorted(xx)[:int(len(xx)*prop)]
    mask = np.in1d(xx,maxx)
    doys_nocloud = t[mask]
    x = np.in1d ( obs_doys, doys_nocloud )
    obs_doys = obs_doys[x]
    vza = np.zeros_like ( obs_doys, dtype=np.float)
    sza = np.zeros_like ( obs_doys, dtype=np.float)
    raa = np.zeros_like ( obs_doys, dtype=np.float)
    rho = np.zeros (( len(bw), obs_doys.shape[0] ))
    sigma_obs = (0.05-0.01)*(bh-bh.min())/(bh.max()-bh.min())
    sigma_obs += 0.01
    sigma_obs = sigma_obs * noise_scalar
    S = np.array([22, 37, 52, 70])
    for i,doy in enumerate(obs_doys):
        j = doy - 1 # location in parameter_grid...
        vza[i] = np.random.rand(1)*vza_var # 15 degs 
        o.date = dd + doy
        sun = ephem.Sun ( o )
        sza[i] = 90. - float(sun.alt )*180./np.pi
        ##sza[i] = S[np.argmin ( np.abs ( xx-S))]
        ##sza[i] = 37
        vaa = np.random.rand(1)*360.
        saa = np.random.rand(1)*360.
        raa[i] = 0.0#vaa - saa
        p = np.r_[parameter_grid[:8,j],0,parameter_grid[8:, j], 0.01,sza[i], vza[i], raa[i], 2 ]
        
        r =  fixnan( np.atleast_2d ( prosail.run_prosail ( *p )) ).squeeze()
        rho[:, i] = np.array ( [r[ band_pass[ii,:]].sum()/bw[ii] \
            for ii in xrange(n_bands) ] )
        rho[:, i] = np.clip (rho[:, i] + np.random.randn ( n_bands )*sigma_obs, 1e-4, 0.999)
    return obs_doys, vza, sza, raa, rho, sigma_obs     
    
def create_observations___OLD ( state, parameter_grid, latitude, longitude, the_time="10:30", \
        b_min = np.array( [ 620., 841, 459, 545, 1230, 1628, 2105] ), \
        b_max = np.array( [ 670., 876, 479, 565, 1250, 1652, 2155] ), \
        every=5, prop=0.4, WINDOW=3, noise_scalar=1):
    """This function creates the observations for  a given temporal evolution
    of parameters, loation, and bands. By default, we take only MODIS bands. 
    The function does a number of other things:
    1.- Calculate missing observations due to simulated cloud
    2.- Add noise
    TODO: There's a problem  with pyephem, gives silly solar altitudes!!!"""
    wv = np.arange ( 400, 2501 )
    n_bands = b_min.shape[0]
    band_pass = np.zeros(( n_bands,2101), dtype=np.bool)
    bw = np.zeros( n_bands )
    bh = np.zeros( n_bands )
    for i in xrange( n_bands ):
        band_pass[i,:] = np.logical_and ( wv >= b_min[i], \
                wv <= b_max[i] )
        bw[i] = b_max[i] - b_min[i]
        bh[i] = ( b_max[i] + b_min[i] )/2.
    o = ephem.Observer()
    o.lat, o.long, o.date = latitude, longitude, "2011/1/1 %s"%the_time
    dd = o.date

    t = np.arange ( 1, 366 )
    obs_doys = np.array ( [ i for i in t if i % every == 0 ] )
    

    #prop = 0.4
    #WINDOW = 3
    weightings = np.repeat(1.0, WINDOW) / WINDOW
    
    xx = np.convolve(np.random.rand(len(t)*100),weightings,'valid')[WINDOW:WINDOW+len(t)]

    maxx = sorted(xx)[:int(len(xx)*prop)]
    mask = np.in1d(xx,maxx)
    doys_nocloud = t[mask]
    x = np.in1d ( obs_doys, doys_nocloud )
    obs_doys = obs_doys[x]
    vza = np.zeros_like ( obs_doys, dtype=np.float)
    sza = np.zeros_like ( obs_doys, dtype=np.float)
    raa = np.zeros_like ( obs_doys, dtype=np.float)
    rho = np.zeros (( len(bw), obs_doys.shape[0] ))
    sigma_obs = (0.01-0.004)*(bh-bh.min())/(bh.max()-bh.min())
    sigma_obs += 0.004
    sigma_obs = sigma_obs * noise_scalar
    S = np.array([22, 37, 52, 60])
    for i,doy in enumerate(obs_doys):
        j = doy - 1 # location in parameter_grid...
        vza[i] = 8#np.random.rand(1)*15. # 15 degs 
        o.date = dd + doy
        sun = ephem.Sun ( o )
        xx = 90. - float(sun.alt )*180./np.pi
        sza[i] = S[np.argmin ( np.abs ( xx-S))]
        vaa = np.random.rand(1)*360.
        saa = np.random.rand(1)*360.
        raa[i] = 0.0#vaa - saa
        p = np.r_[parameter_grid[:8,j],0,parameter_grid[8:, j], 0.01,sza[i], vza[i], raa[i], 2 ]
        
        r =  fixnan( np.atleast_2d ( prosail.run_prosail ( *p )) ).squeeze()
        rho[:, i] = np.array ( [r[ band_pass[ii,:]].sum()/bw[ii] \
            for ii in xrange(n_bands) ] )
        rho[:, i] += np.random.randn ( n_bands )*sigma_obs
        rho[:, i] = np.clip ( rho[:,i], 1e-4, 0.9999)
    return obs_doys, vza, sza, raa, rho, sigma_obs 

def dbl_logistic_model ( p, x ):
    """A double logistic model, as in Sobrino and Juliean, or Zhang et al"""
    return p[0] + p[1]* ( 1./(1+np.exp(p[2]*(365.*x-p[3]))) + \
                          1./(1+np.exp(p[4]*(365.*x-p[5])))  - 1 )

def create_parameter_trajectories ( state, trajectory_funcs ):
    """This function creates the parameter trajectories as in the 
    RSE 2012 paper, just for testing"""
    t = np.arange ( 1, 366 )/365.
    parameter_grid = np.zeros ( (len(state.default_values.keys()), \
         t.shape[0] ) )
    for i, (parameter, default) in \
        enumerate ( state.default_values.iteritems() ):
            if trajectory_funcs.has_key ( parameter ):
                parameter_grid [i, : ] = trajectory_funcs[parameter](t)
            else:
                parameter_grid[i,:] = default
             
    return parameter_grid

def fixnan(x):
    '''
    the RT model sometimes fails so we interpolate over nan
    
    This method replaces the nans in the vector x by their interpolated values
    '''
    for i in xrange(x.shape[0]):
        sample = x[i]
        ww = np.where(np.isnan(sample))
        nww = np.where(~np.isnan(sample))
        sample[ww] = np.interp(ww[0],nww[0],sample[nww])
        x[i] = sample

    return x

def grab_emulators ( vzaf, szaf, emulator_home="/home/ucfajlg/python/emulators/"):
    
    files = glob.glob("%s*.npz" % emulator_home)
    emulator_search_dict = {}
    for f in files:      
        sza, saa,vza, vaa = float(f.split("/")[-1].split("_")[0]), \
                float(f.split("/")[-1].split("_")[1]), float(f.split("/")[-1].split("_")[0]), \
                float(f.split("/")[-1].split("_")[1])
        if saa == vaa:
            emulator_search_dict[ float(f.split("/")[-1].split("_")[0]), \
                            float(f.split("/")[-1].split("_")[2]) ] = f
    emu_locs = [np.argmin(np.sum((np.array(emulator_search_dict.keys()) - \
                                 np.array([szaf[i], vzaf[i]]))**2,axis=1)) \
                 for i in xrange(len(szaf)) ]
    emu_keys = emulator_search_dict.keys()
    unique_emulators = np.unique(emu_locs)

    emulators = {}

    for i, emu_locs in enumerate ( unique_emulators ):
        k = emu_keys[emu_locs]
        emulators[(int(k[0]), int(k[1]))] = gp_emulator.MultivariateEmulator ( dump=emulator_search_dict[k])
    return emulators 

def grab_emulators2 ( vza, sza, raa, emulator_home="/home/jose/emulator/"):
    
    # Locate all available emulators...
    files = glob.glob("%s*.npz" % emulator_home)
    emulator_search_dict = {}
    for f in files:
        emulator_search_dict[ float(f.split("/")[-1].split("_")[0]), \
                                float(f.split("/")[-1].split("_")[2]),
                                float(f.split("/")[-1].split("_")[1]) - \
                                float(f.split("/")[-1].split("_")[3])] = f
    # So we have a dictionary inddexed by SZA, VZA and RAA and mapping to a filename
    # Remove some weirdos...
    #for k in emulator_search_dict.keys():
    #    if (k[1] - int(k[1]) ) != 0.:
    #        emulator_search_dict.pop ( k )
    emu_keys = np.array( emulator_search_dict.keys() )
        
    #emu_locs = [np.argmin(np.sum((emu_keys[:,:2] - \
    #                                 np.array([sza[i], vza[i]]))**2,axis=1)) \
    #             for i in xrange(len(doys)) ]
    emu_locs = [np.sum((emu_keys[:,:2] - \
                                   np.array([sza[i], vza[i]]))**2,axis=1) \
                    for i in xrange(vza.shape[0])]
    emulators = {}
    for i in xrange(len(sza)):
        ang_pass = emu_locs[i] == emu_locs[i][emu_locs[i].argmin()]
        isel =  np.abs( emu_keys[ang_pass][:,-1] - raa[i]).argmin()
        the_emu_key = emu_keys[ang_pass][isel]
        k = the_emu_key
        #print i,sza[i], vza[i], raa[i], emu_keys[ang_pass][isel], \
        #        emulator_search_dict[(k[0], k[1],k[2])]
        if not emulators.has_key ( (int(k[0]), int(k[1]), int(k[2])) ):
            emulators[(int(k[0]), int(k[1]), int(k[2]))] = gp_emulator.MultivariateEmulator \
                    ( dump=emulator_search_dict[(k[0], k[1],k[2])])
    #emu_keys = np.array ( emulators.keys() )
    #emu_locs = np.array ( [np.argmin(np.sum(( emu_keys - mask[i, 2:])**2,axis=1)) \
    #                for i in xrange(vza.shape[0])] )
    return emulators

def spectral_configuration():
    """provides spectral configurations for different sensors."""
    wv = np.arange ( 400, 2501 )

    ### Sentinel 2
    s2_bh = np.array ( [ 443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190])
    s2_bw = np.array ( [20, 65, 35., 30, 15, 15, 20, 115, 20, 20, 90, 180 ])
    s2_b_min = s2_bh - s2_bw*0.5
    s2_b_max = s2_bh + s2_bw*0.5
    s2_band_pass = np.zeros((12,2101), dtype=np.bool)
    s2_n_bands = s2_b_min.shape[0]
    for i in xrange( s2_n_bands ):
        s2_band_pass[i,:] = np.logical_and ( wv >= s2_b_min[i], \
                wv <= s2_b_max[i] )
    ### ProbaV
    proba_bh = np.array ( [ 0.5*(493.+447), 0.5*(610.+690), 0.5*(777.+893), 0.5*(1570.+1650)] )
    proba_bw = np.array ( [ (493.-447), -(610.-690), -(777.-893), -(1570.-1650)] )
    
    proba_b_min = proba_bh - proba_bw*0.5
    proba_b_max = proba_bh + proba_bw*0.5
    proba_band_pass = np.zeros((4,2101), dtype=np.bool)
    proba_n_bands = proba_b_min.shape[0]
    for i in xrange( proba_n_bands ):
        proba_band_pass[i,:] = np.logical_and ( wv >= proba_b_min[i], \
                wv <= proba_b_max[i] )
    ### SLSTR
    slstr_bh = np.array ( [ 550, 659, 865, 1610, 2250])
    slstr_bw = np.array ( [ 30, 30, 30, 30, 30 ])
    slstr_b_min = slstr_bh - slstr_bw*0.5
    slstr_b_max = slstr_bh + slstr_bw*0.5
    slstr_band_pass = np.zeros((5,2101), dtype=np.bool)
    slstr_n_bands = slstr_b_min.shape[0]
    for i in xrange( slstr_n_bands ):
        slstr_band_pass[i,:] = np.logical_and ( wv >= slstr_b_min[i], \
                wv <= slstr_b_max[i] )
    ### OLCI
    
      
    olci_bh = np.array ( [ 400, 412, 442, 490, 510, 560, 620, 665, 681,708,753,865, 885, 900,1020])
    olci_bw = np.array ( [ 15,  10,   10,  10,  10,  10,  10,  10,   8, 10,  8, 20,  10,  10, 40 ])
    olci_b_min = olci_bh - olci_bw*0.5
    olci_b_max = olci_bh + olci_bw*0.5
    olci_band_pass = np.zeros((15,2101), dtype=np.bool)
    olci_n_bands = olci_b_min.shape[0]
    for i in xrange( olci_n_bands ):
        olci_band_pass[i,:] = np.logical_and ( wv >= olci_b_min[i], \
                wv <= olci_b_max[i] )
    return s2_bh, s2_n_bands, s2_b_min, s2_b_max, s2_band_pass, \
           proba_bh, proba_n_bands, proba_b_min, proba_b_max, proba_band_pass, \
           slstr_bh, slstr_n_bands, slstr_b_min, slstr_b_max, slstr_band_pass, \
           olci_bh, olci_n_bands, olci_b_min, olci_b_max, olci_band_pass
    
    
