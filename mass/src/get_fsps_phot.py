from __future__ import print_function


# From James Schombert's version of cosmocalc:
def cosmt(reds):
    import numpy as np
    h = 0.682        # cosmology (DES Y1 + ext)
    wm = 0.301
    wv = 0.699 - 4.165e-5 / (h * h)
    wr = 4.165E-5 / (h * h)
    wk = 0.
    c = 299792.458
    Tyr = 977.8    # coefficent for converting 1/H into Gyr
    nprec = 1000   # time integration precision

    az = 1./(1. + 1.*reds)
    age = 0.
    for i in range(nprec):
        a = az * (float(i) + 0.5) / nprec
        adot = np.sqrt(wk + (wm / a) + (wr / (a * a)) + (wv * a * a))
        age = age + 1. / adot

    cosmt = az * age / nprec
    cosmt = cosmt * (Tyr / (h * 100.))
    return cosmt

def DL(z):
    import numpy as np
    H0 = 68.2
    h = H0/100.
    WV = 0.699 - 4.165e-5 / (h * h)
    WR = 4.165E-5/(h*h)
    WM = 1. - WV - WR
    c = 299792.458
    n = 10000
    az = 1.0/(1.+z)
    DCMR = 0.
    DTT = 0.
    for i in range(n):
        a = az+(1-az)*(i+0.5)/n
        adot = np.sqrt((WM/a)+(WR/(a*a))+(WV*a*a))
        DTT = DTT + 1./adot
        DCMR = DCMR + 1./(a*adot)
    DCMR = (1.-az)*DCMR/n
    DA = az*DCMR
    DL = DA/(az*az)
    DL_Mpc = (c/H0)*DL
    return DL_Mpc



def lognorm(x, mu, sig):
    import numpy as np
    return (1./(x*sig*2.*np.pi))*np.exp(-1.*(np.log(x)-mu)**2/(2*sig*sig))


# get metallicity from FMR
# needs: mass, SFR (Mannucci+ 2010) --> update to Lilly+13
def getMet(mass,SFR):
    import numpy as np
    lm = np.log10(mass) - 10.
    ls = np.log10(SFR)
    getMet = 0.37*lm - 0.14*ls - 0.19*lm*lm + 0.12*lm*ls - 0.054*ls*ls
    getMet = 0.02 * 10.**getMet
    getMet[np.isnan(getMet)] = 0.0001
    return getMet


def priortransform(cube):
    from scipy.special import erfinv
    # definition of the parameter width, by transforming from a unit cube
    cube = cube.copy()
    cube[0] = cube[0] * 5.5 + 7 # basically, this sets bounds on the mass: 7 to 12.5 -> -3 to 2 (A=1 -> 10^10). Working in log10(mass) units.
    cube[1] = cube[1] * 10. # mu: 0 to 10
    cube[2] = cube[2] * 4.8 + 0.2 # sigma: 0.2 to 5 
    cube[3] = 10**(0.3 * erfinv(2.*(cube[3]-0.5))) * 0.998 + 0.001  # metallicity scatter: Gaussian about 0, sigma = 0.3 dex, clip high sigma
    cube[4] = (10**(cube[4]) - 1.) / 4.5  # dust extinction (doubled in birth clouds): 0 to 2., log spaced
    cube[5] = 10**(cube[5]) - 1. # log-spaced redshift sampling, z: 0 to 9.
    return cube

############################
# Main programme down here # 
############################

#=================================================
# Version for applying Halton seq. params.
# Will extend to add in emission lines.
#=================================================

def get_fsps_fluxes(params):
    # params:
    # 0. amplitude (scale fluxes such that A=10 -> M=10^10)
    # 1. mu (lognormal, in Gyr)
    # 2. sigma (lognormal, in Gyr)
    # 3. metallicity scatter
    # 4. dust extinction
    # 5. redshift
    A, mu, sig, met, tau, z = priortransform(params)
    tstep = 1.e-3
    return_frac = 0.4

    """
    Working in units of Gyr, solar masses.

    Steps:
    - compute lognormal SFH since big bang
    - integrate SFH (account for return fraction), down to redshift of interest
    - multiply SFH, so that observed mass @ z = amplitude mass
    - compute FMR metallicity at each timestep
    - apply metallicity deviation
    - set the SFH in FSPS
    - apply dust extinction
    - apply Madau extinction
    - compute fluxes
    - obtain stellar mass
    - scale fluxes to desired amplitude and return
    """
    import numpy as np
    import time
    import subprocess
    import fsps

    # 0. Initialise FSPS
    sps = fsps.StellarPopulation(zcontinuous=3)
    
    # 1. compute lognormal SFH since big bang (hard-coded value for age of univ)
    t = np.linspace(tstep, 13.8, int(13.8/tstep))
    sfh = lognorm(t, mu, sig)

    # 2. Integrate SFH down to redshift of interest. Return fraction = 0.4
    # get age @ z from cosmocalc
    age = cosmt(z)
    #age = cosmt(0.)
    sfh_mass = np.sum(sfh[t<age]) * tstep * (1.-return_frac) * 1.e9

    # 3. multiply SFH, so that observed mass @ z = amplitude mass
    amp_mass = 10**(A)
    sfh *= amp_mass / sfh_mass

    # 4. compute FMR metallicity at each timestep
    # make cumulative SFH (i.e. mass at each age)
    cum_sfh = np.cumsum(sfh) * (1.-return_frac) * tstep * 1.e9
    gas_met = getMet(cum_sfh, sfh)

    # 5. apply metallicity deviation
    gas_met = gas_met * met
    gas_met[gas_met < 0.0001] = 0.0001

    # 6. write SFH.dat file (with appropriate name) - can use the python-fsps method, set_tabular_sfh
    sps.params['sfh'] = 3
    sps.set_tabular_sfh(t, sfh, Z=gas_met)
    #try:
    #    sps.set_tabular_sfh(t, sfh, Z=gas_met)
    #except:
    #    print(gas_met, params)

    # 7. set dust param, dust 1 is young stars
    sps.params['dust1'] = 2. * tau
    sps.params['dust2'] = tau

    # 8. apply Madau extinction
    sps.params['zred'] = z
    sps.params['add_igm_absorption'] = True

     # to get the length of the magnitude array
    mags = sps.get_mags(tage=age)
                        
    # set up flux and amplitude blocks
    fluxes = np.zeros(len(mags))
    #mass = np.zeros(len(zarr))
    
    # 9. compute fluxes - if mags work, this is a simple conversion
    sps.params['tage'] = age
    fluxes = sps.get_mags(tage=age) # these are actually mags, because that is what we will use in the G.P.
    #fluxes = 10**((30. - mags)/2.5)

    # 10. obtain stellar mass
    mass = sps.stellar_mass
    
    # 11. scale fluxes (mags) to desired amplitude and return
    fluxes -= 2.5 * (A - np.log10(mass))

    return fluxes




# Making a second copy of the main program, to return the spectrum.
#
#
#
#

def get_fsps_spec(params):
    # params:
    # 0. amplitude (scale fluxes such that A=10 -> M=10^10)
    # 1. mu (lognormal, in Gyr)
    # 2. sigma (lognormal, in Gyr)
    # 3. metallicity scatter
    # 4. dust extinction
    # 5. redshift
    A, mu, sig, met, tau, z = priortransform(params)
    tstep = 1.e-3
    return_frac = 0.4

    """
    Working in units of Gyr, solar masses.

    Steps:
    - compute lognormal SFH since big bang
    - integrate SFH (account for return fraction), down to redshift of interest
    - multiply SFH, so that observed mass @ z = amplitude mass
    - compute FMR metallicity at each timestep
    - apply metallicity deviation
    - set the SFH in FSPS
    - apply dust extinction
    - apply Madau extinction
    - compute fluxes
    - obtain stellar mass
    - scale fluxes to desired amplitude and return
    """
    import numpy as np
    import time
    import subprocess
    import fsps

    # 0. Initialise FSPS
    sps = fsps.StellarPopulation(zcontinuous=3)
    print("ok")
    
    # 1. compute lognormal SFH since big bang (hard-coded value for age of univ)
    t = np.linspace(tstep, 13.8, int(13.8/tstep))
    sfh = lognorm(t, mu, sig)

    # 2. Integrate SFH down to redshift of interest. Return fraction = 0.4
    # get age @ z from cosmocalc
    age = cosmt(z)
    #age = cosmt(0.)
    sfh_mass = np.sum(sfh[t<age]) * tstep * (1.-return_frac) * 1.e9

    # 3. multiply SFH, so that observed mass @ z = amplitude mass
    amp_mass = 10**(A)
    sfh *= amp_mass / sfh_mass

    # 4. compute FMR metallicity at each timestep
    # make cumulative SFH (i.e. mass at each age)
    cum_sfh = np.cumsum(sfh) * (1.-return_frac) * tstep * 1.e9
    gas_met = getMet(cum_sfh, sfh)

    # 5. apply metallicity deviation
    gas_met = gas_met * met
    gas_met[gas_met < 0.0001] = 0.0001

    # 6. write SFH.dat file (with appropriate name) - can use the python-fsps method, set_tabular_sfh
    sps.params['sfh'] = 3
    sps.set_tabular_sfh(t, sfh, Z=gas_met)
    #try:
    #    sps.set_tabular_sfh(t, sfh, Z=gas_met)
    #except:
    #    print(gas_met, params)

    # 7. set dust param, dust 1 is young stars
    sps.params['dust1'] = 2. * tau
    sps.params['dust2'] = tau

    # 8. apply Madau extinction
    sps.params['zred'] = z
    sps.params['add_igm_absorption'] = True
    print("still ok")

     # to get the length of the magnitude array
    mags = sps.get_mags(tage=age)
                        
    # set up flux and amplitude blocks
    fluxes = np.zeros(len(mags))
    #mass = np.zeros(len(zarr))
    
    # 9. compute fluxes - if mags work, this is a simple conversion
    sps.params['tage'] = age
    fluxes = sps.get_mags(tage=age) # these are actually mags, because that is what we will use in the G.P.
    #fluxes = 10**((30. - mags)/2.5)

    # 10. obtain stellar mass
    mass = sps.stellar_mass
    
    # 11. scale fluxes (mags) to desired amplitude and return
    fluxes -= 2.5 * (A - np.log10(mass))
    print(fluxes)

    # 12. OK, here we add in the bit to get the spectrum
    wave, spec = sps.get_spectrum(tage=age)
    print("got spectrum")

    # 13. Wavelength is in rest-frame (I think), so we move to obs frame.
    wave *= 1. + z

    # 14. Finally, we need to scale the spectrum to the right value.
    # It comes in units of L_sun / Hz, and we also need to adjust for the mass.
    spec *= 10**A / mass
    print(A, mass, spec[200])

    # Solar lumn -> erg / s / Hz = 3.826x10^33
    # Then, div. by 4 pi DL^2 (DL in cm)
    # and the redshift factor
    #spec *= 3.826e33 * (1. + z) / (3.0857e18 * DL(z) * 1.e5)**2

    # And change to our fluxes with zeropoint 30., not 48.6
    # OK, scaling is weird, probably messed some units up.
    # This values seems to work.
    spec *= 4.5e7
    
    return wave, spec
