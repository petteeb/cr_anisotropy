
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import copy
from scipy.stats import chi2
from mapFunctions import mapFunctions as mf
import Ani2 as ani2

'''
A mess of interconnected functions serving to analyze the cosmic ray anisotropy observed in IceCubes data.
'''

#Path to the fits files that will be used. include files from 2011 to 2020
path_to_sidereal_fits = 'MapData/'
# Used to access the original fits files and future generated files. 
yr_range = [2011, 2020]
#Conversion factor (degrees to radians) used like a label
degree = np.pi / 180
#Used in calculating the error in the relative intensities
alpha = 1/20
#Color map used in the creation of skymaps
colormap = cmap=copy.copy(cm.turbo)

NSIDE = 64

npixels = hp.nside2npix(NSIDE)

# manually import useful functions from sphereharm.py, or else...


# make isotropic?
def make_iso(bgMap, ss=100, lmax=20):
    # i will place a map directly into the function.
    #masking
    weight = mask(bgMap, minDec=-15*degree)

    weight[weight==hp.UNSEEN] = 0

    fake_cl = np.zeros((ss, lmax+1))
    
    print(fake_cl.size)

    #I dont know what option 2 is supposed to be or do.

    for n in range(ss):
        # poisson distribution, creates samples from probablitly of an event happening from bg map.

        dummyMap= np.random.poisson(bgMap)
        # calculating the relative intensity from poisson dist.
        rel_int = np.nan_to_num((dummyMap/bgMap)-1) ####################################################################################
        weighted_map =  rel_int * weight
        avg = np.average(weighted_map)
        q = weighted_map - avg

        f_cl = hp.anafast(q, lmax=lmax)

        fake_cl[n] = f_cl
        # so now there is a big array of fake cls?
    
    # makes arrays of fake cls and standard deviation

    f_cl_3siga_1 = []
    f_cl_3sigb_1 = []

    #Appending cls to different sigmas?
    for i, f_cl in enumerate(fake_cl.T):
        f_cl_3siga = np.percentile(f_cl,2.5)
        f_cl_3sigb = np.percentile(f_cl, 97.5)

        f_cl_3siga_1 += [f_cl_3siga]
        f_cl_3sigb_1 += [f_cl_3sigb]
    
    f_cl_3siga = np.asarray(f_cl_3siga_1)
    f_cl_3sigb = np.asarray(f_cl_3sigb_1)

    f_cl_3siga = f_cl_3siga[1:]
    f_cl_3sigb = f_cl_3sigb[1:]

    return f_cl_3siga, f_cl_3sigb



def make_chi2(map1, map2, er1, er2):
    #Current chi^2 definition
    return ((map1-map2)**2)/((er1**2)+(er2**2))



def relint_error(data, bg):
    #Retuirns the defined error for the relative intensity
    return (data/bg)*np.sqrt((1/data)+(alpha/bg))



def monte_carlo(yr, all, number_sim):
    '''
    creates two lists containing the sum  of randomly fluctuated chi^2 maps on using a single years background
    and the other using all of the years background. This will be used to see if the calcuation for the chi^2 used is correct by comparing 
    a model of a chi^2 distribution with the degrees of freedom = the number of unmasked pixels which would be what the 
    IceCube nuetrino detector would be able see.
    
    '''
   
    #Must append unmasked pixels to a new array to avoid having an incorrect number of degrees of freedom.
    bg_yr = np.array([])
    bg_all = np.array([])

    for pixel in mask(yr, minDec=-25*degree):
        if pixel != hp.UNSEEN:
           
            bg_yr = np.append(bg_yr, pixel)

    for pixel in mask(all, minDec=-25*degree):
        if pixel!=hp.UNSEEN:
          
            bg_all = np.append(bg_all, pixel)

    


    print(bg_yr.size, bg_all.size)
    df = bg_yr.size-1
    chi_yr = np.array([])
    chi_all = np.array([])
    #Random fluctuation maps of the original bgs
    d_yr = np.random.poisson(bg_yr)
    d_all = np.random.poisson(bg_all)
    #The relative intensity of the fluctuated data and the original
    r_yr = (d_yr - bg_yr) / bg_yr
    r_all = (d_all  - bg_all) / bg_all


    #errors for the relative intensities
    r_all_er = relint_error(d_all, bg_all)
    r_yr_er = relint_error(d_yr, bg_yr)


    
    # The Monte Carlo simulation begins
    for sim in range(0,number_sim):
        #More random fluctuations
        data = np.random.poisson(bg_yr)
        #Relative intensity is calculated using the fluctuated data
        sim_relint = (data- bg_yr)/bg_yr

        sim_er = relint_error(data, bg_yr)

        sim_yr_chi = make_chi2(sim_relint, r_yr, sim_er, r_yr_er)
        sim_all_chi = make_chi2(sim_relint, r_all, r_all_er, sim_er)
        #For individual comparisons remove sum, keep the number of simulations low in that case.
        

       


        chi_yr = np.append(chi_yr, sim_yr_chi)
        chi_all = np.append(chi_all, sim_all_chi)
    
    

    return chi_yr, chi_all



def error_bars(datamap, bgmap,ss = 100):

    #divides the bg map by the number of pixels, then multiples it by the number of nonmasked pixels
        #bg map = bg map/ sum of map pixels
        
    dataMap = datamap
   
    bgMap = bgmap


    #for data set

        #for map w/o bf

    
        #calculates power spectrum
    #y = hp.anafast(h, lmax = 3*64-1)


    #error bars
    #paramaters
    #should be the same for both maps
    npix = dataMap.size
    nside = hp.npix2nside(npix)
    lmax = 3*nside - 1
    #make fake relint maps
    #ss = sample size
    
        
    fakeCl = np.zeros((ss, lmax+1))
   
    for n in range(ss):
        
        
        relmap = np.random.poisson(dataMap)/bgMap -1
        wbgMap = mf.maskMap(bgMap, -90., -25.)
        wbgMap[wbgMap == hp.UNSEEN] = 0
    
        wbgMap /= wbgMap.sum()
        #bg map = bg map * sum of non zero pixels
        wbgMap *= (bgMap != 0).sum()
    
        w = relmap*wbgMap
        avg = np.average(w)
        h = w - avg
        fCl = hp.anafast(relmap)
        fakeCl[n] = fCl
        

    #makes arrays of fake Cls and standard deviation   
    fCl02_l, fCl16_l, fCl84_l, fCl98_l = ([] for i in range(4))
     #appends Cls to different sigmas
    for i, fCl in enumerate(fakeCl.T):
    
        fCl02 = np.percentile(fCl, 2.5)
        fCl16 = np.percentile(fCl, 16)
        fCl84 = np.percentile(fCl, 84)
        fCl98 = np.percentile(fCl, 97.5)
        
        fCl02_l += [fCl02]
        fCl16_l += [fCl16]
        fCl84_l += [fCl84]
        fCl98_l += [fCl98]

    fCl02 = np.asarray(fCl02_l)
    fCl16 = np.asarray(fCl16_l)
    fCl84 = np.asarray(fCl84_l)
    fCl98 = np.asarray(fCl98_l)
    


    #calculates standard deviation
    #for map w/ BF
    fCl02[fCl02 < 0] = 1e-11
    fCl16[fCl16 < 0] = 1e-11
    #dCl16 = y - fCl16
    #dCl84 = fCl84 - y
    #dCl = [ dCl16, dCl84 ]
    dCl = [fCl16,fCl84]
  
    return dCl



def create_directories(makefolders:bool):
    #oof
    if makefolders == True:
        try:
            os.mkdir('MapImages')
            print('MapImages dir created')
            try:
                os.mkdir('MapImages/ChiSquared')
                print('ChiSquared dir created.')
            except:
                pass
            try:
                os.mkdir('MapImages/RelativeIntensities')
                print('RelativeIntensities dir created.')
            except:
                pass
            try:
                os.mkdir('MapImages/Decompositions')
                print('Decompositions dir created.')
            except:
                pass
        except:
            try:
                os.mkdir('MapImages/ChiSquared')
                print('ChiSquared dir created.')
            except:
                pass
            try:
                os.mkdir('MapImages/RelativeIntensities')
                print('RelativeIntensities dir created.')
            except:
                pass
            try:
                os.mkdir('MapImages/Decompositions')
                print('Decompositions dir created.')
            except:
                pass
            pass
     


def decompose(map, desired_harm, n_side=64):
    #Decomposes map into a desired harmonic fit as I understand it. Entering 1 for desired_harm gives a dipole, 2 a quadrapole etc...
    a_alm = hp.map2alm(map, lmax=desired_harm, iter=10)
    b_alm = hp.map2alm(map, lmax=(desired_harm-1), iter=10)

    a_map = hp.alm2map(a_alm, nside=n_side)
    b_map = hp.alm2map(b_alm, nside=n_side)

    return a_map-b_map



def mask(skymap, minDec= -15* degree, maxDec = 70*degree, mask_value = hp.UNSEEN):
    '''
    Sets those pixels on the skymap which are out side of the IceCube mask region to UNSEEN 
    '''

    npix = skymap.size
    NSIDE = hp.npix2nside(npix)
    masked_skymap = copy.deepcopy(skymap) #What is the point of deepcopy why not just set it equal to "skymap?"

    max_pix= hp.ang2pix(NSIDE, (90*degree)-minDec, 0)

    masked_skymap[0:max_pix] = mask_value

    return masked_skymap



def print_rel_int_maps():

    for year in range(yr_range[0], yr_range[1]+1):

        rel_int_map = hp.read_map('{path}SiderealRelativeIntensities/{year}_rel_int.fits'.format(path=path_to_sidereal_fits, year=year))
        s_rel_int_map = hp.smoothing(rel_int_map, fwhm=5*degree)

        hp.mollview(mask(s_rel_int_map, minDec=-25*degree), rot=[180,0],min=-.001, max=.001,title="{year} Relative Intensity".format(year=year) ,cmap=colormap)
        hp.graticule()
        plt.savefig('MapImages/RelativeIntensities/{year}_rel_int.png'.format(year=year))
        plt.clf()
    
    all_rel_map = hp.read_map('{path}SiderealRelativeIntensities/{year}_rel_int.fits'.format(path=path_to_sidereal_fits, year="all_year"))

    s_all_map = hp.smoothing(all_rel_map, fwhm=5*degree)

    hp.mollview(mask(s_all_map, minDec=-25*degree), rot=[180,0], min=-.001, max=.001, title="All years Relative Intensity", cmap=colormap )
    hp.graticule()
    plt.savefig("MapImages/RelativeIntensities/all_rel_int.png")
    plt.clf()
    plt.close()



def power_spectrum_chi_2():
    '''
    
    Compare the relative intesity's power spectrum of each year to the All maps rel int power spectrum using the chi square method.
    
    '''
    all_map = hp.read_map(path_to_sidereal_fits+'SiderealRelativeIntensities/all_year_rel_int.fits', verbose=False)
    all_cl = hp.anafast(mask(all_map, minDec=-25*degree), lmax = 40, iter=10)
    ell = np.arange(len(all_cl))

    all_error_bars = np.array(error_bars(hp.read_map("MapData/all_data.fits"), hp.read_map('MapData/all_bg.fits')))
    print(all_error_bars)
    print(all_error_bars.size)
    all_cl_err = (all_error_bars[1,:41] - all_error_bars[0,:41])/2


    plt.plot(ell[1:], all_cl[1:],'o')
    plt.ylabel("$C_l$")
    plt.xlabel("l")
    plt.yscale('log')
    plt.ylim(10e-14,10e-8)
    plt.title("All Year Angular Power Spectrum")
    plt.savefig("MapImages/clcomp/all.png")
    plt.clf()

    #no errors yet
    chilist = []

    for year in range(yr_range[0], yr_range[1]+1):

        year_map = hp.read_map('{path_to_sidereal_fits}SiderealRelativeIntensities/{year}_rel_int.fits'.format(path_to_sidereal_fits=path_to_sidereal_fits, year=year), verbose=False)
        year_cl = hp.anafast(mask(year_map, minDec=-25*degree), lmax =40, iter=10)


        year_error_bars = np.array(error_bars(hp.read_map("MapData/IC86-{year}_24H_sid.fits".format(year = year), 0), hp.read_map("MapData/IC86-{year}_24H_sid.fits".format(year=year),1)))
  

        year_cl_err = (year_error_bars[1,:41] - year_error_bars[0,:41])/2

        

        chi2b = ((year_cl - all_cl )**2)/((all_cl_err**2)+(year_cl_err**2))

        plt.plot(ell[1:], year_cl[1:], "o")
        plt.ylabel("$C_l$")
        
        plt.xlabel("l")
        plt.yscale('log')
        plt.title('{year} Angular Power Spectrum'.format(year=year))

        plt.savefig('MapImages/clcomp/a_{year}.png'.format(year=year))
        plt.clf()


        plt.plot(ell[1:], chi2b[1:], "o")
        plt.yscale('log')

        
        plt.ylabel("$C_l$")
        plt.xlabel("l")
        plt.title('{year}, All $\chi^2$'.format(year=year))

        plt.savefig('MapImages/clcomp/chi_{year}.png'.format(year=year))
        plt.clf()

        chilist.append(chi2b[1:41])

    chilist = np.array(chilist)

    for l in range(0, 40):
        y=[]
        x= range(2011,2021)

        for i in range(0, 10):
            y.append(chilist[i,l])

        plt.plot(x,y,'o')
        plt.xlabel('Year')
        plt.title('$\chi^2$ values when l={l}'.format(l=l+1))
        plt.savefig('MapImages/clcomp/Years/{l}_yrs.png'.format(l=l))
        plt.clf()
        #Now calculating the p-values given the chi2. Assuming df=1
        yp =[]
        for val in y:
            yp.append(1-chi2.cdf(val, 1))
        
        plt.plot(x,yp,'o')
        plt.xlabel('Year')
        plt.ylabel('p-value')
        plt.title('P-values when l={l}'.format(l=l+1))
        plt.savefig('MapImages/clcomp/Years/a_pval_{l}_yrs.png'.format(l=l+1))
        plt.clf()

    x = range(2011,2021)
    lis1 =[]
    lis2 =[]
    lis3 =[]

    for i in range(0, 10):
        lis1.append(chilist[i,0])
        lis2.append(chilist[i,1])
        lis3.append(chilist[i,2])

    plt.plot(x, lis1, 'o',label = "l=1", color='b')
    plt.plot(x, lis2, 'o', label="l=2", color='g')
    plt.plot(x, lis3, 'o',label='l=3', color='r')
    plt.ylabel('$C_l$ $\chi^2 values$')
    plt.legend()
    plt.grid()
    plt.xlabel("Year")
    plt.savefig('1-3chi2.png')
    plt.clf()

    plis1=[]
    plis2=[]
    plis3=[]

    for i in range(0, 10):
        plis1.append(1-chi2.cdf(chilist[i,0],1))
        plis2.append(1-chi2.cdf(chilist[i,1],1))
        plis3.append(1-chi2.cdf(chilist[i,2],1))

    plt.plot(x, plis1, 'o',label = "l=1", color='b')
    plt.plot(x, plis2, 'o', label="l=2", color='g')
    plt.plot(x, plis3, 'o',label='l=3', color='r')
    plt.ylabel('$C_l$ p-values')
    plt.legend()
    plt.grid()
    plt.xlabel("Year")
    plt.savefig('1-3pval.png')
    plt.close()



def year_decomposition(lmax):

    for year in range(yr_range[0], yr_range[1]+1):
        map = hp.read_map('{path}SiderealRelativeIntensities/{year}_rel_int.fits'.format(path=path_to_sidereal_fits, year=year))
        for l in range(1, lmax+1):
            if year == 2011:
                a_map = hp.read_map('{path}SiderealRelativeIntensities/{year}_rel_int.fits'.format(path=path_to_sidereal_fits, year="all_year"))
                a_d_map = decompose(mask(a_map, minDec=-15*degree),l)
                a_m_d_map = mask(a_d_map, minDec=-25*degree)
                hp.mollview(a_m_d_map, rot=[180,0], cmap=colormap, title="All Years, l={l}".format( l=l))
                hp.graticule()
                plt.savefig('MapImages/Decompositions/all_l{l}.png'.format(l=l))
                plt.clf()



            d_map = decompose(mask(map, minDec=-15*degree),l)
            m_d_map = mask(d_map, minDec=-25*degree)

            hp.mollview(m_d_map, rot=[180,0], cmap=colormap, title="{year}, l={l}".format(year=year, l=l))
            hp.graticule()
            plt.savefig('MapImages/Decompositions/{year}_l{l}.png'.format(year = l, l=year))
            plt.clf()
            plt.close()

    pass



def construct_relative_intensity_fits(path_to_sidereal_fits):
    '''
    Will construct the relative intensity fits files aswell as those for the all year map.
    '''
    
    #It is likely this if statement will not work so I try.

    if os.path.isdir((path_to_sidereal_fits+'SiderealRelativeIntesities')) == False:
        try:
            os.mkdir(path_to_sidereal_fits+'SiderealRelativeIntensities')
        except:
            print('Path already exists.')
    
    
    #Creating empty arrays so the individual years can be added to them.
    all_year_data = np.zeros(npixels)
    all_year_bg = np.zeros(npixels)

    for year in range(yr_range[0],yr_range[1]+1):
        
        year_data = hp.read_map("{path}IC86-{year}_24H_sid.fits".format(path=path_to_sidereal_fits,year=year), 0)
        year_bg = hp.read_map("{path}IC86-{year}_24H_sid.fits".format(path=path_to_sidereal_fits,year=year), 1)
        year_error = (year_data/year_bg)*np.sqrt((1/year_data)+(alpha/year_bg))
        year_rel_int = (year_data-year_bg)/year_bg

        all_year_data = all_year_data + year_data
        all_year_bg = all_year_bg + year_bg

        hp.write_map('{path_to}SiderealRelativeIntensities/{year}_rel_int.fits'.format(path_to=path_to_sidereal_fits,year=year), year_rel_int, overwrite=True)
        hp.write_map('{path_to}SiderealRelativeIntensities/{year}_rel_int_er.fits'.format(path_to=path_to_sidereal_fits,year=year), year_error, overwrite=True)        

    all_year_rel_int = (all_year_data-all_year_bg)/all_year_bg
    all_year_rel_int_error = (all_year_data/all_year_bg)*np.sqrt((1/all_year_data)+(alpha/all_year_bg))

    hp.write_map('{path_to}SiderealRelativeIntensities/all_year_rel_int_error.fits'.format(path_to = path_to_sidereal_fits), all_year_rel_int_error, overwrite=True)
    hp.write_map('{path_to}SiderealRelativeIntensities/all_year_rel_int.fits'.format(path_to = path_to_sidereal_fits), all_year_rel_int, overwrite=True)
    hp.write_map('{path_to}SiderealRelativeIntensities/all_year_data.fits'.format(path_to=path_to_sidereal_fits), all_year_data, overwrite=True)
    hp.write_map('{path_to}SiderealRelativeIntensities/all_year_bg.fits'.format(path_to=path_to_sidereal_fits), all_year_bg, overwrite=True)

    print('\n\n\nConstructing Relative Intensity fits DONE.\n\n\n')



def construct_chi_2_figures():

    '''
    Creates and saves a set of skymaps and histograms.
    '''

    path_to = path_to_sidereal_fits+'SiderealRelativeIntensities/'
    nbins_all = np.linspace(0, 15)
    nbins_f = np.linspace(0,20)
    save_path = 'MapImages/ChiSquared/'
    unmasked_pix = 14208
    all_rel_int = hp.read_map('{path}all_year_rel_int.fits'.format(path=path_to))
    all_error = hp.read_map('{path}all_year_rel_int_error.fits'.format(path=path_to))

    for year in range(yr_range[0], yr_range[1]+1):

        year_rel_int = hp.read_map('{path}{year}_rel_int.fits'.format(path=path_to, year = year))
        year_error = hp.read_map('{path}{year}_rel_int_er.fits'.format(path=path_to, year = year))

        #Comparing maps to first maps now...
        if year == yr_range[0]:
            print('Chi^2 first year:')
            for yr in range(yr_range[0], yr_range[1]+1):
                syear_rel_int = hp.read_map('{path}{year}_rel_int.fits'.format(path=path_to, year = yr))
                syear_error = hp.read_map('{path}{year}_rel_int_er.fits'.format(path=path_to, year = yr))

                yr_rmse = np.sqrt(((year_rel_int.sum()-syear_rel_int.sum())**2)/unmasked_pix)

                first_chi = ((syear_rel_int - year_rel_int)**2)/((year_error**2)+(syear_error**2))

                fsmoothed_chi_map = hp.smoothing(first_chi, fwhm=5*degree)
                fmasked_s_chi_map = mask(fsmoothed_chi_map, minDec=-25*degree)

                hp.mollview(fmasked_s_chi_map, rot=[180, 0], max = 4.2, title = 'First Year , {year} - $\chi^2$'.format(year=yr),cmap=colormap)

                hp.graticule()
                plt.savefig('{path}{year}_firstyr_chi.png'.format(path=save_path, year=yr))
                plt.clf()
                plt.close()
                #Creating an array of only unmasked values so that the histogram will be accurate.
                fmasked = mask(first_chi, minDec=-25*degree)
                fy_hist_array = np.zeros(0)
                for pixel in fmasked:
                   
                    if pixel != hp.UNSEEN:
                        fy_hist_array = np.append(fy_hist_array, [pixel])

                print(yr , ": ", fy_hist_array.sum(), "     Mean: ", np.mean(fy_hist_array), "      STD: ", np.std(fy_hist_array), "        RMSE: ", f'{yr_rmse:.10f}')
                plt.hist(fy_hist_array, nbins_f)
                plt.yscale('log')
                plt.ylim(1, 10e5)
                plt.xlim(0,20)
                plt.title('First Year , {year} - $\chi^2$'.format(year=yr))
                plt.savefig('{path}first_{year}hist.png'.format(path = save_path, year=yr))
                plt.clf()
                plt.close()
            print("--------------------------","\n", 'Unmasked Pixels: ', fy_hist_array.size, "\n", "*****************************", "\n All Years:")
        
        
        chi_2_map = ((year_rel_int - all_rel_int)**2)/((year_error**2)+(all_error**2))

        rmse = np.sqrt(((all_rel_int.sum() - year_rel_int.sum())**2)/unmasked_pix)

        smoothed_chi_map = hp.smoothing(chi_2_map, fwhm=5*degree)
        masked_s_chi_map = mask(smoothed_chi_map, minDec=-25*degree)
        hp.mollview(masked_s_chi_map, rot=[180, 0], max = 4.3, title = 'All years, {year} - $\chi^2$'.format(year=year),cmap=colormap)
        hp.graticule()
        plt.savefig('{path}01_{year}_all_chi.png'.format(path=save_path, year=year))
        plt.clf()
        plt.close()
        
        masked_chi = mask(chi_2_map, minDec=-25*degree)

        all_hist_array = np.zeros(0)

        for pixel in masked_chi:
            if pixel != hp.UNSEEN:
                all_hist_array = np.append(all_hist_array, [pixel])
        print(year , ": ", all_hist_array.sum(), "      Mean: ", np.mean(all_hist_array), "     STD: ", np.std(all_hist_array),"        RMSE: ", f'{rmse:.10f}')

        plt.hist(all_hist_array, nbins_all)
        plt.axvline(np.mean(all_hist_array), label='Mean: {v}'.format(v=np.mean(all_hist_array)), color="m")
        plt.yscale('log')
        plt.ylim(1, 10e5)
        plt.xlim(0,20)
        plt.legend()
        plt.title('All years, {year} - $\chi^2$'.format(year=year))
        plt.savefig('{path}all_{year}hist.png'.format(path = save_path, year=year))
        plt.clf()
        plt.close()
    print("---------------------------\n",'Unmasked Pixels: ',all_hist_array.size)
    print("\n\n\nChi^2 figures created...\n\n\n")



def test():





    df = 13750
    chi_yr, chi_all = monte_carlo(hp.read_map('MapData/IC86-2011_24H_sid.fits',1), hp.read_map('MapData/all_bg.fits'), 1000)
    



    num_bins = 100
    bins = np.logspace(np.log10(np.amin(chi_yr)), np.log10(np.amax(chi_yr)), 500)
    #bins = np.linspace(np.amin(chi_yr), np.amax(chi_yr), 1000)
    
    weight = np.ones_like(chi_yr)/ chi_yr.sum() *num_bins
    x=np.logspace(np.log10(np.amin(chi_yr)),np.log10(np.amax(chi_yr)),1000)
    model = chi2(df)

    #x=np.linspace(np.amin(chi_yr), 15e3, 1000)


    plt.hist(chi_yr, bins, density=True)
    plt.plot(x, chi2.pdf(x, df))
    
    plt.show()



def test2():

    map1 = hp.read_map("MapData/SiderealRelativeIntensities/2011_rel_int.fits")
    map2 = hp.read_map("MapData/SiderealRelativeIntensities/2019_rel_int.fits")
    er1 = hp.read_map("MapData/SiderealRelativeIntensities/2011_rel_int_er.fits")
    er2 = hp.read_map("MapData/SiderealRelativeIntensities/2019_rel_int_er.fits")


    chisq = make_chi2(map2, map1, er1, er2)

    x=np.logspace(np.log10(np.amin(chisq)),np.log10(np.amax(chisq)),1000)
    
    bins = np.logspace(np.log10(np.amin(chisq)), np.log10(np.amax(chisq)), 200)
    plt.hist(chisq, bins, density=True)
    plt.plot(x, chi2.pdf(x, 1))
    plt.xscale('log')
    plt.yscale('log')
    plt.show()



def test5000(name):
    meanlist = []
    map1 = ani2.top_hat(hp.read_map('MapData/IC86-2016_24H_sid.fits',1), 10)
    map2 = ani2.top_hat(hp.read_map('MapData/all_bg.fits'),10)
    for i in range(0,1):
        
        chi_yr, chi_all = monte_carlo(map1,map2 , 1000)
    
        select_chi = chi_all


        num_bins = 1000
        #df = np.mean(select_chi)
        #meanlist.append(df)
        df=1
        bins = np.logspace(np.log10(1e-8), np.log10(np.amax(select_chi)), 750)
        #bins = np.logspace(np.log10(np.amin(select_chi)), np.log10(np.amax(select_chi)), 100)#for all years
        #bins = np.linspace(np.amin(select_chi)-1, np.amax(select_chi), 100)
    
        
        x=np.logspace(np.log10(1e-8),np.log10(np.amax(select_chi)),1000)
        model = chi2(df)

        #x=np.linspace(np.amin(select_chi), np.amax(select_chi), 1000)


        
    plt.hist(select_chi, bins,  density=True)
        
    plt.plot(x, chi2.pdf(x, df),linewidth=1, label='$\chi^2$ PDF where df ={df}'.format(df=df) )   
    plt.xlabel('$\chi^2 value$')
    plt.ylabel('Counts at value')
    plt.xscale('log')
    plt.title('Relative Intensity $\chi^2$ using single year BG Toy Monte Carlo\n(df = 1')
    
    plt.yscale('log')
    plt.legend()
    plt.savefig('MapImages/MCS/all10DEG.png'.format(name=name))
    plt.clf()
    plt.close
    


if __name__ == '__main__':
    #construct_relative_intensity_fits(path_to_sidereal_fits)
    #create_directories(True)
    #construct_chi_2_figures()
    #year_decomposition(10)
    #print_rel_int_maps()
    #power_spectrum_chi_2()
    test5000('6Top')
 
    print('\n\n\nALL Done')