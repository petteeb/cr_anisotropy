

from cProfile import label
from turtle import title
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm 
import random
import copy
import pandas as pd
from numpy import pi
import matplotlib.colors as pltc
from scipy.stats import chi2
from mapFunctions import mapFunctions as mf
import os

mk_dir = True #if you dont want the program to make a bunch of folders set to false.

'''
Directories for the images and map data are set here (relative to this files location for I am lazy)
'''
img_dir = 'MapImages'
data_dir = 'MapData' #IC86 24h sidereal fits files should be here.
'''
variables and stuff
'''
cmap = copy.copy(cm.turbo)
degree = np.pi/180
'''
Colors used in the histogram coloring methods
'''
t_color = ['slateblue','pink','plum','thistle','tomato','lightsalmon', 'gold', 'khaki', 'lawngreen','lightgreen' ,'springgreen','aqua','turquoise', 'skyblue', 'royalblue']#Gotta be the number of points in each iteration of t_whatever.
bad_color = tuple([0,0,0,0])
space_w =['blueviolet','magenta' ,'red', 'orange', 'yellow' ,'lime', 'forestgreen','deepskyblue']
        #  
range_w = [1.97e-9 , 3.8e-8 ,5.73e-7,6.79e-6,1/15787,1/2149, 1/370, 1/81] #p-value sigmas
repmap = pltc.ListedColormap(['blueviolet', 'magenta', 'red' ,'orange', 'yellow' ,'lime', 'forestgreen','deepskyblue']) # used strictly in the color bar as histogram colorbars look really bad.


'''
Functions.
'''

'''
Gaussian disk injection
'''

def gaussian_disk(vec, core,radius, strength):

    '''
    Vec is the vector the disks center will lie. Core is the angular radius of the area of the highest intensity. Radius is the smoothing radius for the gaussian smoothing
    procedure. Strength is the value for the stated maximum intensity. returns a map. wow.
    '''

    map  = np.zeros(hp.nside2npix(64))
    init_disk = hp.query_disc(64, vec, (core)*degree)

    for i in init_disk:

        map[i] = strength

    f_map = hp.smoothing(map, fwhm=radius*degree)
    
    final_map = np.zeros(hp.nside2npix(64)) + hp.UNSEEN

    mask_disk = hp.query_disc(64, vec, radius*degree)

    for index in mask_disk:

        final_map[index] = f_map[index]





    return final_map



def injected_map(bg_map, g_disk):
    '''
    Takes a background map and injects it with the inputted gaussian disk.
    '''
    index = 0
    map = copy.copy(bg_map)

    for pixel in g_disk:

        if pixel != hp.UNSEEN:
            map[index] = bg_map[index]*(1+pixel)

        else:
            map[index] = bg_map[index]
        index += 1

    return np.random.poisson(map)



def np_injected_map(bg_map, g_disk):

    index = 0
    map = copy.copy(bg_map)

    for pixel in g_disk:

        if pixel != hp.UNSEEN:
            map[index] = bg_map[index]*(1+pixel)

        else:
            map[index] = bg_map[index]
        index += 1

    return map




def s_optim(u_init, l_init, d_val, d_acc, data, data2, bg, bg2, dec, rad, srad):

    '''
    So The disk I make suck so this is a crappy algorythm to create a disk that roughly creates a desired p-value. Who knows how long this will take to run.
    pass it two values you know are below and above what you need and it will flip through searching for a value that falls in a specified range. or you

    '''
    iterations = 0
    value = 0
    pmap = np.array([])
    vec = hp.ang2vec(dec*degree, 180*degree)
    final_input = 0
    c_val = 0

    u = u_init
    l = l_init

    while c_val < d_val - d_acc or c_val > d_val + d_acc:

        disk_str = random.uniform(l, u)
        
        g_disk = gaussian_disk(vec, 1, rad, disk_str)
        map = np_injected_map(data,g_disk)

        chi_m =  chi2map_2(data2, bg2, map, bg, srad, 0)

        pmap = p_map(chi_m)

        c_val = np.amin(get_vals(pmap))

        
        if c_val > d_val - d_acc:

            l = disk_str

        if c_val < d_val + d_acc:

            u = disk_str
        
        if d_val - d_acc <= c_val <= d_val + d_acc:

            value = disk_str
            final_input= (u+l)/2
            break
        print('Current min P-value:', c_val, '\n lower/upper:', [l,u])
        iterations += 1



    print('Input Disk Strength:', final_input, '\nP-value:', c_val, '\nActual Strength:', np.amax(g_disk), '\nTook', iterations, 'Iterations')

    return final_input, pmap, c_val,np.amax(g_disk)


'''
Minimum points and related functions
'''
def find_min(map, n, deg):
    '''
    map = Thats going to have to be a p-value map boss.
    n = Thats going to be the number of times it searches for the lowest value.
    deg = Thats going to be the angular radius of the disk.
    Finds the lowest value in the skymap, then it querys a disk at that loaction and sets those to unseen after noting its position for that year.
    It then searches for the next lowest value, querys the disk again, up to n times.
    Hopefully useful for finding and tracking the locations of separate areas of significance over x amount of years.
    '''    
    locations = []
    values = []
    vec_list = []
    q_rad = deg*degree
    map_f = copy.copy(map)
    npix = len(map)
    nside = 64

    vec = np.transpose(hp.pix2vec(nside, np.arange(npix)))
    
    for i in range(0,n):
        vals = get_vals(map)
        min = np.amin(vals)
        index=0
        for pixel in map:

            if pixel == min:

                values.append(pixel)
                locations.append(hp.pix2vec(nside, index))
                remove_pix = hp.query_disc(nside, vec[index], q_rad)

                vec_list.append(vec[index])
                for pix in remove_pix:
                    map[pix] = hp.UNSEEN
                break
            
            
            index+=1

    for loc in vec_list:
        
        marker = hp.query_disc(nside, loc, 2*degree)

        for pixel in marker:

            map_f[pixel] = -1
        
 

    return locations, map_f



def ot_points(list_op, fyi):
    '''
    Takes a 2d array where each year contains vectors descibing the location of points on a sky map.
    The function then takes a point from the first year and finds the point closest to it in the next and so on and so forth, creating a trail of closest points.
    This is then used to plot a course on another skymap, showing a sort of path i guess. (We define the index of point we wish to follow from the first year(fyi=first year index))
    '''
    initial = list_op[0][fyi]

    pathot = []
    pathot.append(list_op[0][fyi])

    for i in range(1, len(list_op)):

        cdist = np.array([])
        cvec = []

        for q in range(0, len(list_op[i])):

            if i ==1:
                cdist = np.append(cdist, distance(initial, list_op[i][q]))
                
            else:
                
                cdist = np.append(cdist, distance(pathot[i-1], list_op[i][q]))
            
            cvec.append(list_op[i][q])
        
        min = np.argmin(cdist)
        pathot.append(cvec[min])


    return pathot


#OLD
def point_map_etc(deg_smooth):
    point_list =[]

    ref_map = hp.read_map('MapData/SmoothRI/{deg}/2011_rel_int.fits'.format(deg=deg_smooth))    
    ref_er  = hp.read_map('MapData/SmoothRI/{deg}/2011_rel_int_er.fits'.format(deg=deg_smooth))    
    for year in range(2012, 2021):

        map = hp.read_map('MapData/SmoothRI/{deg}/{year}_rel_int.fits'.format(deg=deg_smooth, year=year))    
        er  = hp.read_map('MapData/SmoothRI/{deg}/{year}_rel_int_er.fits'.format(deg=deg_smooth, year=year))   

        pmap = p_maps(ref_map, map, er, ref_er)

        locs ,e_pmap = find_min(pmap, 15,20)
        
        point_list.append(locs)

        histmap = hist_color_2(e_pmap, range_w, space_w)

        hp.mollview(e_pmap, rot=[180,0], title='{year}'.format(year=year),norm='hist' ,cmap=histmap,cbar=False )
        hp.graticule()
        plt.savefig('DSmooth/{deg}/time/{year}.png'.format(deg=deg_smooth, year=year))
        plt.clf()
        plt.close()
    
    
    for i in range(0, 15):

        vectorss = ot_points(point_list, i)
        tmap = t_map(vectorss)

        hp.mollview(tmap, rot=[180,0],cmap=color_map ,title = "P-values over time")
        plt.savefig('DSmooth/{deg}/time/min{i}.png'.format(i=i, deg=deg_smooth))
        plt.clf()
        plt.close()


#NEW
def point_maps(deg):


    point_list = []

    for year in range(2012, 2022):
        chi = chi2map(2011, year, deg, 0)
        pmap = p_map(chi)
        locs, e_pmap = find_min(pmap, 10, 20)
        point_list.append(locs)

    #Lets save point list in a text file and read and plot it over pmaps. using t_path what do you say friend?
    np.save('{deg}sigpoints.npy'.format(deg=deg),point_list)


    for i in range(0, 10):
        vectorss = ot_points(point_list, i)
        tmap = t_map(vectorss)
        hp.mollview(tmap, rot = [180, 0], cmap =cmap, title = 'P-values over time' )
        plt.ylim([-1,0])
        plt.savefig('NewChi/time/{i}.png'.format(i=i))
        plt.clf()
        plt.close()
    

#Throws up some color coded points on the current plot. they relate to minimum p-values if used as intended.
def t_path(point_list):
    

    for i in range(0,15):
        
        thetas = []
        phis = []

        vectors = ot_points(point_list, i)
   
        for vec in vectors:
       
            #Somehow the vectors for querying discs are different that what vec2ang wants? Very cool.
            theta , phi = hp.pix2ang(64, hp.vec2pix(64, vec[0],vec[1], vec[2]))
            thetas.append(theta)
            phis.append(phi)
        
        phis = np.array(phis)
        thetas = np.array(thetas)

        mphi = np.isfinite(phis)
        mtheta = np.isfinite(thetas)

        hp.projplot(thetas[mtheta], phis[mphi],linestyle='-',marker='.',color=t_color[i])
      
            
    

def t_map(pathot):
    map = np.zeros(hp.nside2npix(64))
    scale=1
    for vec in pathot:

        disk = hp.query_disc(64, vec, 2*degree)

        for i in disk:
            map[i] = 1*scale
        
        scale +=1
    
    return map




def distance(vec1, vec2):
    
    dist = np.sqrt(((vec2[0]-vec1[0])**2)+((vec2[1]-vec1[1])**2)+((vec2[2]-vec1[2])**2))
    return dist

'''
General
'''

def get_vals(map):
    vals = np.array([])
    count5 = 0
    
    for pixel in map:
        if pixel != hp.UNSEEN:
            vals = np.append(vals, pixel)
        if pixel < 5.73e-7 and pixel != hp.UNSEEN:
            count5 += 1

    return vals



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



def power_spectrum_chi_2(smoothing):
    '''
    
    Compare the relative intesity's power spectrum of each year to the reference maps rel int power spectrum using the chi square method.
    
    '''
    d_ref = top_hat(hp.read_map('MapData/IC86-2011_24H_sid.fits', 0,verbose=False), smoothing) 
    bg_ref = top_hat(hp.read_map('MapData/IC86-2011_24H_sid.fits', 1,verbose=False),smoothing)

    ref_ri = mask(rel_int(d_ref, bg_ref))
    ref_cl = hp.anafast(ref_ri, lmax = 40, iter=10)
    ell = np.arange(len(ref_cl))

    ref_error_bars = np.array(error_bars(d_ref, bg_ref))
    print(ref_error_bars)
    print(ref_error_bars.size)
    ref_cl_err = (ref_error_bars[1,:41] - ref_error_bars[0,:41])/2


    plt.plot(ell[1:], ref_cl[1:],'o')
    plt.ylabel("$C_l$")
    plt.xlabel("l")
    plt.yscale('log')
    plt.ylim(10e-14,10e-8)
    plt.title("All Year Angular Power Spectrum")
    plt.savefig("NewMapImages/clcomp/2011.png")
    plt.clf()

    #no errors yet
    chilist = []

    for year in range(2012, 2021):

        d_yr =  top_hat(hp.read_map('MapData/IC86-{year}_24H_sid.fits'.format(year=year), 0,verbose=False), smoothing)
        bg_yr =  top_hat(hp.read_map('MapData/IC86-{year}_24H_sid.fits'.format(year=year), 1,verbose=False), smoothing)

        year_map = mask(rel_int(d_yr, bg_yr))

        year_cl = hp.anafast(year_map, lmax =40, iter=10)


        year_error_bars = np.array(error_bars(d_yr, bg_yr))
  

        year_cl_err = (year_error_bars[1,:41] - year_error_bars[0,:41])/2

        

        chi2b = ((year_cl - ref_cl )**2)/((ref_cl_err**2)+(year_cl_err**2))

        plt.plot(ell[1:], year_cl[1:], "o")
        plt.ylabel("$C_l$")
        
        plt.xlabel("l")
        plt.yscale('log')
        plt.title('{year} Angular Power Spectrum'.format(year=year))

        plt.savefig('NewMapImages/clcomp/a_{year}.png'.format(year=year))
        plt.clf()


        plt.plot(ell[1:], chi2b[1:], "o")
        plt.yscale('log')

        
        plt.ylabel("$C_l$")
        plt.xlabel("l")
        plt.title('{year}, All $\chi^2$'.format(year=year))

        plt.savefig('NewMapImages/clcomp/chi_{year}.png'.format(year=year))
        plt.clf()

        chilist.append(chi2b[1:41])

    chilist = np.array(chilist)

    for l in range(0, 40):
        y=[]
        x= range(2012,2021)

        for i in range(0, 9):
            y.append(chilist[i,l])

        plt.plot(x,y,'o')
        plt.xlabel('Year')
        plt.title('$\chi^2$ values when l={l}'.format(l=l+1))
        plt.savefig('NewMapImages/clcomp/Years/{l}_yrs.png'.format(l=l))
        plt.clf()
        #Now calculating the p-values given the chi2. Assuming df=1
        yp =[]
        for val in y:
            yp.append(1-chi2.cdf(val, 1))
        
        plt.plot(x,yp,'o')
        plt.xlabel('Year')
        plt.ylabel('p-value')
        plt.yscale('log')
        plt.title('P-values when l={l}'.format(l=l+1))
        plt.savefig('NewMapImages/clcomp/Years/{smoothing}a_pval_{l}_yrs.png'.format(l=l+1, smoothing=smoothing))
        plt.clf()

    x = range(2012,2021)
    lis1 =[]
    lis2 =[]
    lis3 =[]

    for i in range(0, 9):
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
    plt.savefig('{smoothing}1-3chi2.png'.format(smoothing=smoothing))
    plt.clf()

    plis1=[]
    plis2=[]
    plis3=[]

    for i in range(0, 9):
        plis1.append(1-chi2.cdf(chilist[i,0],1))
        plis2.append(1-chi2.cdf(chilist[i,1],1))
        plis3.append(1-chi2.cdf(chilist[i,2],1))

    plt.plot(x, plis1, 'o',label = "l=1", color='b')
    plt.plot(x, plis2, 'o', label="l=2", color='g')
    plt.plot(x, plis3, 'o',label='l=3', color='r')
    plt.ylabel('$C_l$ p-values')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.xlabel("Year")
    plt.savefig('{smoothing}1-3pval.png'.format(smoothing=smoothing))
    plt.close()



def monte_carlo(yr, number_sim):
    '''
    creates two lists containing the sum  of randomly fluctuated chi^2 maps on using a single years background
    and the other using all of the years background. This will be used to see if the calcuation for the chi^2 used is correct by comparing 
    a model of a chi^2 distribution with the degrees of freedom = the number of unmasked pixels which would be what the 
    IceCube nuetrino detector would be able see.
    
    '''
   
    #Must append unmasked pixels to a new array to avoid having an incorrect number of degrees of freedom.

    bg_yr=mask(top_hat(copy.copy(yr),20))
    


    print(bg_yr.size)

    chi_yr = np.array([])
    
    #Random fluctuation maps of the original bgs
    d_yr = np.array([])

    for pixel in bg_yr:
        if pixel != hp.UNSEEN:
            d_yr = np.append(d_yr, np.random.poisson(pixel))
        else:
            d_yr = np.append(d_yr, pixel)
   
    #The relative intensity of the fluctuated data and the original
    r_yr = mask((d_yr - bg_yr) / bg_yr)
    


    

    
    # The Monte Carlo simulation begins
    for sim in range(0,number_sim):
        #More random fluctuations
        data = np.array([])
        for pixel in bg_yr:
            if pixel != hp.UNSEEN:
                data = np.append(data, np.random.poisson(pixel))
            else:
                data = np.append(data, pixel)
        #Relative intensity is calculated using the fluctuated data
        sim_relint = mask((data- bg_yr)/bg_yr)

        

        sim_yr_chi = mask(bg_yr*(((sim_relint - r_yr )**2)/(r_yr+sim_relint+2)))
        #For individual comparisons remove sum, keep the number of simulations low in that case.
    
        chival = np.array([])

        for pixel in sim_yr_chi:
            if pixel != hp.UNSEEN:
                chival = np.append(chival, pixel)



        chi_yr = np.append(chi_yr, chival)
    
    

    return chi_yr



def sigmap(pmap, bounds, colors, name):
    '''
    Overlays maps such that different levels of significance can be displayed clearly. RIP histogram dream. You will be missed.
    '''
    bad_color = tuple([0,0,0,0])
    out_cm = copy.copy(cm.gist_gray)
    rot=[180,0]
    # out of bounds will now be a nice gray color. very nice.
    hp.mollview(pmap, rot=rot, cmap=out_cm, cbar=False)

    for i in range(1, len(bounds)+1):
        s_map = np.array([])

        if bounds[-i] == bounds[0]:
            
            for pix in pmap:
                if pix <= bounds[0] and pix != hp.UNSEEN:
                    s_map = np.append(s_map, pix)
                else:
                    s_map = np.append(s_map, hp.UNSEEN)
                
            hp.mollview(s_map, rot=rot, cmap=pltc.ListedColormap(colors[0]),reuse_axes=True,title=name, badcolor=bad_color, cbar=False)

       
        else:
       
            for pix in pmap:
                if pix <= bounds[-i] and pix > bounds[-i-1]:
                    s_map  = np.append(s_map, pix)
                
                else:
                    s_map = np.append(s_map, hp.UNSEEN)
            
            hp.mollview(s_map, rot=rot, cmap=pltc.ListedColormap(colors[-i]) ,reuse_axes=True, badcolor= bad_color, cbar=False)

        s_map = np.array([])


    hp.graticule()
    val_list = np.array([])
    plt.ylim([-1,0])

    for val in pmap:
            if val!= hp.UNSEEN:
                    val_list = np.append(val_list, val)

    cbar = plt.colorbar(cm.ScalarMappable(cmap=repmap),location='bottom', aspect = 70, shrink=0.5, boundaries = [ 0,1/8,2/8, 3/8,4/8,5/8,6/8, 7/8, 1])    
    cbar.ax.set_xticklabels(["","$6\sigma$",'$5.5\sigma$' ,'$5\sigma$', '4.5$\sigma$','$4\sigma$' ,'$3.5\sigma$','3$\sigma$','$2.5\sigma$'])
    cbar.set_label('minimum p-value: {:.4e}'.format(np.amin(val_list)))



def top_hat(map, d):
    '''
    Takes a map and desired degrees and smooths the map

    '''
    npix  = len(map)
    nside = hp.npix2nside(npix)
    smooth_rad = d * pi/180.
    smooth_map = np.zeros(npix)
 
    vec = np.transpose(hp.pix2vec(nside, np.arange(npix)))
    for i in range(npix):
        neighbors = hp.query_disc(nside, vec[i], smooth_rad)
        smooth_map[i] += map[neighbors].sum()


    return smooth_map



def rel_int(data, bg):

    return (data-bg) /bg



def mask(skymap, minDec= -25* degree, maxDec = 70*degree, mask_value = hp.UNSEEN):
    '''
    Sets those pixels on the skymap which are out side of the IceCube mask region to UNSEEN 
    '''

    npix = skymap.size
    NSIDE = hp.npix2nside(npix)
    masked_skymap = copy.deepcopy(skymap) #What is the point of deepcopy why not just set it equal to "skymap?"

    max_pix= hp.ang2pix(NSIDE, (90*degree)-minDec, 0)

    masked_skymap[0:max_pix] = mask_value

    return masked_skymap



def make_fit(map, l_max, maxl):
    #Makes a multipole fit. l_max is the max l of the alm map, and maxl is the multipole fits maxl, sorry.
    a_map = hp.map2alm(map, l_max)
    fit = np.array([])
    for m in range(0, maxl+1):

        for l in range(m, maxl+1):

            index = hp.Alm.getidx(l_max, l, m)

            fit    = np.append(fit, a_map[index])
           
    fit_m = hp.alm2map(fit, 64)

    return fit_m



def chi2map(ref_yr, t_yr,smoothing ,mps):
    '''
    returns a chi2 map given a reference year and a test year. Later will include an option to subtract multipoles hende mps(multipole subtraction)
    '''


    r_data = (hp.read_map('{data_dir}/IC86-{ref_yr}_24H_sid.fits'.format(data_dir=data_dir, ref_yr=ref_yr), 0))
    r_bg = (hp.read_map('{data_dir}/IC86-{ref_yr}_24H_sid.fits'.format(data_dir=data_dir, ref_yr=ref_yr), 1))
    t_data = (hp.read_map('{data_dir}/IC86-{ref_yr}_24H_sid.fits'.format(data_dir=data_dir, ref_yr=t_yr), 0))
    t_bg = (hp.read_map('{data_dir}/IC86-{ref_yr}_24H_sid.fits'.format(data_dir=data_dir, ref_yr=t_yr), 1))



    #Smooths maps if smoothing is desired.
    
    if smoothing!=0:

        r_data = top_hat(r_data, smoothing)
        r_bg = top_hat(r_bg, smoothing)
        t_data =top_hat(t_data, smoothing)
        t_bg = top_hat(t_bg, smoothing)
    
    '''
    if smoothing!=0:
        r_data = hp.smoothing(r_data, fwhm=smoothing*degree)
        r_bg = hp.smoothing(r_bg, fwhm=degree*smoothing)
        t_data =hp.smoothing(t_data, fwhm=degree*smoothing)
        t_bg = hp.smoothing(t_bg, fwhm =degree*smoothing)
    '''
    #Makes relative intesnisty maps
    r_rel_int = rel_int(r_data,r_bg)
    t_rel_int = rel_int(t_data,t_bg)

    #Subracts multipole fits if it is desired.
    if mps == 0:
        map = t_bg*(((t_rel_int-r_rel_int)**2)/(r_rel_int+t_rel_int+2))

    else:
        fit_r = make_fit(r_rel_int, 100, mps)
        fit_t = make_fit(t_rel_int, 100, mps)
        r_rel_int = r_rel_int -fit_r
        t_rel_int = t_rel_int -fit_t

        map = t_bg*(((t_rel_int-r_rel_int)**2)/(r_rel_int+t_rel_int+2))

    return mask(map)



def p_map(chi2map):

    pmap =np.array([])

    for pixel in chi2map:
        if pixel == hp.UNSEEN:
            pmap =np.append(pmap, pixel)

        else:
            p_pix = 1-chi2.cdf(pixel, 1)
            pmap = np.append(pmap, p_pix)

    return pmap



def chi2map_2(ref_yr_d,ref_yr_bg, t_yr_d, t_yr_bg,smoothing ,mps):
    '''
    returns a chi2 map given a reference year and a test year maps. Later will include an option to subtract multipoles hende mps(multipole subtraction)
    '''


    r_data = ref_yr_d
    r_bg = ref_yr_bg
    t_data = t_yr_d
    t_bg = t_yr_bg



    #Smooths maps if smoothing is desired.
    if smoothing!=0:

        r_data = top_hat(r_data, smoothing)
        r_bg = top_hat(r_bg, smoothing)
        t_data =top_hat(t_data, smoothing)
        t_bg = top_hat(t_bg, smoothing)

    #Makes relative intesnisty maps
    r_rel_int = rel_int(r_data,r_bg)
    t_rel_int = rel_int(t_data,t_bg)

    #Subracts multipole fits if it is desired.
    if mps == 0:
        map = t_bg*(((t_rel_int-r_rel_int)**2)/(r_rel_int+t_rel_int+2))

    else:
        fit_r = make_fit(r_rel_int, 100, mps)
        fit_t = make_fit(t_rel_int, 100, mps)
        r_rel_int = r_rel_int -fit_r
        t_rel_int = t_rel_int -fit_t

        map = t_bg*(((t_rel_int-r_rel_int)**2)/(r_rel_int+t_rel_int+2))

    return mask(map)

'''
Just some stuff.
'''

def main():
    deg = 20

    minplist = []

    for year in range(2012,2022):

        map = chi2map(2011,year, deg, 0)
        pmap = p_map(map)
        vals = np.array(get_vals(pmap))

        minplist.append(np.sum(vals))
        #sigmap(pmap=pmap, bounds=range_w, colors=space_w, name = 'P-values,{deg} degrees of smoothing, {year}\n(Minus l=7  fits))'.format(year=year, deg=deg))
        
        hp.mollview(pmap, rot=[180, 0], title='P-Value, {year}'.format(year=year), cmap=cmap)
        hp.graticule()
        plt.ylim([-1,0])
        
        plt.savefig('NewChi/{deg}pval_{year}.png'.format(year=year, deg=deg))
        plt.clf()
        plt.close()


    plt.scatter(range(2012,2022), minplist)
    plt.title('Minimum P-values')
    plt.ylabel('P-value')
    plt.xlabel('Year')
    plt.yscale('log')
    plt.show()



    pass



def main2():

    d_ref = hp.read_map('MapData/tyan/a18part1m_signal.fits')*91
    bg_ref = hp.read_map('MapData/tyan/a18part1m_bg.fits')*91
    d_test = hp.read_map('MapData/tyan/ma27a_signal.fits')*366
    bg_test = hp.read_map('MapData/tyan/ma27a_bg.fits')*366 

    relint = (d_ref - bg_ref)/ bg_ref

    hp.mollview(mask(relint))
    vals= get_vals(mask(d_ref))
    print(vals.sum())
    chi = chi2map_2(d_ref, bg_ref, d_test, bg_test, 60, 0)
    pmap = p_map(chi)
    hp.orthview(pmap, rot=[180,90])
    #hp.mollview(pmap, rot=[180,0], cmap=cm.seismic)
    sigmap(pmap=pmap, bounds=range_w, colors=space_w, name = '')
    plt.show()



def simulationfig():

    map = hp.read_map('MapData/IC86-2013_24H_sid.fits', 1)
    
    chiy = monte_carlo(map, 300)

    df=1
    bins = np.logspace(np.log10(1e-8), np.log10(np.amax(chiy)), 750)
    #bins = np.logspace(np.log10(np.amin(select_chi)), np.log10(np.amax(select_chi)), 100)#for all years
    #bins = np.linspace(np.amin(chiy)-1, np.amax(chiy), 100)
    
    #x=np.linspace(np.amin(chiy), np.amax(chiy), 1000)
    x=np.logspace(np.log10(1e-8),np.log10(np.amax(chiy)),1000)
    model = chi2(df)


    plt.hist(chiy, bins,  density=True)
        
    plt.plot(x, chi2.pdf(x, df),linewidth=1, label='$\chi^2$ PDF where df ={df}'.format(df=df) )   
    plt.xlabel('$\chi^2 value$')
    plt.ylabel('Counts at value')
    plt.xscale('log')
    plt.title('Relative Intensity $\chi^2$ using single year BG Toy Monte Carlo\n(df = 1)')
    
    plt.yscale('log')
    plt.legend()
    plt.show()



def gaussian_disk_injection2(smooth, disk_size, disk_str, dec, name,data, data2, bg):


    try:
        os.mkdir('MapImages/DiskSim/a_{name}'.format(name=name))

    except:
        pass

   

    vec = hp.ang2vec(dec*degree, 180*degree)
   
    g_disk = gaussian_disk(vec, 1, disk_size, disk_str)
    

    hp.mollview(g_disk, rot=[180,0], title='Disk Used')
    plt.savefig('MapImages/DiskSim/a_{name}/Disk1.png'.format(name=name))
    plt.clf()
    
    p_list = np.array([])
    
    i_data = np_injected_map(data, g_disk)

        

    chi = chi2map_2(i_data,bg, data2, bg, smooth, 0)

    pmap = p_map(chi)

    p_list = np.amin(get_vals(pmap))
    
    sigmap(pmap, range_w, space_w, 'Injection')

    plt.savefig('MapImages/DiskSim/a_{name}/iter.png'.format(name=name))
    plt.clf()
    plt.close()
   
    
    return np.amax(g_disk),p_list



def gaussian_disk_injection(smooth, disk_size, disk_str, dec, name):


    try:
        os.mkdir('MapImages/DiskSim/{name}'.format(name=name))

    except:
        pass

    bg = hp.read_map('MapData/IC86-2016_24H_sid.fits', 1)

    vec = hp.ang2vec(dec*degree, 180*degree)
   
    g_disk = gaussian_disk(vec, 1, disk_size, disk_str)
    

    hp.mollview(g_disk, rot=[180,0], title='Disk Used')
    plt.savefig('MapImages/DiskSim/{name}/Disk1.png'.format(name=name))
    plt.clf()

    data2 = np.random.poisson(bg)

    p_list = np.array([])

    for i in range(0, 10):

        data = injected_map(bg, g_disk)

        

        chi = chi2map_2(data,bg, data2, bg, smooth, 0)

        pmap = p_map(chi)

        p_list = np.append(p_list, np.amin(get_vals(pmap)))
       
        sigmap(pmap, range_w, space_w, 'Injection')

        plt.savefig('MapImages/DiskSim/{name}/{i}iter.png'.format(name=name, i=i))
        plt.clf()
        plt.close()
    
    
    return np.amax(g_disk),np.average(p_list)
    


def disk_analysis_np():
    # for making a spread sheet

    dic = {
            'Disk_Max':[],
            'Disk_Radius':[],
            'Smoothing_Radius':[],
            'Declination':[],
            'Min_P_val':[]
        }
    bg = hp.read_map('MapData/IC86-2016_24H_sid.fits', 1)
    data = np.random.poisson(bg)
    data2 = np.random.poisson(bg)
    for strength in [1e-3,5e-3,1e-2,3e-2,5e-2,7e-2,8e-2,1e-1,5e-1,1]:

        for radius in [5,10,20,30,40]:

            for smooth in  [20]:

                for degree in range(120,190, 10):

                    disk_val, avg_p = gaussian_disk_injection2(smooth,radius, strength, degree, 'str{str}rad{rad}smt{smt}dec{dec}'.format(
                        str=strength, rad=radius, smt=smooth, dec = degree), data, data2, bg)
                    dic['Min_P_val'].append(avg_p)
                    dic['Declination'].append(degree)
                    dic['Disk_Max'].append(disk_val)
                    dic['Disk_Radius'].append(radius)
                    dic['Smoothing_Radius'].append(smooth)

    df = pd.DataFrame(dic)
    df.to_csv('np_disksimvals.csv')


def test():
    dic = {
            'Disk_Max':[],
            'Input':[],
            'Disk_Radius':[],
            'Smoothing_Radius':[],
            'Declination':[],
            'Min_P_val':[]
        }
    bg = hp.read_map('MapData/IC86-2013_24H_sid.fits', 1)
    bg2 = hp.read_map('MapData/IC86-2011_24H_sid.fits', 1)
    data2 = hp.read_map('MapData/IC86-2011_24H_sid.fits', 0)
    data = hp.read_map('MapData/IC86-2013_24H_sid.fits', 0)
    l = 1e-100
    u = 1
    i =0 

    tol =  [0.005e-5, 0.05e-7, 0.05e-9]
    nam = [4,5,6]
    for sig in [6.3343257e-5, 5.73e-7, 1.97e-09]:

        dic = {
            'Disk_Max':[],
            'Input':[],
            'Disk_Radius':[],
            'Smoothing_Radius':[],
            'Declination':[],
            'Min_P_val':[]
        }

        for disk_rad in [5, 10, 20,30,40,60 ]:

            for dec in range(120, 190, 10):
                disk_str, pmap, c_val, a_s = s_optim(u, l, sig, tol[i], data, data2, bg,bg2, dec, disk_rad, 20)
                dic['Min_P_val'].append(c_val)
                dic['Declination'].append(dec)
                dic['Input'].append(disk_str)
                dic['Disk_Max'].append(a_s)
                dic['Disk_Radius'].append(disk_rad)
                dic['Smoothing_Radius'].append(20)

                sigmap(pmap, range_w, space_w, 'Disk Injection for {sig} sigma\nInjected into 2013 sidereal skymap'.format(sig=nam[i]))
                plt.savefig('Disk/data_{sig}disk{rad}sig{dec}.png'.format(sig=sig, rad=disk_rad, dec=dec))
                plt.clf()
                plt.close()
        i+=1

        df = pd.DataFrame(dic)
        df.to_csv('data_{rad}deg{sig}.csv'.format(rad=disk_rad, sig=sig))
    
def RIMAPS():
    yearcount = []
    for year in range(2011, 2022):
        data = (hp.read_map('MapData/IC86-{ref_yr}_24H_sid.fits'.format(data_dir=data_dir, ref_yr=year), 0))
        bg = (hp.read_map('MapData/IC86-{ref_yr}_24H_sid.fits'.format(data_dir=data_dir, ref_yr=year), 1))
        '''
        data = top_hat((data), 5)
        bg = top_hat((bg), 5)


        relint = mask(rel_int(data, bg))

        hp.mollview(relint,min=-1e-3, max=1e-3 ,rot=[180, 0], title='Relative Intensity {year}'.format(year=year), cmap=cmap)
        hp.graticule()
        plt.ylim([-1,0])
        plt.savefig('{year}relint.png'.format(year=year))
        plt.clf()
        plt.close()
        '''
        yearcount.append(np.sum(data))
        print('Year:', year, 'Data:', np.sum(data), 'BG:',np.sum(bg))
    print(np.sum(np.array(yearcount)))


def disk_analysis():
    # for making a spread sheet

    dic = {
            'Disk_Max':[],
            'Disk_Radius':[],
            'Smoothing_Radius':[],
            'Declination':[],
            'Average_Min_P_val':[]
        }
    
    for strength in [1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1]:

        for radius in [5,10,20,30]:

            for smooth in  [5,10,20]:

                for degree in range(120,190, 10):

                    disk_val, avg_p = gaussian_disk_injection(smooth,radius, strength, degree, 'str{str}rad{rad}smt{smt}dec{dec}'.format(
                        str=strength, rad=radius, smt=smooth, dec = degree
                    ))
                    dic['Average_Min_P_val'].append(avg_p)
                    dic['Declination'].append(degree)
                    dic['Disk_Max'].append(disk_val)
                    dic['Disk_Radius'].append(radius)
                    dic['Smoothing_Radius'].append(smooth)

    df = pd.DataFrame(dic)
    df.to_csv('disksimvals.csv')
        


def disk_graphs():

    for i in [4,5,6]:

        df = pd.read_csv('{i}sig.csv'.format(i=i), usecols=['Disk_Max', 'Disk_Radius', 'Declination'])
        df2 = pd.read_csv('data_{i}sig.csv'.format(i=i), usecols=['Disk_Max', 'Disk_Radius', 'Declination'])


        for slice in range(6, 48, 7):
            sdf = df2[slice-6:slice+1]
            plt.plot(-(sdf['Declination']-90),sdf['Disk_Max'],label='{rad}$\degree$'.format(rad=sdf.iloc[1]['Disk_Radius']),marker='.')

        plt.title('Disk Strength Required for {sig} Sigma p-values\n (Using a 2011 v 2013 $Chi^2$)'.format(sig=i))

        plt.ylabel('Max Disk Strength')
        plt.xlabel('Declination')
        plt.yscale('log')
        plt.legend(fontsize='xx-small', loc='upper left', ncol=1)
        
        plt.savefig('DiskGraphs/data_{sig}graph.png'.format(sig=i))
        plt.clf()
        plt.close()


if __name__== '__main__':
    
    RIMAPS() 
    print('done')
