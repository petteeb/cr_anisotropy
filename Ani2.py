
import AnisotropyAnalysis as ana
import healpy as hp
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib import cm
from mapFunctions import mapFunctions as mf
import matplotlib.colors as pltc
import os, glob, time
from numpy import sqrt, pi
import scipy.optimize as opt
import scipy.stats.distributions as sd
import pandas as pd

'''
notes

15 

'''

color_map = copy.copy(cm.turbo)
degree = np.pi /180
m_region = -25*degree

space_w = ['magenta','dodgerblue','red', 'orange', 'yellow' ,'lime', 'forestgreen']
range_w = [0,5.73e-7, 6.795e-6, 6.334e-5 ,4.65e-4,2.7e-3,1.23e-2]

repmap = pltc.ListedColormap(['dodgerblue', 'red' ,'orange', 'yellow' ,'lime', 'forestgreen'])




def make_fit(map, l_max, maxl):
    #As the darn thing tries to pixelize past lmax=~26
    a_map = hp.map2alm(map, l_max)
    fit = np.array([])
    for m in range(0, maxl+1):

        for l in range(m, maxl+1):

            index = hp.Alm.getidx(l_max, l, m)

            fit    = np.append(fit, a_map[index])
           
    fit_m = hp.alm2map(fit, 64)

    return fit_m



def get_vals(map):
    vals = np.array([])
    count5 = 0
    
    for pixel in map:
        if pixel != hp.UNSEEN:
            vals = np.append(vals, pixel)
        if pixel < 5.73e-7 and pixel != hp.UNSEEN:
            count5 += 1

    return vals, count5



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
    q_rad = deg*ana.degree
    map_f = copy.copy(map)
    npix = len(map)
    nside = 64

    vec = np.transpose(hp.pix2vec(nside, np.arange(npix)))
    
    for i in range(0,n):
        vals, nouse = get_vals(map)
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
        
        marker = hp.query_disc(nside, loc, 2*ana.degree)

        for pixel in marker:

            map_f[pixel] = -1
        
 

    return locations, map_f



def hist_color(map, upto):
    
    
    inside_r = 0
    outside_r = 0
    for value in map:
        if value!=hp.UNSEEN and value<=upto:
            inside_r += 1
        elif value > upto:
            outside_r+=1
            
    out_c = cm.get_cmap('gist_gray', outside_r)
    colors = []
    for pix in range(0,inside_r):
         colors.append('lime')
            
    for color in out_c(range(0,outside_r)):
        colors.append(color)
    
    
    custom = pltc.ListedColormap(colors)
    return custom



def hist_color_2(map, list_of_bounds:list, color_list:list):
    '''
    Histogram weighted color map. list of bounds and color list must be equal length, values larger than last value
    are grayscale.
    
    '''
    colors = []
    
    for i in range(0, len(list_of_bounds)):
        
        #Lowest values are first element in color
        if i == 0:
            for pixel in map:
                
                if pixel < list_of_bounds[0] and pixel != hp.UNSEEN:
                
                    colors.append(color_list[0])
                
         
            
        #Inbetween values are specified colors
        else :
            
            for pixel in map:
                if pixel >= list_of_bounds[i-1] and pixel < list_of_bounds[i]:
                    colors.append(color_list[i])
            
    counter = 0
            
    for pixel in map:
        if pixel >= list_of_bounds[-1]:
            counter +=1
                    
    out_range = cm.get_cmap('gist_gray', counter)
            
    for color in out_range(range(0,counter)):
        colors.append(color)
    #Return nice color map weighted like a histogram with specified bins ;-)
    c_map = pltc.ListedColormap(colors)       
 
    return c_map



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



def p_maps(map1, map2, er1, er2):
    '''
    Using the first map as the expected values, will output a map where every pixel represents the p-value
    Expecting Relative intensity maps.
    
    '''


    p_map = np.array([])

    chi_map = ana.mask(ana.make_chi2(map1, map2, er1, er2), minDec=m_region)

    for pixel in chi_map:

        if pixel == hp.UNSEEN:
            p_map = np.append(p_map, pixel)

        else:
            p_pix = 1-chi2.cdf(pixel, 1)
            p_map = np.append(p_map, p_pix)

    return p_map



def smooth_relint(data,bg,d):

    s_data = top_hat(data, d)
    s_bg = top_hat(bg, d)

    er = ana.relint_error(s_data, s_bg)
    
    return (s_data - s_bg)/s_bg , er



def make_smooth_relint(d):
    '''
    
    
    '''
    try:
        os.mkdir('MapData/SmoothRI/{d}'.format(d=d))
    except:
        pass

    for i in range(2011, 2021):

        data = hp.read_map('MapData/IC86-{yr}_24H_sid.fits'.format(yr=i), 0)
        bg = hp.read_map('MapData/IC86-{yr}_24H_sid.fits'.format(yr=i), 1)

        relint, er = smooth_relint(data, bg, d)

        hp.write_map('MapData/SmoothRI/{d}/{yr}_rel_int.fits'.format(yr=i, d=d), relint, overwrite=True)
        hp.write_map('MapData/SmoothRI/{d}/{yr}_rel_int_er.fits'.format(yr=i, d=d), er, overwrite=True)



def chi_figures(deg):
    expected = hp.read_map('MapData/SmoothRI/2011_rel_int.fits')
    ex_er = hp.read_map('MapData/SmoothRI/2011_rel_int_er.fits')

    for year in range(2012, 2021):

        data = hp.read_map('MapData/SmoothRI/{year}_rel_int.fits'.format(year=year))
        data_er = hp.read_map('MapData/SmoothRI/{year}_rel_int_er.fits'.format(year=year))

        chi = ana.make_chi2(data, expected, ex_er, data_er)

        hp.mollview(ana.mask(chi, minDec=-25*degree), rot=[180,0], cmap=color_map, title = '2011 v {year} $\chi^2$'.format(year=year), unit='$\chi^2$ value')

        hp.graticule()

        plt.savefig('DSmooth/{deg}/{year}.png'.format(year=year, deg=deg))
        plt.clf()
        plt.close()



def make_pmaps(deg, map_path, name):
    list_o_sum =[]
    ex_yr =2011
    expected = hp.read_map('{map_path}/{yr}_rel_int.fits'.format(yr=ex_yr, map_path=map_path, deg=deg))
    exp_er = hp.read_map('MapData/SmoothRI/{deg}/{yr}_rel_int_er.fits'.format(yr=ex_yr, deg=deg))
    
    valex =0
    

    for pix in expected:
            if pix!=hp.UNSEEN:
                valex+=pix
    list_o_sum.append(valex)
    #expected = ana.decompose(expected, 2)
    min_p =[]
    p_sum = np.array([])
    for year in range(ex_yr+1, 2021):

        sign = ''

        

        map2 = hp.read_map('{map_path}/{yr}_rel_int.fits'.format(yr=year,deg=deg ,map_path=map_path))
        map2_er = hp.read_map('MapData/SmoothRI/{deg}/{yr}_rel_int_er.fits'.format(yr=year, deg=deg))

       
        val2 = 0

        for pix in map2:
            if pix!=hp.UNSEEN:
                val2 +=pix
        
        list_o_sum.append(val2)

        if val2> valex:
            sign = '+'
        else:
            sign = '-'
        #map2 = ana.decompose(map2, 2)

        p_map = p_maps(expected, map2, map2_er, exp_er)

        
        val_list = np.array([])


        for val in p_map:
            if val!= hp.UNSEEN:
                    val_list = np.append(val_list, val)

        min_p.append([year, np.amin(val_list)])

        bins = np.logspace(np.log10(np.amin(val_list)), np.log10(np.amax(val_list)), 100)
        p_sum = np.append(p_sum, np.amin(val_list))


        h_color = hist_color_2(p_map,range_w, space_w)

        hp.mollview(p_map, rot=[180,0], cmap=h_color, norm='hist', title='P_value Map for {exp} and {map}'.format(exp=ex_yr, map=year), cbar=False)
        hp.graticule()

        cbar = plt.colorbar(cm.ScalarMappable(cmap=repmap),location='bottom', aspect = 20, shrink=0.5, boundaries = [0,1/6,2/6,3/6,4/6,5/6,1])


        
        cbar.ax.set_xticklabels(["", '$5\sigma$', '4.5$\sigma$','$4\sigma$' ,'$3.5\sigma$','3$\sigma$','$2.5\sigma$'])
        cbar.set_label('minimum p-value: {:.4e}'.format(np.amin(val_list)))

       

        plt.savefig('DSmooth/{deg}/{name}_pmap{year}.png'.format(year=year,name=name ,deg=deg))
        plt.clf()

       
        
    plt.plot(range(2012,2021), p_sum, 'o')
    #plt.hist(val_list, 00)
    #plt.title('P-value Histogram for {yr}'.format(yr=year))
    #plt.xscale('log')
    #plt.yscale('log')
    plt.title('Minimum P-value\n {deg} degrees of smoothing'.format(deg=deg))
    plt.xlabel('Year')
    plt.ylabel('Minimum P-value')
    plt.yscale('log')
    plt.savefig('DSmooth/Degrees/logmin{deg}psum.png'.format(year=year, deg=deg))
    plt.clf()

    plt.close()
    print(list_o_sum)
    return min_p



def degree_ana():
        p_list = []

        for deg in range(1, 21):

            make_smooth_relint(deg)
            #chi_figures(deg)
            p_list.append([deg,make_pmaps(deg)])

            print(deg, 'Finished')
        print(p_list)
        with open(r'/home/bpettee/Documents/list.txt', 'w') as fp:
            fp.write(str(p_list))
        fp.close()
        print('DONE')



def make_small_scale_maps(dir:str,deg ,fit_lmax:int):
    try:
        os.mkdir('MapData/{dir}'.format(dir=dir))
        print('Made')
    except:
        print('no')
        pass
    

    try:
        os.mkdir('MapImages/{dir}'.format(dir=dir))
    except:
        pass


    for year in range(2011,2021):

        raw_map = ana.mask(hp.read_map('MapData/SmoothRI/{deg}/{year}_rel_int.fits'.format(year=year, deg=deg)), minDec=-25*ana.degree)

        fit = make_fit(raw_map, 100, fit_lmax)

        map =  raw_map - fit

        hp.mollview(map, rot=[180,0], cmap=color_map, title='{year} Relative Intensity'.format(year=year))
        hp.graticule()

        plt.savefig('MapImages/{dir}/{year}_rel_int.png'.format(dir=dir, year=year))
        plt.clf()
        plt.close()

        hp.write_map('MapData/{dir}/{year}_rel_int.fits'.format(dir=dir, year=year),map, overwrite=True)



def p_listex():
    
    '''
    Makes a csv file containing minmum pvalues for each degree of smoothing, removed spherical harmonic.
    lmax =1 is a stand in for removing no components!
    '''
    #4 lists to make a dataframe because right now i am too lazy to mess with any indexes
    years = []

    minp  = []

    degs = []

    ls = []

    count5 =[]
    
    for degree in range(1, 21):
        
        for l in range(1, 6):
            if l==1:
                ex_map = hp.read_map('MapData/SmoothRI/{degree}/2011_rel_int.fits'.format(degree=degree, l=l))
                ex_er =  hp.read_map('MapData/SmoothRI/{degree}/2011_rel_int_er.fits'.format(degree=degree))
            else:
                ex_map = hp.read_map('MapData/{degree}DegMinusl{l}/2011_rel_int.fits'.format(degree=degree, l=l))
                ex_er =  hp.read_map('MapData/SmoothRI/{degree}/2011_rel_int_er.fits'.format(degree=degree))

            for year in range(2012, 2021):
                if l==1:
                    map = hp.read_map('MapData/SmoothRI/{degree}/{year}_rel_int.fits'.format(degree=degree, l=l, year=year))
                    er = hp.read_map('MapData/SmoothRI/{degree}/{year}_rel_int_er.fits'.format(degree=degree, year=year))
                else:
                    map = hp.read_map('MapData/{degree}DegMinusl{l}/{year}_rel_int.fits'.format(degree=degree, l=l, year=year))
                    er = hp.read_map('MapData/SmoothRI/{degree}/{year}_rel_int_er.fits'.format(degree=degree, year=year))

                pmap = p_maps(ex_map, map, ex_er, er)
                pval, count = get_vals(pmap)

                degs.append(degree)
                ls.append( l)
                count5.append(count)
                years.append( year)
                minp.append(np.amin(pval))


    minpdict = {
                'Degree of Smoothing':degs,
                'Max l of removed fit':ls,
                'Year':years,
                '#5sig':count5,
                'Minimum p-value':minp
                }
    

    data = pd.DataFrame(minpdict)
    data.to_csv('minp.csv')



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



def distance(vec1, vec2):
    
    dist = sqrt(((vec2[0]-vec1[0])**2)+((vec2[1]-vec1[1])**2)+((vec2[2]-vec1[2])**2))
    return dist



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



def t_map(pathot):
    map = np.zeros(hp.nside2npix(64))
    scale=1
    for vec in pathot:

        disk = hp.query_disc(64, vec, 2*degree)

        for i in disk:
            map[i] = 1*scale
        
        scale +=1
    
    return map






        


def main():
    
    for deg in range(1,21):

        make_smooth_relint(deg)

        for i in range(0,2):
            make_small_scale_maps('{deg}DegMinusl{lm}'.format(lm=i, deg=deg), deg,i)
            make_pmaps(deg,'MapData/{deg}DegMinusl{lm}'.format(lm=i, deg=deg), '{i}maxl'.format(i=i) )
    
    p_listex()



if __name__ == '__main__':
    


    

    point_map_etc(17)
    


    #degree_ana()
    #make_smooth_relint(10)
    



    print('Done')