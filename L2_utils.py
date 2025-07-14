'''
Max Danenhower

This file provides methods to help retrieve Rrs data from the PACE Satellite, use that data to estimate chlorophyll a, cholorphyll b, 
chlorophyll c1+c2, and photoprotective carotenoids (PPC) concentrations using an inversion method, and plot a visualization of those 
pigment concentrations on a color map. These methods uses PACE's level 2 apparent optical properties (AOP) files, which include Rrs data
and their associate uncertainties. Level 2 files contain data from one swath of the PACE satellite, meaning the data are confined to 
the area of the swath. Level 2 files have 1km resolution. 
'''

import sys
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr
import ray

from rrs_inversion_pigments import rrs_inversion_pigments


def load_data(tspan, bbox):
    '''
    Downloads one L2 PACE apparent optical properties (AOP) file that intersects the coordinate box passed in, as well as 
    temperature and salinity files. Data files are saved to local folders named 'L2_data', 'sal_data', and 'temp_data'.

    Parameters:
    -----------
    tspan : tuple of str
        A tuple containing two strings both with format 'YYYY-MM-DD'. The first date in the tuple must predate the second date in the tuple.
    bbox : tuple of floats or ints
        A tuple representing spatial bounds in the form (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).

    Returns:
    --------
    L2_path : string
        A single file path to a PACE L2 AOP file.
    sal_path : string
        A single file path to a salinity file.
    temp_path : string
        A single file path to a temperature file.
    '''

    print('searching for data from', tspan[0], 'to', tspan[1])

    success = True

    rrs_results = earthaccess.search_data(
        short_name='PACE_OCI_L2_AOP',
        bounding_box=bbox,
        temporal=tspan
    )
    if (len(rrs_results) > 0):
        print('collecting rrs data')
        rrs_paths = earthaccess.download(rrs_results, 'rrs_data')
    else:
        print('No L2 AOP data found')
        success = False

    sal_results = earthaccess.search_data(
        short_name='SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5',
        temporal=tspan
    )
    if (len(sal_results) > 0):
        print('collecting salinity data')
        sal_paths = earthaccess.download(sal_results, 'sal_data')
    else:
        print('No salinity data found')
        success = False

    temp_results = earthaccess.search_data(
        short_name='MUR-JPL-L4-GLOB-v4.1',
        temporal=tspan
    )
    if (len(temp_results) > 0):
        print('collecting temperature data')
        temp_paths = earthaccess.download(temp_results, 'temp_data')
    else:
        print('No temperature data found')
        success = False

    if success:
        return rrs_paths[0], sal_paths[0], temp_paths[0]
    else:
        raise Exception('Missing data')

def estimate_inv_pigments(rrs_path, sal_path, temp_path):
    '''
    Uses the rrs_inversion_pigments algorithm to calculate chlorophyll a (Chla), chlorophyll b (Chlb), chlorophyll c1
    +c2 (Chlc12), and photoprotective carotenoids (PPC) given an Rrs spectra, salinity, and temperature. Relies on user input to 
    create a boundary box to estimate pigments for. Pigment values are in units of mg/m^3. 

    See rrs_inversion_pigments file for more information on the inversion estimation method.

    Parameters:
    -----------
    L2_path : str
        A single file path to a PACE L2 AOP file.
    sal_path : str
        A single file path to a salinity file.
    temp_path : str
        A single file path to a temperature file.

    Returns:
    --------
    Xarray dataset 
        Dataset containing the Chla, Chlb, Chlc, and PPC concentration at each lat/lon coordinate
    '''

    # define wavelengths
    sensor_band_params = xr.open_dataset(rrs_path, group='sensor_band_parameters')
    wl_coord = sensor_band_params.wavelength_3d.values
    
    dataset = xr.open_dataset(rrs_path, group='geophysical_data')
    rrs = dataset['Rrs']
    rrs_unc = dataset['Rrs_unc']

    # Add latitude and longitude coordinates to the Rrs and Rrs uncertainty datasets
    dataset = xr.open_dataset(rrs_path, group="navigation_data")
    dataset = dataset.set_coords(("longitude", "latitude"))

    dataset_r = xr.merge((rrs, dataset.coords))
    dataset_ru = xr.merge((rrs_unc, dataset.coords))

    n_bound = dataset_r.latitude.values.max()
    s_bound = dataset_r.latitude.values.min() 
    e_bound = dataset_r.longitude.values.max()
    w_bound = dataset_r.longitude.values.min()
   #n_bound = 25
   #s_bound = 22
   #e_bound = -108
   #w_bound = -111

    print('north',n_bound,'south',s_bound,'east',e_bound,'west',w_bound)

    rrs_box = dataset_r["Rrs"].where(
        (
            (dataset["latitude"] > s_bound) # southern boundary latitude
            & (dataset["latitude"] < n_bound) # northern boundary latitude
            & (dataset["longitude"] < e_bound) # eastern boundary latitude
            & (dataset["longitude"] > w_bound) # western boundary latitude
        ),
        drop=True,
    )

    rrs_unc_box = dataset_ru["Rrs_unc"].where(
        (
            (dataset["latitude"] > s_bound) # southern boundary latitude
            & (dataset["latitude"] < n_bound) # northern boundary latitude
            & (dataset["longitude"] < e_bound) # eastern boundary latitude
            & (dataset["longitude"] > w_bound) # western boundary latitude
        ),
        drop=True,
    )

    '''
    # use climatology files
    sal = xr.open_dataset(sal_path)
    sal['sss02'] = sal['sss02'].assign_coords({
        'Number of Latitudes': sal['Latitude'],
        'Number of Longitudes': sal['Longitude']
    })

    sal = sal.rename({
        'Number of Latitudes': 'lat',
        'Number of Longitudes': 'lon'
    })

    sal = sal.assign_coords({
        "lon": (((sal.lon + 180) % 360) - 180)
    })

    sal = sal.sortby('lon')

    sal = sal['sss02'].sel({"lat": slice(s_bound, n_bound), "lon": slice(w_bound, e_bound)})

    temp = xr.open_dataset(temp_path)
    temp = temp.rename({'fakeDim2': 'Latitude', 'fakeDim3': 'Longitude'})

    temp = temp['data02'].sel({"Latitude": slice(n_bound, s_bound), "Longitude": slice(w_bound, e_bound)})
    
    '''
    sal = xr.open_dataset(sal_path)
    sal = sal["smap_sss"].sel({"latitude": slice(n_bound, s_bound), "longitude": slice(w_bound, e_bound)})

    temp = xr.open_dataset(temp_path)
    temp = temp['analysed_sst'].squeeze() # get rid of extra time dimension
    temp = temp.sel({"lat": slice(s_bound, n_bound), "lon": slice(w_bound, e_bound)})
    temp = temp - 273 # convert from kelvin to celcius

    # mesh salinity and temperature onto the same coordinate system as Rrs and Rrs uncertainty
    sal = sal.interp(longitude=rrs_box.longitude, latitude=rrs_box.latitude, method='nearest')
    temp = temp.interp(lon=rrs_box.longitude, lat=rrs_box.latitude, method='nearest')

    rrs_box['chla'] = (('number_of_lines', 'pixels_per_line'), np.full((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size), np.nan))
    rrs_box['chlb'] = (('number_of_lines', 'pixels_per_line'), np.full((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size), np.nan))
    rrs_box['chlc'] = (('number_of_lines', 'pixels_per_line'), np.full((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size), np.nan))
    rrs_box['ppc'] = (('number_of_lines', 'pixels_per_line'), np.full((rrs_box.number_of_lines.size, rrs_box.pixels_per_line.size), np.nan))

    progress = 1 # keeps track of how many pixels have been calculated
    pixels = rrs_box.number_of_lines.size * rrs_box.pixels_per_line.size

    # for each coordinate estimate the pigment concentrations
    for i in range(len(rrs_box.number_of_lines)):
        for j in range(len(rrs_box.pixels_per_line)):
            # prints total number of pixels and how many have been estimated already
            sys.stdout.write('\rProgress: ' + str(progress) + '/' + str(pixels))
            sys.stdout.flush()
            progress += 1
            r = rrs_box[i][j].to_numpy()
            ru = rrs_unc_box[i][j].to_numpy()
            sal_val = float(sal[i][j].values)
            temp_val = float(temp[i][j].values)
            if not (np.isnan(r[0]) or np.isnan(sal_val) or np.isnan(temp_val)):
               #pigs = rrs_inversion_pigments(r, ru, wl_coord, temp_val, sal_val)[0]
                pigs = np.array([0,0,0,0])
                if not np.isnan(pigs[0]):
                    rrs_box['chla'][i][j] = pigs[0]
                    rrs_box['chlb'][i][j] = pigs[1]
                    rrs_box['chlc'][i][j] = pigs[2]
                    rrs_box['ppc'][i][j] = pigs[3]
    '''

    ray.init(include_dashboard=True)

    futures = []
    coords = []

    for i in range(len(rrs_box.number_of_lines)):
        for j in range(len(rrs_box.pixels_per_line)):
            progress += 1
            sys.stdout.write(f'\rProgress: {progress}/{pixels}')
            sys.stdout.flush()
            r = rrs_box[i][j].to_numpy()
            ru = rrs_unc_box[i][j].to_numpy()
            sal_val = float(sal[i][j].values)
            temp_val = float(temp[i][j].values)
            pigs = rrs_inversion_pigments.remote(r,ru,wl_coord,temp_val,sal_val)
            #pigs = rrs_inversion_pigments(r,ru,wl_coord,temp_val,sal_val)
            futures.append(pigs)
            coords.append((i,j))

    results = ray.get(futures)

    for k, (chla, chlb, chlc, ppc) in enumerate(results):
        i, j = coords[k]
        rrs_box['chla'][i][j] = chla
        rrs_box['chlb'][i][j] = chlb
        rrs_box['chlc'][i][j] = chlc
        rrs_box['ppc'][i][j] = ppc
        sys.stdout.write(f'\rProgress: {k + 1}/{pixels}')
        sys.stdout.flush()
    

    return rrs_box
    '''

def plot_pigments(data, lower_bound, upper_bound, title):
    '''
    Plots the pigment data from an L2 file with lat/lon coordinates using a color map

    Paramaters:
    -----------
    data : Xarray data array
        Contains pigment values to be plotted.
    lower_bound : float
        The lowest value represented on the color scale.
    upper_bound : float
        The upper value represented on the color scale.
    '''

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, cmap.N))
    colors = np.vstack((np.array([1, 1, 1, 1]), colors)) 
    custom_cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(np.linspace(lower_bound, upper_bound, cmap.N)), ncolors=custom_cmap.N) 

    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels={"left": "y", "bottom": "x"})
    data.plot(x="longitude", y="latitude", cmap=custom_cmap, ax=ax, norm=norm)
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
    plt.title(title)
    plt.show()