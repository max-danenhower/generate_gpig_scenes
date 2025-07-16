import os

# Force the working directory to your script folder
os.chdir("C:/Users/AIRS Shared Lab/Desktop/maxd_git/generate_gpig_scenes")

import numpy as np
import xarray as xr
from rrs_inversion_pigments import rrs_inversion_pigments
import ray
import time
from datetime import date, timedelta, datetime
import earthaccess
import sys

def load_data():
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
    # last weeks data
    file_date = date.today() - timedelta(days=7)
    tspan = (file_date, file_date)

    bbox = (90,13,91,14)

    print('searching for data from', tspan[0], 'to', tspan[1],'\n')

    success = True

    rrs_results = earthaccess.search_data(
        short_name='PACE_OCI_L2_AOP_NRT',
        bounding_box=bbox,
        temporal=tspan,
        count=1
    )
    if (len(rrs_results) > 0):
        print('collecting rrs data\n')
        print('L2 AOP filename:', rrs_results[0],'\n')
        rrs_paths = earthaccess.download(rrs_results, 'rrs_data')
    else:
        print('No L2 AOP data found\n')
        success = False

    sal_results = earthaccess.search_data(
        short_name='SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5',
        temporal=tspan,
        count=5
    )
    if (len(sal_results) > 0):
        print('collecting salinity data\n')
        print('salinity filename:',sal_results[4],'\n')
        sal_paths = earthaccess.download(sal_results[4], 'sal_data')
    else:
        print('No salinity data found\n')
        success = False

    temp_results = earthaccess.search_data(
        short_name='MUR-JPL-L4-GLOB-v4.1',
        temporal=tspan,
        count=1
    )
    if (len(temp_results) > 0):
        print('collecting temperature data\n')
        print('temperature filename:',temp_results[0],'\n')
        temp_paths = earthaccess.download(temp_results, 'temp_data')
    else:
        print('No temperature data found\n')
        success = False

    if success:
        return rrs_paths[0], sal_paths[0], temp_paths[0]
    else:
        raise Exception('Missing data')
    
def plot_pigments(data, lower_bound, upper_bound):
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
    plt.show()

def read_data(rrs_path,sal_path,temp_path):
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

    print('L2 granule bounds: north',n_bound,'south',s_bound,'east',e_bound,'west',w_bound,'\n')

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

    return rrs_box, rrs_unc_box, wl_coord, sal, temp


@ray.remote(num_cpus=1)
def run_batch(rrs_batch,rrs_unc_batch,wl,temp_batch,sal_batch):

    results = []
    
    for i in range(rrs_batch.shape[0]):
        if np.isnan(rrs_batch[i][0]) or np.isnan(sal_batch[i]) or np.isnan(temp_batch[i]):
            pigs = np.array([np.nan,np.nan,np.nan,np.nan])
            results.append(pigs)
        else:
            rrs = rrs_batch[i]
            rrs_unc = rrs_unc_batch[i]
            sal = sal_batch[i]
            temp = temp_batch[i]

            pigs = rrs_inversion_pigments(rrs,rrs_unc,wl,float(temp),float(sal))[0]
            results.append(pigs)

    return results

def main():
    # miniconda path: miniconda3\Scripts\activate.bat

    log_path = 'C:/Users/AIRS Shared Lab/Desktop/maxd_git/generate_gpig_scenes/task-log.txt'
    sys.stdout = open(log_path, 'a', buffering=1)
    sys.stderr = sys.stdout

    print(f"\n--- Script started at {datetime.now()} ---\n")

    earthaccess.login(persist=True)

    r_path, s_path, t_path = load_data()

    r,ru,wl,s,t = read_data(r_path,s_path,t_path)

    Rrs_flat = r.stack(pix=('number_of_lines', 'pixels_per_line'))         # shape: (n_pix, 172)
    Rrs_unc_flat = ru.stack(pix=('number_of_lines', 'pixels_per_line')) # same shape
    temp_flat = t.stack(pix=('number_of_lines', 'pixels_per_line'))       # shape: (n_pix,)
    sal_flat = s.stack(pix=('number_of_lines', 'pixels_per_line'))         # shape: (n_pix,)

    n_pix = Rrs_flat.sizes['pix']
    print('number of pixels:',n_pix,'\n')

    Rrs_np = Rrs_flat.values.T       # shape: (n_pix, 172)
    Rrs_unc_np = Rrs_unc_flat.values.T
    temp_np = temp_flat.values.T
    sal_np = sal_flat.values.T

    start = time.time()

    ray.init(include_dashboard=True)

    print('ray availble resources', ray.available_resources(),'\n')

    batch_size = 10_000

    batches = [
        (
            Rrs_np[i:i+batch_size],
            Rrs_unc_np[i:i+batch_size],
            wl,
            temp_np[i:i+batch_size],
            sal_np[i:i+batch_size]
        )
        for i in range(0, len(Rrs_np), batch_size)
    ]

    # Launch Ray tasks
    futures = [run_batch.remote(*b) for b in batches]
    results = ray.get(futures)  # list of lists, flatten if needed
    flat_results = [res for batch in results for res in batch]

    # Get spatial dimensions from original data
    n_lines = r.sizes['number_of_lines']
    n_pixels = r.sizes['pixels_per_line']

    # Convert to 3D array: (n_lines, n_pixels, 4)
    results_array = np.array(flat_results).reshape(n_lines, n_pixels, -1)

    r['chla'].values[:, :] = results_array[:, :, 0]
    r['chlb'].values[:, :] = results_array[:, :, 1]
    r['chlc'].values[:, :] = results_array[:, :, 2]
    r['ppc'].values[:, :]  = results_array[:, :, 3]

    print('total task runtime', time.time()-start,'\n')

    output_str = 'gpig-' + str(date.today() - timedelta(days=7))

    try:
        r.to_netcdf(output_str)
        print('successfully saved results\n')
    except:
        print('error loading results\n')

if __name__ == "__main__":
    main()


