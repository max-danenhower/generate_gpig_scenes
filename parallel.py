import numpy as np
import xarray as xr
from rrs_inversion_pigments import rrs_inversion_pigments
import ray
import time

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
   # n_bound = 25
   # s_bound = 23
   # w_bound = -111
   # e_bound = -109

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

    r_path = 'rrs_data/PACE_OCI.20250204T201251.L2.OC_AOP.V3_0.nc'
    s_path = 'sal_data/SMAP_L3_SSS_20250204_8DAYS_V5.0.nc'
    t_path = 'temp_data/20250204090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'
    r,ru,wl,s,t = read_data(r_path,s_path,t_path)

    Rrs_flat = r.stack(pix=('number_of_lines', 'pixels_per_line'))         # shape: (n_pix, 172)
    Rrs_unc_flat = ru.stack(pix=('number_of_lines', 'pixels_per_line')) # same shape
    temp_flat = t.stack(pix=('number_of_lines', 'pixels_per_line'))       # shape: (n_pix,)
    sal_flat = s.stack(pix=('number_of_lines', 'pixels_per_line'))         # shape: (n_pix,)

    n_pix = Rrs_flat.sizes['pix']

    Rrs_np = Rrs_flat.values.T       # shape: (n_pix, 172)
    Rrs_unc_np = Rrs_unc_flat.values.T
    temp_np = temp_flat.values.T
    sal_np = sal_flat.values.T

    ray.init(include_dashboard=True)

    print('ray availble cores', ray.available_resources())

    batch_size = 1000
    print('batch size',batch_size)

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

    start = time.time()

    # Launch Ray tasks
    futures = [run_batch.remote(*b) for b in batches]
    results = ray.get(futures)  # list of lists, flatten if needed
    flat_results = [res for batch in results for res in batch]

    # Get spatial dimensions from original data
    n_lines = r.sizes['number_of_lines']
    n_pixels = r.sizes['pixels_per_line']

       
    print('time',time.time()-start)
    '''
    # Convert to 3D array: (n_lines, n_pixels, 4)
    results_array = np.array(flat_results).reshape(n_lines, n_pixels, -1)

    for i in range(len(results_array)):
        for j in range(len(results_array[0])):
            chla,chlb,chlc,ppc = results_array[i][j]
            r['chla'][i][j] = chla
            r['chlb'][i][j] = chlb
            r['chlc'][i][j] = chlc
            r['ppc'][i][j] = ppc

    print('time',time.time()-start)

    r.to_netcdf('output')
    '''

if __name__ == "__main__":
    main()




