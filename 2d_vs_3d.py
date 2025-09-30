import xarray as xr
import netCDF4
import cartopy
import numpy as np
import pandas as pd
import metpy
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from Tracking_Functions import moaap

object_names = [['cold clouds', '#737373', '-', 2],
                ['surface cyclones', 'k', '-', 2],
                ['mid-level cyclones', 'k', '--', 2],
                ['anticyclones', '#ff7f00', '-', 2],
                ['MCS', '#33a02c', '-', 2],
                ['moisture streams', 'r', '-', 2],
                ['jets', '#6a3d9a', '-', 2],
                ['Rossby waves', '#8c510a', '-', 3],
                ['mixed Rossby gravity waves', '#bf812d', '-', 1.5],
                ['inertia gravity waves', '#dfc27d','-', 3],
                ['Kelvin waves', '#abd9e9','-', 1.5],
                ['eastward inertia gravity waves', '#4575b4', '-', 3],
                ['fronts', '#cab2d6', '-', 2]]


data_vars = xr.open_dataset('20210701-04_MOAAP-Input_24h.nc')

dT = 1 # time interval of input files [hours]
Mask = np.copy(data_vars['lon']); Mask[:]=1 # tracking is applied globally
DataName = 'ERA5'
time_datetime = pd.to_datetime(np.array(data_vars['time'].values, dtype='datetime64'))

object_split = moaap(
                      data_vars['lon'],
                      data_vars['lat'],
                      time_datetime,
                      dT,
                      Mask,
                      # v850 =  data_vars['V850'].values,
                      # u850 = data_vars['U850'].values,
                      # t850 = data_vars['T850'].values,
                      # q850 = data_vars['Q850'].values,
                      # slp = data_vars['SLP'].values,
                      # ivte = data_vars['IVTE'].values,
                      # ivtn = data_vars['IVTN'].values,
                      # z500 = data_vars['Z500'].values,
                      # v200 = data_vars['V200'].values,
                      # u200 = data_vars['U200'].values,
                      v850 =None, #  data_vars['V850'],
                      u850 = None, #data_vars['U850'],
                      t850 = None, #data_vars['T850'],
                      q850 = None, #data_vars['Q850'],
                      slp = None, #data_vars['SLP'],
                      ivte = None, #data_vars['IVTE'],
                      ivtn = None, #data_vars['IVTN'],
                      z500 = None, #data_vars['Z500'],
                      v200 = None, #data_vars['V200'],
                      u200 = None, #data_vars['U200'],
                      pr   = data_vars['PR'].values,
                      tb   = data_vars['Tb'].values,
                      DataName = DataName,
                      OutputFolder = 'moaap_output/',
                      js_min_anomaly = 12,
                      MinTimeJS = 12,
                        )


data_moaap = xr.open_dataset('moaap_output/202107_ERA5_ObjectMasks__dt-1h_MOAAP-masks.nc')
import pickle
with open('moaap_output/MCSs_202107__dt-1h_MOAAP-masks.pkl', 'rb') as f:
    mcs_charac = pickle.load(f)

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Create a figure and axis with a PlateCarree projection (latitude and longitude)
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                       figsize=(14,6))

# Set the extent of the map (in this case, the entire globe)
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

# Add coastlines and gridlines for reference
ax.coastlines(color='#969696')
ax.gridlines()

# Generate some random data (latitude, longitude, and values)
mcs_mask = np.array(data_moaap['MCS_Tb_Objects'][12,:,:])
mcs_mask[mcs_mask == 0] = np.nan
sc = plt.pcolormesh(data_moaap['lon'],
                    data_moaap['lat'],
                    mcs_mask,
                    cmap = 'nipy_spectral')


# plot MCS tracks
for ii in range(len(mcs_charac.keys())):
  LatLonTrack = mcs_charac[list(mcs_charac.keys())[ii]]['track']
  plt.plot(LatLonTrack[:,1],LatLonTrack[:,0], transform=ccrs.PlateCarree(), lw=1, color='k')

ax.set_extent([-180, 180, -70, 70], crs=ccrs.PlateCarree())

# Add a colorbar to the plot
cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.7, label='MCS mask')

# Set the title of the plot
plt.title('MCS tracks (black lines) and masks at '+str(time_datetime[12])[:16])

plt.savefig("MCS_tracks_masks_3d.png")

print_gif = False
if print_gif:
  for tt in tqdm(range(len(time_datetime))):

    # Create a figure and axis with a PlateCarree projection (latitude and longitude)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                          figsize=(14,6))

    # Set the extent of the map (in this case, the entire globe)
    # ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    # Add coastlines and gridlines for reference
    ax.coastlines(color='#969696')
    ax.gridlines()

    # MCSs
    # if 'MCS_Tb_Objects' in data_moaap.data_vars:
    #     sc = plt.contour(data_moaap['lon'],
    #                     data_moaap['lat'],
    #                     np.array(data_moaap['MCS_Tb_Objects'][tt,:,:])>0,
    #                     colors = '#33a02c', levels=range(0,2,1))
    # ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    # # Mid-Level Cyclones
    # if 'CY_Objects' in data_moaap.data_vars:
    #     sc = plt.contour(data_moaap['lon'],
    #                     data_moaap['lat'],
    #                     np.array(data_moaap['CY_Objects'][tt,:,:])>0,
    #                     colors = 'k', levels=range(0,2,1))

    # # COL
    # if 'COL_Objects' in data_moaap.data_vars:
    #     sc = plt.contour(data_moaap['lon'],
    #                     data_moaap['lat'],
    #                     np.array(data_moaap['COL_Objects'][tt,:,:])>0,
    #                     colors = 'k', levels=range(0,2,1), linestyles = '--')

    # # Anticyclones
    # if 'ACY_Objects' in data_moaap.data_vars:
    #     sc = plt.contour(data_moaap['lon'],
    #                     data_moaap['lat'],
    #                     np.array(data_moaap['ACY_Objects'][tt,:,:])>0,
    #                     colors = '#ff7f00', levels=range(0,2,1))

    # # Jets
    # if 'JET_Objects' in data_moaap.data_vars:
       
    #     sc = plt.contour(data_moaap['lon'],
    #                     data_moaap['lat'],
    #                     np.array(data_moaap['JET_Objects'][tt,:,:])>0,
    #                     colors = '#6a3d9a', levels=range(0,2,1))

    # # ARs
    # if 'AR_Objects' in data_moaap.data_vars:
    #     sc = plt.contour(data_moaap['lon'],
    #                         data_moaap['lat'],
    #                         np.array(data_moaap['AR_Objects'][tt,:,:])>0,
    #                         colors = 'r', levels=range(0,2,1))

    # # Fronts
    # if 'FR_Objects' in data_moaap.data_vars:
    #     sc = plt.contour(data_moaap['lon'],
    #                         data_moaap['lat'],
    #                         np.array(data_moaap['FR_Objects'][tt,:,:])>1,
    #                         colors = '#cab2d6', levels=range(0,2,1),
    #                         linewidths = 0.5)

    # # # Equatorial Rossby Wave
    # if 'ER_Objects' in data_moaap.data_vars:
    #     sc = plt.contour(data_moaap['lon'],
    #                         data_moaap['lat'],
    #                         np.array(data_moaap['ER_Objects'][tt,:,:])>1,
    #                         colors = '#8c510a', levels=range(0,2,1),
    #                         linewidths = 0.5)
    # Define mapping of object types to colors/styles
    object_plotting_config = {
        'MCS_Tb_Objects': {'colors': '#33a02c', 'threshold': 0, 'linewidth': 1},
        'CY_Objects': {'colors': 'k', 'threshold': 0, 'linewidth': 1},
        'COL_Objects': {'colors': 'k', 'threshold': 0, 'linewidth': 1, 'linestyles': '--'},
        'ACY_Objects': {'colors': '#ff7f00', 'threshold': 0, 'linewidth': 1},
        'JET_Objects': {'colors': '#6a3d9a', 'threshold': 0, 'linewidth': 1},
        'AR_Objects': {'colors': 'r', 'threshold': 0, 'linewidth': 1},
        'FR_Objects': {'colors': '#cab2d6', 'threshold': 1, 'linewidth': 0.5},
        'ER_Objects': {'colors': '#8c510a', 'threshold': 1, 'linewidth': 0.5}
    }

    # Plot only available objects
    for obj_name, config in object_plotting_config.items():
        if obj_name in data_moaap.data_vars:
            plot_args = {
                'colors': config['colors'],
                'levels': range(0, 2, 1),
                'linewidths': config.get('linewidth', 1)
            }
            if 'linestyles' in config:
                plot_args['linestyles'] = config['linestyles']
                
            sc = plt.contour(data_moaap['lon'],
                            data_moaap['lat'],
                            np.array(data_moaap[obj_name][tt,:,:]) > config['threshold'],
                            **plot_args)

    # Set the title of the plot
    plt.title('Objects identied by MOAAP at '+str(time_datetime[tt])[:16])

    # create legend
    for ob in range(len(object_names)):
      plt.plot([],[], color = object_names[ob][1], \
               linestyle = object_names[ob][2],\
               lw = object_names[ob][3],\
               label = object_names[ob][0])

    # plt.legend()
    ax.legend(bbox_to_anchor=(1, 0.00), ncol=4)


    # Show the plot
    fig.savefig('images/'+str(tt).zfill(3)+'_CausesOfExtreme_100_daily-extremes.jpg', bbox_inches='tight', dpi=100)

    import glob

    from PIL import Image
    def make_gif(frame_folder):
        frames = [Image.open(image) for image in np.sort(glob.glob(f"{frame_folder}/*.jpg"))]
        frame_one = frames[0]
        frame_one.save("phenomenon.gif", format="GIF", append_images=frames,
                save_all=True, duration=100, loop=0)

    # if __name__ == "__main__":
    make_gif("images/")
