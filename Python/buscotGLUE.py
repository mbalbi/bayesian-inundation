from LisfloodPy.tools import jaccard_fit
from LisfloodGP import Lisflood
from datetime import date


import os, csv
import numpy as np

##
# Project name
today = date.today().strftime("%d%m%y")
project = 'Buscot_grid' # This will serve to name the input files for the .par file
resroot = 'Grid_' + today + '_qvar146_U2' # This serves to name the output files
output_dir = 'results//'+resroot # This will save some of the output files to that folder ('results' folder should already exist)
 
# Input parameters
r_ch = np.arange( 0.01, 0.15, 0.001)
r_fp = np.arange( 0.01, 0.15, 0.001)

# Observations
observed = 'Observations//BuscotFlood92.tiff'

# Create list with all combinations
params = np.array(np.meshgrid(r_ch, r_fp)).T.reshape(-1,2)

# Initialize results csv
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
results_csv = os.path.join( output_dir, 'results.csv')
with open(results_csv, "w", newline='') as f:
    # Writers
    writer = csv.writer(f, delimiter=',', quotechar='"')
    # Headers
    writer.writerow(['r_ch', 'r_fp', 'A', 'B', 'C', 'F', 'resroot'])
    f.flush()

    count = 0
    for param in params:
        count += 1
        # Current run parameters
        r_ch = param[0]
        r_fp = param[1]

        # Save outputs
        outputs = ['max']
        new_name = resroot + '_sim' + str(count)
        predicted = os.path.join( output_dir, new_name+'.max' )

        # Run lisflood
        _ = Lisflood( [], [r_ch, r_fp], output=predicted ) 
        
        # Compare with an observations map
        
        chanmask0 = 'Buscot.bed.asc'
        save_comparison = False
        fit = jaccard_fit( observed, predicted, chanmask0,
                           save_comparison=save_comparison )

        # Save metadata
        datarow = [str(r_ch), str(r_fp), str(fit['A'])[:9],
                   str(fit['B'])[:9], str(fit['C'])[:9], str(fit['F'])[:9],
                   new_name]
        writer.writerow(datarow)
        f.flush()


