#!/usr/bin/env python

# Functions for ICA-AROMA v0.3 beta

import numpy as np
import random
import os
import subprocess
from .. import aroma_mask_out, aroma_mask_edge, aroma_mask_csf

# Denoising types accepted
accepted_den_types = {'nonaggr', 'aggr', 'both', 'no'}

def run_ica(fsl_dir, in_file, out_dir, mel_dir_in, mask, dim, TR):
    """
    This function runs MELODIC and merges the mixture modeled thresholded ICs into a single 4D nifti file

    Parameters
    ---------------------------------------------------------------------------------
    fsl_dir:     Full path of the bin-directory of FSL
    in_file:     Full path to the fMRI data file (nii.gz) on which MELODIC should be run
    out_dir:     Full path of the output directory
    mel_dir_in:   Full path of the MELODIC directory in case it has been run before, otherwise define empty string
    mask:       Full path of the mask to be applied during MELODIC
    dim:        Dimensionality of ICA
    TR:     TR (in seconds) of the fMRI data

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    melodic.ica     MELODIC directory
    melodic_IC_thr.nii.gz   merged file containing the mixture modeling thresholded Z-statistical maps located in melodic.ica/stats/
    """

    # Import needed modules

    # Define the 'new' MELODIC directory and predefine some associated files
    mel_dir = os.path.join(out_dir, 'melodic.ica')
    mel_ic = os.path.join(mel_dir, 'melodic_IC.nii.gz')
    mel_ic_mix = os.path.join(mel_dir, 'melodic_mix')
    mel_ic_thr = os.path.join(out_dir, 'melodic_IC_thr.nii.gz')

    # When a MELODIC directory is specified,
    # check whether all needed files are present.
    # Otherwise... run MELODIC again
    if (len(mel_dir) != 0
            and os.path.isfile(os.path.join(mel_dir_in, 'melodic_IC.nii.gz'))
            and os.path.isfile(os.path.join(mel_dir_in, 'melodic_FTmix'))
            and os.path.isfile(os.path.join(mel_dir_in, 'melodic_mix'))):

        print('  - The existing/specified MELODIC directory will be used.')

        # If a 'stats' directory is present (contains thresholded spatial maps)
        # create a symbolic link to the MELODIC directory.
        # Otherwise, create specific links and
        # run mixture modeling to obtain thresholded maps.
        if os.path.isdir(os.path.join(mel_dir_in, 'stats')):
            os.symlink(mel_dir_in, mel_dir)
        else:
            print('  - The MELODIC directory does not contain the required \'stats\' folder. '
                  'Mixture modeling on the Z-statistical maps will be run.')

            # Create symbolic links to the items in the specified melodic directory
            os.makedirs(mel_dir)
            for item in os.listdir(mel_dir_in):
                os.symlink(os.path.join(mel_dir_in, item),
                           os.path.join(mel_dir, item))

            # Run mixture modeling
            os.system(' '.join([os.path.join(fsl_dir, 'melodic'),
                                '--in=' + mel_ic,
                                '--ICs=' + mel_ic,
                                '--mix=' + mel_ic_mix,
                                '--outdir=' + mel_dir,
                                '--Ostats --mmthresh=0.5']))

    else:
        # If a melodic directory was specified, display that it did not contain all files needed for ICA-AROMA
        # (or that the directory does not exist at all)
        if len(mel_dir_in) != 0:
            if not os.path.isdir(mel_dir_in):
                print('  - The specified MELODIC directory does not exist. MELODIC will be run seperately.')
            else:
                print('  - The specified MELODIC directory does not contain the required files to run ICA-AROMA. '
                      'MELODIC will be run seperately.')

        # Run MELODIC
        os.system(' '.join([os.path.join(fsl_dir, 'melodic'),
                            '--in=' + in_file,
                            '--outdir=' + mel_dir,
                            '--mask=' + mask,
                            '--dim=' + str(dim),
                            '--Ostats --nobet --mmthresh=0.5 --report',
                            '--tr=' + str(TR)]))

    # Get number of components
    cmd = ' '.join([os.path.join(fsl_dir, 'fslinfo'),
                    mel_ic,
                    '| grep dim4 | head -n1 | awk \'{print $2}\''])
    nr_ics = int(float(subprocess.getoutput(cmd)))

    # Merge mixture modeled thresholded spatial maps.
    # Note! In case that mixture modeling did not converge, the file will contain two spatial maps.
    # The latter being the results from a simple null hypothesis test.
    # In that case, this map will have to be used (the first one will be empty).
    for i in range(1, nr_ics + 1):
        # Define thresholded zstat-map file
        z_temp = os.path.join(mel_dir, 'stats', 'thresh_zstat' + str(i) + '.nii.gz')
        cmd = ' '.join([os.path.join(fsl_dir, 'fslinfo'),
                        z_temp,
                        '| grep dim4 | head -n1 | awk \'{print $2}\''])
        len_ic = int(float(subprocess.getoutput(cmd)))

        # Define zeropad for this IC-number and new zstat file
        cmd = ' '.join([os.path.join(fsl_dir, 'zeropad'),
                        str(i),
                        '4'])
        ic_num = subprocess.getoutput(cmd)
        zstat = os.path.join(out_dir, 'thr_zstat' + ic_num)

        # Extract last spatial map within the thresh_zstat file
        os.system(' '.join([os.path.join(fsl_dir, 'fslroi'),
                            z_temp,      # input
                            zstat,      # output
                            str(len_ic - 1),   # first frame
                            '1']))      # number of frames

    # Merge and subsequently remove all mixture modeled Z-maps within the output directory
    os.system(' '.join([os.path.join(fsl_dir, 'fslmerge'),
                        '-t',  # concatenate in time
                        mel_ic_thr,  # output
                        os.path.join(out_dir, 'thr_zstat????.nii.gz')]))  # inputs

    os.system('rm ' + os.path.join(out_dir, 'thr_zstat????.nii.gz'))

    # Apply the mask to the merged file (in case a melodic-directory was predefined and run with a different mask)
    os.system(' '.join([os.path.join(fsl_dir, 'fslmaths'),
                        mel_ic_thr,
                        '-mas ' + mask,
                        mel_ic_thr]))


def register_2_mni(fsl_dir, in_file, out_file, aff_mat, warp):
    """
    This function registers an image (or time-series of images) to MNI152 T1 2mm.
    If no affmat is defined, it only warps (i.e. it assumes that the data has been registered
    to the structural scan associated with the warp-file already).
    If no warp is defined either, it only resamples the data to 2mm isotropic if needed
    (i.e. it assumes that the data has been registered to a MNI152 template).
    In case only an affmat file is defined, it assumes that the data has to be linearly registered to MNI152
    (i.e. the user has a reason not to use non-linear registration on the data).

    Parameters
    ---------------------------------------------------------------------------------
    fsl_dir:     Full path of the bin-directory of FSL
    in_file:     Full path to the data file (nii.gz) which has to be registerd to MNI152 T1 2mm
    out_file:    Full path of the output file
    aff_mat:     Full path of the mat file describing the linear registration (if data is still in native space)
    warp:       Full path of the warp file describing the non-linear registration (if data has not been registered
                    to MNI152 space yet)

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    melodic_IC_mm_MNI2mm.nii.gz merged file containing the mixture modeling thresholded Z-statistical maps registered
        to MNI152 2mm
    """

    # Define the MNI152 T1 2mm template
    fsl_no_bin = fsl_dir.rsplit('/', 2)[0]
    ref = os.path.join(fsl_no_bin, 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')

    # If the no affmat- or warp-file has been specified, assume that the data is already in MNI152 space.
    # In that case, only check if resampling to 2mm is needed
    if (len(aff_mat) == 0) and (len(warp) == 0):
        # Get 3D voxel size
        pix_dim1 = float(subprocess.getoutput('%sfslinfo %s | grep pixdim1 | awk \'{print $2}\'' % (fsl_dir, in_file)))
        pix_dim2 = float(subprocess.getoutput('%sfslinfo %s | grep pixdim2 | awk \'{print $2}\'' % (fsl_dir, in_file)))
        pix_dim3 = float(subprocess.getoutput('%sfslinfo %s | grep pixdim3 | awk \'{print $2}\'' % (fsl_dir, in_file)))

        # If voxel size is not 2mm isotropic, resample the data, otherwise copy the file
        if (pix_dim1 != 2) or (pix_dim2 != 2) or (pix_dim3 != 2):
            os.system(' '.join([os.path.join(fsl_dir, 'flirt'),
                                ' -ref ' + ref,
                                ' -in ' + in_file,
                                ' -out ' + out_file,
                                ' -applyisoxfm 2 -interp trilinear']))
        else:
            os.system('cp ' + in_file + ' ' + out_file)

    # If only a warp-file has been specified, assume that the data has already been registered to the structural scan.
    # In that case, apply the warping without a affmat
    elif (len(aff_mat) == 0) and (len(warp) != 0):
        # Apply warp
        os.system(' '.join([os.path.join(fsl_dir, 'applywarp'),
                            '--ref=' + ref,
                            '--in=' + in_file,
                            '--out=' + out_file,
                            '--warp=' + warp,
                            '--interp=trilinear']))

    # If only an affmat-file has been specified, perform affine registration to MNI
    elif (len(aff_mat) != 0) and (len(warp) == 0):
        os.system(' '.join([os.path.join(fsl_dir, 'flirt'),
                            '-ref ' + ref,
                            '-in ' + in_file,
                            '-out ' + out_file,
                            '-applyxfm -init ' + aff_mat,
                            '-interp trilinear']))

    # If both an affmat- and warp-file have been defined, apply the warping accordingly
    else:
        os.system(' '.join([os.path.join(fsl_dir, 'applywarp'),
                            '--ref=' + ref,
                            '--in=' + in_file,
                            '--out=' + out_file,
                            '--warp=' + warp,
                            '--premat=' + aff_mat,
                            '--interp=trilinear']))

def cross_correlation(a, b):
    """Cross Correlations between columns of two matrices"""
    assert a.ndim == b.ndim == 2
    _, n_cols_a = a.shape
    # nb variables in columns rather than rows hence transpose
    # extract just the cross terms between cols in a and cols in b
    return np.corrcoef(a.T, b.T)[:n_cols_a, n_cols_a:]


def feature_time_series(mel_mix, mc):
    """ This function extracts the maximum RP correlation feature scores. 
    It determines the maximum robust correlation of each component time-series
    with a model of 72 realignment parameters.

    Parameters
    ---------------------------------------------------------------------------------
    mel_mix:     Full path of the melodic_mix text file
    mc:     Full path of the text file containing the realignment parameters

    Returns
    ---------------------------------------------------------------------------------
    maxRPcorr:  Array of the maximum RP correlation feature scores for the components
    of the melodic_mix file"""

    # Import required modules

    # Read melodic mix file (IC time-series), subsequently define a set of squared time-series
    mix = np.loadtxt(mel_mix)

    # Read motion parameter file
    rp6 = np.loadtxt(mc)
    _, n_params = rp6.shape

    # Determine the derivatives of the RPs (add zeros at time-point zero)
    rp6_der = np.vstack((np.zeros(n_params),
                         np.diff(rp6, axis=0)
                         ))

    # Create an RP-model including the RPs and its derivatives
    rp12 = np.hstack((rp6, rp6_der))

    # Add the squared RP-terms to the model
    # add the fw and bw shifted versions
    rp12_1fw = np.vstack((
        np.zeros(2 * n_params),
        rp12[:-1]
    ))
    rp12_1bw = np.vstack((
        rp12[1:],
        np.zeros(2 * n_params)
    ))
    rp_model = np.hstack((rp12, rp12_1fw, rp12_1bw))

    # Determine the maximum correlation between RPs and IC time-series
    n_splits = 1000
    n_mix_rows, n_mix_cols = mix.shape
    n_rows_to_choose = int(round(0.9 * n_mix_rows))

    # Max correlations for multiple splits of the dataset (for a robust estimate)
    max_correls = np.empty((n_splits, n_mix_cols))
    for i in range(n_splits):
        # Select a random subset of 90% of the dataset rows (*without* replacement)
        chosen_rows = random.sample(population=range(n_mix_rows),
                                    k=n_rows_to_choose)

        # Combined correlations between RP and IC time-series, squared and non-squared
        correl_nonsquared = cross_correlation(mix[chosen_rows],
                                              rp_model[chosen_rows])
        correl_squared = cross_correlation(mix[chosen_rows]**2,
                                           rp_model[chosen_rows]**2)
        correl_both = np.hstack((correl_squared, correl_nonsquared))

        # Maximum absolute temporal correlation for every IC
        max_correls[i] = np.abs(correl_both).max(axis=1)

    # Feature score is the mean of the maximum correlation over all the random splits
    # Avoid propagating occasional nans that arise in artificial test cases
    return np.nanmean(max_correls, axis=0)


def feature_frequency(mel_ft_mix, TR):
    """
    This function extracts the high-frequency content feature scores.
    It determines the frequency, as a fraction of the Nyquist frequency,
    at which the higher and lower frequencies explain half
    of the total power between 0.01Hz and Nyquist.

    Parameters
    ---------------------------------------------------------------------------------
    mel_ft_mix:   Full path of the melodic_FTmix text file
    TR:     TR (in seconds) of the fMRI data (float)

    Returns
    ---------------------------------------------------------------------------------
    HFC:        Array of the HFC ('High-frequency content') feature scores
    for the components of the melodic_FTmix file
    """

    # Determine sample frequency
    Fs = 1.0 / TR

    # Determine Nyquist-frequency
    Ny = Fs / 2.0

    # Load melodic_FTmix file
    FT = np.loadtxt(mel_ft_mix)

    # Determine which frequencies are associated with every row in the melodic_FTmix file
    # (assuming the rows range from 0Hz to Nyquist)
    f = Ny * (np.array(list(range(1, FT.shape[0] + 1)))) / (FT.shape[0])

    # Only include frequencies higher than 0.01Hz
    fincl = np.squeeze(np.array(np.where(f > 0.01)))
    FT = FT[fincl, :]
    f = f[fincl]

    # Set frequency range to [0-1]
    f_norm = (f - 0.01) / (Ny - 0.01)

    # For every IC; get the cumulative sum as a fraction of the total sum
    fcumsum_fract = np.cumsum(FT, axis=0) / np.sum(FT, axis=0)

    # Determine the index of the frequency with the fractional cumulative sum closest to 0.5
    idx_cutoff = np.argmin(np.abs(fcumsum_fract - 0.5), axis=0)

    # Now get the fractions associated with those indices index, these are the final feature scores
    HFC = f_norm[idx_cutoff]

    # Return feature score
    return HFC


def feature_spatial(fsl_dir, temp_dir, aroma_dir, mel_ic):
    """
    This function extracts the spatial feature scores.
    For each IC it determines the fraction of the mixture modeled thresholded Z-maps respectively located
    within the CSF or at the brain edges, using predefined standardized masks.

    Parameters
    ---------------------------------------------------------------------------------
    fsl_dir:     Full path of the bin-directory of FSL
    temp_dir:    Full path of a directory where temporary files can be stored (called 'temp_IC.nii.gz')
    aroma_dir:   Full path of the ICA-AROMA directory, containing the mask-files (mask_edge.nii.gz, mask_csf.nii.gz
                    & mask_out.nii.gz)
                NOTE: In the pip-packaged version this should point to the package resources directory
                    (e.g. ica_aroma_py/resources).
    mel_ic:      Full path of the nii.gz file containing mixture-modeled thresholded (p>0.5) Z-maps, registered to
                    the MNI152 2mm template

    Returns
    ---------------------------------------------------------------------------------
    edge_fract:  Array of the edge fraction feature scores for the components of the melIC file
    csf_fract:   Array of the CSF fraction feature scores for the components of the melIC file
    """

    # Get the number of ICs
    num_ics = int(subprocess.getoutput('%sfslinfo %s | grep dim4 | head -n1 | awk \'{print $2}\'' % (fsl_dir, mel_ic)))

    # Loop over ICs
    edge_fract = np.zeros(num_ics)
    csf_fract = np.zeros(num_ics)
    temp_ic = None
    for i in range(0, num_ics):
        # Define temporary IC-file
        temp_ic = os.path.join(temp_dir, 'temp_IC.nii.gz')

        # Extract IC from the merged melodic_IC_thr2MNI2mm file
        os.system(' '.join([os.path.join(fsl_dir, 'fslroi'),
                            mel_ic,
                            temp_ic,
                            str(i),
                  '1']))

        # Change to absolute Z-values
        os.system(' '.join([os.path.join(fsl_dir, 'fslmaths'),
                  temp_ic,
                  '-abs',
                  temp_ic]))

        # Get sum of Z-values within the total Z-map (calculate via the mean and number of non-zero voxels)
        tot_vox = int(subprocess.getoutput(' '.join([os.path.join(fsl_dir, 'fslstats'),
                                                    temp_ic,
                                                    '-V | awk \'{print $1}\''])))

        if not (tot_vox == 0):
            tot_mean = float(subprocess.getoutput(' '.join([os.path.join(fsl_dir, 'fslstats'),
                                                           temp_ic,
                                                           '-M'])))
        else:
            print('     - The spatial map of component ' + str(i + 1) + ' is empty. Please check!')
            tot_mean = 0

        tot_sum = tot_mean * tot_vox

        # Get sum of Z-values of the voxels located within the CSF (calculate via the mean and number
        # of non-zero voxels)
        csf_vox = int(subprocess.getoutput(' '.join([os.path.join(fsl_dir, 'fslstats'),
                                                    temp_ic,
                                                    '-k ' + aroma_mask_csf,
                                                    '-V | awk \'{print $1}\''])))

        if not (csf_vox == 0):
            csf_mean = float(subprocess.getoutput(' '.join([os.path.join(fsl_dir, 'fslstats'),
                                                           temp_ic,
                                                           '-k ' + aroma_mask_csf,
                                                           '-M'])))
        else:
            csf_mean = 0

        csf_sum = csf_mean * csf_vox

        # Get sum of Z-values of the voxels located within the Edge (calculate via the mean and number of
        # non-zero voxels)
        edge_vox = int(subprocess.getoutput(' '.join([os.path.join(fsl_dir, 'fslstats'),
                                                     temp_ic,
                                                     '-k ' + aroma_mask_edge,
                                                     '-V | awk \'{print $1}\''])))

        if not (edge_vox == 0):
            edge_mean = float(subprocess.getoutput(' '.join([os.path.join(fsl_dir, 'fslstats'),
                                                            temp_ic,
                                                            '-k ' + aroma_mask_edge,
                                                            '-M'])))
        else:
            edge_mean = 0

        edge_sum = edge_mean * edge_vox

        # Get sum of Z-values of the voxels located outside the brain (calculate via the mean and number
        # of non-zero voxels)
        out_vox = int(subprocess.getoutput(' '.join([os.path.join(fsl_dir, 'fslstats'),
                                                    temp_ic,
                                                    '-k ' + aroma_mask_out,
                                                    '-V | awk \'{print $1}\''])))

        if not (out_vox == 0):
            out_mean = float(subprocess.getoutput(' '.join([os.path.join(fsl_dir, 'fslstats'),
                                                           temp_ic,
                                                           '-k ' + aroma_mask_out,
                                                           '-M'])))
        else:
            out_mean = 0

        out_sum = out_mean * out_vox

        # Determine edge and CSF fraction
        if not (tot_sum == 0):
            edge_fract[i] = (out_sum + edge_sum) / (tot_sum - csf_sum) if not ((tot_sum - csf_sum) == 0) else 0
            csf_fract[i] = csf_sum / tot_sum
        else:
            edge_fract[i] = 0
            csf_fract[i] = 0

    # Remove the temporary IC-file
    if os.path.isfile(temp_ic):
        os.remove(temp_ic)

    # Return feature scores
    return edge_fract, csf_fract


def classification(out_dir, max_rp_corr, edge_fract, HFC, csf_fract):
    """
    This function classifies a set of components into motion and 
    non-motion components based on four features; 
    maximum RP correlation, high-frequency content, edge-fraction and CSF-fraction

    Parameters
    ---------------------------------------------------------------------------------
    out_dir:     Full path of the output directory
    max_rp_corr:  Array of the 'maximum RP correlation' feature scores of the components
    edge_fract:  Array of the 'edge fraction' feature scores of the components
    HFC:        Array of the 'high-frequency content' feature scores of the components
    csf_fract:   Array of the 'CSF fraction' feature scores of the components

    Return
    ---------------------------------------------------------------------------------
    motion_ics   Array containing the indices of the components identified as motion components

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    classified_motion_ICs.txt   A text file containing the indices of the components identified as motion components """

    # Classify the ICs as motion or non-motion

    # Define criteria needed for classification (thresholds and hyperplane-parameters)
    thr_csf = 0.10
    thr_HFC = 0.35
    hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])

    # Project edge & maxRPcorr feature scores to new 1D space
    x = np.array([max_rp_corr, edge_fract])
    proj = hyp[0] + np.dot(x.T, hyp[1:])

    # Classify the ICs
    motion_ics = np.squeeze(np.array(np.where((proj > 0) + (csf_fract > thr_csf) + (HFC > thr_HFC))))

    # Put the feature scores in a text file
    np.savetxt(os.path.join(out_dir, 'feature_scores.txt'),
               np.vstack((max_rp_corr, edge_fract, HFC, csf_fract)).T)

    # Put the indices of motion-classified ICs in a text file
    txt = open(os.path.join(out_dir, 'classified_motion_ICs.txt'), 'w')
    if motion_ics.size > 1:  # and len(motion_ics) != 0: if motion_ics is not None and
        txt.write(','.join(['{:.0f}'.format(num) for num in (motion_ics + 1)]))
    elif motion_ics.size == 1:
        txt.write('{:.0f}'.format(motion_ics + 1))
    txt.close()

    # Create a summary overview of the classification
    txt = open(os.path.join(out_dir, 'classification_overview.txt'), 'w')
    txt.write('\t'.join(['IC',
                         'Motion/noise',
                         'maximum RP correlation',
                         'Edge-fraction',
                         'High-frequency content',
                         'CSF-fraction']))
    txt.write('\n')
    for i in range(0, len(csf_fract)):
        if (proj[i] > 0) or (csf_fract[i] > thr_csf) or (HFC[i] > thr_HFC):
            classif = "True"
        else:
            classif = "False"
        txt.write('\t'.join(['{:d}'.format(i + 1),
                             classif,
                             '{:.2f}'.format(max_rp_corr[i]),
                             '{:.2f}'.format(edge_fract[i]),
                             '{:.2f}'.format(HFC[i]),
                             '{:.2f}'.format(csf_fract[i])]))
        txt.write('\n')
    txt.close()

    return motion_ics


def denoising(fsl_dir, in_file, out_dir, mel_mix, den_type, den_idx):
    """
    This function classifies the ICs based on the four features; 
    maximum RP correlation, high-frequency content, edge-fraction and CSF-fraction

    Parameters
    ---------------------------------------------------------------------------------
    fsl_dir:     Full path of the bin-directory of FSL
    in_file:     Full path to the data file (nii.gz) which has to be denoised
    out_dir:     Full path of the output directory
    mel_mix:     Full path of the melodic_mix text file
    den_type:    Type of requested denoising ('aggr': aggressive, 'nonaggr': non-aggressive, 'both': both aggressive and non-aggressive
    den_idx:     Indices of the components that should be regressed out

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    denoised_func_data_<denType>.nii.gz:        A nii.gz file of the denoised fMRI data
    """

    # Check if denoising is needed (i.e. are there components classified as motion)
    check = den_idx.size > 0

    if check == 1:
        # Put IC indices into a char array
        if den_idx.size == 1:
            den_idx_str_join = "%d"%(den_idx + 1)
        else:
            denIdxStr = np.char.mod('%i', (den_idx + 1))
            den_idx_str_join = ','.join(denIdxStr)

        print("DENOISE STRING: " + den_idx_str_join)

        # Non-aggressive denoising of the data using fsl_regfilt (partial regression), if requested
        if den_type in ['nonaggr', 'both']:
            os.system(' '.join([os.path.join(fsl_dir, 'fsl_regfilt'),
                                '--in=' + in_file,
                                '--design=' + mel_mix,
                                '--filter="' + den_idx_str_join + '"',
                                '--out=' + os.path.join(out_dir, 'denoised_func_data_nonaggr.nii.gz')]))

        # Aggressive denoising of the data using fsl_regfilt (full regression)
        if den_type in ['aggr', 'both']:
            os.system(' '.join([os.path.join(fsl_dir, 'fsl_regfilt'),
                                '--in=' + in_file,
                                '--design=' + mel_mix,
                                '--filter="' + den_idx_str_join + '"',
                                '--out=' + os.path.join(out_dir, 'denoised_func_data_aggr.nii.gz'),
                                '-a']))
    else:
        print("  - None of the components were classified as motion, so no denoising is applied (a symbolic link to "
              "the input file will be created).")
        if den_type in ['nonaggr', 'both']:
            os.symlink(in_file, os.path.join(out_dir, 'denoised_func_data_nonaggr.nii.gz'))
        if den_type in ['aggr', 'both']:
            os.symlink(in_file, os.path.join(out_dir, 'denoised_func_data_aggr.nii.gz'))
