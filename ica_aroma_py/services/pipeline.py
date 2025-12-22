#!/usr/bin/env python

# Import required modules
import os
import subprocess
import shutil
from pathlib import Path

from . import ICA_AROMA_functions as aromafunc
accepted_denTypes = aromafunc.accepted_denTypes

def run_aroma(
    outDir,
    inFile=None,
    mc=None,
    affmat="",
    warp="",
    mask_in="",
    inFeat=None,
    TR=None,
    denType="nonaggr",
    melDir="",
    dim=0,
    overwrite=False,
    generate_plots=True,
):
    """
    Script to run ICA-AROMA v0.3 beta ('ICA-based Automatic Removal Of Motion Artifacts') on fMRI data.
    See the companion manual for further information.

    This function is the import-friendly entry point (SWANe should call this via import).

    Parameters correspond to the original CLI arguments.
    """

    print('\n------------------------------- RUNNING ICA-AROMA ------------------------------- ')
    print('--------------- \'ICA-based Automatic Removal Of Motion Artifacts\' --------------- \n')

    # Define variables based on the type of input (i.e. Feat directory or specific input arguments),
    # and check whether the specified files exist.
    cancel = False

    if inFeat:
        # Check whether the Feat directory exists
        if not os.path.isdir(inFeat):
            raise FileNotFoundError('The specified Feat directory does not exist.')

        # Define the variables which should be located in the Feat directory
        inFile = os.path.join(inFeat, 'filtered_func_data.nii.gz')
        mc = os.path.join(inFeat, 'mc', 'prefiltered_func_data_mcf.par')
        affmat = os.path.join(inFeat, 'reg', 'example_func2highres.mat')
        warp = os.path.join(inFeat, 'reg', 'highres2standard_warp.nii.gz')

        # Check whether these files actually exist
        if not os.path.isfile(inFile):
            print('Missing filtered_func_data.nii.gz in Feat directory.')
            cancel = True
        if not os.path.isfile(mc):
            print('Missing mc/prefiltered_func_data_mcf.mat in Feat directory.')
            cancel = True
        if not os.path.isfile(affmat):
            print('Missing reg/example_func2highres.mat in Feat directory.')
            cancel = True
        if not os.path.isfile(warp):
            print('Missing reg/highres2standard_warp.nii.gz in Feat directory.')
            cancel = True

        # Check whether a melodic.ica directory exists
        if os.path.isdir(os.path.join(inFeat, 'filtered_func_data.ica')):
            melDir = os.path.join(inFeat, 'filtered_func_data.ica')

    else:
        # Generic mode: inFile and mc are required
        if not inFile:
            print('No input file specified.')
            cancel = True
        else:
            if not os.path.isfile(inFile):
                print('The specified input file does not exist.')
                cancel = True

        if not mc:
            print('No mc file specified.')
            cancel = True
        else:
            if not os.path.isfile(mc):
                print('The specified mc file does does not exist.')
                cancel = True

        if affmat:
            if not os.path.isfile(affmat):
                print('The specified affmat file does not exist.')
                cancel = True

        if warp:
            if not os.path.isfile(warp):
                print('The specified warp file does not exist.')
                cancel = True

    # Parse the arguments which do not depend on whether a Feat directory has been specified
    # (keep same variable names as the original script)
    outDir = str(outDir)
    dim = int(dim)

    # Check if the mask exists, when specified.
    if mask_in:
        if not os.path.isfile(mask_in):
            print('The specified mask does not exist.')
            cancel = True

    # Check if the type of denoising is correctly specified, when specified
    if denType not in accepted_denTypes:
        print('Type of denoising was not correctly specified. Non-aggressive denoising will be run.')
        denType = 'nonaggr'

    # If the criteria for file/directory specifications have not been met. Cancel ICA-AROMA.
    if cancel:
        print('\n----------------------------- ICA-AROMA IS CANCELED -----------------------------\n')
        raise RuntimeError('ICA-AROMA was canceled due to invalid input(s).')

    #------------------------------------------- PREPARE -------------------------------------------#

    # Define the FSL-bin directory
    if "FSLDIR" not in os.environ:
        raise EnvironmentError('FSLDIR environment variable is not set. ICA-AROMA requires FSL.')
    fslDir = os.path.join(os.environ["FSLDIR"], 'bin', '')

    # Create output directory if needed
    if os.path.isdir(outDir) and overwrite is False:
        raise FileExistsError(
            'Output directory {} already exists.\n'
            'AROMA will not continue.\n'
            'Rerun with the overwrite option to explicitly overwrite existing output.'.format(outDir)
        )
    elif os.path.isdir(outDir) and overwrite is True:
        print('Warning! Output directory', outDir, 'exists and will be overwritten.\n')
        shutil.rmtree(outDir)
        os.makedirs(outDir)
    else:
        os.makedirs(outDir)

    # Get TR of the fMRI data, if not specified
    if TR is not None:
        TR = float(TR)
    else:
        cmd = ' '.join([os.path.join(fslDir, 'fslinfo'),
                        inFile,
                        '| grep pixdim4 | awk \'{print $2}\''])
        TR = float(subprocess.getoutput(cmd))

    # Check TR
    if TR == 1:
        print('Warning! Please check whether the determined TR (of ' + str(TR) + 's) is correct!\n')
    elif TR == 0:
        raise ValueError(
            'TR is zero. ICA-AROMA requires a valid TR and will therefore exit. '
            'Please check the header, or define the TR as an additional argument.'
        )

    # Define/create mask. Either by making a copy of the specified mask, or by creating a new one.
    mask = os.path.join(outDir, 'mask.nii.gz')
    if mask_in:
        shutil.copyfile(mask_in, mask)
    else:
        # If a Feat directory is specified, and an example_func is present use example_func to create a mask
        if inFeat and os.path.isfile(os.path.join(inFeat, 'example_func.nii.gz')):
            os.system(' '.join([os.path.join(fslDir, 'bet'),
                                os.path.join(inFeat, 'example_func.nii.gz'),
                                os.path.join(outDir, 'bet'),
                                '-f 0.3 -n -m -R']))
            os.system(' '.join(['mv',
                                os.path.join(outDir, 'bet_mask.nii.gz'),
                                mask]))
            if os.path.isfile(os.path.join(outDir, 'bet.nii.gz')):
                os.remove(os.path.join(outDir, 'bet.nii.gz'))
        else:
            if inFeat:
                print(' - No example_func was found in the Feat directory. A mask will be created including all voxels with varying intensity over time in the fMRI data. Please check!\n')
            os.system(' '.join([os.path.join(fslDir, 'fslmaths'),
                                inFile,
                                '-Tstd -bin',
                                mask]))

    #---------------------------------------- Run ICA-AROMA ----------------------------------------#

    print('Step 1) MELODIC')
    aromafunc.runICA(fslDir, inFile, outDir, melDir, mask, dim, TR)

    print('Step 2) Automatic classification of the components')
    print('  - registering the spatial maps to MNI')
    melIC = os.path.join(outDir, 'melodic_IC_thr.nii.gz')
    melIC_MNI = os.path.join(outDir, 'melodic_IC_thr_MNI2mm.nii.gz')
    aromafunc.register2MNI(fslDir, melIC, melIC_MNI, affmat, warp)

    print('  - extracting the CSF & Edge fraction features')
    # Determine the ICA-AROMA resources directory (mask files)
    aromaDir = Path(__file__).resolve().parents[1] / 'resources'
    edgeFract, csfFract = aromafunc.feature_spatial(fslDir, outDir, str(aromaDir), melIC_MNI)

    print('  - extracting the Maximum RP correlation feature')
    melmix = os.path.join(outDir, 'melodic.ica', 'melodic_mix')
    maxRPcorr = aromafunc.feature_time_series(melmix, mc)

    print('  - extracting the High-frequency content feature')
    melFTmix = os.path.join(outDir, 'melodic.ica', 'melodic_FTmix')
    HFC = aromafunc.feature_frequency(melFTmix, TR)

    print('  - classification')
    motionICs = aromafunc.classification(outDir, maxRPcorr, edgeFract, HFC, csfFract)

    if generate_plots:
        try:
            from .classification_plots import classification_plot
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "You requested the generation of plots, but the 'plots' dependencies are not installed.\n"
                "Install the extra with:\n"
                "  pip install ica-aroma-py[plots]\n"
                "Or disable plots with -np / --noplots."
            ) from e
            
        classification_plot(os.path.join(outDir, 'classification_overview.txt'),
                            outDir)

    if (denType != 'no'):
        print('Step 3) Data denoising')
        aromafunc.denoising(fslDir, inFile, outDir, melmix, denType, motionICs)

    print('\n----------------------------------- Finished -----------------------------------\n')

    # Return some useful outputs
    return {
        "outDir": outDir,
        "TR": TR,
        "melodic_dir": os.path.join(outDir, 'melodic.ica'),
        "motionICs": motionICs,
        "denoised_nonaggr": os.path.join(outDir, 'denoised_func_data_nonaggr.nii.gz'),
        "denoised_aggr": os.path.join(outDir, 'denoised_func_data_aggr.nii.gz'),
    }