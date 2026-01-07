# Import required modules
from nipype.pipeline.engine import Node, Workflow
from nipype.pipeline.plugins import MultiProcPlugin
from .ICA_AROMA_nodes import (GetNiftiTR, IsoResample, FeatureTimeSeries, FeatureFrequency,
                              AromaClassification, AromaClassificationPlot, FeatureSpatial, FeatureSpatialPrep)
from nipype.interfaces.fsl import (BET, ImageMaths, MELODIC, ApplyXFM,
                                   ApplyWarp, FilterRegressor)
from nipype import SelectFiles, IdentityInterface, DataSink
import os
from .ICA_AROMA_functions import accepted_den_types
from .. import aroma_mask_out, aroma_mask_edge, aroma_mask_csf

def generate_aroma_workflow(
    out_dir,
    in_file=None,
    mc=None,
    aff_mat="",
    warp="",
    mask_in="",
    in_feat=None,
    TR=None,
    den_type="nonaggr",
    mel_dir_in="",
    dim=0,
    generate_plots=True,
    aroma_workflow=None,
    result_dir=None,
):
    """
    Script to run ICA-AROMA v0.3 beta ('ICA-based Automatic Removal Of Motion Artifacts') on fMRI data.
    See the companion manual for further information.

    This function is the import-friendly entry point (SWANe should call this via import).

    Parameters correspond to the original CLI arguments.
    """

    print('\n------------------------------- RUNNING ICA-AROMA ------------------------------- ')

    # Define variables based on the type of input (i.e. Feat directory or specific input arguments),
    # and check whether the specified files exist.
    cancel = False

    inputnode = Node(IdentityInterface(fields=[
        "in_file",
        "mc",
        "aff_mat",
        "warp",
        "",
        "",
    ]), name="inputnode")

    if in_feat:
        # Check whether the Feat directory exists
        if not os.path.isdir(in_feat):
            raise FileNotFoundError('The specified Feat directory does not exist.')

        # Define the variables which should be located in the Feat directory
        in_file = os.path.join(in_feat, 'filtered_func_data.nii.gz')
        mc = os.path.join(in_feat, 'mc', 'prefiltered_func_data_mcf.par')
        aff_mat = os.path.join(in_feat, 'reg', 'example_func2highres.mat')
        warp = os.path.join(in_feat, 'reg', 'highres2standard_warp.nii.gz')

        # Check whether these files actually exist
        if not os.path.isfile(in_file):
            print('Missing filtered_func_data.nii.gz in Feat directory.')
            cancel = True
        if not os.path.isfile(mc):
            print('Missing mc/prefiltered_func_data_mcf.mat in Feat directory.')
            cancel = True
        if not os.path.isfile(aff_mat):
            print('Missing reg/example_func2highres.mat in Feat directory.')
            cancel = True
        if not os.path.isfile(warp):
            print('Missing reg/highres2standard_warp.nii.gz in Feat directory.')
            cancel = True

        # Check whether a melodic.ica directory exists
        if os.path.isdir(os.path.join(in_feat, 'filtered_func_data.ica')):
            mel_dir_in = os.path.join(in_feat, 'filtered_func_data.ica')

    else:
        # Generic mode: inFile and mc are required
        if not in_file:
            print('No input file specified.')
            cancel = True
        else:
            if not os.path.isfile(in_file):
                print('The specified input file does not exist.')
                cancel = True

        if not mc:
            print('No mc file specified.')
            cancel = True
        else:
            if not os.path.isfile(mc):
                print('The specified mc file does does not exist.')
                cancel = True

        if aff_mat:
            if not os.path.isfile(aff_mat):
                print('The specified affmat file does not exist.')
                cancel = True

        if warp:
            if not os.path.isfile(warp):
                print('The specified warp file does not exist.')
                cancel = True

    # Parse the arguments which do not depend on whether a Feat directory has been specified
    out_dir = str(out_dir)
    dim = int(dim)

    # Check if the mask exists, when specified.
    if mask_in:
        if not os.path.isfile(mask_in):
            print('The specified mask does not exist.')
            cancel = True

    # Check if the type of denoising is correctly specified, when specified
    if den_type not in accepted_den_types:
        print('Type of denoising was not correctly specified. Non-aggressive denoising will be run.')
        den_type = 'nonaggr'

    # If the criteria for file/directory specifications have not been met. Cancel ICA-AROMA.
    if cancel:
        print('\n----------------------------- ICA-AROMA IS CANCELED -----------------------------\n')
        raise RuntimeError('ICA-AROMA was canceled due to invalid input(s).')

    # ------------------------------------------- PREPARE -------------------------------------------#

    # Define the FSL-bin directory
    if "FSLDIR" not in os.environ:
        raise EnvironmentError('FSLDIR environment variable is not set. ICA-AROMA requires FSL.')
    fsl_dir = os.path.join(os.environ["FSLDIR"], 'bin', '')

    if aroma_workflow is None:
        aroma_workflow = Workflow(name="ica_aroma_execution", base_dir=out_dir)

    aroma_datasink = None
    if result_dir is not None:
        aroma_datasink = Node(DataSink(), name="aroma_datasink")
        aroma_datasink.inputs.base_directory = result_dir

        # Get TR of the fMRI data, if not specified
    get_tr = Node(GetNiftiTR(), name="get_fmri_tr")
    get_tr.inputs.in_file = in_file
    if TR is not None:
        get_tr.inputs.force_tr = TR

    # Define/create mask. Either by making a copy of the specified mask, or by creating a new one.
    mask = Node(IdentityInterface(fields=['mask'], mandatory_inputs=True), name="mask")
    if mask_in:
        mask.inputs.mask = mask_in
    else:
        if in_feat and os.path.isfile(os.path.join(in_feat, 'example_func.nii.gz')):
            mask_bet = Node(BET(), name="create_mask_bet")
            mask_bet.inputs.in_file = os.path.join(in_feat, 'example_func.nii.gz')
            mask_bet.inputs.out_file = "brain.nii.gz"
            mask_bet.inputs.mask = True
            mask_bet.inputs.robust = True
            mask_bet.inputs.frac = 0.3
            aroma_workflow.connect(mask_bet, "mask_file", mask, "mask")

        else:
            mask_maths = Node(ImageMaths(), name="create_mask_maths")
            mask_maths.inputs.in_file = in_file
            mask_maths.inputs.op_string = '-Tstd -bin'
            mask_maths.inputs.out_file = "brain_mask.nii.gz"
            aroma_workflow.connect(mask_maths, "out_file", mask, "mask")

    # ---------------------------------------- Run ICA-AROMA ---------------------------------------- #

    mel_ic = os.path.join(mel_dir_in, 'melodic_IC.nii.gz')
    mel_ic_mix = os.path.join(mel_dir_in, 'melodic_mix')

    # When a MELODIC directory is specified,
    # check whether all needed files are present.
    # Otherwise... run MELODIC again
    if (len(mel_dir_in) != 0 and os.path.isfile(mel_ic)
            and os.path.isfile(os.path.join(mel_dir_in, 'melodic_FTmix'))
            and os.path.isfile(mel_ic_mix)):

        print('  - The existing/specified MELODIC directory will be used.')

        # If a 'stats' directory is present (contains thresholded spatial maps)
        # create a symbolic link to the MELODIC directory.
        # Otherwise, create specific links and
        # run mixture modeling to obtain thresholded maps.
        if os.path.isdir(os.path.join(mel_dir_in, 'stats')):
            melodic = Node(IdentityInterface(fields=['out_dir'], mandatory_inputs=True), name="melodic_fake")
            melodic.inputs.out_dir = mel_dir_in
        else:
            print('  - The MELODIC directory does not contain the required \'stats\' folder. Mixture modeling on '
                  'the Z-statistical maps will be run.')
            # Run mixture modeling
            # TODO: check if we need to specify to run mixture modeling in original melodic dir
            melodic = Node(MELODIC(), name="melodic")
            melodic.inputs.in_files = [mel_ic]
            melodic.inputs.ICs = mel_ic
            melodic.inputs.mix = mel_ic_mix
            melodic.inputs.mm_thresh = 0.5
            melodic.inputs.out_stats = True
    else:
        # If a melodic directory was specified,
        # display that it did not contain all files needed for ICA-AROMA (or that the directory does not exist at all)
        if len(mel_dir_in) != 0:
            if not os.path.isdir(mel_dir_in):
                print('  - The specified MELODIC directory does not exist. MELODIC will be run seperately.')
            else:
                print(
                    '  - The specified MELODIC directory does not contain the required files to run ICA-AROMA. '
                    'MELODIC will be run seperately.')

        # Run MELODIC
        melodic = Node(MELODIC(), name="melodic")
        melodic.inputs.in_files = [in_file]
        melodic.inputs.mm_thresh = 0.5
        melodic.inputs.dim = dim
        melodic.inputs.out_stats = True
        melodic.inputs.no_bet = True
        melodic.inputs.report = True
        aroma_workflow.connect(mask, "mask", melodic, "mask")
        aroma_workflow.connect(get_tr, "TR", melodic, "tr_sec")

    # Select useful Melodic output files
    templates = dict(IC="melodic_IC.nii.gz",
                     mel_mix="melodic_mix",
                     mel_ft_mix="melodic_FTmix",
                     thresh_zstat_files="stats/thresh_zstat*.nii.gz")

    melodic_output = Node(SelectFiles(templates), name="melodic_output")
    melodic_output.inputs.sorted = True
    aroma_workflow.connect(melodic, "out_dir", melodic_output, "melodic_dir")
    aroma_workflow.connect(melodic, "out_dir", melodic_output, "base_directory")

    feature_spatial_prep = Node(FeatureSpatialPrep(), name="feature_spatial_prep")
    aroma_workflow.connect(melodic_output, "thresh_zstat_files", feature_spatial_prep, "in_files")
    aroma_workflow.connect(mask, "mask", feature_spatial_prep, "mask_file")

    # Define the MNI152 T1 2mm template
    fsl_no_bin = fsl_dir.rsplit('/', 2)[0]
    ref = os.path.join(fsl_no_bin, 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')

    registered_file_node = Node(IdentityInterface(fields=['registered_file'], mandatory_inputs=True),
                              name="registered_file_node")

    # If the no affmat-file or warp-file has been specified, assume that the data is already in MNI152 space.
    # In that case, only check if resampling to 2mm is needed
    if (len(aff_mat) == 0) and (len(warp) == 0):
        file_resample = Node(IsoResample(), name="file_resample")
        file_resample.inputs.reference = ref
        file_resample.inputs.dim = 2
        file_resample.inputs.out_file = "melodic_IC_thr_MNI2mm.nii.gz"
        aroma_workflow.connect(feature_spatial_prep, "out_file", file_resample, "in_file")
        aroma_workflow.connect(file_resample, "out_file", registered_file_node, "registered_file")

    # If a warp-file has been specified, apply it and an eventual affmat provided
    elif len(warp) != 0:
        # Apply warp
        applyWarp = Node(ApplyWarp(), name="applyWarp")
        applyWarp.inputs.ref_file = ref
        applyWarp.inputs.field_file = warp
        applyWarp.inputs.out_file = "melodic_IC_thr_MNI2mm.nii.gz"
        if len(aff_mat) != 0:
            applyWarp.inputs.premat = aff_mat
        applyWarp.inputs.interp = "trilinear"
        aroma_workflow.connect(feature_spatial_prep, "out_file", applyWarp, "in_file")
        aroma_workflow.connect(applyWarp, "out_file", registered_file_node, "registered_file")

    # If only an affmat-file has been specified, perform affine registration to MNI
    else:
        apply_mat = Node(ApplyXFM(), name="apply_mat")
        apply_mat.inputs.reference = ref
        apply_mat.inputs.apply_xfm = True
        apply_mat.inputs.in_matrix_file = aff_mat
        apply_mat.inputs.interp = "trilinear"
        apply_mat.inputs.out_file = "melodic_IC_thr_MNI2mm.nii.gz"
        aroma_workflow.connect(feature_spatial_prep, "out_file", apply_mat, "in_file")
        aroma_workflow.connect(apply_mat, "out_file", registered_file_node, "registered_file")

    if aroma_datasink is not None:
        aroma_workflow.connect(registered_file_node, "registered_file", aroma_datasink, "ica_aroma_results.@reg")

    feature_spatial = Node(FeatureSpatial(), name="feature_spatial")
    feature_spatial.inputs.mask_csf =  aroma_mask_csf
    feature_spatial.inputs.mask_edge = aroma_mask_edge
    feature_spatial.inputs.mask_out = aroma_mask_out
    aroma_workflow.connect(registered_file_node, "registered_file", feature_spatial, "in_file")

    feature_time_series = Node(FeatureTimeSeries(), name="feature_time_series")
    feature_time_series.inputs.mc = mc
    aroma_workflow.connect(melodic_output, "mel_mix", feature_time_series, "mel_mix")

    feature_frequency = Node(FeatureFrequency(), name="feature_frequency")
    aroma_workflow.connect(get_tr, "TR", feature_frequency, "TR")
    aroma_workflow.connect(melodic_output, "mel_ft_mix", feature_frequency, "mel_ft_mix")

    aroma_classification = Node(AromaClassification(), name="aroma_classification")
    aroma_workflow.connect(feature_frequency, "HFC", aroma_classification, "HFC")
    aroma_workflow.connect(feature_time_series, "max_rp_corr", aroma_classification, "max_rp_corr")
    aroma_workflow.connect(feature_spatial, "csf_fract", aroma_classification, "csf_fract")
    aroma_workflow.connect(feature_spatial, "edge_fract", aroma_classification, "edge_fract")
    if aroma_datasink is not None:
        aroma_workflow.connect(aroma_classification, "feature_scores",
                               aroma_datasink, "ica_aroma_results.@feature_scores")
        aroma_workflow.connect(aroma_classification, "classified_motion_ics",
                               aroma_datasink, "ica_aroma_results.@classified_motion_ics")
        aroma_workflow.connect(aroma_classification, "classification_overview",
                               aroma_datasink, "ica_aroma_results.@classification_overview")

    if den_type in ['nonaggr', 'both']:
        nonaggr_denoising = Node(FilterRegressor(), name="nonaggr_denoising", mem_gb=5)
        nonaggr_denoising.inputs.in_file = in_file
        nonaggr_denoising.inputs.out_file = "denoised_func_data_nonaggr.nii.gz"
        aroma_workflow.connect(melodic_output, "mel_mix", nonaggr_denoising, "design_file")
        aroma_workflow.connect(aroma_classification, "motion_ics", nonaggr_denoising, "filter_columns")
        if aroma_datasink is not None:
            aroma_workflow.connect(nonaggr_denoising, "out_file", aroma_datasink, "ica_aroma_results.@noaggr")

    if den_type in ['aggr', 'both']:
        aggr_denoising = Node(FilterRegressor(), name="aggr_denoising", mem_gb=5)
        aggr_denoising.inputs.in_file = in_file
        aggr_denoising.inputs.out_file = "denoised_func_data_aggr.nii.gz"
        aggr_denoising.inputs.args = "-a"
        aroma_workflow.connect(melodic_output, "mel_mix", aggr_denoising, "design_file")
        aroma_workflow.connect(aroma_classification, "motion_ics", aggr_denoising, "filter_columns")
        if aroma_datasink is not None:
            aroma_workflow.connect(aggr_denoising, "out_file", aroma_datasink, "ica_aroma_results.@aggr")

    if generate_plots:
        aroma_classification_plot = Node(AromaClassificationPlot(), name="aroma_classification_plot")
        aroma_workflow.connect(aroma_classification, "classification_overview",
                               aroma_classification_plot, "classification_overview_file")
        if aroma_datasink is not None:
            aroma_workflow.connect(aroma_classification_plot, "out_file", aroma_datasink, "ica_aroma_results.@plot")

    return aroma_workflow

def run_aroma_workflow(aroma_workflow, plugin_args=None):

    if plugin_args is None:
        plugin_args = {"mp_context": "fork"}

    aroma_workflow.config["execution"]["crashdump_dir"] = aroma_workflow.base_dir

    aroma_workflow.write_graph(graph2use='exec')
    aroma_workflow.run(plugin=MultiProcPlugin(plugin_args=plugin_args))
    