# Import required modules
from nipype.pipeline.engine import Node, Workflow
from nipype.pipeline.plugins import MultiProcPlugin
from .ICA_AROMA_nodes import (GetNiftiTR, FslNVols, IsoResample, FeatureTimeSeries, FeatureFrequency,
                              AromaClassification, AromaClassificationPlot)
from nipype.interfaces.fsl import (BET, ImageMaths, MELODIC, ExtractROI, Merge as fslMerge, ApplyMask, ApplyXFM,
                                   ApplyWarp, UnaryMaths, ImageStats, Split, FilterRegressor)
from nipype import SelectFiles, MapNode, IdentityInterface, Merge
import os
import argparse
from pathlib import Path


# Denoising types accepted
accepted_denTypes = {'nonaggr', 'aggr', 'both', 'no'}

def generate_aroma_workflow(
    outDir,
    inFile=None,
    mc=None,
    affmat="",
    warp="",
    mask_in="",
    inFeat=None,
    TR=None,
    denType="nonaggr",
    melDirIn="",
    dim=0,
    generate_plots=True,
    aroma_workflow=None
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
            melDirIn = os.path.join(inFeat, 'filtered_func_data.ica')

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

    if aroma_workflow is None:
        aroma_workflow = Workflow(name="ica-aroma", base_dir=outDir)

    # Get TR of the fMRI data, if not specified
    get_tr = Node(GetNiftiTR(), name="get_fmri_tr")
    get_tr.inputs.in_file = inFile
    if TR is not None:
        get_tr.inputs.force_tr = TR

    # Define/create mask. Either by making a copy of the specified mask, or by creating a new one.
    mask = Node(IdentityInterface(fields=['mask'], mandatory_inputs=True), name="mask")
    if mask_in:
        mask.inputs.mask = mask_in
    else:
        if inFeat and os.path.isfile(os.path.join(inFeat, 'example_func.nii.gz')):
            mask_bet = Node(BET(), name="create_mask_bet")
            mask_bet.inputs.in_file = os.path.join(inFeat, 'example_func.nii.gz')
            mask_bet.inputs.out_file = "brain.nii.gz"
            mask_bet.inputs.mask = True
            mask_bet.inputs.robust = True
            mask_bet.inputs.frac = 0.3
            aroma_workflow.connect(mask_bet, "mask_file", mask, "mask")

        else:
            mask_maths = Node(ImageMaths(), name="create_mask_maths")
            mask_maths.inputs.in_file = inFile
            mask_maths.inputs.op_string = '-Tstd -bin'
            mask_maths.inputs.out_file = "brain_mask.nii.gz"
            aroma_workflow.connect(mask_maths, "out_file", mask, "mask")


    #---------------------------------------- Run ICA-AROMA ----------------------------------------#

    melIC = os.path.join(melDirIn, 'melodic_IC.nii.gz')
    melICmix = os.path.join(melDirIn, 'melodic_mix')

    print('Step 1) MELODIC')
    # When a MELODIC directory is specified,
    # check whether all needed files are present.
    # Otherwise... run MELODIC again
    if len(melDirIn) != 0 and os.path.isfile(melIC) and os.path.isfile(os.path.join(melDirIn, 'melodic_FTmix')) and os.path.isfile(melICmix):

        print('  - The existing/specified MELODIC directory will be used.')

        # If a 'stats' directory is present (contains thresholded spatial maps)
        # create a symbolic link to the MELODIC directory.
        # Otherwise, create specific links and
        # run mixture modeling to obtain thresholded maps.
        if os.path.isdir(os.path.join(melDirIn, 'stats')):
            melodic = Node(IdentityInterface(fields=['out_dir'], mandatory_inputs=True), name="melodic_fake")
            melodic.inputs.out_dir = melDirIn
        else:
            print(
                '  - The MELODIC directory does not contain the required \'stats\' folder. Mixture modeling on the Z-statistical maps will be run.')
            # Run mixture modeling
            # TODO: check if we need to specify to run mixture modeling in original melodic dir
            melodic = Node(MELODIC(), name="melodic")
            melodic.inputs.in_files = [melIC]
            melodic.inputs.ICs = melIC
            melodic.inputs.mix = melICmix
            melodic.inputs.mm_thresh = 0.5
            melodic.inputs.out_stats = True
    else:
        # If a melodic directory was specified,
        # display that it did not contain all files needed for ICA-AROMA (or that the directory does not exist at all)
        if len(melDirIn) != 0:
            if not os.path.isdir(melDirIn):
                print('  - The specified MELODIC directory does not exist. MELODIC will be run seperately.')
            else:
                print(
                    '  - The specified MELODIC directory does not contain the required files to run ICA-AROMA. MELODIC will be run seperately.')

        # Run MELODIC
        melodic = Node(MELODIC(), name="melodic")
        melodic.inputs.in_files = [inFile]
        melodic.inputs.mm_thresh = 0.5
        melodic.inputs.dim = dim
        melodic.inputs.out_stats = True
        melodic.inputs.no_bet = True
        melodic.inputs.report = True
        aroma_workflow.connect(mask, "mask", melodic, "mask")
        aroma_workflow.connect(get_tr, "TR", melodic, "tr_sec")

    # Select useful Melodic output files
    templates = dict(IC="melodic_IC.nii.gz",
                     mix="melodic_mix",
                     melFTmix="melodic_FTmix",
                     thresh_zstat_files="stats/thresh_zstat*.nii.gz")

    melodic_output = Node(SelectFiles(templates), name="melodic_output")
    melodic_output.inputs.sorted = True
    aroma_workflow.connect(melodic, "out_dir", melodic_output, "melodic_dir")
    aroma_workflow.connect(melodic, "out_dir", melodic_output, "base_directory")

    getICn = Node(FslNVols(), name="getICn")
    aroma_workflow.connect(melodic_output, "IC", getICn, "in_file")

    # Merge mixture modeled thresholded spatial maps.
    # Note! In case that mixture modeling did not converge, the file will contain two spatial maps.
    # The latter being the results from a simple null hypothesis test.
    # In that case, this map will have to be used (first one will be empty).
    getZstatn = MapNode(FslNVols(),
                        name="getZstatn",
                        iterfield=["in_file"])
    aroma_workflow.connect(melodic_output, "thresh_zstat_files", getZstatn, "in_file")

    def reduce_n_by_1(voln_list):
        return [*map(lambda x: x - 1, voln_list)]

    lastZstat = MapNode(ExtractROI(),
                        name="lastZstat",
                        iterfield=["in_file", "t_min"],
                        #synchronize=True
                        )
    lastZstat.inputs.t_size = 1
    aroma_workflow.connect(melodic_output, "thresh_zstat_files", lastZstat, "in_file")
    aroma_workflow.connect(getZstatn, ("nvols", reduce_n_by_1), lastZstat, "t_min")

    mergeZstat = Node(fslMerge(), name="mergeZstat")
    mergeZstat.inputs.dimension = 't'
    aroma_workflow.connect(lastZstat, "roi_file", mergeZstat, "in_files")

    maskZstat = Node(ApplyMask(), name="maskZstat")
    aroma_workflow.connect(mergeZstat, "merged_file", maskZstat, "in_file")
    aroma_workflow.connect(mask, "mask", maskZstat, "mask_file")

    print('Step 2) Automatic classification of the components')
    print('  - registering the spatial maps to MNI')

    # Define the MNI152 T1 2mm template
    fslnobin = fslDir.rsplit('/', 2)[0]
    ref = os.path.join(fslnobin, 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')

    registeredFileNode = Node(IdentityInterface(fields=['registered_file'], mandatory_inputs=True), name="registeredFileNode")

    # If the no affmat- or warp-file has been specified, assume that the data is already in MNI152 space.
    # In that case, only check if resampling to 2mm is needed
    if (len(affmat) == 0) and (len(warp) == 0):
        fileResample = Node(IsoResample(), name="fileResample")
        fileResample.inputs.reference = ref
        fileResample.inputs.dim = 2
        aroma_workflow.connect(maskZstat, "out_file", fileResample, "in_file")
        aroma_workflow.connect(fileResample, "out_file", registeredFileNode, "registered_file")

    # If a warp-file has been specified, apply it and an eventual affmat provided
    elif len(warp) != 0:
        # Apply warp
        applyWarp = Node(ApplyWarp(), name="applyWarp")
        applyWarp.inputs.ref_file = ref
        applyWarp.inputs.field_file = warp
        if len(affmat) != 0:
            applyWarp.inputs.premat = affmat
        applyWarp.inputs.interp = "trilinear"
        aroma_workflow.connect(maskZstat, "out_file", applyWarp, "in_file")
        aroma_workflow.connect(applyWarp, "out_file", registeredFileNode, "registered_file")

    # If only an affmat-file has been specified, perform affine registration to MNI
    else:
        applyMat = Node(ApplyXFM(), name="applyMat")
        applyMat.inputs.reference = ref
        applyMat.inputs.apply_xfm = True
        applyMat.inputs.in_matrix_file = affmat
        applyMat.inputs.interp = "trilinear"
        aroma_workflow.connect(maskZstat, "out_file", applyMat, "in_file")
        aroma_workflow.connect(applyMat, "out_file", registeredFileNode, "registered_file")

    print('  - extracting the CSF & Edge fraction features')

    # Define the mask files (do NOT rely on the current working directory)
    aromaDir = Path(__file__).resolve().parents[1] / 'resources'
    mask_csf = os.path.join(aromaDir, 'mask_csf.nii.gz')
    mask_edge = os.path.join(aromaDir, 'mask_edge.nii.gz')
    mask_out = os.path.join(aromaDir, 'mask_out.nii.gz')

    # Check whether the masks exist
    if not os.path.isfile(mask_csf):
        raise FileNotFoundError('The specified CSF mask does not exist: ' + mask_csf)
    if not os.path.isfile(mask_edge):
        raise FileNotFoundError('The specified edge mask does not exist: ' + mask_edge)
    if not os.path.isfile(mask_out):
        raise FileNotFoundError('The specified outside-brain mask does not exist: ' + mask_out)

    re_split = Node(Split(), name="re_split")
    re_split.inputs.dimension = "t"
    aroma_workflow.connect(registeredFileNode, "registered_file", re_split, "in_file")

    absValue = Node(UnaryMaths(), name="absValue")
    absValue.inputs.operation = "abs"
    aroma_workflow.connect(registeredFileNode, "registered_file", absValue, "in_file")

    totStat = Node(ImageStats(), name="totStat")
    totStat.inputs.op_string = "-M -V"
    totStat.inputs.split_4d = True
    aroma_workflow.connect(absValue, "out_file", totStat, "in_file")

    apply_csf_mask = Node(ApplyMask(), name="apply_csf_mask")
    apply_csf_mask.inputs.mask_file = mask_csf
    aroma_workflow.connect(absValue, "out_file", apply_csf_mask, "in_file")

    csfStat = Node(ImageStats(), name="csfStat")
    csfStat.inputs.op_string = "-M -V"
    csfStat.inputs.split_4d = True
    aroma_workflow.connect(apply_csf_mask, "out_file", csfStat, "in_file")

    apply_edge_mask = Node(ApplyMask(), name="apply_edge_mask")
    apply_edge_mask.inputs.mask_file = mask_edge
    aroma_workflow.connect(absValue, "out_file", apply_edge_mask, "in_file")

    edgeStat = Node(ImageStats(), name="edgeStat")
    edgeStat.inputs.op_string = "-M -V"
    edgeStat.inputs.split_4d = True
    aroma_workflow.connect(apply_edge_mask, "out_file", edgeStat, "in_file")

    apply_out_mask = Node(ApplyMask(), name="apply_out_mask")
    apply_out_mask.inputs.mask_file = mask_out
    aroma_workflow.connect(absValue, "out_file", apply_out_mask, "in_file")

    outStat = Node(ImageStats(), name="outStat")
    outStat.inputs.op_string = "-M -V"
    outStat.inputs.split_4d = True
    aroma_workflow.connect(apply_out_mask, "out_file", outStat, "in_file")

    mergeStat = Node(Merge(4), name="mergeStat")
    mergeStat.inputs.axis = "hstack"
    aroma_workflow.connect(totStat, "out_stat", mergeStat, "in1")
    aroma_workflow.connect(csfStat, "out_stat", mergeStat, "in2")
    aroma_workflow.connect(edgeStat, "out_stat", mergeStat, "in3")
    aroma_workflow.connect(outStat, "out_stat", mergeStat, "in4")

    print('  - extracting the Maximum RP correlation feature')
    feature_time_series = Node(FeatureTimeSeries(), name="feature_time_series")
    feature_time_series.inputs.mc = mc
    aroma_workflow.connect(melodic_output, "mix", feature_time_series, "melmix")

    print('  - extracting the High-frequency content feature')
    feature_frequency = Node(FeatureFrequency(), name="feature_frequency")
    aroma_workflow.connect(get_tr, "TR", feature_frequency, "TR")
    aroma_workflow.connect(melodic_output, "melFTmix", feature_frequency, "melFTmix")

    print('  - classification')
    aroma_classification = Node(AromaClassification(), name="aroma_classification")
    aroma_workflow.connect(feature_frequency, "HFC", aroma_classification, "HFC")
    aroma_workflow.connect(feature_time_series, "maxRPcorr", aroma_classification, "maxRPcorr")

    def calc_edge_fract(stats):
        import numpy as np
        edgeFract = np.zeros(len(stats))
        i = 0
        for stat in stats:
            totSum = stat[0][0]*stat[0][1]
            csfSum = stat[1][0]*stat[1][1]
            edgeSum = stat[2][0]*stat[2][1]
            outSum = stat[3][0]*stat[3][1]
            if not (totSum == 0):
                edgeFract[i] = (outSum + edgeSum) / (totSum - csfSum) if not ((totSum - csfSum) == 0) else 0
            else:
                edgeFract[i] = 0
            i = i+1
        return edgeFract

    def calc_csf_fract(stats):
        import numpy as np
        csfFract = np.zeros(len(stats))
        i = 0
        for stat in stats:
            totSum = stat[0][0]*stat[0][1]
            csfSum = stat[1][0]*stat[1][1]
            if not (totSum == 0):
                csfFract[i] = csfSum / totSum
            else:
                csfFract[i] = 0
            i = i+1
        return csfFract

    aroma_workflow.connect(mergeStat, ("out", calc_csf_fract), aroma_classification, "csfFract")
    aroma_workflow.connect(mergeStat, ("out", calc_edge_fract), aroma_classification, "edgeFract")

    if (denType == 'nonaggr') or (denType == 'both'):
        nonaggr_denoising = Node(FilterRegressor(), name="nonaggr_denoising")
        nonaggr_denoising.inputs.in_file = inFile
        #nonaggr_denoising.inputs.out_file = "denoised_func_data_nonaggr.nii.gz"
        aroma_workflow.connect(melodic_output, "mix", nonaggr_denoising, "design_file")
        aroma_workflow.connect(aroma_classification, "motionICs", nonaggr_denoising, "filter_columns")

    if (denType == 'aggr') or (denType == 'both'):
        aggr_denoising = Node(FilterRegressor(), name="aggr_denoising")
        aggr_denoising.inputs.in_file = inFile
        #aggr_denoising.inputs.out_file = "denoised_func_data_aggr.nii.gz"
        aggr_denoising.inputs.args = "-a"
        aroma_workflow.connect(melodic_output, "mix", aggr_denoising, "design_file")
        aroma_workflow.connect(aroma_classification, "motionICs", aggr_denoising, "filter_columns")

    if generate_plots:
        aroma_classification_plot = Node(AromaClassificationPlot(), name="aroma_classification_plot")
        aroma_workflow.connect(aroma_classification, "classification_overview", aroma_classification_plot, "classification_overview_file")

    return aroma_workflow

    
def run_aroma_workflow(aroma_workflow):

    plugin_args = {
        "mp_context": "fork",
        "n_procs": 20,
    }

    aroma_workflow.config['execution'] = {'remove_unnecessary_outputs': 'False',
                                      'keep_inputs': 'True'}
    aroma_workflow.write_graph(graph2use='exec')

    aroma_workflow.run(plugin=MultiProcPlugin(plugin_args=plugin_args))

# -------------------------------------------- PARSER --------------------------------------------#

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Script to run ICA-AROMA v0.3 beta ('ICA-based Automatic Removal Of Motion Artifacts') on fMRI data. "
                    "See the companion manual for further information."
    )

    # Required options
    reqoptions = parser.add_argument_group('Required arguments')
    reqoptions.add_argument('-o', '-out', dest="outDir", required=True, help='Output directory name')

    # Required options in non-Feat mode
    nonfeatoptions = parser.add_argument_group('Required arguments - generic mode')
    nonfeatoptions.add_argument('-i', '-in', dest="inFile", required=False, help='Input file name of fMRI data (.nii.gz)')
    nonfeatoptions.add_argument('-mc', dest="mc", required=False, help='File name of the motion parameters obtained after motion realingment (e.g., FSL mcflirt). Note that the order of parameters does not matter, should your file not originate from FSL mcflirt. (e.g., /home/user/PROJECT/SUBJECT.feat/mc/prefiltered_func_data_mcf.par')
    nonfeatoptions.add_argument('-a', '-affmat', dest="affmat", default="", help='File name of the mat-file describing the affine registration (e.g., FSL FLIRT) of the functional data to structural space (.mat file). (e.g., /home/user/PROJECT/SUBJECT.feat/reg/example_func2highres.mat')
    nonfeatoptions.add_argument('-w', '-warp', dest="warp", default="", help='File name of the warp-file describing the non-linear registration (e.g., FSL FNIRT) of the structural data to MNI152 space (.nii.gz). (e.g., /home/user/PROJECT/SUBJECT.feat/reg/highres2standard_warp.nii.gz)')
    nonfeatoptions.add_argument('-m', '-mask', dest="mask", default="", help='File name of the mask to be used for MELODIC (denoising will be performed on the original/non-masked input data)')

    # Required options in Feat mode
    featoptions = parser.add_argument_group('Required arguments - FEAT mode')
    featoptions.add_argument('-f', '-feat', dest="inFeat", required=False, help='Feat directory name (Feat should have been run without temporal filtering and including registration to MNI152)')

    # Optional options
    optoptions = parser.add_argument_group('Optional arguments')
    optoptions.add_argument('-tr', dest="TR", help='TR in seconds', type=float)
    optoptions.add_argument('-den', dest="denType", default="nonaggr", help="Type of denoising strategy: 'no': only classification, no denoising; 'nonaggr': non-aggresssive denoising (default); 'aggr': aggressive denoising; 'both': both aggressive and non-aggressive denoising (seperately)")
    optoptions.add_argument('-md', '-meldir', dest="melDir", default="", help='MELODIC directory name, in case MELODIC has been run previously.')
    optoptions.add_argument('-dim', dest="dim", default=0, help='Dimensionality reduction into #num dimensions when running MELODIC (default: automatic estimation; i.e. -dim 0)', type=int)
    optoptions.add_argument('-ow', '-overwrite', dest="overwrite", action='store_true', help='Overwrite existing output', default=False)
    optoptions.add_argument('-np', '-noplots', dest="generate_plots", action='store_false', help='Plot component classification overview similar to plot in the main AROMA paper', default=True)

    return parser


def main():
    # Keep the original CLI behavior, but delegate all logic to run_aroma()
    parser = _build_arg_parser()
    args = parser.parse_args()

    aroma_workflow = generate_aroma_workflow(
        outDir=args.outDir,
        inFile=args.inFile,
        mc=args.mc,
        affmat=args.affmat,
        warp=args.warp,
        mask_in=args.mask,
        inFeat=args.inFeat,
        TR=args.TR,
        denType=args.denType,
        melDirIn=args.melDir,
        dim=args.dim,
        generate_plots=args.generate_plots,
    )

    run_aroma_workflow(aroma_workflow)


# allow use of module on its own
if __name__ == '__main__':
    main()
