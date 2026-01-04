#!/usr/bin/env python

import argparse

from .pipeline import run_aroma, accepted_den_types

# -------------------------------------------- PARSER --------------------------------------------#
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ICA-AROMA (direct python or nipype workflow)."
    )

    # Required options
    req = parser.add_argument_group('Required arguments')
    req.add_argument('-o', '-out', dest="outDir", required=True, help='Output directory name')

    # Required options in non-Feat mode
    nonfeat = parser.add_argument_group('Required arguments - generic mode')
    nonfeat.add_argument('-i', '-in', dest="inFile", required=False,
                         help='Input file name of fMRI data (.nii.gz)')
    nonfeat.add_argument('-mc', dest="mc", required=False,
                         help='Motion parameters file (e.g., mcflirt .par)')
    nonfeat.add_argument('-a', '-affmat', dest="affmat", default="", help='Affine registration mat file')
    nonfeat.add_argument('-w', '-warp', dest="warp", default="", help='Non-linear warp file')
    nonfeat.add_argument('-m', '-mask', dest="mask", default="", help='Mask for MELODIC')

    # Required options in Feat mode
    feat = parser.add_argument_group('Required arguments - FEAT mode')
    feat.add_argument('-f', '-feat', dest="inFeat", required=False, help='Feat directory name')

    # Optional options
    opt = parser.add_argument_group('Optional arguments')
    opt.add_argument('-tr', dest="TR", help='TR in seconds', type=float)
    opt.add_argument(
        '-den', dest="denType", default="nonaggr",
        help="Denoising: 'no' | 'nonaggr' (default) | 'aggr' | 'both'"
    )
    opt.add_argument('-md', '-meldir', dest="melDir", default="", help='Existing MELODIC directory')
    opt.add_argument('-dim', dest="dim", default=0, type=int,
                     help='MELODIC dimensionality (default 0=auto)')
    opt.add_argument('-ow', '-overwrite', dest="overwrite", action='store_true', default=False,
                     help='Overwrite output dir')
    opt.add_argument('-np', '-noplots', dest="generate_plots", action='store_false',
                     default=True, help='Disable plots')

    # Engine selection
    eng = parser.add_argument_group('Execution engine')
    eng.add_argument(
        '--engine',
        choices=['direct', 'nipype'],
        default='direct',
        help="Execution engine: 'direct' (default) or 'nipype'."
    )

    # Nipype-only options (ignored in direct mode)
    eng.add_argument('--nprocs', type=int, help="(nipype) Number of processes")
    eng.add_argument('--mp-context', dest='mp_context', help="(nipype) mp_context: fork/spawn/...")

    return parser


def main(argv=None):
    """
    Direct example:
    python -m ica_aroma_py.services.cli -o out -i func.nii.gz -mc mc.par

    Nypipe example:
    python -m ica_aroma_py.services.cli --engine nipype -o out -i func.nii.gz -mc mc.par --nprocs 12
    """
    
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    
    # Normalize denType
    if args.denType not in accepted_den_types:
        parser.error(
            f"Invalid -den/denType: {args.denType}. Allowed: {sorted(accepted_den_types)}"
        )

    # ---------- OPTIONAL DEP CHECKS (CLI UX) ----------
    # (A) If engine nipype -> ensure nipype is installed BEFORE importing workflow.py
    if args.engine == "nipype":
        try:
            import nipype  # noqa: F401
        except ModuleNotFoundError:
            parser.error(
                "You choose --engine nipype but 'nipype' is not installed.\n"
                "Install the extra with:\n"
                "  pip install ica-aroma-py[nipype]\n"
                "or:\n"
                "  pip install 'ica-aroma-py[nipype,plots]'"
            )

    # (B) If plots are enabled -> ensure plot deps are installed (optional but recommended)
    # Note: -np / --noplots sets args.generate_plots=False.
    if args.generate_plots:
        try:
            from .classification_plots import classification_plot  # noqa: F401
        except ModuleNotFoundError:
            parser.error(
                "Plot enabled but plot dependencies are not installed.\n"
                "Install the extra with:\n"
                "  pip install ica-aroma-py[plots]\n"
                "Or disable plots with -np / --noplots."
            )

    # ---------- DISPATCH ----------
    if args.engine == "nipype":
        # Lazy import: nipype is optional without this args
        from .workflow import generate_aroma_workflow, run_aroma_workflow

        wf = generate_aroma_workflow(
            out_dir=args.outDir,
            in_file=args.inFile,
            mc=args.mc,
            aff_mat=args.affmat,
            warp=args.warp,
            mask_in=args.mask,
            in_feat=args.inFeat,
            TR=args.TR,
            den_type=args.denType,
            mel_dir_in=args.melDir,
            dim=args.dim,
            generate_plots=args.generate_plots,
            result_dir=args.outDir
        )

        plugin_args = {}
        if args.mp_context:
            plugin_args["mp_context"] = args.mp_context
        if args.nprocs:
            plugin_args["n_procs"] = args.nprocs

        run_aroma_workflow(
            wf,
            plugin_args=plugin_args,
        )
        return 0

    # direct engine
    run_aroma(
        outDir=args.outDir,
        inFile=args.inFile,
        mc=args.mc,
        affmat=args.affmat,
        warp=args.warp,
        mask_in=args.mask,
        inFeat=args.inFeat,
        TR=args.TR,
        denType=args.denType,
        melDir=args.melDir,
        dim=args.dim,
        overwrite=args.overwrite,
        generate_plots=args.generate_plots,
    )
    return 0
