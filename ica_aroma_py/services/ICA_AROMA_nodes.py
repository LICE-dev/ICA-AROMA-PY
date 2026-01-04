# -*- DISCLAIMER: this file contains code derived from Nipype (https://github.com/nipy/nipype/blob/master/LICENSE)  -*-
from nipype.interfaces.fsl import FLIRT, UnaryMaths, ImageStats, ApplyMask, ExtractROI, Merge as fslMerge
from nipype.interfaces.fsl.base import FSLCommand, FSLCommandInputSpec
from nipype.interfaces.base import (traits, TraitedSpec, File, isdefined, BaseInterfaceInputSpec, BaseInterface,
                                    InputMultiObject)
from nibabel import load
import shutil
import os
import numpy as np
from . import ICA_AROMA_functions as AromaFunc

# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.fsl.base.FSLCommandInputSpec)  -*-
class GetNiftiTRInputSpec(FSLCommandInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s pixdim4",
        position="1",
        desc="the input image",
    )
    force_value = traits.Float(mandatory=False, desc="value forced by user")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.TraitedSpec)  -*-
class GetNiftiTROutputSpec(TraitedSpec):
    TR = traits.Float(desc="Repetition Time")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.fsl.base.FSLCommand)  -*-
class GetNiftiTR(FSLCommand):
    """
    Reads the time of repetition from a NIFTI file.

    """

    _cmd = "fslval"
    input_spec = GetNiftiTRInputSpec
    output_spec = GetNiftiTROutputSpec

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = self._outputs()

        if isdefined(self.inputs.force_value) and self.inputs.force_value != -1:
            outputs.TR = self.inputs.force_value
            return outputs

        info = runtime.stdout
        try:
            outputs.TR = float(info)
        except:
            outputs.TR = 0.0

        return outputs


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.fsl.base.FSLCommandInputSpec)  -*-
class FslNVolsInputSpec(FSLCommandInputSpec):
    in_file = File(
        exists=True, mandatory=True, argstr="%s", position="1", desc="the input image"
    )
    force_value = traits.Int(mandatory=False, desc="value forced by user")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.TraitedSpec)  -*-
class FslNVolsOutputSpec(TraitedSpec):
    n_vols = traits.Int(desc="Number of EPI runs")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.FSLCommand)  -*-
class FslNVols(FSLCommand):
    """
    Reads the num. of volumes from a 4d NIFTI file.

    """

    _cmd = "fslnvols"
    input_spec = FslNVolsInputSpec
    output_spec = FslNVolsOutputSpec

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = self._outputs()

        if isdefined(self.inputs.force_value) and self.inputs.force_value != -1:
            outputs.n_vols = self.inputs.force_value
            return outputs

        info = runtime.stdout
        try:
            outputs.n_vols = int(info)
        except ValueError:
            outputs.n_vols = 0

        return outputs


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterfaceInputSpec)  -*-
class IsoResampleInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True, mandatory=True, desc="the input image"
    )
    reference = File(
        exists=True, mandatory=True, desc="the reference image"
    )
    dim = traits.Float(
        desc="the pixel dimension for resampling",
        mandatory=True,
    )
    out_file = File(desc="the output image")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.TraitedSpec)  -*-
class IsoResampleOutputSpec(TraitedSpec):
    out_file = File(desc="the resampled image")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterface)  -*-
class IsoResample(BaseInterface):
    """
    Resample a Nifti file to an isotropic voxel

    """

    input_spec = IsoResampleInputSpec
    output_spec = IsoResampleOutputSpec

    def _run_interface(self, runtime):
        self.inputs.out_file = self._gen_outfilename()

        img = load(self.inputs.in_file)
        vox1, vox2, vox3, _ = img.header.get_zooms()
        # Round to match FSL values
        vox1 = round(vox1, 2)
        vox2 = round(vox2, 2)
        vox3 = round(vox3, 2)

        if vox1 == self.inputs.dim and vox2 == vox1 and vox3 == vox1:
            shutil.copyfile(self.inputs.in_file, self.inputs.out_file)
        else:
            file_resample = FLIRT()
            file_resample.inputs.in_file = self.inputs.in_file
            file_resample.inputs.reference = self.inputs.reference
            file_resample.inputs.apply_isoxfm = self.inputs.dim
            file_resample.inputs.interp = "trilinear"
            file_resample.inputs.out_file = self.inputs.out_file
            file_resample.run()

        return runtime

    def _gen_outfilename(self):
        out_file = self.inputs.out_file
        if not isdefined(out_file) and isdefined(self.inputs.in_file):
            out_file = "reg_" + os.path.basename(self.inputs.in_file)
        return os.path.abspath(out_file)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = self._gen_outfilename()
        return outputs


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterfaceInputSpec)  -*-
class FeatureSpatialInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True, mandatory=True, desc="the input image"
    )
    mask_csf = File(
        exists=True, mandatory=True, desc="the csf mask image"
    )
    mask_out = File(
        exists=True, mandatory=True, desc="the outbrain mask image"
    )
    mask_edge = File(
        exists=True, mandatory=True, desc="the sdge mask image"
    )

# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.TraitedSpec)  -*-
class FeatureSpatialOutputSpec(TraitedSpec):
    edge_fract = traits.Array(mandatory=True,
                              desc="Array of the edge fraction feature scores for the components of the melIC file")
    csf_fract = traits.Array(mandatory=True,
                             desc="Array of the csf fraction feature scores for the components of the melIC file")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterface)  -*-
class FeatureSpatial(BaseInterface):
    """
    This node extracts the spatial feature scores.

    """

    input_spec = FeatureSpatialInputSpec
    output_spec = FeatureSpatialOutputSpec

    def _run_interface(self, runtime):
        abs_value = UnaryMaths()
        abs_value.inputs.operation = "abs"
        abs_value.inputs.in_file = self.inputs.in_file
        abs_value_res = abs_value.run()

        tot_stat = ImageStats()
        tot_stat.inputs.op_string = "-M -V"
        tot_stat.inputs.split_4d = True
        tot_stat.inputs.in_file = abs_value_res.outputs.out_file
        tot_stat_res = tot_stat.run()

        apply_csf_mask = ApplyMask()
        apply_csf_mask.inputs.mask_file = self.inputs.mask_csf
        apply_csf_mask.inputs.in_file = abs_value_res.outputs.out_file
        apply_csf_mask_res = apply_csf_mask.run()

        csf_stat = ImageStats()
        csf_stat.inputs.op_string = "-M -V"
        csf_stat.inputs.split_4d = True
        csf_stat.inputs.in_file = apply_csf_mask_res.outputs.out_file
        csf_stat_res = csf_stat.run()

        apply_edge_mask = ApplyMask()
        apply_edge_mask.inputs.mask_file = self.inputs.mask_edge
        apply_edge_mask.inputs.in_file = abs_value_res.outputs.out_file
        apply_edge_mask_res = apply_edge_mask.run()

        edge_stat = ImageStats()
        edge_stat.inputs.op_string = "-M -V"
        edge_stat.inputs.split_4d = True
        edge_stat.inputs.in_file = apply_edge_mask_res.outputs.out_file
        edge_stat_res = edge_stat.run()

        apply_out_mask = ApplyMask()
        apply_out_mask.inputs.mask_file = self.inputs.mask_out
        apply_out_mask.inputs.in_file = abs_value_res.outputs.out_file
        apply_out_mask_res = apply_out_mask.run()

        out_stat = ImageStats()
        out_stat.inputs.op_string = "-M -V"
        out_stat.inputs.split_4d = True
        out_stat.inputs.in_file = apply_out_mask_res.outputs.out_file
        out_stat_res = out_stat.run()

        tot_stats = tot_stat_res.outputs.out_stat
        out_stats = out_stat_res.outputs.out_stat
        edge_stats = edge_stat_res.outputs.out_stat
        csf_stats = csf_stat_res.outputs.out_stat

        self.edge_fract = np.zeros(len(out_stats))
        self.csf_fract = np.zeros(len(out_stats))

        for i in range(len(out_stats)):
            tot_sum = tot_stats[i][0] * tot_stats[i][1]
            csf_sum = csf_stats[i][0] * csf_stats[i][1]
            edge_sum = edge_stats[i][0] * edge_stats[i][1]
            out_sum = out_stats[i][0] * out_stats[i][1]
            if not (tot_sum == 0):
                self.edge_fract[i] = (out_sum + edge_sum) / (tot_sum - csf_sum) if not ((tot_sum - csf_sum) == 0) else 0
                self.csf_fract[i] = csf_sum / tot_sum
            else:
                self.edge_fract[i] = 0
                self.csf_fract[i] = 0

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["edge_fract"] = self.edge_fract
        outputs["csf_fract"] = self.csf_fract
        return outputs


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterfaceInputSpec)  -*-
class FeatureSpatialPrepInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(File(exists=True), desc="List of zstat files")
    mask_file = File(
        exists=True, mandatory=True, desc="the mask image"
    )
    out_file = File(desc="the output image")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.TraitedSpec)  -*-
class FeatureSpatialPrepOutputSpec(TraitedSpec):
    out_file = File(desc="the preprocessed image")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterface)  -*-
class FeatureSpatialPrep(BaseInterface):
    """
    This node extracts the spatial feature scores.

    """

    input_spec = FeatureSpatialPrepInputSpec
    output_spec = FeatureSpatialPrepOutputSpec

    def _run_interface(self, runtime):
        self.inputs.out_file = self._gen_outfilename()

        # Merge mixture modeled thresholded spatial maps.
        # Note! In case that mixture modeling did not converge, the file will contain two spatial maps.
        # The latter being the results from a simple null hypothesis test.
        # In that case, this map will have to be used (the first one will be empty).

        unique_zstat = []

        for in_file in self.inputs.in_files:
            get_zstat_n = FslNVols()
            get_zstat_n.inputs.in_file = in_file
            get_zstat_n_res = get_zstat_n.run()
            reduced = get_zstat_n_res.outputs.n_vols - 1

            last_zstat = ExtractROI()
            last_zstat.inputs.t_min = reduced
            last_zstat.inputs.t_size = 1
            last_zstat.inputs.in_file = in_file
            last_zstat_res = last_zstat.run()

            unique_zstat.append(last_zstat_res.outputs.roi_file)

        merge_zstat = fslMerge()
        merge_zstat.inputs.dimension = 't'
        merge_zstat.inputs.in_files = unique_zstat
        merge_zstat_res = merge_zstat.run()

        mask_zstat = ApplyMask()
        mask_zstat.inputs.in_file = merge_zstat_res.outputs.merged_file
        mask_zstat.inputs.mask_file = self.inputs.mask_file
        mask_zstat.inputs.out_file = self.inputs.out_file
        mask_zstat.run()

        return runtime

    def _gen_outfilename(self):
        out_file = self.inputs.out_file
        if not isdefined(out_file):
            out_file = "masked_zstat.nii.gz"
        return os.path.abspath(out_file)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = self._gen_outfilename()
        return outputs


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterfaceInputSpec)  -*-
class FeatureTimeSeriesInputSpec(BaseInterfaceInputSpec):
    mel_mix = File(
        exists=True, mandatory=True, desc="melodic_mix text file"
    )
    mc = File(
        exists=True, mandatory=True, desc="file containing the realignment parameters"
    )

# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.TraitedSpec)  -*-
class FeatureTimeSeriesOutputSpec(TraitedSpec):
    max_rp_corr = traits.Array(desc="Array of the maximum RP correlation feature scores for the components of the "
                                    "melodic_mix file")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterface)  -*-
class FeatureTimeSeries(BaseInterface):
    """
    This function extracts the maximum RP correlation feature scores.
    It determines the maximum robust correlation of each component time-series
    with a model of 72 realignment parameters.

    """

    input_spec = FeatureTimeSeriesInputSpec
    output_spec = FeatureTimeSeriesOutputSpec

    def _run_interface(self, runtime):
        self.max_rp_corr = AromaFunc.feature_time_series(self.inputs.mel_mix, self.inputs.mc)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["max_rp_corr"] = self.max_rp_corr
        return outputs


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterfaceInputSpec)  -*-
class FeatureFrequencyInputSpec(BaseInterfaceInputSpec):
    mel_ft_mix = File(
        exists=True, mandatory=True, desc="melodic_mix text file"
    )
    TR = traits.Float(desc="Repetition Time")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.TraitedSpec)  -*-
class FeatureFrequencyOutputSpec(TraitedSpec):
    HFC = traits.Array(desc="Array of the HFC ('High-frequency content') feature scores")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterface)  -*-
class FeatureFrequency(BaseInterface):
    """
    This function extracts the high-frequency content feature scores.
    It determines the frequency, as a fraction of the Nyquist frequency,
    at which the higher and lower frequencies explain half
    of the total power between 0.01Hz and Nyquist.

    """

    input_spec = FeatureFrequencyInputSpec
    output_spec = FeatureFrequencyOutputSpec

    def _run_interface(self, runtime):
        self.HFC = AromaFunc.feature_frequency(self.inputs.mel_ft_mix, self.inputs.TR)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["HFC"] = self.HFC
        return outputs


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterfaceInputSpec)  -*-
class AromaClassificationInputSpec(BaseInterfaceInputSpec):
    max_rp_corr = traits.Array(mandatory=True,
                               desc="Array of the maximum RP correlation feature scores for the components of the "
                                    "melodic_mix file")
    HFC = traits.Array(mandatory=True, desc="Array of the HFC ('High-frequency content') feature scores")
    edge_fract = traits.Array(mandatory=True,
                              desc="Array of the edge fraction feature scores for the components of the melIC file")
    csf_fract = traits.Array(mandatory=True,
                             desc="Array of the csf fraction feature scores for the components of the melIC file")
    feature_scores = File(desc="Feature score file")
    classified_motion_ics = File(desc="Motion IC file")
    classification_overview = File(desc="Overview file")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.TraitedSpec)  -*-
class AromaClassificationOutputSpec(TraitedSpec):
    motion_ics = traits.List(desc="Array of the HFC ('High-frequency content') feature scores")
    feature_scores = File(desc="Feature score file")
    classified_motion_ics = File(desc="Motion IC file")
    classification_overview = File(desc="Overview file")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterface)  -*-
class AromaClassification(BaseInterface):
    """
    This function extracts the high-frequency content feature scores.
    It determines the frequency, as a fraction of the Nyquist frequency,
    at which the higher and lower frequencies explain half
    of the total power between 0.01Hz and Nyquist.

    """

    input_spec = AromaClassificationInputSpec
    output_spec = AromaClassificationOutputSpec

    def _run_interface(self, runtime):
        self.inputs.feature_scores = os.path.abspath("feature_scores.txt")
        self.inputs.classified_motion_ics = os.path.abspath("classified_motion_ICs.txt")
        self.inputs.classification_overview = os.path.abspath("classification_overview.txt")
        motion_ics = AromaFunc.classification(os.path.abspath("."),
                                                   self.inputs.max_rp_corr,
                                                   self.inputs.edge_fract,
                                                   self.inputs.HFC,
                                                   self.inputs.csf_fract)
        self.motion_ics = (motion_ics + 1).tolist()
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["motion_ics"] = self.motion_ics
        outputs["feature_scores"] = os.path.abspath("feature_scores.txt")
        outputs["classified_motion_ics"] = os.path.abspath("classified_motion_ICs.txt")
        outputs["classification_overview"] = os.path.abspath("classification_overview.txt")
        return outputs


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterfaceInputSpec)  -*-
class AromaClassificationPlotInputSpec(BaseInterfaceInputSpec):
    classification_overview_file = File(exists=True, mandatory=True, desc="Classification overview file")
    out_file = File(desc="The component assessment file")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.TraitedSpec)  -*-
class AromaClassificationPlotOutputSpec(TraitedSpec):
    out_file = File(desc="The component assessment file")


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterface)  -*-
class AromaClassificationPlot(BaseInterface):
    """
    This function extracts the high-frequency content feature scores.
    It determines the frequency, as a fraction of the Nyquist frequency,
    at which the higher and lower frequencies explain half
    of the total power between 0.01Hz and Nyquist.

    """

    input_spec = AromaClassificationPlotInputSpec
    output_spec = AromaClassificationPlotOutputSpec

    def _run_interface(self, runtime):
        from .classification_plots import classification_plot
        self.inputs.out_file = os.path.abspath("ICA_AROMA_component_assessment.pdf")
        classification_plot(self.inputs.classification_overview_file, os.path.abspath("."))
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath("ICA_AROMA_component_assessment.pdf")
        return outputs
