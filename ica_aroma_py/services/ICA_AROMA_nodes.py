# -*- DISCLAIMER: this file contains code derived from Nipype (https://github.com/nipy/nipype/blob/master/LICENSE)  -*-
from nipype.interfaces.fsl import FLIRT
from nipype.interfaces.fsl.base import FSLCommand, FSLCommandInputSpec
from nipype.interfaces.base import traits, TraitedSpec, File, isdefined, BaseInterfaceInputSpec, BaseInterface
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
    tot_stat = traits.List(mandatory=True, desc="Mean and number of non-zero voxels within the total Z-map")
    csf_stat = traits.List(mandatory=True, desc="Mean and number of non-zero voxels within the csf Z-map")
    edge_stat = traits.List(mandatory=True, desc="Mean and number of non-zero voxels within the edge Z-map")
    out_stat = traits.List(mandatory=True, desc="Mean and number of non-zero voxels within the out Z-map")

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
        self.edge_fract = np.zeros(len(self.inputs.tot_stat))
        self.csf_fract = np.zeros(len(self.inputs.tot_stat))
        for i in range(len(self.inputs.tot_stat)):
            totSum = self.inputs.tot_stat[i][0] * self.inputs.tot_stat[i][1]
            csfSum = self.inputs.csf_stat[i][0] * self.inputs.csf_stat[i][1]
            edgeSum = self.inputs.edge_stat[i][0] * self.inputs.edge_stat[i][1]
            outSum = self.inputs.out_stat[i][0] * self.inputs.out_stat[i][1]
            if not (totSum == 0):
                self.edge_fract[i] = (outSum + edgeSum) / (totSum - csfSum) if not ((totSum - csfSum) == 0) else 0
                self.csf_fract[i] = csfSum / totSum
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
