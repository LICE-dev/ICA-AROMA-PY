# -*- DISCLAIMER: this file contains code derived from Nipype (https://github.com/nipy/nipype/blob/master/LICENSE)  -*-
from nipype.interfaces.fsl import ApplyXFM, FLIRT
from nipype.interfaces.fsl.base import FSLCommand, FSLCommandInputSpec
from nipype.interfaces.base import traits, TraitedSpec, File, isdefined, BaseInterfaceInputSpec, BaseInterface
from nibabel import load
import numpy as np
import shutil
import os
import random
from .ICA_AROMA_functions import cross_correlation
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

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
    nvols = traits.Int(desc="Number of EPI runs")


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
            outputs.nvols = self.inputs.force_value
            return outputs

        info = runtime.stdout
        try:
            outputs.nvols = int(info)
        except ValueError:
            outputs.nvols = 0

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
            fileResample = FLIRT()
            fileResample.inputs.in_file = self.inputs.in_file
            fileResample.inputs.reference = self.inputs.reference
            fileResample.inputs.apply_isoxfm = self.inputs.dim
            fileResample.inputs.interp = "trilinear"
            fileResample.inputs.out_file = self.inputs.out_file
            fileResample.run()

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
class FeatureTimeSeriesInputSpec(BaseInterfaceInputSpec):
    melmix = File(
        exists=True, mandatory=True, desc="melodic_mix text file"
    )
    mc = File(
        exists=True, mandatory=True, desc="file containing the realignment parameters"
    )

# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.TraitedSpec)  -*-
class FeatureTimeSeriesOutputSpec(TraitedSpec):
    maxRPcorr = traits.Array(desc="Array of the maximum RP correlation feature scores for the components of the melodic_mix file")


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
        # Read melodic mix file (IC time-series), subsequently define a set of squared time-series
        mix = np.loadtxt(self.inputs.melmix)

        # Read motion parameter file
        rp6 = np.loadtxt(self.inputs.mc)
        _, nparams = rp6.shape

        # Determine the derivatives of the RPs (add zeros at time-point zero)
        rp6_der = np.vstack((np.zeros(nparams),
                             np.diff(rp6, axis=0)
                             ))

        # Create an RP-model including the RPs and its derivatives
        rp12 = np.hstack((rp6, rp6_der))

        # Add the squared RP-terms to the model
        # add the fw and bw shifted versions
        rp12_1fw = np.vstack((
            np.zeros(2 * nparams),
            rp12[:-1]
        ))
        rp12_1bw = np.vstack((
            rp12[1:],
            np.zeros(2 * nparams)
        ))
        rp_model = np.hstack((rp12, rp12_1fw, rp12_1bw))

        # Determine the maximum correlation between RPs and IC time-series
        nsplits = 1000
        nmixrows, nmixcols = mix.shape
        nrows_to_choose = int(round(0.9 * nmixrows))

        # Max correlations for multiple splits of the dataset (for a robust estimate)
        max_correls = np.empty((nsplits, nmixcols))
        for i in range(nsplits):
            # Select a random subset of 90% of the dataset rows (*without* replacement)
            chosen_rows = random.sample(population=range(nmixrows),
                                        k=nrows_to_choose)

            # Combined correlations between RP and IC time-series, squared and non squared
            correl_nonsquared = cross_correlation(mix[chosen_rows],
                                                  rp_model[chosen_rows])
            correl_squared = cross_correlation(mix[chosen_rows] ** 2,
                                               rp_model[chosen_rows] ** 2)
            correl_both = np.hstack((correl_squared, correl_nonsquared))

            # Maximum absolute temporal correlation for every IC
            max_correls[i] = np.abs(correl_both).max(axis=1)

        # Feature score is the mean of the maximum correlation over all the random splits
        # Avoid propagating occasional nans that arise in artificial test cases
        self.maxRPcorr = np.nanmean(max_correls, axis=0)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["maxRPcorr"] = self.maxRPcorr
        return outputs


# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterfaceInputSpec)  -*-
class FeatureFrequencyInputSpec(BaseInterfaceInputSpec):
    melFTmix = File(
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
        # Determine sample frequency
        Fs = 1.0 / self.inputs.TR

        # Determine Nyquist-frequency
        Ny = Fs / 2.0

        # Load melodic_FTmix file
        FT = np.loadtxt(self.inputs.melFTmix)

        # Determine which frequencies are associated with every row in the melodic_FTmix file  (assuming the rows range from 0Hz to Nyquist)
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
        self.HFC = f_norm[idx_cutoff]

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["HFC"] = self.HFC
        return outputs

# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterfaceInputSpec)  -*-
class AromaClassificationInputSpec(BaseInterfaceInputSpec):
    maxRPcorr = traits.Array(mandatory=True,
                             desc="Array of the maximum RP correlation feature scores for the components of the melodic_mix file")
    HFC = traits.Array(mandatory=True, desc="Array of the HFC ('High-frequency content') feature scores")
    edgeFract = traits.Array(mandatory=True,
                             desc="Array of the edge fraction feature scores for the components of the melIC file")
    csfFract = traits.Array(mandatory=True,
                            desc="Array of the csf fraction feature scores for the components of the melIC file")
    feature_scores = File(desc="Feature score file")
    classified_motion_ICs = File(desc="Motion IC file")
    classification_overview = File(desc="Overview file")

# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.TraitedSpec)  -*-
class AromaClassificationOutputSpec(TraitedSpec):
    motionICs = traits.List(desc="Array of the HFC ('High-frequency content') feature scores")
    feature_scores = File(desc="Feature score file")
    classified_motion_ICs = File(desc="Motion IC file")
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
        self.inputs.classified_motion_ICs = os.path.abspath("classified_motion_ICs.txt")
        self.inputs.classification_overview = os.path.abspath("classification_overview.txt")

        # Define criteria needed for classification (thresholds and hyperplane-parameters)
        thr_csf = 0.10
        thr_HFC = 0.35
        hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])

        # Project edge & maxRPcorr feature scores to new 1D space
        x = np.array([self.inputs.maxRPcorr, self.inputs.edgeFract])
        proj = hyp[0] + np.dot(x.T, hyp[1:])

        # Classify the ICs
        motionICs = np.squeeze(np.array(np.where((proj > 0) + (self.inputs.csfFract > thr_csf) + (self.inputs.HFC > thr_HFC))))

        # Put the feature scores in a text file
        np.savetxt(self.inputs.feature_scores,
                   np.vstack((self.inputs.maxRPcorr, self.inputs.edgeFract, self.inputs.HFC, self.inputs.csfFract)).T)

        # Put the indices of motion-classified ICs in a text file
        txt = open(self.inputs.classified_motion_ICs, 'w')
        if motionICs.size > 1:  # and len(motionICs) != 0: if motionICs is not None and
            txt.write(','.join(['{:.0f}'.format(num) for num in (motionICs + 1)]))
        elif motionICs.size == 1:
            txt.write('{:.0f}'.format(motionICs + 1))
        txt.close()

        # Create a summary overview of the classification
        txt = open(self.inputs.classification_overview, 'w')
        txt.write('\t'.join(['IC',
                             'Motion/noise',
                             'maximum RP correlation',
                             'Edge-fraction',
                             'High-frequency content',
                             'CSF-fraction']))
        txt.write('\n')
        for i in range(0, len(self.inputs.csfFract)):
            if (proj[i] > 0) or (self.inputs.csfFract[i] > thr_csf) or (self.inputs.HFC[i] > thr_HFC):
                classif = "True"
            else:
                classif = "False"
            txt.write('\t'.join(['{:d}'.format(i + 1),
                                 classif,
                                 '{:.2f}'.format(self.inputs.maxRPcorr[i]),
                                 '{:.2f}'.format(self.inputs.edgeFract[i]),
                                 '{:.2f}'.format(self.inputs.HFC[i]),
                                 '{:.2f}'.format(self.inputs.csfFract[i])]))
            txt.write('\n')
        txt.close()

        self.motionICs = motionICs.tolist()

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["motionICs"] = self.motionICs
        outputs["feature_scores"] = os.path.abspath("feature_scores.txt")
        outputs["classified_motion_ICs"] = os.path.abspath("classified_motion_ICs.txt")
        outputs["classification_overview"] = os.path.abspath("classification_overview.txt")
        return outputs






# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.BaseInterfaceInputSpec)  -*-
class AromaClassificationPlotInputSpec(BaseInterfaceInputSpec):
    classification_overview_file = File(exists=True, mandatory=True, desc="Classification overview file")
    out_file = File(desc="The component assessement file")

# -*- DISCLAIMER: this class extends a Nipype class (nipype.interfaces.base.TraitedSpec)  -*-
class AromaClassificationPlotOutputSpec(TraitedSpec):
    out_file = File(desc="The component assessement file")


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

        self.inputs.out_file = os.path.abspath("ICA_AROMA_component_assessment.pdf")

        ###---Start---###
        # find files
        myfiles = [self.inputs.classification_overview_file]
        print('Found', len(myfiles), 'file(s)')

        mpl.use('Agg')

        # load in data from files
        count = 0
        for m in myfiles:

            res = []

            tmp = open(m, 'r').read().split('\n')

            for t in tmp[1:-1]:
                vals = t.split('\t')
                res.append([vals[1],
                            float(vals[2]),
                            float(vals[3]),
                            float(vals[4]),
                            float(vals[5])])

            if count == 0:
                df = pd.DataFrame.from_records(res)
            else:
                df2 = pd.DataFrame.from_records(res)
                df = pd.concat([df, df2], ignore_index=True)

            count += 1

        # get counts
        ncomp = len(df)
        nmot = len(df.loc[df[0] == "True"])
        print('Found', nmot, 'head motion-related components in a total of', ncomp, 'components.')

        # add dummy components if needed, this is just for making the plots look nice
        tmp = df.loc[df[0] == "True"]
        if len(tmp) < 3:
            df3 = pd.DataFrame.from_records([["True", 1., 1., 0., 0.],
                                             ["True", 1., 1., 0., 0.],
                                             ["True", 1., 1., 0., 0.]])
            df = pd.concat([df, df3], ignore_index=True)
        tmp = df.loc[df[0] == "False"]
        if len(tmp) < 3:
            df3 = pd.DataFrame.from_records([["False", 0., 0., 0., 0.],
                                             ["False", 0., 0., 0., 0.],
                                             ["False", 0., 0., 0., 0.]])
            df = pd.concat([df, df3], ignore_index=True)

        # rename columns
        df = df.rename(index=str, columns={0: 'Motion',
                                           1: 'RP',
                                           2: 'Edge',
                                           3: 'Freq',
                                           4: 'CSF'})

        # Make pretty figure
        # styling
        sns.set_style('white')
        colortrue = "#FFBF17"
        colorfalse = "#69A00A"

        # create figure
        fig = plt.figure(figsize=[12, 4])

        # define grids
        gs = gridspec.GridSpec(4, 7, wspace=1)
        gs00 = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[:, 0:3])
        gs01 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 3:5])
        gs02 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 5:7])

        # define subplots
        # Edge/RP
        ax1 = fig.add_subplot(gs00[1:4, 0:3])
        # distribution edge (ax1 top)
        ax1t = fig.add_subplot(gs00[0, 0:3])
        # distribution RP (ax1 right)
        ax1r = fig.add_subplot(gs00[1:4, 3])
        # Freq
        ax2 = fig.add_subplot(gs01[1:4, :])
        # CSF
        ax3 = fig.add_subplot(gs02[1:4, :])

        # plot Freq
        sns.boxplot(x="Motion",
                    y="Freq",
                    hue="Motion",
                    data=df,
                    ax=ax2,
                    palette={'True': colortrue, 'False': colorfalse},
                    order=['True', 'False'])
        ax2.hlines(0.35, -1, 2, zorder=0, linestyles='dotted', linewidth=0.5)
        ax2.set_ylim([0, 1])
        ax2.set_xlabel('Classification', fontsize=14, labelpad=10)
        ax2.set_ylabel('High-Frequency Content', fontsize=14)
        ax2.set_xticks([0, 1], labels=['Motion', 'Other'])
        ax2.tick_params(axis='both', labelsize=12)
        sns.despine(ax=ax2)

        # plot CSF
        sns.boxplot(x="Motion",
                    y="CSF",
                    hue="Motion",
                    data=df,
                    ax=ax3,
                    palette={'True': colortrue, 'False': colorfalse},
                    order=['True', 'False'])
        ax3.hlines(0.1, -1, 2, zorder=0, linestyles='dotted', linewidth=0.5)
        ax3.set_ylim([0, 1])
        ax3.set_xlabel('Classification', fontsize=14, labelpad=10)
        ax3.set_ylabel('CSF Fraction', fontsize=14)
        ax3.set_xticks([0, 1], labels=['Motion', 'Other'])
        ax3.tick_params(axis='both', labelsize=12)
        sns.despine(ax=ax3)

        # plot Edge/RP relationship
        # obtain projection line
        hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])
        a = -hyp[1] / hyp[2]
        xx = np.linspace(0, 1)
        yy = a * xx - hyp[0] / hyp[2]
        # plot scatter and line
        if len(df) > 100:
            sizemarker = 6
        else:
            sizemarker = 10
        ax1.scatter(x="RP",
                    y="Edge",
                    data=df.loc[df['Motion'] == "False"],
                    color=colorfalse,
                    s=sizemarker)
        # plot true ones on top to see how much the go over the border
        # this gives an indication for how many were selected using the
        # two other features
        ax1.scatter(x="RP",
                    y="Edge",
                    data=df.loc[df['Motion'] == "True"],
                    color=colortrue,
                    s=sizemarker)
        # add decision boundary
        ax1.plot(xx, yy, '.', color="k", markersize=1)
        # styling
        ax1.set_ylim([0, 1])
        ax1.set_xlim([0, 1])
        ax1.set_xlabel('Maximum RP Correlation', fontsize=14, labelpad=10)
        ax1.set_ylabel('Edge Fraction', fontsize=14)
        ax1.set_xticks(np.arange(0, 1.2, 0.2))
        ax1.set_yticks(np.arange(0, 1.2, 0.2))
        ax1.tick_params(axis='both', labelsize=12)

        # plot distributions
        # RP
        sns.histplot(df.loc[df['Motion'] == "True", "RP"],
                     ax=ax1t,
                     color=colortrue,
                     kde=True,
                     stat="density",
                     element="step",
                     alpha=0.2)
        sns.histplot(df.loc[df['Motion'] == "False", "RP"],
                     ax=ax1t,
                     color=colorfalse,
                     kde=True,
                     stat="density",
                     element="step",
                     alpha=0.2)
        ax1t.set_xlim([0, 1])

        # Edge
        sns.histplot(y=df.loc[df['Motion'] == "True", "Edge"],
                     ax=ax1r,
                     color=colortrue,
                     kde=True,
                     stat="density",
                     element="step",
                     alpha=0.2)
        sns.histplot(y=df.loc[df['Motion'] == "False", "Edge"],
                     ax=ax1r,
                     color=colorfalse,
                     kde=True,
                     stat="density",
                     element="step",
                     alpha=0.2)

        ax1r.set_ylim([0, 1])

        # cosmetics
        for myax in [ax1t, ax1r]:
            myax.set_xticks([])
            myax.set_yticks([])
            myax.set_xlabel('')
            myax.set_ylabel('')
            myax.spines['right'].set_visible(False)
            myax.spines['top'].set_visible(False)
            myax.spines['bottom'].set_visible(False)
            myax.spines['left'].set_visible(False)

        # bring tickmarks back
        for myax in fig.get_axes():
            myax.tick_params(which="major", direction='in', length=3)

        # add figure title
        plt.suptitle('Component Assessment', fontsize=20)

        # outtakes
        plt.savefig(self.inputs.out_file, bbox_inches='tight')

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath("ICA_AROMA_component_assessment.pdf")
        return outputs