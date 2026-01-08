import os
aroma_mask_csf = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "resources")), 'mask_csf.nii.gz')
aroma_mask_edge = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "resources")), 'mask_edge.nii.gz')
aroma_mask_out = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "resources")), 'mask_out.nii.gz')

if not os.path.exists(aroma_mask_csf):
    raise Exception