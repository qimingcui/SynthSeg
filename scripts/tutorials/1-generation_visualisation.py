"""

Very simple script to generate an example of the synthetic data used to train SynthSeg.
This is for visualisation purposes, since it uses all the default parameters.



If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""
# %%
import sys, os
from pathlib import Path
sys.path.append(os.path.join( os.getcwd(), '../../'))

from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator

# %%
import tensorflow as tf
print(tf.__version__)  # Should print 2.2.0

from tensorflow import keras
print(keras.__version__)  # Should match tf.__version__
# %%


# generate an image from the label map.
brain_generator = BrainGenerator('/mnt/hdd0/MRI_data/ADNI/synthseg_brain_generator_test/label_map.nii')
im, lab = brain_generator.generate_brain()

# save output image and label map under SynthSeg/generated_examples
utils.save_volume(im, brain_generator.aff, brain_generator.header, '/mnt/hdd0/MRI_data/ADNI/synthseg_brain_generator_test//outputs_tutorial_1/image.nii.gz')
utils.save_volume(lab, brain_generator.aff, brain_generator.header, '/mnt/hdd0/MRI_data/ADNI/synthseg_brain_generator_test//outputs_tutorial_1/labels.nii.gz')
