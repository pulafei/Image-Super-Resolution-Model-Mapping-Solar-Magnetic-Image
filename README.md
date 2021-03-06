# Image-Super-Resolution-Model-Mapping-Solar-Magnetic-Image
## Abstract
With the change in technology, the accuracy of detectors has gradually improved. Inevitably, the data detected by the old detector would not be as clear as the data detected by the new detector. The photospheric line-of-sight magnetic field maps integrated from Michelson Doppler Imager (MDI) and Helioseismic and Magnetic Imager (HMI) detection data have the same problem. This project builds an image super-resolution model to fit the old magnetic field maps related to MDI, which is the old detector, to the new magnetic field maps related to HMI, which is the new detector. This project will be a reference for doing the same type of data clarity enhancement in the future.
## Code Instruction
The project was developed with Google Colab due to the poor performance of the developer's local GPU. The model_test_0530.ipynb is the development code of this project. The capstone_main.ipynb, load_data.py and model.py are split packages based on development code. The development code can be used directly in Google Colab. The split packages can be used in local after setting local environment.
## Environment
tensorflow  
tensorflow_addons   
tensorflow_probability  
matplotlib  
astropy  
## Model Link
This is a model that has been trained. The prediction results of the model are consistent with those on the paper.   
[Model Link](https://drive.google.com/file/d/1yfJfEPf9LQ8wbJEXbo8xzHiNFbyqUkiM/view?usp=sharing)
## Dataset
[Dataset Link](https://drive.google.com/file/d/18cN3ilMD4jNkHnUaLEscFKm30TSk71MD/view?usp=sharing)This is the dataset.
