# MMOR Documentation


## Class Detector
The class loads panoptic segmentation, it includes logic to process predictions, perform hierarchical indexation, and translate class labels to Spanish.
The code implements object detection and segmentation functionalities using the Detectron2 library, allowing detection on both images and video frames.

### Attributes:
```model_type``` Type of model used for detection (Object Detection, Instance Segmentation, Keypoint Detection, LVIS Segmentation, Panoptic Segmentation)

```cfg``` Configuration settings for the model

```predictor``` Object for making predictions using the model

### Methods:
```__init__(self, model_type="OD")``` Initializes the Detector object with a specified model type. Loads the model configuration and weights based on the model type.

```get_attributes(self)``` Returns the model type.

```onImage_d(self, imagePath)``` Performs detection on a single image. Depending on the model type, it uses the predictor to generate predictions and draws instance predictions or panoptic segmentation results on the image. It also performs hierarchical indexation for class labels.

```onVideo_d(self, frame)``` Performs detection on a video frame. Similar to ```onImage_d```, it generates predictions for the frame and applies hierarchical indexation for class labels.

## Class Midas
The Midas class facilitates depth-aware processing of images and video frames using a specified neural network model. It provides methods to handle both images (onImage_m) and video frames (onVideo_m), generating depth-aware representations and allowing for further analysis of the data.

### Attributes
```model_att (torch.nn.Module)``` Neural network model for depth prediction.

```trans_processing (torchvision.transforms.Compose)``` Image transformation pipeline.

```device``` (str) Device to run inference on ('cpu' or 'cuda').

```thresh_m``` (float) Threshold value for determining depth changes.

### Methods
```__init__(self, model_att, trans_processing, device, thresh_m)``` Initialize the class Midas and assign the getter for each parameter.

```onImage_m(self, imagePath)``` Process an image and return mask proximity.

```onVideo_m(self, frame)``` Process a video frame and return the mask proximity. It paramater is ```frame``` (numpy.ndarray): Video frame in BGR format. An it returns a ```numpy.ndarray``` type.

## MobileCam Class
Heredity class from parents Detector and Midas. The MobileCam class combines functionalities from the Midas and Detector classes, providing methods for processing images and real-time video frames. It generates depth-aware representations and identifies relevant objects along with their positions, offering a comprehensive analysis of the input data.

### Methods
```__init__(self, midas_model_att, trans_processing, device, model_type, thresh)```. Initialize class for fusion models and apply getter for parent classes.

```MultOut_img(self, path)``` Processes an image and returns information about relevant objects, their positions, and depth-aware representation. It parameter is ```path``` (str) Path to the input image. Which returns, Audio file (with detected object positions), textual information about the objects, Initial representation of the image before thresholding and Evaluation dictionary containing object information.

```MultOut_RealTime(self)``` Processes real-time video frames and returns information about relevant objects, their positions, and depth-aware representation. And returns ```Audio``` Audio file (with detected object positions).



