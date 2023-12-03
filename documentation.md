# MMOR Documentation


## Class Detector

### Attributes:
´´´model_type´´´: Type of model used for detection (Object Detection, Instance Segmentation, Keypoint Detection, LVIS Segmentation, Panoptic Segmentation)
´´´cfg´´´: Configuration settings for the model
´´´predictor´´´: Object for making predictions using the model
### Methods:
´´´__init__(self, model_type="OD")´´´: Initializes the Detector object with a specified model type. Loads the model configuration and weights based on the model type.
´´´get_attributes(self)´´´: Returns the model type.
´´´onImage_d(self, imagePath)´´´: Performs detection on a single image. Depending on the model type, it uses the predictor to generate predictions and draws instance predictions or panoptic segmentation results on the image. It also performs hierarchical indexation for class labels.
´´´onVideo_d(self, frame)´´´: Performs detection on a video frame. Similar to ´´´onImage_d´´´, it generates predictions for the frame and applies hierarchical indexation for class labels.
Note:
The class loads different model configurations and weights based on the model_type provided during initialization.
For panoptic segmentation, it includes logic to process predictions, perform hierarchical indexation, and translate class labels to Spanish.
The code implements object detection and segmentation functionalities using the Detectron2 library, allowing detection on both images and video frames.

## Class Midas

### Initialization

```python
class Midas:
    def __init__(self, model_att, trans_processing, device, thresh_m):
        """
        Initializes the Midas class.

        Args:
            model_att (torch.nn.Module): Neural network model for depth prediction.
            trans_processing (torchvision.transforms.Compose): Image transformation pipeline.
            device (str): Device to run inference on ('cpu' or 'cuda').
            thresh_m (float): Threshold value for determining depth changes.
        """
```
### Methods
```python
def onImage_m(self, imagePath):
        """
        Process an image and return the depth-aware representation.

        Args:
            imagePath (str): Path to the input image.

        Returns:
            tuple(numpy.ndarray, numpy.ndarray): Tuple containing:
                - Depth-aware representation of the image.
                - Initial representation of the image before thresholding.
        """
def onVideo_m(self, frame):
    """
    Process a video frame and return the depth-aware representation.

    Args:
        frame (numpy.ndarray): Video frame in BGR format.

    Returns:
        numpy.ndarray: Depth-aware representation of the video frame.
    """
```

## MobileCam Class

### Initialization

```python
class MobileCam(Midas, Detector):
    def __init__(self, midas_model_att, trans_processing, device, model_type, thresh):
        """
        Initializes the MobileCam class.

        Args:
            midas_model_att (torch.nn.Module): Neural network model for depth prediction.
            trans_processing (torchvision.transforms.Compose): Image transformation pipeline.
            device (str): Device to run inference on ('cpu' or 'cuda').
            model_type (str): Type of the detector model.
            thresh (float): Threshold value for determining depth changes.
        """


