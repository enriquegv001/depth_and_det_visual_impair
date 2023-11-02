
from VideoStreamColab import js_to_image, bbox_to_bytes, video_stream, video_frame


from setup import call_midas_model
#midas_model, transform, device = call_midas_model()

# Packages for Detectron2
# Class reference from https://github.com/evanshlom/detectron2-panoptic-segmentation-video/blob/main/Detector.py
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import inspect

import cv2
from google.colab.patches import cv2_imshow # just inside colab
import numpy as np

# packages for Midas
import time
import torch
import numpy as np
#import cv2
#import tensorflow as tf
#from tensorflow import keras
#import tensorflow_hub as hub
#from tensorflow.keras.models import load_model

# packages for Mobile Cam
import os 
os.system('pip install gtts pydub') 
from gtts import gTTS #Import Google Text to Speech
from pydub import AudioSegment
from IPython.display import Audio #Import Audio method from IPython's Display Class

class Detector:
    def __init__(self, model_type="OD"):
        self.cfg = get_cfg()
        self.model_type = model_type

        # Load model config and pretrained model
        if model_type=="OD": # object detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type=="IS": # instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type=="KP": # keypoint detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif model_type=="LVIS": # lvis segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif model_type=="PS": # panoptic segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")


        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = DefaultPredictor(self.cfg)

    def get_attributes(self):
      return self.model_type

    def onImage_d(self, imagePath):
        image = cv2.imread(imagePath)
        if self.model_type != "PS":
            predictions = self.predictor(image)
            viz = Visualizer(image[:, :, ::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            instance_mode=ColorMode.SEGMENTATION)
            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        else: # panoptic segmentation predictions
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            #x = [print(x, len(x)) for x in predictions.cpu().numpy()[:10, :10].T] #identify its classes
            #x = [print(x, len(x)) for x in predictions.cpu().numpy()[:, :10]]

            segmentInfo_ = segmentInfo.copy() # save segmentInfo original

        #============================== Algorithm: hierarchy based on things =================================
            # set the class label and class heirarchy inside segmentInfo
            stuff_hierarchy = {'things': 1, 'banner': 1, 'blanket': 2, 'bridge': 1, 'cardboard': 1, 'counter': 1, 'curtain': 2, 'door-stuff': 1, 'floor-wood': 2, 'flower': 3, 'fruit': 2, 'gravel': 3, 'house': 3, 'light': 3, 'mirror-stuff': 1, 'net': 2, 'pillow': 2, 'platform': 1, 'playingfield': 3, 'railroad': 1, 'river': 1, 'road': 1, 'roof': 3, 'sand': 3, 'sea': 3, 'shelf': 1, 'snow': 1, 'stairs': 1, 'tent': 1, 'towel': 2, 'wall-brick': 1, 'wall-stone': 1, 'wall-tile': 1, 'wall-wood': 1, 'water': 1, 'window-blind': 1, 'window': 1, 'tree': 3, 'fence': 1, 'ceiling': 3, 'sky': 3, 'cabinet': 1, 'table': 1, 'floor': 3, 'pavement': 3, 'mountain': 3, 'grass': 3, 'dirt': 3, 'paper': 2, 'food': 2, 'building': 1, 'rock': 1, 'wall': 1, 'rug': 2}
            stuff_cat_id = {i: c for i, c in enumerate(metadata.stuff_classes)} # dict with index and classes

            thing_hierarchy = {'person': 1, 'bicycle': 1, 'car': 1, 'motorcycle':1, 'airplane':3, 'bus':1, 'train':1, 'truck':1, 'boat':3, 'traffic light':2, 'fire hydrant':2, 'stop sign':2, 'parking meter':2, 'bench':2, 'bird':3, 'cat':1, 'dog':1, 'horse':1, 'sheep':1, 'cow':1, 'elephant':1, 'bear':1, 'zebra':1, 'giraffe':1, 'backpack':2, 'umbrella':2, 'handbag':2, 'tie':3, 'suitcase':2, 'frisbee':1, 'skis':2, 'snowboard':2, 'sports ball':2, 'kite':3, 'baseball bat':2, 'baseball glove':2, 'skateboard':1, 'surfboard':3, 'tennis racket':2, 'bottle':2, 'wine glass':2, 'cup':2, 'fork':2, 'knife':2, 'spoon':3, 'bowl':3, 'banana':3, 'apple':3, 'sandwich':3, 'orange':3, 'broccoli':3, 'carrot':3, 'hot dog':3, 'pizza':3, 'donut':3, 'cake':3, 'chair':2, 'couch':2, 'potted plant':3, 'bed':2, 'dining table':2, 'toilet':2, 'tv':3, 'laptop':3, 'mouse':3, 'remote':3, 'keyboard':3, 'cell phone':3, 'microwave':3, 'oven':2, 'toaster':2, 'sink':2, 'refrigerator':2, 'book':3, 'clock':3, 'vase':2, 'scissors':2, 'teddy bear':3, 'hair drier':3, 'toothbrush':3}
            thing_cat_id = {i: c for i,c in enumerate(metadata.thing_classes)}

            for segment in segmentInfo:
              if not segment['isthing']:
                  label = stuff_cat_id[segment['category_id']]
                  segment['class_label'] = label
                  segment['class_hierarchy'] = stuff_hierarchy[label]
              else:
                  label = thing_cat_id[segment['category_id']]
                  segment['class_label'] = label
                  segment['class_hierarchy'] = thing_hierarchy[label]

            # dict that gets the id predictios and the class_hierarchy
            pred_hierarchy_dict_stuff = {x['id']:x['class_hierarchy'] for x in segmentInfo if x['isthing'] == False}
            pred_hierarchy_dict_thing = {x['id']:x['class_hierarchy'] for x in segmentInfo if x['isthing'] == True}

            # get the array that works as a window filter for hierarchy of the classes
            pred_arr = predictions.cpu()
            pred_set_id_stuff = set(pred_hierarchy_dict_stuff.keys())
            pred_set_id_thing = set(pred_hierarchy_dict_thing.keys())

            # here restructure window, need to be working with a three if statements (changes in the stuff, in the thing, in 0)
            def hierarchy_window(x):
              if x in pred_set_id_stuff:
                return pred_hierarchy_dict_stuff[x]
              elif x in pred_set_id_thing:
                return pred_hierarchy_dict_thing[x]
              else:
                return 0
            
            pred_hierarchy = np.vectorize(hierarchy_window)(pred_arr.numpy())

            # restructure info for stuff
            Info_with_label = {'id': [], 'isthing': [], 'category_id': [], 'class_label': [], 'class_hierarchy': []}
            for k in Info_with_label.keys():
              for x in segmentInfo:
                #if x['isthing'] == False:
                Info_with_label[k].append(x[k])

            trans_dict_stuff = {'things': 'objetos', 'banner': 'letrero', 'blanket': 'sábana', 'bridge': 'puente', 'cardboard': 'cartón', 'counter': 'mostrador', 'curtain': 'cortina', 'door-stuff': 'puerta', 'floor-wood': 'piso de madera', 'flower': 'flor', 'fruit': 'fruta', 'gravel': 'grava', 'house': 'casa', 'light': 'luz', 'mirror-stuff': 'espejo', 'net': 'red', 'pillow': 'almohada', 'platform': 'plataforma', 'playingfield': 'cancha de juego', 'railroad': 'ferrocarril', 'river': 'río', 'road': 'camino', 'roof': 'tejado', 'sand': 'arena', 'sea': 'oceano', 'shelf': 'repisas', 'snow': 'nieve', 'stairs': 'escalera', 'tent': 'tienda de campaña', 'towel': 'toalla', 'wall-brick': 'pared de ladrillo', 'wall-stone': 'pared de piedra', 'wall-tile': 'pared de teja', 'wall-wood': 'pared de madera', 'water': 'agua', 'window-blind': 'persiana', 'window': 'ventana', 'tree': 'árbol', 'fence': 'valla', 'ceiling': 'techo', 'sky': 'cielo', 'cabinet': 'gabinete', 'table': 'mesa', 'floor': 'piso', 'pavement': 'pavimento', 'mountain': 'montaña', 'grass': 'pasto', 'dirt': 'tierra', 'paper': 'papel', 'food': 'comida', 'building': 'construcción', 'rock': 'roca', 'wall': 'pared', 'rug': 'alfombra'}
            trans_dict_things = {'person': 'persona', 'bicycle': 'bicicleta', 'car': 'coche', 'motorcycle': 'motocicleta', 'airplane': 'avión', 'bus': 'autobus', 'train': 'tren', 'truck': 'camioneta', 'boat': 'bote', 'traffic light': 'semáforo', 'fire hydrant': 'hidratante', 'stop sign': 'señalización de alto', 'parking meter': 'parquímetro', 'bench': 'banca', 'bird': 'pájaro', 'cat': 'gato', 'dog': 'perro', 'horse': 'caballo', 'sheep': 'oveja', 'cow': 'vaca', 'elephant': 'elefante', 'bear': 'oso', 'zebra': 'cebra', 'giraffe': 'jirafa', 'backpack': 'mochila', 'umbrella': 'sombrilla', 'handbag': 'maleta de mano', 'tie': 'corbata', 'suitcase': 'portafolio', 'frisbee': 'frisbee', 'skis': 'skis', 'snowboard': 'snowboard', 'sports ball': 'balón', 'kite': 'cometa', 'baseball bat': 'bate de baseball', 'baseball glove': 'guante de baseball', 'skateboard': 'patineta', 'surfboard': 'tabla de surf', 'tennis racket': 'raqueta', 'bottle': 'botella', 'wine glass': 'copa de vino', 'cup': 'tasa', 'fork': 'tenedor', 'knife': 'cuchillo', 'spoon': 'cuchara', 'bowl': 'tasón', 'banana': 'plátano', 'apple': 'manzana', 'sandwich': 'sandwich', 'orange': 'naraja', 'broccoli': 'broccoli', 'carrot': 'zanahoria', 'hot dog': 'hot dog', 'pizza': 'pizza', 'donut': 'dona', 'cake': 'pastel', 'chair': 'silla', 'couch': 'sillón', 'potted plant': 'maceta', 'bed': 'cama', 'dining table': 'mesa de comedor', 'toilet': 'escusado', 'tv': 'televisión', 'laptop': 'laptop', 'mouse': 'mouse', 'remote': 'control remoto', 'keyboard': 'teclado', 'cell phone': 'celular', 'microwave': 'microondas', 'oven': 'horno', 'toaster': 'tostador', 'sink': 'lavabo', 'refrigerator': 'refrigerador', 'book': 'libro', 'clock': 'reloj', 'vase': 'jarrón', 'scissors': 'tijeras', 'teddy bear': 'oso de peluche', 'hair drier': 'secadora de pelo', 'toothbrush': 'pasta de dientes'}
            trans_set_stuff = set(trans_dict_stuff.keys())
            trans_set_thing = set(trans_dict_things.keys())
            def translate_label(x):
              if x in trans_set_stuff:
                return trans_dict_stuff[x]
              elif x in trans_set_thing:
                return trans_dict_things[x]
              else:
                return 0

            Info_with_label['class_label'] = np.vectorize(translate_label)(Info_with_label['class_label'])

        #==================================== display for image  ======================================

        # display normal results
        viz = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
        #cv2_imshow(output.get_image()[:,:,::-1])
        #plt.imshow(output)
        #display out colab
        # cv2.imshow("Result", output.get_image()[:,:,::-1])
        # cv2.waitKey(0)

        # display with filter per hierarchy
        #output = viz.draw_panoptic_seg_predictions(pred_arr.to("cpu"), segmentInfo_)
        cv2_imshow(output.get_image()[:,:,::-1])

        return pred_arr, pred_hierarchy, Info_with_label

    def onVideo_d(self, frame):
      predictions, segmentInfo = self.predictor(frame)["panoptic_seg"]
      metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
      #x = [print(x, len(x)) for x in predictions.cpu().numpy()[:10, :10].T] #identify its classes
      #x = [print(x, len(x)) for x in predictions.cpu().numpy()[:, :10]]

      segmentInfo_og = segmentInfo.copy() # save segmentInfo original

      #============================== Algorithm: hierarchy based on things =================================
      """
      # set the class label and class heirarchy inside segmentInfo
      stuff_hierarchy = {'things': 1, 'banner': 1, 'blanket': 2, 'bridge': 1, 'cardboard': 1, 'counter': 1, 'curtain': 2, 'door-stuff': 1, 'floor-wood': 2, 'flower': 3, 'fruit': 2, 'gravel': 3, 'house': 3, 'light': 3, 'mirror-stuff': 1, 'net': 2, 'pillow': 2, 'platform': 1, 'playingfield': 3, 'railroad': 1, 'river': 1, 'road': 1, 'roof': 3, 'sand': 3, 'sea': 3, 'shelf': 1, 'snow': 1, 'stairs': 1, 'tent': 1, 'towel': 2, 'wall-brick': 1, 'wall-stone': 1, 'wall-tile': 1, 'wall-wood': 1, 'water': 1, 'window-blind': 1, 'window': 1, 'tree': 3, 'fence': 1, 'ceiling': 3, 'sky': 3, 'cabinet': 1, 'table': 1, 'floor': 3, 'pavement': 3, 'mountain': 3, 'grass': 3, 'dirt': 3, 'paper': 2, 'food': 2, 'building': 1, 'rock': 1, 'wall': 1, 'rug': 2}
      stuff_hierarchy = {'things': 1, 'banner': 1, 'blanket': 2, 'bridge': 1, 'cardboard': 1, 'counter': 1, 'curtain': 2, 'door-stuff': 1, 'floor-wood': 2, 'flower': 3, 'fruit': 2, 'gravel': 3, 'house': 3, 'light': 3, 'mirror-stuff': 1, 'net': 2, 'pillow': 2, 'platform': 1, 'playingfield': 3, 'railroad': 1, 'river': 1, 'road': 1, 'roof': 3, 'sand': 3, 'sea': 3, 'shelf': 1, 'snow': 1, 'stairs': 1, 'tent': 1, 'towel': 2, 'wall-brick': 1, 'wall-stone': 1, 'wall-tile': 1, 'wall-wood': 1, 'water': 1, 'window-blind': 1, 'window': 1, 'tree': 3, 'fence': 1, 'ceiling': 3, 'sky': 3, 'cabinet': 1, 'table': 1, 'floor': 3, 'pavement': 3, 'mountain': 3, 'grass': 3, 'dirt': 3, 'paper': 2, 'food': 2, 'building': 1, 'rock': 1, 'wall': 1, 'rug': 2}

      stuff_cat_id = {i: c for i, c in enumerate(metadata.stuff_classes)} # dict with index and classes
      #things_cat_id = {i: c for i, c in enumerate(metadata.thing_classes)}

      for segment in segmentInfo:
        if not segment['isthing']:
            label = stuff_cat_id[segment['category_id']]
            segment['class_label'] = label
            segment['class_hierarchy'] = stuff_hierarchy[label]

      # dict that gets the id predictios and the class_hierarchy
      pred_hierarchy_dict = {x['id']:x['class_hierarchy'] for x in segmentInfo if x['isthing'] == False}

      # get the array that works as a window filter for hierarchy of the classes
      pred_arr = predictions.cpu()
      pred_set_id = set(pred_hierarchy_dict.keys())
      pred_hierarchy = np.vectorize(lambda x: pred_hierarchy_dict[x] if x in pred_set_id else 0)(pred_arr.numpy())

      # restructure info for stuff
      Info_with_label = {'id': [], 'isthing': [], 'category_id': [], 'area': [], 'class_label': [], 'class_hierarchy': []}
      for k in Info_with_label.keys():
        for x in segmentInfo:
          if x['isthing'] == False:
            Info_with_label[k].append(x[k])
      """

      # set the class label and class heirarchy inside segmentInfo
      stuff_hierarchy = {'things': 1, 'banner': 1, 'blanket': 2, 'bridge': 1, 'cardboard': 1, 'counter': 1, 'curtain': 2, 'door-stuff': 1, 'floor-wood': 2, 'flower': 3, 'fruit': 2, 'gravel': 3, 'house': 3, 'light': 3, 'mirror-stuff': 1, 'net': 2, 'pillow': 2, 'platform': 1, 'playingfield': 3, 'railroad': 1, 'river': 1, 'road': 1, 'roof': 3, 'sand': 3, 'sea': 3, 'shelf': 1, 'snow': 1, 'stairs': 1, 'tent': 1, 'towel': 2, 'wall-brick': 1, 'wall-stone': 1, 'wall-tile': 1, 'wall-wood': 1, 'water': 1, 'window-blind': 1, 'window': 1, 'tree': 3, 'fence': 1, 'ceiling': 3, 'sky': 3, 'cabinet': 1, 'table': 1, 'floor': 3, 'pavement': 3, 'mountain': 3, 'grass': 3, 'dirt': 3, 'paper': 2, 'food': 2, 'building': 1, 'rock': 1, 'wall': 1, 'rug': 2}
      stuff_cat_id = {i: c for i, c in enumerate(metadata.stuff_classes)} # dict with index and classes

      thing_hierarchy = {'person': 1, 'bicycle': 1, 'car': 1, 'motorcycle':1, 'airplane':3, 'bus':1, 'train':1, 'truck':1, 'boat':3, 'traffic light':2, 'fire hydrant':2, 'stop sign':2, 'parking meter':2, 'bench':2, 'bird':3, 'cat':1, 'dog':1, 'horse':1, 'sheep':1, 'cow':1, 'elephant':1, 'bear':1, 'zebra':1, 'giraffe':1, 'backpack':2, 'umbrella':2, 'handbag':2, 'tie':3, 'suitcase':2, 'frisbee':1, 'skis':2, 'snowboard':2, 'sports ball':2, 'kite':3, 'baseball bat':2, 'baseball glove':2, 'skateboard':1, 'surfboard':3, 'tennis racket':2, 'bottle':2, 'wine glass':2, 'cup':2, 'fork':2, 'knife':2, 'spoon':3, 'bowl':3, 'banana':3, 'apple':3, 'sandwich':3, 'orange':3, 'broccoli':3, 'carrot':3, 'hot dog':3, 'pizza':3, 'donut':3, 'cake':3, 'chair':2, 'couch':2, 'potted plant':3, 'bed':2, 'dining table':2, 'toilet':2, 'tv':3, 'laptop':3, 'mouse':3, 'remote':3, 'keyboard':3, 'cell phone':3, 'microwave':3, 'oven':2, 'toaster':2, 'sink':2, 'refrigerator':2, 'book':3, 'clock':3, 'vase':2, 'scissors':2, 'teddy bear':3, 'hair drier':3, 'toothbrush':3}
      thing_cat_id = {i: c for i,c in enumerate(metadata.thing_classes)}

      for segment in segmentInfo:
        if not segment['isthing']:
            label = stuff_cat_id[segment['category_id']]
            segment['class_label'] = label
            segment['class_hierarchy'] = stuff_hierarchy[label]
        else:
            label = thing_cat_id[segment['category_id']]
            segment['class_label'] = label
            segment['class_hierarchy'] = thing_hierarchy[label]

      # dict that gets the id predictios and the class_hierarchy
      pred_hierarchy_dict_stuff = {x['id']:x['class_hierarchy'] for x in segmentInfo if x['isthing'] == False}
      pred_hierarchy_dict_thing = {x['id']:x['class_hierarchy'] for x in segmentInfo if x['isthing'] == True}

      # get the array that works as a window filter for hierarchy of the classes
      pred_arr = predictions.cpu()
      pred_set_id_stuff = set(pred_hierarchy_dict_stuff.keys())
      pred_set_id_thing = set(pred_hierarchy_dict_thing.keys())

      # here restructure window, need to be working with a three if statements (changes in the stuff, in the thing, in 0)
      def hierarchy_window(x):
        if x in pred_set_id_stuff:
          return pred_hierarchy_dict_stuff[x]
        elif x in pred_set_id_thing:
          return pred_hierarchy_dict_thing[x]
        else:
          return 0

      pred_hierarchy = np.vectorize(hierarchy_window)(pred_arr.numpy())

      # restructure info for stuff
      Info_with_label = {'id': [], 'isthing': [], 'category_id': [], 'class_label': [], 'class_hierarchy': []}
      for k in Info_with_label.keys():
        for x in segmentInfo:
          #if x['isthing'] == False:
          Info_with_label[k].append(x[k])

      # translate prediction to spanish
      trans_dict_stuff = {'things': 'objetos', 'banner': 'letrero', 'blanket': 'sábana', 'bridge': 'puente', 'cardboard': 'cartón', 'counter': 'mostrador', 'curtain': 'cortina', 'door-stuff': 'puerta', 'floor-wood': 'piso de madera', 'flower': 'flor', 'fruit': 'fruta', 'gravel': 'grava', 'house': 'casa', 'light': 'luz', 'mirror-stuff': 'espejo', 'net': 'red', 'pillow': 'almohada', 'platform': 'plataforma', 'playingfield': 'cancha de juego', 'railroad': 'ferrocarril', 'river': 'río', 'road': 'camino', 'roof': 'tejado', 'sand': 'arena', 'sea': 'oceano', 'shelf': 'repisas', 'snow': 'nieve', 'stairs': 'escalera', 'tent': 'tienda de campaña', 'towel': 'toalla', 'wall-brick': 'pared de ladrillo', 'wall-stone': 'pared de piedra', 'wall-tile': 'pared de teja', 'wall-wood': 'pared de madera', 'water': 'agua', 'window-blind': 'persiana', 'window': 'ventana', 'tree': 'árbol', 'fence': 'valla', 'ceiling': 'techo', 'sky': 'cielo', 'cabinet': 'gabinete', 'table': 'mesa', 'floor': 'piso', 'pavement': 'pavimento', 'mountain': 'montaña', 'grass': 'pasto', 'dirt': 'tierra', 'paper': 'papel', 'food': 'comida', 'building': 'construcción', 'rock': 'roca', 'wall': 'pared', 'rug': 'alfombra'}
      trans_dict_things = {'person': 'persona', 'bicycle': 'bicicleta', 'car': 'coche', 'motorcycle': 'motocicleta', 'airplane': 'avión', 'bus': 'autobus', 'train': 'tren', 'truck': 'camioneta', 'boat': 'bote', 'traffic light': 'semáforo', 'fire hydrant': 'hidratante', 'stop sign': 'señalización de alto', 'parking meter': 'parquímetro', 'bench': 'banca', 'bird': 'pájaro', 'cat': 'gato', 'dog': 'perro', 'horse': 'caballo', 'sheep': 'oveja', 'cow': 'vaca', 'elephant': 'elefante', 'bear': 'oso', 'zebra': 'cebra', 'giraffe': 'jirafa', 'backpack': 'mochila', 'umbrella': 'sombrilla', 'handbag': 'maleta de mano', 'tie': 'corbata', 'suitcase': 'portafolio', 'frisbee': 'frisbee', 'skis': 'skis', 'snowboard': 'snowboard', 'sports ball': 'balón', 'kite': 'cometa', 'baseball bat': 'bate de baseball', 'baseball glove': 'guante de baseball', 'skateboard': 'patineta', 'surfboard': 'tabla de surf', 'tennis racket': 'raqueta', 'bottle': 'botella', 'wine glass': 'copa de vino', 'cup': 'tasa', 'fork': 'tenedor', 'knife': 'cuchillo', 'spoon': 'cuchara', 'bowl': 'tasón', 'banana': 'plátano', 'apple': 'manzana', 'sandwich': 'sandwich', 'orange': 'naraja', 'broccoli': 'broccoli', 'carrot': 'zanahoria', 'hot dog': 'hot dog', 'pizza': 'pizza', 'donut': 'dona', 'cake': 'pastel', 'chair': 'silla', 'couch': 'sillón', 'potted plant': 'maceta', 'bed': 'cama', 'dining table': 'mesa de comedor', 'toilet': 'escusado', 'tv': 'televisión', 'laptop': 'laptop', 'mouse': 'mouse', 'remote': 'control remoto', 'keyboard': 'teclado', 'cell phone': 'celular', 'microwave': 'microondas', 'oven': 'horno', 'toaster': 'tostador', 'sink': 'lavabo', 'refrigerator': 'refrigerador', 'book': 'libro', 'clock': 'reloj', 'vase': 'jarrón', 'scissors': 'tijeras', 'teddy bear': 'oso de peluche', 'hair drier': 'secadora de pelo', 'toothbrush': 'pasta de dientes'}
      trans_set_stuff = set(trans_dict_stuff.keys())
      trans_set_thing = set(trans_dict_things.keys())
      def translate_label(x):
        if x in trans_set_stuff:
          return trans_dict_stuff[x]
        elif x in trans_set_thing:
          return trans_dict_things[x]
        else:
          return 0

      Info_with_label['class_label'] = np.vectorize(translate_label)(Info_with_label['class_label'])  

      #==================================== display for image  ======================================

      # display normal results
      viz = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
      output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo_og)
      #cv2_imshow(output.get_image()[:,:,::-1])
      #plt.imshow(output)
      #display out colab
      # cv2.imshow("Result", output.get_image()[:,:,::-1])
      # cv2.waitKey(0)

      # display with filter per hierarchy
      #output = viz.draw_panoptic_seg_predictions(pred_arr.to("cpu"), segmentInfo_)
      #cv2_imshow(output.get_image()[:,:,::-1])


      #cv2.imshow("Result", output.get_image()[:,:,::-1])
      #cv2_imshow(output.get_image()[:,:,::-1])
      #cv2.waitKey(0)
      return pred_arr, pred_hierarchy, Info_with_label, output.get_image()[:,:,::-1][:,:,1]

    def onRealTimeVideo(self, frame):
      image = frame
      if self.model_type != "PS":
          predictions = self.predictor(image)
          viz = Visualizer(image[:, :, ::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
          instance_mode=ColorMode.SEGMENTATION)
          output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

      else:
          predictions, segmentInfo = self.predictor(image)["panoptic_seg"]

          viz = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))

          output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

      #cv2.imshow("Result", output.get_image()[:,:,::-1])
      #cv2_imshow(output.get_image()[:,:,::-1])
      #cv2.waitKey(0)
      return output.get_image()[:,:,::-1]


    def onVideo(self, videoPath):
        # open camera, read to get frame and make panoptic predictions, output save the result and finally display
        cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened()==False):
            print("Error opening video stream or file")
            return

        (success, image) = cap.read() #in case cv2.VideoCaptur(0) here cam is opened

        while success:
            if self.model_type != "PS":
                predictions = self.predictor(image)
                viz = Visualizer(image[:, :, ::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                instance_mode=ColorMode.SEGMENTATION)

                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

            else:
                predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
                print(predictions.shape)
                viz = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

            #cv2_imshow("Result", output.get_image()[:,:,::-1])
            cv2_imshow(output.get_image()[:,:,::-1])


            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

        # missing filter however I will continue, becasue this are to fiew non-relevant objects and maybe will take more time than expected


    def FindPredClassId(self):

          # Load metadata to get class names and IDs
          metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
          """
          meta_list = inspect.getmembers(metadata, predicate=inspect.ismethod)
          r = [print(x) for x in meta_list]

          # Print all class labels
          print('-------------------\nThings Dataset:\n', "IDX  ID  CLASS" , '\n---------------')
          dict_metadataId = metadata.thing_dataset_id_to_contiguous_id
          for idx, class_name in enumerate(metadata.thing_classes):
              print(idx , list(dict_metadataId.keys())[list(dict_metadataId.values()).index(idx)], class_name)

          print('-------------------\nStuff Dataset:\n', "IDX  ID  CLASS" , '\n---------------')
          dict_metadataId = metadata.stuff_dataset_id_to_contiguous_id
          for idx, class_name in enumerate(metadata.stuff_classes):
              print(idx, list(dict_metadataId.keys())[list(dict_metadataId.values()).index(idx)], class_name)
          """
          """
          MetaDict_stuff = {}
          for idx, class_name in enumerate(metadata.stuff_classes):
              #print(idx , list(dict_metadataId.keys())[list(dict_metadataId.values()).index(idx)], class_name)
              MetaDict_stuff[idx] = [class_name]

          return MetaDict_stuff
          """

class Midas:
  def __init__(self, model_att, trans_processing, device, thresh_m):
    self.model_att = model_att
    self.trans_processing = trans_processing
    self.device = device
    self.thresh_m = thresh_m

  def onImage_m(self, imagePath):
      # load image and apply transformers
      img = cv2.imread(imagePath)
      start_time = time.time()
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      #print('trans pross:' , self.trans_processing)
      input_batch = self.trans_processing(img).to( self.device)

      # Predict and resize to original
      with torch.no_grad():
          prediction = self.model_att(input_batch)

          prediction = torch.nn.functional.interpolate( #resize techniques
              prediction.unsqueeze(1),  # just one channel
              size=img.shape[:2], # target size
              mode="bicubic", # interpolation algorithm to reusze, bicubic smoother
              align_corners=False, # input and output corners not net for alignment
          ).squeeze() # remove extra dims

      img_out = prediction.cpu().numpy()
      img_transpose = img_out.T # values are inverted
      #print(img_transpose[:30, -30:])
      #plt.imshow(proximity_out)
      # plt.show()

      #print('prections pos-processing done')

      #===============================Algorithm for proximity===============================
      #print(img_outTranspose)
      proximity_out = img_out.copy()
      uniqeu_out = np.unique(proximity_out)
      img_dis = (proximity_out+min(uniqeu_out))*(255/(max(uniqeu_out)-min(uniqeu_out)))

      q1 = np.percentile(img_out, 25)  # First quartile (Q1)
      q2 = np.percentile(img_out, 50)  # Second quartile (Q2 or median)
      q3 = np.percentile(img_out, 75)  # Third quartile (Q3)

      #proximity_out[proximity_out <= q1] = q1 #far
      #proximity_out[(proximity_out > q1) & (proximity_out <= q2)] = q2 #near
      #proximity_out[proximity_out > q2] = q3 # very near

      proximity_out[proximity_out <= 5] = 5 #far
      proximity_out[(proximity_out > 5) & (proximity_out <= 15)] = 15 #near
      proximity_out[proximity_out > 15] = 20 # very near

      #==================================== display image=======================================
      #plt.imshow(proximity_out)
      #img_dis = (proximity_out/256).astype(np.uint8)
      #cv2.applyColorMap(img_dis, cv2.COLORMAP_PLASMA)
      cv2_imshow(img_dis)
      return proximity_out

      """
            fps = 1 / (time.time() - start_time)
            print('fps: ', fps)
            #return proximity_out
            cv2.putText(proximity_out,f"FPS is {int(fps)}",(15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255,255,0), 1)
            #cv2_imshow(img_out)
            plt.imshow(proximity_out)
      """
      #mymodel = MidasClass(midas, transform)
      #mymodel.onImage_m("./input.jpg")

  def onVideo_m(self, frame):
      # load image and apply transformers
      start_time = time.time()
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      #print('trans pross:' , self.trans_processing)
      input_batch = self.trans_processing(img).to(self.device)

      # Predict and resize to original
      with torch.no_grad():
          prediction = self.model_att(input_batch)

          prediction = torch.nn.functional.interpolate( #resize techniques
              prediction.unsqueeze(1),  # just one channel
              size=img.shape[:2], # target size
              mode="bicubic", # interpolation algorithm to reusze, bicubic smoother
              align_corners=False, # input and output corners not net for alignment
          ).squeeze() # remove extra dims

      img_out = prediction.cpu().numpy()
      img_transpose = img_out.T # values are inverted
      #print(img_transpose[:30, -30:])
      #plt.imshow(proximity_out)
      # plt.show()

      #print('prections pos-processing done')

      #===============================Algorithm for proximity===============================
      #print(img_outTranspose)
      proximity_out = img_out.copy()
      uniqeu_out = np.unique(proximity_out)
      img_dis = (proximity_out+min(uniqeu_out))*(255/(max(uniqeu_out)-min(uniqeu_out)))

      if self.thresh_m < np.percentile(img_out, 75):
        img_out = img_out[img_out < self.thresh_m]
        p1 = np.percentile(img_out, 25)  # First quartile (Q1)
        p2 = np.percentile(img_out, 60)  # Second quartile (Q2 or median)
        p3 = np.percentile(img_out, 70)  # Third quartile (Q3)
      
      else:
        p1 = np.percentile(img_out, 25)  # First quartile (Q1)
        p2 = np.percentile(img_out, 60)  # Second quartile (Q2 or median)
        p3 = np.percentile(img_out, 90)  # Third quartile (Q3)

      proximity_out[proximity_out <= p1] = p1 #far
      proximity_out[(proximity_out > p2) & (proximity_out <= p3)] = p2 #near
      proximity_out[proximity_out > p3] = p3 # very near

      #proximity_out[proximity_out <= q1] = q1 #far
      #proximity_out[(proximity_out > q1) & (proximity_out <= q2)] = q2 #near
      #proximity_out[proximity_out > q2] = q3 # very near

      #==================================== display image=======================================
      #plt.imshow(proximity_out)
      #img_dis = (proximity_out/256).astype(np.uint8)
      #cv2.applyColorMap(img_dis, cv2.COLORMAP_PLASMA)
      #cv2_imshow(img_dis)
      return proximity_out


  def onVideo(self, videoPath):
      cap = cv2.VideoCapture(videoPath)

      if (cap.isOpened()==False):
          print("Error opening video stream or file")
          return

      (success, image) = cap.read() #in case cv2.VideoCaptur(0) here cam is opened

      while success:
            start_time = time.time()
            # pre-processing
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            img_resized = tf.image.resize(img, [384,384], method='bicubic', preserve_aspect_ratio=False)
            img_resized = tf.transpose(img_resized, [2, 0, 1])
            img_input = img_resized.numpy()
            reshape_img = img_input.reshape(1,3,384,384)
            tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

            # predictions
            output = self.MidasModel.signatures['serving_default'](tensor)
            depth_map = output['default'].numpy()

            # post-processing
            depth_map = depth_map.reshape(384, 384)
            depth_map = cv2.resize(depth_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            img_out = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype("uint8") # change into 255 value range
            img_outTranspose = img_out.T # values are inverted

            fps = 1 / (time.time() - start_time)
            cv2.putText(img_out,f"FPS is {int(fps)}",(15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255,255,0), 1)
            cv2_imshow(img_out)

            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

class MobileCam(Midas, Detector):
  def __init__(self, midas_model_att, trans_processing, device, model_type, thresh):
    Midas.__init__(self, midas_model_att, trans_processing, device, thresh)
    Detector.__init__(self, model_type)
    self.model_type = Detector.get_attributes(self)

  def show_attribute(self):
    return self.get_attributes()

  def MultOut_img(self, path):
    # hierarchy for stuff
    segment_arr, hierarchy_arr, SegmentInfo = self.onImage_d(path)
    segment_arr, hierarchy_arr = np.flip(segment_arr.numpy()), np.flip(hierarchy_arr)
    pred_id = SegmentInfo['id']
    pred_class = SegmentInfo['class_label']

    #segment_arr = segment_arr.numpy()
    segment_vrvn, segment_vrn, segment_rvn, segment_rn, segment_nn = segment_arr.copy(), segment_arr.copy(), segment_arr.copy(), segment_arr.copy(), segment_arr.copy()
    segment_vrvn[hierarchy_arr != 1] = 0 # very relevant very near
    segment_rvn[hierarchy_arr != 2] = 0 # relevant very near
    segment_vrn[hierarchy_arr != 1] = 0 # very relevant near
    segment_rn[hierarchy_arr != 2] = 0 # relevant near
    segment_nn[hierarchy_arr != 3] = 0 # not relevant

    # ============================ hierarchy for depth ====================================
    depth_array = self.onImage_m(path)
    #depth_array = depth_array.T
    depth_thresh = np.unique(depth_array)
    segment_vrvn[depth_array != depth_thresh[-1]] = 0 # Very Relevant and very near
    segment_rvn[depth_array != depth_thresh[-1]] = 0 # Relevant and very near
    segment_vrn[depth_array != depth_thresh[-2]] = 0 # Very Relevant and near
    segment_rn[depth_array != depth_thresh[-2]] = 0 # Relevant and near
    print(depth_thresh)
    # ============ Predict the poistion for each object / stuff detected ===================
    h_mod = len(segment_arr) % 3
    w_mod = len(segment_arr[0]) % 3
    if h_mod != 0:
        segment_arr = segment_arr[:-h_mod, :]
    if w_mod != 0:
        segment_arr = segment_arr[:, :-w_mod]

    # get the amount the amount of pixels they correspond for each quadrant
    h = len(segment_arr) // 3
    w = len(segment_arr[0]) // 3
    q_area = h * w

    # devided into grid of 3 x 3
    quad = [segment_arr[:h, :w], segment_arr[:h, w:2*w], segment_arr[:h, 2*w:],
        segment_arr[h:2*h, :w], segment_arr[h:2*h, w:2*w], segment_arr[h:2*h, 2*w:],
        segment_arr[2*h:, :w], segment_arr[2*h:, w:2*w], segment_arr[2*h:, 2*w:]
        ] # quadrants

    # get the prediction for each label
    quad_dict = {0: 'superior izquierda', 1: 'centro superior', 2: 'superior derecha',
                      3: 'medio izquierda' , 4: 'medio centro' , 5:'medio derecha' ,
                      6:'inferior izquierda' , 7: 'centro inferior', 8: 'inderior derecha'}

    quad_dict = {0: 'derecha', 1: 'centro', 2: 'izquierda',
              3: 'derecha' , 4: 'centro' , 5:'izquierda' ,
              6:'derecha' , 7: 'centro', 8: 'izquierda'}

    # class unique class id
    id_dict = {l:np.array([]) for l in pred_id}

    for k in id_dict:
        for q in quad:
            id_dict[k] = np.append(len(q[q == k]), id_dict[k])
        id_dict[k] = id_dict[k]/q_area
        id_dict[k] = quad_dict[np.where(id_dict[k] == max(id_dict[k]))[0][0]] #[::-1] index for quadrant
    #print(id_dict)

    # ========================================= display ==================================
    vr_vn = [(pred_class[pred_id.index(i)], id_dict[i]) for i in np.unique(segment_vrvn) if i != 0]
    #print('\nVery Relevant, Very Near:', vr_vn)
    r_vn = [(pred_class[pred_id.index(i)], id_dict[i]) for i in np.unique(segment_rvn) if i != 0]
    vr_n = [(pred_class[pred_id.index(i)], id_dict[i]) for i in np.unique(segment_vrn) if i != 0]
    r_n = [(pred_class[pred_id.index(i)], id_dict[i]) for i in np.unique(segment_rn) if i != 0]

    text = []
    vr_vn_len = len(vr_vn) > 0
    if vr_vn_len:
      text.append('\nprecaución acercándose a')
      #[print(p[0] + ' '+ p[1] + ' ') for p in vr_vn]
      [text.append(p[0] + ' '+ p[1] + ' ') for p in vr_vn]

    if len(r_vn) > 0:
      if not vr_vn_len:
        text.append('\nprecaución acercándose a')
      [text.append(p[0] + ' '+ p[1] + ' ') for p in r_vn]

    vr_n_l = len(vr_n) > 0
    if vr_n_l:
      text.append('\npróximamente')
      [text.append(p[0] + ' '+ p[1] + ' ') for p in vr_n]

    if len(r_n) > 0:
      if not vr_n_l:
        text.append('\npróximamente')
      [text.append(p[0] + ' '+ p[1] + ' ') for p in r_n]

    # Text to speech automatic play
    text = ', '.join(text)
    #print(text)
    tts = gTTS(text=text, lang='es') 
    tts.save('1.wav') 
    sound_file = '1.wav'
    return Audio(sound_file, autoplay=True)


    """
    #segment_arr = segment_arr.numpy()
    segment_arr_1, segment_arr_2, segment_arr_3 = segment_arr.copy(), segment_arr.copy(), segment_arr.copy()
    segment_arr_1[hierarchy_arr != 1] = 0 # very relevant
    segment_arr_2[hierarchy_arr != 2] = 0 # relevant
    segment_arr_3[hierarchy_arr != 3] = 0 # not relevant

    # hierarchy for depth
    depth_array = self.onImage_m(path)
    depth_array = np.flip(depth_array)
    depth_thresh = np.unique(depth_array)
    segment_arr_1[depth_array != depth_thresh[-1]] = 0 # Very Relevant and very near
    segment_arr_2[depth_array != depth_thresh[-2]] = 0 # Relevant and near

    # ============ Predict the poistion for each object / stuff detected ===================
    h_mod = len(segment_arr) % 3
    w_mod = len(segment_arr[0]) % 3
    if h_mod != 0:
        segment_arr = segment_arr[:-h_mod, :]
    if w_mod != 0:
        segment_arr = segment_arr[:, :-w_mod]

    # get the amount the amount of pixels they correspond for each quadrant
    h = len(segment_arr) // 3
    w = len(segment_arr[0]) // 3
    q_area = h * w

    # devided into grid of 3 x 3
    quad = [segment_arr[:h, :w], segment_arr[:h, w:2*w], segment_arr[:h, 2*w:],
        segment_arr[h:2*h, :w], segment_arr[h:2*h, w:2*w], segment_arr[h:2*h, 2*w:],
        segment_arr[2*h:, :w], segment_arr[2*h:, w:2*w], segment_arr[2*h:, 2*w:]
        ] # quadrants

    # get the prediction for each label
    quad_dict = {0: 'left top', 1: 'center top', 2: 'right top',
                  3: 'middle left' , 4: 'middle center' , 5:'middle right' ,
                  6:'down left' , 7: 'down center', 8: 'down right'}

    # class unique class id
    id_dict = {l:np.array([]) for l in pred_id}

    for k in id_dict:
        for q in quad:
            id_dict[k] = np.append(len(q[q == k]), id_dict[k])
        id_dict[k] = id_dict[k]/q_area
        id_dict[k] = quad_dict[np.where(id_dict[k] == max(id_dict[k]))[0][0]] #[::-1] index for quadrant
    print(id_dict)

    # ========================================= display ==================================
    text_out = [(pred_class[pred_id.index(i)], id_dict[i]) for i in np.unique(segment_arr_1) if i != 0]
    print('\nVery Relevant, Very Near:', text_out)
    text_out = [(pred_class[pred_id.index(i)], id_dict[i]) for i in np.unique(segment_arr_2) if i != 0]
    print('Relevant, Near:', text_out, '\n')
    """
    # ============still missing:
    # Very Relevant and near (1,100)
    # Relevant and near (2, 100)
    # Non Relevant and very near (3,255)
    # Non Relevant and near (3, 100)
    # Very Relevant and far (1,0)
    # Relevant and far (2, 0)
    # Non Relevant and far (3, 0)


  def MultOut_RealTime(self, disp_pred=False):
    # start streaming video from webcam
    video_stream()
    # label for video
    label_html = 'Capturing...'
    # initialze bounding box to empty
    bbox = ''
    count = 0
    while True:
        js_reply = video_frame(label_html, bbox)
        if not js_reply:
            break

        # convert JS response to OpenCV Image
        frame = js_to_image(js_reply["img"])

        # create transparent overlay for bounding box
        bbox_array = np.zeros([480,640,4], dtype=np.uint8)

        # ========================== hierarchy for obj detection ====================================
        segment_arr, hierarchy_arr, SegmentInfo, bbox_array[:,:,3] = self.onVideo_d(frame)
        segment_arr, hierarchy_arr = segment_arr.T, hierarchy_arr.T
        pred_id = SegmentInfo['id']
        pred_class = SegmentInfo['class_label']

        if disp_pred == True:
          bbox_bytes = bbox_to_bytes(bbox_array)
          bbox = bbox_bytes
        else:
          pass

        #print(pred_class)
        segment_arr = segment_arr.numpy()
        segment_vrvn, segment_vrn, segment_rvn, segment_rn, segment_nn = segment_arr.copy(), segment_arr.copy(), segment_arr.copy(), segment_arr.copy(), segment_arr.copy()
        segment_vrvn[hierarchy_arr != 1] = 0 # very relevant very near
        segment_rvn[hierarchy_arr != 2] = 0 # relevant very near
        segment_vrn[hierarchy_arr != 1] = 0 # very relevant near
        segment_rn[hierarchy_arr != 2] = 0 # relevant near
        segment_nn[hierarchy_arr != 3] = 0 # not relevant

        # ============================ hierarchy for depth ====================================
        depth_array = self.onVideo_m(frame)
        depth_array = depth_array.T
        depth_thresh = np.unique(depth_array)

        segment_vrvn[depth_array != depth_thresh[-1]] = 0 # Very Relevant and very near
        segment_rvn[depth_array != depth_thresh[-1]] = 0 # Relevant and very near
        segment_vrn[depth_array != depth_thresh[-2]] = 0 # Very Relevant and near
        segment_rn[depth_array != depth_thresh[-2]] = 0 # Relevant and near

        # ============ Predict the poistion for each object / stuff detected ===================
        h_mod = len(segment_arr) % 3
        w_mod = len(segment_arr[0]) % 3
        if h_mod != 0:
            segment_arr = segment_arr[:-h_mod, :]
        if w_mod != 0:
            segment_arr = segment_arr[:, :-w_mod]

        # get the amount the amount of pixels they correspond for each quadrant
        h = len(segment_arr) #// 3
        w = len(segment_arr[0]) // 3
        q_area = h * w

        """# devided into grid of 3 x 3
        quad = [segment_arr[:h, :w], segment_arr[:h, w:2*w], segment_arr[:h, 2*w:],
            segment_arr[h:2*h, :w], segment_arr[h:2*h, w:2*w], segment_arr[h:2*h, 2*w:],
            segment_arr[2*h:, :w], segment_arr[2*h:, w:2*w], segment_arr[2*h:, 2*w:]
            ] # quadrants"""
        # devided into grid of 3 x 1
        quad = [segment_arr[:, :w], segment_arr[:, w:2*w], segment_arr[:, 2*w:]] # quadrants""

        # get the prediction for each label
        #quad_dict = {0: 'superior izquierda', 1: 'centro superior', 2: 'superior derecha',
        #              3: 'medio izquierda' , 4: 'medio centro' , 5:'medio derecha' ,
        #              6:'inferior izquierda' , 7: 'centro inferior', 8: 'inderior derecha'}

        quad_dict = {0: 'iz', 1: 'fr', 2: 'de'}

        # class unique class id
        id_dict = {l:np.array([]) for l in pred_id}

        for k in id_dict:
            for q in quad:
                id_dict[k] = np.append(len(q[q == k]), id_dict[k])
            #id_dict[k] = id_dict[k]/q_area
            id_dict[k] = quad_dict[np.where(id_dict[k] == max(id_dict[k]))[0][0]] #[::-1] index for quadrant
        #print(id_dict)

        # ========================================= display ==================================
        vr_vn = [(pred_class[pred_id.index(i)], id_dict[i]) for i in np.unique(segment_vrvn) if i != 0]
        #print('\nVery Relevant, Very Near:', vr_vn)
        r_vn = [(pred_class[pred_id.index(i)], id_dict[i]) for i in np.unique(segment_rvn) if i != 0]
        vr_n = [(pred_class[pred_id.index(i)], id_dict[i]) for i in np.unique(segment_vrn) if i != 0]
        r_n = [(pred_class[pred_id.index(i)], id_dict[i]) for i in np.unique(segment_rn) if i != 0]


        text = []
        vr_vn_len = len(vr_vn) > 0
        #if vr_vn_len:
        #  text.append('\nprecaución acercándose a')
        #  #[print(p[0] + ' '+ p[1] + ' ') for p in vr_vn]
        #  [text.append(p[0] + ' '+ p[1] + ' ') for p in vr_vn]

        #if len(r_vn) > 0:
          #if not vr_vn_len:
          #text.append('\nprecaución acercándose a')
          #[text.append(p[0] + ' '+ p[1] + ' ') for p in r_vn]
            
        vr_n_l = len(vr_n) > 0
        if vr_n_l:
          text.append('\npróximamente')
          iz, fr, de = ['izquieda: '], ['frente: '], ['derecha: ']
          for p in vr_n:
            if p[1] == 'iz':                 
              iz.append(p[0] + ' ')
            elif p[1] == 'fr':                 
              fr.append(p[0] + ' ')
            elif p[2] == 'de':
               de.append(p[0] + ' ')
          text.append(' '.join(iz) + ' '.join(fr), ' '.join(de))
            
        if len(r_n) > 0:
          #if not vr_n_l:
          #  text.append('\npróximamente')
          #[text.append(p[0] + ' '+ p[1] + ' ') for p in r_n]
            pass
        if len(text)==0:
          text =['  Sin objetos relevantes  ']

        # Text to speech automatic play
        text = ', '.join(text)
        #print(text)
        tts = gTTS(text=text, lang='es') 
        tts.save('1.wav') 
        sound_file = '1.wav'
        return Audio(sound_file, autoplay=True)
        
        #cv2.waitKey(3)

