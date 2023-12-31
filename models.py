from VideoStreamColab import js_to_image, bbox_to_bytes, video_stream, video_frame

from setup import call_midas_model

# Packages for Detectron2
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
import matplotlib.pyplot as plt


# packages for Mobile Cam
import os 
os.system('pip install gtts pydub') 
from gtts import gTTS #Import Google Text to Speech
from pydub import AudioSegment
from IPython.display import display, Audio, clear_output #Import Audio method from IPython's Display Class

class Detector: # Class reference from https://github.com/evanshlom/detectron2-panoptic-segmentation-video/blob/main/Detector.py
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

        #============================== Hierarchical Indexation  =================================
            # semantic segementation hierarchy dictionary
            stuff_hierarchy = {'things': 1, 'banner': 1, 'blanket': 3, 'bridge': 1, 'cardboard': 1, 'counter': 1, 'curtain': 2, 'door-stuff': 1, 'floor-wood': 2, 'flower': 3, 'fruit': 2, 'gravel': 3, 'house': 3, 'light': 3, 'mirror-stuff': 1, 'net': 2, 'pillow': 3, 'platform': 1, 'playingfield': 3, 'railroad': 1, 'river': 1, 'road': 1, 'roof': 3, 'sand': 3, 'sea': 3, 'shelf': 1, 'snow': 1, 'stairs': 1, 'tent': 1, 'towel': 2, 'wall-brick': 1, 'wall-stone': 1, 'wall-tile': 1, 'wall-wood': 1, 'water': 1, 'window-blind': 3, 'window': 1, 'tree': 1, 'fence': 1, 'ceiling': 3, 'sky': 3, 'cabinet': 1, 'table': 1, 'floor': 3, 'pavement': 3, 'mountain': 3, 'grass': 3, 'dirt': 3, 'paper': 2, 'food': 2, 'building': 3, 'rock': 1, 'wall': 1, 'rug': 2}
            stuff_cat_id = {i: c for i, c in enumerate(metadata.stuff_classes)} # dict with index and classes
            #instance segementation hierarchy dictionary
            thing_hierarchy = {'person': 1, 'bicycle': 1, 'car': 1, 'motorcycle':1, 'airplane':3, 'bus':1, 'train':1, 'truck':1, 'boat':3, 'traffic light':1, 'fire hydrant':1, 'stop sign':1, 'parking meter':1, 'bench':1, 'bird':3, 'cat':1, 'dog':1, 'horse':1, 'sheep':1, 'cow':1, 'elephant':1, 'bear':1, 'zebra':1, 'giraffe':1, 'backpack':1, 'umbrella':1, 'handbag':1, 'tie':3, 'suitcase':1, 'frisbee':1, 'skis':1, 'snowboard':1, 'sports ball':1, 'kite':3, 'baseball bat':1, 'baseball glove':1, 'skateboard':1, 'surfboard':3, 'tennis racket':1, 'bottle':2, 'wine glass':1, 'cup':1, 'fork':1, 'knife':1, 'spoon':2, 'bowl':3, 'banana':3, 'apple':3, 'sandwich':3, 'orange':3, 'broccoli':3, 'carrot':3, 'hot dog':3, 'pizza':3, 'donut':3, 'cake':3, 'chair':1, 'couch':1, 'potted plant':1, 'bed':1, 'dining table':1, 'toilet':1, 'tv':3, 'laptop':3, 'mouse':3, 'remote':3, 'keyboard':3, 'cell phone':3, 'microwave':3, 'oven':2, 'toaster':2, 'sink':2, 'refrigerator':1, 'book':3, 'clock':3, 'vase':2, 'scissors':2, 'teddy bear':3, 'hair drier':2, 'toothbrush':2}
            thing_cat_id = {i: c for i,c in enumerate(metadata.thing_classes)}

            # map hierarchy to each object and append in segmentInfo
            for segment in segmentInfo:
              if not segment['isthing']:
                  label = stuff_cat_id[segment['category_id']]
                  segment['class_label'] = label
                  segment['class_hierarchy'] = stuff_hierarchy[label]
              else:
                  label = thing_cat_id[segment['category_id']]
                  segment['class_label'] = label
                  segment['class_hierarchy'] = thing_hierarchy[label]

            # semantic and instance dictionaries with predicted objects id and they corresponding hierarchy
            pred_hierarchy_dict_stuff = {x['id']:x['class_hierarchy'] for x in segmentInfo if x['isthing'] == False}
            pred_hierarchy_dict_thing = {x['id']:x['class_hierarchy'] for x in segmentInfo if x['isthing'] == True}

            # select objects from semantic and instance segmentation in the frame
            pred_arr = predictions.cpu()
            pred_set_id_stuff = set(pred_hierarchy_dict_stuff.keys())
            pred_set_id_thing = set(pred_hierarchy_dict_thing.keys())

            # get hierarchy mask, need to be working with a three if statements (changes in the stuff, in the thing, in 0)
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

            
            # Simple translation
            trans_dict_stuff = {'things': 'objetos', 'banner': 'letrero', 'blanket': 'sábana', 'bridge': 'puente', 'cardboard': 'cartón', 'counter': 'mostrador', 'curtain': 'cortina', 'door-stuff': 'puerta', 'floor-wood': 'piso', 'flower': 'flor', 'fruit': 'fruta', 'gravel': 'grava', 'house': 'casa', 'light': 'luz', 'mirror-stuff': 'espejo', 'net': 'red', 'pillow': 'almohada', 'platform': 'plataforma', 'playingfield': 'cancha de juego', 'railroad': 'ferrocarril', 'river': 'río', 'road': 'calle', 'roof': 'tejado', 'sand': 'arena', 'sea': 'oceano', 'shelf': 'repisas', 'snow': 'nieve', 'stairs': 'escalera', 'tent': 'tienda de campaña', 'towel': 'toalla', 'wall-brick': 'pared', 'wall-stone': 'pared', 'wall-tile': 'pared de teja', 'wall-wood': 'pared', 'water': 'agua', 'window-blind': 'persiana', 'window': 'ventana', 'tree': 'arbol', 'fence': 'valla', 'ceiling': 'techo', 'sky': 'cielo', 'cabinet': 'gabinete', 'table': 'mesa', 'floor': 'piso', 'pavement': 'pavimento', 'mountain': 'montaña', 'grass': 'pasto', 'dirt': 'tierra', 'paper': 'papel', 'food': 'comida', 'building': 'edificio', 'rock': 'roca', 'wall': 'pared', 'rug': 'alfombra'}
            trans_dict_things = {'person': 'persona', 'bicycle': 'bicicleta', 'car': 'coche', 'motorcycle': 'motocicleta', 'airplane': 'avión', 'bus': 'autobus', 'train': 'tren', 'truck': 'camioneta', 'boat': 'bote', 'traffic light': 'semáforo', 'fire hydrant': 'hidratante', 'stop sign': 'señalización de alto', 'parking meter': 'parquímetro', 'bench': 'banca', 'bird': 'pájaro', 'cat': 'gato', 'dog': 'perro', 'horse': 'caballo', 'sheep': 'oveja', 'cow': 'vaca', 'elephant': 'elefante', 'bear': 'oso', 'zebra': 'cebra', 'giraffe': 'jirafa', 'backpack': 'mochila', 'umbrella': 'sombrilla', 'handbag': 'maleta de mano', 'tie': 'corbata', 'suitcase': 'portafolio', 'frisbee': 'frisbee', 'skis': 'skis', 'snowboard': 'snowboard', 'sports ball': 'balón', 'kite': 'cometa', 'baseball bat': 'bate de baseball', 'baseball glove': 'guante de baseball', 'skateboard': 'patineta', 'surfboard': 'tabla de surf', 'tennis racket': 'raqueta', 'bottle': 'botella', 'wine glass': 'copa de vino', 'cup': 'tasa', 'fork': 'tenedor', 'knife': 'cuchillo', 'spoon': 'cuchara', 'bowl': 'tasón', 'banana': 'plátano', 'apple': 'manzana', 'sandwich': 'sandwich', 'orange': 'naraja', 'broccoli': 'broccoli', 'carrot': 'zanahoria', 'hot dog': 'hot dog', 'pizza': 'pizza', 'donut': 'dona', 'cake': 'pastel', 'chair': 'silla', 'couch': 'sillón', 'potted plant': 'maceta', 'bed': 'cama', 'dining table': 'mesa', 'toilet': 'escusado', 'tv': 'televisión', 'laptop': 'laptop', 'mouse': 'mouse', 'remote': 'control remoto', 'keyboard': 'teclado', 'cell phone': 'celular', 'microwave': 'microondas', 'oven': 'horno', 'toaster': 'tostador', 'sink': 'lavabo', 'refrigerator': 'refrigerador', 'book': 'libro', 'clock': 'reloj', 'vase': 'jarrón', 'scissors': 'tijeras', 'teddy bear': 'oso de peluche', 'hair drier': 'secadora de pelo', 'toothbrush': 'pasta de dientes'}
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
        cv2_imshow(output.get_image()[:,:,::-1])

        return pred_arr, pred_hierarchy, Info_with_label

    def onVideo_d(self, frame):
      predictions, segmentInfo = self.predictor(frame)["panoptic_seg"]
      metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
      
      #============================== Hierarchical Indexation  =================================
      # set the class label and class heirarchy inside segmentInfo
      stuff_hierarchy = {'things': 1, 'banner': 1, 'blanket': 3, 'bridge': 1, 'cardboard': 1, 'counter': 1, 'curtain': 2, 'door-stuff': 1, 'floor-wood': 2, 'flower': 3, 'fruit': 2, 'gravel': 3, 'house': 3, 'light': 3, 'mirror-stuff': 1, 'net': 2, 'pillow': 3, 'platform': 1, 'playingfield': 3, 'railroad': 1, 'river': 1, 'road': 1, 'roof': 3, 'sand': 3, 'sea': 3, 'shelf': 2, 'snow': 1, 'stairs': 1, 'tent': 1, 'towel': 2, 'wall-brick': 2, 'wall-stone': 2, 'wall-tile': 2, 'wall-wood': 2, 'water': 1, 'window-blind': 3, 'window': 1, 'tree': 1, 'fence': 1, 'ceiling': 3, 'sky': 3, 'cabinet': 1, 'table': 1, 'floor': 3, 'pavement': 3, 'mountain': 3, 'grass': 3, 'dirt': 3, 'paper': 2, 'food': 2, 'building': 3, 'rock': 1, 'wall': 1, 'rug': 2}
      stuff_cat_id = {i: c for i, c in enumerate(metadata.stuff_classes)} # dict with index and classes

      thing_hierarchy = {'person': 1, 'bicycle': 1, 'car': 1, 'motorcycle':1, 'airplane':3, 'bus':1, 'train':1, 'truck':1, 'boat':3, 'traffic light':1, 'fire hydrant':1, 'stop sign':1, 'parking meter':1, 'bench':1, 'bird':3, 'cat':1, 'dog':1, 'horse':1, 'sheep':1, 'cow':1, 'elephant':1, 'bear':1, 'zebra':1, 'giraffe':1, 'backpack':1, 'umbrella':1, 'handbag':1, 'tie':3, 'suitcase':1, 'frisbee':1, 'skis':1, 'snowboard':1, 'sports ball':1, 'kite':3, 'baseball bat':1, 'baseball glove':1, 'skateboard':1, 'surfboard':3, 'tennis racket':1, 'bottle':2, 'wine glass':1, 'cup':1, 'fork':1, 'knife':1, 'spoon':2, 'bowl':3, 'banana':3, 'apple':3, 'sandwich':3, 'orange':3, 'broccoli':3, 'carrot':3, 'hot dog':3, 'pizza':3, 'donut':3, 'cake':3, 'chair':1, 'couch':1, 'potted plant':1, 'bed':1, 'dining table':1, 'toilet':1, 'tv':3, 'laptop':3, 'mouse':3, 'remote':3, 'keyboard':3, 'cell phone':3, 'microwave':3, 'oven':2, 'toaster':2, 'sink':2, 'refrigerator':1, 'book':3, 'clock':3, 'vase':2, 'scissors':2, 'teddy bear':3, 'hair drier':2, 'toothbrush':2}
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
      trans_dict_stuff = {'things': 'objetos', 'banner': 'letrero', 'blanket': 'sábana', 'bridge': 'puente', 'cardboard': 'cartón', 'counter': 'mostrador', 'curtain': 'cortina', 'door-stuff': 'puerta', 'floor-wood': 'piso', 'flower': 'flor', 'fruit': 'fruta', 'gravel': 'grava', 'house': 'casa', 'light': 'luz', 'mirror-stuff': 'espejo', 'net': 'red', 'pillow': 'almohada', 'platform': 'plataforma', 'playingfield': 'cancha de juego', 'railroad': 'ferrocarril', 'river': 'río', 'road': 'calle', 'roof': 'tejado', 'sand': 'arena', 'sea': 'oceano', 'shelf': 'repisas', 'snow': 'nieve', 'stairs': 'escalera', 'tent': 'tienda de campaña', 'towel': 'toalla', 'wall-brick': 'pared', 'wall-stone': 'pared', 'wall-tile': 'pared de teja', 'wall-wood': 'pared', 'water': 'agua', 'window-blind': 'persiana', 'window': 'ventana', 'tree': 'árbol', 'fence': 'valla', 'ceiling': 'techo', 'sky': 'cielo', 'cabinet': 'gabinete', 'table': 'mesa', 'floor': 'piso', 'pavement': 'pavimento', 'mountain': 'montaña', 'grass': 'pasto', 'dirt': 'tierra', 'paper': 'papel', 'food': 'comida', 'building': 'edificio', 'rock': 'roca', 'wall': 'pared', 'rug': 'alfombra'}
      trans_dict_things = {'person': 'persona', 'bicycle': 'bicicleta', 'car': 'coche', 'motorcycle': 'motocicleta', 'airplane': 'avión', 'bus': 'autobus', 'train': 'tren', 'truck': 'camioneta', 'boat': 'bote', 'traffic light': 'semáforo', 'fire hydrant': 'hidratante', 'stop sign': 'señalización de alto', 'parking meter': 'parquímetro', 'bench': 'banca', 'bird': 'pájaro', 'cat': 'gato', 'dog': 'perro', 'horse': 'caballo', 'sheep': 'oveja', 'cow': 'vaca', 'elephant': 'elefante', 'bear': 'oso', 'zebra': 'cebra', 'giraffe': 'jirafa', 'backpack': 'mochila', 'umbrella': 'sombrilla', 'handbag': 'maleta de mano', 'tie': 'corbata', 'suitcase': 'portafolio', 'frisbee': 'frisbee', 'skis': 'skis', 'snowboard': 'snowboard', 'sports ball': 'balón', 'kite': 'cometa', 'baseball bat': 'bate de baseball', 'baseball glove': 'guante de baseball', 'skateboard': 'patineta', 'surfboard': 'tabla de surf', 'tennis racket': 'raqueta', 'bottle': 'botella', 'wine glass': 'copa de vino', 'cup': 'tasa', 'fork': 'tenedor', 'knife': 'cuchillo', 'spoon': 'cuchara', 'bowl': 'tasón', 'banana': 'plátano', 'apple': 'manzana', 'sandwich': 'sandwich', 'orange': 'naraja', 'broccoli': 'broccoli', 'carrot': 'zanahoria', 'hot dog': 'hot dog', 'pizza': 'pizza', 'donut': 'dona', 'cake': 'pastel', 'chair': 'silla', 'couch': 'sillón', 'potted plant': 'maceta', 'bed': 'cama', 'dining table': 'mesa', 'toilet': 'escusado', 'tv': 'televisión', 'laptop': 'laptop', 'mouse': 'mouse', 'remote': 'control remoto', 'keyboard': 'teclado', 'cell phone': 'celular', 'microwave': 'microondas', 'oven': 'horno', 'toaster': 'tostador', 'sink': 'lavabo', 'refrigerator': 'refrigerador', 'book': 'libro', 'clock': 'reloj', 'vase': 'jarrón', 'scissors': 'tijeras', 'teddy bear': 'oso de peluche', 'hair drier': 'secadora de pelo', 'toothbrush': 'pasta de dientes'}
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
      return pred_arr, pred_hierarchy, Info_with_label

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

      #===============================Depth Awareness===============================
      proximity_out = img_out.copy()
      unique_out = np.unique(proximity_out)
      # array with the new values
      proximity_out = (proximity_out-min(unique_out))*(255/(max(unique_out)-min(unique_out)))
        
      print('\nnormalized to range(0,255):', min(np.unique(proximity_out)), max(np.unique(proximity_out)))

   
      initial_out_img = proximity_out.copy()
      #if self.thresh_m > np.percentile(proximity_out, 75):
      if np.std(initial_out_img) > self.thresh_m:
        #print('px reduced', 'std', np.std(initial_out_img))
        new_px_dis = proximity_out[proximity_out > np.percentile(proximity_out, 70)] # cut the pixels distribution
        p1 = np.percentile(new_px_dis, 25)  # First quartile (Q1)
        p2 = np.percentile(new_px_dis, 65)  # Second quartile (Q2 or median)
        p3 = np.percentile(new_px_dis, 99.9)  # Third quartile (Q3)
      
      else:
        #print('px mantained', 'std', np.std(initial_out_img))
        p1 = np.percentile(proximity_out, 25)  # First quartile (Q1)
        p2 = np.percentile(proximity_out, 65)  # Second quartile (Q2 or median)
        p3 = np.percentile(proximity_out, 99.9)  # Third quartile (Q3)

      #print('Percentiles: ', p1, p2, p3)
      proximity_out[proximity_out <= p2] = p1 #far
      proximity_out[(proximity_out > p2) & (proximity_out <= p3)] = p2 #near
      proximity_out[proximity_out > p3] = p3 # very near

      # rescaling for visualization
      proximity_out[proximity_out == p1] = 0 #far
      proximity_out[proximity_out == p2] = 150 #near
      proximity_out[proximity_out == p3] = 255 # very near
      #just the display is working for rotation, but is programed for 180° rot
      #cv2_imshow(np.rot90(proximity_out, 2))
      cv2_imshow(proximity_out)  
      
      return proximity_out, initial_out_img 


  def onVideo_m(self, frame):
      # load image and apply transformers
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

      #===============================Depth Awareness===============================
      proximity_out = img_out.copy()
      unique_out = np.unique(proximity_out)
      # array with the new values
      proximity_out = (proximity_out-min(unique_out))*(255/(max(unique_out)-min(unique_out)))

      initial_out_img = proximity_out.copy()
      #if self.thresh_m > np.percentile(proximity_out, 75):
      if np.std(initial_out_img) > self.thresh_m:
        #print('px reduced', 'std', np.std(initial_out_img))
        new_px_dis = proximity_out[proximity_out > np.percentile(proximity_out, 70)] # cut the pixels distribution
        p1 = np.percentile(new_px_dis, 25)  # First quartile (Q1)
        p2 = np.percentile(new_px_dis, 68)  # Second quartile (Q2 or median)
        p3 = np.percentile(new_px_dis, 99.9)  # Third quartile (Q3)
      
      else:
        #print('px mantained', 'std', np.std(initial_out_img))
        p1 = np.percentile(proximity_out, 25)  # First quartile (Q1)
        p2 = np.percentile(proximity_out, 68)  # Second quartile (Q2 or median)
        p3 = np.percentile(proximity_out, 99.9)  # Third quartile (Q3)

      proximity_out[proximity_out <= p2] = p1 #far
      proximity_out[(proximity_out > p2) & (proximity_out <= p3)] = p2 #near
      proximity_out[proximity_out > p3] = p3 # very near

      # rescaling for visualization
      proximity_out[proximity_out == p1] = 0 #far
      proximity_out[proximity_out == p2] = 150 #near
      proximity_out[proximity_out == p3] = 255 # very near
      return proximity_out

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
    segment_vrvn, segment_vrn, segment_rvn, segment_rn, segment_vrf = segment_arr.copy(), segment_arr.copy(), segment_arr.copy(), segment_arr.copy(), segment_arr.copy()
    segment_vrvn[hierarchy_arr != 1] = 0 # very relevant very near
    segment_rvn[hierarchy_arr != 2] = 0 # relevant very near
    segment_vrn[hierarchy_arr != 1] = 0 # very relevant near
    segment_rn[hierarchy_arr != 2] = 0 # relevant near
    segment_vrf[hierarchy_arr != 1] = 0 # very relevant far

    # ============================ hierarchy for depth ====================================
    depth_array, initial_out_img = self.onImage_m(path)
    #depth_array = depth_array.T
    depth_array = np.rot90(depth_array, 2)
    depth_thresh = np.unique(depth_array)
    segment_vrvn[depth_array != depth_thresh[-1]] = 0 # Very Relevant and very near
    segment_rvn[depth_array != depth_thresh[-1]] = 0 # Relevant and very near
    segment_vrn[depth_array != depth_thresh[-2]] = 0 # Very Relevant and near
    segment_rn[depth_array != depth_thresh[-2]] = 0 # Relevant and near
    segment_vrf[depth_array != depth_thresh[-3]] = 0 # Very relevant far
    
    #test visualization
    #print('unique depthmap:', np.unique(depth_thresh), '\nnear and very relevant')
    seg_out = segment_vrn.copy()
    seg_out[seg_out != 0] = 150
    cv2_imshow(np.rot90(seg_out, 2))
    #cv2_imshow(seg_out)  
    # ============ Predict the poistion for each object / stuff detected ===================
    h_mod = len(segment_arr) % 3
    w_mod = len(segment_arr[0]) % 3
    if h_mod != 0:
        segment_arr = segment_arr[:-h_mod, :]
    if w_mod != 0:
        segment_arr = segment_arr[:, :-w_mod]

    # get the amount the amount of pixels they correspond for each quadrant
    h = len(segment_arr) # // 3
    w = len(segment_arr[0]) // 3
    
    # devided into grid of 3 x 1
    quad = [segment_arr[:, :w], segment_arr[:, w:2*w], segment_arr[:, 2*w:]] # quadrants""
      
    quad_dict = {0: 'iz', 1: 'fr', 2: 'de'}

    # class unique class id
    id_dict_pos = {l:np.array([]) for l in pred_id}

    for k in id_dict_pos:
        for q in quad:
            id_dict_pos[k] = np.append(len(q[q == k]), id_dict_pos[k])
        #id_dict_pos[k] = id_dict_pos[k]/q_area
        id_dict_pos[k] = quad_dict[np.where(id_dict_pos[k] == max(id_dict_pos[k]))[0][0]] #[::-1] index for quadrant
    #print(id_dict_pos)

    # ========================================= display ==================================
    vr_vn = [(pred_class[pred_id.index(i)], id_dict_pos[i]) for i in np.unique(segment_vrvn) if i != 0]
    r_vn = [(pred_class[pred_id.index(i)], id_dict_pos[i]) for i in np.unique(segment_rvn) if i != 0]
    vr_n = [(pred_class[pred_id.index(i)], id_dict_pos[i]) for i in np.unique(segment_vrn) if i != 0]
    r_n = [(pred_class[pred_id.index(i)], id_dict_pos[i]) for i in np.unique(segment_rn) if i != 0]
    vrf = [pred_class[pred_id.index(i)]for i in np.unique(segment_vrf) if i != 0]
    #print('vr_vn:', vr_vn, '\nr_vn:', r_vn, '\nvr_n:', vr_n, '\nr_n:', r_n)
    eval_dict = {"vrvn": [[x[0] for x in vr_vn]], "vrn": [[x[0] for x in vr_n]], "vrf": [vrf]} 

    text = []
    vr_n_l = len(vr_n) > 0
    if vr_n_l:
      text.append(' ')
      iz, fr, de = [' su izquierda '], ['l frente '], [' su derecha ']
      for p in vr_n:
        if p[1] == 'iz':                             
          iz.append(p[0] + ' ')
        elif p[1] == 'fr':                 
          fr.append(p[0] + ' ')
        elif p[1] == 'de':
            de.append(p[0] + ' ')
      if len(iz) > 1:
        text.append(' '.join(iz))
      if len(fr) > 1:
        text.append(' '.join(fr))
      if len(de) > 1:
        text.append(' '.join(de))
        
    if len(r_n) > 0:
        pass
    if len(text)==0:
      text =['  Sin objetos relevantes  ']

    # Text to speech automatic play
    text = ', a'.join(text)
    #print(text)
    tts = gTTS(text=text, lang='es') 
    tts.save('1.wav') 
    sound_file = '1.wav'
    return Audio(sound_file, autoplay=True), text, initial_out_img, eval_dict

  def MultOut_RealTime(self): #disp_pred=False
    # start streaming video from webcam
    video_stream()
    # label for video
    label_html = 'Capturing...'
    # initialze bounding box to empty
    bbox = ''
    count = 0
    #while True:
    js_reply = video_frame(label_html, bbox)
    if not js_reply:
      text ='  No se esta grabando  '
      #print(text)
      tts = gTTS(text=text, lang='es') 
      tts.save('1.wav') 
      sound_file = '1.wav'
      return Audio(sound_file, autoplay=True)
    else:
      # convert JS response to OpenCV Image
      frame = js_to_image(js_reply["img"])
      # ========================== hierarchy for obj detection ====================================
      segment_arr, hierarchy_arr, SegmentInfo = self.onVideo_d(frame)
      segment_arr, hierarchy_arr = segment_arr.T, hierarchy_arr.T
      pred_id = SegmentInfo['id']
      pred_class = SegmentInfo['class_label']
      segment_arr = segment_arr.numpy()

      # ============================ hierarchy for depth ====================================
      #startB = time.time()
      depth_array = self.onVideo_m(frame)
      #print('midas:', time.time() - startB)
      depth_array = depth_array.T

      pred_id = SegmentInfo['id']
      pred_class = SegmentInfo['class_label']

      segment_veryrel = segment_arr.copy()
      segment_veryrel[hierarchy_arr != 1] = 0 
      segment_vrvn, segment_vrn, segment_vrf = segment_veryrel.copy(), segment_veryrel.copy(), segment_veryrel.copy()
      
      # ============================ hierarchy for depth ====================================
      depth_thresh = np.unique(depth_array)
      mask_vrvn = (depth_array == depth_thresh[-1])
      mask_vrn = (depth_array == depth_thresh[-2])
      mask_vrf = (depth_array == depth_thresh[-3])
      segment_vrvn[~mask_vrvn] = 0
      segment_vrn[~mask_vrn] = 0
      segment_vrf[~mask_vrf] = 0

      # ------------------- hierarchy for depth, using object percentage ---------------------
      # For each object detect the number of pixels that correspond to each 
      # To the segment_relevant matrix get the layers for each object
      # Do a for loop for the class in each layer and count the pixels that each layer matches just very relevant classes
      unique_vn, counts_vn = np.unique(segment_vrvn, return_counts = True)  
      unique_n, counts_n = np.unique(segment_vrn, return_counts = True)  
      unique_f, counts_f = np.unique(segment_vrf, return_counts = True)

      # Predict the poistion for each object / stuff detected 
      h_mod = len(segment_arr) % 3
      w_mod = len(segment_arr[0]) % 3
      if h_mod != 0:
          segment_arr_cut = segment_arr[:-h_mod, :]
      if w_mod != 0:
          segment_arr_cut = segment_arr[:, :-w_mod]

      # get the amount the amount of pixels they correspond for each quadrant 
      # INVERTED TO THE IMAGES
      h = len(segment_arr_cut) // 3 

      # devided into grid of 3 x 1
      quad = [segment_arr_cut[:h, :], segment_arr_cut[h:2*h, :], segment_arr_cut[2*h:, :]] 

      quad_dict = {0: 'iz', 1: 'fr', 2: 'de'}

      # dict for near obj
      pred_depth_pos = {}

      # Determine the highest frequency for each class
      # iteration on the classes and in case they are in the unique values. Then get the unique counts
      for c in unique_n:
        if c != 0:
          vn, n, f = 0,0,0
          if c in unique_vn:
              vn = counts_vn[np.where(np.array(unique_vn) == c)][0]
          if c in unique_n:
              n = counts_n[np.where(np.array(unique_n) == c)][0]
          if c in unique_f:
              f = counts_f[np.where(np.array(unique_f) == c)][0]

          # select just the objects in the middle with more pixels than percent percent liminit
          if n / (vn + n + f) >= 0.40:
            #get the position
            pred_depth_pos[c] = quad_dict[np.argmax([len(q[q == c]) for q in quad])] 

        #change output text to prural
        def count_obj(pos_list):
          unique_w, counts_w = np.unique(pos_list, return_counts = True)
          list_out = unique_w.tolist().copy()
          for word, count in zip(unique_w, counts_w):
            if count > 1:
              list_out.remove(word)
              list_out.append(f"{count} {word[:-1]}s")
          return list_out

        text = []
        if len(pred_depth_pos) > 0:
          text.append(' ')
          iz, fr, de = [' su izquierda '], ['l frente '], [' su derecha ']
          for p in pred_depth_pos:
            pred_lab = pred_class[pred_id.index(p)]
            pred_pos = pred_depth_pos[p]
            if pred_pos == 'iz':                             
              iz.append(pred_lab + ' ')
            elif pred_pos == 'fr':                 
              fr.append(pred_lab + ' ')
            elif pred_pos == 'de':
                de.append(pred_lab + ' ')

          iz, fr, de = count_obj(iz), count_obj(fr), count_obj(de)

          if len(iz) > 1:
            text.append(' '.join(iz))
          if len(fr) > 1:
            text.append(' '.join(fr))
          if len(de) > 1:
            text.append(' '.join(de))

      if len(text)==0:
        text =' Sin objetos relevantes '
        tts = gTTS(text=text, lang='es') 
        tts.save('1.wav') 
        sound_file = '1.wav'
        return Audio(sound_file, autoplay=False)
      
      else:
        text = ', a'.join(text)
        tts = gTTS(text=text, lang='es') 
        tts.save('1.wav') 
        sound_file = '1.wav'
        return Audio(sound_file, autoplay=True)