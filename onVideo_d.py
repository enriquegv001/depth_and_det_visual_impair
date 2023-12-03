def onVideo_d(self, frame):
    predictions, segmentInfo = self.predictor(frame)["panoptic_seg"]
    metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
    
    segmentInfo_og = segmentInfo.copy() # save segmentInfo original

    #============================== Algorithm: hierarchy based on things =================================
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

    #==================================== display for image  ======================================

    # display normal results
    #viz = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
    #output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo_og)
    #cv2.imshow("Result", output.get_image()[:,:,::-1])
    #cv2_imshow(output.get_image()[:,:,::-1])
    return pred_arr, pred_hierarchy, Info_with_label#, output.get_image()[:,:,::-1][:,:,1]
