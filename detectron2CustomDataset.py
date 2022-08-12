# coding=utf-8, take from https://github.com/PhilippeBaumstimler/PRE_detectron2
import pycocotools
import numpy as np
import os
from path import Path
import cv2 as cv
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import glob


def create_dataset_dict(instance_file_list, image_file_list, instance_decode):
    """
    :param instance_file_list : liste des chemins vers les masques d'instances ordonnés
    :param image_file_list: liste des chemins vers les images ordonnées
    :param instance_decode: fonction qui pour une instance donné, renvoie l'id de l'objet associé
    :return dataset_dicts : dictionnaire regroupant toutes les informations sur
                            toutes les instances de chaque masque d'instances
    :return selon le format utilisable par detectron2
    """
    dataset_dicts = []
    print("Creating custom dataset ...")
    for instance_file, image_file in tqdm(zip(instance_file_list, image_file_list)):
        instance_dict = {}
        instance_img = Image.open(instance_file)
        instance_img = np.array(instance_img)
        height, width = instance_img.shape
        image_name = os.path.basename(image_file)

        instance_dict["file_name"] = image_file
        instance_dict["image_id"] = image_name
        instance_dict["height"] = height
        instance_dict["width"] = width

        objects = []
        mask = instance_img.copy()
        for label in np.unique(instance_img):
            object_dict = {}
            target = np.zeros(instance_img.shape)
            roi = np.zeros(instance_img.shape)
            trainId = instance_decode(label)
            if trainId != 255 and trainId != 0:
                target[:, :][mask == label] = 255
                roi[:, :][mask == label] = 1
                target = target.astype(np.uint8)
                roi = roi.astype(np.uint8)
                contours, _ = cv.findContours(target, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                # if len(contours)>1:
                xmin = width
                ymin = height
                xmax = 0
                ymax = 0
                for cont in contours:
                    x, y, w, h = cv.boundingRect(cont)
                    if x < xmin:
                        xmin = x
                    if y < ymin:
                        ymin = y
                    if x + w > xmax:
                        xmax = x + w
                    if y + h > ymax:
                        ymax = y + h
                from detectron2.structures import BoxMode
                object_dict["bbox"] = [xmin, ymin, xmax, ymax]
                object_dict["bbox_mode"] = BoxMode.XYXY_ABS
                object_dict["segmentation"] = pycocotools.mask.encode(np.asarray(roi, order="F"))
                object_dict["category_id"] = trainId - 1
                objects.append(object_dict)
                instance_dict["annotations"] = objects
        dataset_dicts.append(instance_dict)
    return dataset_dicts


####################################### KITTI CUSTOM DATASET #######################################
## Création de la dataset personnalisé KITTI_seg selon le format entrainable par detectron2, 11 classes d'instances


def kitti_seg_instance(train, decoder, dir=None):
    """
    :param train: True si dataset d'entrainement, False si validation
    :param decoder: decoder
    :param dir: chemin d'accès vers la dataset KITTI_seg, par défaut None et se réfère à la variable
                d'environnement KITTI_SEG_DATASET, initialisé avant la boucle d'entrainement dans le script
                detectron2Train2.py
    :return dataset_dicts : dictionnaire regroupant toutes les informations
                            sur toutes les instances de chaque masque d'instances
    """
    if dir == None:
        dataset_dir = os.environ["KITTI_SEG_DATASET"]
    else:
        dataset_dir = dir
    if train:
        # Chemin d'accès vers les fichiers d'images
        training_dir = os.path.join(dataset_dir, "training")
        train_instance_dir = os.path.join(training_dir, "instance")
        train_image_dir = os.path.join(training_dir, "image_2")

        # Instanciation d'une liste de chemin vers vers les fichiers d'instances png
        train_instance_file_list = [os.path.join(train_instance_dir, f) for f in os.listdir(train_instance_dir) if
                                    os.path.isfile(os.path.join(train_instance_dir, f))]
        train_instance_file_list.sort()
        train_image_file_list = [os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) if
                                 os.path.isfile(os.path.join(train_image_dir, f))]
        train_image_file_list.sort()
        return create_dataset_dict(train_instance_file_list, train_image_file_list, decoder)
    else:
        validation_dir = os.path.join(dataset_dir, "validation")
        val_instance_dir = os.path.join(validation_dir, "instance")
        val_image_dir = os.path.join(validation_dir, "image_2")

        # Instanciation d'une liste de chemin vers vers les fichiers d'instances png
        val_instance_file_list = [os.path.join(val_instance_dir, f) for f in os.listdir(val_instance_dir) if
                                  os.path.isfile(os.path.join(val_instance_dir, f))]
        val_instance_file_list.sort()
        val_image_file_list = [os.path.join(val_image_dir, f) for f in os.listdir(val_image_dir) if
                               os.path.isfile(os.path.join(val_image_dir, f))]
        val_image_file_list.sort()
        return create_dataset_dict(val_instance_file_list, val_image_file_list, decoder)


def kitti_decode(instance_id):
    """
    :param instance_id : id de l'instance à décoder
    :return trainId : valeur de l'id d'entrainement après décodage
    """
    from cityscapesscripts.helpers.labels import id2label
    return id2label[instance_id // 256].trainId


def create_kitti_dataset():
    from detectron2.data import DatasetCatalog, MetadataCatalog
    DatasetCatalog.register("kitti_seg_instance_train", lambda: kitti_seg_instance(True, kitti_decode))
    DatasetCatalog.register("kitti_seg_instance_val", lambda: kitti_seg_instance(False, kitti_decode))
    MetadataCatalog.get("kitti_seg_instance_train").thing_classes = ["dynamic", "person", "rider", "car", "truck",
                                                                     "bus", "caravan", "trailer", "train", "motorcycle",
                                                                     "bicycle"]
    MetadataCatalog.get("kitti_seg_instance_val").thing_classes = ["dynamic", "person", "rider", "car", "truck",
                                                                   "bus", "caravan", "trailer", "train", "motorcycle",
                                                                   "bicycle"]
    MetadataCatalog.get("kitti_seg_instance_val").evaluator_type = "kitti_instance"
    return


####################################### KITTI CUSTOM DATASET - Version 8 classes #######################################
## Création de la dataset personnalisé KITTI_seg selon le format entrainable par detectron2, 8 classes d'instances

def kitti_decode8(instance_id):
    from cityscapesscripts.helpers.labels import id2label
    trainId = id2label[instance_id // 256].trainId
    if trainId in [0, 1, 7, 8, 255]:
        return 255
    else:
        if trainId < 7:
            return trainId - 1
        else:
            return trainId - 3


def create_kitti_dataset8():
    from detectron2.data import DatasetCatalog, MetadataCatalog
    DatasetCatalog.register("kitti_seg_instance_train8", lambda: kitti_seg_instance(True, kitti_decode8))
    DatasetCatalog.register("kitti_seg_instance_val8", lambda: kitti_seg_instance(False, kitti_decode8))
    MetadataCatalog.get("kitti_seg_instance_train8").thing_classes = ["person", "rider", "car", "truck",
                                                                      "bus", "train", "motorcycle", "bicycle"]
    MetadataCatalog.get("kitti_seg_instance_val8").thing_classes = ["person", "rider", "car", "truck",
                                                                    "bus", "train", "motorcycle", "bicycle"]
    MetadataCatalog.get("kitti_seg_instance_val8").evaluator_type = "kitti_instance"
    return


####################################### CITYSCAPES PM CUSTOM DATASET #######################################


def cityscapes_pm_seg_instance(train, decoder, dir=None):
    """ Prend en entrée le chemin d'accès vers la dataset KITTI_seg
        Renvoi la liste de dictionnaires de la dataset selon le format définit pour detectron2
    """
    if dir == None:
        dataset_dir = os.environ["CITYSCAPES_DATASET"]
    else:
        dataset_dir = dir
    # Chemin d'accès vers les fichiers d'images
    image_dir = os.path.join(dataset_dir, "leftImg8bit")
    gt_dir = os.path.join(dataset_dir, "gtFine")

    if train:
        train_instances = os.path.join(gt_dir, "train", "*", "*_gtFine_instanceIds.png")
        train_images = os.path.join(image_dir, "train", "*", "*_leftImg8bit.png")

        # Instanciation d'une liste de chemin vers vers les fichiers d'instances png
        train_instance_file_list = glob.glob(train_instances)
        train_instance_file_list.sort()
        train_image_file_list = glob.glob(train_images)
        train_image_file_list.sort()
        train_image_file_list = glob.glob(train_images)
        train_image_file_list.sort()
        return create_dataset_dict(train_instance_file_list, train_image_file_list, decoder)
    else:
        val_instances = os.path.join(gt_dir, "val", "*", "*_gtFine_instanceIds.png")
        val_images = os.path.join(image_dir, "val", "*", "*_leftImg8bit.png")

        # Instanciation d'une liste de chemin vers vers les fichiers d'instances png
        val_instance_file_list = glob.glob(val_instances)
        val_instance_file_list.sort()
        val_image_file_list = glob.glob(val_images)
        val_image_file_list.sort()

        return create_dataset_dict(val_instance_file_list, val_image_file_list, decoder)


def cityscapes_pm_decode(instance_id):
    """
    :param instance_id : id de l'instance à décoder
    :return trainId : valeur de l'id d'entrainement après décodage
    """
    from cityscapesscripts.helpers.labels import id2label
    if instance_id in [0, 255]:
        return instance_id
    if instance_id // 1000 == 0:
        return id2label[instance_id].trainId
    else:
        return id2label[instance_id // 1000].trainId


def create_cityscapes_pm_dataset():
    from detectron2.data import DatasetCatalog, MetadataCatalog
    DatasetCatalog.register("cityscapes_pm_instance_train", lambda: kitti_seg_instance(True, cityscapes_pm_decode))
    DatasetCatalog.register("cityscapes_pm_instance_val", lambda: kitti_seg_instance(False, cityscapes_pm_decode))
    MetadataCatalog.get("cityscapes_pm_instance_train").thing_classes = ["dynamic", "person", "rider", "car", "truck",
                                                                         "bus", "caravan", "trailer", "train",
                                                                         "motorcycle", "bicycle"]
    MetadataCatalog.get("cityscapes_pm_instance_val").thing_classes = ["dynamic", "person", "rider", "car", "truck",
                                                                       "bus", "caravan", "trailer", "train",
                                                                       "motorcycle", "bicycle"]
    MetadataCatalog.get("cityscapes_pm_instance_val").evaluator_type = "cityscapes_pm_instance"
    return


####################################### Test visualisation

if __name__ == "__main__":
    """ Permet de vérifier la bonne implémentation de la custom dataset
    """
    import random
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog

    create_kitti_dataset()
    kitti_seg_instance_metadata = MetadataCatalog.get("kitti_seg_instance_train")

    dataset_path = Path("~/StageEnsta/kitti/data_semantics")
    dataset_dicts = kitti_seg_instance(True, kitti_decode, dir=dataset_path)

    for d in random.sample(dataset_dicts, 3):
        print(d["file_name"])
        img = cv.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_seg_instance_metadata, scale=0.8)
        out = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(20, 20))
        plt.imshow(out.get_image())
        plt.show()
