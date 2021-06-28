import os

NOTEBOOK=False


HEALTHY = "NORMAL"
SICK = "PNEUMONIA"

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
LABEL_MODE = 'categorical'

CLASS_NAME_PATH = os.path.join(os.getcwd(),"task-3-modeling/src/const/class_names.txt")
