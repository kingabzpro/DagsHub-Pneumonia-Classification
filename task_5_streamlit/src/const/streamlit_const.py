import os

BASE_PATH = os.getcwd()
DAGSHUB_IMAGE_PATH = os.path.join(BASE_PATH, "task_5_streamlit/images/dagshub.png")

HEALTHY_IMAGE_ONE_PATH = os.path.join(BASE_PATH,
                                      "task-1-data-labeling/data/labeled-data/val/NORMAL/NORMAL2-IM-1427-0001.jpeg")
HEALTHY_IMAGE_TWO_PATH = os.path.join(BASE_PATH,
                                      "task-1-data-labeling/data/labeled-data/val/NORMAL/NORMAL2-IM-1430-0001.jpeg")
SICK_IMAGE_ONE_PATH = os.path.join(BASE_PATH,
                                   "task-1-data-labeling/data/labeled-data/val/PNEUMONIA/person1946_bacteria_4874.jpeg")
SICK_IMAGE_TWO_PATH = os.path.join(BASE_PATH,
                                   "task-1-data-labeling/data/labeled-data/val/PNEUMONIA/person1946_bacteria_4875.jpeg")

HEADER = "Pneumonia-Classification"
SUB_HEADER = "Classifying chest X-Ray images for Pneumonia"
SHORT_DESCRIPTION = """
                    This application is
                    """
IMAGE_POOL_DESCRIPTION = "You can choose one of the following images in the sidebar menu " \
                         "for the model to predict for Pneumonia"

SELECT_BOX_TEXT = "Choose an image for the model to predict?"
SUPPORTED_IMG_TYPE = ["png", "jpg", "jpeg"]
WARNING_MSG = "Unfortunately we have bad news, our model detects that you have Pneumonia"
SUCCESS_MSG =  "We have GREAT news! Based on our model, you don't have Pneumonia!"

BIG_FONT= "h1"
MID_FONT= "h3"
SMALL_FONT = "p"