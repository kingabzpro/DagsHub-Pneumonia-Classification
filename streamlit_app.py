import streamlit as st
import cv2 as cv
from PIL import Image
import numpy as np
import tensorflow as tf
from src.const.general_const import PROD_MODEL_PATH, IMG_SIZE, CLASS_NAME_PATH

# const
import os

BASE_PATH = os.getcwd()
DAGSHUB_IMAGE_PATH = os.path.join(BASE_PATH, "task-5-streamlit-app/images/dagshub.png")

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


# todo: Add side bar with option to choose from library or upload new image
# todo: Add SHORT_DESCRIPTION
# todo: Add message with prediction & color red/green
# todo: Change val set to be traced by git & change img path here

# todo: not streamlit - Add wights to the model, change basepath in const to general const

def markdown_format(font_size,content):
    st.markdown(f"<{font_size} style='text-align: center; color: black;'>{content}</{font_size}>",
                unsafe_allow_html=True)

def load_n_resize_image(image_path):
    pil_img = Image.open(image_path)
    return cv.resize(cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR), IMG_SIZE)


def load_image_pool():
    healthy = [load_n_resize_image(HEALTHY_IMAGE_ONE_PATH), load_n_resize_image(HEALTHY_IMAGE_TWO_PATH)]
    sick = [load_n_resize_image(SICK_IMAGE_ONE_PATH), load_n_resize_image(SICK_IMAGE_TWO_PATH)]
    return {'healthy': healthy, 'sick': sick}


def present_pool(col, col_name, img_list):
    name_list = []
    for row in range(len(img_list)):
        col.image(img_list[row], use_column_width=True, caption=col_name + f" {row + 1}")
        name_list.append(col_name + f" {row + 1}")
    return name_list


def display_prediction(pred):
    if pred == 'sick':
        st.warning('Unfortunately we have bad news, our model detects that you have Pneumonia')
    else:
        st.success("We have GREAT news! Based on our model, you don't have Pneumonia!")


@st.cache(suppress_st_warning=True)
def get_prediction(img):
    with open(CLASS_NAME_PATH, "r") as textfile:
        class_names = textfile.read().split(',')

    img_expand = np.expand_dims(img, 0)

    model = tf.keras.models.load_model(PROD_MODEL_PATH)
    predictions = model.predict(img_expand)
    display_prediction(class_names[np.rint(predictions[0][0]).astype(int)])


def predict_for_selectbox(selectbox, my_bar, latest_iteration):
    img_class = selectbox.split()[0]
    img_position = int(selectbox.split()[-1]) - 1
    img = dict_of_img_lists[img_class][img_position]
    my_bar.progress(50)

    latest_iteration.text('Processing image')
    get_prediction(img)
    my_bar.progress(100)


def predict_for_file_buffer(file_buffer, my_bar, latest_iteration):
    latest_iteration.text('Loading image')
    img = load_n_resize_image(file_buffer)
    markdown_format('h3', "Your chest X-ray")
    st.image(img, use_column_width=True)
    my_bar.progress(50)

    latest_iteration.text('Processing image')
    get_prediction(img)
    my_bar.progress(100)



if __name__ == '__main__':
    # Page configuration
    st.set_page_config(page_title=HEADER, page_icon="🤒",
                       initial_sidebar_state='collapsed')

    # Base Design
    st.image(image=DAGSHUB_IMAGE_PATH)
    markdown_format('h1', HEADER)
    markdown_format('h3', SUB_HEADER)
    markdown_format('p', SHORT_DESCRIPTION)
    latest_iteration = st.empty()
    my_bar = st.progress(0)

    # Show pool of images
    dict_of_img_lists = load_image_pool()
    with st.beta_expander("Image Pool"):
        markdown_format('h3', IMAGE_POOL_DESCRIPTION)
        col1, col2 = st.beta_columns(2)
        healthy_sidebar_list = present_pool(col1, "healthy", dict_of_img_lists['healthy'])
        sick_sidebar_list = present_pool(col2, "sick", dict_of_img_lists['sick'])

    # Sidebar
    selectbox = st.sidebar.selectbox("Choose an image for the model to predict?",
                                     [None] + healthy_sidebar_list + sick_sidebar_list)

    file_buffer = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"])

    # Predict for user selection
    if selectbox:
        predict_for_selectbox(selectbox, my_bar, latest_iteration)
        dict_of_img_lists = load_image_pool()

    if file_buffer:
        predict_for_file_buffer(file_buffer, my_bar, latest_iteration)

