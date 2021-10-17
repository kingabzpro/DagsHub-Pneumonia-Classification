from src.const.general_const import IMG_SIZE, CLASS_NAME_PATH, CLASS_MODE, BATCH_SIZE, PROD_MODEL_PATH
from .const.evaluating_const import PROCESSED_VAL_PATH, DH_LOG_MET_PATH
from src.utiles.functions import load_dataset
from keras.models import Model
import tensorflow as tf
import scipy as sp

BATCH_SIZE = 16

def get_cam(chexnet, path=False):
    if path:
        chexnet = tf.keras.models.load_model(chexnet)

    gap_weights = model.layers[-1].get_weights()[0]
    cam = Model(inputs=model.input, outputs=(model.layers[-3].output, model.layers[-1].output))

if __name__ == '__main__':

    cam = get_cam(input('path to saved model'), path=True)
    features, results = cam_model.predict(load_dataset(PROCESSED_VAL_PATH, BATCH_SIZE, IMG_SIZE, CLASS_MODE))

    for idx in range(BATCH_SIZE):
        xi_features = features[idx, :, :, :]
        height_roomout = IMAGE_DIM.shape[1] / xi_features.shape[0]
        width_roomout = IMAGE_DIM.shape[2] / xi_features.shape[1]

        cam_features = sp.ndimage.zoom(xi_features, (height_roomout, width_roomout, 1), order=2)
        pred = np.argmax(results[idx])
        cam_features = xi_features

        plt.figure(facecolor='white')
        cam_weights = gap_weights[:, pred]
        cam_output = np.dot(cam_features, cam_weights)

        buf = 'Predicted Class = ' + str(pred) + ', Probability = ' + str(results[idx][pred])
        plt.xlabel(buf)
        plt.imshow(np.squeeze(X_test[idx],-1), alpha=0.5)
        plt.imshow(cam_output, cmap='jet', alpha=0.5)
        plt.show()
