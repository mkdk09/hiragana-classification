from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

jpg_name = './hiragana73/U304A/1971_1352040_0240'

model=load_model('./sample_hiragana_cnn_model.h5')

img_path = (jpg_name + '.png')
img = img_to_array(load_img(img_path, target_size=(28,28), grayscale=True))
img_nad = img_to_array(img)/255
img_nad = img_nad[None, ...]

pred = model.predict(img_nad, batch_size=1, verbose=0)
score = np.max(pred)
print('label:', np.argmax(pred))
print('score:',score)