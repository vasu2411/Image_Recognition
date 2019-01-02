from PIL import Image
import numpy as np
from keras.models import load_model

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
          'frog', 'horse', 'ship', 'truck']

#load trained model		  
model = load_model('Trained_model.h5')

#user input for image path
input_path = input('Enter image file pathname: ')
input_image = Image.open(input_path)

#resize image of user to 32*32
input_image = input_image.resize((32, 32), resample=Image.LANCZOS)

image_array = np.array(input_image)
image_array = image_array.astype('float32')
image_array /= 255.0
image_array = image_array.reshape(1, 32, 32, 3)

#predicts class of image
answer = model.predict(image_array)
input_image.show()
print(labels[np.argmax(answer)])