import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import datetime 

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def display_morph(original, new):
  plt.figure(figsize=(15, 15))
  display_list = [original, new]
  title = ['Input Image', 'Morphologically Modified']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def create_mask(pred_mask):
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

def test_show_predictions(dataset=None, num=1):
  for image in dataset.take(num):
    pred_mask = model.predict(image)
    display([image[0], create_mask(pred_mask)])


def plot_model_history(model_history, save = True, show = False):
  
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if show:
      plt.show()
    if save:
      path = '../results/histories/'
      Path(path).mkdir(parents=True, exist_ok=True)
      plt.savefig(path + 'model_history_accuracy_'  + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M') +'.png')

    # summarize history for loss
    plt.clf()
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if show:
      plt.show()
    if save:
      path = '../results/histories/'
      Path(path).mkdir(parents=True, exist_ok=True)
      plt.savefig(path + 'model_history_loss_'  + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M') +'.png')
    plt.clf()
