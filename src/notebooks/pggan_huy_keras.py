""" reference Progeressive growing gan 2017: https://arxiv.org/abs/1710.10196 """



from google.colab import drive
drive.mount('/content/drive/')

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Reshape
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend

from numpy import asarray
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from math import sqrt
from matplotlib import pyplot

from skimage import io
from PIL import Image
from skimage.transform import resize
import os 
import glob 
import numpy as np
import cv2
from numpy.random import randint,randn

# use for fade newblock
class FadeBlock(Add):


	# init with default value
  def __init__(self, alpha=0.0, **kwargs):
    super(FadeBlock, self).__init__(**kwargs)
    self.alpha = backend.variable(alpha, name='ws_alpha')
 
	# output a weighted sum of inputs
  def _merge_function(self, inputs):
    assert (len(inputs) == 2)
    # ((1-a) * input1) + (a * input2)
    output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
    return output

# minibatch stev
class Minibatchstev(Layer):
  # init this layer
  def __init__(self,**kwargs):
    super(Minibatchstev,self).__init__(**kwargs)

  def call(self, inputs):
    # calculate the mean value cross the batch 
    mean_3d = backend.mean(inputs, axis = 0,keepdims = True)
    # calculate squared different (variance) between pixel value (inputs and mean)
    squared_diff = backend.square(inputs - mean)
    # calculate mean of variance
    mean_squared_diff = backend.mean(squared_diff,axis = 0,keepdims = True)
    # add a small value to avoid a blow-up when we calculate stdev (ensure the mean is not zero)
    mean_squared_diff += 1e-8
    # sqrt of the variance
    stdev = backend.sqrt(mean_squared_diff)
    # calculate the mean stdev across each pixel
    mean_pix = backend.mean(stdev,keepdims = True)
    shape = backend.shape(inputs)
    output = backend.tile(mean_pix,(shape[0], shape[1], shape[2], 1))
    # concat output with mean_
    combined = backend.concatenate([inputs, output], axis=-1)
    return combined
  
  def caculate_output_shape(self,input_shape):
    input_shape = list(input_shape)
    input_shape[-1]+=1
    return tuple(input_shape)

# pixelNormalization

class PixelNormalization(Layer):
  #initalize the layer
  def __init__(self,**kwargs):
    super(PixelNormalization,self).__init__(**kwargs)

  # calculate pixelnormal
  def call(self,inputs):
    # calculate square pixel values
    values = inputs**2
    # calculate the mean pixel values
    mean_values = backend.mean(values, axis = -1 ,keepdims = True)
    # ensure the mean is not zero
    mean_values += 1.0e-8
    # calculate the sqrt of the mean squared value (L2 norm)
    l2 = backend.sqrt(mean_values)
		# normalize values by the l2 norm
    normalized = inputs / l2
    return normalized

  # define the putput shape of the layer
  def compute_output_shape(self,input_shape):
    return input_shape

#calculate wloss
def wassertein_loss(ytrue,ypred):
  return backend.mean(ytrue*ypred)

# grow D block
def add_D_block(old_model,n_input_layer = 3):
  """ 
  this function return 2 version:
  1.model without newblock
  2.model with newblock (use fadedblock)
  """
  #init weight
  init = RandomNormal(stddev = 0.02)
  # store shape of model
  shape_model = list(old_model.input.shape)
  # new size of input (double)
  input_shape = (shape_model[-2]*2,shape_model[-2]*2,shape_model[-1])
  in_image = Input(shape = input_shape)

  # from RGB block

  d = Conv2D(128,(1,1),padding = 'same',kernel_initializer= init,kernel_constraint = max_norm(1))(in_image)
  d = LeakyReLU(alpha = 0.2)(d)
  
  # new block
  d = Conv2D(128,(3,3),padding = 'same',kernel_initializer = init,kernel_constraint=max_norm(1.0))(d)
  d = LeakyReLU(alpha = 0.2)(d)
  d = Conv2D(128,(3,3),padding = 'same',kernel_initializer = init,kernel_constraint=max_norm(1.0))(d)
  d = LeakyReLU(alpha = 0.2)(d)
  d = AveragePooling2D()(d)

  new_block = d

  # skip input layer,conv(1,1) and activation so start with n_input_layer = 3
  for i in range(n_input_layer,len(old_model.layers)):
    d = old_model.layers[i](d)
  # model without newblock
  model1 = Model(in_image,d)
  # compile mode1
  model1.compile(loss = wassertein_loss, optimizer = Adam(learning_rate = 0.001,beta_1=0,beta_2=0.99,epsilon=10e-8))
    
  # downsample the new larger image
  downsample = AveragePooling2D()(in_image)
  # connect old input to downsampled new input

  # downsample througt the 1x1 conv
  old_block = old_model.layers[1](downsample)
  # thought the activation
  old_block = old_model.layers[2](old_block)

  # fade use FadeBlock
  d = FadeBlock()([old_block,new_block])

  # skip input layer,conv(1,1) and activation so start with n_input_layer = 3
  for i in range(n_input_layer,len(old_model.layers)):
    d = old_model.layers[i](d)
  # model without newblock
  model2 = Model(in_image,d)
  # compile mode2
  model2.compile(loss = wassertein_loss, optimizer = Adam(learning_rate = 0.001,beta_1=0,beta_2=0.99,epsilon=10e-8))
  return [model1,model2]

def discriminator(blocks,input_shape = (4,4,3)):
  init = RandomNormal(stddev=0.02)
  model_list = list()
  in_image = Input(shape = input_shape)

  #conv 1x1
  d = Conv2D(128,(1,1),padding='same',kernel_initializer='he_normal')(in_image)
  d = LeakyReLU(alpha=0.2)(d)
  #conv 3x3
  d = Minibatchstev()(d)
  d = Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal')(d)
  d = LeakyReLU(alpha=0.2)(d)
  #conv 4x4
  d = Conv2D(128,(4,4),padding ='same',kernel_initializer='he_normal')(d)
  d = LeakyReLU(alpha=0.2)(d)

  #dense output
  d = Flatten()(d)
  out_class = Dense(1)(d)

  # define mode
  model = Model(in_image,out_class)
  # compile mode
  model.compile(loss = wassertein_loss,optimizer = Adam(learning_rate=0.001,beta_1=0,beta_2=0.99,epsilon= 10e-8))
  # store model
  model_list.append([model,model])

  for i in range(1,blocks):
    models = add_D_block(model_list[i-1][0])
    model_list.append(models)
  return model_list

# grow G
def add_G_block(old_model):
  # init weight
  init = RandomNormal(stddev = 0.02)
  # upsample the output of the last block
  upsampling = UpSampling2D()(old_model.layers[-2].output)

  # define newblock
  g = Conv2D(128,(3,3),padding='same',kernel_regularizer=init,kernel_constraint=max_norm(1))(upsampling)
  g = PixelNormalization()(g)
  g = LeakyReLU(alpha=0.2)(g)
  g = Conv2D(128,(3,3),padding='same',kernel_initializer=init,kernel_constraint=max_norm(1))(g)
  g = PixelNormalization()(g)
  g = LeakyReLU(alpha=0.2)(g)
  # new ouput 
  out_image = Conv2D(3,(1,1),padding='same',kernel_initializer=init,kernel_constraint=max_norm(1))(g)
  #old version
  model1=Model(old_model.input,out_image)

  # old output 
  out_old = old_model.layers[-1]
  out_image2 = out_old(upsampling)

  # old output 
  #out_image2 = Conv2D(3,(1,1),padding='same',kernel_initializer=init,kernel_constraint=max_norm(1))(upsampling)
  merged = FadeBlock()([out_image2,out_image])
  # new version model
  model2 = Model(old_model.input,merged) 
  return [model1,model2]

def generator(latent_dim,blocks,in_dim = 4):
  """
  in this function G isnt compile
  G is trained via the D using W loss
  """
  # init weigjt
  init = RandomNormal(stddev=0.2)
  # make list to store
  model_list = list ()
  #input
  input = Input(shape=(latent_dim,))

  g = Dense(128 * in_dim * in_dim,kernel_initializer=init,kernel_constraint=max_norm(1))(input)
  g = Reshape((in_dim,in_dim,128))(g)
  # conv 4x4
  g = Conv2D(128,(3,3),padding='same',kernel_initializer=init,kernel_constraint=max_norm(1.0))(g)
  g = PixelNormalization()(g)
  g = LeakyReLU(alpha=0.2)(g)
  # conv 3x3
  g = Conv2D(128,(3,3),padding='same',kernel_initializer=init,kernel_constraint=max_norm(1.0))(g)
  g = PixelNormalization()(g)
  g = LeakyReLU(alpha=0.2)(g)
  # conv 1x1 ,output block (from RGB)
  out_img = Conv2D(3,(1,1),padding='same',kernel_initializer=init,kernel_constraint=max_norm(1))(g)
  # define model
  model = Model(input,out_img)
  # store
  model_list.append([model,model])
  for i in range(1,blocks):
    old_model = model_list[i-1][0]
    #creat new model for new resolution
    models = add_G_block(old_model)
    #store
    model_list.append(models)
  return model_list

def Composite(D_model,G_model):
  model_list = list()
  for i in range(len(D_model)):
    g_model,d_model = G_model[i],D_model[i]
    # non fade in model
    d_model[0].trainable = False
    model1 = Sequential()
    model1.add(g_model[0])
    model1.add(d_model[0])
    model1.compile(loss = wassertein_loss,optimizer = Adam(learning_rate=0.001,beta_1=0,beta_2=0.99,epsilon=1e-07))
    # fade in model
    d_model[1].trainable = False
    model2 = Sequential()
    model2.add(g_model[0])
    model2.add(d_model[0])
    model2.compile(loss = wassertein_loss,optimizer = Adam(learning_rate=0.001,beta_1=0,beta_2=0.99,epsilon=1e-07))
    #store
    model_list.append([model1,model2])
  return model_list

def load_data_image(img_path):
  X = []
  files = glob.glob(os.path.join(img_path, '*.jpg'))
  for file in files:
    img = cv2.imread(file)
    img = cv2.resize(img,(128,128))
    X.append(img)
  X = [np.random.random((128,128,3)) for i in range(len(X))]
  X = np.concatenate([arr[np.newaxis] for arr in X])
  X = X.astype('float32')
  # scale [-1 1]
  X = (X - 127.5)/127.5
  return X

def scale_dataset(dataset, new_shape):
 # scale image to preferred size
	images_list = list()
	for image in dataset:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

def select_real_sample(dataset,n_sample):
  # choose random sample
  indx = randint(0,dataset.shape[0],n_sample)
  x = dataset[indx]
  # generate label
  y = ones((n_sample,1))
  return x,y

def generate_x_input(latent_dim, n_sample):
  # random input 
  x_input = randn(latent_dim*n_sample)
  # reshape into a batch of inputs for network
  x_input = x_input.reshape(n_sample,latent_dim)
  return x_input

def generate_fake_img(generator,latent_dim,n_samples):
  # generate a noise
  x_input = generate_x_input(latent_dim,n_sample)
  # generate
  x  = generator.predict(x_input)
  # lable for fake img
  y = -ones((n_samples,1))
  return x,y

def update_alpha_fadeblock(models, step,n_steps):
  alpha = step/float(n_steps-1)
  #update alpha
  for model in models:
    for layer in model.layers:
      if(isinstance(layer,FadeBlock)):
        backend.set_value(layer.alpha,alpha)

def summarize_performance(status, g_model, latent_dim, n_samples=25):
	# devise name
	gen_shape = g_model.output_shape
	name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
	# generate images
	X, _ = generate_fake_img(g_model, latent_dim, n_samples)
	# normalize pixel values to the range [0,1]
	X = (X - X.min()) / (X.max() - X.min())
	# plot real images
	square = int(sqrt(n_samples))
	for i in range(n_samples):
		pyplot.subplot(square, square, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X[i])
	# save plot to file
	filename1 = 'plot_%s.png' % (name)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%s.h5' % (name)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

def train_epochs(g_model,d_model,gan_model,dataset,n_epochs,n_batchs,fadein = False):
  # the number of batchs per training epoch
  batchs_per_epoch = int (dataset.shape[0]/n_batchs)
  # the number of training iterations
  n_steps = batchs_per_epoch * n_epochs
  for i in range(n_steps):
    if fadein:
      # update alpha for all fadeblock layer when fading in
      update_alpha_fadeblock([g_model,d_model,gan_model],i,n_steps)
    # creat real and fake data
    X_real, y_real = select_real_sample(dataset,int(n_batchs/2))
    X_fake, y_fake = generate_fake_img(g_model,latent_dim,int(n_batchs/2))
    # update discriminator
    d_loss1 = d_model.train_on_batch(X_real,y_real)
    d_loss2 = d_model.train_on_batch(X_fake,y_fake)
    # update the generator via the discriminator' error
    z_input = generate_latent_points(latent_dim, n_batch)
    y_real2 = ones((n_batch, 1))
    g_loss = gan_model.train_on_batch(z_input, y_real2)
    print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))

def train(g_models,d_models,gan_models,dataset,lantent_dim,e_norm, e_fadein , n_batch):
  # fit the first model in list models
  g_model, d_model, gan_model = g_models[0][0], d_models[0][0], gan_models[0][0]
  # get output shape
  gen_shape = g_model.output_shape
  scaled_data = scale_dataset(dataset,gen_shape[1:])
  print('Scaled Data', scaled_data.shape)
  train_epochs(g_model,d_model,gan_model,scaled_data,e_norm[0],n_batchs[0])
  summarize_performance('tuned', g_model, latent_dim)
  for i in range(1,len(g_models)):
    [g_model,g_fade] = g_models[i]
    [d_model, d_fade] = d_models[i]
    [gan_model, gan_fade] = gan_models[i]
    # scale dataset to appropriate size
    gen_shape = g_model.output_shape
    scaled_data = scale_dataset(dataset, gen_shape[1:])
    print('Scaled Data', scaled_data.shape)
    # train fade-in models for next level of growth
    train_epochs(g_fade,d_fade,gan_fade,scaled_data,e_fade[i],n_batch[i],True)
    summarize_performance('faded', g_fade, latent_dim)
    # train normal or straight-through models
    train_epochs(g_model, d_model, gan_model, scaled_data, e_norm[i], n_batch[i])
    summarize_performance('tuned', g_model, latent_dim)

img_path = '/content/drive/My Drive/data/data/augmented data/yes'
n_blocks = 6
latent_dim = 100
d_models = discriminator(n_blocks)
g_models = generator(latent_dim,n_blocks)
gan_models = Composite(d_models,g_models)
dataset = load_data_image(img_path)
n_batch = [16, 16, 16, 8, 4, 4]
n_epochs = [5, 8, 8, 10, 10, 10]
train(g_models,d_models,gan_models,dataset,latent_dim,n_epochs,n_batch)

#model = tf.keras.models.load_model('model.h5')
#latent_dim=100
#n_images = 10
#input = generate_x_input(latent_dim,n_images)
#X = model.predict(input)



