

import os
import sys
import logging
import numpy as np

import torch

from . import model_settings

__all__ = ['BaseGenerator']


def get_temp_logger(logger_name='logger'):
  if not logger_name:
    raise ValueError(f'Input `logger_name` should not be empty!')

  logger = logging.getLogger(logger_name)
  if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

  return logger


class BaseGenerator(object):

  def __init__(self, model_name, logger=None):

    self.model_name = model_name
    for key, val in model_settings.MODEL_POOL[model_name].items():
      setattr(self, key, val)
    self.use_cuda = model_settings.USE_CUDA
    self.batch_size = model_settings.MAX_IMAGES_ON_DEVICE
    self.logger = logger or get_temp_logger(model_name + '_generator')
    self.model = None
    self.run_device = 'cuda' if self.use_cuda else 'cpu'
    self.cpu_device = 'cpu'

    # Check necessary settings.
    self.check_attr('gan_type')
    self.check_attr('latent_space_dim')
    self.check_attr('resolution')
    self.min_val = getattr(self, 'min_val', -1.0)
    self.max_val = getattr(self, 'max_val', 1.0)
    self.output_channels = getattr(self, 'output_channels', 3)
    self.channel_order = getattr(self, 'channel_order', 'RGB').upper()
    assert self.channel_order in ['RGB', 'BGR']

    # Build model and load pre-trained weights.
    self.build()
    if os.path.isfile(getattr(self, 'model_path', '')):
      self.load()
    elif os.path.isfile(getattr(self, 'tf_model_path', '')):
      self.convert_tf_model()
    else:
      self.logger.warning(f'No pre-trained model will be loaded!')

    # Change to inference mode and GPU mode if needed.
    assert self.model
    self.model.eval().to(self.run_device)

  def check_attr(self, attr_name):

    if not hasattr(self, attr_name):
      raise AttributeError(
          f'`{attr_name}` is missing for model `{self.model_name}`!')

  def build(self):

    raise NotImplementedError(f'Should be implemented in derived class!')

  def load(self):

    raise NotImplementedError(f'Should be implemented in derived class!')

  def convert_tf_model(self, test_num=10):

    raise NotImplementedError(f'Should be implemented in derived class!')

  def sample(self, num):

    raise NotImplementedError(f'Should be implemented in derived class!')

  def preprocess(self, latent_codes):
    raise NotImplementedError(f'Should be implemented in derived class!')

  def easy_sample(self, num):

    return self.preprocess(self.sample(num))

  def synthesize(self, latent_codes):
    raise NotImplementedError(f'Should be implemented in derived class!')

  def get_value(self, tensor):
    if isinstance(tensor, np.ndarray):
      return tensor
    if isinstance(tensor, torch.Tensor):
      return tensor.to(self.cpu_device).detach().numpy()
    raise ValueError(f'Unsupported input type `{type(tensor)}`!')

  def postprocess(self, images):
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')

    images_shape = images.shape
    if len(images_shape) != 4 or images_shape[1] not in [1, 3]:
      raise ValueError(f'Input should be with shape [batch_size, channel, '
                       f'height, width], where channel equals to 1 or 3. '
                       f'But {images_shape} is received!')
    images = (images - self.min_val) * 255 / (self.max_val - self.min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    if self.channel_order == 'BGR':
      images = images[:, :, :, ::-1]

    return images

  def easy_synthesize(self, latent_codes, **kwargs):
    
    outputs = self.synthesize(latent_codes, **kwargs)
    if 'image' in outputs:
      outputs['image'] = self.postprocess(outputs['image'])

    return outputs

  def get_batch_inputs(self, latent_codes):
    total_num = latent_codes.shape[0]
    for i in range(0, total_num, self.batch_size):
      yield latent_codes[i:i + self.batch_size]
