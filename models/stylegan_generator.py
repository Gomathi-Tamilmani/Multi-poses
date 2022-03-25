
import os
import numpy as np

import torch

from . import model_settings
from .stylegan_generator_model import StyleGANGeneratorModel
from .base_generator import BaseGenerator

__all__ = ['StyleGANGenerator']



class StyleGANGenerator(BaseGenerator):
  

  def __init__(self, model_name, logger=None):
    self.truncation_psi = model_settings.STYLEGAN_TRUNCATION_PSI
    self.truncation_layers = model_settings.STYLEGAN_TRUNCATION_LAYERS
    self.randomize_noise = model_settings.STYLEGAN_RANDOMIZE_NOISE
    self.model_specific_vars = ['truncation.truncation']
    super().__init__(model_name, logger)
    self.num_layers = (int(np.log2(self.resolution)) - 1) * 2
    assert self.gan_type == 'stylegan'

  def build(self):
    self.check_attr('w_space_dim')
    self.check_attr('fused_scale')
    self.model = StyleGANGeneratorModel(
        resolution=self.resolution,
        w_space_dim=self.w_space_dim,
        fused_scale=self.fused_scale,
        output_channels=self.output_channels,
        truncation_psi=self.truncation_psi,
        truncation_layers=self.truncation_layers,
        randomize_noise=self.randomize_noise)

  def load(self):
    self.logger.info(f'Loading pytorch model from `{self.model_path}`.')
    state_dict = torch.load(self.model_path)
    for var_name in self.model_specific_vars:
      state_dict[var_name] = self.model.state_dict()[var_name]
    self.model.load_state_dict(state_dict)
    self.logger.info(f'Successfully loaded!')
    self.lod = self.model.synthesis.lod.to(self.cpu_device).tolist()
    self.logger.info(f'  `lod` of the loaded model is {self.lod}.')

  def convert_tf_model(self, test_num=3):
    import sys
    import pickle
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    sys.path.append(model_settings.BASE_DIR + '/stylegan_tf_official')

    self.logger.info(f'Loading tensorflow model from `{self.tf_model_path}`.')
    tf.compat.v1.InteractiveSession()
    with open(self.tf_model_path, 'rb') as f:
      tf_model = pickle.load(f)
    self.logger.info(f'Successfully loaded!')

    self.logger.info(f'Converting tensorflow model to pytorch version.')

    
    tf_vars = dict(tf_model.__getstate__()['variables'])
    tf_vars.update(
        dict(tf_model.components.mapping.__getstate__()['variables']))
    tf_vars.update(
        dict(tf_model.components.synthesis.__getstate__()['variables']))
    state_dict = self.model.state_dict()
    for pth_var_name, tf_var_name in self.model.pth_to_tf_var_mapping.items():
      if 'ToRGB_lod' in tf_var_name:
        lod = int(tf_var_name[len('ToRGB_lod')])
        lod_shift = 10 - int(np.log2(self.resolution))
        tf_var_name = tf_var_name.replace(f'{lod}', f'{lod - lod_shift}')
      if tf_var_name not in tf_vars:
        self.logger.debug(f'Variable `{tf_var_name}` does not exist in '
                          f'tensorflow model.')
        continue
      self.logger.debug(f'  Converting `{tf_var_name}` to `{pth_var_name}`.')
      var = torch.from_numpy(np.array(tf_vars[tf_var_name]))
      if 'weight' in pth_var_name:
        if 'dense' in pth_var_name:
          var = var.permute(1, 0)
        elif 'conv' in pth_var_name:
          var = var.permute(3, 2, 0, 1)
      state_dict[pth_var_name] = var
    self.logger.info(f'Successfully converted!')

    self.logger.info(f'Saving pytorch model to `{self.model_path}`.')
    for var_name in self.model_specific_vars:
      del state_dict[var_name]
    torch.save(state_dict, self.model_path)
    self.logger.info(f'Successfully saved!')

    self.load()

    # Official tensorflow model can only run on GPU.
    if test_num <= 0 or not tf.test.is_built_with_cuda():
      return
    self.logger.info(f'Testing conversion results.')
    self.model.eval().to(self.run_device)
    total_distance = 0.0
    for i in range(test_num):
      latent_code = self.easy_sample(1)
      tf_output = tf_model.run(latent_code,  # latents_in
                               None,  # labels_in
                               truncation_psi=self.truncation_psi,
                               truncation_cutoff=self.truncation_layers,
                               randomize_noise=self.randomize_noise)
      pth_output = self.synthesize(latent_code)['image']
      distance = np.average(np.abs(tf_output - pth_output))
      self.logger.debug(f'  Test {i:03d}: distance {distance:.6e}.')
      total_distance += distance
    self.logger.info(f'Average distance is {total_distance / test_num:.6e}.')

  def sample(self, num, latent_space_type='Z'):
    latent_space_type = latent_space_type.upper()
    if latent_space_type == 'W':
      latent_codes = np.random.randn(num, self.w_space_dim)
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    return latent_codes.astype(np.float32)

  def preprocess(self, latent_codes, latent_space_type='Z'):

    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    latent_space_type = latent_space_type.upper()
    if latent_space_type == 'Z':
      latent_codes = np.random.randn(num, self.latent_space_dim)
    elif latent_space_type == 'W':
      latent_codes = latent_codes.reshape(-1, self.w_space_dim)
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    return latent_codes.astype(np.float32)

  def easy_sample(self, num, latent_space_type='Z'):
    return self.preprocess(self.sample(num, latent_space_type),
                           latent_space_type)

  def synthesize(self,
                 latent_codes,
                 latent_space_type='Z',
                 generate_style=False,
                 generate_image=True):
    
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    results = {}

    latent_space_type = latent_space_type.upper()
    latent_codes_shape = latent_codes.shape
    # Generate from Z space.
    if latent_space_type == 'W':
      if not (len(latent_codes_shape) == 2 and
              latent_codes_shape[1] == self.w_space_dim):
        raise ValueError(f'Latent_codes should be with shape [batch_size, '
                         f'w_space_dim], where `batch_size` no larger than '
                         f'{self.batch_size}, and `w_space_dim` equal to '
                         f'{self.w_space_dim}!\n'
                         f'But {latent_codes_shape} received!')
      ws = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      ws = ws.to(self.run_device)
      wps = self.model.truncation(ws)
      results['w'] = latent_codes
      results['wp'] = self.get_value(wps)
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    if generate_style:
      for i in range(self.num_layers):
        style = self.model.synthesis.__getattr__(
            f'layer{i}').epilogue.style_mod.dense(wps[:, i, :])
        results[f'style{i:02d}'] = self.get_value(style)

    if generate_image:
      images = self.model.synthesis(wps)
      results['image'] = self.get_value(images)

    return results
