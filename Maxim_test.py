from Maxim_model import *
if __name__ =="__main__":
    _MODEL_CONFIGS = {
        'variant': '',
        'dropout_rate': 0.0,
        'num_outputs': 3,
        'use_bias': True,
        'num_supervision_scales': 3,
    }
    _MODEL_VARIANT_DICT = {
    'Denoising': 'S-3',
    'Deblurring': 'S-3',
    'Deraining': 'S-2',
    'Dehazing': 'S-2',
    'Enhancement': 'S-2',
}
    for task, var in _MODEL_VARIANT_DICT.items():

      model_configs = ml_collections.ConfigDict(_MODEL_CONFIGS)
      model_configs.variant = _MODEL_VARIANT_DICT[task]
      model = Model(**model_configs,in_channel=3)
      test_sub_net =model #MAXIM(features=32,in_channel=3,in_channel_y=3)
      out = test_sub_net(torch.randint(0,100,[7,3,256,256],dtype=torch.float),train=True)
      print(f"Task {task}, Model-{var}  TEST DONE!" )