
from .utils import TLinear
from deepspeed.utils import OnDevice
from deepspeed.inference.config import DeepSpeedInferenceConfig
from deepspeed.module_inject import replace_transformer_layer
from deepspeed.module_inject.replace_policy import replace_policies
from deepspeed.module_inject.utils import policy_to_ds_container
from .engine import DeepSpeedEngine

class DeepSpeedHybridEngine(DeepSpeedEngine):
    r"""DeepSpeed engine for training and inference."""
    def __init__(self, args, model, **kwargs):
        self._configure_with_arguments(args, mpu)
        
        if self._config.initialize_with_linear_transposed:
            self.convert_to_linear_transposed(model)

        super().__init__(args, model, **kwargs)
    
    def convert_to_linear_transposed(self, model):
        def _replace_linear_layer(r_module, prev_name=''):
            for name, child in r_module.named_children():
                if child.__class__ in [torch.nn.Linear]:
                    setattr(r_module, name, TLinear(child))
                else:
                    _replace_linear_layer(child, name)
            return r_module
        _replace_linear_layer(model)

    def new_inference_container(self, orig_layer, policy_cls, layer_id):
        policy = policy_cls(orig_layer, inference=True)
        _container = policy_to_ds_container(policy=policy,
                                            config=DeepSpeedInferenceConfig(),
                                            model_config=None,
                                            layer_id=layer_id,
                                            child=orig_layer)
        _container.set_dtype(self._config.fp16_enabled)
        _container.initialize_tensors()
        
        _container.create_module(set_empty_params=True)
        _container.set_params_wo_copy()
        return _container

    def populate_all_inference_policies(self):
        self.inference_policies = {}
        for plcy in replace_policies:
            _ = plcy(None)
            if isinstance(plcy._orig_layer_class, list):
                for orig_layer_class in plcy._orig_layer_class:
                    self.inference_policies.update({orig_layer_class: (new_inference_container, plcy)})
            elif plcy._orig_layer_class is not None:
                self.inference_policies.update({plcy._orig_layer_class: (new_inference_container, plcy)})

    def create_inference_containers(self, module, layer_id=0):
        for name, child in self.module.named_children():
            if child.__class__ in self.inference_policies:
                self._inference_containers.append(self.inference_policies[child.__class__][0](child,
                                                            policies[child.__class__][-1],
                                                            layer_id))
                self._inference_container_map[name] = len(self._inference_containers) - 1
                self._orig_fwds[name] = child.forward
                layer_id += 1
            else:
                self.create_inference_containers(child, layer_id=layer_id)

    def create_inference_module(self):
        self._inference_containers = []
        self._inference_container_map = {}
        self._orig_fwds = {}
        self.populate_all_inference_policies()
        self.create_inference_containers(self.module)
        
    def eval(self):
        for name, child in self.module.named_children():
            if name in self._inference_container_map:
                container_idx = self._inference_container_map[name]
                child.forward = self._inference_containers[container_idx].module.forward
        super().eval()
    
    def train(self):
        for name, child in self.module.named_children():
            if name in self._orig_fwds:
                child.forward = self._orig_fwds[name]
        super().train()

