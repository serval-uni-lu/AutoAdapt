
from typing import Optional, Union
from opendelta.utils.signature import get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
import torch.nn as nn
import torch
from opendelta.delta_models.layers.activations import Activations
from opendelta import BaseDeltaConfig
import opendelta.utils.logging as logging
import numpy as np
from opendelta import global_setting
from dataclasses import dataclass, field

logger = logging.get_logger(__name__)


class InterFaceMixin:
    def __init__(self):
        self._axis_order = global_setting.axis_order
        self._reverse_axis_order = np.argsort(self._axis_order).tolist()

    def _transpose(self, tensor):
        if tensor.dim() == 3:
            return tensor.permute(*self._axis_order)
        else:
            return tensor



    def _reverse_transpose(self, tensor):
        if tensor.dim() == 3:
            return tensor.permute(*self._reverse_axis_order).contiguous()
        else:
            return tensor

    def _convert_data_type(self, tensor):
        self._data_type_record = tensor.dtype
        self._device_record = tensor.device
        return tensor.to(torch.float32).to(self._get_device())

    def _reverse_data_type(self, tensor):
        return tensor.to(self._data_type_record).to(self._device_record)





class AdapterLayer(nn.Module, InterFaceMixin):
    r"""A layer of adapter tuning module.
    """
    layer_count = 0

    @classmethod
    def count_layer(cls):
        cls.layer_count += 1

    @classmethod
    def get_layer_count(cls):
        return cls.layer_count

    def __init__(self, bottleneck_dim=[], 
                        non_linearity='gelu_new', 
                        dropout_rate =  0.0 , 
                        normalization = None , 
                        skip_connection = False ,  
                        device=None, backend="hf"):
        super().__init__()
        InterFaceMixin.__init__(self)
        self.bottleneck_dim = bottleneck_dim
        self.init_device = device
        self.instantiated = False
        self.non_linearity = non_linearity.lower()
        self.dropout_rate = dropout_rate 
        self.normalization = normalization
        self.skip_connection = skip_connection 
        self.backend = backend

        self.layer_id = AdapterLayer.get_layer_count()

        AdapterLayer.count_layer()


    



    def _get_device(self):
        if self.instantiated:
            return self.modulelist.down_proj.weight.device
        else:
            return self.init_device
        


      


    def instantiate(self, hiddens):


        self.hidden_dim = hiddens.shape[-1]
        self.hidden_dtype = hiddens.dtype
        self.modulelist = nn.Sequential()

        # Down Projection
        
        self.modulelist.add_module("down_proj", nn.Linear(  self.hidden_dim, 
                                                            self.bottleneck_dim[0], 
                                                            device=self.init_device, 
                                                            dtype=self.hidden_dtype
            )
        )


        self.modulelist.add_module("non_linear", Activations(self.non_linearity))

        if self.normalization : 
            self.modulelist.add_module("adapter_norm", nn.LayerNorm(self.bottleneck_dim[0]))



         # Optional Dropout after non-linearity
        if self.dropout_rate > 0.0:
            self.modulelist.add_module("adapter_dropout", nn.Dropout(self.dropout_rate))

    

        '''
        # Bottleneck Layers (if more than one dimension is provided)
        input_size = self.bottleneck_dim[0]
        for idx, layer_dim in enumerate(self.bottleneck_dim[1:]):
            # Mid Projections
            self.modulelist.add_module(
                f"mid_proj{idx}", 
                nn.Linear(
                    input_size, 
                    layer_dim, 
                    device=self.init_device, 
                    dtype=self.hidden_dtype
                )
            )
            self.modulelist.add_module(
                f"non_linear{idx}", 
                Activations(self.non_linearity)
            )

            # Optional Normalization after each mid projection
            if self.normalization is not None:
                if self.normalization.lower() == "layernorm":
                    self.modulelist.add_module(
                        f"norm_mid{idx}",
                        nn.LayerNorm(layer_dim, eps=1e-12)
                    )
                elif self.normalization.lower() == "batchnorm":
                    self.modulelist.add_module(
                        f"norm_mid{idx}",
                        nn.BatchNorm1d(layer_dim)
                    )
                else:
                    raise ValueError(f"Unsupported normalization type: {self.normalization}")

            # Optional Dropout after each mid non-linearity
            if self.dropout_rate > 0.0:
                self.modulelist.add_module(
                    f"dropout_mid{idx}",
                    nn.Dropout(self.dropout_rate)
                )

            

            input_size = layer_dim

        '''


        '''
        input_size = self.bottleneck_dim[0]
       
        for idx , layer_dim in enumerate (self.bottleneck_dim[1:]) : 
           
            # mid projections 
            self.modulelist.add_module("mid_proj"+ str(idx),nn.Linear(input_size, layer_dim, device=self.init_device, dtype=self.hidden_dtype ))
            # select non-linearity
            self.modulelist.add_module("non_linear"+ str(idx), Activations(self.non_linearity.lower()))

            input_size = layer_dim
    
        self.modulelist.add_module("up_proj", nn.Linear(input_size, self.hidden_dim,  device=self.init_device, dtype=self.hidden_dtype))
        '''

        # Up Projection
        self.modulelist.add_module(
            "up_proj",
            nn.Linear(
                self.bottleneck_dim[-1], 
                self.hidden_dim, 
                device=self.init_device, 
                dtype=self.hidden_dtype
            )
        )
        



        # TODO:
        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        #if self.normalization:
             #self.adapter_norm_after = nn.LayerNorm(self.hidden_dim)

        self.instantiated = True
        # initialize the weight, which is important for fast convergence and better performance.
        self.apply(self._init_weight)
        if self.backend == 'bmt':
            import bmtrain as bmt
            self.modulelist = bmt.BMTrainModelWrapper(self.modulelist)


                


    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm1d):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()


    def post_forward(self, output):
        r""" Get the hidden_states from the PLM's layer output, pass it into the adapter,
        then combined with the main hidden_states. Finally pass it into the subsequent layer.

        """
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError

        hiddens = self._transpose(hiddens)
        # if self.backend == 'hf':
        #     hiddens = self._convert_data_type(hiddens)
        # elif self.backend == 'bmt': # if bmt, left the convertion to bmt
        #     pass

        if not self.instantiated:
            # self.hidden_dim = hiddens.shape[-1]
            # logger.debug(f"Got hidden dim hidden_dim {self.hidden_dim}")
            self.instantiate(hiddens=hiddens)

        # from IPython import embed; embed(header="14135315")
 
        adapter_output = self.modulelist(hiddens)


        
        if self.skip_connection:
            modified_output = adapter_output + hiddens
        else:
            modified_output = adapter_output  # If skip_connection is False
    

        #modified_output = adapter_output + hiddens # TODO option: disable residual_connection

        ## changed : added a layer norm after residual connection 
        #if self.normalization :

            #modified_output = self.adapter_norm_after(modified_output)

        modified_output = self._reverse_transpose(modified_output)

        # if self.backend == 'hf':
        #     # print("!"*100)
        #     modified_output = self._reverse_data_type(modified_output)
        # elif self.backend == 'bmt': # if bmt, left the convertion to bmt
        #     print("!"*100)
        #     pass


        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output





class AdapterConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`~AdapterModel`

    """
    def __init__(
        self,
        bottleneck_dim: Optional[List[int]]=[24,32], # custom
        non_linearity: Optional[str]='gelu_new', # custom
        dropout_rate: Optional[float]=0.0 ,     # custom
        normalization: Optional[str]=None,      # custom
        skip_connection:Optional[bool]=False,   # custom
        **kwargs
    ):
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])



class AdapterModel(DeltaBase):
    r""" The implementation of Adapter(`Parameter-Efficient Transfer Learning for NLP <https://arxiv.org/abs/1902.00751>`_ ) .
    Add adapter to the designated ``modified_modules``. In sequential paradigm, The modules' output is then passed into the adapter's
    post_forward.

    .. note::
        We **assume** the output of the modified module is the hidden state or a tuple where hidden state is the
        first element. This is true for most PLMs. However, we admit that currently it's not rigorous, We will improve
        it in the next version. Currently, if you encount an error here for you backbone, you can modify the code to
        get the hidden state.

    class attributes:
        - default_modified_modules = ["attn", "ff"] According to the Adapter paper, we add adapter to the attention layer
          and feed forward layer.
        - delta_type = "adapter"

    Args:
        backbone_model (:obj:`transformers.PretrainedModels`): The backbone model to be modified.
        bottleneck_dim (:obj:`int`): The dimension of the adapter's bottleneck.
        non_linearity (:obj:`str`): The non linearity of the adapter.
        modified_modules (:obj:`List[str]`): modules to add adapter after them.
        unfrozen_modules (:obj:`List[str]`, *optional*, default to :obj:`None`): The modules that should be unfrozen together with the adapter parameters.
        common_structure (:obj:`bool`): whether using name-based addressing witha common structure mapping.
        backend (:obj:`str`): choose the backend of plm, 'hf' for huggingface transformers,'bmt' for bmtrain. 

    """
    config_class = AdapterConfig
    delta_type = "adapter"
    default_modified_modules = ["attn@.proj@", "ff@.w2@"]
    _supported_backends = ['hf', 'bmt']
    _need_pseudo_data = True
    def __init__(self,
                 backbone_model: nn.Module,
                 bottleneck_dim: Optional[List[int]]=[], # custom
                 non_linearity: Optional[str]='gelu_new', # custom
                 dropout_rate: Optional[float]=0.0 ,     # custom
                 normalization: Optional[str]=None,      # custom
                 skip_connection:Optional[bool]=False,   # custom
                 modified_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[bool] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 backend: Optional[str] = 'hf',
                 #device : Optional[object] = None,
                 ):
        DeltaBase.__init__(self,
                           backbone_model,
                           modified_modules=modified_modules,
                           exclude_modules=exclude_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           interactive_modify=interactive_modify,
                           backend=backend,
                           )
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.delta_modules = nn.ModuleList()

        self.add_all_delta_to_backbone(self.backbone_model,
                                   self.modified_modules,
                                   )
  
    
    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        adapterlayer = self.new_module_like(ref)
        self.insert_sequential_module(ref, delta_module=adapterlayer, delta_name="adapter")

    def new_module_like(self, module):
        module_device = get_device(module) #self.device #
        adapterlayer = AdapterLayer(bottleneck_dim=self.bottleneck_dim, 
                                    non_linearity=self.non_linearity, 
                                    dropout_rate=self.dropout_rate ,
                                    normalization=self.normalization , 
                                    skip_connection=self.skip_connection , 
                                    device=module_device, 
                                    backend=self.backend)
        self.delta_modules.append(adapterlayer)
        return adapterlayer

