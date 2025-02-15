
from torch.utils.data.distributed import Dataset 
import torch
import json
import random
import logging
import os
import numpy as np

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger("name")






class InputFeatures_defect ( object) : 

    def __init__(self,
                   code_tokens,
                   code_ids,
                   label): 
        self.code_tokens = code_tokens
        self.code_ids =  code_ids
        self.label = label





def convert_examples_to_features_defect(js,tokenizer,args):
  
    code=''.join(js['code_tokens'])
    code_tokens=tokenizer.tokenize(code)[:args.code_length-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures_defect(code_tokens,code_ids,js['target'])






class TextDataset_defect(Dataset):
    def __init__(self, tokenizer, args, file_path=None, nb_samples = None , is_test=None,lang=None):
        self.examples = []
        
        data=[]
            
        with open(file_path) as f:
            for line in f:
                line = json.loads(line.strip())
                js = {}
                code = ' '.join(line['func'].split())
                label = int(line['target'])
                js['code_tokens'] = code
                js['target'] = label
                data.append(js)

        if 'train' in file_path : 
            size =   int (args.train_data_rate_defect * len(data))
        else : 
            size =  len(data)
        
        if nb_samples : 
            size = nb_samples
       
        for js in data[:size]:
            self.examples.append(convert_examples_to_features_defect(js,tokenizer,args))
        
        
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].label))



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False




def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True







class InputFeatures_clone ( object) : 

    def __init__(self,
                   code_ids,
                   label): 
        
        self.code_ids = code_ids
        self.label = label





def convert_examples_to_features_clone(js,tokenizer,args):

    ids_args =  ((js['code1'], js['code2']))

    result = tokenizer(*ids_args, padding="max_length", max_length=args.code_length-2, truncation='longest_first')
 

    return InputFeatures_clone( result['input_ids'] , js['label'])





class TextDataset_clone(Dataset):
    def __init__(self, tokenizer, args, file_path=None, nb_samples= None , is_test=None,lang=None):
        self.examples = []
       
        logger.info("Preparing the Dataset...\n")
        url_to_code = {}
        with open("./datasets/dataset_clone/data.jsonl") as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                code = ' '.join(js['func'].split())
                url_to_code[js['idx']] = code

        data = []
    
        with open(file_path) as f:
            for line in f:
                js = {}
                line = line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue

                js['code1'] = url_to_code[url1]
                js['code2']= url_to_code[url2]
                js['label']= int(label)
                data.append(js)

        
        if 'train' in file_path : 
            size =   int (args.train_data_rate_clone * len(data))
        else : 
            size =  len(data)
        if nb_samples : 
            size = nb_samples

        for js in data[:size]:
            
            self.examples.append(convert_examples_to_features_clone(js,tokenizer,args))  


     



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].label))







class InputFeatures_code_search(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url
        self.task = "code_search"

        
def convert_examples_to_features_code_search(js,tokenizer,args):
    """convert examples to token ids"""
    code_length = 256
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:code_length -4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures_code_search(code_tokens,code_ids,nl_tokens,nl_ids,js['url'] if "url" in js else js["retrieval_idx"])

class TextDataset_code_search(Dataset):
    def __init__(self, tokenizer, args, file_path=None , nb_samples= None):
        self.examples = []
        data = []
        logger.info("Preparing the code search Dataset...\n")
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js) 
        if 'train' in file_path : 
            size =   int (args.train_data_rate_code_search * len(data))
        else : 
            size =  len(data) 
        if nb_samples : 
            size = nb_samples

        for js in data[:size]:
            self.examples.append(convert_examples_to_features_code_search(js,tokenizer,args))                      
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids), self.examples[i].task)
            








#--------------flaky tests -----------------------------------------------------------------------------------------------------

class InputFeatures_flakyTest( object) : 

    def __init__(self,
                   code_tokens,
                   code_ids,
                   label): 
        self.code_tokens = code_tokens
        self.code_ids =  code_ids
        self.label = label
        self.task = "flakiness_detect"



class TextDataset_flakyTest(Dataset):
    def __init__(self, tokenizer, args, file_path=None, nb_samples= None, is_test=None,lang=None):
        self.examples = []
        

        logger.info("Preparing the flakeFlager Dataset...\n")
        data=[]
            
    
        with open(file_path) as f:
            data_list = json.load(f)
            for line in data_list:
                js = {}
                code = ' '.join(line['code'].split())
                label = int(line['label'])
                js['code_tokens'] = code
                js['label'] = label
                data.append(js)

        if 'train' in file_path : 
            size =   int (args.train_data_rate_flaky * len(data))
        else : 
            size =  len(data)
            
        if nb_samples : 
            size = nb_samples
       
        for js in data[:size]:
            self.examples.append(convert_examples_to_features_flakyTest(js,tokenizer,args))
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].label),self.examples[i].task )
    



def convert_examples_to_features_flakyTest(js,tokenizer,args):
    
        code=''.join(js['code_tokens'])
        code_tokens=tokenizer.tokenize(code)[:args.code_length-2]
        code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.code_length - len(code_ids)
        code_ids+=[tokenizer.pad_token_id]*padding_length
        return InputFeatures_flakyTest(code_tokens,code_ids,js['label'])





def save_best_model(model, args, checkpoint_prefix='checkpoint-best-acc'):
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_path = os.path.join(output_dir, 'model.bin')
    torch.save(model_to_save.state_dict(), output_path)
    logger.info("Saving model checkpoint to %s", output_path)





def count_trainable_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)






def count_rate_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage =  np.round ( (trainable_params / total_params) , 4 )
    return  percentage




def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)



