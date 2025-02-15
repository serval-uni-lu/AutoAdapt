import torch.nn as nn 
import torch



class Model_classification(nn.Module):   
    def __init__(self, encoder , config ):
        super(Model_classification, self).__init__()
        self.encoder = encoder
        self.config= config
        if 'codet5' in config._name_or_path.lower() :
            self.hidden_size = config.d_model #.embed_dim
            self.encoder = encoder.encoder
            
        else : 
            self.hidden_size = config.hidden_size

        self.classification_head =  nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()

        )

    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            if "codet5" in self.config._name_or_path.lower() :
                outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))
                outputs = outputs.last_hidden_state
                attention_mask =code_inputs.ne(1) 
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
                sum_embeddings = torch.sum(outputs * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                outputs = sum_embeddings / sum_mask
              

            else :
                outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]
             
            outputs = self.classification_head(outputs).squeeze()
            return outputs
        
        else: # just in case we want to add specific actions for nl outputs 
            if "codet5" in self.config._name_or_path.lower() :
                outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))
                encoder_outputs = outputs.last_hidden_state

                # Mean pooling
                attention_mask =nl_inputs.ne(1) 
                attn_scores = self.pooler(encoder_outputs)
                attn_scores = attn_scores.squeeze(-1).masked_fill(attention_mask == 0, float('-inf'))
                attn_weights = torch.softmax(attn_scores, dim=-1) 
                attn_weights = attn_weights.unsqueeze(-1)
                weighted_sum = torch.sum(encoder_outputs * attn_weights, dim=1) 
                outputs = weighted_sum

            else :
                outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]
             
            outputs = self.classification_head(outputs).squeeze()
            return outputs
        


        

class  Model_codeSearch(nn.Module):   
    def __init__(self, encoder , config ):
        super( Model_codeSearch , self).__init__()
        self.encoder = encoder
        self.config= config
        if 'codet5' in config._name_or_path.lower() :
            self.hidden_size = config.d_model #.embed_dim
            self.encoder = encoder.encoder
            
        else : 
            self.hidden_size = config.hidden_size
       
    
    def forward(self, code_inputs=None, nl_inputs=None, task=None): 
        if code_inputs is not None:

            if "codet5" in self.config._name_or_path.lower() :
                outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))
                outputs = outputs.last_hidden_state
                attention_mask =code_inputs.ne(1) 
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
                sum_embeddings = torch.sum(outputs * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                outputs = sum_embeddings / sum_mask
            else :
                outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]

            return torch.nn.functional.normalize(outputs, p=2, dim=1).squeeze()
           
        else:
            if "codet5" in self.config._name_or_path.lower() :
                outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))
                outputs = outputs.last_hidden_state
                attention_mask =nl_inputs.ne(1) 
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
                sum_embeddings = torch.sum(outputs * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                outputs = sum_embeddings / sum_mask
            else :
                outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]

            return torch.nn.functional.normalize(outputs, p=2, dim=1).squeeze()
          