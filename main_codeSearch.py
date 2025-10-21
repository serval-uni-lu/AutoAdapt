
from opendelta import AdapterModel , LoraModel , PrefixModel
import argparse
import logging
import os
import pprint
import torch
import numpy as np
from model import Model
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm
import torch.nn as nn
import transformers
from optimization import * 
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.functional import binary_cross_entropy , binary_cross_entropy_with_logits
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaTokenizer , RobertaModel , AutoModel , AutoTokenizer , AutoConfig)
from utilities import *
from sklearn.metrics import recall_score, precision_score, f1_score
os.environ["TOKENIZERS_PARALLELISM"] = "false"




def train_codeSearch(args, model, tokenizer , train_dataloader_code_search , eval_dataloader_code_search , test_dataloader_code_search=None):
    """ Train the model """
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader_code_search) * args.num_train_epochs)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader_code_search.dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader_code_search)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,best_mrr = 0,0,0 
    train_loss  , eval_MRR = [] , []
    for idx in range(args.num_train_epochs): 
        LOSSes = []
        for step,batch in enumerate(train_dataloader_code_search):
            #get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)
        
            
            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores*20, torch.arange(code_inputs.size(0), device=scores.device))
            
            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            LOSSes.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
        train_loss.append(np.mean(LOSSes))
        #evaluate    
        logger.info("***** Running evaluation *****")
        eval_results_code_search = evaluate_code_search(args, model,eval_dataloader_code_search, eval_dataloader_code_search , eval_when_training=True)
        eval_MRR.append(eval_results_code_search['eval_mrr'])

        for key, value in eval_results_code_search.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if eval_results_code_search['eval_mrr']>best_mrr:
            best_mrr = eval_results_code_search['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best eval mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)  
            
            if not args.do_optimization : 
                logger.info("***** Running Test *****")
                test_results_code_search = evaluate_code_search(args, model, test_dataloader_code_search, test_dataloader_code_search , eval_when_training=False)
                
                for key, value in test_results_code_search.items():
                        logger.info("  %s = %s", key, round(value,4))  

                save_best_model(model, args , checkpoint_prefix="models/best_model_codeSearch")  
            
    if not args.do_optimization : 
        save_best_model(model, args , checkpoint_prefix="models/final_model_codeSearch")  
        test_final = evaluate_code_search(args, model, test_dataloader_code_search, test_dataloader_code_search ) 
            
    return train_loss , eval_MRR


def evaluate_code_search(args, model, query_dataloader , code_dataloader ,eval_when_training=False):
    logger.info("  Num queries = %d", len(code_dataloader.dataset))
    logger.info("  Num codes = %d", len(code_dataloader.dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    code_vecs = [] 
    nl_vecs = []
    for batch in query_dataloader:  
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())  
    #model.train()    
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    
    scores = np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls = []
    code_urls = []
    for example in query_dataloader.dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataloader.dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls,sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr":float(np.mean(ranks))
    }

    return result





def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='./', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--task", default="code_search", type=str, 
                        help="Name of the task")
    parser.add_argument("--train_data_file_CodeSearch", default="./datasets/code_search/train.jsonl", type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--eval_data_file_CodeSearch", default="./datasets/code_search/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file_CodeSearch", default="./datasets/code_search/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default="", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    parser.add_argument("--model_name_or_path", default='microsoft/unixcoder-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="microsoft/unixcoder-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/unixcoder-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--do_train", default=None , type = bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.") 
    parser.add_argument("--do_optimization", default=None , type = bool,
                        help="Whether to run adapter optimization")  
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--train_data_rate_code_search", default=0.0001, type=float,
                        help="The size of the train dataset")
    parser.add_argument("--nb_samples", default=1000, type=int,
                        help="Total number of train samples.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--local_rank', default=-1 ,type=int,
                        help="random seed for initialization")
    parser.add_argument('--population_size', default=5 ,type=int,
                        help="population size on the evolutionary optimization algorithm")
    parser.add_argument('--sample_size', default=3 ,type=int,
                        help="sample size on the evolutionary optimization algorithm")
    parser.add_argument('--cycles', default=2 ,type=int,
                        help="number of cycles on the evolutionary optimization algorithm")
    parser.add_argument('--optimization_history_file', default=None ,type=str,
                        help="saving the history of optimization")
    parser.add_argument('--stats_file', default=None ,type=str,
                        help="saving the optimization statistics ")
    args = parser.parse_args()
    set_seed(seed=args.seed)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1 #torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)



    config = AutoConfig.from_pretrained(args.model_name_or_path , trust_remote_code=True )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path ,trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path,config=config , trust_remote_code=True)  

    
    # Ensure the unified Model sees a list and switches to embedding mode
    if not hasattr(config, "tasks") or config.tasks is None:
        config.tasks = ["code_search"]
    elif isinstance(config.tasks, (str, bytes)):
        config.tasks = [config.tasks.lower()]
    else:
        config.tasks = [str(t).lower() for t in config.tasks]


    #get training dataset
    train_dataset_code_search = TextDataset_code_search(tokenizer, args, args.train_data_file_CodeSearch , nb_samples=None )#args.nb_samples)
    train_dataloader_code_search = DataLoader(train_dataset_code_search, sampler=RandomSampler(train_dataset_code_search), batch_size=args.train_batch_size,num_workers=4)
    eval_dataset_code_search = TextDataset_code_search(tokenizer, args, args.eval_data_file_CodeSearch , nb_samples=None)
    eval_dataloader_code_search = DataLoader(eval_dataset_code_search, sampler=SequentialSampler(eval_dataset_code_search), batch_size=args.eval_batch_size,num_workers=4)
    test_dataset_code_search = TextDataset_code_search(tokenizer, args, args.test_data_file_CodeSearch , nb_samples=None)
    test_dataloader_code_search = DataLoader(test_dataset_code_search, sampler=SequentialSampler(test_dataset_code_search), batch_size=args.train_batch_size,num_workers=4)
    

    
    if args.do_optimization: 

        history, population, best_of_all , stats=  regularized_evolution(args, config , train_dataloader_code_search , eval_dataloader_code_search)
        #pop_list = list(population)
        #sorted_pop = sorted(pop_list, key=lambda x: x[1], reverse=True)
        #with open("./logs_optim/final_population_codeSearch_unixcoder.json", "w") as f:
            #json.dump(sorted_pop, f)

    else : 
        
        """use if you want to train with standards PEFT modules"""
        #delta = AdapterModel(model , bottleneck_dim=[24] )
        #delta = LoraModel(model)
        #delta = PrefixModel(model)
        #delta.freeze_module(exclude=["deltas" ])
        #delta.log()
        #model = Model( model , config)
        #model.to(args.device)
        
        
        """specifiying your own architecture here """
        x_list = [ [{'insert_modules': ('attention.self', 'intermediate', 'output'), 'bottleneck_dim': (16, 64, 128), 'non_linearity': 'gelu', 'dropout_rate': 0.2, 'normalization': 'layer_norm', 'skip_connection': True}, 0, 0, {'insert_modules': ('intermediate', 'attention.self'), 'bottleneck_dim': (64, 32), 'non_linearity': 'swish', 'dropout_rate': 0.3, 'normalization': 'layer_norm', 'skip_connection': True}, 0, 0, 0, 0, 0, 0, {'insert_modules': ('attention.output', 'intermediate', 'attention.self'), 'bottleneck_dim': (32, 64, 16), 'non_linearity': 'silu', 'dropout_rate': 0.0, 'normalization': None, 'skip_connection': True}, {'insert_modules': ('output', 'attention.self'), 'bottleneck_dim': (256, 16), 'non_linearity': 'leakyrelu', 'dropout_rate': 0.1, 'normalization': 'layer_norm', 'skip_connection': True}]]
        
        
        if args.do_train:
            
            """train with different architectures specified in x_list. comment this if you want to train with standard PEFT modules"""
            for x in x_list : 
                set_seed(seed=args.seed)
                model = AutoModel.from_pretrained(args.model_name_or_path,config=config , trust_remote_code=True)  
                logger.info(x)
                model = get_delta_model(model , x , args.device)
                model = Model( model , config)
                model.to(args.device)
                
                """to use in both standard PEFT modules and custom architectures"""
                results = train_codeSearch(args , model ,tokenizer , 
                                           train_dataloader_code_search , 
                                           eval_dataloader_code_search , 
                                           test_dataloader_code_search)
                
                print("train results", results)

    
        if args.do_eval:
                checkpoint_prefix = 'models/final_model_codeSearch/model.bin'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
                model.load_state_dict(torch.load(output_dir) , strict=False)      
                eval_dataset_code_search = TextDataset_code_search(tokenizer, args, args.eval_data_file_CodeSearch, nb_samples=None)
                eval_dataloader_code_search = DataLoader(eval_dataset_code_search, sampler=SequentialSampler(eval_dataset_code_search), batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
                result_task1 = evaluate_code_search(args, model, eval_dataloader_code_search, eval_dataloader_code_search)
                logger.info("\n***** Eval results *****")
                for key , value in result_task1.items() : 
                    logger.info("  %s = %s", key, str(value))




        if args.do_test:
                checkpoint_prefix = 'models/best_model_codeSearch/model.bin'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
                model.load_state_dict(torch.load(output_dir),  strict=False)    
                test_dataset_vul= TextDataset_defect(tokenizer, args,args.test_data_file_vul)
                test_dataloader_vul = DataLoader(test_dataset_vul  , sampler=SequentialSampler(test_dataset_vul ), batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
                task1_test_result = evaluate_code_search(args, model, test_dataloader_vul ) 
                

      



   
       
if __name__ == "__main__":
    main()