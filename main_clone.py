
from myOpenDelta.opendelta import AdapterModel , LoraModel , PrefixModel
import argparse
import logging
import os
import torch
import numpy as np
from model import Model
from tqdm import tqdm
import torch.nn as nn
import transformers
from torch.nn.functional import binary_cross_entropy , binary_cross_entropy_with_logits
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler 
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer , RobertaForSequenceClassification , AutoModel , AutoConfig , AutoTokenizer)
import torch.distributed as dis
from torch.nn.parallel import DistributedDataParallel as DDP
from utilities import *
from optimization import *
from sklearn.metrics import recall_score, precision_score, f1_score
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.utils.logging.set_verbosity_error()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("name")





def train_clone(args, model,  tokenizer, train_dataloader , eval_dataloader , test_dataloader=None):
    """ Train the model """

    optimizer =torch.optim.Adam(model.parameters(), lr=args.learning_rate )
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1, num_training_steps=max_steps)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    best_acc=  - np.inf
    model.zero_grad()
    loss_fn = nn.BCELoss()
    early_stopper = EarlyStopper(patience=3, min_delta=0.03)
    results =  {}
    for idx in range(args.num_train_epochs): 
        LOSSes, ACCs =  [], []
        #bar = tqdm(train_dataloader,total=len(train_dataloader))
        for step, batch in enumerate(train_dataloader) : #enumerate(bar):

            model.train()
            code_inputs = batch[0].to(args.device)  
            labels =  batch[1].to(args.device)  
            labels= labels.float().squeeze()
            logits = model(code_inputs=code_inputs)
            loss = loss_fn(logits,labels)
            accuracy = (logits.round() == labels ).float().mean().item()*100.0
            # perfom a backward step 
            LOSSes.append(loss.item() )
            # add current accuracies to accuracy arrays 
            ACCs.append(accuracy)
        
            # update progress bar
            #bar.set_description("Epoch {} Train Loss {}  Accuracy {}  ".format(idx, round(np.mean(LOSSes), 3) , np.round(np.mean(ACCs))))
            if (step+1)%100 == 0:
                logger.info("Epoch {} Step {} Train Loss {}   Accuracy {} ".format(idx, step, round(np.mean(LOSSes), 3) ,  round(np.mean(ACCs), 3) ))
            
            loss.backward()
            
            # optimizer step 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        results.setdefault('train_loss', []).append(round(np.mean(LOSSes),3))
        results.setdefault('train_acc', []).append(round(np.mean(ACCs),3))
        eval_results = evaluate_clone(args, model, eval_dataloader)
        results.setdefault('eval_loss', []).append(eval_results['eval_loss'])
        results.setdefault('eval_acc', []).append(eval_results['eval_acc'])
        results.setdefault('eval_f1', []).append(eval_results['f1_score'])
        results.setdefault('eval_precision', []).append(eval_results['precision'])
        results.setdefault('eval_recall', []).append(eval_results['recall'])

        for key, value in eval_results.items():
            logger.info("  %s = %s", key, value)  


        if eval_results['f1_score']>best_acc:
            best_acc=eval_results['f1_score']
            logger.info("\n "+"*"*30)  
            logger.info("  Best F1 score :%s",round(best_acc,4))
            logger.info("  "+"*"*30)   
            if not args.do_optimization : 
        
                #save_best_model(model, args , checkpoint_prefix="models/best_model_clone")
                test_result =   test_clone(args, model, test_dataloader)  
        
        #if early_stopper.early_stop(round(eval_results['eval_loss'],3)):             
            #break
    if not args.do_optimization : 
        save_best_model(model, args , checkpoint_prefix="models/final_model_clone")
        final_test_result =   test_clone(args, model, test_dataloader)
    
    return results  





# run validation for both tasks 
def evaluate_clone(args, model, eval_dataloader_clone ):
        
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples clone detection = %d", len(eval_dataloader_clone.dataset))
        logger.info("  Batch size = %d ", args.eval_batch_size)

        model.eval()
        loss_fn = nn.BCELoss()

        eval_loss = 0.0
        nb_eval_steps = 0
        logits = []
        labels = []
        for batch in eval_dataloader_clone:
            inputs = batch[0].to(args.device)
            label = batch[1].to(args.device)
            with torch.no_grad():
                logit = model(code_inputs=inputs)
                label = label.float().squeeze()
                lm_loss = loss_fn(logit, label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())
            nb_eval_steps += 1
        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)
        preds = logits.round()
        eval_acc = np.mean(labels ==  preds)
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.tensor(eval_loss)
        recall = recall_score(labels , preds)
        precision = precision_score(labels , preds , zero_division=0)
        f1 = f1_score(labels , preds)
        result = {
            "eval_loss": round(float(perplexity),4),
            "eval_acc": round(eval_acc, 4),
            "f1_score" : round(f1, 4),
            "recall" : round(recall,4),
            "precision" : round(precision,4)}

        return result





# Run test for one task 

def test_clone(args, model, test_dataloader):

    logits = []
    labels = []

    for batch in test_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            logit = model(code_inputs=inputs)
            label = label.float().squeeze()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.round()
    acc = np.mean(labels ==  preds)
    recall = recall_score(labels , preds)
    precision = precision_score(labels , preds , zero_division=0)
    f1 = f1_score(labels , preds)

    result = {
    
            "test_acc": round(acc, 4),
            "test_f1_score" : round(f1, 4),
            "test_recall" : round(recall,4),
            "test_precision" : round(precision,4)
        }
    logger.info("\n***** Test Results for clone detection ")
    logger.info("\n{}\n".format(result ))

    return result




def main():



    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="clone_detection", type=str, 
                        help="Name of the task")
    parser.add_argument("--train_data_file", default="./datasets/dataset_clone/train.txt", type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default='./', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_classes", default=1, type=int,
                        help="The number of classes for the classification model")
    parser.add_argument("--eval_data_file", default="./datasets/dataset_clone/valid.txt", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default="./datasets/dataset_clone/test.txt", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    parser.add_argument("--model_name_or_path", default='microsoft/graphcodebert-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--do_optimization", default=None, type=bool,
                        help="Whether to run adapter optimization")  
    parser.add_argument("--do_train", default=True, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=None, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=None, type=bool,
                        help="Whether to run eval on the test set.") 
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--train_data_rate_clone", default=0.0001, type= float,
                        help="Data size for train")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--nb_samples", default=100, type=int,
                        help="Total number of train samples.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--local_rank', default=-1 ,type=int,
                        help="random seed for initialization")
    parser.add_argument('--population_size', default=3 ,type=int,
                        help="population size on the evolutionary optimization algorithm")
    parser.add_argument('--sample_size', default=2 ,type=int,
                        help="sample size on the evolutionary optimization algorithm")
    parser.add_argument('--cycles', default=2 ,type=int,
                        help="number of cycles on the evolutionary optimization algorithm")
    parser.add_argument('--optimization_history_file', default=None ,type=str,
                        help="saving the history of optimization")
    parser.add_argument('--stats_file', default=None ,type=str,
                        help="saving the optimization statistics ")
    
    
    
    args = parser.parse_args()
    set_seed(seed=args.seed)
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1 #torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)
    config = AutoConfig.from_pretrained(args.model_name_or_path , num_labels = args.num_classes ,  trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path ,  trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path,config=config ,  trust_remote_code=True)  
    
    # Ensure the unified Model sees a list and picks classification behavior
    if not hasattr(config, "tasks") or config.tasks is None:
        config.tasks = ["clone_detection"]
    elif isinstance(config.tasks, (str, bytes)):
        config.tasks = [config.tasks.lower()]
    else:
        config.tasks = [str(t).lower() for t in config.tasks]


    train_dataset=TextDataset_clone(tokenizer, args, args.train_data_file, nb_samples = None) #args.nb_samples)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4,pin_memory=True )
    
    eval_dataset = TextDataset_clone(tokenizer, args,args.eval_data_file , nb_samples=41541 )
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset , sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)


    test_dataset = TextDataset_clone(tokenizer, args,args.test_data_file  ,nb_samples= 41541 ) 
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size , num_workers=4,pin_memory=True)



    if args.do_optimization: 
        logger.info("Starting optimization...")
        history, population, best_of_all, stats  =  regularized_evolution(args, config, train_dataloader , eval_dataloader )
    else : 
        
        
        # enable to insert default adapter / lora/ prefix , with a fixed adapter across all layers 
        #delta = AdapterModel(model , bottleneck_dim=[24])
        #delta = LoraModel(model)
        #delta = PrefixModel(model)
        #delta.freeze_module(exclude=["deltas" ])
        #delta.log()
        #model = Model( model , config)
        #if args.n_gpu > 1:
            #model = torch.nn.DataParallel(model, device_ids=[0,1])
        #model.to(args.device)
        # top 3 adapter configs 
        
        
        x_list = [ [{'insert_modules': ('attention.self', 'intermediate', 'output'), 'bottleneck_dim': (16, 64, 128), 'non_linearity': 'gelu', 'dropout_rate': 0.2, 'normalization': 'layer_norm', 'skip_connection': True}, 0, 0, {'insert_modules': ('intermediate', 'attention.self'), 'bottleneck_dim': (64, 32), 'non_linearity': 'swish', 'dropout_rate': 0.3, 'normalization': 'layer_norm', 'skip_connection': True}, 0, 0, 0, 0, 0, 0, {'insert_modules': ('attention.output', 'intermediate', 'attention.self'), 'bottleneck_dim': (32, 64, 16), 'non_linearity': 'silu', 'dropout_rate': 0.0, 'normalization': None, 'skip_connection': True}, {'insert_modules': ('output', 'attention.self'), 'bottleneck_dim': (256, 16), 'non_linearity': 'leakyrelu', 'dropout_rate': 0.1, 'normalization': 'layer_norm', 'skip_connection': True}]
        ]
        
        if args.do_train:
            
            for x in x_list : 
                set_seed(seed=args.seed)
                model = AutoModel.from_pretrained(args.model_name_or_path,config=config ,  trust_remote_code=True)  
                logger.info(x)
                model = get_delta_model(model , x, args.device)
                model = Model( model , config)
                if args.n_gpu > 1:
                    model = torch.nn.DataParallel(model, device_ids=[1])
                model.to(args.device)
                results = train_clone(args , model ,tokenizer ,  
                                    train_dataloader , 
                                    eval_dataloader , 
                                    test_dataloader)
                
                logger.info("train results : \n {}\n".format(results))
                logger.info("*"*130)


        if args.do_eval:
            checkpoint_prefix = 'models/final_model_clone/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir) , strict=False)      

            eval_dataset_clone= TextDataset_clone(tokenizer, args,args.eval_data_file_clone)
            eval_dataloader_clone = DataLoader(eval_dataset_clone  , sampler=SequentialSampler(eval_dataset_clone ), batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
        
            result_task1= evaluate_clone(args, model, eval_dataloader_clone  )

            logger.info("\n***** Eval results *****")
            for key , value in result_task1.items() : 
                logger.info("  %s = %s", key, str(value))
    
        
        if args.do_test:
            checkpoint_prefix = 'models/best_model_clone/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir),  strict=False)    
            test_dataset_clone= TextDataset_clone(tokenizer, args,args.test_data_file_clone)
            test_dataloader_clone = DataLoader(test_dataset_clone  , sampler=SequentialSampler(test_dataset_clone ), batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
            task1_test_result = test_clone(args, model, test_dataloader_clone ) 



if __name__ == "__main__":
    main()