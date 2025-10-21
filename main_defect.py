from myOpenDelta.opendelta import AdapterModel , LoraModel , PrefixModel
import argparse
import logging
import os
import torch
import numpy as np
from model import Model
from tqdm import tqdm
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy , binary_cross_entropy_with_logits
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaTokenizer ,RobertaModel , AutoTokenizer, T5ForConditionalGeneration , AutoConfig , AutoModel)
from sklearn.metrics import recall_score, precision_score, f1_score
from utilities import *
from optimization import *
from transformers import T5Config 



os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_defect(args, model,  tokenizer, train_dataloader , eval_dataloader_defect , test_dataloader_defect=None ):
    """ Train the model """

    
    optimizer =torch.optim.Adam(model.parameters(), lr=args.learning_rate )
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1, num_training_steps=max_steps)

    logger.info("***** Running training for defect detection *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    best_acc=  - np.inf
    model.zero_grad()
    loss_fn = nn.BCELoss()
    early_stopper = EarlyStopper(patience=3, min_delta=0.03)
    results =  {}

    test_result = 0 

    for idx in range(args.num_train_epochs): 

        LOSSes, ACCs =  [], []
        #bar = tqdm(train_dataloader,total=len(train_dataloader))
        for step, batch in enumerate(train_dataloader) : #enumerate(bar) 
            model.train()
            code_inputs = batch[0].to(args.device)  
            labels =  batch[1].to(args.device)  
            labels= labels.float().squeeze()
            logits = model(code_inputs=code_inputs)
            loss = loss_fn(logits,labels)
            accuracy = (logits.round() == labels ).float().mean().item()*100.0
            LOSSes.append(loss.item() )
            # add current accuracies to accuracy arrays 
            ACCs.append(accuracy)
            if (step+1)%100 == 0:
                print("Epoch {} Step {} Train Loss {}   Accuracy {} ".format(idx, step, round(np.mean(LOSSes), 3) ,  round(np.mean(ACCs), 3) ))
            #bar.set_description("Epoch {} Train Loss {}   Accuracy {} ".format(idx, round(np.mean(LOSSes), 3) ,  round(np.mean(ACCs), 3) ))
            
            loss.backward()
            
            # optimizer step 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
        
        results.setdefault('train_loss', []).append(round(np.mean(LOSSes),3))
        results.setdefault('train_acc', []).append(round(np.mean(ACCs),3))
        eval_results = evaluate_defect(args, model, eval_dataloader_defect)
        
        
        results.setdefault('eval_loss', []).append(round(eval_results['eval_loss'],3))
        results.setdefault('eval_acc', []).append(round(eval_results['eval_acc'],3))
        results.setdefault('eval_f1', []).append(round(eval_results['f1_score'],3))
        results.setdefault('eval_precision', []).append(round(eval_results['precision'],3))
        results.setdefault('eval_recall', []).append(round(eval_results['recall'],3))


        for key, value in eval_results.items():
            logger.info("  %s = %s", key, round(value,4))  
        eval_perf = eval_results['eval_acc'] 
        if  eval_perf>best_acc:
            best_acc= eval_perf
            logger.info("\n "+"*"*30)  
            logger.info("  Best validation performance :%s",round(best_acc,4))
            logger.info("  "+"*"*30) 
            if not args.do_optimization : 
                test_result =   test_defect(args, model, test_dataloader_defect)  
                save_best_model(model, args , checkpoint_prefix="models/best_model_defect")
            
            # save best model 

        #if early_stopper.early_stop(round(eval_results['eval_loss'],3)):             
            #break
    
    if not args.do_optimization : 
        save_best_model(model, args , checkpoint_prefix="models/final_model_defect")
        final_test_result =   test_defect(args, model, test_dataloader_defect)  
    
    return results 





def evaluate_defect(args, model, eval_dataloader_vul):
    logger.info("\n***** Running evaluation *****")
    logger.info("  Num examples vulnerability detection = %d", len(eval_dataloader_vul.dataset))
    logger.info("  Batch size = %d ", args.eval_batch_size)

    model.eval()
    loss_fn = nn.BCELoss()
    eval_loss = 0.0
    nb_eval_steps = 0
    logits = []
    labels = []
    for batch in eval_dataloader_vul:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            logit = model(code_inputs=inputs)
            label = label.float().squeeze()
            # Compute loss
            lm_loss = loss_fn(logit, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1

    # Concatenate all logits and labels
    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Binarize predictions
    preds = logits.round()
    
    # Calculate metrics
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds)
    result = {
        "eval_loss": round(float(perplexity), 4),
        "eval_acc": round(eval_acc, 4),
        "f1_score": round(f1, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4)
    }
    return result



# Run test for one task 

def test_defect(args, model, test_dataloader):

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
    logger.info("***** Test Results for task defect detection ")
    logger.info(result )

    return result





def main():



    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file_defect", default="./datasets/dataset_defect/train.jsonl", type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--task", default="defect_detection", type=str, 
                        help="Name of the task")

    parser.add_argument("--eval_data_file_defect", default="./datasets/dataset_defect/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file_defect", default="./datasets/dataset_defect/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    
    parser.add_argument("--output_dir", default='./', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_classes", default=1, type=int,
                        help="The number of classes for the classification model")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    parser.add_argument("--model_name_or_path", default='microsoft/unixcoder-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="microsoft/unixcoder-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/unixcoder-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--do_optimization", default=None, type=bool,
                        help="Whether to run adapter optimization")  
    parser.add_argument("--do_train", default=None, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=None, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=None, type=bool,
                        help="Whether to run eval on the test set.") 
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--train_data_rate_defect", default=1.0, type= float,
                        help="Data size for train")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--nb_samples", default=None, type=int,
                        help="Total number of train samples.")
    parser.add_argument("--nb_samples_valid", default=None, type=int,
                        help="Total number of validation samples.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--local_rank', default=-1 ,type=int,
                        help="random seed for initialization")
    parser.add_argument('--population_size', default=None ,type=int,
                        help="population size on the evolutionary optimization algorithm")
    parser.add_argument('--sample_size', default=None ,type=int,
                        help="sample size on the evolutionary optimization algorithm")
    parser.add_argument('--cycles', default=None ,type=int,
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

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels = args.num_classes , trust_remote_code=True )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path , trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path,config=config , trust_remote_code=True) 
    
    if not hasattr(config, "tasks") or config.tasks is None:
        config.tasks = ["defect_detection"]
    elif isinstance(config.tasks, (str, bytes)):
        config.tasks = [config.tasks.lower()]
    else:
        config.tasks = [str(t).lower() for t in config.tasks]
    
    
    train_dataset=TextDataset_defect(tokenizer, args, args.train_data_file_defect, nb_samples =None) #args.nb_samples)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4, pin_memory=True )
    

    # prepare validation data 
    eval_dataset_defect= TextDataset_defect(tokenizer, args,args.eval_data_file_defect,nb_samples=None) #args.nb_samples_valid )
    eval_dataloader_defect = DataLoader(eval_dataset_defect  , sampler=SequentialSampler(eval_dataset_defect ), batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # prepare test dataloaders 
    test_dataset_defect= TextDataset_defect(tokenizer, args,args.test_data_file_defect, nb_samples=None) #args.nb_samples_valid)
    test_dataloader_defect = DataLoader(test_dataset_defect  , sampler=SequentialSampler(test_dataset_defect ), batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)



    if args.do_optimization: 
        logger.info("Starting optimization...")
        history, population, best_of_all , stats=  regularized_evolution(args, config , train_dataloader , eval_dataloader_defect)
    
    else : 

        
        """
        # to test top configs 
        
        x_list = [ 
            [{'insert_modules': ('layer.0', 'layer.1.DenseReluDense'), 'bottleneck_dim': (64, 128), 'non_linearity': 'relu', 'dropout_rate': 0.1, 'normalization': 'layer_norm', 'skip_connection': True}, 0, {'insert_modules': ('layer.1',), 'bottleneck_dim': (128,), 'non_linearity': 'leakyrelu', 'dropout_rate': 0.0, 'normalization': None, 'skip_connection': True}, {'insert_modules': ('layer.1.DenseReluDense',), 'bottleneck_dim': (128,), 'non_linearity': 'gelu_new', 'dropout_rate': 0.0, 'normalization': 'layer_norm', 'skip_connection': True}, 0, {'insert_modules': ('layer.0', 'layer.1', 'layer.1.DenseReluDense'), 'bottleneck_dim': (64, 256, 128), 'non_linearity': 'gelu_new', 'dropout_rate': 0.25, 'normalization': 'layer_norm', 'skip_connection': True}, 0, {'insert_modules': ('layer.1.DenseReluDense',), 'bottleneck_dim': (64,), 'non_linearity': 'silu', 'dropout_rate': 0.3, 'normalization': None, 'skip_connection': True}, 0, {'insert_modules': ('layer.1', 'layer.0', 'layer.1.DenseReluDense'), 'bottleneck_dim': (256, 128, 64), 'non_linearity': 'silu', 'dropout_rate': 0.2, 'normalization': 'layer_norm', 'skip_connection': True}, {'insert_modules': ('layer.0', 'layer.0.SelfAttention'), 'bottleneck_dim': (64, 32), 'non_linearity': 'relu', 'dropout_rate': 0.3, 'normalization': None, 'skip_connection': True}, 0], 
            [0, {'insert_modules': ('layer.0.SelfAttention',), 'bottleneck_dim': (16,), 'non_linearity': 'silu', 'dropout_rate': 0.1, 'normalization': None, 'skip_connection': True}, {'insert_modules': ('layer.1.DenseReluDense',), 'bottleneck_dim': (128,), 'non_linearity': 'gelu_new', 'dropout_rate': 0.25, 'normalization': 'layer_norm', 'skip_connection': True}, {'insert_modules': ('layer.1.DenseReluDense', 'layer.0', 'layer.1'), 'bottleneck_dim': (128, 64, 128), 'non_linearity': 'tanh', 'dropout_rate': 0.3, 'normalization': 'layer_norm', 'skip_connection': True}, {'insert_modules': ('layer.1', 'layer.0', 'layer.0.SelfAttention'), 'bottleneck_dim': (128, 64, 16), 'non_linearity': 'tanh', 'dropout_rate': 0.1, 'normalization': None, 'skip_connection': True}, {'insert_modules': ('layer.1',), 'bottleneck_dim': (256,), 'non_linearity': 'leakyrelu', 'dropout_rate': 0.2, 'normalization': None, 'skip_connection': True}, {'insert_modules': ('layer.1.DenseReluDense', 'layer.0.SelfAttention'), 'bottleneck_dim': (64, 16), 'non_linearity': 'leakyrelu', 'dropout_rate': 0.3, 'normalization': None, 'skip_connection': True}, 0, {'insert_modules': ('layer.0', 'layer.1.DenseReluDense', 'layer.0.SelfAttention'), 'bottleneck_dim': (128, 128, 32), 'non_linearity': 'gelu_new', 'dropout_rate': 0.3, 'normalization': None, 'skip_connection': True}, {'insert_modules': ('layer.0.SelfAttention', 'layer.0', 'layer.1.DenseReluDense'), 'bottleneck_dim': (32, 128, 64), 'non_linearity': 'tanh', 'dropout_rate': 0.0, 'normalization': None, 'skip_connection': True}, {'insert_modules': ('layer.1.DenseReluDense',), 'bottleneck_dim': (128,), 'non_linearity': 'gelu_new', 'dropout_rate': 0.0, 'normalization': 'layer_norm', 'skip_connection': True}, {'insert_modules': ('layer.0', 'layer.1.DenseReluDense', 'layer.0.SelfAttention'), 'bottleneck_dim': (64, 64, 16), 'non_linearity': 'swish', 'dropout_rate': 0.3, 'normalization': 'layer_norm', 'skip_connection': True}], 
            [{'insert_modules': ('layer.0.SelfAttention', 'layer.1.DenseReluDense', 'layer.0'), 'bottleneck_dim': (32, 128, 64), 'non_linearity': 'relu', 'dropout_rate': 0.1, 'normalization': None, 'skip_connection': True}, 0, {'insert_modules': ('layer.1.DenseReluDense', 'layer.0', 'layer.1'), 'bottleneck_dim': (128, 64, 256), 'non_linearity': 'swish', 'dropout_rate': 0.15, 'normalization': 'layer_norm', 'skip_connection': True}, {'insert_modules': ('layer.1', 'layer.0.SelfAttention'), 'bottleneck_dim': (256, 32), 'non_linearity': 'tanh', 'dropout_rate': 0.2, 'normalization': 'layer_norm', 'skip_connection': True}, {'insert_modules': ('layer.1.DenseReluDense',), 'bottleneck_dim': (128,), 'non_linearity': 'relu', 'dropout_rate': 0.15, 'normalization': 'layer_norm', 'skip_connection': True}, {'insert_modules': ('layer.0', 'layer.1'), 'bottleneck_dim': (64, 128), 'non_linearity': 'gelu_new', 'dropout_rate': 0.15, 'normalization': 'layer_norm', 'skip_connection': True}, {'insert_modules': ('layer.0.SelfAttention', 'layer.0'), 'bottleneck_dim': (32, 64), 'non_linearity': 'gelu', 'dropout_rate': 0.25, 'normalization': None, 'skip_connection': True}, 0, 0, {'insert_modules': ('layer.1', 'layer.0', 'layer.1.DenseReluDense'), 'bottleneck_dim': (256, 128, 64), 'non_linearity': 'silu', 'dropout_rate': 0.2, 'normalization': 'layer_norm', 'skip_connection': True}, 0, {'insert_modules': ('layer.0', 'layer.0.SelfAttention'), 'bottleneck_dim': (64, 32), 'non_linearity': 'relu', 'dropout_rate': 0.3, 'normalization': None, 'skip_connection': True}], 

        ]
        """
        
        
        # to finetune with a fixed adapter across all layers 
        delta = AdapterModel(model , bottleneck_dim=[24])
        #delta = LoraModel(model)
        #delta = PrefixModel(model)
        delta.freeze_module(exclude=["deltas" ])
        delta.log()
        model = Model( model , config)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=[0])
        model.to(args.device)
        print(model)
        if args.do_train:
            
            # loop for training with different configs in x_list 
            """
            for x in x_list : 
                set_seed(seed=args.seed)
                model = AutoModel.from_pretrained(args.model_name_or_path,config=config , trust_remote_code=True) 
                print('\n',x,'\n')
                model = get_delta_model(model , x)
                model = Model( model , config)
                model.to(args.device)
            """
            results = train_defect(args , model ,tokenizer , 
                                       train_dataloader , 
                                       eval_dataloader_defect , 
                                       test_dataloader_defect)
                
            print("train results", results)

    
        if args.do_eval:
                checkpoint_prefix = 'models/best_model_defect/model.bin'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
                model.load_state_dict(torch.load(output_dir) , strict=False)      
                eval_dataset_vul= TextDataset_defect(tokenizer, args,args.eval_data_file_defect)
                eval_dataloader_vul = DataLoader(eval_dataset_vul  , sampler=SequentialSampler(eval_dataset_vul ), batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
                result_task1= evaluate_defect(args, model, eval_dataloader_vul  )
                logger.info("\n***** Eval results *****")
                for key , value in result_task1.items() : 
                    logger.info("  %s = %s", key, str(value))




        if args.do_test:
                checkpoint_prefix = 'models/best_model_vul/model.bin'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
                model.load_state_dict(torch.load(output_dir),  strict=False)    
                test_dataset_vul= TextDataset_defect(tokenizer, args,args.test_data_file_defect)
                test_dataloader_vul = DataLoader(test_dataset_vul  , sampler=SequentialSampler(test_dataset_vul ), batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
                task1_test_result = test_defect(args, model, test_dataloader_vul ) 
                        

        
if __name__ == "__main__":
    main()