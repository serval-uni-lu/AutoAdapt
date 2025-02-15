
import collections 
from copy import deepcopy
import random
import math
import numpy as np
from model import Model_classification , Model_codeSearch
from torch.nn.functional import binary_cross_entropy , binary_cross_entropy_with_logits
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel ,AutoModel, AutoTokenizer)
from myOpenDelta.opendelta import AdapterModel
import numpy as np 
import random 
import traceback
from main_defect import train_defect 
from main_codeSearch import train_codeSearch
from main_clone import train_clone
from utilities import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("name")
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)


INSERT_LOCATIONS = ['attention.output', 'intermediate','output' , "attention.self"]

INSERT_LOCATION_T5 =  ['layer.0' , 'layer.0.SelfAttention' , 'layer.1' , 'layer.1.DenseReluDense']


ACTIVATIONS = ['gelu_new', 'relu' , 'leakyrelu', 'gelu' , 'tanh', 'silu' , 'swish']
MAX_FNN_SIZE = 3
FAIL_COMPILE = 0
CROSSOVER_RATE = 0.6
DROPOUT =  [0.0 , 0.1 , 0.15 , 0.2 , 0.25 , 0.3]

FFN_SIZES = {
  'attention.self': [16, 32],
  'attention.output': [32, 64],
  'intermediate': [64, 128],
  'output': [128, 256],
  'layer.0' : [64, 128],
  'layer.0.SelfAttention' : [16, 32],
  'layer.1' :[128, 256] ,
  'layer.1.DenseReluDense' : [64, 128]
}







def dict_to_tuple (dic) : 

    if isinstance(dic , dict) : 
        for n , p in dic.items() : 
            if isinstance(p, list) :
                dic[n] = tuple(p)
    return tuple(dic)


def get_config_hash(adapter_params):
    return hash(tuple([dict_to_tuple(d) if d != 0 else d for d in adapter_params]))



def save_statistics(stats, output_dir , filename):
    """
    Saves the statistics dictionary as a JSON file.
    
    Args:
        stats (dict): Dictionary containing per-cycle statistics.
        output_dir (str): Directory to save the JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, filename)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Evolution statistics saved to {stats_path}")




def generate_binary_list(size ) : 
    """
    Generates a randomized binary list of given size with at least one zero and one one.
    Useful for creating binary masks for selecting layers or components.

    Args:
        size (int): Length of the binary list to generate.

    Returns:
        list: A shuffled binary list containing a mix of 0s and 1s.
    """

    zero_count = random.randint(0, size-2)
    one_count = size - zero_count
    my_list = [0]*zero_count + [1]*one_count
    random.shuffle(my_list)

    return my_list







def random_ffn (insert_modules) : 

    """
    Args:
        insert_modules (list): List of module locations where adapters will be inserted.

    Returns:
        list: A list of randomly chosen FFN sizes, one for each location in `insert_modules`.
    """

    size_list = []
    for location in insert_modules : 
        size_list.append(random.choice(FFN_SIZES[location]))

    return size_list
 






def random_insert_modules (config) : 
    """
    Returns:
        list: A randomized subset of module locations from `INSERT_LOCATIONS`.
    """
    model_name = config._name_or_path.lower() 
    nb_locations = random.randint(1,len(INSERT_LOCATIONS)-1)

    if 'codet5' in model_name : 
        return random.sample(INSERT_LOCATION_T5 , k=nb_locations)
    else :
        return random.sample(INSERT_LOCATIONS , k=nb_locations)






def random_configuration(config):
    """
    Generates a random adapter configuration. 
    This configuration determines where adapters are inserted, their bottleneck sizes, 
    activation functions, dropout rates, normalization techniques, and whether skip connections are used.

    Returns:
        dict: A dictionary containing the random configuration for the adapter.
    """
      

    insert_modules = random_insert_modules(config)
    bottleneck_sizes = random_ffn(insert_modules)

    configuration = {
        "insert_modules": list(insert_modules),
        "bottleneck_dim": list(bottleneck_sizes),
        "non_linearity": random.choice(ACTIVATIONS),
        "dropout_rate": random.choice(DROPOUT),  
        "normalization": random.choice([None, "layer_norm"]),
        "skip_connection":  True  # random.choice([True, False])
    }
    return configuration







def random_adapter_parameters (config) : 

    """
    Generates random adapter parameters for each layer in a transformer model.

    Args:
        config: A configuration object containing model details.

    Returns:
        list: A list where each element corresponds to a layer.
              If the layer has an adapter, the element is a configuration dictionary.
              Otherwise, the element is 0 (no adapter for that layer).
    """
    

    adapter_layers = generate_binary_list(config.num_hidden_layers)
    parameters = list()
    
    for layer_idx , insert_decision in enumerate(adapter_layers) : 
        if insert_decision==1 : 
            configuration = random_configuration(config)
            parameters.append(configuration)
        else : 
            parameters.append(0)

    return parameters 







def get_delta_model ( model, adapter_parameters:dict , device= "cpu")  : 

    """
    Integrates adapter configurations into a given model using `AdapterModel`.

    For each layer in the transformer, this function checks if an adapter is specified.
    If an adapter is present, it constructs the corresponding adapter module and integrates it into the model.

    Args:
        model: The backbone model (e.g., a transformer model like Roberta).
        adapter_parameters (dict): A dictionary containing adapter configurations for each layer.
                                   Layers without adapters are marked as 0.

    Returns:
        model: The updated model with adapters applied to the specified layers and locations.
    """

    delta_model = None
    model_name = model.config._name_or_path.lower()  

    if "salesforce" in model_name:
        prefix = "encoder.block."
    elif "microsoft" in model_name:
        prefix = "layer."
    else:
        prefix = "layer."

    for layer_id, adapter_param in enumerate(adapter_parameters) :
        if adapter_param != 0 :  
            for module_id , module in  enumerate(adapter_param['insert_modules']) : 
                insert_module =  prefix + str(layer_id) + '.' + str(module).strip()

                
                delta_model = AdapterModel(backbone_model= model ,
                                    modified_modules= [insert_module],
                                    bottleneck_dim=[adapter_param['bottleneck_dim'][module_id]] ,
                                    non_linearity=adapter_param['non_linearity'],
                                    dropout_rate=adapter_param['dropout_rate'],
                                    normalization=adapter_param['normalization'],
                                    skip_connection=adapter_param['skip_connection'],
                                    #device=device
                                    
                                    )
               
           
    delta_model.freeze_module(exclude=["deltas" ])
    delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=False)
    logger.info(count_rate_trainable_parameters(model.encoder)*100)
    return model





def regularized_evolution(args, config, train_datalaoder , eval_dataloader,  cycles=20 , population_size=20, sample_size=10, init_population=None, init_history=None):

    best_of_all = [[] , 0]
    population = collections.deque()
    history = []  
    hash_pop = []

    # statistiques 
    best_fitness_per_cycle = []
    avg_fitness_per_cycle = []
    std_fitness_per_cycle = []
    best_accuracy_per_cycle = []
    avg_accuracy_per_cycle = []
    avg_trainable_rate_per_cycle = []
    unique_configs_per_cycle = []
    best_fscore_per_cycle = []
    avg_fscore_per_cycle = []


    # Override parameters if provided
    population_size = args.population_size if args.population_size else population_size
    cycles = args.cycles if args.cycles else cycles
    sample_size = args.sample_size if args.sample_size else sample_size


    
    logger.info("\n\nInitialization of the population, may take a while ...\n\n")
   
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if init_population is None:
        while len(population) < population_size:
            model = AutoModel.from_pretrained(args.model_name_or_path,config=config) 
            delta_parameters = random_adapter_parameters(config)
            model = get_delta_model(model , delta_parameters , args.device)

            if args.task =="code_search" : 
                delta_model =  Model_codeSearch( model, config )
            else :

                delta_model = Model_classification( model, config )


            logger.info(" ==> Delta parameters \n {} \n".format( delta_parameters ))
            adapter_params = tuple([dict_to_tuple(d)  if d!=0 else d for d in delta_parameters])

            
            if args.n_gpu > 1:
                delta_model = torch.nn.DataParallel(delta_model, device_ids=[0])

            delta_model.to(args.device)
            
            if hash(adapter_params) in hash_pop:
                continue

            hash_pop.append(hash(adapter_params))

        
            metric , perfomance , f1_score , trainale_rate = fitness(delta_model,  args , tokenizer , train_datalaoder , eval_dataloader)
            logger.info(f"==> Fitness : {metric}")
            logger.info(f"==> Eval Accuracy : {perfomance}" )
            logger.info(f"==> Eval f1 score : {f1_score}")
            logger.info(f"==> Parameters % : {trainale_rate*100}\n")

            

            if metric == 0 :
                continue
        
            population.append((delta_parameters, metric , perfomance , f1_score, trainale_rate))
            history.append((delta_parameters, metric , perfomance , f1_score, trainale_rate))
    else:
        population = init_population
        for individual in population:
            hash_pop.append(hash(tuple(individual[0])))
    if init_history is not None:
        history = init_history
    
    
    # Carry out evolution in cycles

    logger.info("\n\n Carry out evolution in cycles... \n\n")

    while len(history) - population_size < cycles:
        
            sample = []
        
            while len(sample) < sample_size:
                candidate = random.choice(list(population))
                sample.append(candidate)

            random_number =  random.random()
            if random_number < CROSSOVER_RATE:
                # do crossover 
                logger.info("CROSSOVER\n")
                sample.sort(key=lambda i: i[1])
                parent1 , parent2 = sample[-1][0] , sample[-2][0]
                child =  crossover(parent1, parent2 , config)

                
            else : 
                # do mutation 
                logger.info("MUTATION\n")
                # The parent is the best model in the sample
                parent = max(sample, key=lambda i: i[1])
                # Create the child model by mutating its architecture
                child = mutate(parent[0] , config)

            logger.info(f"\n{child}\n")

            model = AutoModel.from_pretrained(args.model_name_or_path,config=config) 
            child_model =   get_delta_model(model , child , args.device)

            if args.task =="code_search" : 
                child_model =  Model_codeSearch( child_model, config)
            else :

                child_model = Model_classification( child_model, config )
            
            if args.n_gpu > 1:
                child_model= torch.nn.DataParallel(child_model, device_ids=[0])

            child_model.to(args.device)
            child_metric ,child_perfomance, child_f1_score, child_trainale_rate = fitness(child_model,  args , tokenizer, train_datalaoder, eval_dataloader)
         
            if child_metric == FAIL_COMPILE:
                  continue
            # If not failed, then add to the population
            population.append((child, child_metric , child_perfomance,child_f1_score, child_trainale_rate))
            # and add to the history
            history.append((child,child_metric,child_perfomance, child_f1_score, child_trainale_rate))
            # Remove the oldest model.
            population.popleft()
            # Best model in the current population
            best_candidate = max(list(population), key=lambda i: i[1])
            if best_candidate[1] > best_of_all[1]:
              # Best model during whole calculation
              best_of_all = best_candidate


            logger.info("\n\nCycle {0}, the best candidate in pop : score {1} / Eval acc {2} / f1 score {3} / parameters {4}. And best in the run : score {5} / eval acc {6}/ f1 score {7} / parameters {8} "\
                  .format(len(history) - population_size, best_candidate[1], best_candidate[2], best_candidate[3] ,best_candidate[4], best_of_all[1] ,  best_of_all[2]  ,  best_of_all[3] ,  best_of_all[4]))
            
            #population_scores = [x[1] for x in list(population)]
            #population_mean = np.mean(population_scores)
            #population_std = np.round (np.std(population_scores),3)
         
            #logger.info("Mean {0} and standard deviation {1} of score in the population\n\n".format(population_mean, population_std))



            logger.info ("\n\n Statistics\n\n")

            # Extract fitness scores and other metrics from the current population
            population_fitness = [individual[1] for individual in list(population)]
            population_accuracy = [individual[2] for individual in list(population)]
            population_fscore = [individual[3] for individual in list(population)]
            population_trainable_rate = [individual[4] for individual in list(population)]
            population_configs = [get_config_hash(individual[0]) for individual in list(population)]
            
            # Compute statistics
            best_fitness = np.round(max(population_fitness),3)
            avg_fitness = np.round( np.mean(population_fitness),3)
            std_fitness =  np.round(np.std(population_fitness),3)
            best_accuracy =  np.round(max(population_accuracy),3)
            avg_accuracy =  np.round(np.mean(population_accuracy),3)
            best_fscore =  np.round(max(population_fscore),3)
            avg_fscore =  np.round(np.mean(population_fscore),3)
            avg_trainable_rate =  np.round(np.mean(population_trainable_rate),3)
            
            unique_configs = len(set(population_configs))
            
            # Append to statistics lists
            best_fitness_per_cycle.append(best_fitness)
            avg_fitness_per_cycle.append(avg_fitness)
            std_fitness_per_cycle.append(std_fitness)
            best_accuracy_per_cycle.append(best_accuracy)
            avg_accuracy_per_cycle.append(avg_accuracy)
            best_fscore_per_cycle.append(best_fscore)
            avg_fscore_per_cycle.append(avg_fscore)
            avg_trainable_rate_per_cycle.append(avg_trainable_rate)
            unique_configs_per_cycle.append(unique_configs)
            
            # Logging the statistics
            cycle = len(history) - population_size
            logger.info(f"Cycle {cycle } Statistics:")
            logger.info(f"  Best Fitness: {best_fitness}")
            logger.info(f"  Average Fitness: {avg_fitness}")
            logger.info(f"  Fitness Std Dev: {std_fitness}")
            logger.info(f"  Best accuracy: {best_accuracy}")
            logger.info(f"  Average accuracy: {avg_accuracy}")
            logger.info(f"  Best f1 score: {best_fscore}")
            logger.info(f"  Average f1 score: {avg_fscore}")
            logger.info(f"  Average Trainable Rate: {avg_trainable_rate}")
            logger.info(f"  Unique Configurations: {unique_configs}\n")

    with open(os.path.join(args.output_dir, args.optimization_history_file ),'w') as f:
        for candidate in history:
            f.write(str(candidate)+'\n')


    stats = {
        'best_fitness': best_fitness_per_cycle,
        'avg_fitness': avg_fitness_per_cycle,
        'std_fitness': std_fitness_per_cycle,
        'best_accuracy': best_accuracy_per_cycle,
        'avg_accuracy': avg_accuracy_per_cycle,
        'best_fscore': best_fscore_per_cycle,
        'avg_fscore': avg_fscore_per_cycle,
        'avg_trainable_rate': avg_trainable_rate_per_cycle,
        'unique_configs': unique_configs_per_cycle
                                                        }

    save_statistics(stats, args.output_dir , args.stats_file)
    

    logger.info("Evolutionary Search Completed.\n")
    logger.info(f"Best of all candidates: {best_of_all}\n")
    
    return history, population, best_of_all , stats 







def mutate(adapter_structure , config) : 
    """
    Perform a mutation operation on the adapter structure of a model.

    Args:
        adapter_structure (List[Union[int, Dict[str, Any]]]): A list specifying 
            adapter configurations for each layer. If an element is `0`, it indicates 
            that layer has no adapter; if it is a dictionary, it contains the adapter 
            configuration for that layer.
        config: A model configuration object (e.g., from HuggingFace Transformers)

    Returns:
        List[Union[int, Dict[str, Any]]]: A mutated copy of the original adapter structure.
    """
    parent_hash = hash(tuple ([ dict_to_tuple(d)  if d!=0 else d for d in adapter_structure ]))
    model_len = config.num_hidden_layers-1
    new_structure = deepcopy(adapter_structure)
   
    while True:
        random_or_swap = random.randint(0, 1) # swap if 0 , random otherwise 
        
        if random_or_swap== 1 :
            layer_index = random.randint(0, model_len)
            new_element = random_configuration(config)
            #new_element["non_linearity"] = [ d['non_linearity'] for d in adapter_structure if d!=0][0]
            new_structure[layer_index] = new_element
        else :
            while True : 
                first_layer = random.randint(0 ,model_len )
                second_layer = random.randint(0 , model_len )
                if first_layer == second_layer :
                    continue 
                else :
                    new_structure[second_layer] = adapter_structure[first_layer]
                    new_structure[first_layer] = adapter_structure[second_layer]
        
                    break 
        
        if hash(tuple ([ dict_to_tuple(d) if d!=0 else d for d in new_structure ])) != parent_hash:
            break
        else:
            continue
    return new_structure





def crossover (parent1 , parent2 , config) : 

    """
    Perform a one-point crossover on two 'parent' adapter structures.

    Args:
        parent1 (List[Any]): The adapter configuration for the first parent, 
    
        parent2 (List[Any]): The adapter configuration for the second parent.
        config: A configuration object that should have 'num_hidden_layers' 

    Returns:
        List[Any]: A new adapter configuration (the child) resulting from crossover between parent1 and parent2.
    """
    #one point cross over 
    crossover_point = random.randint(1 , config.num_hidden_layers-2)
    child = []
    child[:crossover_point] = parent1[:crossover_point]
    child[crossover_point:] = parent2[crossover_point:]
    
    return child

  



def fitness(model, args, tokenizer, train_data, eval_data):
    """
    Calculates a 'fitness' value for a given model on a specific task. 
    The fitness is computed as (accuracy + f_score) / 2 - 0.5 * (trainable_rate), 

    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        args: An object containing all relevant hyperparameters and settings,
              including 'task' and 'device'.
        tokenizer: The tokenizer used for processing input text (e.g., RobertaTokenizer).
        train_data: A DataLoader or other dataset structure for training.
        eval_data: A DataLoader or dataset structure for evaluation.

    Returns:
        tuple: A tuple containing:
            - fit (float): The computed fitness metric.
            - accuracy (float): The maximum evaluation accuracy (or MRR).
            - f_score (float): The maximum F1 score (or, for code_search, the same as accuracy).
            - trainable_rate (float): The fraction of parameters that are trainable.
    """

    # Initialize default metrics
    fit = 0
    accuracy = 0
    f_score = 0

    # Compute the fraction of trainable parameters
    trainable_rate = count_rate_trainable_parameters(model)

    # Temporarily set logger level to WARNING to reduce verbosity during training
    logger.setLevel(logging.WARNING)

    try:
        # Evaluate on the specified task
        if args.task == "defect_detection":
            results = train_defect(args, model, tokenizer,train_dataloader=train_data,eval_dataloader_defect=eval_data)
            accuracy = np.round(results['eval_acc'][-1], 3)
            f_score = np.round(results['eval_f1'][-1], 3)

        elif args.task == "clone_detection":
            results = train_clone(args, model, tokenizer , train_dataloader=train_data , eval_dataloader=eval_data)
            accuracy = np.round(results['eval_acc'][-1], 3)
            f_score = np.round(results['eval_f1'][-1], 3)

        elif args.task == "code_search":
            # For code search, we assume MRR is used as the metric
            train_loss, eval_MRR = train_codeSearch(args, model, tokenizer,train_dataloader_code_search=train_data , eval_dataloader_code_search=eval_data)
            accuracy = np.round(eval_MRR[-1], 3)
            f_score = accuracy

        else:
            print("add a valid task name")

    except Exception as e:
        print(e)
        print(traceback.format_exc())

    fit = (accuracy + f_score) / 2 - 0.5 * trainable_rate

    # Restore logger to INFO level
    logger.setLevel(logging.INFO)

    return np.round(fit, 3), accuracy, f_score, trainable_rate
