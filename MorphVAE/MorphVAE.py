# This file defines the training process of MorphVAE
# For replication, please first follow the instructions on evolutiongym.github.io/ to properly install evogym,
#  and then place our codes where appropriate. 

import os, time, numpy as np, shutil, random, math, torch, sys

# set directories
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo.run import run_ppo
from ppo.arguments import get_args
from evogym import sample_robot, hashable, BASELINE_ENV_NAMES
from evogym.utils import is_connected, has_actuator, get_full_connectivity
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure
from VAE import VAE
from vec2morph import vec_to_morph, morph_to_vec, operator
import pyro
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    JitTraceEnum_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    config_enumerate,
)
from pyro.optim import Adam

def run_vae(experiment_name, structure_shape, pop_size, max_evaluations, train_iters, num_cores, args, generations):
    
    # =======================================================================================
    # 1. Initialize the VAE model
    # =======================================================================================
    model = VAE(y_size=args.num_tasks, x_size=args.width,
                h_size=args.num_robo_types,
                hidden_layers_g = args.hidden_layers_g,
                hidden_layers_p = args.hidden_layers_p, y_emb=128, h_emb=128, 
                config_enum="parallel") # VAE形态模型的初始化
    guide = config_enumerate(model.guide, expand=True) # 后验分布 
    operator_2 = operator(args.width) 
    # ========================================================================================
    # =======================================================================================
    
    # initialize a dictionary used for storing fitness scores 
    #  (to calculate sample probabilities in continuous natural selection)
    fitnesses = {}
    ids = {}
    for k in range(args.num_tasks):
        fitnesses[k] = []
        ids[k] = []
    
    random_prop = 0  # sample robots with VAE, rather than the built-in function in EvoGym
    
    for generation in range(generations):
        torch.cuda.empty_cache()
        
        ### STARTUP: MANAGE DIRECTORIES ###
        home_path = os.path.join(root_dir, "saved_data", experiment_name)
        start_gen = 0

        ### DEFINE TERMINATION CONDITION ###    
        tc = TerminationCondition(train_iters)
        is_continuing = False  

        try:
            os.makedirs(home_path)
        except:
            if generation == 0:
                print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
                print("Override? (y/n/c): ", end="")
                ans = input()
                if ans.lower() == "y":
                    shutil.rmtree(home_path)
                    print()
                elif ans.lower() == "c":
                    print("Enter gen to start training on (0-indexed): ", end="")
                    start_gen = int(input())
                    is_continuing = True
                    print()
                else:
                    return
            else:
                pass


        ### STORE META-DATA ##
        if not is_continuing:
            temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
            vae_loss_path_sup = os.path.join(root_dir, "saved_data", experiment_name, "vae_loss_sup.txt")
            vae_loss_path_unsup = os.path.join(root_dir, "saved_data", experiment_name, "vae_loss_unsup.txt")
            valid_num_path = os.path.join(root_dir, "saved_data", experiment_name, "valid_num.txt")
            
            try:
                os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))
            except:
                pass

            f = open(temp_path, "w")
            f.write(f'POP_SIZE: {pop_size}\n')
            f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
            f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
            f.write(f'TRAIN_ITERS: {train_iters}\n')

            f.write(f'NUM_TASKS: {args.num_tasks}\n')
            f.write(f'NUM_ROBO_TYPES: {args.num_robo_types}\n')
            f.write(f'NUM_ORGAN_TYPES: {args.num_organs}\n')
            f.write(f'NUM_VOXEL_TYPES: {args.num_voxels}\n')
            f.close()

        else:
            pass

        ### UPDATE NUM SURVIORS ###		
        num_evaluations	= pop_size * generation # number of evaluated robot designs
        survivor_rate = 1 # keep all robot designs in the morphology pool
        num_survivors = math.ceil(survivor_rate*pop_size)
        explore = math.ceil(pop_size*args.explore_rate*(generations-generation-1)/(generations-1))
        
        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "structure")
        save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller")
        save_path_organ = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "organ")
        save_path_type = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation),"type")
        
        try:
            os.makedirs(save_path_structure)
            os.makedirs(save_path_controller)
            os.makedirs(save_path_organ)
            os.makedirs(save_path_type)
        except:
            pass
        
        structures = []
        robo_type_dist = {}
        nums_robots = {}
        nums_robots_valid = {}
        
        for task_id in range(args.num_tasks):
            nums_robots[task_id] = 0
            nums_robots_valid[task_id] = 0
        
        for task_id in range(args.num_tasks):
            population_structure_hashes = {}
            tasks = torch.zeros((pop_size, args.num_tasks))
            tasks[:,task_id] = 1
           
            num_random_samples = 0
            num_explorations = 0
            num_exploitations = 0
            
            while len(population_structure_hashes) != pop_size and nums_robots[task_id]<400: 
                # set an upper bound for the number of repeated attempts of VAE in exploration-exploitation rebalancing
                if num_random_samples<random_prop*pop_size: # random sample
                    robos, _ = sample_robot((args.width,args.width))
                    robos = torch.tensor([robos])
                    _,_,robo_type = model.guide(xs=morph_to_vec(robos).float(), ys=tasks[0:1,:].float(), 
                                                hs=None) # infer about latents
                    num_random_samples += 1
                    flag = "ran"
                    
    # =======================================================================================
    #    2. generate morphologies using VAE
    # =======================================================================================
                else:
                    robos_vec,_,robo_type,_ = model.model(xs=None, ys=tasks, hs=None)
                    robo_type = robo_type.detach().numpy()
                    nums_robots[task_id] += tasks.shape[0]
                    if generation>0:
                        sims = torch.mm(robos_vec, all_bodies[task_id].T)/25 # 相同体素的比例
                    robos = vec_to_morph(robos_vec, operator_2, args.width)
                    #flag = "vae"
                    
                    
                ## keep record of the proportion of valid robots
                for robo_idx in range(robos.shape[0]):
                    temp_robo = robos[robo_idx,:,:]
                    temp_robo = temp_robo.numpy()
                    if is_connected(temp_robo) and has_actuator(temp_robo):
                        nums_robots_valid[task_id] += 1
                
                print(robos.shape[0])
                for robo_idx in range(robos.shape[0]):
                    if len(population_structure_hashes) == pop_size:
                            break
                    temp_robo = robos[robo_idx,:,:]
                    temp_robo = temp_robo.numpy()
                    
                    if generation>0 and is_connected(temp_robo) and has_actuator(temp_robo) and hashable(temp_robo) not in population_structure_hashes: 
                        # exploration-exploitation rebalancing (except for the first generation)
                        if (sims[robo_idx,:]>=args.sim_thr).sum()==0 and num_explorations<explore: # exploration
                            flag = "ore"
                            temp_structure = (temp_robo, get_full_connectivity(np.array(temp_robo)))
                            structures.append(Structure(*temp_structure, str(len(population_structure_hashes))+flag, task_id=task_id))
                            robo_type_dist[str(task_id)+"-"+str(len(population_structure_hashes))] = robo_type[robo_idx]
                            population_structure_hashes[hashable(temp_structure[0])] = True
                            num_explorations += 1
                        if (sims[robo_idx,:]>=args.sim_thr).sum()!=0 and num_exploitations<pop_size-explore: # exploitation
                            flag = "oit"
                            temp_structure = (temp_robo, get_full_connectivity(np.array(temp_robo)))
                            structures.append(Structure(*temp_structure, str(len(population_structure_hashes))+flag, task_id=task_id))
                            robo_type_dist[str(task_id)+"-"+str(len(population_structure_hashes))] = robo_type[robo_idx]
                            population_structure_hashes[hashable(temp_structure[0])] = True
                            num_exploitations += 1
                    if generation==0 and is_connected(temp_robo) and has_actuator(temp_robo) and hashable(temp_robo) not in population_structure_hashes: 
                        # in first generation, directly generate
                        flag = "vae"
                        temp_structure = (temp_robo, get_full_connectivity(np.array(temp_robo)))
                        structures.append(Structure(*temp_structure, str(len(population_structure_hashes))+flag, task_id=task_id))
                        robo_type_dist[str(task_id)+"-"+str(len(population_structure_hashes))] = robo_type[robo_idx]
                        population_structure_hashes[hashable(temp_structure[0])] = True
                        
                        
            if len(population_structure_hashes) != pop_size and num_explorations<explore:
                # upper bound of attempts met, but not enough exploration samples: use the built-in sampling function in EvoGym
                for iii in range(explore - num_explorations):
                    robos, _ = sample_robot((args.width,args.width))
                    robos = torch.tensor([robos])
                    _,_,robo_type = model.guide(xs=morph_to_vec(robos).float(), ys=tasks[0:1,:].float(), 
                                                hs=None) # infer about latents
                    robo_type = robo_type.detach().numpy()
                    flag = "ran"
                    for robo_idx in range(robos.shape[0]):
                        temp_robo = robos[robo_idx,:,:]
                        temp_robo = temp_robo.numpy()
                        temp_structure = (temp_robo, get_full_connectivity(np.array(temp_robo)))
                        structures.append(Structure(*temp_structure, str(len(population_structure_hashes))+flag, task_id=task_id))
                        robo_type_dist[str(task_id)+"-"+str(len(population_structure_hashes))] = robo_type[robo_idx]
                        population_structure_hashes[hashable(temp_structure[0])] = True
                        
                        
            if len(population_structure_hashes) != pop_size and num_exploitations<pop_size-explore:
                # upper bound of attempts met, but not enough exploitation samples: directly generate with VAE
                while len(population_structure_hashes) != pop_size:
                    robos_vec,_,robo_type,_ = model.model(xs=None, ys=tasks, hs=None)
                    robo_type = robo_type.detach().numpy()
                    robos = vec_to_morph(robos_vec, operator_2, args.width)
                    flag = "vae"
                    for robo_idx in range(robos.shape[0]):
                        if len(population_structure_hashes) == pop_size:
                            break
                        temp_robo = robos[robo_idx,:,:]
                        temp_robo = temp_robo.numpy()
                        if is_connected(temp_robo) and has_actuator(temp_robo) and hashable(temp_robo) not in population_structure_hashes: 
                            temp_structure = (temp_robo, get_full_connectivity(np.array(temp_robo)))
                            structures.append(Structure(*temp_structure, str(len(population_structure_hashes))+flag, task_id=task_id))
                            robo_type_dist[str(task_id)+"-"+str(len(population_structure_hashes))] = robo_type[robo_idx]
                            population_structure_hashes[hashable(temp_structure[0])] = True
    # =======================================================================================
    # =======================================================================================
        
        
        ### SAVE POPULATION DATA ###
        # save valid proportion
        for task_id in range(args.num_tasks):
            f = open(valid_num_path, "a")
            out = ""
            out += str(generation) + "\t\t" + str(task_id) + "\t\t" + str(nums_robots[task_id]) + "\t\t" + str(nums_robots_valid[task_id]) + "\t\t" + str(nums_robots_valid[task_id]/nums_robots[task_id]) + "\n"
            f.write(out)
            f.close()
        
        # save robot designs
        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, "task-{}_struct-{}".format(structures[i].task_id, structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)
            temp_path = os.path.join(save_path_type,"task-{}_struct-{}".format(structures[i].task_id, structures[i].label))
            np.savez(temp_path, robo_type_dist[str(structures[i].task_id) + "-" + str(structures[i].label[:-3])])

    # =======================================================================================
    # 3. robot evaluation and saving 
    # =======================================================================================
        #better parallel
        group = mp.Group()
        for structure in structures:
            ppo_args = (int(structure.label[:-3])%4, args, task_id_name_mapper[structure.task_id], structure, tc, (save_path_controller, structure.label))
            group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
        group.run_jobs(num_cores) 

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()

        tasks_structures = []
        for task_id in range(args.num_tasks):
            task_structures = []
            for robo_idx in range(pop_size):
                task_structures.append(structures[task_id * pop_size + robo_idx]) 
            task_structures = sorted(task_structures, key=lambda task_structure: task_structure.fitness, reverse=True) 
            tasks_structures.append(task_structures)

        #SAVE RANKING TO FILE
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "output"+str(generation)+".txt")
        f = open(temp_path, "w")

        out = ""
        for task_structures in tasks_structures:
            for task_structure in task_structures:
                out += str(task_structure.task_id) + "\t\t" + str(task_structure.label) + "\t\t" + str(task_structure.fitness) + "\n"
        f.write(out)
        f.close()

        ### CHECK EARLY TERMINATION ###
        if num_evaluations >= max_evaluations:
            print(f'Trained exactly {num_evaluations} robots')
            return

        ### SAVE DESIGNS AND FITNESS INTO THE MORPHOLOGY POOL ###
        tasks_survivors = []
        for k, task_structures in enumerate(tasks_structures):
            task_survivors = task_structures[:num_survivors]
            tasks_survivors.append(task_survivors)
            fitnesses[k] += [kk.fitness for kk in task_survivors]
            ids[k] += [str(generation)+" "+str(kk.label)[:-3] for kk in task_survivors]
        
        survivor_bodies = []
        survivor_robo_type = []
        for task_survivors in tasks_survivors:
            for survivor in task_survivors:
                survivor_bodies.append(np.array(survivor.body))
                survivor_robo_type.append(np.array(robo_type_dist[str(survivor.task_id)+"-"+str(survivor.label)[:-3]]))
        survivor_bodies = torch.tensor(np.array(survivor_bodies))
        survivor_robo_type = torch.tensor(np.array(survivor_robo_type))
        
        survivor_tasks = [torch.randint(1,(1,num_survivors)) + task_id for task_id in range(args.num_tasks)]
        survivor_tasks = torch.hstack(survivor_tasks) 
        survivor_tasks_vec = []
        for t in survivor_tasks.squeeze():
            survivor_task = [0]*args.num_tasks
            survivor_task[t] = 1
            survivor_tasks_vec.append(survivor_task)
        survivor_tasks = torch.tensor(survivor_tasks_vec)
        
        
        if generation == 0:
            all_bodies = {}
            all_tasks = {}
            for k in range(args.num_tasks):
                all_bodies[k] = morph_to_vec(survivor_bodies)[(pop_size*k):(pop_size*(k+1)),:].float()
                all_tasks[k] = survivor_tasks[(pop_size*k):(pop_size*(k+1)),:]
        else:
            for k in range(args.num_tasks):
                all_bodies[k] = torch.cat((all_bodies[k], morph_to_vec(survivor_bodies)[(pop_size*k):(pop_size*(k+1)),:].float()), dim=0)
                all_tasks[k] = torch.cat((all_tasks[k], survivor_tasks[(pop_size*k):(pop_size*(k+1)),:]), dim=0)
        
    # =======================================================================================
    # =======================================================================================
        
        
    # =======================================================================================
    # 4. update VAE through continuous natural selection
    # =======================================================================================
        
        # sample_size = 50 
        sample_size = min(int(len(fitnesses[0])*0.5), 50)
        sample_probs = {}
        for k in range(args.num_tasks):
            sample_prob = np.exp(np.array(fitnesses[k])*args.tau) / np.exp(np.array(fitnesses[k])*args.tau).sum()
            sample_probs[k] = sample_prob
      
        print("Updating VAE...")
        
        num_iters_vae = int(((generations-generation-1)/(generations-1))*(args.num_iters_vae_low-args.num_iters_vae_up)+args.num_iters_vae_up)
        for step in range(num_iters_vae): # number of updates
            
            for k in range(args.num_tasks):
                sample_ids = np.random.choice(range(pop_size*(generation+1)), sample_size, replace=True, p=sample_probs[k])
                if k==0:
                    sample_bodies = all_bodies[k][sample_ids,:]
                    sample_tasks = all_tasks[k][sample_ids,:]
                else:
                    sample_bodies = torch.cat((sample_bodies, all_bodies[k][sample_ids,:]), dim=0)
                    sample_tasks = torch.cat((sample_tasks, all_tasks[k][sample_ids,:]), dim=0)
            
            loss_fn = lambda model, guide: TraceEnum_ELBO().differentiable_loss(model,guide,sample_bodies.float(),sample_tasks.float(),None)
            with pyro.poutine.trace(param_only=True) as param_capture:
                loss = loss_fn(model.model,guide)
            params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())
            if generation==0:
                optimizer_model = torch.optim.Adam(params, lr=args.vae_lr, betas=args.vae_betas)
            
            loss_pos = loss.detach().numpy().item()
            
            loss.backward()
            optimizer_model.step()
            optimizer_model.zero_grad()
           
            f = open(vae_loss_path_unsup, "a")

            out = ""
            out += str(generation) + "\t\t" + str(step) + "\t\t" + "loss" + "\t\t" + str(loss_pos) + "\n"
            f.write(out)

            f.close()
            
        # save VAE parameters after update in each generation
        params_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "mymodelparams.pt")
        pyro.get_param_store().save(params_path)
        
        torch.save(model.emb_h, os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "emb_h.pt"))
        torch.save(model.emb_y, os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "emb_y.pt"))
        
        print("Updating finished! ")
    # =======================================================================================
    # =======================================================================================

    
    
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    seed = 0 # 1,2
    random.seed(seed)
    np.random.seed(seed)

    structure_shape=(5, 5)
    pop_size = 25
    
    max_evaluations=1e100
    train_iters = 1000
    num_cores = 15 
    generations = 40

    env_tasks = ["Walker-v0", "Climber-v0", "UpStepper-v0", "Catcher-v0", "Carrier-v0", "Pusher-v0"]
    task_name_id_mapper = {}
    task_id_name_mapper = {}
    for env_task in env_tasks:
        task_name_id_mapper[env_task] = len(task_name_id_mapper) 
        task_id_name_mapper[task_name_id_mapper[env_task]] = env_task 

    args = get_args() 
    args.width = structure_shape[0]
    args.num_tasks = len(env_tasks)
    args.num_voxels = 5

    args.hidden_layers_g = [128, 128, 128]
    args.hidden_layers_p = [128 ,128, 128]
   
    # number of updates of VAE in each generation linearly increases from low to up
    args.num_iters_vae_low = 50 
    args.num_iters_vae_up = 250 
    
    args.num_processes = 4
    args.num_steps = 128
    args.vae_lr = 0.0001
    args.vae_betas = (0.95,0.999)
    
    # Two variants: MorphVAE-H and -L
    args.tau = 1.5 # 0.7 
    args.sim_thr = 0 # 0.75
    args.explore = 0
    args.explore_rate = 0 # 0.5
    
    args.no_cuda = False
    args.cuda = True
    
    experiment_name = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    run_vae(experiment_name, structure_shape, pop_size, max_evaluations, train_iters, num_cores, 
        args=args, generations=generations)