
td3_bo_batch = {
    'name' : 'Experiment 1: BO Batch',
    'data_folder' : 'data_exp_td3_bo_batch',
    'iterations_init' : 300,
    'iterations' : 100,
    'design_cycles' : 55,
    'state_batch_size' : 32,
    'initial_episodes' : 3,
    'use_gpu' : True,
    'use_cpu_for_rollout': False,
    'cuda_device': 0,
    'exploration_strategy': 'random',
    'design_optim_method' : 'bo_batch',
    'save_networks' : True,
    'rl_method' : 'TD3',
    'steps_per_episodes': 1000,
    'rl_algorithm_config' : dict(
        algo_params_pop=dict(
            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,
            discount=0.99,
            reward_scale=1.0,
            policy_learning_rate=3e-4,
            qf_learning_rate=3e-4,
            policy_and_target_update_period=2,
            tau=0.005,
            alpha=2.5,
            ),
        algo_params_adapt=dict(
            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,
            discount=0.99,
            reward_scale=1.0,
            policy_learning_rate=3e-4,
            qf_learning_rate=3e-4,
            policy_and_target_update_period=2,
            tau=0.005,
            alpha=0.4,
            Kp=3e-5,
            Kd=8e-5,
            target_reward=10000, ),

        net_size=200,
        network_depth=4,
        copy_from_global=True,
        indiv_updates=1000,
        pop_updates=250,
        batch_size=256,
    ),
    'env' : dict(
        env_name='HalfCheetah_mujoco',  # HalfCheetah, HalfCheetah_mujoco, Ant_mujoco
        render=False,
        record_video=False,
    ),
    'seed': 1029,

    'pso_params': dict(
        n_particles=700,
        iters=250,),

    'bo_params': dict(
        init_points = 30,
        n_iter = 30,
    )
}

td3_bo_sim = {
    'name' : 'Experiment 2: BO using Simulations',
    'data_folder' : 'data_exp_td3_bo_sim',
    'iterations_init' : 300,
    'iterations' : 100,
    'design_cycles' : 55,
    'state_batch_size' : 32,
    'initial_episodes' : 3,
    'use_gpu' : True,
    'use_cpu_for_rollout': False,
    'cuda_device': 0,
    'exploration_strategy': 'random',
    'design_optim_method' : 'bo_sim',
    'save_networks' : True,
    'rl_method' : 'TD3',
    'steps_per_episodes': 1000,
    'rl_algorithm_config' : dict(
        algo_params=dict(
            target_policy_noise = 0.2,
            target_policy_noise_clip = 0.5,
            discount = 0.99,
            reward_scale = 1.0,
            policy_learning_rate = 3e-4,
            qf_learning_rate = 3e-4,
            policy_and_target_update_period = 2,
            tau = 0.005,
            ),

        algo_params_pop=dict(
            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,
            discount=0.99,
            reward_scale=1.0,
            policy_learning_rate=3e-4,
            qf_learning_rate=3e-4,
            policy_and_target_update_period=2,
            tau=0.005,
            alpha=2.5,
            ),

        algo_params_adapt=dict(
            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,
            discount=0.99,
            reward_scale=1.0,
            policy_learning_rate=3e-4,
            qf_learning_rate=3e-4,
            policy_and_target_update_period=2,
            tau=0.005,
            alpha=0.4,
            Kp=3e-5,
            Kd=8e-5,
            target_reward=10000, ),

        net_size=200,
        network_depth=4,
        copy_from_global=True,
        indiv_updates=5,
        pop_updates=5,
        batch_size=256,
    ),
    'env' : dict(
        env_name='HalfCheetah_mujoco',  # HalfCheetah, HalfCheetah_mujoco
        render=False,
        record_video=False,
    ),
    'seed': 1029,

    'pso_params': dict(
        n_particles=70,
        iters = 60,),

    'bo_params': dict(
        init_points=30,
        n_iter=30,
    )}

config_dict = {
    'td3_bo_batch': td3_bo_batch,
    'td3_bo_sim': td3_bo_sim,}
