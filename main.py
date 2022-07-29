import gym
from envs.GuardedMaze import GuardedMaze
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback

# env = gym.make("CartPole-v1")
# env = GuardedMaze()
# env.seed(0)


# game_id = 'Seaquest-v4'
game_id = 'Breakout-v4'
# game_id = 'MsPacman-v4'
env = make_atari_env(game_id, n_envs=10, seed=3)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)
# env = VecVideoRecorder(env, f"./videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0)

# wrapper_kwargs = {'terminal_on_life_loss':False}
eval_env = make_atari_env(game_id, n_envs=1, seed=3)#, wrapper_kwargs=wrapper_kwargs)
eval_env = VecFrameStack(eval_env, n_stack=4)







total_time = 1_000_000


run = wandb.init(
    project="conformal_dqn",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
)
wandb.run.name = wandb.run.id + '_vanila_'  + game_id
wandb.run.save()
wandb_callback = WandbCallback()
eval_callback = EvalCallback(eval_env, eval_freq=1000,
                             deterministic=True, render=False,n_eval_episodes=50)
callbacks = [eval_callback, wandb_callback]
model = DQN("CnnPolicy", env, train_freq=500, gradient_steps=500, verbose=1, conformal=False, buffer_size=1_000_000, tensorboard_log='./dqn_tensorboard', real_return=False)
model.learn(total_timesteps=total_time, callback=callbacks, tb_log_name='dqn_vanila')
model.save("./models/dqn_vanila")
del model
run.finish()
#



run = wandb.init(
    project="conformal_dqn",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
)
wandb.run.name = wandb.run.id + '_aci_'  + game_id
wandb.run.save()
wandb_callback = WandbCallback()
eval_callback = EvalCallback(eval_env, eval_freq=1000,
                             deterministic=True, render=False,n_eval_episodes=50, robust=False)
callbacks = [eval_callback, wandb_callback]

model = DQN("CnnConformalDQN",
            env,
            train_freq=500,
            gradient_steps=500, verbose=1,
            conformal=True,
            adaptive_conformal=True,
            buffer_size=1_000_000,
            # exploration_initial_eps=0, #no need for eps-greedy(?)
            # policy_kwargs={'robust_policy':True},
            # policy_kwargs={'ofu_policy':True},
            tensorboard_log='./dqn_tensorboard',
            real_return=False)
# model = DQN("ConformalDQN", env, train_freq=1000, gradient_steps=100, verbose=1, conformal=True,adaptive_conformal=True, policy_kwargs={'robust_policy':True})
model.learn(total_timesteps=total_time, callback=callbacks, tb_log_name='dqn_aci')
model.save("./models/dqn_aci")
del model
run.finish()



run = wandb.init(
    project="conformal_dqn",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
)
wandb.run.name = wandb.run.id + '_aci_'  + game_id
wandb.run.save()
wandb_callback = WandbCallback()
eval_callback = EvalCallback(eval_env, eval_freq=1000,
                             deterministic=True, render=False,n_eval_episodes=50, robust=False)
callbacks = [eval_callback, wandb_callback]

model = DQN("CnnConformalDQN",
            env,
            train_freq=500,
            gradient_steps=500, verbose=1,
            conformal=True,
            adaptive_conformal=True,
            buffer_size=1_000_000,
            # exploration_initial_eps=0, #no need for eps-greedy(?)
            policy_kwargs={'robust_policy':True},
            # policy_kwargs={'ofu_policy':True},
            tensorboard_log='./dqn_tensorboard',
            real_return=False)
# model = DQN("ConformalDQN", env, train_freq=1000, gradient_steps=100, verbose=1, conformal=True,adaptive_conformal=True, policy_kwargs={'robust_policy':True})
model.learn(total_timesteps=total_time, callback=callbacks, tb_log_name='dqn_aci_robust')
model.save("./models/dqn_aci_robust")
del model
run.finish()



run = wandb.init(
    project="conformal_dqn",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
)
wandb.run.name = wandb.run.id + '_ci_'  + game_id
wandb.run.save()
wandb_callback = WandbCallback()
eval_callback = EvalCallback(eval_env, eval_freq=1000,
                             deterministic=True, render=False,n_eval_episodes=50, robust=False)
callbacks = [eval_callback, wandb_callback]
model = DQN("CnnConformalDQN",
            env,
            train_freq=500,
            gradient_steps=500, verbose=1,
            conformal=True,
            adaptive_conformal=False,
            buffer_size=1_000_000,
            # exploration_initial_eps=0, #no need for eps-greedy(?)
            # policy_kwargs={'ofu_policy':True},
            tensorboard_log='./dqn_tensorboard',
            real_return = False)
# model = DQN("ConformalDQN", env, train_freq=1000, gradient_steps=100, verbose=1, conformal=True,adaptive_conformal=True, policy_kwargs={'robust_policy':True})
model.learn(total_timesteps=total_time, callback=callbacks, tb_log_name='dqn_ci')
model.save("./models/dqn_ci")
del model
run.finish()
#
# # model = DQN("CnnConformalDQN",
# #             env,
# #             train_freq=500,
# #             gradient_steps=100, verbose=1,
# #             conformal=True,
# #             adaptive_conformal=False,
# #             reset_calib_buffer=True,
# #             buffer_size=1_000_000,
# #             policy_kwargs={'ofu_policy':True},
# #             tensorboard_log='./dqn_tensorboard')
# # # model = DQN("ConformalDQN", env, train_freq=1000, gradient_steps=100, verbose=1, conformal=True,adaptive_conformal=True, policy_kwargs={'robust_policy':True})
# # model.learn(total_timesteps=total_time, callback=callbacks, tb_log_name='dqn_ci_reset')
# # model.save("./models/dqn_ci_w_reset")
#
#
