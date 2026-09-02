'''


python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 50000


python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --num_envs 1


/home/libai/08_amp/legged_lab/logs/rsl_rl/g1_amp/2026-08-11_17-41-02/model_17400.pt

python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-Play-v0 --num_envs 16 --checkpoint /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_amp/2026-08-11_17-41-02/model_17400.pt


+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                       Available Environments in Isaac Lab LeggedLab Extension                                                                        |
+--------+---------------------------------------------------+------------------------------------------+----------------------------------------------------------------------------------------------+
| S. No. | Task Name                                         | Entry Point                              | Config                                                                                       |
+--------+---------------------------------------------------+------------------------------------------+----------------------------------------------------------------------------------------------+
|   1    | LeggedLab-Isaac-AMP-Flat-Atom01-v0                | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.atom01.amp_flat_env_cfg:Atom01AmpFlatEnvCfg           |
|   2    | LeggedLab-Isaac-AMP-Flat-Atom01-Play-v0           | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.atom01.amp_flat_env_cfg:Atom01AmpFlatEnvCfg_PLAY      |
|   3    | LeggedLab-Isaac-AMP-G1-v0                         | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfg                         |
|   4    | LeggedLab-Isaac-AMP-G1-Play-v0                    | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfg_PLAY                    |
|   5    | LeggedLab-Isaac--Deepmimic-G1-v0                  | legged_lab.envs:ManagerBasedAnimationEnv | legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg       |
|   6    | LeggedLab-Isaac--Deepmimic-G1-Play-v0             | legged_lab.envs:ManagerBasedAnimationEnv | legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg_PLAY  |
|   7    | LeggedLab-Isaac--Deepmimic-G1-Debug-v0            | legged_lab.envs:ManagerBasedAnimationEnv | legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg_DEBUG |
|   8    | LeggedLab-Isaac-Velocity-Flat-Unitree-Go2-v0      | isaaclab.envs:ManagerBasedRLEnv          | legged_lab.tasks.locomotion.velocity.config.go2.flat_env_cfg:Go2FlatEnvCfg                   |
|   9    | LeggedLab-Isaac-Velocity-Flat-Unitree-Go2-Play-v0 | isaaclab.envs:ManagerBasedRLEnv          | legged_lab.tasks.locomotion.velocity.config.go2.flat_env_cfg:Go2FlatEnvCfg_PLAY              |
+--------+---------------------------------------------------+------------------------------------------+----------------------------------------------------------------------------------------------+



python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 50000


python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --num_envs 1

python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-v0 \
  --headless \
  --resume \
  --load_run 2026-08-11_17-41-02 \
  --checkpoint model_17400.pt \
  --max_iterations 32600

  
cd 00_isaaclab/IsaacLab


python scripts/tools/convert_urdf.py /home/libai/08_amp/unitree_ros/robots/g1_description/g1_29dof.urdf /home/libai/08_amp/unitree_ros/robots/g1_description/usd/g1_29dof.usd --joint-target-type position



+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                       Available Environments in Isaac Lab LeggedLab Extension                                                                        |
+--------+---------------------------------------------------+------------------------------------------+----------------------------------------------------------------------------------------------+
| S. No. | Task Name                                         | Entry Point                              | Config                                                                                       |
+--------+---------------------------------------------------+------------------------------------------+----------------------------------------------------------------------------------------------+
|   1    | LeggedLab-Isaac-AMP-Flat-Atom01-v0                | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.atom01.amp_flat_env_cfg:Atom01AmpFlatEnvCfg           |
|   2    | LeggedLab-Isaac-AMP-Flat-Atom01-Play-v0           | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.atom01.amp_flat_env_cfg:Atom01AmpFlatEnvCfg_PLAY      |
|   3    | LeggedLab-Isaac-AMP-G1-v0                         | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfg                         |
|   4    | LeggedLab-Isaac-AMP-G1-v1                         | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfgV1                       |
|   5    | LeggedLab-Isaac-AMP-G1-Play-v0                    | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfg_PLAY                    |
|   6    | LeggedLab-Isaac-AMP-G1-Play-v1                    | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfg_PLAY_V1                 |
|   7    | LeggedLab-Isaac--Deepmimic-G1-v0                  | legged_lab.envs:ManagerBasedAnimationEnv | legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg       |
|   8    | LeggedLab-Isaac--Deepmimic-G1-Play-v0             | legged_lab.envs:ManagerBasedAnimationEnv | legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg_PLAY  |
|   9    | LeggedLab-Isaac--Deepmimic-G1-Debug-v0            | legged_lab.envs:ManagerBasedAnimationEnv | legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg_DEBUG |
|   10   | LeggedLab-Isaac-Velocity-Flat-Unitree-Go2-v0      | isaaclab.envs:ManagerBasedRLEnv          | legged_lab.tasks.locomotion.velocity.config.go2.flat_env_cfg:Go2FlatEnvCfg                   |
|   11   | LeggedLab-Isaac-Velocity-Flat-Unitree-Go2-Play-v0 | isaaclab.envs:ManagerBasedRLEnv          | legged_lab.tasks.locomotion.velocity.config.go2.flat_env_cfg:Go2FlatEnvCfg_PLAY              |
+--------+---------------------------------------------------+------------------------------------------+----------------------------------------------------------------------------------------------+


python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v1 --num_envs 1

python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v1 --headless --max_iterations 50000 --num_envs 6000


+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                        Available Environments in Isaac Lab LeggedLab Extension                                                                        |
+--------+---------------------------------------------------+------------------------------------------+-----------------------------------------------------------------------------------------------+
| S. No. | Task Name                                         | Entry Point                              | Config                                                                                        |
+--------+---------------------------------------------------+------------------------------------------+-----------------------------------------------------------------------------------------------+
|   1    | LeggedLab-Isaac-AMP-Flat-Atom01-v0                | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.atom01.amp_flat_env_cfg:Atom01AmpFlatEnvCfg            |
|   2    | LeggedLab-Isaac-AMP-Flat-Atom01-Play-v0           | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.atom01.amp_flat_env_cfg:Atom01AmpFlatEnvCfg_PLAY       |
|   3    | LeggedLab-Isaac-AMP-G1-v0                         | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfg                          |
|   4    | LeggedLab-Isaac-AMP-G1-v1                         | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfgV1                        |
|   5    | LeggedLab-Isaac-AMP-G1-Play-v0                    | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfg_PLAY                     |
|   6    | LeggedLab-Isaac-AMP-G1-Play-v1                    | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfg_PLAY_V1                  |
|   7    | LeggedLab-Isaac-AMP-G1-assist-v0                  | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1_assist.g1_assist_amp_env_cfg:G1AssistAmpEnvCfg      |
|   8    | LeggedLab-Isaac-AMP-G1-assist-Play-v0             | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1_assist.g1_assist_amp_env_cfg:G1AssistAmpEnvCfg_PLAY |
|   9    | LeggedLab-Isaac--Deepmimic-G1-v0                  | legged_lab.envs:ManagerBasedAnimationEnv | legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg        |
|   10   | LeggedLab-Isaac--Deepmimic-G1-Play-v0             | legged_lab.envs:ManagerBasedAnimationEnv | legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg_PLAY   |
|   11   | LeggedLab-Isaac--Deepmimic-G1-Debug-v0            | legged_lab.envs:ManagerBasedAnimationEnv | legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg_DEBUG  |
|   12   | LeggedLab-Isaac-Velocity-Flat-Unitree-Go2-v0      | isaaclab.envs:ManagerBasedRLEnv          | legged_lab.tasks.locomotion.velocity.config.go2.flat_env_cfg:Go2FlatEnvCfg                    |
|   13   | LeggedLab-Isaac-Velocity-Flat-Unitree-Go2-Play-v0 | isaaclab.envs:ManagerBasedRLEnv          | legged_lab.tasks.locomotion.velocity.config.go2.flat_env_cfg:Go2FlatEnvCfg_PLAY               |
+--------+---------------------------------------------------+------------------------------------------+-----------------------------------------------------------------------------------------------+



python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-assist-v0 --num_envs 1

python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-assist-v0 --headless --max_iterations 50000 --num_envs 6000


python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-assist-Play-v0 --num_envs 16 --checkpoint /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_amp/2026-08-13_13-45-16/model_29400.pt



帮我搭建任务LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v0和LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v0
1. 目标：训练外骨骼辅助g1关节hip_pitch_joint运动,即外骨骼hip_pitch_assist_joint输出扭矩来辅助hip_pitch_joint运动
2. 仅使用三层PPO进行训练，输入：assist_joint的关节角度、角速度、输入扭矩的0.5秒的历史值，输出2个assist_joint的关节扭矩，输入-》256 -》64 -》 16 -》输出
3. 机器人除assist关节，其他关节驱动参考程序/home/libai/08_amp/legged_lab/scripts/rsl_rl/play_g1_assist_exoskeleton.py,使得机器人正常行走
4. 奖励设置： 4.1 assist_joint 输出平滑 4.2 期望assist_joint输出扭矩与hip_pitch_joint输出扭矩同向且不超过hip_pitch_joint输出扭矩 4.3 终止惩罚
5. 终止指令：5.1 参考任务LeggedLab-Isaac-AMP-G1-assist-v0，引入不带外骨骼的终止条件 5.2 加入assist_joint与hip_pitch_joint相差超过4度则终止(穿模)
6. 6.1 设置 sim_dt为0.001,(更好的碰撞处理) 6.2 decimation 为 20 6.3 episode_length_s 20.0 
全部新建文件，不要影响其他任务。若有其它以为疑问，请向我提问


+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                                                      Available Environments in Isaac Lab LeggedLab Extension                                                                                      |
+--------+---------------------------------------------------+------------------------------------------+---------------------------------------------------------------------------------------------------------------------------+
| S. No. | Task Name                                         | Entry Point                              | Config                                                                                                                    |
+--------+---------------------------------------------------+------------------------------------------+---------------------------------------------------------------------------------------------------------------------------+
|   1    | LeggedLab-Isaac-AMP-Flat-Atom01-v0                | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.atom01.amp_flat_env_cfg:Atom01AmpFlatEnvCfg                                        |
|   2    | LeggedLab-Isaac-AMP-Flat-Atom01-Play-v0           | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.atom01.amp_flat_env_cfg:Atom01AmpFlatEnvCfg_PLAY                                   |
|   3    | LeggedLab-Isaac-AMP-G1-v0                         | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfg                                                      |
|   4    | LeggedLab-Isaac-AMP-G1-v1                         | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfgV1                                                    |
|   5    | LeggedLab-Isaac-AMP-G1-Play-v0                    | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfg_PLAY                                                 |
|   6    | LeggedLab-Isaac-AMP-G1-Play-v1                    | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg:G1AmpEnvCfg_PLAY_V1                                              |
|   7    | LeggedLab-Isaac-AMP-G1-assist-v0                  | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1_assist.g1_assist_amp_env_cfg:G1AssistAmpEnvCfg                                  |
|   8    | LeggedLab-Isaac-AMP-G1-assist-Play-v0             | legged_lab.envs:ManagerBasedAmpEnv       | legged_lab.tasks.locomotion.amp.config.g1_assist.g1_assist_amp_env_cfg:G1AssistAmpEnvCfg_PLAY                             |
|   9    | LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v0      | isaaclab.envs:ManagerBasedRLEnv          | legged_lab.tasks.locomotion.amp.config.g1_assist_exoskeleton.g1_assist_exoskeleton_env_cfg:G1AssistExoskeletonEnvCfg      |
|   10   | LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v0 | isaaclab.envs:ManagerBasedRLEnv          | legged_lab.tasks.locomotion.amp.config.g1_assist_exoskeleton.g1_assist_exoskeleton_env_cfg:G1AssistExoskeletonEnvCfg_PLAY |
|   11   | LeggedLab-Isaac--Deepmimic-G1-v0                  | legged_lab.envs:ManagerBasedAnimationEnv | legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg                                    |
|   12   | LeggedLab-Isaac--Deepmimic-G1-Play-v0             | legged_lab.envs:ManagerBasedAnimationEnv | legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg_PLAY                               |
|   13   | LeggedLab-Isaac--Deepmimic-G1-Debug-v0            | legged_lab.envs:ManagerBasedAnimationEnv | legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg_DEBUG                              |
|   14   | LeggedLab-Isaac-Velocity-Flat-Unitree-Go2-v0      | isaaclab.envs:ManagerBasedRLEnv          | legged_lab.tasks.locomotion.velocity.config.go2.flat_env_cfg:Go2FlatEnvCfg                                                |
|   15   | LeggedLab-Isaac-Velocity-Flat-Unitree-Go2-Play-v0 | isaaclab.envs:ManagerBasedRLEnv          | legged_lab.tasks.locomotion.velocity.config.go2.flat_env_cfg:Go2FlatEnvCfg_PLAY                                           |
+--------+---------------------------------------------------+------------------------------------------+---------------------------------------------------------------------------------------------------------------------------+


python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v0 --num_envs 1


python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v0 --headless --max_iterations 5000 --num_envs 6000


python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v0 --num_envs 16 --checkpoint /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_ppo/2026-08-14_11-06-58/model_1000.pt


参考程序/home/libai/06_assist_test/isaaclab/assist/scripts/rsl_rl/play_assist.py ,写一个任务LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v0的单个机器人play,并画图


固定vx 0.7 vy 0.0 yaw 0.0 
python scripts/rsl_rl/play_g1_assist_exoskeleton_plot.py \
  --checkpoint logs/rsl_rl/g1_assist_exoskeleton_ppo/2026-08-14_11-06-58/model_1000.pt \
  --real-time


python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v0 --num_envs 16 --checkpoint /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_ppo/2026-08-14_14-00-08/model_2000.pt


python scripts/mujoco/sim2sim_g1_assist_exoskeleton.py \
  --assist-policy /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_ppo/2026-08-14_14-00-08/exported/policy.pt

  
python scripts/mujoco/sim2sim_g1_assist_exoskeleton.py   --assist-policy /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_ppo/2026-08-14_14-00-08/exported/policy.pt

运动时还可以，但站立时有点力气互怼


python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v0 \
  --headless \
  --resume \
  --load_run 2026-08-14_14-00-08 \
  --checkpoint model_2000.pt \
  --max_iterations 2000 \
  --num_envs 6000


python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v0 \
  --headless \
  --num_envs 6000 \
  --max_iterations 5000 \
  --resume \
  --load_run 2026-08-14_14-00-08 \
  --checkpoint model_2000.pt

  

python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v0 --num_envs 16 --checkpoint /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_ppo/2026-08-17_10-05-41/model_4400.pt

python scripts/mujoco/sim2sim_g1_assist_exoskeleton.py \
  --assist-policy /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_ppo/2026-08-17_10-05-41/exported/policy.pt


##
新建任务  LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v1 和  LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v1



  python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v1 \
  --headless \
  --max_iterations 5000 \
  --num_envs 6000

本设备采用强化学习作为主体，搭建人形机器人训练框架，采集丰富的人行走轨迹，进行动作重映射，使用模仿学习。使得人形机器人行走模型接近于人体。
再采用“冻结行走策略+独立外骨骼策略”的分层结构，采用减少人形机器人能耗的训练策略，实现本外骨骼助力策略。

python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v1 --num_envs 16 --checkpoint /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_v1_ppo/2026-08-18_13-27-09/model_4999.pt

 python scripts/rsl_rl/play_g1_assist_exoskeleton_plot.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v1 \
  --checkpoint /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_v1_ppo/2026-08-18_13-27-09/model_4999.pt \
  --real-time \
  'env.viewer.eye=[4.0,4.0,4.0]' \
  'env.viewer.lookat=[0.0,0.0,1.0]'



  env.viewer.origin_type=asset_root \
  env.viewer.asset_name=robot \
  env.viewer.env_index=0 \
  'env.viewer.eye=[3.0,-3.0,1.5]' \
  'env.viewer.lookat=[0.0,0.0,0.8]'


  env.viewer.origin_type=world \
  'env.viewer.eye=[4.0,4.0,4.0]' \
  'env.viewer.lookat=[0.0,0.0,1.0]'

 env.viewer.origin_type=asset_root \
  env.viewer.asset_name=robot \
  env.viewer.env_index=0 \
  'env.viewer.eye=[3.0,-3.0,1.5]' \
  'env.viewer.lookat=[0.0,0.0,0.8]'

env.viewer.origin_type=asset_body \
env.viewer.asset_name=robot \
env.viewer.body_name=torso_link \
'env.viewer.eye=[3.0,-3.0,1.0]' \
'env.viewer.lookat=[0.0,0.0,0.0]'




  python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2 \
  --num_envs 8000 \
  --headless



  python scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v2 \
  --checkpoint logs/rsl_rl/g1_assist_exoskeleton_v2_ppo/2026-08-31_10-43-42/model_2000.pt \
  --num_envs 16

  



python scripts/mujoco/sim2sim_g1_assist_exoskeleton_2.py



调整任务 LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2

obs 加入角度
reward 

  python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2-v1 \
  --num_envs 8000 \
  --headless

  


conda run --no-capture-output -n env_isaaclab_2 \
python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2 \
  --num_envs 4096 \
  --headless \
  --max_iterations 2000



python scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v2 \
  --checkpoint /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_v2_ppo/2026-09-01_10-02-55/model_1999.pt \
  --num_envs 16


python scripts/mujoco/sim2sim_g1_assist_exoskeleton_2.py



python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2 \
  --headless \
  --num_envs 6000 \
  --max_iterations 3000 \
  --resume \
  --load_run 2026-09-01_10-02-55 \
  --checkpoint model_1999.pt

  
python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2 \
  --headless \
  --num_envs 6000 \
  --max_iterations 2000 \
  --run_name dynamics_tracking_w8


python scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2.py \
  --assist-policy /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_v2_ppo/2026-09-02_10-00-19_dynamics_tracking_w8/exported/policy.pt \
  --vx 0.7 \
  --output-dir /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_v2_ppo/2026-09-02_10-00-19_dynamics_tracking_w8/sim2sim_model_1999
  

# 用途：在 MuJoCo Sim2Sim 中完全关闭左右髋部外骨骼助力，以 0 Nm 作为
# assist actuator 的输出，测试无助力时机器人在零速度指令下的表现。
# 步态策略及后部滑块、圆柱机构的零位 PD 控制仍然启用。
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
python scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2.py \
  --disable-assist \
  --vx 0.0 \
  --vy 0.0 \
  --yaw 0.0 \
  --output-dir /home/libai/08_amp/legged_lab/logs/sim2sim_analysis_exoskeleton_2_no_assist


python scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2.py \
  --assist-policy /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_v2_ppo/2026-09-02_10-00-19_dynamics_tracking_w8/exported/policy.pt \
  --disable-assist \
  --vx 0.7 \
  --output-dir /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_v2_ppo/2026-09-02_10-00-19_dynamics_tracking_w8/sim2sim_model_1999
  

# 用途：在 MuJoCo Sim2Sim 中测试最终助力执行器输入系数。以下示例仅在
# data.ctrl 最终写入时把助力扭矩乘以 0.5；策略推理、obs、扭矩历史记录和
# 40 Nm/s 目标平滑过程均不变，最终执行器输出仍受正负 8 Nm 限幅约束。
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
python scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2.py \
  --assist-policy /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_v2_ppo/2026-09-02_10-00-19_dynamics_tracking_w8/exported/policy.pt \
  --assist-torque-scale 3.0 \
  --vx 0.7 \
  --vy 0.0 \
  --yaw 0.0 \
  --output-dir /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_exoskeleton_v2_ppo/2026-09-02_10-00-19_dynamics_tracking_w8/sim2sim_torque_scale_1p0
  

# 用途：为当前 model_1999 生成 v2 Sim2Real 的 NumPy 权重和固定参数 C99
# 源码，并依次验证 TorchScript→NumPy、NumPy→C 的数值一致性。
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
python sim2real/g1_assist_exoskeleton_v2/export_assist_policy_numpy.py

conda run --no-capture-output -n env_isaaclab_2 \
python sim2real/g1_assist_exoskeleton_v2/test_numpy_assist_policy.py

conda run --no-capture-output -n env_isaaclab_2 \
python sim2real/g1_assist_exoskeleton_v2/export_assist_policy_c.py

conda run --no-capture-output -n env_isaaclab_2 \
python sim2real/g1_assist_exoskeleton_v2/test_c_assist_policy.py


# 用途：在 MuJoCo 中使用纯 NumPy v2 助力策略进行部署前验证；步态策略仍使用
# TorchScript。最终电机输入系数设为 0.5，policy、obs、历史和平滑过程不缩放。
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
python sim2real/g1_assist_exoskeleton_v2/sim2sim_numpy_assist.py \
  --assist-torque-scale 0.5 \
  --vx 0.7 \
  --vy 0.0 \
  --yaw 0.0 \
  --output-dir /home/libai/08_amp/legged_lab/sim2real/g1_assist_exoskeleton_v2/output

'''
