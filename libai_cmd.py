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


'''