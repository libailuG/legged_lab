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


'''