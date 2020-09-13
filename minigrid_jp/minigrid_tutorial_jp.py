from typing import Dict, Optional, List, Any

import gym
from gym_minigrid.envs import EmptyRandomEnv5x5
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from plugins.minigrid_plugin.minigrid_models import MiniGridSimpleConvRNN
from plugins.minigrid_plugin.minigrid_sensors import EgocentricMiniGridSensor
from plugins.minigrid_plugin.minigrid_tasks import MiniGridTaskSampler, MiniGridTask
from core.algorithms.onpolicy_sync.losses.ppo import PPO, PPOConfig
from core.base_abstractions.experiment_config import ExperimentConfig, TaskSampler
from core.base_abstractions.sensor import SensorSuite
from utils.experiment_utils import TrainingPipeline, Builder, PipelineStage, LinearDecay

class MiniGridTutorialExperimentConfig(ExperimentConfig):

    # [1]
    @classmethod
    def tag(cls) -> str:
        return "MiniGridTutorial_for_Japanese"
    '''実験のタグを設定します（任意）'''
    
    # [2]
    @staticmethod
    def make_env(*args, **kwargs):
        return EmptyRandomEnv5x5()
    '''
    GitHub:gym_minigridの、EmptyRandomEnv5x5()を実験環境とします。
    このタスクは1施行が最大100ステップです。100ステップでゴールでたどり着けないと失敗。
    
    https://github.com/maximecb/gym-minigrid/search?q=EmptyRandomEnv5x5%28%29&unscoped_q=EmptyRandomEnv5x5%28%29
    '''
    
    # [3]
    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return MiniGridTaskSampler(**kwargs)
    '''plugins.minigrid_plugin.minigrid_tasksのMiniGridTaskSamplerを使用します。
    https://github.com/allenai/allenact/blob/master/plugins/minigrid_plugin/minigrid_tasks.py
    '''
    
    # [4] 
    '''TaskSamplerでの1stepでのタスクサンプルの定義です。おまじない的です'''
    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="train")

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="valid")

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="test")

    # [5]
    '''MiniGridTaskSamplerで、呼び出される各タスクのサンプル関数を定義します。
    ■訓練時は後ほど定義する、トータルstep数の間、ずっとstepし続けます（ゴールや失敗時には自動で環境をリセット）。
    そのため、何回タスクが実施されるかは不明です。トータルsteps数を満たすまでタスクを繰り返します。
    ■検証時とテスト時は、途中保存のタイミングで、max_tasks回、タスクを実行しその平均の結果を求めます
    （検証は20回、テスト時は40回）
    '''
    def _get_sampler_args(self, process_ind: int, mode: str) -> Dict[str, Any]:
        """訓練・検証・テストのタスクサンプルの設定を行います。
        # Parameters
        process_ind : index of the current task sampler
        mode:  one of `train`, `valid`, or `test`
        """
        if mode == "train":
            max_tasks = None  # タスクの回数です
            task_seeds_list = None  # no predefined random seeds for training
            deterministic_sampling = False  # randomly sample tasks in training
        else:
            max_tasks = 20 + 20 * (mode == "test")  # タスク回数（検証20回、テスト40回）

            # one seed for each task to sample:
            # - ensures different seeds for each sampler, and
            # - ensures a deterministic set of sampled tasks.
            task_seeds_list = list(
                range(process_ind * max_tasks, (process_ind + 1) * max_tasks)
            )

            deterministic_sampling = (
                True  # deterministically sample task in validation/testing
            )

        return dict(
            max_tasks=max_tasks,  # see above
            # builder for third-party environment (defined below)
            env_class=self.make_env,
            sensors=self.SENSORS,  # sensors used to return observations to the agent
            env_info=dict(),  # parameters for environment builder (none for now)
            task_seeds_list=task_seeds_list,  # see above
            deterministic_sampling=deterministic_sampling,  # see above
        )
    
    
    # [6]
    SENSORS = [
        EgocentricMiniGridSensor(agent_view_size=5, view_channels=3),
    ]

    '''状態観測のSensorsを設定します。このクラスはフォルダpluginsに用意済みです
    from plugins.minigrid_plugin.minigrid_sensors import EgocentricMiniGridSensor
    https://github.com/allenai/allenact/blob/master/plugins/minigrid_plugin/minigrid_sensors.py
    
    view_channels=3は、各迷路のタイルの状態を設定します。
    3の場合は、
    - タイルのタイプ（壁や床やドアやゴールなど11種類）
    - タイルの色（赤や緑など6種類）
    - タイルの状態（open, closed, lockedの3種類）です。
    ※詳細は本notebookの最後記載します。
    
    
    agent_view_size=5は、エージェントが観測できる範囲（横・縦）の長さを示します。
    agent_view_size=5だと、こんな観測範囲です。
    ＊ ＊ ＊ ＊ ＊
    ＊ ＊ ＊ ＊ ＊
    ＊ ＊  A ＊ ＊
    ＊ ＊ ＊ ＊ ＊
    ＊ ＊ ＊ ＊ ＊
    
    ただし、この範囲のうち、自分（Actor）が向いている真横より前のみ見えます(部分観測)。 
    上記でAが上を向いていると、以下の観測範囲となります。ただし、壁などが途中にあると、その先は観測できません。
    
    ＊ ＊ ＊ ＊ ＊
    ＊ ＊ ＊ ＊ ＊
    ＊ ＊  A ＊ ＊
    ｘ ｘ ｘ ｘ ｘ
    ｘ ｘ ｘ ｘ ｘ
    
    なお、行動（Action）は、なお、エージェントができる行動（Action）は、"left", "right", "forward"で、
    それぞれ、「その場で左に向く」、「その場で右に向く」、「1歩前に進む」の3タイプです。
    '''

    # [7]
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return MiniGridSimpleConvRNN(
            action_space=gym.spaces.Discrete(
                len(MiniGridTask.class_action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            num_objects=cls.SENSORS[0].num_objects,
            num_colors=cls.SENSORS[0].num_colors,
            num_states=cls.SENSORS[0].num_states,
        )
    '''AllenActに RNNActorCritic()が用意されていて、これを迷路用にチューニングしています。
    https://github.com/allenai/allenact/blob/master/plugins/minigrid_plugin/minigrid_models.py#L141
    
    RNNActorCritic()はリカレントニューラルネットワークRNNを使用したメモリを搭載したタイプのActor-Criticのディープラーニングモデルです。
    RNN型のメモリを使用することで、今回の迷路課題のように部分観測であり、過去の自分の状態と観測した状態の情報を保持できるようにしています。
    '''

   # [8] 
    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> Dict[str, Any]:
        return {
            "nprocesses": 128 if mode == "train" else 16,
            "gpu_ids": [],
        }
    '''
    使用するサブプロセスの数を定義。
    訓練時は128個、検証とテスト時は16個とします。
    gpu_idsのリストを空にして、GPUを使用せず、cpuのみを使用するように定義しています。
    '''
    
    
    # [9]
    '''深層強化学習の訓練・検証・テストのパイプラインを組み立てます。
    今回はPPOを使用します。ここで種々、PPOの設定を与えます。
    ネットワークパラメータの最適化にはAdamを使用し、学習率を線形に小さくしてきます。
    '''
    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        ppo_steps = int(150000)
        return TrainingPipeline(
            named_losses=dict(ppo_loss=PPO(**PPOConfig)),  # type:ignore
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"],
                              max_stage_steps=ppo_steps)
            ],
            optimizer_builder=Builder(optim.Adam, dict(lr=1e-4)),
            num_mini_batch=4,
            update_repeats=3,
            max_grad_norm=0.5,
            num_steps=16,
            gamma=0.99,
            use_gae=True,
            gae_lambda=0.95,
            advance_scene_rollout_period=None,
            save_interval=10000,  # 約1万stepに1度、検証、もしくはテストを実施します
            metric_accumulate_interval=1,
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(
                    steps=ppo_steps)}  # type:ignore
            ),
        )
class MiniGridTutorialExperimentConfig(ExperimentConfig):

    # [1]
    @classmethod
    def tag(cls) -> str:
        return "MiniGridTutorial_for_Japanese"
    '''実験のタグを設定します（任意）'''
    
    # [2]
    @staticmethod
    def make_env(*args, **kwargs):
        return EmptyRandomEnv5x5()
    '''
    GitHub:gym_minigridの、EmptyRandomEnv5x5()を実験環境とします。
    このタスクは1施行が最大100ステップです。100ステップでゴールでたどり着けないと失敗。
    
    https://github.com/maximecb/gym-minigrid/search?q=EmptyRandomEnv5x5%28%29&unscoped_q=EmptyRandomEnv5x5%28%29
    '''
    
    # [3]
    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return MiniGridTaskSampler(**kwargs)
    '''plugins.minigrid_plugin.minigrid_tasksのMiniGridTaskSamplerを使用します。
    https://github.com/allenai/allenact/blob/master/plugins/minigrid_plugin/minigrid_tasks.py
    '''
    
    # [4] 
    '''TaskSamplerでの1stepでのタスクサンプルの定義です。おまじない的です'''
    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="train")

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="valid")

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="test")

    # [5]
    '''MiniGridTaskSamplerで、呼び出される各タスクのサンプル関数を定義します。
    ■訓練時は後ほど定義する、トータルstep数の間、ずっとstepし続けます（ゴールや失敗時には自動で環境をリセット）。
    そのため、何回タスクが実施されるかは不明です。トータルsteps数を満たすまでタスクを繰り返します。
    ■検証時とテスト時は、途中保存のタイミングで、max_tasks回、タスクを実行しその平均の結果を求めます
    （検証は20回、テスト時は40回）
    '''
    def _get_sampler_args(self, process_ind: int, mode: str) -> Dict[str, Any]:
        """訓練・検証・テストのタスクサンプルの設定を行います。
        # Parameters
        process_ind : index of the current task sampler
        mode:  one of `train`, `valid`, or `test`
        """
        if mode == "train":
            max_tasks = None  # タスクの回数です
            task_seeds_list = None  # no predefined random seeds for training
            deterministic_sampling = False  # randomly sample tasks in training
        else:
            max_tasks = 20 + 20 * (mode == "test")  # タスク回数（検証20回、テスト40回）

            # one seed for each task to sample:
            # - ensures different seeds for each sampler, and
            # - ensures a deterministic set of sampled tasks.
            task_seeds_list = list(
                range(process_ind * max_tasks, (process_ind + 1) * max_tasks)
            )

            deterministic_sampling = (
                True  # deterministically sample task in validation/testing
            )

        return dict(
            max_tasks=max_tasks,  # see above
            # builder for third-party environment (defined below)
            env_class=self.make_env,
            sensors=self.SENSORS,  # sensors used to return observations to the agent
            env_info=dict(),  # parameters for environment builder (none for now)
            task_seeds_list=task_seeds_list,  # see above
            deterministic_sampling=deterministic_sampling,  # see above
        )
    
    
    # [6]
    SENSORS = [
        EgocentricMiniGridSensor(agent_view_size=5, view_channels=3),
    ]

    '''状態観測のSensorsを設定します。このクラスはフォルダpluginsに用意済みです
    from plugins.minigrid_plugin.minigrid_sensors import EgocentricMiniGridSensor
    https://github.com/allenai/allenact/blob/master/plugins/minigrid_plugin/minigrid_sensors.py
    
    view_channels=3は、各迷路のタイルの状態を設定します。
    3の場合は、
    - タイルのタイプ（壁や床やドアやゴールなど11種類）
    - タイルの色（赤や緑など6種類）
    - タイルの状態（open, closed, lockedの3種類）です。
    ※詳細は本notebookの最後記載します。
    
    
    agent_view_size=5は、エージェントが観測できる範囲（横・縦）の長さを示します。
    agent_view_size=5だと、こんな観測範囲です。
    ＊ ＊ ＊ ＊ ＊
    ＊ ＊ ＊ ＊ ＊
    ＊ ＊  A ＊ ＊
    ＊ ＊ ＊ ＊ ＊
    ＊ ＊ ＊ ＊ ＊
    
    ただし、この範囲のうち、自分（Actor）が向いている真横より前のみ見えます(部分観測)。 
    上記でAが上を向いていると、以下の観測範囲となります。ただし、壁などが途中にあると、その先は観測できません。
    
    ＊ ＊ ＊ ＊ ＊
    ＊ ＊ ＊ ＊ ＊
    ＊ ＊  A ＊ ＊
    ｘ ｘ ｘ ｘ ｘ
    ｘ ｘ ｘ ｘ ｘ
    
    なお、行動（Action）は、なお、エージェントができる行動（Action）は、"left", "right", "forward"で、
    それぞれ、「その場で左に向く」、「その場で右に向く」、「1歩前に進む」の3タイプです。
    '''

    # [7]
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return MiniGridSimpleConvRNN(
            action_space=gym.spaces.Discrete(
                len(MiniGridTask.class_action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            num_objects=cls.SENSORS[0].num_objects,
            num_colors=cls.SENSORS[0].num_colors,
            num_states=cls.SENSORS[0].num_states,
        )
    '''AllenActに RNNActorCritic()が用意されていて、これを迷路用にチューニングしています。
    https://github.com/allenai/allenact/blob/master/plugins/minigrid_plugin/minigrid_models.py#L141
    
    RNNActorCritic()はリカレントニューラルネットワークRNNを使用したメモリを搭載したタイプのActor-Criticのディープラーニングモデルです。
    RNN型のメモリを使用することで、今回の迷路課題のように部分観測であり、過去の自分の状態と観測した状態の情報を保持できるようにしています。
    '''

   # [8] 
    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> Dict[str, Any]:
        return {
            "nprocesses": 128 if mode == "train" else 16,
            "gpu_ids": [],
        }
    '''
    使用するサブプロセスの数を定義。
    訓練時は128個、検証とテスト時は16個とします。
    gpu_idsのリストを空にして、GPUを使用せず、cpuのみを使用するように定義しています。
    '''
    
    
    # [9]
    '''深層強化学習の訓練・検証・テストのパイプラインを組み立てます。
    今回はPPOを使用します。ここで種々、PPOの設定を与えます。
    ネットワークパラメータの最適化にはAdamを使用し、学習率を線形に小さくしてきます。
    '''
    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        ppo_steps = int(150000)
        return TrainingPipeline(
            named_losses=dict(ppo_loss=PPO(**PPOConfig)),  # type:ignore
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"],
                              max_stage_steps=ppo_steps)
            ],
            optimizer_builder=Builder(optim.Adam, dict(lr=1e-4)),
            num_mini_batch=4,
            update_repeats=3,
            max_grad_norm=0.5,
            num_steps=16,
            gamma=0.99,
            use_gae=True,
            gae_lambda=0.95,
            advance_scene_rollout_period=None,
            save_interval=10000,  # 約1万stepに1度、検証、もしくはテストを実施します
            metric_accumulate_interval=1,
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(
                    steps=ppo_steps)}  # type:ignore
            ),
        )
