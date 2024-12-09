from dataclasses import dataclass
import importlib
from typing import Any
from .달려라_하니 import racing_with_out_transformer as 달려라하니

@dataclass
class submission:
    model: type[object]
    model_path: str
    memory: type[object]
    hyperparameters: dict[str, Any]

submission_달려라하니 = submission(달려라하니.DQN, r"submissions\달려라_하니\model_weights_760_Run_HANI.pth", 달려라하니.ReplayMemory, {
        "batch_size":64,        # 배치 크기를 32로 설정
		"eps_start":1.0,        # 탐험 비율을 높게 시작
		"eps_end":0.1,          # 탐험 비율의 하한값을 조금 높임
		"eps_decay":1000,       # 탐험 비율 감소 속도를 느리게 설정
		"gamma":0.98,            # 감쇠 계수 증가로 보상의 미래 중요성을 조금 더 강조
        "lr":0.001})

dict_reappearance = {
    "달려라하니":submission_달려라하니
}

