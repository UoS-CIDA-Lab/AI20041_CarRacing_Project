from dataclasses import dataclass
import importlib
from typing import Any
from .달려라_하니 import racing_with_out_transformer as 달려라하니
from .F1_1024 import module as F1_1024
from .U1 import DQN_upgrade as U1
from .U2 import main as U2
from .G1 import train as G1
from .G2 import main as G2
from .김성민 import train as 김성민

@dataclass
class submission:
    model: type[object]
    model_path: str
    hyperparameters: dict[str, Any]

submission_달려라하니 = submission(달려라하니.DQN, "./submissions/달려라_하니/model_weights_760_Run_HANI.pth", {
        "batch_size":64,        # 배치 크기를 32로 설정
		"eps_start":1.0,        # 탐험 비율을 높게 시작
		"eps_end":0.1,          # 탐험 비율의 하한값을 조금 높임
		"eps_decay":1000,       # 탐험 비율 감소 속도를 느리게 설정
		"gamma":0.98,            # 감쇠 계수 증가로 보상의 미래 중요성을 조금 더 강조
        "lr":0.001})

submission_F1_1024 = submission(F1_1024.DQN, "./submissions/F1_1024/model_weights_903.pth",{
    })
submission_U1 = submission(U1.DQN, "./submissions/U1/model_weights_1000.pth", {
    })

submission_U2 = submission(U2.DQN, "./submissions/U2/model_weights_370.pth", {
        })

submission_G1 = submission(G1.DQN, "./submissions/G1/model_weights_1000.pth", {
        })

submission_G2 = submission(G2.DQN, "./submissions/G2/model_weights_3000.pth", {
        })

submission_김성민 = submission(김성민.DQN, "./submissions/김성민/model_weights_1000.pth", {
        })

dict_reappearance = {
    "달려라하니":submission_달려라하니,
    "F1_1024":submission_F1_1024,
    "U1":submission_U1,
    "U2":submission_U2,
    "G1":submission_G1,
    "G2":submission_G2,
    "김성민":submission_김성민,
}



