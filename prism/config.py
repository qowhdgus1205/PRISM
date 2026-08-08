# config.py

# Configuration for the SimCLR encoder and Target MLP training.

# General settings
DEVICE = "auto"
SEED = 42
OUTPUT_DIR = "./runs/default"
DATA_PATH = str(__import__('pathlib').Path(__file__).parent.parent.parent / "data" / "ldc2025_train.csv")
SPLIT_DIR = str(__import__('pathlib').Path(__file__).parent.parent.parent / "splits")
USE_FIXED_SPLIT = True

# Column names
INPUT_COLS = ['N', 'p_int', 'EGR', 'T_int', 'p_diff', 'FM', 'EAR', 'IT']
ACP_COLS = ['MFB10', 'MFB50', 'MFB90', 'p_max']
TARGET_COLS = ['eta_i', 'vol_eff', 'T_exh', 'IMEP']

# SimCLR/contrastive encoder settings
LATENT_DIM = 128
PROJ_DIM = 128
EPOCHS_CONTRASTIVE = 1000
BATCH_SIZE_CONTRASTIVE = 128
LR_CONTRASTIVE = 1e-4
ACP_LAMBDA = 0.4
TEMPERATURE = 0.5
ALPHA = 0.4  # Mixup strength (0.4: less aggressive, cleaner interpolation)
BASE_WEIGHT = 1.0
MIX_WEIGHT = 0.5  # Reduced: mixed samples are regularizer, not equal-weight signal

# MLP/regressor settings
MLP_EPOCHS = 1000
MLP_LR = 5e-4
PATIENCE = 500
BATCH_SIZE = 128
VAL_RATIO = 0.1
TEST_RATIO = 0.1
FEASIBLE_STAGE1_MIXUP_ALPHA = 0.0  # >0 applies supervised mixup to X -> ACP_hat training

# Gaussian Noise Augmentation settings (weak / medium / strong 3단계)
USE_GAUSSIAN_NOISE_AUGMENTATION = True
NOISE_STD_WEAK   = 0.005   # view1: 약한 노이즈, dropout 없음
NOISE_STD_MED    = 0.01    # view2: 중간 노이즈, 약한 dropout
NOISE_STD_STRONG = 0.02    # view3: 강한 노이즈, feature dropout
GAUSSIAN_NOISE_STD = 0.01  # backward compat
SIMCLR_VIEW_NAMES = ["weak", "medium"]  # default to 2-view; add "strong" for ablations

# ACP Auxiliary Regression (latent → ACP 직접 예측으로 강한 supervision)
ACP_REG_LAMBDA = 0.5       # auxiliary ACP regression loss 가중치
ACP_HEAD_LOSS = "smooth_l1"  # "mse" or "smooth_l1"
ACP_HEAD_HUBER_BETA = 0.5

# Derived ACP Features settings
# ACPHead가 raw ACP 4개만 예측하도록 False로 설정
# (파생피처 포함 시 ACPHead 목표가 복잡해져 latent space 붕괴)
USE_DERIVED_ACP_FEATURES = False

# ACP similarity settings
ACP_SIM_USE_STANDARDIZATION = True   # z-score ACP targets before similarity / ACPHead supervision
ACP_SIGMA_MODE = "global_median"     # "global_median", "batch_median", or "fixed"
ACP_SIGMA = None                     # used when ACP_SIGMA_MODE == "fixed"
ACP_SIGMA_MAX_SAMPLES = 2048         # subsample size for global sigma estimation

# Physics-informed contrastive: ACP k-NN positive pairs
USE_ACP_POSITIVES = True    # use ACP-neighbor positives by default
ACP_K_NEIGHBORS = 3         # number of ACP nearest neighbors to add as positives

# Physics-aware mixup settings
MIXUP_MODE = "random"       # "random": uniform permutation (더 다양한 mixing 신호)
LOCAL_MIXUP_K = 3           # (local 모드일 때만 사용)
LOCAL_MIXUP_PRESERVE_ANCHOR = False  # lambda 편향 제거: symmetric Beta 그대로 사용
MIXUP_SCHEDULE = True        # True: mix_weight를 코사인으로 감쇠 (초반 강, 후반 약)
MIXUP_WEIGHT_MAX = 0.5       # 초반 mix_weight
MIXUP_WEIGHT_MIN = 0.05      # 후반 mix_weight

# Structural ablation flags (각 개선 요소 개별 on/off)
USE_LAYER_NORM_ENCODER = True   # True: encoder hidden LayerNorm / False: BatchNorm (원래 방식)
USE_LR_SCHEDULE = True          # True: cosine annealing + warmup / False: 고정 LR
USE_ACP_ZSCORE = True           # True: kNN 거리에 Z-score 표준화 / False: F.normalize (원래 방식)

# Physics-informed regression: monotonicity constraint
USE_MONO_LOSS = False        # set True to enable monotonicity penalty
MONO_LAMBDA = 0.05           # weight for monotonicity loss

# Inference/encoding settings
ENCODE_BATCH_SIZE = 4096

# ---------------------------------------------------------------------------
# Stage 2 mode
# ---------------------------------------------------------------------------
# "finetune"        : X → [pretrained encoder, unfrozen] → z → TargetMLP → Y  ← 기본값 (원래 의도)
# "z_only"          : X → [frozen encoder] → z → TargetMLP → Y               ← ablation
# "xz"              : X → [frozen encoder] → z, [X,z] → TargetMLP → Y        ← ablation
# "simclr_feasible"    : X → [frozen encoder] → acp_head(z) → [X,ACP_hat] → Y     ← ablation
# "simclr_xz_feasible" : X → [frozen encoder] → z, acp_head(z), [X,z,ACP_hat] → Y ← hybrid
STAGE2_MODE = "finetune"

# Fine-tuning: encoder gets lower LR than head (standard transfer learning)
FINETUNE_ENCODER_LR_SCALE = 0.1  # encoder_lr = MLP_LR * FINETUNE_ENCODER_LR_SCALE

STAGE2_HEAD = "mlp"  # "mlp" | "thin" | "linear"
STAGE2_ACP_AUX_LAMBDA = 0.0  # >0 enables ACP auxiliary loss during Stage-2 fine-tuning

# ---------------------------------------------------------------------------
# Pluggable backbone selection
# ---------------------------------------------------------------------------
ENCODER_BACKBONE = "mlp"   # "mlp" | "cnn" | "ftt"

# FT-Transformer hyperparams (used when ENCODER_BACKBONE == "ftt")
FT_DIM          = 64
FT_DEPTH        = 6
FT_HEADS        = 8
FT_ATTN_DROPOUT = 0.1
FT_FF_DROPOUT   = 0.1
