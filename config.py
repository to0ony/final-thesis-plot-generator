# -------------------------
# Model Configuration
# -------------------------

# Model architecture hyperparameters
BLOCK_SIZE = 256
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
MODEL_TYPE = None

# Training hyperparameters
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
MAX_ITERS = 8000
EVAL_INTERVAL = 400
LEARNING_RATE = 3e-4
EVAL_ITERS = 50
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 500

def get_model_config(vocab_size):
    """
    Kreira model konfiguraciju
    """
    from mingpt.model import GPT
    
    config = GPT.get_default_config()
    config.vocab_size = vocab_size
    config.block_size = BLOCK_SIZE
    config.n_layer = N_LAYER
    config.n_head = N_HEAD
    config.n_embd = N_EMBD
    config.model_type = MODEL_TYPE
    
    return config
