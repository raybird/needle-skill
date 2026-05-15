import sys
import traceback

_MODEL = None
_PARAMS = None
_TOKENIZER = None


def _ensure_needle():
    try:
        import needle
    except ImportError:
        print("Needle is not installed. Run 'needle-skill setup' first.", file=sys.stderr)
        sys.exit(1)


def load_model(checkpoint_path):
    global _MODEL, _PARAMS, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _PARAMS, _TOKENIZER

    _ensure_needle()
    from needle.model.run import load_checkpoint
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.dataset.dataset import get_tokenizer

    print(f"Loading model from {checkpoint_path}...", file=sys.stderr)
    _PARAMS, config = load_checkpoint(checkpoint_path)
    _MODEL = SimpleAttentionNetwork(config)
    _TOKENIZER = get_tokenizer()
    param_count = sum(x.size for x in __import__("jax").tree.leaves(_PARAMS))
    print(f"Model loaded: {param_count:,} parameters", file=sys.stderr)


def generate(query, tools, max_gen_len=512, seed=0, constrained=False):
    _ensure_needle()
    from needle.model.run import generate as needle_generate

    if _MODEL is None:
        print("Model not loaded. Call load_model() first.", file=sys.stderr)
        sys.exit(1)

    return needle_generate(
        _MODEL, _PARAMS, _TOKENIZER,
        query, tools=tools,
        max_gen_len=max_gen_len, seed=seed,
        stream=False, constrained=constrained,
    )


def is_loaded():
    return _MODEL is not None


def unload():
    global _MODEL, _PARAMS, _TOKENIZER
    _MODEL = None
    _PARAMS = None
    _TOKENIZER = None