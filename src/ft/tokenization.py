UNK_TOKEN = "<unk>"


def train_bpe_tokenizer(text: str, vocab_size: int) -> dict:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=[UNK_TOKEN])

    # Train BPE only on the training split to avoid leaking validation/test text.
    tokenizer.train_from_iterator([text], trainer)
    return {
        "type": "bpe",
        "vocab_size_target": vocab_size,
        "unk_token": UNK_TOKEN,
        "tokenizer_json": tokenizer.to_str(),
    }


def load_bpe_tokenizer(tokenizer_meta: dict):
    from tokenizers import Tokenizer

    if tokenizer_meta.get("type") != "bpe":
        raise ValueError("Expected BPE tokenizer metadata.")
    return Tokenizer.from_str(tokenizer_meta["tokenizer_json"])


def encode_text(text: str, tokenizer_meta: dict) -> list[int]:
    tokenizer = load_bpe_tokenizer(tokenizer_meta)
    return tokenizer.encode(text).ids


def decode_tokens(token_ids: list[int], tokenizer_meta: dict) -> str:
    tokenizer = load_bpe_tokenizer(tokenizer_meta)
    return tokenizer.decode([int(token_id) for token_id in token_ids])


def tokenizer_vocab_size(tokenizer_meta: dict) -> int:
    return load_bpe_tokenizer(tokenizer_meta).get_vocab_size()
