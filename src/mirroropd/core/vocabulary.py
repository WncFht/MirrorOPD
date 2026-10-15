"""Shared token identities and exclusion of padded, unsampleable output rows."""

from mirroropd.identity import content_id


def align_logits(logits, vocab_size):
    if type(vocab_size) is not int or not 0 < vocab_size <= logits.shape[-1]:
        raise ValueError("Legal vocabulary must fit the model output head")
    return logits[..., :vocab_size]


def validate_tokenizers(student, teacher):
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("Distillation requires identical token-to-ID semantics")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(student, name) != getattr(teacher, name):
            raise ValueError(f"Tokenizer {name} differs between student and teacher")
    if not student.chat_template or student.chat_template != teacher.chat_template:
        raise ValueError("A shared chat template is required; configure it explicitly when absent")
    return content_id(
        {
            "vocabulary": student.get_vocab(),
            "bos": student.bos_token_id,
            "eos": student.eos_token_id,
            "pad": student.pad_token_id,
            "template": student.chat_template,
        }
    )
