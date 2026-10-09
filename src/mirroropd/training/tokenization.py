"""Validate the shared-prefix assumption of the compact teacher cache."""


def assert_shared_tokenization(student, teacher, *, require_chat=False):
    vocab = student.get_vocab()
    other = teacher.get_vocab()
    mismatches = [token for token, index in vocab.items() if other.get(token) != index]
    if vocab != other:
        raise ValueError(
            "Shared-prefix MirrorOPD requires identical token-to-ID semantics; "
            f"mismatched student tokens: {mismatches[:12]}"
        )
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(student, field) != getattr(teacher, field):
            raise ValueError(f"Shared-prefix MirrorOPD requires matching {field}")
    if require_chat:
        if not student.chat_template or not teacher.chat_template:
            raise ValueError("Native chat setting requires both chat templates")
        if student.chat_template != teacher.chat_template:
            raise ValueError("Shared-prefix MirrorOPD requires matching native chat templates")
        for messages in (
            [{"role": "user", "content": "Calculate 3 + 5."}],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Calculate 7 - 2."},
            ],
        ):
            kwargs = {"tokenize": True, "add_generation_prompt": True}
            if student.apply_chat_template(messages, **kwargs) != teacher.apply_chat_template(
                messages, **kwargs
            ):
                raise ValueError("Native templates produce different prompt IDs")
    return {"shared_vocab": True, "shared_special_ids": True, "native_chat_checked": require_chat}
