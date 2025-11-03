## Hugging Face Workflows

### Tokenization Best Practices

#### Chat Template Tokenization
**CRITICAL**: When tokenizing strings that have already been generated from an applied chat template, **always set `add_special_tokens=False`**.

```python
# ✓ CORRECT - String already has chat template applied
messages = [{"role": "user", "content": "Hello"}]
formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
tokens = tokenizer(formatted_text, add_special_tokens=False)

# ✗ INCORRECT - Will add special tokens twice
tokens = tokenizer(formatted_text, add_special_tokens=True)  # DON'T DO THIS
```

**Rationale**: Chat templates already include special tokens (BOS, EOS, etc.). Adding them again during tokenization will duplicate these tokens and corrupt the input.

### Generation Scripts

**CRITICAL**: When creating generation scripts, **always use batched generation** for efficiency.

```python
# ✓ CORRECT - Batched generation
inputs = tokenizer(prompts, padding=True, padding="longest", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)

# ✗ INCORRECT - Loop-based generation (slow)
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=100)
```

### Tokenizer Padding Configuration

**CRITICAL**: Set padding side based on the operation:

#### For Generation (Inference)
**Always use LEFT padding with LONGEST padding strategy**:

```python
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token  # if pad_token not set

# Then do batched generation - pad to longest sequence in batch
inputs = tokenizer(prompts, padding="longest", return_tensors="pt")
outputs = model.generate(**inputs)
```

**Rationale**:
- Left padding ensures all sequences end at the same position, which is necessary for autoregressive generation to work correctly in batches
- `padding="longest"` is efficient - only pads to the longest sequence in the current batch, not to a fixed max length

#### For Training (Fine-tuning)
**Always use RIGHT padding with MAX_LENGTH padding strategy**:

```python
tokenizer.padding_side = "right"

# Then tokenize training data - pad to max_length
inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
```

**Rationale**:
- Right padding is standard for training and ensures padding tokens are masked correctly in loss calculations
- `padding="max_length"` ensures consistent tensor shapes across all batches, which is important for training stability

### Quick Reference

| Operation | Padding Side | Padding Strategy |
|-----------|--------------|------------------|
| Training  | RIGHT        | `"max_length"`   |
| Generation | LEFT        | `"longest"`      |
