import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from time import time
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLSdpaAttention, apply_multimodal_rotary_pos_emb, Qwen2VLAttention, repeat_kv
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.models.qwen2_vl.modeling_qwen2_vl import Cache, StaticCache

config_MNIST = Qwen2VLConfig(
    vocab_size=10,  # MNIST has 10 classes (digits 0-9)
    hidden_size=64,  # Reduced hidden size for simplicity
    intermediate_size=128,  # Suitable for a lightweight model
    num_hidden_layers=2,  # Few layers are enough for MNIST
    num_attention_heads=4,  # Reduced number of attention heads
    num_key_value_heads=2,  # Grouped Query Attention for efficiency
    max_position_embeddings=28 * 28,  # Maximum sequence length for flattened MNIST images
    initializer_range=0.02,  # Default initialization range
    rms_norm_eps=1e-5,  # Default RMSNorm epsilon
    use_cache=False,  # Cache is unnecessary for a single-shot classification task
    tie_word_embeddings=True,  # Tie word embeddings for consistency
    rope_theta=10000.0,  # Default RoPE theta
    use_sliding_window=False,  # Sliding window attention is unnecessary for MNIST
    sliding_window=None,  # No sliding window
    max_window_layers=0,  # No sliding window layers
    attention_dropout=0.1,  # Introduce dropout for regularization
    rope_scaling={"rope_type": "default", "mrope_section": 32},  # Add 'mrope_section'
)

class MNISTClassifier(nn.Module):
    def __init__(self, attention):
        super(MNISTClassifier, self).__init__()
        self.attention = attention
        self.classifier = nn.Linear(config_MNIST.hidden_size, 10)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        # Flatten MNIST images and project to hidden size
        inputs = inputs.view(batch_size, 28 * 28).unsqueeze(-1)  # Shape: (batch_size, 784, 1)
        inputs = inputs.repeat(1, 1, 64)  # Expand to match hidden size (config.hidden_size = 64)

        # Generate position_ids for RoPE
        seq_length = inputs.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)  # Shape: (seq_length,)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, seq_length)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # Shape: (3, batch_size, seq_length)

        # Compute attention outputs
        attention_output, _, _ = self.attention(hidden_states=inputs, position_ids=position_ids)

        # Global average pooling to aggregate sequence information
        pooled_output = attention_output.mean(dim=1)

        # Compute logits
        logits = self.classifier(pooled_output)

        return logits

class EfficientQwen2VLSdpaAttention(Qwen2VLAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        total_qkv_dim = self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim
        self.qkv_proj = nn.Linear(self.hidden_size, total_qkv_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        # Unified projection for Q, K, V
        qkv_states = self.qkv_proj(hidden_states)
        query_size = self.num_heads * self.head_dim
        key_value_size = self.num_key_value_heads * self.head_dim * 2

        # Direct slicing for Q, K, V tensors
        query_states = qkv_states[:, :, :query_size].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_value_states = qkv_states[:, :, query_size:].view(
            bsz, q_len, 2, self.num_key_value_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)

        key_states, value_states = key_value_states

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        # Rotary positional embedding
        cos, sin = (
            self.rotary_emb(value_states, position_ids)
            if position_embeddings is None
            else position_embeddings
        )
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # Efficient repetition of keys and values
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Efficient mask slicing and preparation
        causal_mask = attention_mask[:, :, :, : key_states.size(-2)] if attention_mask is not None else None

        # Ensure tensors are contiguous only if required
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # SDPA with efficient backend
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=causal_mask is None and q_len > 1,
        )

        # Reshape and project efficiently
        attn_output = self.o_proj(attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size))

        return attn_output, None, past_key_value

# Dataset preparation remains the same
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training and evaluation function
def train_and_evaluate(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        start_time = time()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    elapsed_time = time() - start_time
    return accuracy, elapsed_time


# Compare original and efficient classifiers
original_model = MNISTClassifier(Qwen2VLSdpaAttention(config_MNIST))
efficient_model = MNISTClassifier(EfficientQwen2VLSdpaAttention(config_MNIST))

print("Training Original Classifier...")
original_accuracy, original_time = train_and_evaluate(original_model)

print("\nTraining Efficient Classifier...")
efficient_accuracy, efficient_time = train_and_evaluate(efficient_model)

print(f"\nOriginal Classifier - Accuracy: {original_accuracy:.2f}%, Elapsed Time: {original_time:.2f} seconds")
print(f"Efficient Classifier - Accuracy: {efficient_accuracy:.2f}%, Elapsed Time: {efficient_time:.2f} seconds")
