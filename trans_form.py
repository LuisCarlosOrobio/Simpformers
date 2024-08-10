import numpy as np

# Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    matmul_qk = np.matmul(Q, K.transpose(0, 1, 3, 2))  # (batch_size, num_heads, seq_len_q, seq_len_k)
    dk = Q.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = np.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

# Multi-Head Attention Layer
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.Wq = np.random.randn(d_model, d_model)
        self.Wk = np.random.randn(d_model, d_model)
        self.Wv = np.random.randn(d_model, d_model)
        self.dense = np.random.randn(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]
        
        Q = np.dot(q, self.Wq)
        K = np.dot(k, self.Wk)
        V = np.dot(v, self.Wv)
        
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        scaled_attention, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        
        output = np.dot(concat_attention, self.dense)
        return output

# Positional Encoding
def positional_encoding(seq_len, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(seq_len)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding

# Feed-Forward Network
class FeedForwardNetwork:
    def __init__(self, d_model, dff):
        self.W1 = np.random.randn(d_model, dff)
        self.W2 = np.random.randn(dff, d_model)

    def forward(self, x):
        return np.dot(np.maximum(0, np.dot(x, self.W1)), self.W2)

# Layer Normalization
class LayerNormalization:
    def __init__(self, d_model, epsilon=1e-6):
        self.gamma = np.ones((1, d_model))
        self.beta = np.zeros((1, d_model))
        self.epsilon = epsilon

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized_x = (x - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * normalized_x + self.beta

# Transformer Encoder Layer
class EncoderLayer:
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)
    
    def forward(self, x, mask):
        attn_output = self.mha.forward(x, x, x, mask)
        out1 = self.layernorm1.forward(attn_output + x)
        
        ffn_output = self.ffn.forward(out1)
        out2 = self.layernorm2.forward(ffn_output + out1)
        
        return out2

# Transformer Encoder
class Encoder:
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = np.random.randn(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(self.num_layers)]
    
    def forward(self, x, mask):
        seq_len = x.shape[1]
        
        # Corrected: Lookup embedding instead of using dot product
        x = self.embedding[x] * np.sqrt(self.d_model)  # Shape: (batch_size, seq_len, d_model)
        x += self.pos_encoding[:, :seq_len, :]
        
        for i in range(self.num_layers):
            x = self.enc_layers[i].forward(x, mask)
        
        return x

# Example usage of the Transformer Encoder

# Parameters
d_model = 128
num_heads = 8
dff = 512
num_layers = 4
input_vocab_size = 10000
maximum_position_encoding = 1000
seq_len = 10
batch_size = 2

# Initialize the Encoder
encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding)

# Create some random input data
x = np.random.randint(0, input_vocab_size, size=(batch_size, seq_len))

# Forward pass through the Encoder
output = encoder.forward(x, mask=None)

# Print the output shape to verify
print("Output shape:", output.shape)
