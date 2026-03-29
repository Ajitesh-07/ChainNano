import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class TransformerBlock(nn.Module):
    # Added board dimensions to properly size the positional encoding
    def __init__(self, channels, board_height, board_width, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        
        seq_len = board_height * board_width
        
        # 1. DYNAMIC POSITIONAL ENCODING: Now scales to any board size (e.g., 9x6 = 54)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, channels) * 0.02)
        
        self.qkv_proj = nn.Linear(channels, channels * 3)
        self.out_proj = nn.Linear(channels, channels)
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x):
        batch_size, c, h, w = x.size()
        seq_len = h * w
        
        # Flatten the spatial dimensions
        x_flat = x.view(batch_size, c, seq_len).permute(0, 2, 1).contiguous() 
        
        # Apply Positional Encoding
        x_flat = x_flat + self.pos_embed
        
        qkv = self.qkv_proj(x_flat)        
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) 
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Flash Attention
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, c)
        attn_out = self.out_proj(attn_out)
        
        x_flat = self.norm1(x_flat + attn_out)
        
        ffn_out = self.ffn(x_flat)
        x_flat = self.norm2(x_flat + ffn_out)
        
        # Reshape back to 2D image format
        out = x_flat.permute(0, 2, 1).contiguous().view(batch_size, c, h, w)
        return out

class RRTBlock(nn.Module):
    def __init__(self, channels, board_height, board_width):
        super().__init__()
        self.res1 = ResBlock(channels)
        self.res2 = ResBlock(channels)
        self.transformer = TransformerBlock(channels, board_height, board_width)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.transformer(x)
        return x

class ChainReactionResTNet(nn.Module):
    # Defaults set for an 8-channel input and a standard 9x6 board.
    def __init__(self, input_channels=8, hidden_channels=128, num_rrt_blocks=2, board_height=12, board_width=8):
        super().__init__()
        
        self.board_height = board_height
        self.board_width = board_width
        
        self.start_block = ConvBlock(input_channels, hidden_channels)
        
        self.backbone = nn.Sequential(
            *[RRTBlock(hidden_channels, board_height, board_width) for _ in range(num_rrt_blocks)]
        )
        
        # THE FIX: Action space is just tapping a cell. 
        # Output is 1 channel representing the board, which we flatten.
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1) # 1 channel output
        )
        
        # THE FIX: Value head dynamically scaled to board size
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_height * board_width, 128), # Reduced fully connected size
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.start_block(x)
        x = self.backbone(x)
        
        # Policy output shape: [Batch, Board_Height * Board_Width]
        policy_logits = self.policy_head(x)
        policy_logits = policy_logits.reshape(policy_logits.size(0), -1) 
        
        value = self.value_head(x)
        
        return policy_logits, value
    
def benchmark_inference(input_channels=8, batch_size=256, channels=256, blocks=4):
    # Ensure we are using the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not detected. Benchmarking on CPU.")
        
    # 1. Initialize the model with your exact specs
    model = ChainReactionResTNet(input_channels=input_channels, hidden_channels=channels, num_rrt_blocks=blocks)
    print(sum(p.numel() for p in model.parameters()))
    model.to(device)
    
    # 2. The bf16 Magic: Convert all model weights to bfloat16
    model.to(torch.bfloat16)
    model.eval() # Set to evaluation mode (disables dropout/batchnorm updates)
    
    # 3. Create dummy input tensor matching the (Batch, Channels, 8, 8) shape
    # We create it directly in bf16 on the GPU to avoid transfer overhead
    dummy_input = torch.randn(batch_size, input_channels, 12, 8, dtype=torch.bfloat16, device=device)
    
    print(f"--- Benchmarking ResTNet ({channels} Channels, {blocks} RRT Blocks) ---")
    print(f"Precision: bfloat16")
    print(f"Batch Size: {batch_size}")
    
    # 4. Warmup
    # CUDA API calls are asynchronous. We must warm up the GPU so it allocates
    # the necessary memory and gets out of idle power states.
    print("Warming up Tensor Cores...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
            
    # 5. The Actual Benchmark
    print("Running 100 batched forward passes...")
    iterations = 100
    
    torch.cuda.synchronize() # Wait for all previous operations to finish
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(iterations):
            # We can optionally use torch.autocast, but since we casted the model 
            # and inputs directly to bf16, this is already pure bf16 math.
            policy, value = model(dummy_input)
            
    torch.cuda.synchronize() # Wait for all benchmark operations to finish
    end_time = time.perf_counter()
    
    # 6. Calculate Metrics
    total_time = end_time - start_time
    time_per_batch = (total_time / iterations) * 1000 # in milliseconds
    nodes_per_second = (batch_size * iterations) / total_time
    
    print("\n--- Results ---")
    print(f"Total Time: {total_time:.3f} seconds")
    print(f"Latency per batch ({batch_size} nodes): {time_per_batch:.2f} ms")
    print(f"Throughput: {nodes_per_second:,.0f} Nodes Per Second (NPS)")
    
    # 7. Check the 8GB VRAM envelope
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak VRAM Usage: {peak_vram:.2f} MB")

if __name__ == "__main__":
    # Test with a batch size of 256, which is typical for a multithreaded MCGS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benchmark_inference(input_channels=8, batch_size=256, channels=128, blocks=2)

    model = ChainReactionResTNet(input_channels=8, hidden_channels=128, num_rrt_blocks=2).to(device)

    model.to(torch.bfloat16)
    model.eval()

    # 2. Create a bfloat16 dummy input
    dummy_input = torch.zeros(1, 8, 12, 8, dtype=torch.bfloat16, device=device)

    # 3. Trace and Export
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model = traced_model.cpu()
    traced_model.save("chain_reaction_bf16.pt")
    print("Model successfully exported to chess_50k_bf16.pt in Pure bfloat16!")