import os

output_dir = r""
# Ensure directory exists
os.makedirs(output_dir, exist_ok=True)

def generate_shader(reg_count):
    filename = f"r{reg_count}.comp"
    
    # Target ALU ops per iter = 768
    # repeats * reg_count = 768
    # repeats = 768 / reg_count
    repeats = int(768 / reg_count)
    if repeats < 1:
        repeats = 1
        
    # Strategy: Split into arrays of max size 64 to avoid potential compiler issues with huge single arrays
    # although unrolled access usually handles it, splitting is safer and matches previous style.
    arrays_config = []
    remaining = reg_count
    idx = 0
    while remaining > 0:
        size = min(remaining, 64)
        arrays_config.append((f"r{idx}", size, 1.0 + idx * 0.1)) # slightly different start vals
        remaining -= size
        idx += 1

    code = []
    code.append("#version 450")
    code.append("layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;")
    code.append("layout(set=0,binding=0) buffer Data {")
    code.append("    float data[];")
    code.append("} InOut;")
    code.append("layout(push_constant) uniform PC {")
    code.append("    int iters;")
    code.append("} pc;")
    code.append("")
    code.append("void main() {")
    code.append("    uint idx = gl_GlobalInvocationID.x;")
    code.append("    if(idx >= InOut.data.length()) return;")
    code.append("    float v = InOut.data[idx];")
    
    # Declare arrays
    for name, size, _ in arrays_config:
        code.append(f"    float {name}[{size}];")
    
    # Initialize arrays (Unrolled)
    code.append("    // Initialize")
    for name, size, start_val in arrays_config:
        for i in range(size):
            val = start_val + i * 0.001
            code.append(f"    {name}[{i}] = v * {val:.4f};")
            
    # Computation loop
    code.append("    for(int i=0;i<pc.iters;i++) {")
    
    # Inner loop for ALU intensity consistency
    code.append(f"        for(int k=0; k<{repeats}; k++) {{")
    
    # Unrolled computation
    for name, size, _ in arrays_config:
        for i in range(size):
            code.append(f"            {name}[{i}] = fma({name}[{i}], 1.0001, 0.0001);")
            
    code.append("        }") # End inner loop
    code.append("    }") # End outer loop
    
    # Summation (Unrolled)
    code.append("    float s = 0.0;")
    for name, size, _ in arrays_config:
        for i in range(size):
            code.append(f"    s += {name}[{i}];")
            
    code.append("    InOut.data[idx] = v + s * 0.00001;")
    code.append("}")
    code.append("")

    with open(os.path.join(output_dir, filename), "w") as f:
        f.write("\n".join(code))
    print(f"Generated {filename} (Regs: {reg_count}, Repeats: {repeats}, Total Ops: {reg_count * repeats})")

# Generate for a range of register counts
# Including r16, r32 to ensure baseline consistency
register_counts = [16, 32, 64, 96, 128, 192, 256]

for count in register_counts:
    generate_shader(count)
