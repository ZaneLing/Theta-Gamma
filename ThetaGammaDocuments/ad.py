
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# RNG for reproducibility
rng = np.random.default_rng(42)

# Settings
L = 5       # number of hops
N = 300     # samples for CI estimation
hops = np.arange(1, L+1)

def make_attention_data(trend='decay', noise_level=0.03):
    if trend == 'decay':
        base = np.linspace(0.85, 0.45, L)   # baseline decays
    elif trend == 'stable':
        base = np.linspace(0.86, 0.80, L)   # HORA stays relatively stable
    elif trend == 'partial':
        base = np.linspace(0.86, 0.65, L)   # partial improvement (ablation)
    data = rng.normal(loc=base, scale=noise_level, size=(N, L))
    return np.clip(data, 0.0, 1.0)

def make_entropy_data(trend='collapse', noise_level=0.08):
    if trend == 'collapse':
        base = np.linspace(1.6, 0.4, L)     # baseline entropy collapses
    elif trend == 'stable':
        base = np.linspace(1.55, 1.45, L)   # HORA preserves entropy
    elif trend == 'partial':
        base = np.linspace(1.55, 0.9, L)    # ablation partial
    data = rng.normal(loc=base, scale=noise_level, size=(N, L))
    return np.clip(data, 0.01, None)

# Simulated data
alpha_baseline = make_attention_data('decay')
alpha_hora = make_attention_data('stable')
alpha_hora_no_theta = make_attention_data('partial')

entropy_baseline = make_entropy_data('collapse')
entropy_hora = make_entropy_data('stable')
entropy_hora_no_gamma = make_entropy_data('partial')

def mean_ci(arr):
    m = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    ci = 1.96 * sem
    return m, ci

# Save paths (modify if you want)
fname1 = 'attention_decay_square.png'
fname2 = 'logical_space_square.png'
out_path = 'hora_simulated_plots.png'

# Create attention-decay square figure
fig, ax = plt.subplots(figsize=(3.2, 3.2))
# Make square (depends on matplotlib version)
try:
    ax.set_box_aspect(1)
except Exception:
    pass

for name, arr in [('baseline', alpha_baseline),
                  ('HORA', alpha_hora),
                  ('HORA w/o θ', alpha_hora_no_theta)]:
    m, ci = mean_ci(arr)
    ax.plot(hops, m, label=name)
    ax.fill_between(hops, m-ci, m+ci, alpha=0.18)

ax.set_xlabel('Hop index')
ax.set_ylabel('Avg alignment to initial query')
ax.set_xticks(hops)
ax.set_ylim(0.35, 0.92)
ax.set_title('Attention decay: alignment vs. hop')
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(fname1, dpi=300)
plt.close(fig)

# Create logical-space (entropy) square figure
fig, ax = plt.subplots(figsize=(3.2, 3.2))
try:
    ax.set_box_aspect(1)
except Exception:
    pass

for name, arr in [('baseline', entropy_baseline),
                  ('HORA', entropy_hora),
                  ('HORA w/o γ', entropy_hora_no_gamma)]:
    m, ci = mean_ci(arr)
    ax.plot(hops, m, label=name)
    ax.fill_between(hops, m-ci, m+ci, alpha=0.18)

ax.set_xlabel('Hop index')
ax.set_ylabel('Path diversity (Shannon entropy)')
ax.set_xticks(hops)
ax.set_ylim(0.2, 1.8)
ax.set_title('Logical space: path diversity vs. hop')
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(fname2, dpi=300)
plt.close(fig)

# Concatenate horizontally
imgs = [Image.open(fname1), Image.open(fname2)]
heights = [im.size[1] for im in imgs]
max_h = max(heights)
resized = []
for im in imgs:
    if im.size[1] != max_h:
        w = int(im.size[0] * (max_h / im.size[1]))
        resized.append(im.resize((w, max_h), Image.LANCZOS))
    else:
        resized.append(im)
total_width = sum(im.size[0] for im in resized)
combined = Image.new('RGB', (total_width, max_h), (255,255,255))
x_off = 0
for im in resized:
    combined.paste(im, (x_off, 0))
    x_off += im.size[0]

combined.save(out_path, dpi=(300,300))
print(f"Saved combined figure to: {out_path}")

