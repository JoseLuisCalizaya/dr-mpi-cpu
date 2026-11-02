import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Cargar archivos procesados
df_strong = pd.read_csv('summary_strong_scaling.csv')
df_bw = pd.read_csv('summary_bandwidth.csv')

# Configuración de la cuadrícula 2x3 idéntica al paper
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

grid_config = [
    [('DotProduct_DR', 'Dot Product', 'speedup'),
     ('Reduce_DR', 'Reduce', 'speedup'),
     ('Inclusive_Scan_DR', 'Inclusive Scan', 'speedup')],
    [('BlackScholes_DR', 'Black Scholes', 'speedup'),
     ('Stream_Triad_DR', 'Stream Triad', 'bandwidth'),
     ('GemvEq_DR', 'GEMV (GemvEq)', 'speedup')]
]

ranks_all = [1, 2, 4, 6]

for r_idx in range(2):
    for c_idx in range(3):
        bench, title, metric = grid_config[r_idx][c_idx]
        ax = axes[r_idx, c_idx]
        
        # Línea de referencia ideal
        ax.plot(ranks_all, ranks_all if metric == 'speedup' else [ranks_all[0]]*4, 
                '--', label='perfect scaling' if metric=='speedup' else 'ref', color='gray', alpha=0.7)
        
        if metric == 'speedup':
            sub = df_strong[df_strong['benchmark'] == bench].sort_values('ranks')
            ax.plot(sub['ranks'], sub['speedup'], marker='o', color='#ff7f0e', linewidth=2, label='mhp (CPU Local)')
            ax.set_ylabel('Speedup')
            ax.set_ylim(bottom=0)
        elif metric == 'bandwidth':
            sub_stream = df_bw[df_bw['benchmark'] == 'Stream_Triad_DR'].sort_values('ranks')
            sub_dot = df_bw[df_bw['benchmark'] == 'DotProduct_DR'].sort_values('ranks')
            
            ax.plot(sub_stream['ranks'], sub_stream['bandwidth_GiB_s'], marker='o', color='#ff7f0e', linewidth=2, label='mhp (STREAM)')
            ax.plot(sub_dot['ranks'], sub_dot['bandwidth_GiB_s'], marker='s', color='#1f77b4', linewidth=2, label='mhp (DotProduct)')
            ax.axhline(y=21.3, color='r', linestyle=':', label='Límite DDR4 (~21.3 GiB/s)')
            ax.set_ylabel('Bandwidth (GiB/s)')
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Procesos MPI (Rangos)')
        ax.set_xticks(ranks_all)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig('figura2_replica_final.png', dpi=300)
print("Gráfica generada exitosamente en 'figura2_replica_final.png'")
