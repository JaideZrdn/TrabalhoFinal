"""
=============================================================================
TRABALHO FINAL - FIA (ICC260)
Tarefa 2: Raciocínio Espacial com Logic Tensor Networks (LTNtorch)
=============================================================================

PREDICADOS RELACIONAIS IMPLEMENTADOS:
  Horizontais:
    leftOf(x, y)      → x está à esquerda de y
    rightOf(x, y)     → x está à direita de y
    closeTo(x, y)     → x está próximo de y (kernel gaussiano)
    inBetween(x, y, z)→ x está entre y e z
    lastOnTheLeft(x)  → x é o objeto mais à esquerda de todos
    lastOnTheRight(x) → x é o objeto mais à direita de todos

  Verticais (Tarefa 3 embutida):
    below(x, y)       → x está abaixo de y
    above(x, y)       → x está acima de y
    canStack(x, y)    → x pode ser empilhado sobre y

AXIOMAS IMPLEMENTADOS:
  Irreflexividade:  ∀x ¬LeftOf(x,x)
  Assimetria:       ∀x,y LeftOf(x,y) → ¬LeftOf(y,x)
  Inverso H:        ∀x,y LeftOf(x,y) ↔ RightOf(y,x)
  Transitividade H: ∀x,y,z LeftOf(x,y) ∧ LeftOf(y,z) → LeftOf(x,z)
  Inverso V:        ∀x,y Below(x,y) ↔ Above(y,x)
  Transitividade V: ∀x,y,z Below(x,y) ∧ Below(y,z) → Below(x,z)
  Supervisão:       Labels geométricos reais ancorando o aprendizado
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ltn

# ─────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────────
SEED       = 42
N_OBJECTS  = 25
EPOCHS     = 800
LR         = 0.001
THRESHOLD  = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)

SHAPES  = ["Circle", "Square", "Cylinder", "Cone", "Triangle"]
COLORS  = ["Red", "Green", "Blue"]


# ═════════════════════════════════════════════
# BLOCO 1 — GERAÇÃO DE DADOS (reaproveitado da Tarefa 1)
# ═════════════════════════════════════════════

def generate_objects(n=25, seed=None):
    """Gera n objetos aleatórios. Retorna tensor (n,11) + labels."""
    if seed is not None:
        np.random.seed(seed)

    data, shape_labels, color_labels = [], [], []

    for _ in range(n):
        x     = np.random.uniform(0.0, 1.0)
        y     = np.random.uniform(0.0, 1.0)
        size  = float(np.random.randint(0, 2))

        color_idx = np.random.randint(0, 3)
        color_vec = [0.0, 0.0, 0.0]; color_vec[color_idx] = 1.0

        shape_idx = np.random.randint(0, 5)
        shape_vec = [0.0]*5; shape_vec[shape_idx] = 1.0

        data.append([x, y] + color_vec + shape_vec + [size])
        shape_labels.append(SHAPES[shape_idx])
        color_labels.append(COLORS[color_idx])

    return torch.tensor(data, dtype=torch.float32), shape_labels, color_labels


def plot_scene(objects_tensor, shape_labels, color_labels,
               title="Cena CLEVR — Raciocínio Espacial"):
    marker_map = {"Circle":"o","Square":"s","Cylinder":"D","Cone":"^","Triangle":"v"}
    color_map  = {"Red":"red","Green":"green","Blue":"blue"}
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x (horizontal)"); ax.set_ylabel("y (vertical)")
    ax.grid(True, alpha=0.3)
    data = objects_tensor.numpy()
    for i in range(len(data)):
        x, y = data[i,0], data[i,1]
        size = data[i,10]
        ms   = 220 if size == 1.0 else 90
        ax.scatter(x, y,
                   c=color_map[color_labels[i]],
                   marker=marker_map[shape_labels[i]],
                   s=ms, edgecolors='black', linewidths=0.8, zorder=3)
        ax.annotate(str(i), (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7)

    # Eixo de referência
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.4, label='centro x')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, label='centro y')

    legend_shapes = [mpatches.Patch(color='gray',
                     label=f"{s} ({marker_map[s]})") for s in SHAPES]
    legend_colors = [mpatches.Patch(color=color_map[c], label=c) for c in COLORS]
    ax.legend(handles=legend_shapes + legend_colors,
              loc='upper left', fontsize=8, framealpha=0.8)
    plt.tight_layout()
    plt.savefig("cena_espacial.png", dpi=150)
    plt.show()
    print("→ cena_espacial.png salva.")


# ═════════════════════════════════════════════
# BLOCO 2 — ARQUITETURAS DAS MLPs
# ═════════════════════════════════════════════

class MLPUnario(nn.Module):
    """
    Predicado UNÁRIO: recebe features de 1 objeto → retorna [0,1].
    Usado em: lastOnTheLeft, lastOnTheRight.
    Entrada: vetor (11,)
    """
    def __init__(self, input_size=11, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),          nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)


class MLPBinario(nn.Module):
    """
    Predicado BINÁRIO: recebe features de 2 objetos CONCATENADOS → [0,1].
    Entrada: vetor (22,)  = [obj_x (11) | obj_y (11)]
    Usado em: leftOf, rightOf, closeTo, below, above, canStack.

    Por que concatenar?
    O modelo precisa VER os dois objetos juntos para aprender a relação
    entre eles. A concatenação é o jeito mais simples de fazer isso.
    """
    def __init__(self, input_size=22, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, 1),          nn.Sigmoid()
        )
    def forward(self, x, y):
        # Concatena os dois objetos ao longo da última dimensão
        xy = torch.cat([x, y], dim=-1)
        return self.net(xy)


class MLPTernario(nn.Module):
    """
    Predicado TERNÁRIO: recebe 3 objetos → [0,1].
    Entrada: vetor (33,) = [obj_x (11) | obj_y (11) | obj_z (11)]
    Usado em: inBetween(x, y, z).
    """

    def __init__(self, input_size=33, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, x, y, z):
        # Recebe os 3 tensores separadamente e concatena na última dimensão
        xyz = torch.cat([x, y, z], dim=-1)
        return self.net(xyz)


# ═════════════════════════════════════════════
# BLOCO 3 — LABELS GEOMÉTRICOS (ground truth)
#
# Em vez de anotar manualmente, calculamos diretamente
# das coordenadas x,y dos objetos. Isso é possível porque
# temos acesso ao ambiente simulado.
# ═════════════════════════════════════════════

def gerar_labels_espaciais(objects_tensor, sigma=0.15):
    """
    Gera os labels geométricos verdadeiros para todos os pares de objetos.

    Retorna dicionários booleanos:
        left_labels[(i,j)]   = True  se obj_i está à esquerda de obj_j
        right_labels[(i,j)]  = True  se obj_i está à direita de obj_j
        close_labels[(i,j)]  = True  se distância euclidiana < sigma
        below_labels[(i,j)]  = True  se obj_i está abaixo de obj_j
        above_labels[(i,j)]  = True  se obj_i está acima de obj_j
        between_labels[(i,j,k)] = True se obj_i está entre obj_j e obj_k
    """
    n = len(objects_tensor)
    coords = objects_tensor[:, :2].numpy()   # pega só x e y

    left_labels, right_labels   = {}, {}
    close_labels                = {}
    below_labels, above_labels  = {}, {}
    between_labels              = {}

    for i in range(n):
        for j in range(n):
            xi, xj = coords[i, 0], coords[j, 0]
            yi, yj = coords[i, 1], coords[j, 1]
            dist   = np.sqrt((xi - xj)**2 + (yi - yj)**2)

            # Relações horizontais
            left_labels[(i,j)]  = bool(xi < xj)
            right_labels[(i,j)] = bool(xi > xj)
            close_labels[(i,j)] = bool(dist < sigma and i != j)

            # Relações verticais
            below_labels[(i,j)] = bool(yi < yj)
            above_labels[(i,j)] = bool(yi > yj)

        # InBetween: obj_i está entre obj_j e obj_k?
        for j in range(n):
            for k in range(n):
                if i == j or i == k or j == k:
                    between_labels[(i,j,k)] = False
                    continue
                xj, xk = coords[j,0], coords[k,0]
                x_min, x_max = min(xj,xk), max(xj,xk)
                # i está entre j e k se sua coordenada x estiver no intervalo
                between_labels[(i,j,k)] = bool(x_min < coords[i,0] < x_max)

    return left_labels, right_labels, close_labels, below_labels, above_labels, between_labels


def idx_mais_a_esquerda(objects_tensor):
    """Retorna o índice do objeto com menor coordenada x."""
    return int(torch.argmin(objects_tensor[:, 0]).item())

def idx_mais_a_direita(objects_tensor):
    """Retorna o índice do objeto com maior coordenada x."""
    return int(torch.argmax(objects_tensor[:, 0]).item())


# ═════════════════════════════════════════════
# BLOCO 4 — CONFIGURAÇÃO LTN
# ═════════════════════════════════════════════

def setup_ltn_t2(objects_tensor):
    """
    Instancia variáveis LTN e todos os predicados da Tarefa 2.

    Variáveis LTN:
        x, y, z → percorrem todos os objetos (para quantificadores ∀ e ∃)

    Predicados:
        Unários:  lastOnTheLeft, lastOnTheRight
        Binários: leftOf, rightOf, closeTo, below, above, canStack
        Ternário: inBetween
    """
    # Variáveis — cada uma itera sobre todos os N objetos
    x = ltn.Variable("x", objects_tensor)
    y = ltn.Variable("y", objects_tensor)
    z = ltn.Variable("z", objects_tensor)

    predicados = {
        # ── Horizontais ──────────────────────────────────
        "leftOf":        ltn.Predicate(MLPBinario()),
        "rightOf":       ltn.Predicate(MLPBinario()),
        "closeTo":       ltn.Predicate(MLPBinario()),
        "inBetween":     ltn.Predicate(MLPTernario()),
        "lastOnTheLeft": ltn.Predicate(MLPUnario()),
        "lastOnTheRight":ltn.Predicate(MLPUnario()),

        # ── Verticais ────────────────────────────────────
        "below":         ltn.Predicate(MLPBinario()),
        "above":         ltn.Predicate(MLPBinario()),
        "canStack":      ltn.Predicate(MLPBinario()),
    }

    return x, y, z, predicados


# ═════════════════════════════════════════════
# BLOCO 5 — AXIOMAS DA KB
# ═════════════════════════════════════════════

def compute_kb_t2(x, y, z, predicados, objects_tensor,
                  shape_labels, labels_espaciais):
    """
    Calcula a satisfatibilidade de TODOS os axiomas da Tarefa 2.

    ┌─────────────────────────────────────────────────────────┐
    │  AXIOMAS HORIZONTAIS                                    │
    │                                                         │
    │  1. Irreflexividade:                                    │
    │     ∀x  ¬LeftOf(x,x)                                   │
    │     → nenhum objeto está à esquerda de si mesmo         │
    │                                                         │
    │  2. Assimetria:                                         │
    │     ∀x,y  LeftOf(x,y) → ¬LeftOf(y,x)                  │
    │     → se x está à esq de y, y NÃO pode estar à esq x   │
    │                                                         │
    │  3. Inverso:                                            │
    │     ∀x,y  LeftOf(x,y) ↔ RightOf(y,x)                  │
    │     → esquerda e direita são opostos                    │
    │                                                         │
    │  4. Transitividade:                                     │
    │     ∀x,y,z  (LeftOf(x,y) ∧ LeftOf(y,z)) → LeftOf(x,z) │
    │     → cadeia posicional                                 │
    │                                                         │
    │  AXIOMAS VERTICAIS                                      │
    │                                                         │
    │  5. Inverso:  ∀x,y  Below(x,y) ↔ Above(y,x)           │
    │  6. Transitividade: ∀x,y,z                             │
    │     Below(x,y) ∧ Below(y,z) → Below(x,z)              │
    │                                                         │
    │  SUPERVISÃO (ancoragem nos dados reais)                 │
    │  7-14. Labels geométricos verdadeiros                   │
    └─────────────────────────────────────────────────────────┘
    """
    p  = predicados

    # Operadores fuzzy
    Not    = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And    = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Or     = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
    Impl   = ltn.Connective(ltn.fuzzy_ops.ImpliesLuk())   # Łukasiewicz
    Equiv  = ltn.Connective(ltn.fuzzy_ops.ImpliesLuk())   # usamos Impl nos dois sentidos
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2),      quantifier="e")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    # ──────────────────────────────────────────
    # AXIOMA 1: Irreflexividade
    # ∀x ¬LeftOf(x,x)
    # ──────────────────────────────────────────
    ax1_irreflexividade = Forall(x, Not(p["leftOf"](x, x)))

    # ──────────────────────────────────────────
    # AXIOMA 2: Assimetria
    # ∀x,y  LeftOf(x,y) → ¬LeftOf(y,x)
    #
    # Em lógica fuzzy Łukasiewicz:
    #   A → B  =  min(1, 1 - A + B)
    # Quando A=1 (x está à esq de y com certeza),
    # forçamos B=1 (y NÃO está à esq de x com certeza)
    # ──────────────────────────────────────────
    ax2_assimetria = Forall(
        [x, y],
        Impl(p["leftOf"](x, y),
             Not(p["leftOf"](y, x)))
    )

    # ──────────────────────────────────────────
    # AXIOMA 3: Inverso (bicondicional)
    # ∀x,y  LeftOf(x,y) ↔ RightOf(y,x)
    #
    # Implementado como:
    #   (A→B) ∧ (B→A)
    # ──────────────────────────────────────────
    ax3_inverso_h = Forall(
        [x, y],
        And(
            Impl(p["leftOf"](x, y),  p["rightOf"](y, x)),
            Impl(p["rightOf"](y, x), p["leftOf"](x, y))
        )
    )

    # ──────────────────────────────────────────
    # AXIOMA 4: Transitividade
    # ∀x,y,z  LeftOf(x,y) ∧ LeftOf(y,z) → LeftOf(x,z)
    #
    # Este axioma é o mais "poderoso" para raciocínio em cadeia.
    # Ele diz: se A está à esq de B, e B está à esq de C,
    # então A DEVE estar à esq de C.
    # ──────────────────────────────────────────
    ax4_transitividade = Forall(
        [x, y, z],
        Impl(
            And(p["leftOf"](x, y), p["leftOf"](y, z)),
            p["leftOf"](x, z)
        )
    )

    # ──────────────────────────────────────────
    # AXIOMA 5: Inverso Vertical
    # ∀x,y  Below(x,y) ↔ Above(y,x)
    # ──────────────────────────────────────────
    ax5_inverso_v = Forall(
        [x, y],
        And(
            Impl(p["below"](x, y), p["above"](y, x)),
            Impl(p["above"](y, x), p["below"](x, y))
        )
    )

    # ──────────────────────────────────────────
    # AXIOMA 6: Transitividade Vertical
    # ∀x,y,z  Below(x,y) ∧ Below(y,z) → Below(x,z)
    # ──────────────────────────────────────────
    ax6_transitividade_v = Forall(
        [x, y, z],
        Impl(
            And(p["below"](x, y), p["below"](y, z)),
            p["below"](x, z)
        )
    )

    # ──────────────────────────────────────────
    # AXIOMA 7: lastOnTheLeft
    # ∃x (∀y leftOf(x,y))
    # → existe um objeto que está à esquerda de TODOS os outros
    # ──────────────────────────────────────────
    ax7_last_left = Exists(x, Forall(y, p["leftOf"](x, y)))

    # ──────────────────────────────────────────
    # AXIOMA 8: lastOnTheRight
    # ∃x (∀y rightOf(x,y))
    # ──────────────────────────────────────────
    ax8_last_right = Exists(x, Forall(y, p["rightOf"](x, y)))

    # ──────────────────────────────────────────
    # SUPERVISÃO COM LABELS REAIS
    # Para cada par (i,j), se sabemos que obj_i está à esquerda
    # de obj_j, forçamos leftOf(obj_i, obj_j) ≈ 1.0
    # ──────────────────────────────────────────
    (left_labels, right_labels, close_labels,
     below_labels, above_labels, between_labels) = labels_espaciais

    n = len(objects_tensor)

    sup_left, sup_right = [], []
    sup_close           = []
    sup_below, sup_above = [], []
    sup_between         = []
    sup_can_stack       = []

    # Usamos uma AMOSTRA dos pares para eficiência
    # (n*n pares seria 625 — pode ser lento com gradientes)
    sample_pairs = [(i,j) for i in range(n) for j in range(n) if i != j]
    np.random.shuffle(sample_pairs)
    sample_pairs = sample_pairs[:120]   # 120 pares amostrados

    for (i, j) in sample_pairs:
        oi = ltn.Constant(objects_tensor[i])
        oj = ltn.Constant(objects_tensor[j])

        # leftOf
        v = p["leftOf"](oi, oj)
        sup_left.append(v.value if left_labels[(i,j)] else 1.0 - v.value)

        # rightOf
        v = p["rightOf"](oi, oj)
        sup_right.append(v.value if right_labels[(i,j)] else 1.0 - v.value)

        # closeTo
        v = p["closeTo"](oi, oj)
        sup_close.append(v.value if close_labels[(i,j)] else 1.0 - v.value)

        # below
        v = p["below"](oi, oj)
        sup_below.append(v.value if below_labels[(i,j)] else 1.0 - v.value)

        # above
        v = p["above"](oi, oj)
        sup_above.append(v.value if above_labels[(i,j)] else 1.0 - v.value)

        # canStack: x pode ser empilhado sobre y se y não for Cone nem Triangle
        j_shape = shape_labels[j]
        can = (j_shape not in ["Cone", "Triangle"])
        v = p["canStack"](oi, oj)
        sup_can_stack.append(v.value if can else 1.0 - v.value)

    # InBetween: amostra de trios
    sample_trios = [(i,j,k) for i in range(n)
                             for j in range(n)
                             for k in range(n)
                             if i!=j and i!=k and j!=k]
    np.random.shuffle(sample_trios)
    sample_trios = sample_trios[:80]

    for (i, j, k) in sample_trios:
        oi = ltn.Constant(objects_tensor[i])
        oj = ltn.Constant(objects_tensor[j])
        ok = ltn.Constant(objects_tensor[k])
        v  = p["inBetween"](oi, oj, ok)
        is_between = between_labels[(i,j,k)]
        sup_between.append(v.value if is_between else 1.0 - v.value)

    # Supervisão do objeto mais à esquerda / direita
    idx_left  = idx_mais_a_esquerda(objects_tensor)
    idx_right = idx_mais_a_direita(objects_tensor)
    sup_last_left  = p["lastOnTheLeft"](ltn.Constant(objects_tensor[idx_left])).value
    sup_last_right = p["lastOnTheRight"](ltn.Constant(objects_tensor[idx_right])).value

    def mean_stack(lst):
        return torch.mean(torch.stack(lst))

    sat_sup_left      = mean_stack(sup_left)
    sat_sup_right     = mean_stack(sup_right)
    sat_sup_close     = mean_stack(sup_close)
    sat_sup_below     = mean_stack(sup_below)
    sat_sup_above     = mean_stack(sup_above)
    sat_sup_between   = mean_stack(sup_between)
    sat_sup_canstack  = mean_stack(sup_can_stack)
    sat_sup_ll        = sup_last_left
    sat_sup_lr        = sup_last_right

    # ──────────────────────────────────────────
    # SatAgg FINAL
    # Combina axiomas lógicos + supervisão
    # ──────────────────────────────────────────
    sat_total = SatAgg(
        ax1_irreflexividade,
        ax2_assimetria,
        ax3_inverso_h,
        ax4_transitividade,
        ax5_inverso_v,
        ax6_transitividade_v,
        ax7_last_left,
        ax8_last_right,
        sat_sup_left,
        sat_sup_right,
        sat_sup_close,
        sat_sup_below,
        sat_sup_above,
        sat_sup_between,
        sat_sup_canstack,
        sat_sup_ll,
        sat_sup_lr,
    )

    detalhes = {
        "ax1_irreflexividade":   ax1_irreflexividade.value.item(),
        "ax2_assimetria":        ax2_assimetria.value.item(),
        "ax3_inverso_h":         ax3_inverso_h.value.item(),
        "ax4_transitividade_h":  ax4_transitividade.value.item(),
        "ax5_inverso_v":         ax5_inverso_v.value.item(),
        "ax6_transitividade_v":  ax6_transitividade_v.value.item(),
        "ax7_last_left":         ax7_last_left.value.item(),
        "ax8_last_right":        ax8_last_right.value.item(),
        "sup_leftOf":            sat_sup_left.item(),
        "sup_rightOf":           sat_sup_right.item(),
        "sup_closeTo":           sat_sup_close.item(),
        "sup_below":             sat_sup_below.item(),
        "sup_above":             sat_sup_above.item(),
        "sup_inBetween":         sat_sup_between.item(),
        "sup_canStack":          sat_sup_canstack.item(),
    }

    return sat_total, detalhes


# ═════════════════════════════════════════════
# BLOCO 6 — TREINAMENTO
# ═════════════════════════════════════════════

def treinar_t2(objects_tensor, shape_labels, labels_espaciais,
               epochs=EPOCHS, lr=LR, verbose=True):
    """
    Loop de treinamento para a Tarefa 2.
    Maximiza SatAgg de todos os axiomas espaciais.
    """
    x, y, z, predicados = setup_ltn_t2(objects_tensor)

    todos_params = []
    for pred in predicados.values():
        todos_params += list(pred.parameters())
    optimizer = torch.optim.Adam(todos_params, lr=lr)

    historico = []

    print(f"\n{'='*60}")
    print(f"  TREINAMENTO — TAREFA 2: RACIOCÍNIO ESPACIAL")
    print(f"  {N_OBJECTS} objetos | {epochs} épocas | lr={lr}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        optimizer.zero_grad()

        sat_total, detalhes = compute_kb_t2(
            x, y, z, predicados, objects_tensor,
            shape_labels, labels_espaciais
        )

        loss = 1.0 - sat_total
        loss.backward()
        optimizer.step()

        historico.append(sat_total.item())

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"  Época {epoch:4d} | SatAgg={sat_total.item():.4f} | Loss={loss.item():.4f}")

    print(f"\n  Satisfatibilidade final por axioma:")
    print(f"  {'─'*45}")
    for nome, val in detalhes.items():
        status = "✓" if val >= 0.7 else ("~" if val >= 0.4 else "✗")
        print(f"  {status}  {nome:<28}: {val:.4f}")

    return predicados, historico, detalhes


# ═════════════════════════════════════════════
# BLOCO 7 — MÉTRICAS DE CLASSIFICAÇÃO
# ═════════════════════════════════════════════

def calcular_metricas_t2(predicados, objects_tensor,
                          labels_espaciais, shape_labels):
    """
    Calcula Acurácia, Precisão, Recall e F1 para cada predicado espacial.
    """
    (left_labels, right_labels, close_labels,
     below_labels, above_labels, between_labels) = labels_espaciais

    n = len(objects_tensor)

    print(f"\n{'='*60}")
    print(f"  MÉTRICAS DE CLASSIFICAÇÃO (threshold={THRESHOLD})")
    print(f"{'='*60}")

    resultados = {}

    def metricas_binario(pred_name, label_dict, todos_pares, threshold=THRESHOLD):
        """Calcula métricas para predicado binário."""
        y_true, y_pred_scores = [], []

        for (i, j) in todos_pares:
            oi = ltn.Constant(objects_tensor[i])
            oj = ltn.Constant(objects_tensor[j])
            with torch.no_grad():
                score = predicados[pred_name](oi, oj).value.item()
            y_true.append(float(label_dict[(i,j)]))
            y_pred_scores.append(score)

        y_true  = np.array(y_true)
        y_pred  = (np.array(y_pred_scores) > threshold).astype(float)

        TP = ((y_pred == 1) & (y_true == 1)).sum()
        TN = ((y_pred == 0) & (y_true == 0)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        FN = ((y_pred == 0) & (y_true == 1)).sum()

        acuracia = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0
        precisao = TP/(TP+FP)             if (TP+FP)>0        else 0
        recall   = TP/(TP+FN)             if (TP+FN)>0        else 0
        f1       = (2*precisao*recall/(precisao+recall)
                    if (precisao+recall)>0 else 0)

        print(f"\n  {pred_name}:")
        print(f"    TP={TP:3} TN={TN:3} FP={FP:3} FN={FN:3}")
        print(f"    Acurácia={acuracia:.4f} | Precisão={precisao:.4f} "
              f"| Recall={recall:.4f} | F1={f1:.4f}")

        return {"Acurácia":acuracia,"Precisão":precisao,
                "Recall":recall,"F1":f1}

    # Pares de avaliação (excluindo i==j)
    pares = [(i,j) for i in range(n) for j in range(n) if i!=j]

    resultados["leftOf"]  = metricas_binario("leftOf",  left_labels,  pares)
    resultados["rightOf"] = metricas_binario("rightOf", right_labels, pares)
    resultados["closeTo"] = metricas_binario("closeTo", close_labels, pares)
    resultados["below"]   = metricas_binario("below",   below_labels, pares)
    resultados["above"]   = metricas_binario("above",   above_labels, pares)

    # canStack
    can_stack_labels = {}
    for i in range(n):
        for j in range(n):
            j_shape = shape_labels[j]
            can_stack_labels[(i,j)] = (i != j and
                                       j_shape not in ["Cone","Triangle"])
    resultados["canStack"] = metricas_binario("canStack", can_stack_labels, pares)

    # InBetween (ternário — amostra menor para eficiência)
    trios = [(i,j,k) for i in range(n) for j in range(n) for k in range(n)
             if i!=j and i!=k and j!=k]
    np.random.shuffle(trios)
    trios = trios[:300]

    y_true_bt, y_pred_bt = [], []
    for (i,j,k) in trios:
        oi = ltn.Constant(objects_tensor[i])
        oj = ltn.Constant(objects_tensor[j])
        ok = ltn.Constant(objects_tensor[k])
        with torch.no_grad():
            score = predicados["inBetween"](oi, oj, ok).value.item()
        y_true_bt.append(float(between_labels[(i,j,k)]))
        y_pred_bt.append(score)

    y_true_bt = np.array(y_true_bt)
    y_pred_bt_bin = (np.array(y_pred_bt) > THRESHOLD).astype(float)
    TP = ((y_pred_bt_bin==1)&(y_true_bt==1)).sum()
    TN = ((y_pred_bt_bin==0)&(y_true_bt==0)).sum()
    FP = ((y_pred_bt_bin==1)&(y_true_bt==0)).sum()
    FN = ((y_pred_bt_bin==0)&(y_true_bt==1)).sum()
    ac = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0
    pr = TP/(TP+FP) if (TP+FP)>0 else 0
    rc = TP/(TP+FN) if (TP+FN)>0 else 0
    f1 = 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0
    print(f"\n  inBetween (ternário, {len(trios)} trios):")
    print(f"    TP={TP:3} TN={TN:3} FP={FP:3} FN={FN:3}")
    print(f"    Acurácia={ac:.4f} | Precisão={pr:.4f} | Recall={rc:.4f} | F1={f1:.4f}")
    resultados["inBetween"] = {"Acurácia":ac,"Precisão":pr,"Recall":rc,"F1":f1}

    return resultados


# ═════════════════════════════════════════════
# BLOCO 8 — CONSULTAS ESPECIAIS
# ═════════════════════════════════════════════

def executar_consultas(predicados, objects_tensor, shape_labels, labels_espaciais):
    """
    Demonstra consultas lógicas específicas pedidas no enunciado.

    CONSULTAS:
    (a) Opcional: "Existe ao menos 1 objeto à Esquerda de todos os Quadrados?"
        ∃x (∀y (IsSquare(y) → LeftOf(x,y)))

    (b) Opcional: "Todo Quadrado está à Direita de todo Círculo?"
        ∀x,y (IsSquare(x) ∧ IsCircle(y)) → RightOf(x,y)

    (c) "Qual é o objeto mais à esquerda?"
    (d) "Qual é o objeto mais à direita?"
    (e) "Quais pares de objetos estão próximos?"
    (f) "Qual objeto está no meio de outros dois?"
    """
    print(f"\n{'='*60}")
    print(f"  CONSULTAS LÓGICAS")
    print(f"{'='*60}")

    n   = len(objects_tensor)
    p   = predicados
    (left_labels, right_labels, close_labels,
     below_labels, above_labels, between_labels) = labels_espaciais

    with torch.no_grad():

        # ── (c) Objeto mais à esquerda ──────────────────────
        scores_left = []
        for i in range(n):
            oi = ltn.Constant(objects_tensor[i])
            s  = p["lastOnTheLeft"](oi).value.item()
            scores_left.append(s)
        idx_ll = np.argmax(scores_left)
        print(f"\n  (c) Objeto mais à ESQUERDA:")
        print(f"      → Objeto #{idx_ll} | score={scores_left[idx_ll]:.4f}"
              f" | shape={shape_labels[idx_ll]}"
              f" | x={objects_tensor[idx_ll,0]:.3f}")
        print(f"      (geometricamente: #{idx_mais_a_esquerda(objects_tensor)}"
              f" | x={objects_tensor[idx_mais_a_esquerda(objects_tensor),0]:.3f})")

        # ── (d) Objeto mais à direita ───────────────────────
        scores_right = []
        for i in range(n):
            oi = ltn.Constant(objects_tensor[i])
            s  = p["lastOnTheRight"](oi).value.item()
            scores_right.append(s)
        idx_lr = np.argmax(scores_right)
        print(f"\n  (d) Objeto mais à DIREITA:")
        print(f"      → Objeto #{idx_lr} | score={scores_right[idx_lr]:.4f}"
              f" | shape={shape_labels[idx_lr]}"
              f" | x={objects_tensor[idx_lr,0]:.3f}")
        print(f"      (geometricamente: #{idx_mais_a_direita(objects_tensor)}"
              f" | x={objects_tensor[idx_mais_a_direita(objects_tensor),0]:.3f})")

        # ── (e) Pares próximos ──────────────────────────────
        print(f"\n  (e) Pares PRÓXIMOS (closeTo > {THRESHOLD}):")
        proximos = []
        for i in range(n):
            for j in range(i+1, n):
                oi = ltn.Constant(objects_tensor[i])
                oj = ltn.Constant(objects_tensor[j])
                s  = p["closeTo"](oi, oj).value.item()
                if s > THRESHOLD:
                    proximos.append((i, j, s))
        if proximos:
            for (i, j, s) in sorted(proximos, key=lambda t: -t[2])[:5]:
                print(f"      Obj #{i} ↔ #{j} | score={s:.4f}"
                      f" | real={close_labels.get((i,j), '?')}")
        else:
            print(f"      Nenhum par encontrado acima do threshold.")

        # ── (f) Objeto no meio ──────────────────────────────
        print(f"\n  (f) Objetos inBetween (score > {THRESHOLD}):")
        between_found = []
        sample = [(i,j,k) for i in range(n)
                           for j in range(n)
                           for k in range(n)
                           if i!=j and i!=k and j!=k]
        np.random.shuffle(sample)
        for (i,j,k) in sample[:500]:
            oi = ltn.Constant(objects_tensor[i])
            oj = ltn.Constant(objects_tensor[j])
            ok = ltn.Constant(objects_tensor[k])
            s  = p["inBetween"](oi, oj, ok).value.item()
            if s > THRESHOLD:
                between_found.append((i,j,k,s))
        if between_found:
            for (i,j,k,s) in sorted(between_found, key=lambda t: -t[3])[:3]:
                print(f"      Obj #{i} entre #{j} e #{k} | score={s:.4f}"
                      f" | real={between_labels.get((i,j,k),'?')}")
        else:
            print(f"      Nenhum trio encontrado.")


# ═════════════════════════════════════════════
# BLOCO 9 — VISUALIZAÇÃO DOS RESULTADOS
# ═════════════════════════════════════════════

def plot_relacoes_espaciais(objects_tensor, predicados,
                             shape_labels, color_labels):
    """
    Plota a cena com setas indicando as relações leftOf aprendidas
    (apenas os pares com score > 0.8 para não poluir o gráfico).
    """
    marker_map = {"Circle":"o","Square":"s","Cylinder":"D","Cone":"^","Triangle":"v"}
    color_map  = {"Red":"red","Green":"green","Blue":"blue"}
    n    = len(objects_tensor)
    data = objects_tensor.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (pred_name, title, lbl_fn) in enumerate([
        ("leftOf",  "Relação: leftOf aprendida",  None),
        ("closeTo", "Relação: closeTo aprendida", None),
    ]):
        ax = axes[ax_idx]
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

        # Plota objetos
        for i in range(n):
            ms = 200 if data[i,10]==1.0 else 80
            ax.scatter(data[i,0], data[i,1],
                       c=color_map[color_labels[i]],
                       marker=marker_map[shape_labels[i]],
                       s=ms, edgecolors='black', linewidths=0.8, zorder=3)
            ax.annotate(str(i), (data[i,0], data[i,1]),
                        textcoords="offset points", xytext=(5,3), fontsize=7)

        # Plota relações aprendidas
        with torch.no_grad():
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    oi = ltn.Constant(objects_tensor[i])
                    oj = ltn.Constant(objects_tensor[j])
                    s  = predicados[pred_name](oi, oj).value.item()
                    if s > 0.80:
                        dx = data[j,0] - data[i,0]
                        dy = data[j,1] - data[i,1]
                        ax.annotate("",
                            xy=(data[j,0], data[j,1]),
                            xytext=(data[i,0], data[i,1]),
                            arrowprops=dict(
                                arrowstyle="->",
                                color="steelblue" if pred_name=="leftOf" else "orange",
                                alpha=min(1.0, s),
                                lw=1.2
                            )
                        )

    plt.tight_layout()
    plt.savefig("relacoes_espaciais.png", dpi=150)
    plt.show()
    print("→ relacoes_espaciais.png salva.")


def plot_historico_t2(historico):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(historico, color="steelblue", lw=1.5)
    ax.set_xlabel("Época"); ax.set_ylabel("SatAgg")
    ax.set_title("Satisfatibilidade durante Treinamento — Tarefa 2")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("historico_t2.png", dpi=150)
    plt.show()
    print("→ historico_t2.png salva.")


# ═════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL
# ═════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  TAREFA 2: RACIOCÍNIO ESPACIAL COM LTN")
    print("="*60)

    # 1. Dados
    print("\n[1/5] Gerando objetos...")
    objects_tensor, shape_labels, color_labels = generate_objects(N_OBJECTS, seed=SEED)
    size_labels = objects_tensor[:, 10].tolist()

    # 2. Labels geométricos (ground truth a partir das coords)
    print("[2/5] Calculando labels espaciais geométricos...")
    labels_espaciais = gerar_labels_espaciais(objects_tensor, sigma=0.15)
    left_labels = labels_espaciais[0]
    n_left = sum(1 for v in left_labels.values() if v)
    print(f"      Pares leftOf verdadeiros: {n_left} / {N_OBJECTS*(N_OBJECTS-1)}")

    # 3. Plot da cena
    print("[3/5] Plotando cena...")
    plot_scene(objects_tensor, shape_labels, color_labels)

    # 4. Treino
    print("[4/5] Treinando...")
    x, y, z, predicados = setup_ltn_t2(objects_tensor)
    predicados_treinados, historico, detalhes_finais = treinar_t2(
        objects_tensor, shape_labels, labels_espaciais,
        epochs=EPOCHS, lr=LR
    )

    # 5. Métricas
    print("[5/5] Avaliando métricas...")
    metricas = calcular_metricas_t2(
        predicados_treinados, objects_tensor,
        labels_espaciais, shape_labels
    )

    # 6. Consultas
    executar_consultas(
        predicados_treinados, objects_tensor,
        shape_labels, labels_espaciais
    )

    # 7. Plots
    plot_relacoes_espaciais(
        objects_tensor, predicados_treinados,
        shape_labels, color_labels
    )
    plot_historico_t2(historico)

    print("\n" + "="*60)
    print("  TAREFA 2 CONCLUÍDA!")
    print("  Arquivos gerados:")
    print("    → cena_espacial.png")
    print("    → relacoes_espaciais.png")
    print("    → historico_t2.png")
    print("="*60)
