"""
=============================================================================
TRABALHO FINAL - FIA (ICC260)
Tarefa 4: Raciocínio Composto com Logic Tensor Networks (LTNtorch)
=============================================================================

CONSULTAS IMPLEMENTADAS:

  Q1 — Filtragem Composta:
       "Existe algum objeto Pequeno que esteja Abaixo de um Cilindro
        E à Esquerda de um Quadrado?"
       ∃x (IsSmall(x) ∧ ∃y(IsCylinder(y) ∧ Below(x,y)) ∧ ∃z(IsSquare(z) ∧ LeftOf(x,z)))

  Q2 — Dedução de Posição Absoluta:
       "Existe um Cone Verde que está InBetween dois outros objetos quaisquer?"
       ∃x,y,z (IsCone(x) ∧ IsGreen(x) ∧ InBetween(x,y,z))

  Q3 — Restrição de Proximidade:
       "Se dois objetos são Triângulos e estão Próximos, devem ter o mesmo Tamanho."
       ∀x,y (IsTriangle(x) ∧ IsTriangle(y) ∧ CloseTo(x,y)) → SameSize(x,y)

NOTA: Esta tarefa REUTILIZA os predicados treinados nas Tarefas 1 e 2.
      A ideia central do raciocínio composto é COMBINAR predicados já
      aprendidos em fórmulas mais complexas — sem retreinar nada.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ltn

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
SEED      = 42
N_OBJECTS = 25
EPOCHS    = 600
LR        = 0.001
THRESHOLD = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)

SHAPES = ["Circle", "Square", "Cylinder", "Cone", "Triangle"]
COLORS = ["Red", "Green", "Blue"]


# ═════════════════════════════════════════════
# BLOCO 1 — GERAÇÃO DE DADOS
# (idêntica às tarefas anteriores — centralizada aqui para o arquivo ser
#  auto-contido e executável de forma independente)
# ═════════════════════════════════════════════

def generate_objects(n=25, seed=None):
    if seed is not None:
        np.random.seed(seed)
    data, shape_labels, color_labels = [], [], []
    for _ in range(n):
        x        = np.random.uniform(0.0, 1.0)
        y        = np.random.uniform(0.0, 1.0)
        size     = float(np.random.randint(0, 2))
        color_idx = np.random.randint(0, 3)
        color_vec = [0.0]*3; color_vec[color_idx] = 1.0
        shape_idx = np.random.randint(0, 5)
        shape_vec = [0.0]*5; shape_vec[shape_idx] = 1.0
        data.append([x, y] + color_vec + shape_vec + [size])
        shape_labels.append(SHAPES[shape_idx])
        color_labels.append(COLORS[color_idx])
    return torch.tensor(data, dtype=torch.float32), shape_labels, color_labels


def gerar_labels_espaciais(objects_tensor, sigma=0.15):
    n      = len(objects_tensor)
    coords = objects_tensor[:, :2].numpy()
    left_labels, right_labels = {}, {}
    close_labels              = {}
    below_labels, above_labels = {}, {}
    between_labels            = {}
    for i in range(n):
        for j in range(n):
            xi, xj = coords[i,0], coords[j,0]
            yi, yj = coords[i,1], coords[j,1]
            dist   = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            left_labels[(i,j)]  = bool(xi < xj)
            right_labels[(i,j)] = bool(xi > xj)
            close_labels[(i,j)] = bool(dist < sigma and i != j)
            below_labels[(i,j)] = bool(yi < yj)
            above_labels[(i,j)] = bool(yi > yj)
        for j in range(n):
            for k in range(n):
                if i==j or i==k or j==k:
                    between_labels[(i,j,k)] = False; continue
                xj, xk   = coords[j,0], coords[k,0]
                x_min, x_max = min(xj,xk), max(xj,xk)
                between_labels[(i,j,k)] = bool(x_min < coords[i,0] < x_max)
    return (left_labels, right_labels, close_labels,
            below_labels, above_labels, between_labels)


# ═════════════════════════════════════════════
# BLOCO 2 — DEFINIÇÃO DE TODOS OS PREDICADOS
# (Tarefas 1 + 2 + 4 juntos num único lugar)
# ═════════════════════════════════════════════

class MLPUnario(nn.Module):
    def __init__(self, inp=11, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp,h), nn.ReLU(),
                                  nn.Linear(h,1),   nn.Sigmoid())
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


def criar_todos_predicados():
    """
    Instancia TODOS os predicados do trabalho num único dicionário.
    Retorna o dict para ser passado para treino e consultas.
    """
    return {
        # ── Tarefa 1: Forma e Tamanho ──────────────────────
        "isCircle":      ltn.Predicate(MLPUnario()),
        "isSquare":      ltn.Predicate(MLPUnario()),
        "isCylinder":    ltn.Predicate(MLPUnario()),
        "isCone":        ltn.Predicate(MLPUnario()),
        "isTriangle":    ltn.Predicate(MLPUnario()),
        "isSmall":       ltn.Predicate(MLPUnario()),
        "isBig":         ltn.Predicate(MLPUnario()),

        # ── Tarefa 1: Cor ───────────────────────────────────
        # (necessário para Q2: "Cone Verde")
        "isRed":         ltn.Predicate(MLPUnario()),
        "isGreen":       ltn.Predicate(MLPUnario()),
        "isBlue":        ltn.Predicate(MLPUnario()),

        # ── Tarefa 2: Espacial Horizontal ───────────────────
        "leftOf":        ltn.Predicate(MLPBinario()),
        "rightOf":       ltn.Predicate(MLPBinario()),
        "closeTo":       ltn.Predicate(MLPBinario()),
        "inBetween":     ltn.Predicate(MLPTernario()),
        "lastOnTheLeft": ltn.Predicate(MLPUnario()),
        "lastOnTheRight":ltn.Predicate(MLPUnario()),

        # ── Tarefa 2: Espacial Vertical ─────────────────────
        "below":         ltn.Predicate(MLPBinario()),
        "above":         ltn.Predicate(MLPBinario()),
        "canStack":      ltn.Predicate(MLPBinario()),

        # ── Tarefa 4 (novo): Mesmo Tamanho ──────────────────
        # sameSize(x,y): ambos têm o mesmo tamanho (0.0 ou 1.0)
        # Aprendido pela supervisão: sameSize(i,j) = True se size_i == size_j
        "sameSize":      ltn.Predicate(MLPBinario()),
    }


# ═════════════════════════════════════════════
# BLOCO 3 — KB UNIFICADA
# Todos os axiomas das tarefas 1, 2 e 4 juntos
# ═════════════════════════════════════════════

def compute_kb_unificada(x, y, z, p, objects_tensor,
                          shape_labels, color_labels,
                          size_labels, labels_espaciais):
    """
    Calcula SatAgg da KB completa (Tarefas 1 + 2 + 4).

    Estrutura:
        Axiomas Lógicos (regras do mundo)   → sem labels, só lógica
        Supervisão por Labels (dados reais) → ancora o aprendizado
    """
    Not    = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And    = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Or     = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
    Impl   = ltn.Connective(ltn.fuzzy_ops.ImpliesLuk())
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2),      quantifier="e")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    (left_labels, right_labels, close_labels,
     below_labels, above_labels, between_labels) = labels_espaciais

    n = len(objects_tensor)

    # ══════════════════════════════════════════
    # AXIOMAS LÓGICOS
    # ══════════════════════════════════════════

    # ── T1: Unicidade e completude de forma ──
    ax_unic_forma = Forall(x, And(
        And(Not(And(p["isCircle"](x),   p["isSquare"](x))),
            Not(And(p["isCircle"](x),   p["isCylinder"](x)))),
        And(Not(And(p["isCircle"](x),   p["isCone"](x))),
            Not(And(p["isSquare"](x),   p["isCylinder"](x))))
    ))
    ax_comp_forma = Forall(x, Or(Or(Or(
        p["isCircle"](x), p["isSquare"](x)),
        Or(p["isCylinder"](x), p["isCone"](x))),
        p["isTriangle"](x)
    ))

    # ── T1: Unicidade e completude de tamanho ─
    ax_unic_tam = Forall(x, Not(And(p["isSmall"](x), p["isBig"](x))))
    ax_comp_tam = Forall(x, Or(p["isSmall"](x), p["isBig"](x)))

    # ── T1: Unicidade de cor ──────────────────
    ax_unic_cor = Forall(x, And(
        Not(And(p["isRed"](x), p["isGreen"](x))),
        Not(And(p["isRed"](x), p["isBlue"](x)))
    ))

    # ── T2: Irreflexividade ───────────────────
    ax_irr = Forall(x, Not(p["leftOf"](x, x)))

    # ── T2: Assimetria ────────────────────────
    ax_assim = Forall([x, y], Impl(p["leftOf"](x,y), Not(p["leftOf"](y,x))))

    # ── T2: Inverso horizontal ────────────────
    ax_inv_h = Forall([x, y], And(
        Impl(p["leftOf"](x,y),  p["rightOf"](y,x)),
        Impl(p["rightOf"](y,x), p["leftOf"](x,y))
    ))

    # ── T2: Transitividade horizontal ─────────
    ax_trans_h = Forall([x, y, z], Impl(
        And(p["leftOf"](x,y), p["leftOf"](y,z)),
        p["leftOf"](x,z)
    ))

    # ── T2: Inverso vertical ──────────────────
    ax_inv_v = Forall([x, y], And(
        Impl(p["below"](x,y), p["above"](y,x)),
        Impl(p["above"](y,x), p["below"](x,y))
    ))

    # ── T2: Transitividade vertical ───────────
    ax_trans_v = Forall([x, y, z], Impl(
        And(p["below"](x,y), p["below"](y,z)),
        p["below"](x,z)
    ))

    # ══════════════════════════════════════════
    # AXIOMAS DA TAREFA 4
    # ══════════════════════════════════════════

    # ── T4/Q3: Restrição de Proximidade ──────
    #
    # ∀x,y (IsTriangle(x) ∧ IsTriangle(y) ∧ CloseTo(x,y)) → SameSize(x,y)
    #
    # Leitura: SE dois triângulos estão próximos,
    #          ENTÃO eles devem ter o mesmo tamanho.
    #
    # Por que isso é interessante para LTN?
    # Porque é uma regra que CORRELACIONA espaço com atributo.
    # O gradiente vai empurrar sameSize(x,y) → 1.0 sempre que
    # x e y forem triângulos próximos.
    ax_q3_proximidade = Forall(
        [x, y],
        Impl(
            And(And(p["isTriangle"](x), p["isTriangle"](y)),
                p["closeTo"](x, y)),
            p["sameSize"](x, y)
        )
    )

    # ── T4: sameSize é simétrico ─────────────
    # ∀x,y  SameSize(x,y) → SameSize(y,x)
    ax_same_size_sim = Forall(
        [x, y],
        Impl(p["sameSize"](x, y), p["sameSize"](y, x))
    )

    # ══════════════════════════════════════════
    # SUPERVISÃO COM LABELS REAIS
    # ══════════════════════════════════════════

    def mean_stack(lst):
        return torch.mean(torch.stack(lst))

    # ── Supervisão de forma ───────────────────
    shape_to_pred = {"Circle":"isCircle","Square":"isSquare",
                     "Cylinder":"isCylinder","Cone":"isCone","Triangle":"isTriangle"}
    color_to_pred = {"Red":"isRed","Green":"isGreen","Blue":"isBlue"}

    sup_forma, sup_cor, sup_tam = [], [], []
    for i in range(n):
        oi = ltn.Constant(objects_tensor[i])
        # Forma correta → 1.0; outras → 0.0
        for sh, pr in shape_to_pred.items():
            v = p[pr](oi).value
            sup_forma.append(v if shape_labels[i]==sh else 1.0-v)
        # Cor
        for co, pr in color_to_pred.items():
            v = p[pr](oi).value
            sup_cor.append(v if color_labels[i]==co else 1.0-v)
        # Tamanho
        v = p["isSmall"](oi).value
        sup_tam.append(v if size_labels[i]==0.0 else 1.0-v)

    # ── Supervisão espacial (amostra de pares) ─
    pares = [(i,j) for i in range(n) for j in range(n) if i!=j]
    np.random.shuffle(pares)
    pares = pares[:120]

    sup_left, sup_right  = [], []
    sup_close            = []
    sup_below, sup_above = [], []
    sup_samesize         = []
    sup_canstack         = []

    for (i,j) in pares:
        oi = ltn.Constant(objects_tensor[i])
        oj = ltn.Constant(objects_tensor[j])

        v = p["leftOf"](oi,oj).value
        sup_left.append(v if left_labels[(i,j)] else 1.0-v)

        v = p["rightOf"](oi,oj).value
        sup_right.append(v if right_labels[(i,j)] else 1.0-v)

        v = p["closeTo"](oi,oj).value
        sup_close.append(v if close_labels[(i,j)] else 1.0-v)

        v = p["below"](oi,oj).value
        sup_below.append(v if below_labels[(i,j)] else 1.0-v)

        v = p["above"](oi,oj).value
        sup_above.append(v if above_labels[(i,j)] else 1.0-v)

        # sameSize: True se os dois têm o mesmo tamanho
        same = (size_labels[i] == size_labels[j])
        v = p["sameSize"](oi,oj).value
        sup_samesize.append(v if same else 1.0-v)

        # canStack: True se j não é Cone nem Triangle
        can = (i != j and shape_labels[j] not in ["Cone","Triangle"])
        v = p["canStack"](oi,oj).value
        sup_canstack.append(v if can else 1.0-v)

    # ── Supervisão inBetween (amostra de trios) ─
    trios = [(i,j,k) for i in range(n)
                      for j in range(n)
                      for k in range(n)
                      if i!=j and i!=k and j!=k]
    np.random.shuffle(trios)
    trios = trios[:80]

    sup_between = []
    for (i,j,k) in trios:
        oi = ltn.Constant(objects_tensor[i])
        oj = ltn.Constant(objects_tensor[j])
        ok = ltn.Constant(objects_tensor[k])
        v  = p["inBetween"](oi,oj,ok).value
        sup_between.append(v if between_labels[(i,j,k)] else 1.0-v)

    # ══════════════════════════════════════════
    # SATAGGG FINAL
    # ══════════════════════════════════════════
    sat_total = SatAgg(
        # Axiomas lógicos
        ax_unic_forma, ax_comp_forma,
        ax_unic_tam,   ax_comp_tam,
        ax_unic_cor,
        ax_irr,        ax_assim,
        ax_inv_h,      ax_trans_h,
        ax_inv_v,      ax_trans_v,
        ax_q3_proximidade,
        ax_same_size_sim,
        # Supervisão
        mean_stack(sup_forma),
        mean_stack(sup_cor),
        mean_stack(sup_tam),
        mean_stack(sup_left),
        mean_stack(sup_right),
        mean_stack(sup_close),
        mean_stack(sup_below),
        mean_stack(sup_above),
        mean_stack(sup_between),
        mean_stack(sup_samesize),
        mean_stack(sup_canstack),
    )

    detalhes = {
        "ax_unicidade_forma":    ax_unic_forma.value.item(),
        "ax_completude_forma":   ax_comp_forma.value.item(),
        "ax_unicidade_tam":      ax_unic_tam.value.item(),
        "ax_completude_tam":     ax_comp_tam.value.item(),
        "ax_unicidade_cor":      ax_unic_cor.value.item(),
        "ax_irreflexividade":    ax_irr.value.item(),
        "ax_assimetria":         ax_assim.value.item(),
        "ax_inverso_h":          ax_inv_h.value.item(),
        "ax_transitividade_h":   ax_trans_h.value.item(),
        "ax_inverso_v":          ax_inv_v.value.item(),
        "ax_transitividade_v":   ax_trans_v.value.item(),
        "ax_Q3_proximidade":     ax_q3_proximidade.value.item(),
        "ax_samesize_simetria":  ax_same_size_sim.value.item(),
    }

    return sat_total, detalhes


# ═════════════════════════════════════════════
# BLOCO 4 — TREINAMENTO UNIFICADO
# ═════════════════════════════════════════════

def treinar_unificado(objects_tensor, shape_labels, color_labels,
                       size_labels, labels_espaciais,
                       epochs=EPOCHS, lr=LR):
    """
    Treina todos os predicados (T1 + T2 + T4) juntos numa única KB.
    """
    predicados = criar_todos_predicados()

    x = ltn.Variable("x", objects_tensor)
    y = ltn.Variable("y", objects_tensor)
    z = ltn.Variable("z", objects_tensor)

    params = []
    for pred in predicados.values():
        params += list(pred.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    historico = []

    print(f"\n{'='*60}")
    print(f"  TREINAMENTO UNIFICADO (Tarefas 1 + 2 + 4)")
    print(f"  {N_OBJECTS} objetos | {epochs} épocas | lr={lr}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        optimizer.zero_grad()

        sat_total, detalhes = compute_kb_unificada(
            x, y, z, predicados, objects_tensor,
            shape_labels, color_labels, size_labels, labels_espaciais
        )

        loss = 1.0 - sat_total
        loss.backward()
        optimizer.step()
        historico.append(sat_total.item())

        if epoch % 100 == 0 or epoch == epochs-1:
            print(f"  Época {epoch:4d} | SatAgg={sat_total.item():.4f} | Loss={loss.item():.4f}")

    print(f"\n  Satisfatibilidade final por axioma:")
    print(f"  {'─'*50}")
    for nome, val in detalhes.items():
        status = "✓" if val >= 0.7 else ("~" if val >= 0.4 else "✗")
        print(f"  {status}  {nome:<30}: {val:.4f}")

    return predicados, historico, detalhes


# ═════════════════════════════════════════════
# BLOCO 5 — CONSULTAS COMPOSTAS (Q1, Q2, Q3)
# ═════════════════════════════════════════════

def executar_consultas_t4(predicados, objects_tensor,
                           shape_labels, color_labels,
                           size_labels, labels_espaciais):
    """
    Avalia as 3 consultas compostas da Tarefa 4 e exibe os resultados.

    ┌──────────────────────────────────────────────────────────────┐
    │  Q1: FILTRAGEM COMPOSTA                                      │
    │  "Existe algum objeto Pequeno abaixo de um Cilindro          │
    │   E à Esquerda de um Quadrado?"                              │
    │                                                              │
    │  ∃x (IsSmall(x)                                             │
    │       ∧ ∃y(IsCylinder(y) ∧ Below(x,y))                     │
    │       ∧ ∃z(IsSquare(z)   ∧ LeftOf(x,z)))                   │
    │                                                              │
    │  COMO AVALIAR NO LTN:                                        │
    │  Para cada candidato x, calculamos o grau de verdade da      │
    │  fórmula inteira. O candidato com maior score é a resposta.  │
    │                                                              │
    │  Q2: DEDUÇÃO DE POSIÇÃO ABSOLUTA                             │
    │  "Existe um Cone Verde entre dois outros objetos?"           │
    │  ∃x,y,z (IsCone(x) ∧ IsGreen(x) ∧ InBetween(x,y,z))       │
    │                                                              │
    │  Q3: VERIFICAÇÃO DE REGRA                                    │
    │  "A regra ∀x,y (Tri∧Tri∧Close → SameSize) é satisfeita?"   │
    │  → Calcula satisfatibilidade e mostra os pares críticos.     │
    └──────────────────────────────────────────────────────────────┘
    """
    p = predicados
    n = len(objects_tensor)

    Not    = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And    = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Impl   = ltn.Connective(ltn.fuzzy_ops.ImpliesLuk())
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2),      quantifier="e")
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")

    (left_labels, right_labels, close_labels,
     below_labels, above_labels, between_labels) = labels_espaciais

    print(f"\n{'='*60}")
    print(f"  CONSULTAS COMPOSTAS — TAREFA 4")
    print(f"{'='*60}")

    # ──────────────────────────────────────────
    # Q1: FILTRAGEM COMPOSTA
    # ──────────────────────────────────────────
    print(f"\n  ┌─ Q1: Filtragem Composta ─────────────────────────")
    print(f"  │  ∃x (IsSmall(x) ∧ ∃y(Cyl(y)∧Below(x,y)) ∧ ∃z(Sq(z)∧Left(x,z)))")

    # Para cada candidato x, calculamos o score da fórmula
    # "IsSmall(x) ∧ ∃y(Cyl(y)∧Below(x,y)) ∧ ∃z(Sq(z)∧Left(x,z))"
    scores_q1 = []

    with torch.no_grad():
        for i in range(n):
            oi = ltn.Constant(objects_tensor[i])

            # Componente 1: IsSmall(x)
            small_i = p["isSmall"](oi).value

            # Componente 2: ∃y (IsCylinder(y) ∧ Below(x, y))
            # Para cada candidato y, calcula Cyl(y) ∧ Below(x,y)
            # depois agrega com Exists (máximo suave)
            scores_cyl_below = []
            for j in range(n):
                if i == j: continue
                oj = ltn.Constant(objects_tensor[j])
                cyl_j     = p["isCylinder"](oj).value
                below_ij  = p["below"](oi, oj).value
                scores_cyl_below.append((cyl_j * below_ij).squeeze())

            exist_cyl_below = torch.max(torch.stack(scores_cyl_below))

            # Componente 3: ∃z (IsSquare(z) ∧ LeftOf(x, z))
            scores_sq_left = []
            for k in range(n):
                if i == k: continue
                ok = ltn.Constant(objects_tensor[k])
                sq_k     = p["isSquare"](ok).value
                left_ik  = p["leftOf"](oi, ok).value
                scores_sq_left.append((sq_k * left_ik).squeeze())

            exist_sq_left = torch.max(torch.stack(scores_sq_left))

            # Score final para o candidato i:
            # IsSmall(i) ∧ ∃y(...) ∧ ∃z(...)
            # Em lógica produto: multiplicação
            score_i = (small_i.squeeze()
                       * exist_cyl_below
                       * exist_sq_left)
            scores_q1.append(score_i.item())

    # Melhor candidato
    melhor_i  = int(np.argmax(scores_q1))
    melhor_sc = scores_q1[melhor_i]

    # Verificação geométrica
    candidatos_reais = []
    for i in range(n):
        if size_labels[i] != 0.0: continue          # deve ser pequeno
        tem_cyl_abaixo = any(
            shape_labels[j]=="Cylinder" and below_labels[(i,j)]
            for j in range(n) if j!=i
        )
        tem_sq_esquerda = any(
            shape_labels[k]=="Square" and left_labels[(i,k)]
            for k in range(n) if k!=i
        )
        if tem_cyl_abaixo and tem_sq_esquerda:
            candidatos_reais.append(i)

    print(f"  │")
    print(f"  │  Candidato com maior score LTN:")
    print(f"  │    Objeto #{melhor_i} | score={melhor_sc:.4f}"
          f" | shape={shape_labels[melhor_i]}"
          f" | size={'Pequeno' if size_labels[melhor_i]==0 else 'Grande'}"
          f" | cor={color_labels[melhor_i]}")
    print(f"  │")
    print(f"  │  Candidatos geométricos reais: {candidatos_reais if candidatos_reais else 'Nenhum'}")

    if candidatos_reais:
        acertou = melhor_i in candidatos_reais
        print(f"  │  Resposta LTN {'✓ CORRETA' if acertou else '✗ INCORRETA'}")
    else:
        print(f"  │  (Não há objeto que satisfaça a condição nesta cena)")
    print(f"  └{'─'*50}")

    # ──────────────────────────────────────────
    # Q2: DEDUÇÃO DE POSIÇÃO ABSOLUTA
    # ──────────────────────────────────────────
    print(f"\n  ┌─ Q2: Dedução de Posição Absoluta ───────────────")
    print(f"  │  ∃x,y,z (IsCone(x) ∧ IsGreen(x) ∧ InBetween(x,y,z))")

    scores_q2 = []   # (score, i, j, k)

    with torch.no_grad():
        # Para eficiência, filtramos primeiro os candidatos x que são Cone Verde
        candidatos_cone_verde = [
            i for i in range(n)
            if shape_labels[i]=="Cone" and color_labels[i]=="Green"
        ]

        if not candidatos_cone_verde:
            print(f"  │  Não há Cone Verde nesta cena.")
            print(f"  │  Verificando quais objetos são Cones e/ou Verdes...")
            cones   = [i for i in range(n) if shape_labels[i]=="Cone"]
            verdes  = [i for i in range(n) if color_labels[i]=="Green"]
            print(f"  │    Cones: {cones} | Verdes: {verdes}")
        else:
            for i in candidatos_cone_verde:
                oi       = ltn.Constant(objects_tensor[i])
                cone_i   = p["isCone"](oi).value.squeeze()
                green_i  = p["isGreen"](oi).value.squeeze()

                for j in range(n):
                    if j == i: continue
                    for k in range(n):
                        if k == i or k == j: continue
                        oj = ltn.Constant(objects_tensor[j])
                        ok = ltn.Constant(objects_tensor[k])
                        between_ijk = p["inBetween"](oi,oj,ok).value.squeeze()

                        score = (cone_i * green_i * between_ijk).item()
                        scores_q2.append((score, i, j, k))

        if scores_q2:
            scores_q2.sort(key=lambda t: -t[0])
            top = scores_q2[0]
            print(f"  │  Melhor tripla encontrada:")
            print(f"  │    x=#{top[1]} (Cone Verde) está entre"
                  f" y=#{top[2]} e z=#{top[3]}")
            print(f"  │    Score LTN = {top[0]:.4f}")
            print(f"  │    Verificação geométrica: {between_labels.get((top[1],top[2],top[3]),'?')}")

            # Satisfatibilidade global da fórmula ∃x,y,z(...)
            # como média dos top scores
            top5_scores = [s[0] for s in scores_q2[:5]]
            sat_q2 = max(top5_scores)
            print(f"  │  SatAgg Q2 = {sat_q2:.4f}")
        else:
            print(f"  │  Nenhuma tripla avaliada.")
    print(f"  └{'─'*50}")

    # ──────────────────────────────────────────
    # Q3: VERIFICAÇÃO DA REGRA DE PROXIMIDADE
    # ──────────────────────────────────────────
    print(f"\n  ┌─ Q3: Restrição de Proximidade ──────────────────")
    print(f"  │  ∀x,y (IsTriangle(x) ∧ IsTriangle(y) ∧ CloseTo(x,y))")
    print(f"  │         → SameSize(x,y)")

    # Avalia a regra para TODOS os pares de triângulos próximos
    triangulos = [i for i in range(n) if shape_labels[i]=="Triangle"]
    print(f"  │  Triângulos na cena: {triangulos}")

    sat_q3_valores = []
    pares_criticos = []   # pares onde a regra é mais "testada"

    with torch.no_grad():
        for i in triangulos:
            for j in triangulos:
                if i == j: continue
                oi = ltn.Constant(objects_tensor[i])
                oj = ltn.Constant(objects_tensor[j])

                tri_i   = p["isTriangle"](oi).value.squeeze()
                tri_j   = p["isTriangle"](oj).value.squeeze()
                close   = p["closeTo"](oi, oj).value.squeeze()
                same_sz = p["sameSize"](oi, oj).value.squeeze()

                # Premissa: IsTriangle(x) ∧ IsTriangle(y) ∧ CloseTo(x,y)
                premissa = tri_i * tri_j * close

                # Implicação de Łukasiewicz: min(1, 1 - premissa + conclusao)
                impl = torch.clamp(1.0 - premissa + same_sz, max=1.0)
                sat_q3_valores.append(impl.item())

                # Par crítico: premissa alta (realmente são triângulos próximos)
                if premissa.item() > 0.3:
                    pares_criticos.append({
                        "i": i, "j": j,
                        "premissa": premissa.item(),
                        "same_size_real": size_labels[i]==size_labels[j],
                        "same_size_score": same_sz.item(),
                        "impl_score": impl.item()
                    })

    if sat_q3_valores:
        sat_q3_media = np.mean(sat_q3_valores)
        print(f"  │  SatAgg Q3 = {sat_q3_media:.4f}")

        if pares_criticos:
            print(f"  │  Pares de triângulos próximos (premissa > 0.3):")
            for pc in pares_criticos:
                real = "✓" if pc["same_size_real"] else "✗"
                print(f"  │    #{pc['i']}↔#{pc['j']} | premissa={pc['premissa']:.3f}"
                      f" | sameSize_score={pc['same_size_score']:.3f}"
                      f" | impl={pc['impl_score']:.3f} | real={real}")
        else:
            print(f"  │  Nenhum par com premissa forte (triângulos não estão próximos).")
            print(f"  │  → Regra satisfeita vacuamente (premissa sempre ≈ 0)")
    else:
        print(f"  │  Nenhum triângulo na cena — regra satisfeita vacuamente.")
    print(f"  └{'─'*50}")

    return {
        "scores_q1": scores_q1,
        "scores_q2": scores_q2[:10] if scores_q2 else [],
    }


# ═════════════════════════════════════════════
# BLOCO 6 — MÉTRICAS COMPLETAS
# ═════════════════════════════════════════════

def calcular_metricas_completas(predicados, objects_tensor,
                                 shape_labels, color_labels,
                                 size_labels, labels_espaciais,
                                 threshold=THRESHOLD):
    """
    Calcula Acurácia, Precisão, Recall e F1 para todos os predicados.
    Retorna tabela formatada para o relatório.
    """
    p = predicados
    n = len(objects_tensor)

    (left_labels, right_labels, close_labels,
     below_labels, above_labels, between_labels) = labels_espaciais

    print(f"\n{'='*60}")
    print(f"  MÉTRICAS COMPLETAS (threshold={threshold})")
    print(f"{'='*60}")

    resultados = {}

    def avaliar_unario(pred_name, y_true_list):
        y_true, y_pred = [], []
        with torch.no_grad():
            for i in range(n):
                oi = ltn.Constant(objects_tensor[i])
                s  = p[pred_name](oi).value.item()
                y_true.append(y_true_list[i])
                y_pred.append(float(s > threshold))
        return _metricas(pred_name, np.array(y_true), np.array(y_pred))

    def avaliar_binario(pred_name, label_dict, pares):
        y_true, y_pred = [], []
        with torch.no_grad():
            for (i,j) in pares:
                oi = ltn.Constant(objects_tensor[i])
                oj = ltn.Constant(objects_tensor[j])
                s  = p[pred_name](oi,oj).value.item()
                y_true.append(float(label_dict[(i,j)]))
                y_pred.append(float(s > threshold))
        return _metricas(pred_name, np.array(y_true), np.array(y_pred))

    def _metricas(nome, y_true, y_pred):
        TP = ((y_pred==1)&(y_true==1)).sum()
        TN = ((y_pred==0)&(y_true==0)).sum()
        FP = ((y_pred==1)&(y_true==0)).sum()
        FN = ((y_pred==0)&(y_true==1)).sum()
        ac = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0
        pr = TP/(TP+FP)             if (TP+FP)>0        else 0
        rc = TP/(TP+FN)             if (TP+FN)>0        else 0
        f1 = 2*pr*rc/(pr+rc)        if (pr+rc)>0        else 0
        print(f"  {nome:<18} | Ac={ac:.3f} Pr={pr:.3f} Re={rc:.3f} F1={f1:.3f}"
              f" | TP={TP:3} TN={TN:3} FP={FP:3} FN={FN:3}")
        return {"Acurácia":ac,"Precisão":pr,"Recall":rc,"F1":f1}

    pares = [(i,j) for i in range(n) for j in range(n) if i!=j]

    print(f"\n  {'Predicado':<18} | {'Ac':>5} {'Pr':>5} {'Re':>5} {'F1':>5}"
          f" | TP   TN   FP   FN")
    print(f"  {'─'*58}")

    # Forma
    for sh, pr_name in [("Circle","isCircle"),("Square","isSquare"),
                         ("Cylinder","isCylinder"),("Cone","isCone"),
                         ("Triangle","isTriangle")]:
        gt = [1.0 if shape_labels[i]==sh else 0.0 for i in range(n)]
        resultados[pr_name] = avaliar_unario(pr_name, gt)

    # Cor
    for co, pr_name in [("Red","isRed"),("Green","isGreen"),("Blue","isBlue")]:
        gt = [1.0 if color_labels[i]==co else 0.0 for i in range(n)]
        resultados[pr_name] = avaliar_unario(pr_name, gt)

    # Tamanho
    gt_small = [1.0 if size_labels[i]==0.0 else 0.0 for i in range(n)]
    gt_big   = [1.0 if size_labels[i]==1.0 else 0.0 for i in range(n)]
    resultados["isSmall"] = avaliar_unario("isSmall", gt_small)
    resultados["isBig"]   = avaliar_unario("isBig",   gt_big)

    # Espaciais binários
    resultados["leftOf"]  = avaliar_binario("leftOf",  left_labels,  pares)
    resultados["rightOf"] = avaliar_binario("rightOf", right_labels, pares)
    resultados["closeTo"] = avaliar_binario("closeTo", close_labels, pares)
    resultados["below"]   = avaliar_binario("below",   below_labels, pares)
    resultados["above"]   = avaliar_binario("above",   above_labels, pares)

    # sameSize
    same_sz_labels = {(i,j): (size_labels[i]==size_labels[j] and i!=j)
                      for i in range(n) for j in range(n)}
    resultados["sameSize"] = avaliar_binario("sameSize", same_sz_labels, pares)

    # canStack
    can_st_labels = {(i,j): (i!=j and shape_labels[j] not in ["Cone","Triangle"])
                     for i in range(n) for j in range(n)}
    resultados["canStack"] = avaliar_binario("canStack", can_st_labels, pares)

    return resultados


# ═════════════════════════════════════════════
# BLOCO 7 — PLOTS
# ═════════════════════════════════════════════

def plot_historico(historico, titulo="Satisfatibilidade — KB Unificada"):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(historico, color="steelblue", lw=1.5)
    ax.set_xlabel("Época"); ax.set_ylabel("SatAgg")
    ax.set_title(titulo); ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Sat. máxima')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("historico_t4.png", dpi=150)
    plt.show()
    print("→ historico_t4.png salvo.")


def plot_scores_q1(scores_q1, shape_labels, size_labels, color_labels):
    """Barras horizontais dos scores Q1 para cada objeto candidato."""
    n  = len(scores_q1)
    ys = list(range(n))
    cores = ["#e74c3c" if s > THRESHOLD else "#95a5a6" for s in scores_q1]
    labels_obj = [
        f"#{i} {shape_labels[i][:3]} "
        f"{'P' if size_labels[i]==0 else 'G'} "
        f"{color_labels[i][:1]}"
        for i in range(n)
    ]
    fig, ax = plt.subplots(figsize=(7, 8))
    bars = ax.barh(ys, scores_q1, color=cores, edgecolor='black', linewidth=0.5)
    ax.set_yticks(ys); ax.set_yticklabels(labels_obj, fontsize=8)
    ax.axvline(THRESHOLD, color='black', ls='--', lw=1, label=f'threshold={THRESHOLD}')
    ax.set_xlabel("Score Q1"); ax.set_title("Q1 — Score por candidato x")
    ax.legend(); ax.set_xlim(0, 1.05); ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig("scores_q1.png", dpi=150)
    plt.show()
    print("→ scores_q1.png salvo.")


def plot_metricas_resumo(resultados):
    """Heatmap com F1 de todos os predicados."""
    nomes = list(resultados.keys())
    f1s   = [resultados[k]["F1"] for k in nomes]

    fig, ax = plt.subplots(figsize=(10, 5))
    cores = ["#2ecc71" if v >= 0.7 else ("#f39c12" if v >= 0.4 else "#e74c3c")
             for v in f1s]
    bars = ax.bar(nomes, f1s, color=cores, edgecolor='black', linewidth=0.5)
    ax.axhline(0.7, color='green',  ls='--', alpha=0.5, label='F1 ≥ 0.7 (bom)')
    ax.axhline(0.4, color='orange', ls='--', alpha=0.5, label='F1 ≥ 0.4 (razoável)')
    ax.set_ylabel("F1 Score"); ax.set_title("F1 por Predicado — Tarefa 4")
    ax.set_ylim(0, 1.05); ax.legend()
    plt.xticks(rotation=40, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.2f}", ha='center', fontsize=7)
    plt.tight_layout()
    plt.savefig("metricas_f1.png", dpi=150)
    plt.show()
    print("→ metricas_f1.png salvo.")


# ═════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL
# ═════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  TAREFA 4: RACIOCÍNIO COMPOSTO COM LTN")
    print("="*60)

    # 1. Dados
    print("\n[1/6] Gerando objetos...")
    objects_tensor, shape_labels, color_labels = generate_objects(N_OBJECTS, seed=SEED)
    size_labels = objects_tensor[:, 10].tolist()

    print("  Distribuição:")
    for sh in SHAPES:
        cnt = shape_labels.count(sh)
        print(f"    {sh:<10}: {'█'*cnt} ({cnt})")

    # 2. Labels geométricos
    print("\n[2/6] Calculando labels espaciais...")
    labels_espaciais = gerar_labels_espaciais(objects_tensor, sigma=0.15)

    # 3. Treino unificado
    print("\n[3/6] Treinando KB unificada...")
    predicados, historico, detalhes = treinar_unificado(
        objects_tensor, shape_labels, color_labels,
        size_labels, labels_espaciais,
        epochs=EPOCHS, lr=LR
    )

    # 4. Consultas compostas
    print("\n[4/6] Executando consultas compostas...")
    resultados_consultas = executar_consultas_t4(
        predicados, objects_tensor,
        shape_labels, color_labels,
        size_labels, labels_espaciais
    )

    # 5. Métricas
    print("\n[5/6] Calculando métricas completas...")
    metricas = calcular_metricas_completas(
        predicados, objects_tensor,
        shape_labels, color_labels,
        size_labels, labels_espaciais
    )

    # 6. Plots
    print("\n[6/6] Gerando gráficos...")
    plot_historico(historico)
    plot_scores_q1(resultados_consultas["scores_q1"],
                   shape_labels, size_labels, color_labels)
    plot_metricas_resumo(metricas)

    print("\n" + "="*60)
    print("  TAREFA 4 CONCLUÍDA!")
    print("  Arquivos gerados:")
    print("    → historico_t4.png")
    print("    → scores_q1.png")
    print("    → metricas_f1.png")
    print("="*60)
