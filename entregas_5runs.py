"""
=============================================================================
TRABALHO FINAL - FIA (ICC260)
SEÇÃO 4 DE ENTREGAS: 5 Execuções com Datasets Distintos
=============================================================================

Este script executa o pipeline completo (Tarefas 3.2 + 3.3) 5 vezes,
cada vez com um dataset aleatório diferente, e consolida os resultados
em tabelas prontas para o relatório do GitHub.

SAÍDAS GERADAS:
    → resultados_5runs.md     (tabela Markdown para o relatório)
    → grafico_5runs.png       (gráfico comparativo das métricas)
    → historicos_5runs.png    (curvas de SatAgg das 5 execuções)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import ltn

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
N_OBJECTS = 25
EPOCHS    = 500     # reduzido para rodar 5x mais rápido
LR        = 0.001
THRESHOLD = 0.5
N_RUNS    = 5

# Seeds diferentes para garantir datasets distintos
SEEDS = [42, 7, 123, 999, 2024]

SHAPES = ["Circle", "Square", "Cylinder", "Cone", "Triangle"]
COLORS = ["Red", "Green", "Blue"]


# ═════════════════════════════════════════════
# BLOCO 1 — GERAÇÃO DE DADOS
# ═════════════════════════════════════════════

def generate_objects(n=25, seed=None):
    if seed is not None:
        np.random.seed(seed)
    data, shape_labels, color_labels = [], [], []
    for _ in range(n):
        x         = np.random.uniform(0.0, 1.0)
        y         = np.random.uniform(0.0, 1.0)
        size      = float(np.random.randint(0, 2))
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
    close_labels = {}
    below_labels, above_labels = {}, {}
    between_labels = {}
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
                xj, xk = coords[j,0], coords[k,0]
                x_min, x_max = min(xj,xk), max(xj,xk)
                between_labels[(i,j,k)] = bool(x_min < coords[i,0] < x_max)
    return (left_labels, right_labels, close_labels,
            below_labels, above_labels, between_labels)


# ═════════════════════════════════════════════
# BLOCO 2 — PREDICADOS
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


def criar_predicados():
    return {
        "isCircle":      ltn.Predicate(MLPUnario()),
        "isSquare":      ltn.Predicate(MLPUnario()),
        "isCylinder":    ltn.Predicate(MLPUnario()),
        "isCone":        ltn.Predicate(MLPUnario()),
        "isTriangle":    ltn.Predicate(MLPUnario()),
        "isSmall":       ltn.Predicate(MLPUnario()),
        "isBig":         ltn.Predicate(MLPUnario()),
        "isRed":         ltn.Predicate(MLPUnario()),
        "isGreen":       ltn.Predicate(MLPUnario()),
        "isBlue":        ltn.Predicate(MLPUnario()),
        "leftOf":        ltn.Predicate(MLPBinario()),
        "rightOf":       ltn.Predicate(MLPBinario()),
        "closeTo":       ltn.Predicate(MLPBinario()),
        "inBetween":     ltn.Predicate(MLPTernario()),
        "lastOnTheLeft": ltn.Predicate(MLPUnario()),
        "lastOnTheRight":ltn.Predicate(MLPUnario()),
        "below":         ltn.Predicate(MLPBinario()),
        "above":         ltn.Predicate(MLPBinario()),
        "canStack":      ltn.Predicate(MLPBinario()),
        "sameSize":      ltn.Predicate(MLPBinario()),
    }


# ═════════════════════════════════════════════
# BLOCO 3 — KB COMPLETA
# ═════════════════════════════════════════════

def compute_kb(x_var, y_var, z_var, p, objects_tensor,
               shape_labels, color_labels, size_labels, labels_espaciais):

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

    # ── Axiomas lógicos ───────────────────────
    ax_unic_forma = Forall(x_var, And(
        And(Not(And(p["isCircle"](x_var), p["isSquare"](x_var))),
            Not(And(p["isCircle"](x_var), p["isCylinder"](x_var)))),
        And(Not(And(p["isCircle"](x_var), p["isCone"](x_var))),
            Not(And(p["isSquare"](x_var), p["isCylinder"](x_var))))
    ))
    ax_comp_forma = Forall(x_var, Or(Or(Or(
        p["isCircle"](x_var), p["isSquare"](x_var)),
        Or(p["isCylinder"](x_var), p["isCone"](x_var))),
        p["isTriangle"](x_var)))
    ax_unic_tam   = Forall(x_var, Not(And(p["isSmall"](x_var), p["isBig"](x_var))))
    ax_comp_tam   = Forall(x_var, Or(p["isSmall"](x_var), p["isBig"](x_var)))
    ax_irr        = Forall(x_var, Not(p["leftOf"](x_var, x_var)))
    ax_assim      = Forall([x_var, y_var],
                        Impl(p["leftOf"](x_var,y_var), Not(p["leftOf"](y_var,x_var))))
    ax_inv_h      = Forall([x_var, y_var], And(
                        Impl(p["leftOf"](x_var,y_var),  p["rightOf"](y_var,x_var)),
                        Impl(p["rightOf"](y_var,x_var), p["leftOf"](x_var,y_var))))
    ax_trans_h    = Forall([x_var, y_var, z_var],
                        Impl(And(p["leftOf"](x_var,y_var), p["leftOf"](y_var,z_var)),
                             p["leftOf"](x_var,z_var)))
    ax_inv_v      = Forall([x_var, y_var], And(
                        Impl(p["below"](x_var,y_var), p["above"](y_var,x_var)),
                        Impl(p["above"](y_var,x_var), p["below"](x_var,y_var))))
    ax_trans_v    = Forall([x_var, y_var, z_var],
                        Impl(And(p["below"](x_var,y_var), p["below"](y_var,z_var)),
                             p["below"](x_var,z_var)))

    # Q3: Triângulos próximos têm mesmo tamanho
    ax_q3 = Forall([x_var, y_var],
                Impl(And(And(p["isTriangle"](x_var), p["isTriangle"](y_var)),
                         p["closeTo"](x_var, y_var)),
                     p["sameSize"](x_var, y_var)))

    ax_same_sim = Forall([x_var, y_var],
                     Impl(p["sameSize"](x_var,y_var), p["sameSize"](y_var,x_var)))

    # ── Supervisão ───────────────────────────
    def mean_stack(lst):
        return torch.mean(torch.stack(lst))

    shape_to_pred = {"Circle":"isCircle","Square":"isSquare",
                     "Cylinder":"isCylinder","Cone":"isCone","Triangle":"isTriangle"}
    color_to_pred = {"Red":"isRed","Green":"isGreen","Blue":"isBlue"}

    sup_forma, sup_cor, sup_tam = [], [], []
    for i in range(n):
        oi = ltn.Constant(objects_tensor[i])
        for sh, pr in shape_to_pred.items():
            v = p[pr](oi).value
            sup_forma.append(v if shape_labels[i]==sh else 1.0-v)
        for co, pr in color_to_pred.items():
            v = p[pr](oi).value
            sup_cor.append(v if color_labels[i]==co else 1.0-v)
        v = p["isSmall"](oi).value
        sup_tam.append(v if size_labels[i]==0.0 else 1.0-v)

    pares = [(i,j) for i in range(n) for j in range(n) if i!=j]
    np.random.shuffle(pares); pares = pares[:120]

    sup_left, sup_right, sup_close = [], [], []
    sup_below, sup_above = [], []
    sup_samesize, sup_canstack = [], []

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
        same = (size_labels[i] == size_labels[j])
        v = p["sameSize"](oi,oj).value
        sup_samesize.append(v if same else 1.0-v)
        can = (shape_labels[j] not in ["Cone","Triangle"])
        v = p["canStack"](oi,oj).value
        sup_canstack.append(v if can else 1.0-v)

    trios = [(i,j,k) for i in range(n) for j in range(n) for k in range(n)
             if i!=j and i!=k and j!=k]
    np.random.shuffle(trios); trios = trios[:80]
    sup_between = []
    for (i,j,k) in trios:
        oi = ltn.Constant(objects_tensor[i])
        oj = ltn.Constant(objects_tensor[j])
        ok = ltn.Constant(objects_tensor[k])
        v  = p["inBetween"](oi,oj,ok).value
        sup_between.append(v if between_labels[(i,j,k)] else 1.0-v)

    sat_total = SatAgg(
        ax_unic_forma, ax_comp_forma, ax_unic_tam, ax_comp_tam,
        ax_irr, ax_assim, ax_inv_h, ax_trans_h,
        ax_inv_v, ax_trans_v, ax_q3, ax_same_sim,
        mean_stack(sup_forma), mean_stack(sup_cor), mean_stack(sup_tam),
        mean_stack(sup_left), mean_stack(sup_right), mean_stack(sup_close),
        mean_stack(sup_below), mean_stack(sup_above),
        mean_stack(sup_between), mean_stack(sup_samesize), mean_stack(sup_canstack),
    )

    # Satisfatibilidade individual das fórmulas das consultas
    sat_formulas = {
        # Tarefas 3.2
        "F_irr":       ax_irr.value.item(),
        "F_assim":     ax_assim.value.item(),
        "F_inv_h":     ax_inv_h.value.item(),
        "F_trans_h":   ax_trans_h.value.item(),
        "F_inv_v":     ax_inv_v.value.item(),
        "F_trans_v":   ax_trans_v.value.item(),
        # Tarefas 3.3
        "F_unic_forma":ax_unic_forma.value.item(),
        "F_comp_forma":ax_comp_forma.value.item(),
        "F_unic_tam":  ax_unic_tam.value.item(),
        "F_comp_tam":  ax_comp_tam.value.item(),
        "F_Q3_prox":   ax_q3.value.item(),
    }

    return sat_total, sat_formulas


# ═════════════════════════════════════════════
# BLOCO 4 — TREINO DE UMA EXECUÇÃO
# ═════════════════════════════════════════════

def executar_um_run(seed, run_id):
    """
    Executa pipeline completo para um dataset com seed específico.
    Retorna dicionário com todas as métricas e satisfatibilidades.
    """
    print(f"\n{'─'*55}")
    print(f"  RUN {run_id}/5  |  seed={seed}")
    print(f"{'─'*55}")

    # Dados
    torch.manual_seed(seed)
    np.random.seed(seed)
    objects_tensor, shape_labels, color_labels = generate_objects(N_OBJECTS, seed=seed)
    size_labels      = objects_tensor[:, 10].tolist()
    labels_espaciais = gerar_labels_espaciais(objects_tensor, sigma=0.15)

    # Predicados e variáveis LTN
    predicados = criar_predicados()
    x_var = ltn.Variable("x", objects_tensor)
    y_var = ltn.Variable("y", objects_tensor)
    z_var = ltn.Variable("z", objects_tensor)

    params = []
    for pred in predicados.values():
        params += list(pred.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    historico_sat = []
    sat_formulas_final = {}

    # Loop de treino
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        sat_total, sat_formulas = compute_kb(
            x_var, y_var, z_var, predicados, objects_tensor,
            shape_labels, color_labels, size_labels, labels_espaciais
        )
        loss = 1.0 - sat_total
        loss.backward()
        optimizer.step()
        historico_sat.append(sat_total.item())
        if epoch == EPOCHS - 1:
            sat_formulas_final = sat_formulas

    sat_final = historico_sat[-1]
    print(f"  SatAgg final: {sat_final:.4f}")

    # ── Avaliação das consultas Q1, Q2, Q3 ───
    sat_q1, sat_q2, sat_q3 = avaliar_consultas(
        predicados, objects_tensor,
        shape_labels, color_labels,
        size_labels, labels_espaciais
    )
    sat_formulas_final["Q1_filtComposta"]   = sat_q1
    sat_formulas_final["Q2_posAbsoluta"]    = sat_q2
    sat_formulas_final["Q3_proxTriangulos"] = sat_q3

    # ── Métricas de classificação ─────────────
    metricas = calcular_metricas(
        predicados, objects_tensor,
        shape_labels, color_labels,
        size_labels, labels_espaciais
    )

    return {
        "run":        run_id,
        "seed":       seed,
        "sat_final":  sat_final,
        "sat_formulas": sat_formulas_final,
        "metricas":   metricas,
        "historico":  historico_sat,
    }


# ═════════════════════════════════════════════
# BLOCO 5 — AVALIAÇÃO DE Q1, Q2, Q3
# ═════════════════════════════════════════════

def avaliar_consultas(predicados, objects_tensor,
                       shape_labels, color_labels,
                       size_labels, labels_espaciais):
    """
    Retorna o grau de satisfatibilidade de Q1, Q2, Q3.
    """
    p = predicados
    n = len(objects_tensor)
    (left_labels, right_labels, close_labels,
     below_labels, above_labels, between_labels) = labels_espaciais

    with torch.no_grad():
        # Q1: ∃x (IsSmall(x) ∧ ∃y(Cyl∧Below) ∧ ∃z(Sq∧Left))
        scores_q1 = []
        for i in range(n):
            oi       = ltn.Constant(objects_tensor[i])
            small_i  = p["isSmall"](oi).value.squeeze()
            sc_cb    = []
            sc_sl    = []
            for j in range(n):
                if i==j: continue
                oj = ltn.Constant(objects_tensor[j])
                sc_cb.append((p["isCylinder"](oj).value * p["below"](oi,oj).value).squeeze())
                sc_sl.append((p["isSquare"](oj).value   * p["leftOf"](oi,oj).value).squeeze())
            exist_cb = torch.max(torch.stack(sc_cb))
            exist_sl = torch.max(torch.stack(sc_sl))
            scores_q1.append((small_i * exist_cb * exist_sl).item())
        sat_q1 = max(scores_q1)   # ∃x → máximo

        # Q2: ∃x,y,z (Cone(x) ∧ Green(x) ∧ InBetween(x,y,z))
        scores_q2 = []
        for i in range(n):
            oi      = ltn.Constant(objects_tensor[i])
            cone_i  = p["isCone"](oi).value.squeeze()
            green_i = p["isGreen"](oi).value.squeeze()
            sc_bt   = []
            for j in range(n):
                if j==i: continue
                for k in range(n):
                    if k==i or k==j: continue
                    oj = ltn.Constant(objects_tensor[j])
                    ok = ltn.Constant(objects_tensor[k])
                    bt = p["inBetween"](oi,oj,ok).value.squeeze()
                    sc_bt.append(bt)
            if sc_bt:
                exist_bt = torch.max(torch.stack(sc_bt))
                scores_q2.append((cone_i * green_i * exist_bt).item())
        sat_q2 = max(scores_q2) if scores_q2 else 0.0

        # Q3: ∀x,y (Tri∧Tri∧Close → SameSize) — média da implicação
        vals_q3 = []
        for i in range(n):
            for j in range(n):
                if i==j: continue
                oi = ltn.Constant(objects_tensor[i])
                oj = ltn.Constant(objects_tensor[j])
                tri_i  = p["isTriangle"](oi).value.squeeze()
                tri_j  = p["isTriangle"](oj).value.squeeze()
                close  = p["closeTo"](oi,oj).value.squeeze()
                same   = p["sameSize"](oi,oj).value.squeeze()
                prem   = tri_i * tri_j * close
                impl   = torch.clamp(1.0 - prem + same, max=1.0)
                vals_q3.append(impl.item())
        sat_q3 = float(np.mean(vals_q3)) if vals_q3 else 1.0

    return sat_q1, sat_q2, sat_q3


# ═════════════════════════════════════════════
# BLOCO 6 — MÉTRICAS DE CLASSIFICAÇÃO
# ═════════════════════════════════════════════

def calcular_metricas(predicados, objects_tensor,
                       shape_labels, color_labels,
                       size_labels, labels_espaciais, threshold=THRESHOLD):
    p = predicados
    n = len(objects_tensor)
    (left_labels, right_labels, close_labels,
     below_labels, above_labels, between_labels) = labels_espaciais

    def _calc(y_true, y_pred):
        y_true = np.array(y_true); y_pred = np.array(y_pred)
        TP = ((y_pred==1)&(y_true==1)).sum()
        TN = ((y_pred==0)&(y_true==0)).sum()
        FP = ((y_pred==1)&(y_true==0)).sum()
        FN = ((y_pred==0)&(y_true==1)).sum()
        ac = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0.0
        pr = TP/(TP+FP)             if (TP+FP)>0        else 0.0
        rc = TP/(TP+FN)             if (TP+FN)>0        else 0.0
        f1 = 2*pr*rc/(pr+rc)        if (pr+rc)>0        else 0.0
        return {"Acurácia":round(ac,4),"Precisão":round(pr,4),
                "Recall":round(rc,4),"F1":round(f1,4)}

    def avaliar_unario(pred_name, gt_list):
        y_true, y_pred = [], []
        with torch.no_grad():
            for i in range(n):
                oi = ltn.Constant(objects_tensor[i])
                s  = p[pred_name](oi).value.item()
                y_true.append(gt_list[i])
                y_pred.append(float(s > threshold))
        return _calc(y_true, y_pred)

    def avaliar_binario(pred_name, label_dict, pares):
        y_true, y_pred = [], []
        with torch.no_grad():
            for (i,j) in pares:
                oi = ltn.Constant(objects_tensor[i])
                oj = ltn.Constant(objects_tensor[j])
                s  = p[pred_name](oi,oj).value.item()
                y_true.append(float(label_dict[(i,j)]))
                y_pred.append(float(s > threshold))
        return _calc(y_true, y_pred)

    pares = [(i,j) for i in range(n) for j in range(n) if i!=j]

    shape_to_pred = {"Circle":"isCircle","Square":"isSquare",
                     "Cylinder":"isCylinder","Cone":"isCone","Triangle":"isTriangle"}
    color_to_pred = {"Red":"isRed","Green":"isGreen","Blue":"isBlue"}

    resultados = {}

    for sh, pr in shape_to_pred.items():
        gt = [1.0 if shape_labels[i]==sh else 0.0 for i in range(n)]
        resultados[pr] = avaliar_unario(pr, gt)

    for co, pr in color_to_pred.items():
        gt = [1.0 if color_labels[i]==co else 0.0 for i in range(n)]
        resultados[pr] = avaliar_unario(pr, gt)

    resultados["isSmall"] = avaliar_unario("isSmall",
        [1.0 if size_labels[i]==0.0 else 0.0 for i in range(n)])
    resultados["isBig"]   = avaliar_unario("isBig",
        [1.0 if size_labels[i]==1.0 else 0.0 for i in range(n)])
    resultados["leftOf"]  = avaliar_binario("leftOf",  left_labels,  pares)
    resultados["rightOf"] = avaliar_binario("rightOf", right_labels, pares)
    resultados["closeTo"] = avaliar_binario("closeTo", close_labels, pares)
    resultados["below"]   = avaliar_binario("below",   below_labels, pares)
    resultados["above"]   = avaliar_binario("above",   above_labels, pares)

    same_sz = {(i,j): (size_labels[i]==size_labels[j] and i!=j)
               for i in range(n) for j in range(n)}
    resultados["sameSize"] = avaliar_binario("sameSize", same_sz, pares)

    can_st = {(i,j): (i!=j and shape_labels[j] not in ["Cone","Triangle"])
              for i in range(n) for j in range(n)}
    resultados["canStack"] = avaliar_binario("canStack", can_st, pares)

    return resultados


# ═════════════════════════════════════════════
# BLOCO 7 — GERAÇÃO DO RELATÓRIO MARKDOWN
# ═════════════════════════════════════════════

def gerar_markdown(todos_runs):
    """
    Gera o arquivo resultados_5runs.md com todas as tabelas
    formatadas para o relatório do GitHub.
    """
    linhas = []
    linhas.append("# Resultados das 5 Execuções — Tarefas 3.2 e 3.3\n")
    linhas.append("> Cada execução usa um dataset aleatório distinto (seed diferente).  \n")
    linhas.append(f"> Objetos: {N_OBJECTS} | Épocas: {EPOCHS} | LR: {LR} | Threshold: {THRESHOLD}\n\n")

    # ── Tabela 1: SatAgg geral por run ────────
    linhas.append("## 1. Satisfatibilidade Geral (SatAgg)\n")
    linhas.append("| Run | Seed | SatAgg Final |\n")
    linhas.append("|-----|------|--------------|\n")
    sats = []
    for r in todos_runs:
        linhas.append(f"| {r['run']} | {r['seed']} | {r['sat_final']:.4f} |\n")
        sats.append(r['sat_final'])
    linhas.append(f"| **Média** | — | **{np.mean(sats):.4f}** |\n")
    linhas.append(f"| **Desvio** | — | **{np.std(sats):.4f}** |\n\n")

    # ── Tabela 2: Satisfatibilidade por fórmula ─
    linhas.append("## 2. Satisfatibilidade por Fórmula\n")
    linhas.append("> Fórmulas das Tarefas 3.2 (espacial) e 3.3 (raciocínio composto)\n\n")

    todas_formulas = list(todos_runs[0]["sat_formulas"].keys())
    header = "| Fórmula | " + " | ".join([f"Run {r['run']}" for r in todos_runs])
    header += " | Média |\n"
    sep    = "|---------|" + "--------|"*len(todos_runs) + "-------|\n"
    linhas.append(header)
    linhas.append(sep)

    for formula in todas_formulas:
        vals = [r["sat_formulas"][formula] for r in todos_runs]
        row  = f"| `{formula}` | "
        row += " | ".join([f"{v:.4f}" for v in vals])
        row += f" | **{np.mean(vals):.4f}** |\n"
        linhas.append(row)
    linhas.append("\n")

    # ── Tabela 3: Métricas por predicado (média das 5 runs) ─
    linhas.append("## 3. Métricas por Predicado (Média das 5 Execuções)\n\n")

    todos_preds = list(todos_runs[0]["metricas"].keys())
    linhas.append("| Predicado | Acurácia | Precisão | Recall | F1 |\n")
    linhas.append("|-----------|----------|----------|--------|----|\n")

    for pred in todos_preds:
        ac_vals = [r["metricas"][pred]["Acurácia"] for r in todos_runs]
        pr_vals = [r["metricas"][pred]["Precisão"] for r in todos_runs]
        rc_vals = [r["metricas"][pred]["Recall"]   for r in todos_runs]
        f1_vals = [r["metricas"][pred]["F1"]       for r in todos_runs]
        linhas.append(
            f"| `{pred}` | {np.mean(ac_vals):.4f} ± {np.std(ac_vals):.3f} "
            f"| {np.mean(pr_vals):.4f} ± {np.std(pr_vals):.3f} "
            f"| {np.mean(rc_vals):.4f} ± {np.std(rc_vals):.3f} "
            f"| {np.mean(f1_vals):.4f} ± {np.std(f1_vals):.3f} |\n"
        )
    linhas.append("\n")

    # ── Tabela 4: Métricas por run (F1 resumido) ─
    linhas.append("## 4. F1 Score por Predicado e por Execução\n\n")

    header = "| Predicado | " + " | ".join([f"Run {r['run']}" for r in todos_runs]) + " |\n"
    sep    = "|-----------|" + "-------|"*len(todos_runs) + "\n"
    linhas.append(header)
    linhas.append(sep)

    for pred in todos_preds:
        row = f"| `{pred}` | "
        row += " | ".join([f"{r['metricas'][pred]['F1']:.4f}" for r in todos_runs])
        row += " |\n"
        linhas.append(row)
    linhas.append("\n")

    # ── Seção de análise ──────────────────────
    linhas.append("## 5. Observações\n\n")
    linhas.append("- **Satisfatibilidade**: valores próximos de 1.0 indicam que o LTN ")
    linhas.append("aprendeu a satisfazer os axiomas lógicos definidos na KB.\n")
    linhas.append("- **Variância entre runs**: datasets distintos produzem distribuições ")
    linhas.append("diferentes de objetos, impactando a dificuldade de cada axioma.\n")
    linhas.append("- **Predicados espaciais** (`leftOf`, `below`) tendem a ter F1 alto ")
    linhas.append("por serem determinísticos a partir das coordenadas.\n")
    linhas.append("- **Q3** pode ser satisfeita vacuamente se não houver par de triângulos ")
    linhas.append("próximos no dataset — fenômeno esperado em lógica de implicação.\n")

    md = "".join(linhas)
    with open("resultados_5runs.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("→ resultados_5runs.md salvo.")
    return md


# ═════════════════════════════════════════════
# BLOCO 8 — GRÁFICOS COMPARATIVOS
# ═════════════════════════════════════════════

def plot_historicos(todos_runs):
    """Curvas de SatAgg das 5 execuções num único gráfico."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cores = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6"]
    for r in todos_runs:
        ax.plot(r["historico"], color=cores[r["run"]-1],
                lw=1.5, label=f"Run {r['run']} (seed={r['seed']})")
    ax.set_xlabel("Época"); ax.set_ylabel("SatAgg")
    ax.set_title("Satisfatibilidade durante Treinamento — 5 Execuções")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.4)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("historicos_5runs.png", dpi=150)
    plt.show()
    print("→ historicos_5runs.png salvo.")


def plot_metricas_comparativas(todos_runs):
    """
    Gráfico de barras agrupadas: F1 médio ± desvio para cada predicado.
    """
    todos_preds = list(todos_runs[0]["metricas"].keys())
    f1_medias   = []
    f1_desvios  = []
    for pred in todos_preds:
        vals = [r["metricas"][pred]["F1"] for r in todos_runs]
        f1_medias.append(np.mean(vals))
        f1_desvios.append(np.std(vals))

    x      = np.arange(len(todos_preds))
    cores  = ["#2ecc71" if v >= 0.7 else ("#f39c12" if v >= 0.4 else "#e74c3c")
              for v in f1_medias]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(x, f1_medias, yerr=f1_desvios, capsize=4,
                  color=cores, edgecolor='black', linewidth=0.5, error_kw={"lw":1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(todos_preds, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel("F1 Score (média ± desvio)")
    ax.set_title("F1 por Predicado — Média de 5 Execuções")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.7, color='green',  ls='--', alpha=0.5, label='F1 ≥ 0.7')
    ax.axhline(0.4, color='orange', ls='--', alpha=0.5, label='F1 ≥ 0.4')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    for bar, v, dv in zip(bars, f1_medias, f1_desvios):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + dv + 0.03, f"{v:.2f}",
                ha='center', fontsize=7)
    plt.tight_layout()
    plt.savefig("grafico_5runs.png", dpi=150)
    plt.show()
    print("→ grafico_5runs.png salvo.")


def plot_sat_formulas(todos_runs):
    """Heatmap de satisfatibilidade por fórmula × run."""
    formulas = list(todos_runs[0]["sat_formulas"].keys())
    matriz   = np.array([[r["sat_formulas"][f] for f in formulas]
                          for r in todos_runs])   # shape: (5, n_formulas)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matriz, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(formulas)))
    ax.set_xticklabels(formulas, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(N_RUNS))
    ax.set_yticklabels([f"Run {r['run']} (s={r['seed']})" for r in todos_runs])
    ax.set_title("Satisfatibilidade por Fórmula e Execução (verde=1.0, vermelho=0.0)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(N_RUNS):
        for j in range(len(formulas)):
            ax.text(j, i, f"{matriz[i,j]:.2f}",
                    ha='center', va='center', fontsize=7,
                    color='black' if 0.3 < matriz[i,j] < 0.8 else 'white')
    plt.tight_layout()
    plt.savefig("heatmap_formulas.png", dpi=150)
    plt.show()
    print("→ heatmap_formulas.png salvo.")


# ═════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL
# ═════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  ENTREGAS — 5 EXECUÇÕES COM DATASETS DISTINTOS")
    print("  Tarefas 3.2 (Espacial) + 3.3 (Raciocínio Composto)")
    print("="*55)

    todos_runs = []

    for run_id, seed in enumerate(SEEDS, start=1):
        resultado = executar_um_run(seed, run_id)
        todos_runs.append(resultado)

        # Resumo rápido do run
        print(f"\n  Resumo Run {run_id}:")
        print(f"    SatAgg final : {resultado['sat_final']:.4f}")
        print(f"    Q1 (filtComp): {resultado['sat_formulas']['Q1_filtComposta']:.4f}")
        print(f"    Q2 (posAbs)  : {resultado['sat_formulas']['Q2_posAbsoluta']:.4f}")
        print(f"    Q3 (proxTri) : {resultado['sat_formulas']['Q3_proxTriangulos']:.4f}")
        f1_left = resultado['metricas']['leftOf']['F1']
        f1_bel  = resultado['metricas']['below']['F1']
        print(f"    F1 leftOf    : {f1_left:.4f}")
        print(f"    F1 below     : {f1_bel:.4f}")

    # Relatório Markdown
    print("\n" + "="*55)
    print("  Gerando relatório Markdown...")
    gerar_markdown(todos_runs)

    # Gráficos
    print("  Gerando gráficos...")
    plot_historicos(todos_runs)
    plot_metricas_comparativas(todos_runs)
    plot_sat_formulas(todos_runs)

    print("\n" + "="*55)
    print("  CONCLUÍDO! Arquivos gerados:")
    print("    → resultados_5runs.md    (tabelas para o GitHub)")
    print("    → historicos_5runs.png   (curvas de treinamento)")
    print("    → grafico_5runs.png      (F1 médio por predicado)")
    print("    → heatmap_formulas.png   (sat. por fórmula × run)")
    print("="*55)
