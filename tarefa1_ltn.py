"""
=============================================================================
TRABALHO FINAL - FIA (ICC260)
Tarefa 1: Taxonomia e Formas com Logic Tensor Networks (LTNtorch)
=============================================================================

INSTALAÇÃO (rode antes):
    pip install ltnorch torch matplotlib numpy

ESTRUTURA DO VETOR DE OBJETO (tamanho 11):
    [0]   x        → posição horizontal (0.0 a 1.0)
    [1]   y        → posição vertical   (0.0 a 1.0)
    [2]   Red      → cor vermelha  (one-hot)
    [3]   Green    → cor verde     (one-hot)
    [4]   Blue     → cor azul      (one-hot)
    [5]   Circle   → forma círculo   (one-hot)
    [6]   Square   → forma quadrado  (one-hot)
    [7]   Cylinder → forma cilindro  (one-hot)
    [8]   Cone     → forma cone      (one-hot)
    [9]   Triangle → forma triângulo (one-hot)
    [10]  Size     → tamanho (0.0=pequeno, 1.0=grande)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ltn  # pip install ltnorch

# ─────────────────────────────────────────────
# SEMENTE ALEATÓRIA para reprodutibilidade
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1. GERAÇÃO DE DADOS
#    25 objetos aleatórios com posição, cor e forma
# ─────────────────────────────────────────────

N_OBJECTS = 25  # número de objetos na cena

SHAPES  = ["Circle", "Square", "Cylinder", "Cone", "Triangle"]  # índices 5-9
COLORS  = ["Red", "Green", "Blue"]                               # índices 2-4
SHAPE_IDX_START = 5   # onde começa o bloco de forma no vetor
COLOR_IDX_START = 2   # onde começa o bloco de cor no vetor

def generate_objects(n=25, seed=None):
    """
    Gera n objetos aleatórios.
    Retorna um tensor de shape (n, 11) e listas de labels para plot.
    """
    if seed is not None:
        np.random.seed(seed)

    data = []
    shape_labels = []
    color_labels = []

    for _ in range(n):
        x     = np.random.uniform(0.0, 1.0)   # posição x
        y     = np.random.uniform(0.0, 1.0)   # posição y
        size  = float(np.random.randint(0, 2)) # 0=pequeno, 1=grande

        # Cor one-hot: escolhe 1 das 3 cores
        color_idx = np.random.randint(0, 3)
        color_vec = [0.0, 0.0, 0.0]
        color_vec[color_idx] = 1.0

        # Forma one-hot: escolhe 1 das 5 formas
        shape_idx = np.random.randint(0, 5)
        shape_vec = [0.0, 0.0, 0.0, 0.0, 0.0]
        shape_vec[shape_idx] = 1.0

        obj = [x, y] + color_vec + shape_vec + [size]
        data.append(obj)
        shape_labels.append(SHAPES[shape_idx])
        color_labels.append(COLORS[color_idx])

    tensor = torch.tensor(data, dtype=torch.float32)
    return tensor, shape_labels, color_labels


def plot_scene(objects_tensor, shape_labels, color_labels, title="Cena CLEVR Simplificada"):
    """
    Plota os objetos no espaço 2D.
    Forma → marcador, Cor → cor do marcador, Tamanho → tamanho do marcador.
    """
    marker_map = {
        "Circle":   "o",   # círculo
        "Square":   "s",   # quadrado
        "Cylinder": "D",   # losango (aproximação)
        "Cone":     "^",   # triângulo apontando para cima
        "Triangle": "v",   # triângulo apontando para baixo
    }
    color_map = {
        "Red":   "red",
        "Green": "green",
        "Blue":  "blue",
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    data = objects_tensor.numpy()

    for i in range(len(data)):
        x, y   = data[i, 0], data[i, 1]
        size   = data[i, 10]
        marker = marker_map[shape_labels[i]]
        color  = color_map[color_labels[i]]
        ms     = 200 if size == 1.0 else 80  # grande vs pequeno

        ax.scatter(x, y, c=color, marker=marker, s=ms,
                   edgecolors='black', linewidths=0.8, zorder=3)
        ax.annotate(str(i), (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7)

    # Legenda de formas
    legend_shapes = [
        mpatches.Patch(color='gray', label=f"{s} ({marker_map[s]})")
        for s in SHAPES
    ]
    legend_colors = [
        mpatches.Patch(color=color_map[c], label=c)
        for c in COLORS
    ]
    ax.legend(handles=legend_shapes + legend_colors,
              loc='upper left', fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig("cena_clevr.png", dpi=150)
    plt.show()
    print("→ Cena salva em: cena_clevr.png")


# ─────────────────────────────────────────────
# 2. PREDICADOS LTN
#    Cada predicado é uma MLP que recebe features e retorna [0,1]
# ─────────────────────────────────────────────

class MLP_Predicado(nn.Module):
    """
    Rede neural pequena para um predicado unário.
    Entrada: vetor de features do objeto (tamanho input_size)
    Saída:   escalar em [0,1] representando grau de verdade
    """
    def __init__(self, input_size=11, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # garante saída entre 0 e 1
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 3. CONFIGURAÇÃO LTN
#    Constantes, variáveis e predicados
# ─────────────────────────────────────────────

def setup_ltn(objects_tensor):
    """
    Configura os componentes LTN para a Tarefa 1.
    Retorna: dicionário com predicados e variável x
    """

    # --- Constantes LTN ---
    # Cada objeto vira uma constante no universo do LTN
    # ltn.Constant encapsula um tensor e o rastreia para gradientes
    constants = [ltn.Constant(objects_tensor[i]) for i in range(len(objects_tensor))]

    # --- Variável LTN ---
    # Uma variável percorre todos os objetos (usada nos quantificadores ∀ e ∃)
    # ltn.Variable é como um "para todo x" — processa todos os objetos de uma vez
    x = ltn.Variable("x", objects_tensor)

    # --- Predicados de Forma ---
    # Cada um é uma MLP independente com seus próprios pesos
    isCircle   = ltn.Predicate(MLP_Predicado(input_size=11))
    isSquare   = ltn.Predicate(MLP_Predicado(input_size=11))
    isCylinder = ltn.Predicate(MLP_Predicado(input_size=11))
    isCone     = ltn.Predicate(MLP_Predicado(input_size=11))
    isTriangle = ltn.Predicate(MLP_Predicado(input_size=11))

    # --- Predicados de Tamanho ---
    isSmall = ltn.Predicate(MLP_Predicado(input_size=11))
    isBig   = ltn.Predicate(MLP_Predicado(input_size=11))

    predicados = {
        "isCircle":   isCircle,
        "isSquare":   isSquare,
        "isCylinder": isCylinder,
        "isCone":     isCone,
        "isTriangle": isTriangle,
        "isSmall":    isSmall,
        "isBig":      isBig,
    }

    return x, predicados, constants


# ─────────────────────────────────────────────
# 4. AXIOMAS (Base de Conhecimento - KB)
#    As regras lógicas que guiam o treinamento
# ─────────────────────────────────────────────

def compute_satisfatibilidade(x, predicados, objects_tensor, shape_labels, size_labels):
    """
    Calcula a satisfatibilidade de todos os axiomas da KB.

    AXIOMAS IMPLEMENTADOS:

    1. Unicidade de Forma (Mutual Exclusion):
       ∀x, ¬(isCircle(x) ∧ isSquare(x) ∧ isCylinder(x) ∧ isCone(x) ∧ isTriangle(x))
       → Um objeto NÃO pode ser mais de uma forma ao mesmo tempo.

    2. Completude (Coverage):
       ∀x, (isCircle(x) ∨ isSquare(x) ∨ isCylinder(x) ∨ isCone(x) ∨ isTriangle(x))
       → Todo objeto DEVE ser pelo menos uma das formas.

    3. Supervisão por Labels (dados de treino):
       Se sabemos que o objeto i é um Círculo, forçamos isCircle(obj_i) ≈ 1.0
       → Isso ancora o aprendizado nos dados reais.

    4. Unicidade de Tamanho:
       ∀x, ¬(isSmall(x) ∧ isBig(x))

    5. Completude de Tamanho:
       ∀x, (isSmall(x) ∨ isBig(x))
    """

    # Operadores lógicos fuzzy do LTN
    # Not(a)    = 1 - a
    # And(a, b) = produto: a * b  (suave e diferenciável)
    # Or(a, b)  = probabilistic sum: a + b - a*b
    # Forall    = média geométrica (pMeanError) sobre todos os objetos
    Not    = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And    = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Or     = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    p = predicados  # atalho

    # ── Axioma 1: Unicidade de Forma ──────────────────────────────────
    # ∀x ¬(C(x) ∧ S(x)) ∧ ¬(C(x) ∧ Cy(x)) ∧ ... (todos os pares)
    # Simplificação: o AND de todos negado equivale a dizer que
    # no máximo 1 forma é verdadeira.
    # Implementamos verificando cada par:
    ax_unicidade_forma = Forall(x,
        And(
            And(
                Not(And(p["isCircle"](x),   p["isSquare"](x))),
                Not(And(p["isCircle"](x),   p["isCylinder"](x)))
            ),
            And(
                Not(And(p["isCircle"](x),   p["isCone"](x))),
                Not(And(p["isSquare"](x),   p["isCylinder"](x)))
            )
        )
    )

    # ── Axioma 2: Completude de Forma ─────────────────────────────────
    # ∀x (Circle(x) ∨ Square(x) ∨ Cylinder(x) ∨ Cone(x) ∨ Triangle(x))
    ax_completude_forma = Forall(x,
        Or(
            Or(
                Or(p["isCircle"](x), p["isSquare"](x)),
                Or(p["isCylinder"](x), p["isCone"](x))
            ),
            p["isTriangle"](x)
        )
    )

    # ── Axioma 3: Unicidade de Tamanho ────────────────────────────────
    # ∀x ¬(isSmall(x) ∧ isBig(x))
    ax_unicidade_tam = Forall(x,
        Not(And(p["isSmall"](x), p["isBig"](x)))
    )

    # ── Axioma 4: Completude de Tamanho ───────────────────────────────
    # ∀x (isSmall(x) ∨ isBig(x))
    ax_completude_tam = Forall(x,
        Or(p["isSmall"](x), p["isBig"](x))
    )

    # ── Axioma 5: Supervisão por Labels (dados de treino) ─────────────
    # Para cada objeto, forçamos o predicado correto a ser verdadeiro.
    # Ex: Se obj_i é um Círculo → isCircle(obj_i) deve ser ≈ 1.0
    supervisao_forma = []
    supervisao_tam   = []

    shape_to_pred = {
        "Circle":   "isCircle",
        "Square":   "isSquare",
        "Cylinder": "isCylinder",
        "Cone":     "isCone",
        "Triangle": "isTriangle",
    }

    for i, (shape, size) in enumerate(zip(shape_labels, size_labels)):
        obj_i = ltn.Constant(objects_tensor[i])

        # Supervisão de forma
        pred_name = shape_to_pred[shape]
        supervisao_forma.append(p[pred_name](obj_i))

        # Supervisão de tamanho
        if size == 0.0:
            supervisao_tam.append(p["isSmall"](obj_i))
        else:
            supervisao_tam.append(p["isBig"](obj_i))

    # Agrega as supervisões individuais
    # (média dos graus de verdade para cada label correto)
    sat_sup_forma = torch.mean(torch.stack([s.value for s in supervisao_forma]))
    sat_sup_tam   = torch.mean(torch.stack([s.value for s in supervisao_tam]))

    # ── SatAgg: satisfatibilidade geral da KB ─────────────────────────
    # Combina todos os axiomas com média geométrica ponderada
    sat_total = SatAgg(
        ax_unicidade_forma,   # regra lógica
        ax_completude_forma,  # regra lógica
        ax_unicidade_tam,     # regra lógica
        ax_completude_tam,    # regra lógica
        sat_sup_forma,        # supervisão de forma
        sat_sup_tam           # supervisão de tamanho
    )

    return sat_total, {
        "unicidade_forma":  ax_unicidade_forma.value.item(),
        "completude_forma": ax_completude_forma.value.item(),
        "unicidade_tam":    ax_unicidade_tam.value.item(),
        "completude_tam":   ax_completude_tam.value.item(),
        "supervisao_forma": sat_sup_forma.item(),
        "supervisao_tam":   sat_sup_tam.item(),
    }


# ─────────────────────────────────────────────
# 5. TREINAMENTO
# ─────────────────────────────────────────────

def treinar(objects_tensor, shape_labels, size_labels,
            epochs=500, lr=0.001, verbose=True):
    """
    Treina todos os predicados maximizando a satisfatibilidade da KB.

    Parâmetros:
        objects_tensor : tensor (N, 11) dos objetos
        shape_labels   : lista de strings com a forma de cada objeto
        size_labels    : lista de floats (0.0=pequeno, 1.0=grande)
        epochs         : número de épocas de treino
        lr             : taxa de aprendizado
        verbose        : imprime progresso a cada 50 épocas
    """

    x, predicados, constants = setup_ltn(objects_tensor)

    # Coleta todos os parâmetros de todos os predicados
    todos_params = []
    for pred in predicados.values():
        todos_params += list(pred.parameters())

    optimizer = torch.optim.Adam(todos_params, lr=lr)

    historico_sat = []

    print(f"\n{'='*55}")
    print(f"  TREINAMENTO - TAREFA 1")
    print(f"  {N_OBJECTS} objetos | {epochs} épocas | lr={lr}")
    print(f"{'='*55}")

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Calcula satisfatibilidade de todos os axiomas
        sat_total, axiomas_sat = compute_satisfatibilidade(
            x, predicados, objects_tensor, shape_labels, size_labels
        )

        # Loss = 1 - satisfatibilidade  (queremos MAXIMIZAR sat → MINIMIZAR loss)
        loss = 1.0 - sat_total

        loss.backward()
        optimizer.step()

        historico_sat.append(sat_total.item())

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"  Época {epoch:4d} | SatAgg={sat_total.item():.4f} | Loss={loss.item():.4f}")

    print(f"\n  Satisfatibilidade final por axioma:")
    for nome, val in axiomas_sat.items():
        print(f"    {nome:<20}: {val:.4f}")

    return predicados, historico_sat


# ─────────────────────────────────────────────
# 6. AVALIAÇÃO: MÉTRICAS DE CLASSIFICAÇÃO
# ─────────────────────────────────────────────

def calcular_metricas(predicados, objects_tensor, shape_labels, size_labels, threshold=0.5):
    """
    Calcula Acurácia, Precisão, Recall e F1 para cada predicado de forma/tamanho.

    O threshold padrão é 0.5:
        se predicado(x) > 0.5 → modelo diz "SIM, é essa forma"
        se predicado(x) ≤ 0.5 → modelo diz "NÃO"
    """

    print(f"\n{'='*55}")
    print(f"  MÉTRICAS DE CLASSIFICAÇÃO (threshold={threshold})")
    print(f"{'='*55}")

    shape_to_pred = {
        "Circle":   "isCircle",
        "Square":   "isSquare",
        "Cylinder": "isCylinder",
        "Cone":     "isCone",
        "Triangle": "isTriangle",
    }

    resultados = {}

    with torch.no_grad():
        for shape, pred_name in shape_to_pred.items():
            # Ground truth: 1 se o objeto tem essa forma, 0 caso contrário
            y_true = torch.tensor(
                [1.0 if s == shape else 0.0 for s in shape_labels]
            )

            # Predição do modelo (grau de verdade)
            x_var = ltn.Variable("x", objects_tensor)
            scores = predicados[pred_name](x_var).value.squeeze()

            # Binariza pela threshold
            y_pred = (scores > threshold).float()

            TP = ((y_pred == 1) & (y_true == 1)).sum().item()
            TN = ((y_pred == 0) & (y_true == 0)).sum().item()
            FP = ((y_pred == 1) & (y_true == 0)).sum().item()
            FN = ((y_pred == 0) & (y_true == 1)).sum().item()

            acuracia  = (TP + TN) / (TP + TN + FP + FN) if (TP+TN+FP+FN) > 0 else 0
            precisao  = TP / (TP + FP)                   if (TP + FP) > 0       else 0
            recall    = TP / (TP + FN)                   if (TP + FN) > 0       else 0
            f1        = (2 * precisao * recall / (precisao + recall)
                         if (precisao + recall) > 0 else 0)

            resultados[pred_name] = {
                "Acurácia":  acuracia,
                "Precisão":  precisao,
                "Recall":    recall,
                "F1":        f1,
                "TP": TP, "TN": TN, "FP": FP, "FN": FN
            }

            print(f"\n  {pred_name}:")
            print(f"    TP={TP:2.0f} TN={TN:2.0f} FP={FP:2.0f} FN={FN:2.0f}")
            print(f"    Acurácia={acuracia:.4f} | Precisão={precisao:.4f} "
                  f"| Recall={recall:.4f} | F1={f1:.4f}")

        # Predicados de tamanho
        for size_label, pred_name, val in [("Pequeno", "isSmall", 0.0),
                                            ("Grande",  "isBig",   1.0)]:
            y_true = torch.tensor(
                [1.0 if s == val else 0.0 for s in size_labels]
            )
            x_var  = ltn.Variable("x", objects_tensor)
            scores = predicados[pred_name](x_var).value.squeeze()
            y_pred = (scores > threshold).float()

            TP = ((y_pred == 1) & (y_true == 1)).sum().item()
            TN = ((y_pred == 0) & (y_true == 0)).sum().item()
            FP = ((y_pred == 1) & (y_true == 0)).sum().item()
            FN = ((y_pred == 0) & (y_true == 1)).sum().item()

            acuracia = (TP + TN) / (TP + TN + FP + FN) if (TP+TN+FP+FN) > 0 else 0
            precisao = TP / (TP + FP)                   if (TP + FP) > 0       else 0
            recall   = TP / (TP + FN)                   if (TP + FN) > 0       else 0
            f1       = (2 * precisao * recall / (precisao + recall)
                        if (precisao + recall) > 0 else 0)

            resultados[pred_name] = {
                "Acurácia": acuracia, "Precisão": precisao,
                "Recall": recall, "F1": f1
            }

            print(f"\n  {pred_name} ({size_label}):")
            print(f"    TP={TP:2.0f} TN={TN:2.0f} FP={FP:2.0f} FN={FN:2.0f}")
            print(f"    Acurácia={acuracia:.4f} | Precisão={precisao:.4f} "
                  f"| Recall={recall:.4f} | F1={f1:.4f}")

    return resultados


# ─────────────────────────────────────────────
# 7. PLOT DO HISTÓRICO DE TREINAMENTO
# ─────────────────────────────────────────────

def plot_treinamento(historico_sat, titulo="Satisfatibilidade durante Treinamento"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(historico_sat, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Época")
    ax.set_ylabel("SatAgg")
    ax.set_title(titulo)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Sat. máxima')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("historico_treinamento.png", dpi=150)
    plt.show()
    print("→ Histórico salvo em: historico_treinamento.png")


# ─────────────────────────────────────────────
# 8. EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  TAREFA 1: TAXONOMIA E FORMAS")
    print("  Logic Tensor Networks com LTNtorch")
    print("="*55)

    # ── 8.1 Gerar dados ──────────────────────────────────────
    print("\n[1/4] Gerando 25 objetos aleatórios...")
    objects_tensor, shape_labels, color_labels = generate_objects(N_OBJECTS, seed=SEED)

    # Extrair labels de tamanho (índice 10 do vetor)
    size_labels = objects_tensor[:, 10].tolist()

    print(f"  Distribuição de formas:")
    for shape in SHAPES:
        count = shape_labels.count(shape)
        bar = "█" * count
        print(f"    {shape:<10}: {bar} ({count})")

    print(f"\n  Distribuição de tamanho:")
    n_small = sum(1 for s in size_labels if s == 0.0)
    n_big   = N_OBJECTS - n_small
    print(f"    Pequeno: {'█'*n_small} ({n_small})")
    print(f"    Grande:  {'█'*n_big} ({n_big})")

    # ── 8.2 Plotar cena ──────────────────────────────────────
    print("\n[2/4] Plotando cena...")
    plot_scene(objects_tensor, shape_labels, color_labels)

    # ── 8.3 Treinar ──────────────────────────────────────────
    print("\n[3/4] Treinando predicados LTN...")
    predicados, historico_sat = treinar(
        objects_tensor, shape_labels, size_labels,
        epochs=500, lr=0.001
    )

    # ── 8.4 Avaliar métricas ─────────────────────────────────
    print("\n[4/4] Calculando métricas...")
    metricas = calcular_metricas(
        predicados, objects_tensor, shape_labels, size_labels
    )

    # ── 8.5 Plot do histórico ────────────────────────────────
    plot_treinamento(historico_sat)

    print("\n" + "="*55)
    print("  TAREFA 1 CONCLUÍDA!")
    print("  Arquivos gerados:")
    print("    → cena_clevr.png")
    print("    → historico_treinamento.png")
    print("="*55)
