# Plano de Implementação (v2) — Raciocínio Espacial Neuro-Simbólico com LTN (lib `ltn`, TensorFlow)

> Atualiza o `plano_implementacao_LTN.md` anterior com base nas respostas do professor e nos tutoriais oficiais 1–4 da lib `ltn` (TensorFlow, **não** `ltntorch`). MVP para rodar em Google Colab.

---

## 0. Decisões de design (confirmadas pelo professor)

1. **O "dataset" é uma única cena** com 25 objetos = 5 formas × 5 instâncias cada (não são imagens; são vetores de features de 11 dimensões, como na Seção 2 do PDF).
2. **Treina-se uma única vez** nessa cena. O papel dos axiomas de irreflexividade/assimetria/inverso/transitividade não é "decorar" a cena, mas ensinar a regra de forma que ela generalize.
3. **Avaliação = generalização**: depois de treinado, o modelo (sem novos gradientes) é testado em **5 cenas novas geradas aleatoriamente** (5 seeds diferentes), reportando por cena: satisfatibilidade (SatAgg e por axioma), acurácia, precisão, recall e F1 (fórmulas 6–9 do PDF/imagem de métricas).
4. Coordenadas e rótulos (forma/cor/tamanho) são conhecidos por construção (dados sintéticos) — usados para métricas e para uma **âncora de supervisão fraca** dos predicados relacionais (`leftOf`/`below`), já que os axiomas lógicos puros são subdeterminados (uma rede que sempre devolve 0, por exemplo, satisfaz trivialmente irreflexividade e assimetria).
5. **Simplificações do MVP**:
   - Um único classificador genérico `IsA(objeto, rótulo_onehot)` cobre todas as formas e os dois tamanhos (em vez de 7 redes separadas) — mesmo padrão do predicado `C` no tutorial `3-knowledgebase_and_learning.ipynb`.
   - Apenas `LeftOf` e `Below` são MLPs relacionais treinadas de fato. `RightOf`/`Above` são o mesmo predicado com argumentos trocados. `CloseTo` é uma fórmula fechada (kernel Gaussiano). `InBetween`, `lastOnTheLeft`, `lastOnTheRight`, `canStack` são apenas composições lógicas (consultas, sem parâmetros treináveis).

---

## 1. Estratégia geral (linguagem natural)

1. **Setup**: `pip install ltn`, imports, e **um único ponto de definição da seed** (ver Seção 4) para garantir reprodutibilidade total (geração de dados, inicialização de pesos, otimizador).
2. **Geração de dados**: `generate_scene(seed)` cria 25 objetos com coordenadas (x,y) sorteadas com **rejeição de sobreposição** (nenhum par de objetos pode ficar a uma distância menor que `MIN_DIST`), mais cor/forma/tamanho one-hot.
3. **Visualização**: `plot_scene()` — scatter 2D com marcador por forma, cor por cor RGB, tamanho do marcador por tamanho (pedido na Tarefa 1).
4. **Vocabulário lógico**: `Not/And/Or/Implies/Equiv/Forall/Exists` na "configuração produto estável" (notebook `2b-operators_and_gradients.ipynb`), com `stable=True` nos agregadores.
5. **Predicados aprendidos**: `IsA(x, label_onehot)` (Tarefa 1.2), `LeftOf(x,y)` (Tarefa 2.1), `Below(x,y)` (Tarefa 3), todos MLPs pequenas via `ltn.Predicate`.
6. **Predicados/funções compostos** (sem peso próprio, via `ltn.Predicate.Lambda`): `RightOf`, `Above`, `CloseTo`, `InBetween`, `lastOnTheLeft`, `lastOnTheRight`, `canStack`.
7. **Base de Conhecimento**: função `axioms(features, labels)` que monta a lista de fórmulas (Tarefas 1–3) e agrega com `ltn.Wrapper_Formula_Aggregator`.
8. **Treinamento**: um único loop, minimizando `1 - SatAgg` com Adam, sobre a cena de treino.
9. **Avaliação/generalização**: `evaluate_scene(features, labels)` gera métricas de satisfatibilidade + classificação (Acurácia/Precisão/Recall/F1) numa cena nova, sem treinar.
10. **Repetição**: 5 seeds de teste diferentes → `pandas.DataFrame` consolidado, salvo em CSV.
11. **Tarefa 4**: funções `query_*` para as 3 fórmulas de raciocínio composto do enunciado.
12. **Ponto extra (XAI)**: para cada consulta, reportar a "melhor testemunha" (indexação do tensor via `.take()`) e, opcionalmente, uma checagem de consequência lógica por refutação (notebook `4-reasoning.ipynb`).

---

## 2. Sequência de passos (células do Colab)

| # | Célula | Conteúdo |
|---|--------|----------|
| 1 | Setup | `!pip install ltn -q`; imports (`ltn`, `tensorflow`, `numpy`, `pandas`, `matplotlib`, `itertools`) |
| 2 | **Seed única + Config** | `SEED_GLOBAL`, `set_all_seeds(seed)`, constantes de forma/cor, dimensão do vetor, hiperparâmetros |
| 3 | Geração de dados | `generate_scene(seed)` com checagem anti-sobreposição |
| 4 | Visualização | `plot_scene()` |
| 5 | Vocabulário lógico | `Not/And/Or/Implies/Equiv/Forall/Exists/formula_aggregator` |
| 6 | Constantes de rótulo | `ltn.Constant` one-hot por forma/tamanho |
| 7 | Predicados aprendidos | `IsAModel`, `RelModel`; instanciar `IsA`, `LeftOf`, `Below` |
| 8 | Predicados compostos | `RightOf`, `Above`, `CloseTo`, `InBetween`, `lastOnTheLeft`, `lastOnTheRight`, `canStack` |
| 9 | Base de Conhecimento | `axioms(features, labels)` — Tarefas 1, 2, 3 |
| 10 | Treinamento | loop único sobre a cena de treino |
| 11 | Avaliação single-run | `evaluate_scene()` |
| 12 | Loop de 5 execuções | 5 seeds de teste → `DataFrame` → CSV |
| 13 | Tarefa 4 | consultas compostas |
| 14 | Ponto extra (XAI) | testemunhas + refutação lógica |

---

## 3. Esqueleto de código

```python
# =========================================================
# 1. SETUP
# =========================================================
!pip install ltn -q

import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import ltn

# =========================================================
# 2. SEED ÚNICA + CONFIG
# =========================================================
SEED_GLOBAL = 42          # <<< ÚNICO PONTO QUE CONTROLA TODA A ALEATORIEDADE

def set_all_seeds(seed=SEED_GLOBAL):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_all_seeds(SEED_GLOBAL)   # chamado uma vez, aqui, antes de tudo

SHAPES = ["circle", "square", "cylinder", "cone", "triangle"]
COLORS = ["red", "green", "blue"]
N_PER_SHAPE = 5                       # 5 objetos por forma -> 25 no total
VEC_DIM = 11                          # x,y,r,g,b,circle,square,cylinder,cone,triangle,size
MIN_DIST = 0.08                       # distância mínima entre quaisquer dois objetos (evita sobreposição)
LR = 0.001
EPOCHS = 3000
P_FORALL, P_EXISTS = 2, 6
CLOSE_TO_GAMMA = 8.0

# =========================================================
# 3. GERAÇÃO DE DADOS (sem sobreposição de coordenadas)
# =========================================================
def _sample_non_overlapping_points(n, rng, min_dist=MIN_DIST, max_tries=10_000):
    """Amostra n pontos em [0,1]^2 tais que nenhum par fique a distância < min_dist."""
    pts = []
    tries = 0
    while len(pts) < n:
        p = rng.uniform(0.0, 1.0, size=2)
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
        tries += 1
        if tries > max_tries:
            raise RuntimeError("Não foi possível posicionar objetos sem sobreposição; "
                                "reduza MIN_DIST ou o número de objetos.")
    return np.array(pts, dtype=np.float32)

def generate_scene(seed, n_per_shape=N_PER_SHAPE):
    """Gera uma cena com len(SHAPES)*n_per_shape objetos, sem coordenadas sobrepostas."""
    rng = np.random.default_rng(seed)   # gerador local, isolado -> reprodutível por cena
    n = len(SHAPES) * n_per_shape
    coords = _sample_non_overlapping_points(n, rng)

    rows, shape_idx, color_idx, size_idx = [], [], [], []
    k = 0
    for s_i, shape in enumerate(SHAPES):
        for _ in range(n_per_shape):
            x, y = coords[k]; k += 1
            c_i = rng.integers(0, len(COLORS))
            sz = rng.integers(0, 2)  # 0 = pequeno, 1 = grande
            color_onehot = np.eye(len(COLORS))[c_i]
            shape_onehot = np.eye(len(SHAPES))[s_i]
            rows.append(np.concatenate([[x, y], color_onehot, shape_onehot, [float(sz)]]))
            shape_idx.append(s_i); color_idx.append(c_i); size_idx.append(sz)

    features = np.array(rows, dtype=np.float32)
    labels = dict(shape_idx=np.array(shape_idx), color_idx=np.array(color_idx),
                  size_idx=np.array(size_idx))
    return features, labels

# =========================================================
# 4. VISUALIZAÇÃO
# =========================================================
MARKERS = {"circle": "o", "square": "s", "cylinder": "D", "cone": "^", "triangle": "v"}
COLOR_MAP = {"red": "red", "green": "green", "blue": "blue"}

def plot_scene(features, labels, title="Cena"):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(features)):
        shape = SHAPES[labels["shape_idx"][i]]
        color = COLORS[labels["color_idx"][i]]
        size = 160 if labels["size_idx"][i] == 1 else 60
        ax.scatter(features[i, 0], features[i, 1], marker=MARKERS[shape],
                   c=COLOR_MAP[color], s=size, edgecolors="black")
        ax.annotate(str(i), (features[i, 0], features[i, 1]), fontsize=7)
    ax.set_title(title); ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    plt.show()

# =========================================================
# 5. VOCABULÁRIO LÓGICO ("configuração produto estável")
# =========================================================
Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Equiv = ltn.Wrapper_Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.And_Prod(),
                                                     ltn.fuzzy_ops.Implies_Reichenbach()))
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=P_FORALL, stable=True), semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=P_EXISTS, stable=True), semantics="exists")
formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=P_FORALL, stable=True))

# =========================================================
# 6. CONSTANTES DE RÓTULO
# =========================================================
def onehot_constant(index, size):
    v = np.zeros(size, dtype=np.float32); v[index] = 1.0
    return ltn.Constant(v, trainable=False)

shape_constants = {s: onehot_constant(i, len(SHAPES)) for i, s in enumerate(SHAPES)}
size_constants = {"small": onehot_constant(0, 2), "big": onehot_constant(1, 2)}

# =========================================================
# 7. PREDICADOS APRENDIDOS
# =========================================================
class IsAModel(tf.keras.Model):
    """Classificador genérico: objeto (11d) + rótulo one-hot (5d ou 2d) -> [0,1]."""
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(16, activation="elu")
        self.d2 = tf.keras.layers.Dense(1, activation="sigmoid")
    def call(self, inputs):
        obj, label = inputs
        return tf.squeeze(self.d2(self.d1(tf.concat([obj, label], axis=1))), axis=-1)

class RelModel(tf.keras.Model):
    """MLP relacional genérica: dois objetos concatenados (22d) -> [0,1]."""
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(16, activation="elu")
        self.d2 = tf.keras.layers.Dense(1, activation="sigmoid")
    def call(self, inputs):
        a, b = inputs
        return tf.squeeze(self.d2(self.d1(tf.concat([a, b], axis=1))), axis=-1)

IsA = ltn.Predicate(IsAModel())
LeftOf = ltn.Predicate(RelModel())
Below = ltn.Predicate(RelModel())

def isShapePredicate(shape_name):
    return lambda x: IsA([x, shape_constants[shape_name]])

isCircle, isSquare, isCylinder = isShapePredicate("circle"), isShapePredicate("square"), isShapePredicate("cylinder")
isCone, isTriangle = isShapePredicate("cone"), isShapePredicate("triangle")
isSmall = lambda x: IsA([x, size_constants["small"]])
isBig = lambda x: IsA([x, size_constants["big"]])
shape_predicates = [isCircle, isSquare, isCylinder, isCone, isTriangle]

# =========================================================
# 8. PREDICADOS/FUNÇÕES COMPOSTOS
# =========================================================
RightOf = ltn.Predicate.Lambda(lambda args: LeftOf([args[1], args[0]]).tensor)
Above = ltn.Predicate.Lambda(lambda args: Below([args[1], args[0]]).tensor)

CloseTo = ltn.Predicate.Lambda(
    lambda args: tf.exp(-CLOSE_TO_GAMMA * tf.reduce_sum(
        tf.square(args[0][:, 0:2] - args[1][:, 0:2]), axis=1))
)

def InBetween(x, y, z):
    return Or(And(LeftOf([y, x]), RightOf([z, x])),
              And(LeftOf([z, x]), RightOf([y, x])))

def canStack(x, y):
    not_cone_or_tri = And(Not(isCone(y)), Not(isTriangle(y)))
    return And(not_cone_or_tri, CloseTo([x, y]))

# --- "professores" (ground truth) usados só como âncora de supervisão ---
GT_LeftOf = ltn.Predicate.Lambda(
    lambda args: tf.sigmoid(20.0 * (args[1][:, 0] - args[0][:, 0]))  # x_b - x_a
)
GT_Below = ltn.Predicate.Lambda(
    lambda args: tf.sigmoid(20.0 * (args[1][:, 1] - args[0][:, 1]))  # y_b - y_a
)

# =========================================================
# 9. BASE DE CONHECIMENTO (AXIOMAS)
# =========================================================
def axioms(features, labels):
    x = ltn.Variable("x", features)
    y = ltn.Variable("y", features)
    z = ltn.Variable("z", features)

    ax = []

    # --- Tarefa 1: taxonomia de forma/tamanho ---
    shape_terms = [p(x) for p in shape_predicates]
    for p_i, p_j in itertools.combinations(shape_terms, 2):
        ax.append(Forall(x, Not(And(p_i, p_j))))          # unicidade (todos os pares)
    ax.append(Forall(x, Or(*shape_terms)))                 # completude de forma
    ax.append(Forall(x, Not(And(isSmall(x), isBig(x)))))   # unicidade de tamanho
    ax.append(Forall(x, Or(isSmall(x), isBig(x))))         # completude de tamanho

    # supervisão direta de forma/tamanho (rótulos conhecidos por construção)
    for s_i, pred in enumerate(shape_predicates):
        mask = labels["shape_idx"] == s_i
        if mask.any():
            xs = ltn.Variable(f"x_shape_{s_i}", features[mask])
            ax.append(Forall(xs, pred(xs)))
    for sz_i, pred in [(0, isSmall), (1, isBig)]:
        mask = labels["size_idx"] == sz_i
        if mask.any():
            xs = ltn.Variable(f"x_size_{sz_i}", features[mask])
            ax.append(Forall(xs, pred(xs)))

    # --- Tarefa 2: raciocínio horizontal ---
    ax.append(Forall(x, Not(LeftOf([x, x]))))                                # irreflexividade
    ax.append(Forall([x, y], Implies(LeftOf([x, y]), Not(LeftOf([y, x])))))  # assimetria
    ax.append(Forall([x, y], Equiv(LeftOf([x, y]), RightOf([y, x]))))        # inverso
    ax.append(Forall([x, y, z],
        Implies(And(LeftOf([x, y]), LeftOf([y, z])), LeftOf([x, z]))))      # transitividade
    ax.append(Forall([x, y], Equiv(LeftOf([x, y]), GT_LeftOf([x, y]))))     # âncora de significado

    # --- Tarefa 3: raciocínio vertical ---
    ax.append(Forall(x, Not(Below([x, x]))))
    ax.append(Forall([x, y], Implies(Below([x, y]), Not(Below([y, x])))))
    ax.append(Forall([x, y], Equiv(Below([x, y]), Above([y, x]))))
    ax.append(Forall([x, y, z],
        Implies(And(Below([x, y]), Below([y, z])), Below([x, z]))))
    ax.append(Forall([x, y], Equiv(Below([x, y]), GT_Below([x, y]))))

    sat = formula_aggregator(ax)
    return sat.tensor if hasattr(sat, "tensor") else sat

# =========================================================
# 10. TREINAMENTO (uma única cena)
# =========================================================
set_all_seeds(SEED_GLOBAL)   # garante que a cena de treino também é reprodutível
train_features, train_labels = generate_scene(seed=SEED_GLOBAL)
plot_scene(train_features, train_labels, "Cena de treino")

trainable_vars = IsA.trainable_variables + LeftOf.trainable_variables + Below.trainable_variables
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

history = []
for epoch in range(EPOCHS):
    with tf.GradientTape() as tape:
        sat = axioms(train_features, train_labels)
        loss = 1.0 - sat
    grads = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))
    if epoch % 200 == 0:
        history.append((epoch, float(sat)))
        print(f"Epoch {epoch}: SatAgg = {float(sat):.4f}")

# =========================================================
# 11. AVALIAÇÃO (satisfatibilidade + métricas de classificação)
# =========================================================
def binary_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (np.asarray(y_pred_prob) > threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1)); tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0)); fn = np.sum((y_pred == 0) & (y_true == 1))
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1)

def evaluate_scene(features, labels):
    results = {"sat_kb": float(axioms(features, labels))}

    x_all = ltn.Variable("x_all", features)
    for name, pred, key, cls in [("isCircle", isCircle, "shape_idx", 0),
                                  ("isSquare", isSquare, "shape_idx", 1),
                                  ("isCylinder", isCylinder, "shape_idx", 2),
                                  ("isCone", isCone, "shape_idx", 3),
                                  ("isTriangle", isTriangle, "shape_idx", 4),
                                  ("isSmall", isSmall, "size_idx", 0),
                                  ("isBig", isBig, "size_idx", 1)]:
        prob = pred(x_all).tensor.numpy()
        y_true = (labels[key] == cls).astype(int)
        results[name] = binary_metrics(y_true, prob)

    n = len(features)
    xi, yi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    xa = ltn.Variable("xa", features[xi.flatten()])
    xb = ltn.Variable("xb", features[yi.flatten()])
    for name, pred, axis in [("leftOf", LeftOf, 0), ("below", Below, 1)]:
        prob = pred([xa, xb]).tensor.numpy()
        true = (features[xi.flatten(), axis] < features[yi.flatten(), axis]).astype(int)
        results[name] = binary_metrics(true, prob)

    return results

# =========================================================
# 12. LOOP DE 5 EXECUÇÕES (GENERALIZAÇÃO)
# =========================================================
TEST_SEEDS = [SEED_GLOBAL + 1000 + i for i in range(5)]  # derivadas da seed global -> reprodutível

all_runs = []
for run_id, seed in enumerate(TEST_SEEDS):
    feats, labs = generate_scene(seed)
    metrics = evaluate_scene(feats, labs)
    metrics["run"] = run_id
    all_runs.append(metrics)

flat_rows = []
for run in all_runs:
    row = {"run": run["run"], "sat_kb": run["sat_kb"]}
    for k, v in run.items():
        if isinstance(v, dict):
            for mk, mv in v.items():
                row[f"{k}_{mk}"] = mv
    flat_rows.append(row)

df_metrics = pd.DataFrame(flat_rows)
df_metrics.to_csv("metrics_5_runs.csv", index=False)
print(df_metrics.describe())

# =========================================================
# 13. TAREFA 4 — CONSULTAS COMPOSTAS
# =========================================================
def query_small_below_cylinder_left_square(features):
    x = ltn.Variable("x", features); y = ltn.Variable("y", features); z = ltn.Variable("z", features)
    formula = Exists(x, And(isSmall(x),
                    And(Exists(y, And(isCylinder(y), Below([x, y]))),
                        Exists(z, And(isSquare(z), LeftOf([x, z]))))))
    return float(formula.tensor)

def query_green_cone_inbetween(features):
    x = ltn.Variable("x", features); y = ltn.Variable("y", features); z = ltn.Variable("z", features)
    is_green = ltn.Predicate.Lambda(lambda a: a[:, 3])  # índice 3 = canal Green no one-hot de cor
    formula = Exists([x, y, z], And(isCone(x), And(is_green(x), InBetween(x, y, z))))
    return float(formula.tensor)

def query_triangles_close_same_size(features, labels):
    x = ltn.Variable("x", features); y = ltn.Variable("y", features)
    same_size = ltn.Predicate.Lambda(
        lambda a: tf.cast(tf.equal(a[0][:, -1], a[1][:, -1]), tf.float32))
    formula = Forall([x, y],
        Implies(And(isTriangle(x), And(isTriangle(y), CloseTo([x, y]))), same_size([x, y])))
    return float(formula.tensor)

# =========================================================
# 14. PONTO EXTRA — EXPLICAÇÃO (XAI)
# =========================================================
def explain_left_of(features, i, j):
    xi = ltn.Constant(features[i], trainable=False)
    xj = ltn.Constant(features[j], trainable=False)
    truth = float(LeftOf([xi, xj]).tensor)
    print(f"LeftOf(obj{i}, obj{j}) = {truth:.3f}  "
          f"(x_i={features[i,0]:.2f}, x_j={features[j,0]:.2f})")
    return truth

def best_witness_for_exists(features, formula_var_name="x"):
    """Exemplo: encontrar o objeto que melhor satisfaz isSmall(x) ∧ ∃y (isCylinder(y) ∧ below(x,y))."""
    x = ltn.Variable(formula_var_name, features)
    y = ltn.Variable("y", features)
    per_object = And(isSmall(x), Exists(y, And(isCylinder(y), Below([x, y]))))
    values = per_object.tensor.numpy()
    best_i = int(np.argmax(values))
    print(f"Melhor testemunha: objeto {best_i} com valor {values[best_i]:.3f}")
    return best_i, values
```

---

## 4. Reprodutibilidade (ponto único de controle)

- **`SEED_GLOBAL`** (célula 2) é a única constante que qualquer pessoa precisa mudar para reproduzir/variar o experimento inteiro.
- `set_all_seeds()` fixa `random`, `numpy` e `tensorflow` **antes** de qualquer geração de dado ou inicialização de peso.
- `generate_scene(seed)` usa um `np.random.default_rng(seed)` **local** (não o estado global), então cada cena (treino e as 5 de teste) é determinística e isolada — mudar a ordem das chamadas não muda os resultados de cada cena individualmente.
- As seeds de teste (`TEST_SEEDS`) são **derivadas** de `SEED_GLOBAL` (`SEED_GLOBAL + 1000 + i`) em vez de hardcoded soltas, então todo o experimento (treino + 5 avaliações) depende de um único número.
- Antes do treino, chama-se `set_all_seeds(SEED_GLOBAL)` de novo, para garantir que a inicialização dos pesos das redes (`IsAModel`, `RelModel`) seja sempre a mesma, independente de quantas células foram executadas antes (comum em notebooks).

## 5. Anti-sobreposição de coordenadas

- `_sample_non_overlapping_points` faz **rejection sampling**: sorteia um ponto, aceita apenas se a distância euclidiana a todos os pontos já aceitos for ≥ `MIN_DIST` (padrão 0.08 no espaço unitário [0,1]²).
- Isso garante, por construção, que nenhuma fórmula geométrica (`leftOf`, `below`, `closeTo`, `inBetween`) opere sobre um par de objetos empatados em x ou em y — o que quebraria a premissa de irreflexividade/assimetria (ex.: dois objetos exatamente na mesma posição não têm uma resposta bem definida de "quem está à esquerda de quem").
- Com 25 objetos e `MIN_DIST=0.08` a rejeição converge rapidamente; se for aumentado o número de objetos ou o `MIN_DIST`, o `max_tries` evita loop infinito e lança um erro explicativo.

---

## 6. Observações / pontos que podem ser modificados ou melhorados

1. **Âncora de supervisão (`GT_LeftOf`/`GT_Below`)**: é a parte mais "não-puramente-lógica" do MVP — sem ela, os axiomas lógicos sozinhos são subdeterminados (uma rede constante os satisfaz trivialmente). Aqui ela foi implementada como um predicado `Lambda` diferenciável (sigmoide da diferença de coordenadas) e entra na KB como mais um axioma (`Equiv`), o que é mais limpo do que somar um termo solto fora do `formula_aggregator` (like no plano v1). Vale documentar essa escolha no relatório e, se quiser, testar remover essa âncora e ver a KB colapsar — é uma boa demonstração para o relatório.
2. **`IsA` único vs. predicados separados**: reduz bastante o código; se o professor exigir redes fisicamente distintas por predicado, basta trocar `IsAModel` por 7 instâncias de uma MLP simples (um predicado por forma/tamanho).
3. **`canStack`**: o enunciado é vago ("mesmas dimensões" ou "centroide em distância horizontal estável"); o esqueleto usa `CloseTo` na posição x/y completa como proxy simples — ajustar/documentar o critério escolhido.
4. **Threshold fixo de 0.5**: suficiente e mais fácil de justificar para o MVP; pode evoluir para busca de threshold ótimo por F1 depois.
5. **Épocas/arquitetura (16 unidades, 3000 épocas)**: ponto de partida; ajustar observando a curva de SatAgg (idealmente perto de 1.0 nos axiomas lógicos puros antes de otimizar as métricas de classificação).
6. **`stable=True`**: ativado por padrão nos agregadores (evita gradientes explodindo/vanishing em casos extremos, conforme `2b-operators_and_gradients.ipynb`); pode ser desligado para comparar curvas de treino no relatório.
7. **Repetição de treino nas 5 execuções**: o plano treina **uma vez** e testa em 5 cenas novas (conforme orientação do professor — mede generalização de raciocínio, não estabilidade de treino). Uma variante opcional para o relatório é também retreinar do zero em cada uma das 5 execuções e comparar a variância.
8. **`ltn.diag`**: não usado neste MVP porque as fórmulas relacionais precisam do produto cartesiano completo (todos-contra-todos). Se, no futuro, quiser comparar pares específicos "zipados" (ex.: só pares pré-definidos), usar `ltn.diag(x,y)` dentro do quantificador evita o custo O(n²) desnecessário.
9. **Ponto extra (XAI)**: o esqueleto já oferece testemunha (melhor indivíduo que satisfaz uma fórmula existencial) e valor de verdade da sub-fórmula. Para uma explicação mais robusta, pode-se aplicar a técnica de **raciocínio por refutação** do notebook `4-reasoning.ipynb`: tentar encontrar uma grounding que satisfaça a KB treinada mas viole uma conclusão específica (ex.: "todo quadrado está à direita de todo círculo") — se a busca falhar (KB nunca atinge o threshold `q` sem satisfazer também a conclusão), isso é evidência de que a conclusão é uma consequência lógica aprendida, não uma coincidência dos dados de treino.
10. **Métricas de `leftOf`/`below` na avaliação**: comparam a predição do modelo com o ground-truth geométrico (comparação direta de coordenadas), igual ao usado na âncora de treino — é esperado que essas métricas fiquem altas por construção; o valor mais informativo para o relatório é a **satisfatibilidade dos axiomas puramente lógicos** (irreflexividade, assimetria, transitividade) nas cenas de teste, que mostra se o "raciocínio" generalizou de fato.
