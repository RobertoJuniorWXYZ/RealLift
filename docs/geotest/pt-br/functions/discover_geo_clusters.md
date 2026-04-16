# `reallift.geo.discovery.discover_geo_clusters`

Na biblioteca RealLift, a função `discover_geo_clusters` é o motor principal para o design experimental de testes incrementais baseados em geografia (Geo Experiments). Sua finalidade é prever e identificar a combinação ótima de sub-regiões de controle para formar um **Controle Sintético** de alta correlação e baixo erro estocástico em relação a uma região de tratamento alvo, **antes do período da intervenção.**

## Assinatura

```python
def discover_geo_clusters(
    filepath: str,
    date_col: str,
    geos: list = None,
    n_treatment: int = 3,
    fixed_treatment: list = None,
    start_date: str = None,
    end_date: str = None,
    use_elasticnet: bool = True,
    search_mode: str = "auto",
    verbose: bool = True,
    show_results: bool = True
) -> list
```

---

## 1. Objetivo

A função resolve o seguinte problema de otimização combinatória aplicada à Inferência Causal:

> **Dado um conjunto de $n$ séries temporais geográficas observadas no período pré-intervenção, particionar essas geografias em grupo de tratamento $(T)$ e grupo de controle/doador $(D)$ de modo que o Controle Sintético construído a partir de $D$ reproduza com máxima fidelidade a trajetória histórica de $T$.**

A qualidade dessa partição é o fator determinante para a credibilidade de todo o teste incremental subsequente.

O problema é não-trivial porque:
1. O espaço de buscas é combinatório: $C(n, k)$ partições possíveis crescem fatorialamente.
2. A construção do Controle Sintético envolve uma otimização convexa restrita aninhada dentro de cada avaliação.
3. A métrica de ranqueamento precisa equilibrar dois critérios potencialmente conflitantes (erro baixo vs. correlação alta).


---

## 2. Entradas (Inputs)

A função recebe dados tabulares em formato CSV com a seguinte estrutura esperada:

### 2.1 Estrutura do CSV

| Coluna | Descrição | Tipo | Restrições |
|:---|:---|:---|:---|
| `date_col` | Coluna temporal indexadora | `str` / `datetime` | Parseável como data; formato `YYYY-MM-DD` ou `DD/MM/YYYY` |
| `geo_1, geo_2, ..., geo_n` | Uma coluna por geografia | `float` / `int` | Valores ≥ 0 (exigido pela transformação logarítmica) |

### 2.2 Requisitos formais dos dados

- **Granularidade temporal:** Diária (a função agrega duplicatas via `groupby(date_col).sum()`).
- **Completude:** Todas as geografias devem ter observações em todas as datas. Valores ausentes resultam em `NaN` após a transformação log-diff e causam falha silenciosa na otimização.
- **Positividade estrita:** Como a primeira transformação aplicada é $\log(x)$, valores zero ou negativos produzem $-\infty$ ou `NaN`. Se uma geografia possui dias com valor zero, ela deve ser excluída ou imputada antes da chamada.
- **Horizonte mínimo:** O período pré-intervenção deve conter observações suficientes para que a diferenciação ($T-1$ observações úteis) e a otimização do ElasticNet e CVXPY sejam numericamente estáveis. Na prática, recomenda-se $T \geq 30$ dias.


---

## 3. Parâmetros

### 3.1 Parâmetros obrigatórios

| Parâmetro | Tipo | Descrição |
|:---|:---|:---|
| `filepath` | `str` | Caminho absoluto ou relativo para o arquivo CSV de entrada (fator diário por geografias). |
| `date_col` | `str` | Coluna string/datetime representando a data de cada observação sequencial. |

### 3.2 Parâmetros opcionais (configuração do experimento)

| Parâmetro | Tipo | Default | Descrição |
|:---|:---|:---|:---|
| `geos` | `list[str]` | `None` | Especifica ou restringe a gama total de praças passíveis de leitura na base de dados. Se `None`, usa todas as colunas numéricas do CSV. |
| `n_treatment` | `int` | `3` | Número de geografias no grupo de tratamento. Utilizado caso `fixed_treatment` for `None`. O software gerará partições combinatórias contendo esse *N* número de lugares em cada simulação. |
| `fixed_treatment` | `list[str]` | `None` | Lista nominal das geografias que sua equipe ativamente escolheu intervir. Desabilita o combinatório `n_treatment` e analisa o cenário ótimo focado unicamente para salvaguardar os dados dos alvos escolhidos. |
| `start_date` | `str` | `None` | Data inicial do período de análise no formato `YYYY-MM-DD`. Filtra registros anteriores, certificando de testar as sinergias passadas **sem influência** do próprio tratamento (Data Leak). |
| `end_date` | `str` | `None` | Data final do período de análise no formato `YYYY-MM-DD`. Filtra registros posteriores. |

### 3.3 Hiperparâmetros algorítmicos

| Parâmetro | Tipo | Default | Descrição |
|:---|:---|:---|:---|
| `use_elasticnet` | `bool` | `True` | Ativa a pré-filtragem de controles via ElasticNet. Quando `False`, todos os controles entram diretamente na otimização CVXPY com pesos uniformes como base inicial. |
| `search_mode` | `str` | `"auto"` | Estratégia de busca: `"exhaustive"`, `"ranking"` ou `"auto"`. Ver seção 5.1 para detalhamento completo. |

### 3.4 Parâmetros de interface

| Parâmetro | Tipo | Default | Descrição |
|:---|:---|:---|:---|
| `verbose` | `bool` | `True` | Exibe logs progressivos e barra de progresso (via `tqdm`). |
| `show_results` | `bool` | `True` | Imprime a tabela formatada de resultados ao final da execução. |

### 3.5 Hiperparâmetros internos (não expostos na API)

A função realiza *grid search* sobre dois hiperparâmetros do ElasticNet:

| Hiperparâmetro | Grid | Significado |
|:---|:---|:---|
| `alpha` | `[0.001, 0.01, 0.1]` | Intensidade da regularização. Valores menores → menos penalização → mais controles sobrevivem. |
| `l1_ratio` | `[0.2, 0.5, 0.8]` | Balanço entre L1 (Lasso, esparsidade) e L2 (Ridge, suavidade). Valores altos → mais esparsidade → menos controles. |

Isso gera uma grade de $3 \times 3 = 9$ configurações avaliadas **para cada combinação de tratamento**. Apenas o melhor resultado local (menor `std_residual`) é retido por combinação.


---

## 4. Base Teórica e Hipóteses

### 4.1 Fundamento: Método de Controle Sintético (Abadie et al., 2003, 2010)

A função implementa uma variante do Synthetic Control Method (SCM). O SCM propõe que, na ausência de um experimento randomizado, a trajetória contrafactual de uma unidade tratada pode ser estimada como uma **combinação convexa ponderada** de unidades não tratadas:

$$
\hat{Y}_{T,t} = \sum_{j \in D} w_j \cdot Y_{j,t} \quad \text{com} \quad w_j \geq 0, \quad \sum_j w_j = 1
$$

onde $\hat{Y}_{T,t}$ é o valor sintético no tempo $t$, $Y_{j,t}$ é o valor observado do doador $j$, e $w_j$ é o peso otimizado.

### 4.2 Hipóteses assumidas

1. **Paralelismo latente (Parallel Trends generalizado):** Existe uma combinação linear convexa de controles que replica a trajetória do tratamento no período pré-intervenção. Se essa combinação não existir, o Controle Sintético é inerentemente inadequado para o problema.

2. **Estabilidade estrutural:** Os padrões de co-movimento observados no período pré-intervenção se mantêm no período pós-intervenção, exceto pelo efeito causal do tratamento. Choques exógenos assimétricos (ex: desastre natural localizado) violam essa hipótese.

3. **Não-antecipação:** As unidades de tratamento não alteram seu comportamento antes da data de início do tratamento. A função opera exclusivamente sobre dados pré-intervenção, mas se houver vazamento temporal, o ajuste histórico será inflado artificialmente.

4. **Ausência de interferência (SUTVA):** O tratamento de uma geo não afeta os outcomes das geos de controle. Em contextos de marketing digital com spillover geográfico, essa hipótese pode ser violada.

### 4.3 Extensão implementada: Filtrando a "Multidão" com ElasticNet

O SCM clássico (Abadie) não realiza seleção de variáveis — todas as unidades do donor pool participam da otimização. Com muitas cidades no controle, comparar todas geraria muito ruído estocástico (*overfitting*). A função introduz uma etapa prévia de **regularização ElasticNet** que avalia simultaneamente dezenas de doadores e naturalmente "zera" a relevância das cidades que não ajudam na aproximação do Tratamento:

$$
\min_{w} \frac{1}{2T} \lVert Xw - y \rVert_2^2 + \alpha \cdot \lambda_1 \lVert w \rVert_1 + \frac{\alpha}{2} (1 - \lambda_1) \lVert w \rVert_2^2
$$

A penalização L1 induz esparsidade (zera coeficientes de controles irrelevantes), enquanto a L2 estabiliza a solução quando controles são multicolineares.

**A Regra do Positivo:** Se uma cidade candidata recebe um peso negativo (ou seja, ela cresce isoladamente quando o alvo principal cai), o vetor a descarta. Controles exigem similaridade real vetorial, e não correlações espelhadas bizarras.

*(Obs: Quando acionado através do parâmetro `use_elasticnet=False` nos bastidores puristas do DiD, o algoritmo intercede para criar controles sob restrição fixa puramente baseado na correlação vetorial holística [1/N], evitando penalização convexa).*

### 4.4 Simplificações

- **Normalização por média:** A otimização CVXPY normaliza séries dividindo pela média temporal ($y / \bar{y}$, $X / \bar{X}$), ao invés de usar a normalização clássica V-ponderada de Abadie. Isso simplifica o problema mas perde a capacidade de ponderar covariáveis auxiliares.
- **Sem intercepto:** O modelo sintético não inclui intercepto ($\hat{Y} = Xw$, não $\hat{Y} = Xw + b$), forçando o controle a explicar tanto o nível quanto a dinâmica.
- **Grid search discreto:** Os hiperparâmetros do ElasticNet são buscados em grade fixa de 9 pontos, não via validação cruzada. Isso é pragmático (velocidade), mas pode não encontrar o ótimo global de regularização.


---

## 5. Mecanismo / Transformação

A função executa um *pipeline* sequencial de 6 estágios para cada candidato a grupo de tratamento.

### 5.0 Preparação dos dados

```
CSV bruto → parse de datas → filtro temporal → agrupamento diário → ordenação cronológica
```

A saída é um DataFrame $\mathbf{D} \in \mathbb{R}^{T \times n}$ onde $T$ é o número de dias e $n$ é o número de geos.

### 5.1 Seleção da estratégia de busca (`search_mode`)

O comportamento de busca muda dependendo de como a função é chamada:

| Condição | Modo efetivo | Comportamento |
|:---|:---|:---|
| `fixed_treatment` fornecido | `fixed` | Avalia cada geo fixo individualmente, com isolamento cruzado entre todos os fixos |
| `n_treatment == 1` | `exhaustive` | Cada geo é testado como tratamento unitário ($C(n,1) = n$ iterações) |
| `search_mode="exhaustive"` | `exhaustive` | Gera todas as $C(n, k)$ combinações e avalia cada uma como grupo |
| `search_mode="ranking"` | `ranking` | Heurística gulosa em duas fases (ver 5.1.2) |
| `search_mode="auto"` | dinâmico | $C(n,k) > 1000 \Rightarrow$ `ranking`; caso contrário, `exhaustive` |

#### 5.1.1 Modo Exaustivo (`"exhaustive"`) — Busca Combinatória Completa

A busca exaustiva testa **todas** as combinações matemáticas possíveis $C(n, k)$, onde $n$ é o número total de geografias e $k$ é o `n_treatment`:

$$
C(n, k) = \frac{n!}{k! \cdot (n - k)!}
$$

**Exemplos práticos de escala:**

| Geos ($n$) | Tratamentos ($k$) | Combinações $C(n,k)$ |
|:---:|:---:|---:|
| 10 | 3 | 120 |
| 15 | 3 | 455 |
| 20 | 5 | 15.504 |
| 25 | 5 | 53.130 |
| 27 | 5 | 80.730 |
| 30 | 5 | 142.506 |

**Fluxo algorítmico:**

Cada combinação é avaliada e ranqueada pelo **Synthetic Error Ratio (SER)** — uma métrica que penaliza erro alto e/ou correlação baixa entre o tratamento e seu Controle Sintético. Formalmente, $\text{SER} = \sigma_e / (\rho + 10^{-6})$, onde $\sigma_e$ é o desvio padrão do resíduo e $\rho$ é a correlação de Pearson (ver detalhamento completo na seção 6).

```
Para CADA combinação C(n,k):
  1. Definir grupo de tratamento (ex: [SP, RJ, MG, BA, PR])
  2. Definir donor pool = todas as geos EXCETO o tratamento
  3. Criar a série-alvo: y = média(tratamentos)
  4. Aplicar ElasticNet para filtrar controles irrelevantes
  5. Otimizar pesos via CVXPY (Controle Sintético)
  6. Calcular métricas: std_residual, correlation, SER
  7. Armazenar resultado
                    ↓
Ordenar TODAS as combinações por SER crescente
                    ↓
Retornar as TOP 5 como RECOMMENDATION 0..4
```

**Característica fundamental:** cada iteração avalia o grupo de tratamento **como unidade composta** — a série-alvo `y` é a média dos `k` geos agindo juntos. Isso captura **sinergias e redundâncias** entre geos que uma busca individual jamais detectaria.

**Resultado — 5 Alternativas Independentes:**

O modo exaustivo retorna as **5 melhores combinações completas**, ranqueadas por SER:

```
RECOMMENDATION 0 | Treatment: [Geo3, Geo8, Geo15, Geo21, Geo27] | SER: 0.312  ← melhor
RECOMMENDATION 1 | Treatment: [Geo3, Geo8, Geo15, Geo21, Geo24] | SER: 0.318
RECOMMENDATION 2 | Treatment: [Geo2, Geo8, Geo15, Geo21, Geo27] | SER: 0.325
RECOMMENDATION 3 | Treatment: [Geo3, Geo9, Geo15, Geo21, Geo27] | SER: 0.331
RECOMMENDATION 4 | Treatment: [Geo4, Geo8, Geo15, Geo21, Geo27] | SER: 0.337
```

Se a RECOMMENDATION 0 não for viável por restrições operacionais (ex: uma praça indisponível para campanha), o usuário pode adotar a RECOMMENDATION 1 com confiança — cada alternativa foi **independentemente otimizada** sobre o espaço completo.

---

#### 5.1.2 Modo Ranking (`"ranking"`) — Busca Gulosa em Duas Fases

A busca por ranking é uma **heurística gulosa** (*greedy*) que reduz drasticamente o número de avaliações ao assumir que os melhores geos individuais formam o melhor grupo.

**Fluxo algorítmico:**

```
FASE 1 — Triagem Individual (n avaliações):
  Para CADA geo isoladamente:
    1. Tratar a geo como tratamento unitário
    2. Donor pool = todas as demais geos
    3. Ajustar Controle Sintético
    4. Calcular SER individual
                    ↓
  Ordenar as n geos por SER crescente
  Selecionar as top-k (melhores individuais)

FASE 2 — Reavaliação com Exclusão Cruzada (k avaliações):
  Para CADA geo do top-k:
    1. Tratar a geo como tratamento
    2. Donor pool = todas as geos EXCETO todas as top-k
    3. Reajustar Controle Sintético com isolamento total
    4. Recalcular SER
                    ↓
Retornar os k clusters como TEST CLUSTER 0..k-1
```

**Total de avaliações:** $n + k$ (ex: 27 + 5 = 32), contra 80.730 no exaustivo.

**Resultado — Um Único Grupo:**

O ranking retorna **um único grupo de tratamento** composto pelos top-k geos individuais. Não há alternativas — se o grupo não servir, é necessário reconfigurar manualmente os parâmetros.

> [!WARNING]
> **Falácia Composicional da Busca Gulosa**
> O ranking assume que os melhores geos individuais formam o melhor grupo. Isso nem sempre é verdade. Duas geos podem ser individualmente excelentes, mas altamente correlacionadas entre si — quando combinadas, "disputam" os mesmos controles e degradam a qualidade do Controle Sintético conjunto. O modo exaustivo é imune a esse problema porque avalia cada combinação como grupo completo.

---

#### 5.1.3 Modo Auto (`"auto"`) — Seleção Adaptativa (padrão)

O modo automático decide entre exaustivo e ranking com base na cardinalidade do problema:

```python
if C(n, k) > 1000:
    usar "ranking"
else:
    usar "exhaustive"
```

Isso garante que problemas pequenos (até ~1.000 combinações) recebam a solução ótima completa, enquanto problemas grandes evitam tempos proibitivos de execução.

---

#### Tabela Comparativa dos Modos de Busca

| Característica | `"exhaustive"` | `"ranking"` | `"auto"` |
|:---|:---|:---|:---|
| **O que cada iteração testa** | Grupo completo de $k$ geos | 1 geo isolado | Depende de $C(n,k)$ |
| **Nº de avaliações** | $C(n, k)$ | $n + k$ | Adaptativo |
| **Captura sinergia entre geos** | Sim | Não | Depende |
| **Alternativas retornadas** | Top 5 Recommendations | Grupo único | Depende |
| **Velocidade** | Lenta para $C(n,k)$ grande | Rápida | Adaptativa |
| **Quando usar** | $C(n,k) \leq 10.000$ ou precisão máxima | Bases com muitas geos ($n > 30$) | Uso geral |

> [!TIP]
> **Recomendação Prática:** Para decisões de alto impacto financeiro, force `search_mode="exhaustive"` mesmo com tempos de execução maiores. As 5 alternativas retornadas oferecem flexibilidade operacional que a busca gulosa não proporciona. Reserve `"ranking"` para prototipagem rápida ou bases com dezenas de geos onde o exaustivo seria computacionalmente inviável.

### 5.2 Transformação log-diff (`log_diff_transform`)

Para evitar que o modelo seja enganado por tendências de longo prazo ou distorções de escala, aplica-se uma transformação de log e diferença. Para cada série geográfica $Y_{j,t}$:

$$
Z_{j,t} = \Delta \log Y_{j,t} = \log Y_{j,t} - \log Y_{j,t-1} = \log\left(\frac{Y_{j,t}}{Y_{j,t-1}}\right)
$$

**Propósito triplo:**

1. **Estacionarização:** Remove tendências determinísticas. A série resultante representa taxas de crescimento relativas, não níveis absolutos.
2. **Normalização de escala:** Séries com magnitudes muito diferentes (ex: São Paulo vs. Macapá) tornam-se comparáveis em escala logarítmica.
3. **Estabilização de variância:** O logaritmo comprime a cauda direita, reduzindo a heterocedasticidade típica de séries de receita/vendas.

**Consequência:** A otimização posterior opera sobre **retornos logarítmicos diários**, não sobre valores brutos. Isso foca o modelo em alinhar a *dinâmica relativa* (padrões de sub/sobre-performance dia a dia), tornando a série estacionária.

A operação consome 1 observação (a primeira diferença elimina $t=0$), reduzindo a série de $T$ para $T-1$ pontos.

### 5.3 Pré-filtragem de controles (ElasticNet)

Com `use_elasticnet=True`, cada combinação é submetida ao seguinte procedimento:

1. **Padronização:** A matriz de controles $X \in \mathbb{R}^{(T-1) \times (n-k)}$ é padronizada (média zero, variância unitária) via `StandardScaler`.

2. **Grid Search:** Para cada par $(\alpha, \lambda_1)$ no grid $\{0.001, 0.01, 0.1\} \times \{0.2, 0.5, 0.8\}$, ajusta-se:

$$
\hat{w} = \arg\min_{w} \frac{1}{2(T-1)} \lVert Xw - y \rVert_2^2 + \alpha \lambda_1 \lVert w \rVert_1 + \frac{\alpha(1-\lambda_1)}{2} \lVert w \rVert_2^2
$$

3. **Seleção positiva:** Apenas controles com $\hat{w}_j > 0$ são selecionados. Se todos os coeficientes forem $\leq 0$, o controle com maior $\lvert \hat{w}_j \rvert$ é forçado como selecionado (fallback de segurança).

**Intuição:** O ElasticNet funciona como um "filtro de relevância" que separa a multidão de candidatos. Controles que não contribuem para explicar a variação do tratamento recebem peso zero (contribuição L1/Lasso). Controles com relação negativa (movem-se em direção oposta ao tratamento) são explicitamente excluídos pela regra do coeficiente positivo.

### 5.4 Otimização do Controle Sintético (CVXPY)

Os controles que sobrevivem ao filtro ElasticNet passam pela otimização convexa que define os pesos definitivos. Esta etapa opera sobre os **dados originais** (não transformados), garantindo que os pesos reflitam o ajuste em escala real.

**Normalização prévia:**

$$
\tilde{y}_t = \frac{y_t}{\bar{y}}, \qquad \tilde{X}_{j,t} = \frac{X_{j,t}}{\bar{X}_j}
$$

onde $\bar{y} = \frac{1}{T}\sum_t y_t$ e $\bar{X}_j = \frac{1}{T}\sum_t X_{j,t}$

**Problema de otimização:**

$$
\min_{w} \sum_{t=1}^{T} \left(\tilde{y}_t - \sum_{j} w_j \tilde{X}_{j,t}\right)^2
$$

sujeito a:

$$
w_j \geq 0 \quad \forall j, \qquad \sum_j w_j = 1
$$

Resolvido via solver SCS (Splitting Conic Solver) do pacote `cvxpy`.

**Poda final:** Controles com $w_j < 0.001$ (0.1%) são removidos. Os pesos remanescentes são renormalizados para somar 1:

$$
w_j^* = \frac{w_j}{\sum_{i} w_i} \quad \text{para } w_j \geq 0.001
$$

Se alguma cidade receber um peso insignificante (abaixo de 0.1%), ela é podada do *donor pool* definitivo.

### 5.5 Avaliação do ajuste (métricas)

Com os pesos finais $w^*$ e os controles podados, a série predita é calculada no **espaço log-diff** (transformado):

$$
\hat{y}_t = \sum_j w_j^* Z_{j,t} \quad \text{(sem ElasticNet)}
$$

ou

$$
\hat{y}_t = f_{\text{ElasticNet}}(X_{\text{scaled}}) \quad \text{(com ElasticNet, refit nos controles selecionados)}
$$

O resíduo é $e_t = y_t - \hat{y}_t$, e as métricas computadas são:

| Métrica | Fórmula | Interpretação |
|:---|:---|:---|
| `std_residual` | $\sigma_e = \sqrt{\frac{1}{T-1}\sum_t (e_t - \bar{e})^2}$ | Dispersão do erro diário. Baixo = sintético estável. |
| `rmspe` | $\text{RMSPE} = \sqrt{\frac{1}{T-1}\sum_t e_t^2}$ | Magnitude absoluta do erro. Inclui viés. |
| `correlation` | $\rho = \text{corr}(y, \hat{y})$ | Sincronismo direcional. Alto = movimentos em fase. |
| `synthetic_error_ratio` | $\text{SER} = \frac{\sigma_e}{\rho + 10^{-6}}$ | Métrica de ranqueamento final. Ver seção 6. |

### 5.6 Seleção do melhor hiperparâmetro local

Dentro das 9 configurações do grid ElasticNet, apenas o candidato com **menor `std_residual`** é retido para representar aquela combinação de tratamento. Isso significa que cada combinação de geos contribui com exatamente um resultado para o ranqueamento global.


---

## 6. Função Objetivo: Synthetic Error Ratio (SER)

### 6.1 Métrica de ranqueamento

O algoritmo final avalia as oscilações preditas em relação às reais com o grupo vencedor e ranqueia todas as opções baseando-se em duas variáveis fundamentais:

- **Resíduo Padrão (`std_residual`)**: Mede o erro absoluto histórico diário entre o real e o simulado. Representa a largura da banda de erro — quanto mais "largo", menos precisa é a partição (queremos baixo).
- **Correlação (`correlation`)**: Mede o nível de sincronismo dos movimentos dos "picos e vales" das linhas de predição (queremos alto).

Para resolver desempates clássicos da Inferência Causal (ex: modelos muito correlatos porém com altos níveis absolutos de erro que alargam o intervalo de confiança do Geo-Teste), a função cria uma heurística penalizadora:

$$
\text{SER} = \frac{\sigma_e}{\rho + \epsilon}, \quad \epsilon = 10^{-6}
$$

O algoritmo divide a largura da banda de erro pela correlação, forçando o ranqueamento priorizar severamente modelos cujas curvas possuam o menor desvio padrão histórico, *desde que* sigam juntas. Por isso o "Cluster 0" garantidamente representará o Controle Sintético historicamente estático e metodologicamente mais seguro.

### 6.2 Por que não usar apenas o RMSPE?

O RMSPE, métrica padrão da literatura (Abadie et al., 2010), mede o erro absoluto de predição. Porém, em contextos corporativos com alta volatilidade:

- Um RMSPE baixo pode advir de um controle "plano" (flat) cuja média se aproxima da média do tratamento por acaso, sem rastrear dinâmicas sazonais.
- Esses **"Controles Zumbis"** passam no critério RMSPE clássico, mas colapsam fora da amostra quando choques exógenos atingem o tratamento e o sintético não reage.

O denominador $\rho$ penaliza esse cenário: um controle plano terá $\rho \approx 0$, inflando o SER para $+\infty$ e eliminando-o do ranqueamento.

### 6.3 Por que não usar apenas a correlação?

A correlação mede sincronia direcional, mas é invariante a escala. Dois sinais com $\rho = 0.99$ podem ter resíduos de magnitudes totalmente diferentes. Sem o $\sigma_e$ no numerador, o ranqueamento priorizaria modelos "sincronizados de longe" — correlatos, mas com bandas de erro largas que tornam o Geo-Teste estatisticamente fraco (intervalos de confiança amplos no pós-tratamento).

> [!NOTE]
> **Nota Acadêmica sobre a Heurística Operacional**
> Na estatística clássica (Abadie et al., 2010), a otimização de Controles Sintéticos visa exclusivamente a minimização do Erro Quadrático Médio de Predição (RMSPE) no período pré-tratamento. Contudo, em Causalidade Aplicada a negócios e marketing, a minimização absoluta do RMSPE sofre de um fenômeno conhecido como *supressão de variância*, onde a modelagem pode selecionar matrizes "fixas" ou "planas" que historicamente têm erro passivo baixo, mas não reagem de forma equivalente a choques sazonais exógenos.
>
> O **Synthetic Error Ratio** não se propõe a ser um novo teorema fundamental, mas sim atua como uma **Função de Perda Regularizadora** (Loss Function) puramente pragmática. Assumindo a Correlação como o "Sinal" (comportamento estrutural latente) e o Resíduo como o "Ruído" (variância estocástica inútil), a divisão matemática simula a maximização inversa do *Signal-to-Noise Ratio (SNR)*. Isso força empiricamente a exclusão dos famigerados "Controles Zumbis" e garante Robustez Fora da Amostra (Out-of-Sample) frente à instabilidade do mundo corporativo moderno.

### 6.4 Comportamento da otimização

A função **não otimiza o SER diretamente** via gradiente ou solver. O SER é uma métrica de avaliação *post-hoc*: cada configuração (combinação + hiperparâmetros) é avaliada independentemente e o SER é calculado ao final. O ranqueamento é feito por ordenação simples do vetor de SERs.

Formalmente, a solução retornada é:

$$
S^* = \arg\min_{S \in \mathcal{C}} \text{SER}(S)
$$

onde a minimização é feita por enumeração (exaustivo) ou por heurística gulosa (ranking).


---

## 7. Saídas (Outputs)

### 7.1 Tipo

`list[dict]` — Lista de dicionários, onde cada dicionário representa uma partição tratamento/controle avaliada.

### 7.2 Comportamento de retorno por modo

| Modo | Quantidade retornada | Label no output |
|:---|:---|:---|
| `exhaustive` | Top 5 (das $C(n,k)$ avaliadas) | `RECOMMENDATION 0..4` |
| `ranking` | $k$ clusters (um por geo do tratamento) | `TEST CLUSTER 0..k-1` |
| `fixed` | $|F|$ clusters (um por geo fixo) | `TEST CLUSTER 0..|F|-1` |

### 7.3 Schema do dicionário

```python
{
    "treatment":            list[str],    # Geos no grupo de tratamento
    "control":              list[str],    # Geos no donor pool final (pós-poda)
    "control_weights":      list[float],  # Pesos ∈ (0,1], Σ = 1.0
    "correlation":          float,        # ρ ∈ [-1, 1] (Pearson, espaço log-diff)
    "std_residual":         float,        # σ_e ≥ 0 (espaço log-diff)
    "rmspe":                float,        # RMSPE ≥ 0 (espaço log-diff)
    "synthetic_error_ratio": float,       # SER = σ_e / (ρ + 1e-6)
    "n_controls":           int,          # Nº de controles pré-poda CVXPY
    "alpha":                float,        # Melhor α do ElasticNet
    "l1_ratio":             float         # Melhor λ₁ do ElasticNet
}
```

### 7.4 Ordenação

A lista é sempre retornada em ordem **crescente** de SER (quando `use_elasticnet=True`) ou RMSPE (quando `use_elasticnet=False`). O índice `0` é sempre o melhor design experimental encontrado.


---

## 8. Interpretação dos Resultados

### 8.1 Leitura do RECOMMENDATION 0

Considere o seguinte resultado hipotético:

```
RECOMMENDATION 0
  treatment:            [curitiba, florianopolis, porto_alegre]
  control:              [belo_horizonte, goiania, brasilia, recife]
  control_weights:      [0.42, 0.31, 0.18, 0.09]
  correlation:          0.9213
  std_residual:         0.0387
  synthetic_error_ratio: 0.0420
```

**Interpretação linha a linha:**

- **`treatment`**: Essas 3 cidades devem receber a intervenção (campanha, holdout, etc.). A série-alvo do modelo é a média aritmética diária dessas 3 cidades.
- **`control`**: Essas 4 cidades compõem o Controle Sintético. Elas **não devem receber tratamento** durante o experimento.
- **`control_weights`**: Belo Horizonte contribui com 42% do sintético, Goiânia com 31%, Brasília com 18% e Recife com 9%. Isso significa que o contrafactual será: $\hat{Y}_t = 0.42 \cdot \text{BH}_t + 0.31 \cdot \text{GO}_t + 0.18 \cdot \text{BSB}_t + 0.09 \cdot \text{REC}_t$.
- **`correlation = 0.9213`**: No período pré-intervenção, as variações diárias (log-diff) do tratamento e do sintético se movem na mesma direção em 92% dos dias. Esse nível de sincronia é alto e indicativo de um bom ajuste estrutural.
- **`std_residual = 0.0387`**: A dispersão do erro diário (em escala log-diff) é de ~3.9%. Isso implica que, na ausência de tratamento, espera-se que a previsão do sintético oscile ±3.9% ao redor do real em um dia típico.
- **`synthetic_error_ratio = 0.0420`**: Métrica consolidada. Valores abaixo de 0.1 são geralmente indicativos de ajuste excelente. Valores acima de 0.5 sugerem partição de baixa qualidade.

### 8.2 Quando se preocupar

| Sinal | Diagnóstico | Ação |
|:---|:---|:---|
| SER > 0.3 | Ajuste fraco — sintético não replica tratamento | Aumentar o período histórico, remover geos ruidosas, ou reduzir `n_treatment` |
| $\rho < 0.7$ | Dessincronização estrutural | Investigar heterogeneidade geográfica; considerar segregar por cluster regional |
| $\sigma_e > 0.1$ | Resíduo largo — intervalo de confiança será amplo | Aceitar menor poder estatístico ou buscar geos com melhor co-movimento |
| Peso > 0.8 em um único controle | Dependência excessiva | Risco de que choque idiossincrático nesse controle invalide todo o sintético |


---

## 9. Limitações

### 9.1 Limitações matemáticas

1. **Otimalidade local no grid:** A busca de hiperparâmetros ElasticNet usa um grid discreto de 9 pontos. A combinação ótima $(\alpha^*, \lambda_1^*)$ pode residir fora do grid, especialmente para dados com estrutura de regularização atípica.

2. **Dualidade da avaliação:** O ElasticNet é ajustado no espaço log-diff (estacionário), mas os pesos CVXPY são otimizados no espaço original (nível). Essa mudança de espaço entre os estágios 5.3 e 5.4 pode introduzir inconsistências: um controle considerado relevante em retornos logarítmicos pode ter ajuste fraco em nível, e vice-versa.

3. **Ausência de validação cruzada temporal:** A avaliação usa todo o período pré-intervenção tanto para ajuste quanto para avaliação (in-sample). Não há split treino/validação temporal, o que pode levar a *overfitting* em períodos curtos.

### 9.2 Limitações computacionais

1. **Explosão combinatória:** $C(50, 10) \approx 10^{10}$. Para bases com $n > 30$ e $k > 5$, o modo exaustivo é computacionalmente inviável. O modo `ranking` é uma solução pragmática, mas sacrifica a garantia de otimalidade global.

2. **Custo por iteração:** Cada avaliação envolve 9 fits ElasticNet + 1 otimização CVXPY (solver SCS). Para $C(n,k) = 80.730$, isso resulta em aproximadamente 726.570 ajustes de modelo.

### 9.3 Limitações estatísticas

1. **Sem correção para múltiplas comparações:** O ranqueamento de $C(n,k)$ combinações pelo SER seleciona o melhor ajuste *ex-post*. Em bases com muitas geos e poucas observações, o "melhor" pode ser um artefato de seleção (análogo ao *p-hacking* em testes de hipóteses).

2. **Pressuposto de estabilidade violável:** A função não testa formalmente se os padrões de co-movimento são estáveis ao longo do tempo. Uma correlação $\rho = 0.95$ computada sobre 90 dias pode mascarar uma quebra estrutural aos 60 dias que compromete a validade futura.

3. **SER como heurística:** O SER não possui fundamento axiomático (não é derivado de um modelo probabilístico). Ele é motivado pela analogia Signal-to-Noise, mas não há prova formal de que minimizar o SER minimiza o viés da estimativa do efeito causal.

### 9.4 Limitações práticas

1. **Valores zero:** Séries com zeros frequentes (dias sem dado, feriados) geram `log(0) = -∞`, corrompendo toda a pipeline. Requerem tratamento prévio (drop, imputação, ou transformação $\log(x + 1)$ — esta última **não** é implementada internamente).

2. **Geos com séries muito curtas:** Se `start_date`/`end_date` limitam o horizonte a < 15 dias, a diferenciação + grid search pode operar com $T < 14$ observações, insuficientes para estabilidade numérica do ElasticNet.


---

## 10. Casos de Uso

### 10.1 Quando usar

| Cenário | Configuração recomendada |
|:---|:---|
| Desenho de Geo-Teste antes de campanha digital | `n_treatment=3..5`, `search_mode="exhaustive"` (se viável) |
| Validação de holdout pré-definido pela equipe | `fixed_treatment=["geo1", "geo2"]` |
| Prototipagem rápida com muitas geos (>30) | `search_mode="ranking"` |
| Análise de sensibilidade a seleção de tratamento | Comparar `clusters[0]` a `clusters[4]` no modo exaustivo |
| Experimentos sem regularização (DiD puro) | `use_elasticnet=False` |

### 10.2 Quando NÃO usar

| Cenário | Motivo | Alternativa |
|:---|:---|:---|
| Dados com muitos zeros (ex: vendas binárias) | Transformação $\log$ falha | Pré-processar com $\log(x+1)$ ou usar DiD clássico |
| Geo singular (n=1 tratamento, n=1 controle) | Sem grau de liberdade para otimizar | Usar Bayesian Structural Time Series (CausalImpact) |
| Séries com < 15 observações pré-tratamento | Instabilidade numérica do ElasticNet e CVXPY | Coletar mais dados ou usar método não-paramétrico |
| Tratamento com spillover geográfico forte | Violação de SUTVA | Usar buffers geográficos ou randomização por cluster |
| Efeitos de tratamento esperados em nível (não em crescimento) | A avaliação opera em log-diff; efeitos de nível constante podem ser diluídos | Considerar modelo em nível com covariáveis |


---

## 11. Exemplo Simples

### 11.1 Setup

Considere um CSV (`vendas.csv`) com 60 dias de vendas diárias de 6 cidades:

| data | SP | RJ | BH | CWB | POA | REC |
|:---|---:|---:|---:|---:|---:|---:|
| 2026-01-01 | 1000 | 800 | 400 | 300 | 250 | 200 |
| 2026-01-02 | 1020 | 810 | 410 | 305 | 255 | 198 |
| ... | ... | ... | ... | ... | ... | ... |
| 2026-03-01 | 1150 | 870 | 430 | 320 | 270 | 210 |

### 11.2 Chamada

```python
from reallift.geo.discovery import discover_geo_clusters

# Exemplo Exploratório: Busca exaustiva do melhor design com 2 tratamentos.
clusters = discover_geo_clusters(
    filepath="vendas.csv",
    date_col="data",
    n_treatment=2,
    search_mode="exhaustive",
    verbose=True
)
```

### 11.3 O que acontece internamente

1. **Combinações geradas:** $C(6, 2) = 15$

   ```
   (SP,RJ), (SP,BH), (SP,CWB), (SP,POA), (SP,REC),
   (RJ,BH), (RJ,CWB), (RJ,POA), (RJ,REC),
   (BH,CWB), (BH,POA), (BH,REC),
   (CWB,POA), (CWB,REC),
   (POA,REC)
   ```

2. **Para cada combinação** (ex: `(SP, RJ)`):
   - Tratamento: $y_t = \frac{\text{SP}_t + \text{RJ}_t}{2}$
   - Donor pool: `[BH, CWB, POA, REC]`
   - Aplica `log_diff_transform` → 59 observações de retornos diários
   - Roda 9 configs ElasticNet → filtra controles → CVXPY otimiza pesos
   - Calcula métricas → retém o melhor dos 9 configs

3. **Total de avaliações:** $15 \times 9 = 135$ fits de modelo

4. **Ordenação:** As 15 combinações são ordenadas por SER

### 11.4 Saída (hipotética)

```python
clusters[0]
# {
#     "treatment": ["CWB", "POA"],
#     "control": ["BH", "REC"],
#     "control_weights": [0.72, 0.28],
#     "correlation": 0.9450,
#     "std_residual": 0.0310,
#     "synthetic_error_ratio": 0.0328,
#     "n_controls": 2,
#     "alpha": 0.01,
#     "l1_ratio": 0.5
# }
```

### 11.5 Interpretação

O algoritmo identificou que Curitiba + Porto Alegre, quando tratadas simultaneamente, são melhor replicadas por um sintético composto de 72% Belo Horizonte + 28% Recife. Esse foi o melhor design dentre os 15 possíveis, com SER = 0.0328.

Se o praticante não puder usar Curitiba no tratamento (restrição operacional), pode adotar `clusters[1]` com confiança — essa alternativa foi avaliada de forma independente sobre o espaço completo, não como uma variação marginal da primeira.

### 11.6 Exemplo com Tratamento Fixo

```python
# Exemplo Direcionado: Forçando agrupamentos de controle para unidades de Campanha Ativas.
melhores_clusters = discover_geo_clusters(
    filepath="base_historica_vendas.csv",
    date_col="data_faturamento",
    fixed_treatment=["sao_paulo", "rio_de_janeiro"],
    start_date="2026-01-01"
)

for sub_testes in melhores_clusters:
     print(f"Alvo: {sub_testes['treatment'][0]} => Explicado por: {sub_testes['control']}")
```

---

## Referências

- Abadie, A., & Gardeazabal, J. (2003). *The Economic Costs of Conflict: A Case Study of the Basque Country.* American Economic Review, 93(1), 113-132.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). *Synthetic Control Methods for Comparative Case Studies.* Journal of the American Statistical Association, 105(490), 493-505.
- Zou, H., & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net.* Journal of the Royal Statistical Society: Series B, 67(2), 301-320.
- O'Donoghue, B., Chu, E., Parikh, N., & Boyd, S. (2016). *Conic Optimization via Operator Splitting and Homogeneous Self-Dual Embedding.* Journal of Optimization Theory and Applications, 169(3), 1042-1068.
