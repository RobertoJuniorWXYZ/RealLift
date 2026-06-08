# RealLift — Roadmap

## Feature 1: Secondary KPIs

**Objetivo:** Estimar lift também em métricas secundárias (ex: novos usuários, ticket médio, leads) usando os mesmos pesos sintéticos aprendidos sobre a métrica primária.

**Lógica central:**
Os pesos w* são otimizados na métrica primária. Para cada KPI secundária, aplica-se os mesmos pesos:

```
ŷ_s^(0) = Σ wᵢ* · y_s_i
Δy_s = ȳ_s_treated − ŷ_s^(0)
```

Os mesmos blocos MBB e permutações do placebo são reutilizados — sem nova otimização.

**Requisito:** as KPIs secundárias devem ter a mesma segmentação geográfica e granularidade temporal da métrica primária.

**API sugerida (run):**

```python
results = rl_post.run(
    treatment_start_date="2025-04-01",
    treatment_end_date="2025-04-28",
    doe=doe,
    scenario=0,
    secondary_kpis=["new_users", "avg_ticket", "leads"],  # colunas no DataFrame
)
```

**Output:**
- `results.secondary_effects` — DataFrame com lift absoluto, lift %, CI e p-valor por KPI
- `results.plot_secondary_effects()` — visualização comparativa entre KPIs

**Decisões em aberto:**
- Normalizar as KPIs antes de aplicar os pesos? (Relevante quando as escalas são muito diferentes.)
- Exibir `avg_ticket` como lift na média ou lift no total? (Ambos têm interpretações válidas.)

---

## Feature 2: Custo / Investimento → ROAS e iROAS

**Objetivo:** Receber dados de investimento por geo e calcular eficiência incremental da campanha.

**Métricas:**

| Métrica | Fórmula | Interpretação |
|---|---|---|
| **ROAS** | Receita Incremental Total / Investimento Total | Retorno bruto sobre gasto |
| **iROAS** | Receita Incremental / Investimento Incremental | Retorno sobre o *delta* de investimento vs. baseline |
| **CPL / CPA** | Investimento Total / Leads ou Conversões Incrementais | Custo por resultado incremental (com Feature 1) |

**Requisito:** investimento com a mesma segmentação geográfica da métrica primária.

**API sugerida:**

```python
results = rl_post.run(
    treatment_start_date="2025-04-01",
    treatment_end_date="2025-04-28",
    doe=doe,
    scenario=0,
    investment_col="spend",   # coluna no DataFrame com investimento diário por geo
)
```

**Output:**
- `results.roas` — float
- `results.iroas` — float
- `results.iroas_ci` — intervalo de confiança via MBB (bootstrap no lift propaga para o iROAS)
- `results.plot_iroas_distribution()` — distribuição bootstrap do iROAS

### Extensão opcional: Adstock

**Objetivo:** Modelar o efeito de carryover do investimento na resposta de lift — útil quando a campanha tem efeito residual após o período de tratamento.

**Modelo:**
```
Adstock_t = spend_t + λ · Adstock_{t-1}
Lift_t ~ α + β · Adstock_t
```

`λ` é o parâmetro de decaimento (0 = sem carryover, 1 = acumulação total). Pode ser fixado pelo usuário ou estimado por grid search sobre o R² do modelo.

**Ativação sugerida:**

```python
results = rl_post.run(
    ...
    investment_col="spend",
    adstock={"decay": 0.5},        # λ fixo
    # adstock={"decay": "auto"},   # estima λ por grid search
)
```

**Quando faz sentido usar:** campanhas de TV, OOH, ou qualquer mídia com efeito de memória. Não recomendado para canais de resposta direta (paid search, performance).

---

## Feature 3: Pré-Clusterização Regional de Geos

**Objetivo:** Permitir que um único `GeoExperiment` execute múltiplos sub-experimentos regionais independentes, garantindo que tratamento e donor pool de cada região não se misturem.

**O problema que resolve:** em um experimento nacional, um geo de São Paulo pode ter dinâmicas de mercado completamente diferentes de um geo gaúcho. Se o sintético de um geo paulista for construído com geos do Sul como donors, os pesos w* estão sendo otimizados sobre um contrafactual implausível — violando a premissa central do SCM. A pré-clusterização resolve isso restringindo cada sub-experimento ao seu próprio pool regional.

**Como funciona:** o usuário fornece uma coluna de região no DataFrame. O pipeline roda de forma independente para cada região:

1. SER e seleção de geos restrita aos geos da região
2. Otimização convexa usando apenas o donor pool regional
3. Inferência (MBB + placebo) dentro da região
4. Resultados por região + agregação ponderada nacional

**Requisito:** cada região precisa ter geos suficientes para formar um pool de controle válido. Regiões com poucos geos podem ser agrupadas manualmente antes de passar ao pipeline.

**API sugerida:**

```python
rl = RealLift.GeoExperiment(
    "geo_daily_sales.csv",
    date_col="date",
    region_col="region",   # coluna com rótulo da região por geo
)

doe = rl.design(
    pct_treatment=[0.10],
    experiment_days=[28],
    # design roda independentemente por região
)

results = rl_post.run(
    treatment_start_date="2025-04-01",
    treatment_end_date="2025-04-28",
    doe=doe,
    scenario=0,
)
```

**Output:**
- `results.regional_effects` — DataFrame com lift, CI e p-valor por região
- `results.plot_regional_effects()` — forest plot comparando regiões
- `results.aggregate_effect` — lift nacional agregado, ponderado por receita de cada região

**Implicações:**
- **Validade interna mais alta**: o contrafactual de cada geo é construído com peers genuinamente comparáveis.
- **Heterogeneidade regional visível**: conecta diretamente com S4 (HTE) — a variação de lift entre regiões é informação estratégica (ex: Sul responde melhor que Nordeste).
- **Risco de poder**: regiões menores terão menos geos disponíveis para o pool, reduzindo a qualidade do sintético. Um alerta automático quando o pool regional tiver menos de N geos seria útil.
- **Agregação nacional não trivial**: combinar CIs de sub-experimentos independentes requer cuidado — a soma dos lifts regionais ponderados por receita é estimável, mas a incerteza conjunta precisa ser propagada corretamente (ex: via simulação conjunta dos bootstraps).

**Esforço:** médio-alto. A lógica de inferência já existe; o trabalho está em paralelizar o pipeline por região, garantir que a seleção de tratamento/controle respeite os limites regionais, e implementar a agregação com propagação de incerteza.

---

## Feature 3B: Pré-Clusterização por Escala de Mercado

**Objetivo:** Agrupar geos por volume de vendas antes do design do experimento, garantindo que tratamento e donor pool sejam compostos por praças de porte comparável.

**Distinção da Feature 3:** a Feature 3 separa geos por região geográfica (Sul, Sudeste, etc.). Esta feature separa por escala de mercado — uma SP não entra no mesmo pool que uma cidade pequena do interior, independente de região.

**Motivação de negócio:** o cliente que trata geos de grande volume não sabe como se comportam suas pequenas praças, e vice-versa. Misturar escalas no mesmo experimento pode produzir um lift agregado que não representa nenhum dos dois grupos.

**Pré-requisito crítico: qualidade dos dados.** Esta feature só faz sentido se os dados passaram pelo `clean()` com qualidade suficiente. A média de um geo com muitos zeros, imputações excessivas ou séries inconsistentes é uma referência não confiável para clusterização — agrupar por uma escala espúria é pior do que não agrupar. Geos com qualidade abaixo de um limiar mínimo devem ser excluídos antes da clusterização por escala.

**Como funciona:**
1. Após o `clean()`, calcular a média histórica de cada geo
2. Agrupar por quantis de volume (ex: quartis → 4 pools de escala)
3. Design e inferência rodam independentemente dentro de cada pool
4. Resultados por pool de escala + agregação ponderada

**API sugerida:**

```python
doe = rl.design(
    pct_treatment=[0.10],
    experiment_days=[28],
    scale_clusters=4,        # número de grupos por volume (quartis)
    min_quality_score=0.7,   # excluir geos abaixo desse score do clean()
)
```

**Output:**
- `results.scale_cluster_effects` — lift por faixa de volume (pequenas, médias, grandes praças)
- `results.plot_scale_cluster_effects()` — comparativo entre faixas

**Implicações:**
- Combinável com Feature 3 (regional): pode-se ter clusters por região *e* por escala simultaneamente, embora isso exija volume de dados suficiente em cada célula.
- Sem qualidade de dados garantida, a clusterização por escala é ruído — o pré-requisito do `clean()` não é opcional.

**Esforço:** médio. A lógica de clusterização por quantil é simples; o trabalho está na integração com o score de qualidade do `clean()` e na garantia de pools mínimos viáveis por faixa.

---

## Feature 4 (sugestão): Janela de Efeito Pós-Campanha

**Objetivo:** Medir por quanto tempo o lift persiste após o fim do período de tratamento — relevante para calcular o valor total da campanha além da janela experimental.

**Lógica:** estender o período de análise N dias além de `treatment_end_date` e plotar o decay do lift diário. Nenhuma nova otimização — apenas aplicar o contrafactual já estimado ao período pós.

**API sugerida:**

```python
results.plot_post_treatment_decay(days=14)
```

---

---

## Sugestões Adicionais

### S1: Decomposição Temporal do Lift

**O que é:** em vez de reportar apenas o lift agregado do período, calcular o lift diário ao longo de todo o período de tratamento.

**Como funciona:** o contrafactual ŷ^(0) já é gerado dia a dia. Basta não agregar — expor `Δy_t = y_t − ŷ_t^(0)` como série temporal.

**Implicações:**
- Revela **ramp-up**: campanhas de mídia de massa frequentemente levam dias para atingir efeito pleno.
- Revela **wear-out**: o lift pode decair na segunda metade do período, indicando saturação precoce.
- Identifica **picos pontuais** (ex: um dia de promoção) que inflam ou deflam o agregado.
- Conecta diretamente com a Feature 3 (Janela Pós-Campanha) — o decay pós-tratamento é a continuação natural dessa série.

**Output sugerido:**
- `results.daily_lift` — Series com lift diário
- `results.plot_daily_lift()` — gráfico de linha com CI diário via MBB

**Esforço:** baixo. O dado já existe no pipeline — é só uma questão de exposição.

---

### S2: Análise de Contribuição do Pool de Controle

**O que é:** ranking dos pesos w* por geo de controle, mostrando quais geos sustentam o sintético e em que proporção.

**Como funciona:** os pesos já são calculados na otimização convexa. Basta expô-los ordenados, com visualização.

**Implicações:**
- **Auditabilidade**: permite que o analista valide se os geos de controle fazem sentido de negócio (ex: um geo de controle com w* = 0.6 deve ser parecido com o tratado).
- **Fragilidade do design**: se um único geo concentra >50% do peso, o sintético é frágil — qualquer choque idiossincrático naquele geo contamina o contrafactual. Isso pode virar um alerta automático.
- **Iteração no design**: serve de feedback para o `design()` — o analista pode excluir manualmente geos com pesos muito concentrados e re-rodar.

**Output sugerido:**
- `results.control_weights` — DataFrame com geo, peso w*, e % de contribuição
- `results.plot_control_weights()` — gráfico de barras ranqueado

**Esforço:** baixo. Os pesos já existem no objeto interno.

---

### S3: Checagem Retrospectiva de Poder

**O que é:** após o experimento, comparar o lift observado com a curva MDE projetada no `design()` — respondendo "o experimento estava adequadamente dimensionado para o efeito que encontramos?"

**Como funciona:** o `design()` projeta o MDE mínimo detectável dado o tamanho e duração do experimento. O `run()` produz o lift observado e seu CI. A checagem cruza os dois.

**Implicações:**
- Se o lift observado está **abaixo do MDE projetado** e ainda assim é significativo, o design foi conservador — pode-se usar experimentos menores no futuro.
- Se o lift observado está **próximo do MDE projetado** e o resultado não é significativo, o experimento estava no limite de poder — a conclusão correta é "inconclusivo", não "sem efeito".
- Evita o erro comum de interpretar um resultado não significativo como prova de ausência de lift, quando na verdade o experimento nunca teve poder suficiente para detectá-lo.

**Output sugerido:**
- `results.power_check` — dict com `observed_lift`, `mde_projected`, `was_adequately_powered` (bool)
- Alerta automático no `results.summary()` quando `observed_lift < mde_projected`

**Esforço:** baixo-médio. Requer que `run()` receba o `doe` com as projeções de poder (já é o caso na API atual).

---

### S4: Heterogeneidade de Efeito por Geo Tratado (HTE)

**O que é:** quando há múltiplos geos no grupo de tratamento, estimar o lift individualmente para cada geo em vez de apenas o agregado.

**Como funciona:** para cada geo tratado, o pipeline já constrói um sintético individual (via `plot_cluster_effects`). O que falta é consolidar esses efeitos individuais em uma análise comparativa com CIs.

**Implicações:**
- **Targeting**: identifica quais geos respondem melhor à intervenção — insumo direto para decidir onde concentrar investimento nas próximas campanhas.
- **Segmentação estratégica**: geos de alta resposta podem ter características em comum (tamanho, densidade, perfil demográfico) que revelam o público mais receptivo.
- **Robustez**: se o efeito agregado é positivo mas metade dos geos tem lift negativo, o resultado agregado pode ser espúrio ou mascarar heterogeneidade real.
- **Interação com Feature 2**: o iROAS por geo revela onde o investimento é mais eficiente — não apenas se a campanha funcionou, mas *onde* funcionou mais.

**Output sugerido:**
- `results.geo_effects` — DataFrame com lift absoluto, lift %, CI e p-valor por geo tratado
- `results.plot_geo_effects()` — forest plot com efeito e CI por geo

**Esforço:** médio. A infraestrutura de inferência por geo já existe; o esforço está em agregar, testar e expor consistentemente.

---

### S5: Score de Qualidade do Design

**O que é:** um índice único (0–100) que resume a confiança no design experimental, combinando múltiplas dimensões de qualidade em um número comunicável para stakeholders não técnicos.

**Componentes sugeridos:**

| Componente | Peso sugerido | Fonte |
|---|---|---|
| Qualidade do fit pré-período (R²) | 30% | Resíduos do SCM no pré-período |
| Ghost Lift p-valor | 25% | `check_ghost_lift` do `design()` |
| Distribuição de SER do pool | 20% | Qualidade dos geos de controle selecionados |
| Poder estatístico projetado | 25% | MDE vs. lift esperado declarado pelo usuário |

**Implicações:**
- **Comunicação**: um score único é mais acessível do que quatro métricas separadas para quem aprova orçamento ou interpreta resultados.
- **Comparabilidade**: permite comparar designs de experimentos diferentes ("o design de abril tinha score 82, o de junho teve 91").
- **Gatekeeping**: pode ser usado como critério mínimo — ex: não rodar experimento com score < 60.
- **Risco**: pesos fixos podem ser enganosos em casos extremos (ex: R² alto mas Ghost Lift alto). Deve sempre ser acompanhado das métricas individuais.

**Output sugerido:**
- `doe.design_score` — float (0–100)
- `doe.design_score_breakdown` — dict com contribuição de cada componente
- Exibido automaticamente no PDF do DoE Report

**Esforço:** médio. Os inputs já existem; o trabalho está em calibrar os pesos e definir as transformações de normalização.

---

### S6: Curva de Resposta ao Investimento

**O que é:** usando múltiplos experimentos históricos em diferentes níveis de spend, estimar a curva de resposta — onde o iROAS começa a cair e onde a campanha entra em saturação.

**Como funciona:** cada experimento produz um par (investimento_total, lift_incremental). Com 3+ pontos, é possível ajustar uma curva de resposta (ex: função de potência ou logarítmica) e estimar o ponto de máximo iROAS e o ponto de saturação.

**Implicações:**
- **Planejamento de budget**: responde "quanto devo investir para maximizar o retorno incremental?" com base em evidência empírica, não em benchmark de mercado.
- **Argumento para escala**: se a curva mostra iROAS ainda alto no nível atual de investimento, há justificativa clara para aumentar budget.
- **Argumento para corte**: se a curva mostra saturação, há justificativa para realocar investimento para outros canais ou geos.
- **Requisito crítico**: exige pelo menos 2–3 experimentos históricos com níveis de investimento diferentes. Com apenas um experimento, só é possível estimar o iROAS pontual — a curva requer variação.

**Output sugerido:**
- `RealLift.ResponseCurve([results_1, results_2, results_3])` — nova classe que agrega múltiplos `ExperimentResult`
- `curve.plot_response_curve()` — curva ajustada com pontos observados e região de incerteza
- `curve.optimal_spend` — ponto de máximo iROAS estimado

**Esforço:** alto. Requer nova classe, fitting de curva, e acumulação de experimentos históricos. Faz mais sentido como feature de médio prazo, após acúmulo de 2–3 experimentos com Feature 2 implementada.

---

## Notas Gerais

- Todas as features acima são extensões do método de inferência existente — não alteram a otimização dos pesos nem o design de experimento.
- Features 1 e 2 são independentes e podem ser desenvolvidas em paralelo.
- A extensão Adstock de Feature 2 deve ser implementada por último, pois depende dos outputs de ROAS/iROAS estar estáveis.
