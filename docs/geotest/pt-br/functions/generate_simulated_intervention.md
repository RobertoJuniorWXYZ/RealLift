# `reallift.simulation.generate_simulated_intervention`

Extrapola uma base de dados pré-existente (CSV) para criar um período de intervenção sintético, injetando um lift sobre o comportamento projetado. É a ferramenta ideal para validar o **Poder Estatístico** de um cenário de DoE antes de colocar o investimento em campo.

## Assinatura

```python
def generate_simulated_intervention(
    filepath,
    treatment_geos,
    days=None,
    start_date=None,
    end_date=None,
    lift=0.05,
    date_col="date",
    noise_std=None,
    random_seed=42,
    plot=True,
    save_csv=False,
    file_name="simulated_intervention.csv",
    as_integer=False
) -> pd.DataFrame
```

---

## Funcionamento (Weekday-Mean Forecast)

Para cada geografia, a função constrói o período pós-teste com base na **média histórica por dia da semana** (segunda, terça, ..., domingo). Isso preserva naturalmente a sazonalidade semanal (comportamento de varejo, fins de semana, etc.) sem nenhuma modelagem paramétrica complexa.

**Etapas por geo:**
1. Calcula a média de todos os registros do pré-teste para cada dia da semana (0=Seg … 6=Dom)
2. Projeta os `days` dias futuros usando a média do respectivo dia da semana como baseline
3. Adiciona ruído gaussiano estimado dos resíduos do pré-teste (`std` por weekday)
4. Aplica o floor em zero (volumes não podem ser negativos)
5. Multiplica pelo fator `(1 + lift)` nos geos tratados

---

## Parâmetros

### Dados de Entrada

| Parâmetro | Tipo | Descrição |
|:---|:---|:---|
| `filepath` | `str` | Caminho para o CSV com os dados históricos (pré-teste). |
| `date_col` | `str` | Nome da coluna de datas no CSV. Default: `"date"`. |
| `treatment_geos` | `list` | Lista de geos que receberão o impacto do lift. |

### Definição do Período Pós-Teste

Use **uma** das duas formas abaixo:

| Parâmetro | Tipo | Descrição |
|:---|:---|:---|
| `days` | `int` | Número de dias a simular a partir do último dia do pré-teste. |
| `start_date` + `end_date` | `str` | Datas explícitas do período pós-teste (formato `'YYYY-MM-DD'`, ambas inclusivas). O `days` é calculado automaticamente. |

> [!IMPORTANT]
> É obrigatório fornecer `days` **ou** o par `start_date` + `end_date`. Se nenhum for fornecido, a função lançará `ValueError`.

### Controle do Efeito

| Parâmetro | Tipo | Default | Descrição |
|:---|:---|:---|:---|
| `lift` | `float` ou `[min, max]` | `0.05` | Efeito incremental relativo (ex: `0.14` = +14%). Se for uma lista `[min, max]`, sorteia um valor dentro do intervalo para cada geo tratada. |
| `noise_std` | `float` ou `[min, max]` | `None` | Desvio padrão do ruído aditivo. Se `None`, estimado automaticamente dos resíduos do pré-teste. |

### Saída

| Parâmetro | Tipo | Default | Descrição |
|:---|:---|:---|:---|
| `random_seed` | `int` | `42` | Semente para reprodutibilidade do ruído. |
| `as_integer` | `bool` | `False` | Se `True`, arredonda todos os valores para inteiros. |
| `plot` | `bool` | `True` | Exibe o gráfico com o período completo (pré + pós-teste). |
| `save_csv` | `bool` | `False` | Salva o DataFrame completo em CSV. |
| `file_name` | `str` | `"simulated_intervention.csv"` | Nome do arquivo de saída. |

---

## Retorno

Retorna um `pd.DataFrame` contendo a **união** dos dados originais (pré-teste) com os dados simulados (intervenção), mantendo as mesmas colunas e a coluna de data.

---

## Exemplos de Uso

### Forma 1 — por número de dias

```python
from reallift import generate_simulated_intervention

df_sim = generate_simulated_intervention(
    filepath="cleaned_geo_data.csv",
    treatment_geos=["sao_paulo", "rio_de_janeiro", "campinas"],
    days=28,
    lift=0.14,
    date_col="dia",
    random_seed=42,
    save_csv=True,
    as_integer=True,
    file_name="lift_simulation.csv"
)
```

### Forma 2 — por datas explícitas

```python
from reallift import generate_simulated_intervention

df_sim = generate_simulated_intervention(
    filepath="cleaned_geo_data.csv",
    treatment_geos=["sao_paulo", "rio_de_janeiro", "campinas"],
    start_date="2026-01-01",
    end_date="2026-04-10",   # 100 dias, inclusive
    lift=0.14,
    date_col="dia",
    random_seed=42,
    save_csv=True,
    as_integer=True,
    file_name="lift_simulation.csv"
)
```

---

## Relação com o DoE

O fluxo padrão é usar `generate_simulated_intervention` logo após o `design_of_experiments`, alimentando os geos selecionados pelo melhor cenário diretamente:

```python
# Melhor cenário do DoE
best_scenario = doe["scenarios"][0]

df_sim = generate_simulated_intervention(
    filepath="cleaned_geo_data.csv",
    treatment_geos=best_scenario["treatment_pool"],
    days=28,
    lift=0.14,
    date_col="dia",
    as_integer=True,
    save_csv=True,
    file_name="lift_simulation.csv"
)
```
