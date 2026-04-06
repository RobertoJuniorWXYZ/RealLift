# `reallift.simulation.generate_simulated_intervention`

Extrapola uma base de dados pré-existente (CSV) para criar um período de intervenção sintético, injetando um lift linearmente acrescido ao comportamento projetado. É a ferramenta ideal para testar o **Poder Estatístico** de um cenário real de DoE antes de colocar o investimento em campo.

## Assinatura

```python
def generate_simulated_intervention(
    filepath,
    days,
    treatment_geos,
    lift=0.05,
    date_col="date",
    trend_slope=0.05,
    seasonality_amplitude=10,
    seasonality_period=7,
    noise_std=2,
    random_seed=42,
    plot=True,
    save_csv=False,
    file_name="simulated_intervention.csv",
    as_integer=False
) -> pd.DataFrame
```

## Funcionamento (Grounding)

Diferente de `generate_geo_data`, que cria tudo do zero, esta função **ancora** a simulação no último ponto observado de cada geo no seu CSV original. Ela calcula um `offset` para garantir que a transição entre o mundo real e o mundo simulado seja contínua em termos de nível, tendência e sazonalidade.

## Parâmetros Principais

- **`filepath`**: Caminho para o CSV com os dados históricos (pré-teste).
- **`days`**: Duração do cenário de intervenção a ser simulado (ex: 21 dias).
- **`treatment_geos`**: Lista de geos que receberão o impacto do lift.
- **`lift`**: O efeito incremental relativo (ex: 0.10 = 10% de aumento sobre a tendência).
- **`trend_slope` / `seasonality_amplitude`**: Devem ser estimados ou aproximados do comportamento real dos seus dados para que a simulação seja verossímil.

## Retorno

Retorna um `pd.DataFrame` contendo a união dos dados originais (pré-teste) com os dados novos (intervenção simulada).

## Exemplo de Fluxo

```python
from reallift.simulation import generate_simulated_intervention

# Criando um cenário de teste para geo_17 e geo_0 com 5% de lift
df_sim = generate_simulated_intervention(
    filepath="meu_historico.csv",
    days=21,
    treatment_geos=["geo_17", "geo_0"],
    lift=0.05,
    save_csv=True,
    file_name="teste_ab_geolift.csv"
)
```
