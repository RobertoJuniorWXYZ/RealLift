# `reallift.simulation.generate_geo_data`

Gera bases de dados sintéticas para testes geográficos, permitindo simular cenários complexos de tendência, sazonalidade e ruído. É a ferramenta principal para validar a sensibilidade do modelo (MDE) antes de rodar experimentos reais.

## Assinatura

```python
def generate_geo_data(
    start_date="2022-01-01",
    end_date="2022-06-30",
    n_geos=5,
    freq="D",
    trend_slope=0.05,
    seasonality_amplitude=10,
    seasonality_period=7,
    noise_std=2,
    treatment_geos=None,
    treatment_start=None,
    lift=0.2,
    random_seed=42,
    plot=True,
    save_csv=False,
    file_name="synthetic_geolift.csv",
    pre_file_name="synthetic_geolift_pre.csv",
    base_value=50.0,
    as_integer=False,
    pre_only=False
) -> tuple
```

## Parâmetros Principais

- **`n_geos`**: Quantidade de unidades geográficas a serem geradas.
- **`trend_slope`**: Inclinação da tendência linear (crescimento orgânico).
- **`seasonality_amplitude`**: Magnitude da oscilação senoidal (ex: ciclos semanais).
- **`noise_std`**: Desvio padrão do ruído branco inserido na série. Pode ser um `float` ou uma lista `[min, max]` para aleatoriedade por geo.
- **`lift`**: O efeito incremental a ser injetado nos geos de tratamento. Pode ser um valor fixo (0.05 = 5%) ou um intervalo para sorteio.
- **`pre_only`**: (Novo) Se `True`, a função gera e retorna apenas o período "pré-teste", ignorando a injeção de lift e datas futuras. Essencial para fluxos de **Design of Experiments (DoE)**.

## Retorno

Retorna uma tupla `(df_full, df_pre, treatment_geos)`:
1. `df_full`: DataFrame completo com todo o período gerado.
2. `df_pre`: DataFrame filtrado apenas com o período anterior ao `treatment_start`.
3. `treatment_geos`: Lista das geos que foram (ou seriam) sorteadas para o tratamento.

## Exemplo de Uso

```python
from reallift.simulation import generate_geo_data

# Gerando dados pré-teste para um DoE rápido
df, df_pre, treated = generate_geo_data(
    n_geos=30,
    noise_std=[2, 8],
    pre_only=True,
    save_csv=True,
    pre_file_name="meu_historico.csv"
)
```
