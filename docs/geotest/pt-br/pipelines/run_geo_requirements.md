# `reallift.pipelines.geo_pipeline.run_geo_requirements`

A função `run_geo_requirements` atua como a interface mestre de **Design de Experimentos (DoE)**. Seu uso é mandatório **antes** de você iniciar qualquer intervenção ou gasto nas regiões afetadas. Ela projeta matematicamente as premissas de custo, tempo e agrupamento para que o teste geográfico tenha sucesso no futuro.

## Assinatura

```python
def run_geo_requirements(
    filepath: str,
    date_col: str,
    treatment_start_date: str = None,
    n_treatment: int = 1,
    fixed_treatment: list = None,
    n_folds: int = 5,
    mde: float = 0.015,
    max_days: list = [21, 60],
    verbose: bool = True
) -> dict
```

## Arquitetura da Pipeline (Design Level)

Para embasar a viabilidade do teste (Decisão Go/No-Go), a pipeline processa apenas metadados pré-intervenção orquestrando as requisições primárias do projeto:

1. **Agrupamento Ótimo (`find_best_geo_clusters`)**: Se não houver tratamento fixado, ele irá explorar a base massiva para elencar quais as "n" regiões do seu negócio têm mais simetria comportamental com outros pares. Se houver, pareia com exatidão as fixadas unicamente.
2. **Avaliação Pragmática (`validate_geo_groups`)**: Sendo a fase anterior a campanha, assume-se rígidos `n_folds=5` de praxe para bater os dados da base intensivamente com janelas rolantes de séries temporais. Testando fortemente a viabilidade elástica por Cross-Validation.
3. **Cálculo de Requisitos (`estimate_duration`)**: Baseado no MDE (Lift Mínimo esperado ditado por sua regra de negócio original), ele insere ruídos sintéticos retrospectivos na base idênticos a essa porcentagem, verificando *se* a variância natural da base consegue detectar a elevação sem confundi-la, e projetando a **menor duração exigida em dias seguidos**.

> [!NOTE] 
> Esta função pula conscientemente os métodos estritos de Efeito Causal (`run_synthetic_control`) e Intervalos Empíricos (`run_placebo_tests`) visto que, na perspectiva do tempo, o post-treatment ainda não existe para gerar Lifts factuais ou distribuições de probabilidade. Ela é pura mensuração premonitória.

## Retorno (*Output*)

O terminal exibe relatórios condensados de *Cluster Assignments* (quem pareia com quem para designar geograficamente a campanha) e recomenda um **Setup de Experimento**, validando estatisticamente o menor período contínuo possível para estabilizar um *Statistical Power* de $\approx 80\%$.

O dicionário retornado aglutina as inferências por grupo isolado avaliado:

```python
{
    "clusters": [...],       # O array master de agrupamentos (Controles e Fatores ElasticNet)
    "results": [
        {
            "cluster": {...},      # Composição atual
            "validation": {...},   # Baseline testado: (r2_test_avg, mape_test_avg, wape_test)
            "duration": {
               "best_power": 0.85, # O Poder estatístico máximo projetado com a duração
               "best_days": 21,    # O tempo cravado que o experimento DEVE durar em dias
            }
        }
    ]
}
```
