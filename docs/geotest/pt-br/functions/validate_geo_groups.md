# `reallift.geo.validation.validate_geo_groups`

A função `validate_geo_groups` é o motor de auditoria de *Overfitting* estático ou temporal para os grupos de geografias candidatas. Sua finalidade é assegurar que o Controle Sintético escolhido não decorre simplesmente de um ajuste espúrio perfeito aos ruídos passados, mas que ele possua uma capacidade preditiva genuína *Out-of-Sample*.

## Assinatura

```python
def validate_geo_groups(
    filepath: str,
    date_col: str,
    splits: list,
    treatment_start_date: str = None,
    train_test_split: float = 0.8,
    n_folds: int = 1,
    plot: bool = True,
    export_csv: bool = False,
    output_prefix: str = "geo_validation",
    cluster_idx: int = None,
    verbose: bool = True
) -> dict
```

## Fundamentação Matemática e Algorítmica

A validação resolve empiricamente o problema de viés-variância na fase pré-tratamento. 

### Validação Cruzada de Séries Temporais (*Time Series Cross-Validation*)
Se `n_folds > 1`, a função aplica uma rotina baseada no `TimeSeriesSplit` preservando a causalidade flecha-do-tempo. Ele particiona a matriz de pré-tratamento em $K$ janelas deslizantes sequenciais.
Em cada *fold*:
1. Os pesos estritos convexos sintéticos ($w$) são otimizados (via *Solver SCS* cooperando com o `cvxpy`) enraizados puramente nos dados do `Train Set` de sua respectiva partição.
2. Essa equação de matriz otimizada é cegamente aplicada para inferir o contrafactual da variância temporal no espaço adjacente do `Test Set`.
3. Os resíduos de predição cega (OOF) guiam métricas absolutas rigorosas como *Out-Of-Fold* $R^2$ e Erro Percentual Absoluto (MAPE e WAPE).

### Validação Estática (*Static Trend Split*)
Se `n_folds = 1` (default), aplica-se um fracionamento determinístico definido diretamente pelo limiar `train_test_split` (usualmente reservando os 80% iniciais do tempo pra Treino, os 20% finais como Teste Virgem). 

### Métricas de Penalização (OOF R2 Gap)
O algorítmo é cauteloso e acionará um *Warning* visível na sua tela de terminal (`⚠️ High R2 gap...`) se a assertividade preditiva da base cega cair sistematicamente mais de `0.20` pontos frente o treinamento. O R-Quadrado decorrido no treino nunca deve polarizar catastroficamente com os testes finais; distanciamentos severos constituem de fato "Assinaturas de Ruído".

## Retorno (*Output*)

Devolve um super dicionário Python englobando:
- `summary`: DataFrame com arrays indexados de `r2_train`, `r2_test`, `mape_test` cruzados por cada rodada.
- `outputs`: Lista serializada de DataFrames para visualização paramétrica com matriz temporal limpa e `residuals`.
