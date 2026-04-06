# Synthetic Error Ratio: Otimização de Precisão em Marketing Science
**Autor: Roberto Junior**

> *Inovação Metodológica - Framework RealLift*
> *Complementar ao: [RealLift Overview](./reallift_overview.md)*

---

## 1. O Problema da Escolha de Doadores

No Método do Controle Sintético (Abadie et al., 2010), o objetivo é criar um "clone" digital de uma cidade através de uma média ponderada de outras cidades (controles). Tradicionalmente, o algoritmo busca minimizar o **RMSPE (Root Mean Squared Prediction Error)** — a distância média entre a cidade real e a sintética no período pré-teste.

### A Armadilha dos "Controles Zumbis"
Em mercados de alta volatilidade (como o brasileiro), a busca exclusiva pelo menor erro RMSPE pode levar a um viés perigoso. O algoritmo pode escolher cidades com baixa variação histórica (séries estacionárias ou "planas") apenas porque elas não aumentam o ruído.

No RealLift, chamamos estes doadores de **"Controles Zumbis"**:
- Eles dão um RMSPE baixo (excelente encaixe visual na média).
- Eles têm **zero correlação** com os picos e vales da unidade tratada.
- Eles falham catastroficamente fora da amostra (out-of-sample) porque não reagem aos mesmos choques do mercado.

---

## 2. A Solução: Coeficiente SER

Para mitigar o risco de escolher de forma míope apenas pelo erro residual, o framework RealLift utiliza o **Synthetic Error Ratio (SER)** como sua *Loss Function* principal de ranking na fase de Design de Experimentos (DoE).

### Formulação Matemática

$$ SER = \frac{\text{Std. Residual}}{\rho(Y_\text{Trat}, Y_\text{Sint}) + \epsilon} $$

- **Numerador ($\sigma$):** Mantém o Resíduo de Predição (quanto menor a distância, melhor).
- **Denominador ($\rho$):** Aplica a Correlação Linear de Pearson entre a unidade tratada e o sintético gerado.
- **Epsilon ($\epsilon$):** Um fator de estabilidade para evitar divisões por zero.

---

## 3. Comportamento e Vantagens

O SER atua como uma barreira pragmática e barata ao ruído. Ele penaliza agressivamente combinações que tentam ser "clones estáticos".

1.  **Penalização de Zumbis**: Se a correlação tende a zero, o coeficiente $SER$ explode, enviando aquele cenário para o fundo do ranking de recomendações.
2.  **Recompensa ao Co-movimento**: O SER aceita um erro absoluto ligeiramente maior (Numerador) se, em troca, ganhar uma sincronia maior nos picos e vales (Denominador). 
3.  **Filtragem Proativa**: No DoE, rodamos milhares de combinações geográficas. O SER permite que o modelo escolha, de forma autônoma, os cenários que têm maior probabilidade de permanecer estáveis durante a campanha real.

## Conclusão

O **Synthetic Error Ratio** é o coração da inteligência de seleção do RealLift. Ao equilibrar a precisão numérica (Resíduo) com a sincronia comportamental (Correlação), ele garante que seus experimentos geográficos sejam ancorados em contrafactuais vivos e reagentes à realidade do mercado.
