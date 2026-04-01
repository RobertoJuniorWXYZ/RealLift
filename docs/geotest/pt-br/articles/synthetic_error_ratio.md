# Synthetic Error Ratio: Uma Alternativa Heurística para Otimização de Controles Sintéticos em Marketing Science
**Autor: Roberto Junior**

> *Documentação Metodológica - Framework RealLift*

---

## Resumo Executivo
O Método do Controle Sintético (SCM), introduzido por Abadie et al. (2010), estabeleceu um padrão ouro para a inferência causal moderna. Sua interpretabilidade e rigor matemático o tornaram extremamente popular. Contudo, a aplicação do SCM em cenários de negócios de alta frequência temporal — como dados diários de vendas e investimentos de mídia no varejo — traz desafios específicos. A altíssima volatilidade e os choques exógenos constantes podem, em algumas situações, levar o modelo a sofrer de *viés de interpolação*.

Este artigo revisa o funcionamento clássico do SCM e as propostas da literatura recente (como regressões regulares e métodos aumentados) para lidar com séries temporais ruidosas. Com profundo respeito aos métodos consolidados, apresentamos o coeficiente **Synthetic Error Ratio (SER)**, uma métrica heurística alternativa adotada pelo framework *RealLift*. O objetivo do SER não é substituir a econometria clássica, mas oferecer uma alternativa pragmática e altamente interpretável para Cientistas de Dados que precisam balancear o rigor estatístico com a narativa de negócios em painéis corporativos.

---

## 1. O Controle Sintético Clássico e o Viés de Interpolação

A beleza do SCM de Abadie reside na sua formulação intuitiva. O objetivo central é criar uma versão "sintética" da unidade tratada combinando unidades do grupo de controle (doadores). 

A otimização é desenhada sob restrições estritas e elegantes:
1. Nenhuma unidade de controle recebe peso negativo ($w_i \ge 0$).
2. A soma total dos pesos equivale a 100% ($\sum w_i = 1$).

Sob essas regras, o SCM busca a combinação de pesos que minimize o erro quadrático de predição (RMSPE) no período anterior à intervenção. Essa pureza torna o modelo incrivelmente didático para executivos. 

### A Limitação Prática (O "Controle Zumbi")
O SCM foi desenhado primariamente para dados de variação menos caótica (como o PIB trimestral de estados). No *Marketing Science*, usamos dados hiper-voláteis, como vendas diárias sob campanhas sazonais.
Neste cenário agressivo, a busca exclusiva pela minimização da distância do Erro (RMSPE) pode criar um viés indesejado. O algoritmo pode, matematicamente, preferir selecionar doadores com baixa variação histórica (séries estacionárias, quase "planas") e combiná-las para que cruzem exatamente o meio da média volátil da unidade tratada.

No jargão prático, costumamos chamar essas séries planas de **"Controles Zumbis"**. O algoritmo os prefere porque, in-sample, a combinação deles não piora ativamente o ruído passivo (baixo RMSPE). No entanto, o pressuposto de um contrafactual ideal é o *co-movimento*: a suposição de que o grupo doador reagirá de forma semelhante aos mesmos choques do mercado. As séries "planas" não reagem aos picos; portanto, falham sistematicamente fora da amostra (out-of-sample) durante a campanha de mídia, comprometendo o teste.

---

## 2. A Evolução da Literatura e Abordagens Alternativas

Em resposta aos desafios de interpolação e ausência de co-movimento, pesquisadores propuseram evoluções extraordinárias ao método. Ambas abordagens abaixo solucionam o problema estatístico, mas trazem diferentes adaptações para as regras originais de Abadie:

### A. Regressão Regularizada (Doudchenko & Imbens, 2016)
Este notável trabalho propõe flexibilizar as premissas de Abadie. Ao invés da otimização convexa restrita, os autores sugerem estimar os pesos utilizando regressões penalizadas elásticas (Elastic Net, Lasso, Ridge) e permitem a inclusão de um intercepto.
* **O Trade-Off e a Limitação:** A solução é impecável do ponto de vista do fit matemático, eliminando o erro sistemático entre tendência e nível. Contudo, ela permite pesos de doadores que sejam negativos e somas que não batam 100%. Em ambientes corporativos, a explicação da composição do painel de controle ("Temos 1.3 de influência no RS e -0.4 na BA") perde a força da transparência original cobiçada pela liderança.

### B. Augmented Synthetic Control Method - ASCM (Ben-Michael et al., 2021)
O ASCM unifica o melhor dos dois mundos. Ele roda o algoritmo restrito de Abadie primeiro. E para corrigir o resíduo estatístico (o *bias* onde o SCM clássico falha em espelhar as tendências da unidade tratada), ele aplica uma correção final na predição usando uma regressão tipo Ridge como "estimador de viés".
* **O Trade-Off e a Limitação:** Trata-se de uma solução definitiva para a Ciência de Dados robusta, e é a base de excelentes ferramentas na indústria. A complexidade do cálculo adicional no *backend*, no entanto, adiciona uma camada mais profunda de modelagem, abstraindo levemente de onde exatamente o preenchimento do volume derivou no cálculo absoluto exibido.

---

## 3. A Proposta da RealLift: O Synthetic Error Ratio (SER)

Diante do catálogo riquíssimo da literatura, a **RealLift** propõe uma trilha ramificada. O objetivo não é ser mais rigoroso matematicamente que Doudchenko ou o ASCM, mas ser **pragmático**: proteger a inviolabilidade visual da "Pizza de 100%" de pesos positivos de Abadie, sem depender de "correções" no valor final projetado da curva.

O motor atua em três fluxos sequenciais mitigadores:
1. **Seleção e Purificação de Variáveis (Filtro ElasticNet):** O método utiliza os mesmos princípios do ElasticNet (Doudchenko), mas o limita estritamente a um *Feature Selector* prévio. Doadores que têm correlação nula ou inversa recebem força estritamente zero, eliminando o inchaço matricial.
2. **A Manutenção Clássica (SCM CVXPY):** Usando apenas a malha correlacionada altamente confiável filtrada no passo 1, devolvemos a regressão para a maravilhosa simplicidade convexa de pesos positivos limitados a 100%. 

Aqui chegamos à inovação. Se operarmos nas regras de Abadie apenas no cenário do passo 2, ainda estamos reféns do algoritmo escolher o subgrupo que gerar o menor erro residual (RMSPE), arriscando abraçar uma métrica perfeitamente engessada. 

Para quebrar esse impasse, inserimos o **Synthetic Error Ratio**.

---

## 4. Como o SER Penaliza a Falta de Movimento

Em um grid exaustivo testando combinações geográficas candidatas para formar o arranjo sintético, qual deles apresenta o menor *Mínimo Efeito Detectável (MDE)* real corporativo?
A RealLift ranqueia as opções simuladas aplicando a seguinte métrica balizadora como sua *Loss Function Heurística*:

$$ SER = \frac{\text{Std. Residual}}{\rho(Y_\text{Trat}, Y_\text{Sint}) + \epsilon} $$

*   **O Numerador:** Mantém o Resíduo de Predição (A Distância clássica, quanto mais apertada, melhor).
*   **O Denominador:** Aplica a Correlação Linear de Pearson entre os picos e vales do grupo tratado e da simulação sintética.

**Como ele atua:**
A regra de negócio é empírica: *O doador precisa oscilar no exato mesmo momento sociológico do mercado tratado.* 
Se a simulação encontrar doadores milagrosamente estáveis (os temidos *Zumbis*), que geram um erro artificialmente baixo pelo seu aspecto achatado e sem dispersão, a sua Correlação de Pearson ($\rho$) tenderá a se afastar do 1. Essa divisão agirá como um severo *malus*, explodindo o coeficiente $SER$ e despencando a simulação para o fundo do ranking aceitável.

O $SER$ recompensa combinações ligeiramente imperfeitas – que podem gerar picos matemáticos com um pouco mais de distância em valores brutos (aumentando o Numerador) – mas que acompanham de forma sagrada os vales comportamentais e sazonalidades orgânicas ativas do Brasil (enchendo o Denominador).

## Conclusão
O *Marketing Science* demanda soluções interpretáveis nos conselhos de decisão sem abrir mão da integridade da amostragem em produção. O **Synthetic Error Ratio** consolida a proposta da biblioteca *RealLift* em resgatar a elegância dogmática de Abadie enquanto estipula uma barreira pragmática e barata aos ruídos das volatilidades contínuas que permeiam os anúncios digitais de alta escala.
