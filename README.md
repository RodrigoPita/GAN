# Gerando Imagens no estilo de Satélite com GANs
Projeto para a disciplina de Álgebra Linear Algorítmica (2019.2) dada pelo professor João Paixão
<br>
Caso prefira ver pelo Google Collab, pode clicar neste <a href="https://colab.research.google.com/drive/1Px6zMkv5_IvK4l3Z_MwxWohvGvps-o8W?usp=sharing">link</a>.


# Redes Adversariais Generativas (GANs)

Minha ideia consiste em fazer um gerador de mapas para jogos, utilizando Redes Neurais Artificiais (RNAs), noções de Aprendizado de Máquinas
**[<a href = "https://www.coursera.org/learn/machine-learning/">10</a>]**,
mais especificamente o modelo proposto por Ian J. Goodfellow 
**[<a href = "https://arxiv.org/pdf/1406.2661.pdf">1</a>,
<a href = "https://colab.research.google.com/drive/1v9w6Eg8dAwXTf1qte2tTgB1mJN4qKFax">3</a>,
<a href = "https://github.com/eriklindernoren/Keras-GAN">8</a>,
<a href = "https://machinelearningmastery.com/generative_adversarial_networks/">9</a>,
<a href = "http://hunterheidenreich.com/blog/what-is-a-gan/">22</a>,
<a href = "https://github.com/RobStelling/GANs">23</a>]**, em seu paper publicado em 2014, sobre Redes Generativas Adversariais (GANs). Esse método é uma forma de treinar duas redes neurais simultaneamente. Mas para poder falar melhor sobre as GANs, precisamos aprender o que são Redes Neurais Artificiais!

## Redes Neurais Artificiais (RNAs)
<div align = "center"><img src= "https://3.bp.blogspot.com/-hNRMQ-sJNnw/XcNvq0JqoII/AAAAAAABCbM/r1OMZEbsW4YYsyvinAnQRL-wKX8Z8d_SwCK4BGAYYCw/s1600/Redes%2BAdversariais%2BGenerativas%2B%2528GANs%2529%2B-%2BSIAc%2B%25286%2529.png"></div>

A RNA representada no diagrama acima é do estilo "Feedforward", ou seja, ela recebe um dado e passa adiante até chegar na saída.
As RNAs, no geral, são modelos compostos por camadas de "neurônios" 
**[<a href = "https://colab.research.google.com/drive/1XvxwddJNk2gDpxSdCpNcbcc2w59sI1iT">26</a>]**
representadas pelos círculos roxos acima.

Cada neurônio funciona como uma função, ou seja, ele recebe uma entrada, faz algo com ela e passa o sinal para frente. Na imagem podemos ver as três camadas principais de uma Rede Neural, a mais da esquerda, chamada **Camada de Entrada**, de onde a informação é alimentada para Rede, as **Camadas Ocultas**, representadas pela caixa preta, que são as camadas internas da Rede, onde todos os cálculos ocorrem para transformar o significado de um dado de entrada num dado de saída e por fim, as **Camadas de Saída**, de onde recebemos os dados esperados.

> Para o computador, essas camadas são representadas como vetores 
**[<a href = "https://towardsdatascience.com/linear-algebra-explained-in-the-context-of-deep-learning-8fcb8fca1494">29</a>]**. Tomemos X como o vetor que representa a camada de Entrada, com cada componente sendo referente a cada neurônio.
<br>
<br>
$$X =
 \begin{bmatrix}
  x1\\
  x2\\
  x3\\
 \end{bmatrix}
$$
<br>
Para passar a informação à camada seguinte (Camada Oculta), precisararemos fazer o produto entre a matriz W dos pesos, que serão responsáveis por determinar, na primeira camada, quais pixels mais importam para a camada oculta e X. Outra forma de visualizar essa etapa é que ela não passa de uma soma ponderada dos valores de entrada, além disso, adicionamos um viés (bias) para tentar aproximar onde o valor do próximo neurônio começa a ser significativo;
<br>
<br>
$$\begin{bmatrix}
  a1\\
  a2\\
  a3\\
 \end{bmatrix}
 =
 \begin{bmatrix}
  w1 && w2 && w3\\
  w4 && w5 && w6\\
  w7 && w8 && w9\\
 \end{bmatrix}
 .
 \begin{bmatrix}
  x1\\
  x2\\
  x3\\
 \end{bmatrix} + bias
$$
<br>
Para finalizar, temos a função de ativação (sigmoid, ReLU, ...) que irá escalar a saída para um número real entre 0 e 1;
<br>
<br>
$$\begin{bmatrix}
  h1\\
  h2\\
  h3\\
 \end{bmatrix}
 =
 f\begin{pmatrix}
 \begin{bmatrix}
  a1\\
  a2\\
  a3\\
 \end{bmatrix}
 \end{pmatrix} (Ativação)
 $$
<br>
<br>
E esse processo se repete por quantas forem as camadas ocultas, até chegar na camada de saída.

Mas para que as RNAs tenham resultados aceitáveis, elas precisam ser treinadas. Quando a Rede acabou de ser implementada, suponha que nós temos uma **Rede Discriminativa** (Rede que classifica "coisas", real ou falso) recém implementada e queremos classificar imagens de quadros como sendo no estilo de Van Gogh ou não. Para isso, vamos criar uma lista (vetor) com todas as obras do pintor e chamaremos essa de Imagens Reais e outra com obras de outros pintores, a qual chamaremos de Imagens Falsas. 
<br>
Agora vamos iterar pelas duas listas, entregando cada imagem, já classificada como real ou falsa, para a Rede, que as lerá como matrizes de pixels com tamanho igual a proporção de pixels. Os pixels serão recebidos pelos neurônios da camada de entrada, passarão pelas camadas ocultas até chegar na saída. Ao acabar esse processo para todas as imagens das duas listas, a Rede estará devidamente treinada e pronta para ser testada. Para a classificação de imagens, dois são os métodos de treinamento mais comumente usados, uma rede simples, pixel a pixel ou uma rede convolucional.
<br>
<br>
<div align = "center"><img src = "https://1.bp.blogspot.com/-_wzsl8WtJAo/XcNt7RXWeRI/AAAAAAABCbA/Jf7mj6lC_wUuFdloA2GfpbnM_ZhVjsFtQCK4BGAYYCw/s1600/Redes%2BAdversariais%2BGenerativas%2B%2528GANs%2529%2B-%2BSIAc%2B%25285%2529.png"></div>

## Redes Neurais Artificiais (RNAs)
<div align = "center"><img src= "https://3.bp.blogspot.com/-hNRMQ-sJNnw/XcNvq0JqoII/AAAAAAABCbM/r1OMZEbsW4YYsyvinAnQRL-wKX8Z8d_SwCK4BGAYYCw/s1600/Redes%2BAdversariais%2BGenerativas%2B%2528GANs%2529%2B-%2BSIAc%2B%25286%2529.png"></div>

A RNA representada no diagrama acima é do estilo "Feedforward", ou seja, ela recebe um dado e passa adiante até chegar na saída.
As RNAs, no geral, são modelos compostos por camadas de "neurônios" 
**[<a href = "https://colab.research.google.com/drive/1XvxwddJNk2gDpxSdCpNcbcc2w59sI1iT">26</a>]**
representadas pelos círculos roxos acima.

Cada neurônio funciona como uma função, ou seja, ele recebe uma entrada, faz algo com ela e passa o sinal para frente. Na imagem podemos ver as três camadas principais de uma Rede Neural, a mais da esquerda, chamada **Camada de Entrada**, de onde a informação é alimentada para Rede, as **Camadas Ocultas**, representadas pela caixa preta, que são as camadas internas da Rede, onde todos os cálculos ocorrem para transformar o significado de um dado de entrada num dado de saída e por fim, as **Camadas de Saída**, de onde recebemos os dados esperados.

> Para o computador, essas camadas são representadas como vetores 
**[<a href = "https://towardsdatascience.com/linear-algebra-explained-in-the-context-of-deep-learning-8fcb8fca1494">29</a>]**. Tomemos X como o vetor que representa a camada de Entrada, com cada componente sendo referente a cada neurônio.
<br>
<br>
$$X =
 \begin{bmatrix}
  x1\\
  x2\\
  x3\\
 \end{bmatrix}
$$
<br>
Para passar a informação à camada seguinte (Camada Oculta), precisararemos fazer o produto entre a matriz W dos pesos, que serão responsáveis por determinar, na primeira camada, quais pixels mais importam para a camada oculta e X. Outra forma de visualizar essa etapa é que ela não passa de uma soma ponderada dos valores de entrada, além disso, adicionamos um viés (bias) para tentar aproximar onde o valor do próximo neurônio começa a ser significativo;
<br>
<br>
$$\begin{bmatrix}
  a1\\
  a2\\
  a3\\
 \end{bmatrix}
 =
 \begin{bmatrix}
  w1 && w2 && w3\\
  w4 && w5 && w6\\
  w7 && w8 && w9\\
 \end{bmatrix}
 .
 \begin{bmatrix}
  x1\\
  x2\\
  x3\\
 \end{bmatrix} + bias
$$
<br>
Para finalizar, temos a função de ativação (sigmoid, ReLU, ...) que irá escalar a saída para um número real entre 0 e 1;
<br>
<br>
$$\begin{bmatrix}
  h1\\
  h2\\
  h3\\
 \end{bmatrix}
 =
 f\begin{pmatrix}
 \begin{bmatrix}
  a1\\
  a2\\
  a3\\
 \end{bmatrix}
 \end{pmatrix} (Ativação)
 $$
<br>
<br>
E esse processo se repete por quantas forem as camadas ocultas, até chegar na camada de saída.

Mas para que as RNAs tenham resultados aceitáveis, elas precisam ser treinadas. Quando a Rede acabou de ser implementada, suponha que nós temos uma **Rede Discriminativa** (Rede que classifica "coisas", real ou falso) recém implementada e queremos classificar imagens de quadros como sendo no estilo de Van Gogh ou não. Para isso, vamos criar uma lista (vetor) com todas as obras do pintor e chamaremos essa de Imagens Reais e outra com obras de outros pintores, a qual chamaremos de Imagens Falsas. 
<br>
Agora vamos iterar pelas duas listas, entregando cada imagem, já classificada como real ou falsa, para a Rede, que as lerá como matrizes de pixels com tamanho igual a proporção de pixels. Os pixels serão recebidos pelos neurônios da camada de entrada, passarão pelas camadas ocultas até chegar na saída. Ao acabar esse processo para todas as imagens das duas listas, a Rede estará devidamente treinada e pronta para ser testada. Para a classificação de imagens, dois são os métodos de treinamento mais comumente usados, uma rede simples, pixel a pixel ou uma rede convolucional.
<br>
<br>
<div align = "center"><img src = "https://1.bp.blogspot.com/-_wzsl8WtJAo/XcNt7RXWeRI/AAAAAAABCbA/Jf7mj6lC_wUuFdloA2GfpbnM_ZhVjsFtQCK4BGAYYCw/s1600/Redes%2BAdversariais%2BGenerativas%2B%2528GANs%2529%2B-%2BSIAc%2B%25285%2529.png"></div>

### Pixel a Pixel
A tecnica mais comum é a que a rede discriminadora, ao receber o vetor de pixels relativos à imagem de entrada, vai iterar por cada pixel, comparando com os pixels respectivos de todas as imagens dos bancos de dados. Ao fazer isso, ela vai gerar uma probabilidade como resposta, dizendo o quão provável é da imagem ser verdadeira ou falsa. 

<div align = "center"><img height = "250" width = "200" src = "https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed_small.gif"></div>

### Convolução
Já essa tecnica 
**[<a href = "https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53">27</a>]**
é um pouco mais rebuscada, pois ela não olha um pixel por vez, mas sim, uma janela de pixels e vai tirando as informações da imagem, preservando um pouco melhor o contexto em que os pixels estão inseridos, diferentemente do método anterior, no qual a Rede só olha pro pixel em questão.

<div align = "center"><img  height = "250" width = "200" src = "https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed.gif"></div>

## Voltando para as GANs
Para entender a ideia das GANs, suponha que há um estudante **(G)** de artes que decide falsificar obras e vendê-las como originais, porém no começo, ele não tem ideia de como falsificá-las, então acaba cometendo erros, o especialista **(D)** chega e percebe as falhas na obra e impede o falsificador de vendê-la. A cada vez que isso acontece, o falsificador vai aprendendo melhor sobre o que configura uma obra ser autêntica ou falsa 
**[<a href = "http://hunterheidenreich.com/blog/what-is-a-gan/">22</a>]**. 
Os dois ficarão nesse "joguinho" de um fazer a obra e o outro apontar que é falsa até um ponto que o especialista não mais consiga detectar um erro.
<div align = "center"><img src= "https://2.bp.blogspot.com/-60c6crHCjdM/XcN2jRLv8MI/AAAAAAABCbY/kuwCoE5GaV8NvN4Yu0btEWW3uAibwQ6igCK4BGAYYCw/s1600/Redes%2BAdversariais%2BGenerativas%2B%2528GANs%2529%2B-%2BSIAc%2B%25287%2529.png"></div>

A primeira rede é o especialista, uma rede Discriminativa **(D)**, ou seja, ela vai receber um conjunto de dados, continuando o exemplo da analogia, de imagens de obras de Van Gogh, sendo essas classificadas como as imagens reais 
**[<a href = "https://colab.research.google.com/drive/1LYaIy4zYBEx-3wWNkOFpt-nbfMvW5ouN">2</a>]**. Além disso, vai receber as imagens geradas pela outra rede, o falsificador, uma rede Geradora **(G)** e classificará essas como falsas. Durante esse processo de treinamento das redes, a **G** irá "aprendendo" o que significa ser uma imagem real para **D**, até que depois de iterações suficientes, supondo que as arquiteturas das redes estejam devidamente equilibradas, as redes convirjam para um limite teórico denominado Equilíbrio de Nash 
**[<a href = "https://corporatefinanceinstitute.com/resources/knowledge/economics/nash-equilibrium-game-theory/">4</a>,
<a href = "https://www.khanacademy.org/economics-finance-domain/ap-microeconomics/imperfect-competition/oligopoly-and-game-theory/v/prisoners-dilemma-and-nash-equilibrium">5</a>]**,
onde **G** gera imagens "tão bem" a ponto da **D** não "saber" mais se classifica como real **(1)** ou falsa **(0)**, sempre fica em **0.5**, podendo ser uma ou a outra.

<div align = "center"><img src = "https://1.bp.blogspot.com/-ZA5scLaJ29M/XcN3MMcHOpI/AAAAAAABCbk/XZC45CdXhR86jsm4CH9hW7dK-v3OUVbTgCK4BGAYYCw/s1600/Redes%2BAdversariais%2BGenerativas%2B%2528GANs%2529%2B-%2BSIAc%2B%25288%2529.png"></div>

Aqui, podemos ver algumas das inúmeras aplicações possíveis das GANs 
**[<a href = "https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900">15</a>]**, 
como por exemplo:

* Tranferência de estilos entre imagens com a CycleGAN **[<a href = "https://junyanz.github.io/CycleGAN/">11</a>]**;

<div align = "center"><img height = "100" width = "650" src= "https://miro.medium.com/max/1812/1*Rm1C3K6ln1dvlnFgJppp6w.png"></div>


* Geração de músicas com a MidiNET **[<a href = "https://arxiv.org/pdf/1703.10847.pdf">12</a>]**;

<div align = "center"><img height = "277" width = "850" src= "https://blogger.googleusercontent.com/img/a/AVvXsEgm1vvgJsX0cvm0rirgddVjKXbToyNs7-aB3UEjWft-N9AUr0MbHIKWLgkF_U08_GgOT1fYAEOWGsT2z32lkPRRLydj4uhiDypwTKmXjir9nbmUv1QW1nA_La8ycsNZyy3VGIknMO35wWrOUrGVQsQHlpT2Uieg7Bew9_lxV-06UMa5VSwkpgJIRvQ"></div>

* Tentativa de reprodução da voz humana a partir da geração de espectrogramas **[<a href = "https://www.youtube.com/watch?v=jSsMqjMcRAg">13</a>,
<a href = "http://arss.sourceforge.net/">14</a>]**;
* Geração de personagens de animes **[<a href = "https://arxiv.org/pdf/1708.05509.pdf">16</a>]**;
<div align = "center"><img height = "250" width = "250" src= "https://miro.medium.com/max/1082/1*4oqZHrOOZRDzMsJA_eqW1g.png"></div>

* Geração de imagens de rosto humano ultra realistas da NVIDIA **[<a href = "https://www.lyrn.ai/2018/12/26/a-style-based-generator-architecture-for-generative-adversarial-networks/">17</a>,
<a href = "https://arxiv.org/pdf/1710.10196.pdf">18</a>]**;
<div align = "center"><img height = "200" width = "350" src= "https://miro.medium.com/max/1682/1*UsiBSjHy8ut5GSAT8f7bIQ.png"></div>

* Texto para imagem com StackGAN **[<a href = "https://github.com/hanzhanggit/StackGAN">19</a>,
<a href = "https://arxiv.org/pdf/1710.10916.pdf">28</a>]**;
<div align = "center"><img height = "200" width = "600" src= "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRRKcHl1lDL_0zdBYqnhAcdjHmC4g8Lp-plCmYCcvZ8Y8kxFZIa"></div>

* Geração de imagens do Padrão MNIST **[<a href = "https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/">20</a>,
<a href = "https://colab.research.google.com/drive/1TBP5YeGUgubC7MmMad1J3Ok-lXdebPEn">21</a>,
<a href = "https://github.com/RobStelling/GANs">23</a>]**;
<div align = "center"><img height = "200" width = "200" src= "https://2.bp.blogspot.com/-Thak0vJBjmg/XcJyleDspbI/AAAAAAABCZM/jqJJWTLOPXId0V1eBZg4ZPCxalhOYIp8QCK4BGAYYCw/s320/mnist_real.png"></div>

# Trabalho

Esse trabalho, tem por objetivo gerar imagens de cidades como as feitas por satélites. Para coletar as imagens de satélites para o conjundo de dados Reais, usei alguns sites específicos, voltados para esse tema
**[<a href = "https://gisgeography.com/free-satellite-imagery-data-list/">6</a>, 
<a href = "https://geology.com/satellite/">7</a>,
<a href = "https://www.google.com.br/intl/pt-BR/earth/">24</a>
<a href = "https://www.mapbox.com/maps/satellite/">25</a>]**, 
quanto a resolução das imagens, todas foram redimensionadas para estarem em **108x78 px**. Durante o treinamento, percebi que apesar de demorar mais, para esse problema, o método de convoluções traz resultados melhores, pois ele elimina a maior parte dos ruídos, diferentemente do método simples, que lê pixel a pixel. A proporção das imagens foi escolhida também, já pensando na possibilidade de convolção, visto que temos que conseguir dividir os tamanhos da largura e comprimento da imagem, nesse caso, por 3 e 2, para poder aplicar o método adequadamente.
<div align = "center"><img src = "https://2.bp.blogspot.com/-FhQPyDbNbEk/XcN3Yu9yx7I/AAAAAAABCbs/QJkvze9FU-4Wm1EpJInVNCkf83uuLP4KgCK4BGAYYCw/s1600/Redes%2BAdversariais%2BGenerativas%2B%2528GANs%2529%2B-%2BSIAc%2B%25289%2529.png"></div>
