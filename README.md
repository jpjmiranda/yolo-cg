LINK PARA ACESSO DO DATASET - https://universe.roboflow.com/aivle5-f7j14/one-piece-uuyxt

# YOLO-CG — Treinamento e Inferência com YOLOv8

Projeto desenvolvido para a disciplina de **Visão Computacional**, com o objetivo de aplicar, na prática, o processo de criação de um detector de objetos personalizado utilizando **YOLOv8**.

O projeto contempla as etapas de preparação do dataset, organização dos arquivos no formato YOLO, treinamento do modelo com dados customizados e realização de inferência em vídeo utilizando o modelo treinado.

---

## 1. Objetivo do Projeto

O objetivo principal deste projeto é treinar um modelo de detecção de objetos utilizando o **YOLOv8**, com um dataset personalizado baseado em personagens de **One Piece**.

A proposta consiste em demonstrar o fluxo completo de trabalho em visão computacional, passando pelas seguintes etapas:

1. Organização de um dataset customizado no formato YOLO;
2. Configuração do ambiente Python;
3. Treinamento de um modelo YOLOv8;
4. Geração dos pesos treinados do modelo;
5. Utilização do modelo treinado para realizar inferência;
6. Aplicação da inferência em vídeo;
7. Registro e documentação do processo em repositório GitHub.

---

## 2. Contexto do Projeto

Este projeto foi desenvolvido com base nas aulas práticas da disciplina, principalmente nas aulas relacionadas ao treinamento customizado e inferência com YOLO.

O fluxo seguido foi:

```text
Aula 09 → Organização do dataset customizado
Aula 10 → Treinamento do modelo YOLOv8
Aula 11 → Inferência utilizando o modelo treinado
```

Dessa forma, o projeto não se limita apenas ao treinamento do modelo, mas demonstra todo o processo necessário para criar uma aplicação simples de detecção de objetos.

---

## 3. Tecnologias Utilizadas

As principais tecnologias utilizadas neste projeto foram:

- Python
- Poetry
- Jupyter Notebook
- Ultralytics YOLOv8
- PyTorch
- OpenCV
- NumPy
- Roboflow
- CUDA
- NVIDIA RTX 3060 8GB
- Git
- GitHub

---

## 4. Estrutura do Repositório

A estrutura geral do projeto está organizada da seguinte forma:

```text
yolo-cg/
│
├── dataset_onepiece/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── README.dataset.txt
│   └── README.roboflow.txt
│
├── notebooks/
│   ├── teste.ipynb
│   └── aula11_inferencia_yolo.ipynb
│
├── videos_teste/
│   └── video_onepiece.mp4
│
├── src/
│   └── aula_09/
│
├── tests/
│
├── .gitignore
├── poetry.lock
├── pyproject.toml
└── README.md
```

---

## 5. Dataset Customizado

O dataset utilizado neste projeto foi organizado no formato YOLOv8.

O dataset está localizado na pasta:

```text
dataset_onepiece/
```

Dentro dessa pasta, existem os arquivos e diretórios necessários para o treinamento do YOLO:

```text
dataset_onepiece/
│
├── data.yaml
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── README.dataset.txt
└── README.roboflow.txt
```

A pasta `train` contém os dados utilizados para treinamento do modelo.

A pasta `valid` contém os dados utilizados para validação durante o treinamento.

As imagens ficam dentro das pastas `images`, enquanto os arquivos de anotação ficam dentro das pastas `labels`.

Cada imagem possui um arquivo `.txt` correspondente contendo as coordenadas das caixas delimitadoras no formato YOLO.

---

## 6. Arquivo data.yaml

O arquivo `data.yaml` é um dos arquivos mais importantes do projeto, pois é ele que informa ao YOLO onde estão as imagens de treino, as imagens de validação e quais são as classes do dataset.

O arquivo está localizado em:

```text
dataset_onepiece/data.yaml
```

Esse arquivo possui informações como:

```yaml
train: train/images
val: valid/images

nc: 9

names:
  - Blook
  - Chyopa
  - Franky
  - Luffy
  - Nami
  - NicoRobin
  - Sangdi
  - Usopp
  - Zoro
```

---

## 7. Classes do Dataset

O dataset possui 9 classes, representando personagens de One Piece.

As classes utilizadas são:

```text
0 - Blook
1 - Chyopa
2 - Franky
3 - Luffy
4 - Nami
5 - NicoRobin
6 - Sangdi
7 - Usopp
8 - Zoro
```

A quantidade total de classes é definida no arquivo `data.yaml` pelo campo:

```yaml
nc: 9
```

O campo `names` indica o nome de cada classe utilizada no treinamento.

---

## 8. Aula 09 — Organização do Dataset

Na Aula 09, o foco foi a preparação do dataset customizado para ser utilizado no treinamento com YOLO.

A principal etapa dessa aula foi garantir que o dataset estivesse organizado corretamente no padrão esperado pelo YOLOv8.

A estrutura necessária é:

```text
dataset_onepiece/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

A pasta `images` contém as imagens utilizadas no treinamento e na validação.

A pasta `labels` contém os arquivos de anotação no formato `.txt`.

Cada arquivo de anotação possui as informações das caixas delimitadoras, seguindo o padrão:

```text
classe x_centro y_centro largura altura
```

Esses valores são normalizados, ou seja, variam entre 0 e 1.

Essa etapa foi fundamental, pois o YOLO depende diretamente da estrutura correta do dataset para conseguir realizar o treinamento.

---

## 9. Aula 10 — Treinamento do Modelo YOLOv8

Na Aula 10, foi realizado o treinamento do modelo YOLOv8 utilizando o dataset personalizado.

O modelo base utilizado foi:

```python
yolov8n.pt
```

Esse modelo é uma versão leve do YOLOv8, adequada para testes, estudos e treinamentos mais rápidos.

O treinamento utiliza o arquivo `data.yaml` para localizar as imagens e as classes do dataset.

Exemplo de código utilizado para treinamento:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset_onepiece/data.yaml",
    epochs=5,
    imgsz=416,
    batch=8,
    device=0
)
```

Descrição dos principais parâmetros:

| Parâmetro | Função |
|---|---|
| `data` | Caminho para o arquivo `data.yaml` |
| `epochs` | Quantidade de épocas de treinamento |
| `imgsz` | Tamanho das imagens utilizadas no treinamento |
| `batch` | Quantidade de imagens processadas por vez |
| `device` | Define se o treino será feito na CPU ou GPU |

No caso deste projeto, foi utilizada GPU NVIDIA com CUDA disponível, por isso o parâmetro usado foi:

```python
device=0
```

Esse valor indica que o treinamento deve utilizar a GPU principal da máquina.

---

## 10. Verificação da GPU com CUDA

Durante o desenvolvimento do projeto, foi verificado se a GPU estava disponível para acelerar o treinamento.

O código utilizado para testar a disponibilidade da CUDA foi:

```python
import torch

print("CUDA disponível?", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA do Torch:", torch.version.cuda)
```

A saída esperada, quando a GPU está funcionando corretamente, é semelhante a:

```text
CUDA disponível? True
GPU: NVIDIA GeForce RTX 3060
CUDA do Torch: 12.8
```

Inicialmente, o PyTorch estava instalado apenas na versão CPU, o que fazia com que o comando `torch.cuda.is_available()` retornasse `False`.

Para resolver isso, foi necessário reinstalar o PyTorch com suporte CUDA.

Comando utilizado:

```bash
poetry run python -m pip uninstall -y torch torchvision torchaudio
poetry run python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Após a instalação correta, o ambiente passou a reconhecer a GPU normalmente.

---

## 11. Resultado do Treinamento

Após o treinamento, o YOLO gera automaticamente uma pasta dentro de:

```text
runs/detect/
```

O nome da pasta pode variar conforme a quantidade de treinamentos realizados.

Exemplos:

```text
runs/detect/train/
runs/detect/train2/
runs/detect/train3/
runs/detect/train4/
```

Dentro da pasta de treino, são gerados diversos arquivos importantes.

A estrutura principal é semelhante a:

```text
runs/detect/train/
│
├── weights/
│   ├── best.pt
│   └── last.pt
│
├── results.png
├── confusion_matrix.png
├── val_batch0_pred.jpg
└── args.yaml
```

---

## 12. Arquivos Gerados pelo YOLO

Durante o treinamento, o YOLO gera arquivos que ajudam a avaliar o desempenho do modelo.

Os principais arquivos são:

| Arquivo | Descrição |
|---|---|
| `best.pt` | Melhor modelo gerado durante o treinamento |
| `last.pt` | Modelo salvo na última época |
| `results.png` | Gráfico com métricas do treinamento |
| `confusion_matrix.png` | Matriz de confusão do modelo |
| `val_batch0_pred.jpg` | Exemplo de predição feita na validação |
| `args.yaml` | Arquivo com os parâmetros usados no treinamento |

O arquivo mais importante é:

```text
best.pt
```

Esse arquivo representa o melhor modelo treinado e é utilizado posteriormente na etapa de inferência.

---

## 13. Aula 11 — Inferência com YOLO

Na Aula 11, o modelo treinado foi utilizado para realizar inferência.

Inferência é o processo de usar um modelo já treinado para detectar objetos em novos dados, como imagens ou vídeos.

Neste projeto, a inferência foi realizada em vídeo.

O modelo carregado para inferência foi o arquivo:

```text
best.pt
```

Esse arquivo foi gerado durante o treinamento da Aula 10.

---

## 14. Inferência em Vídeo

O vídeo de teste foi colocado na pasta:

```text
videos_teste/
```

Exemplo de estrutura:

```text
videos_teste/
└── video_onepiece.mp4
```

O código utilizado para carregar o modelo treinado foi:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
```

Em alguns casos, como o YOLO pode criar pastas `train`, `train2`, `train3` e assim por diante, também foi utilizado código para localizar automaticamente a pasta de treino mais recente.

Exemplo:

```python
from pathlib import Path
import os

CWD = Path.cwd()

possible_detect_dirs = [
    CWD / "runs" / "detect",
    CWD.parent / "runs" / "detect",
    CWD.parent.parent / "runs" / "detect",
]

train_folders = []

for detect_dir in possible_detect_dirs:
    if detect_dir.exists():
        train_folders.extend(list(detect_dir.glob("train*")))

if not train_folders:
    raise FileNotFoundError("Nenhuma pasta de treino encontrada. Rode primeiro a aula 10.")

latest_train = max(train_folders, key=os.path.getctime)

best_model = latest_train / "weights" / "best.pt"

print("Último treino encontrado:", latest_train)
print("Modelo encontrado:", best_model)

if not best_model.exists():
    raise FileNotFoundError("best.pt não encontrado dentro da pasta weights.")
```

Depois de localizar o modelo, ele foi carregado com:

```python
from ultralytics import YOLO
import torch

print("CUDA disponível?", torch.cuda.is_available())

model = YOLO(str(best_model))

print("Modelo carregado com sucesso!")
print("Classes:", model.names)
```

---

## 15. Código de Inferência em Vídeo

O código principal de inferência em vídeo utilizado foi:

```python
from pathlib import Path

video_path = CWD.parent / "videos_teste" / "video_onepiece.mp4"

if video_path.exists():
    print("Vídeo encontrado:")
    print(video_path)
else:
    raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
```

Após confirmar que o vídeo existe, a inferência foi executada com:

```python
results = model.predict(
    source=str(video_path),
    conf=0.5,
    device=0,
    save=True,
    project=str(CWD / "runs" / "inferencias_video"),
    name="aula11_video_onepiece",
    exist_ok=True
)

print("Inferência no vídeo finalizada!")
print("Resultado salvo em:", results[0].save_dir)
```

Descrição dos parâmetros:

| Parâmetro | Função |
|---|---|
| `source` | Caminho do vídeo utilizado na inferência |
| `conf` | Confiança mínima para considerar uma detecção |
| `device` | Define o uso da GPU |
| `save` | Salva o resultado da inferência |
| `project` | Define a pasta onde o resultado será salvo |
| `name` | Define o nome da pasta de saída |
| `exist_ok` | Permite reutilizar a pasta sem gerar erro |

---

## 16. Exibição do Vídeo com Resultado

Após a inferência, o vídeo com as detecções é salvo automaticamente na pasta definida no parâmetro `project`.

O código utilizado para localizar e exibir o vídeo resultante foi:

```python
from IPython.display import Video, display
from pathlib import Path

save_dir = Path(results[0].save_dir)

videos_resultado = (
    list(save_dir.glob("*.mp4")) +
    list(save_dir.glob("*.avi")) +
    list(save_dir.glob("*.mov"))
)

if videos_resultado:
    video_resultado = videos_resultado[0]
    print("Vídeo com detecção:")
    print(video_resultado)
    display(Video(str(video_resultado), embed=True, width=800))
else:
    print("Nenhum vídeo de resultado encontrado.")
```

---

## 17. Fluxo Completo do Projeto

O fluxo completo desenvolvido neste projeto pode ser representado da seguinte forma:

```text
Coleta/seleção das imagens
        ↓
Anotação das imagens
        ↓
Exportação no formato YOLOv8
        ↓
Organização do dataset
        ↓
Configuração do data.yaml
        ↓
Treinamento com YOLOv8
        ↓
Geração do best.pt
        ↓
Carregamento do modelo treinado
        ↓
Inferência em vídeo
        ↓
Vídeo final com detecções
```

---

## 18. Como Executar o Projeto

Para executar este projeto, primeiro clone o repositório:

```bash
git clone https://github.com/jpjmiranda/yolo-cg.git
```

Entre na pasta do projeto:

```bash
cd yolo-cg
```

Instale as dependências com Poetry:

```bash
poetry install
```

Ative o ambiente virtual:

```bash
poetry shell
```

Também é possível executar comandos diretamente com:

```bash
poetry run python nome_do_arquivo.py
```

Para utilizar o ambiente no Jupyter Notebook, instale o kernel:

```bash
poetry run python -m ipykernel install --user --name yolo-cg --display-name "Python (yolo-cg)"
```

Depois, abra o notebook no VS Code ou Jupyter e selecione o kernel:

```text
Python (yolo-cg)
```

---

## 19. Execução do Treinamento

O treinamento pode ser executado em um notebook localizado na pasta:

```text
notebooks/
```

Exemplo de código para treinamento:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset_onepiece/data.yaml",
    epochs=5,
    imgsz=416,
    batch=8,
    device=0
)
```

Caso o computador não possua GPU ou o CUDA não esteja disponível, é possível treinar usando CPU:

```python
results = model.train(
    data="dataset_onepiece/data.yaml",
    epochs=5,
    imgsz=416,
    batch=2,
    device="cpu"
)
```

O treinamento em CPU é mais lento, por isso a GPU é recomendada.

---

## 20. Execução da Inferência

A inferência em vídeo pode ser executada no notebook:

```text
notebooks/aula11_inferencia_yolo.ipynb
```

O vídeo deve estar localizado em:

```text
videos_teste/video_onepiece.mp4
```

O modelo utilizado deve ser o arquivo `best.pt`, gerado no treinamento.

Exemplo de inferência:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(
    source="videos_teste/video_onepiece.mp4",
    conf=0.5,
    device=0,
    save=True
)
```

Após a execução, o YOLO salva o vídeo processado com as detecções.

---

## 21. Possíveis Problemas e Soluções

### Problema: CUDA disponível aparece como False

Quando o comando:

```python
torch.cuda.is_available()
```

retorna `False`, significa que o PyTorch não está reconhecendo a GPU.

Isso pode acontecer quando o PyTorch foi instalado na versão CPU.

Solução utilizada:

```bash
poetry run python -m pip uninstall -y torch torchvision torchaudio
poetry run python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Depois disso, é necessário reiniciar o kernel do notebook.

---

### Problema: O arquivo best.pt não foi encontrado

O arquivo `best.pt` só é criado depois que o treinamento é executado com sucesso.

O caminho esperado é semelhante a:

```text
runs/detect/train/weights/best.pt
```

ou:

```text
runs/detect/train2/weights/best.pt
```

ou ainda:

```text
runs/detect/train3/weights/best.pt
```

A numeração muda conforme novos treinamentos são executados.

---

### Problema: O vídeo não foi encontrado

O vídeo precisa estar na pasta correta:

```text
videos_teste/
```

O nome utilizado no notebook precisa ser exatamente o mesmo nome do arquivo.

Exemplo:

```text
video_onepiece.mp4
```

Caso o vídeo tenha outro nome, é necessário alterar o caminho no código.

---

### Problema: Erro de memória da GPU

Caso ocorra erro de memória durante o treinamento, é possível reduzir o valor de `batch`.

Exemplo:

```python
batch=4
```

ou:

```python
batch=2
```

Também é possível reduzir o tamanho da imagem:

```python
imgsz=320
```

Essas alterações reduzem o consumo de memória da GPU.

---

## 22. Arquivos Ignorados pelo Git

Alguns arquivos gerados durante o treinamento podem ser grandes e não precisam necessariamente ser enviados para o GitHub.

Exemplos:

```text
runs/
*.pt
*.onnx
*.engine
*.cache
```

Esses arquivos podem ser incluídos no `.gitignore` para evitar o envio de arquivos pesados.

O arquivo `best.pt`, por exemplo, pode ser gerado novamente executando o treinamento.

---

## 23. Importância do Projeto

Este projeto é importante porque demonstra uma aplicação prática de visão computacional com detecção de objetos.

Durante o desenvolvimento, foram trabalhados conceitos como:

- Dataset customizado;
- Anotação de imagens;
- Estrutura de arquivos no formato YOLO;
- Uso de arquivo YAML;
- Treinamento supervisionado;
- Modelos pré-treinados;
- Transfer learning;
- Inferência em vídeo;
- Uso de GPU;
- Organização de projeto no GitHub;
- Documentação técnica.

Além disso, o projeto mostra como um modelo treinado pode ser reutilizado para analisar novos arquivos, como vídeos, aplicando as classes aprendidas durante o treinamento.

---

## 24. Conclusão

O projeto apresentou o processo completo de desenvolvimento de um detector de objetos personalizado com YOLOv8.

Inicialmente, foi realizada a organização do dataset no formato adequado para o YOLO. Em seguida, o modelo `yolov8n.pt` foi utilizado como base para o treinamento com as classes personalizadas do dataset One Piece.

Após o treinamento, o arquivo `best.pt` foi gerado e utilizado na etapa de inferência. Por fim, o modelo treinado foi aplicado em um vídeo de teste, demonstrando a capacidade do YOLO em detectar objetos/personagens em novos dados.

Com isso, o projeto conclui as etapas práticas de dataset, treinamento e inferência, demonstrando o uso de visão computacional em uma aplicação real.

---

## 25. Autor

Projeto desenvolvido por **João Miranda** para a disciplina de **Computação Gráfica**.
