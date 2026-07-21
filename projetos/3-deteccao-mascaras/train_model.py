import shutil

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Detecção de Máscaras Faciais (Fine-tuning do YOLO11n)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo pré-treinado YOLO11n: YOLO("yolo11n.pt")
#      (única exceção à regra de "sem modelos pré-treinados" do processo seletivo)
#   2. Fazer fine-tuning em dataset/data.yaml, em CPU (device="cpu"),
#      com um número de épocas modesto (ex: 15-30)
#   3. Copiar os pesos resultantes (results.save_dir / "weights" / "best.pt")
#      para "model.pt", na raiz desta pasta
# ---------------------------------------------------------------------------

# insira seu código aqui
print("Iniciando treinamento YOLO\n")

model = YOLO("yolo11n.pt")

results = model.train(
    data="dataset/data.yaml",
    epochs=15,
    batch=16,
    imgsz=640,
    device="cpu",
    workers=8,
    verbose=True
) 
print("Treinamento concluído. Realizando cópia dos arquivos\n")

try:
    shutil.copy(results.save_dir/"weights"/"best.pt", "model.pt")
    print("Cópia dos pesos foi concluída\n")
except FileNotFoundError:
    print("Erro: Cópia falhou. Não foi possível achar os pesos\n")
except PermissionError:
    print("Erro: Cópia falhou. Permissão negada\n")
except Exception as e:
    print(f"Erro: Cópia falhou. {e}\n")
    
# Dica de estrutura (não é obrigatório seguir exatamente assim):
#
# model = YOLO("yolo11n.pt")
# results = model.train(
#     data="dataset/data.yaml",
#     epochs=...,
#     imgsz=...,
#     batch=...,
#     device="cpu",
# )
# shutil.copy(results.save_dir / "weights" / "best.pt", "model.pt")
