from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite via model.export(format="tflite")
#      (a Ultralytics gera automaticamente "model.tflite" na mesma pasta)
# ---------------------------------------------------------------------------

# insira seu código aqui
print("Iniciando a otimização do modelo 'model.pt'\n")

model = YOLO("model.pt")

#format foi modificado de "tflite" para "litert", já que esse tipo foi depreciado e
#a documentação recomenda a utilização desta nova. O mesmo arquivo "model.tflife" é 
#gerado
model.export(format="tflite", imgsz=640)

print("Exportação e otimização finalizadas\n")
# Dica de estrutura (não é obrigatório seguir exatamente assim):
#
# model = YOLO("model.pt")
# model.export(format="tflite", imgsz=...)
