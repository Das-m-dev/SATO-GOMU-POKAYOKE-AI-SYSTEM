import torch, cv2, numpy, pandas, matplotlib

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())   # False is expected here
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)

x = torch.randn(1000, 1000)
print("CPU matmul OK:", (x @ x).shape)