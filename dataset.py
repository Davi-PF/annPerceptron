import numpy as np


class Dataset:
    def __init__(self, tamanho, num_caracteristicas):
        self.tamanho = tamanho
        self.num_caracteristicas = num_caracteristicas
        self.entradas = None
        self.saidas_desejadas = None

    def gerar_dados(self, media=0, desvio_padrao=1):
        np.random.seed(0)
        self.entradas = np.random.normal(loc=media, scale=desvio_padrao, size=(self.tamanho, self.num_caracteristicas))
        self.entradas[:, -1] = -1  # Adicionando o elemento fictício
        self.saidas_desejadas = np.random.choice([1, -1], self.tamanho)
