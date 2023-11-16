import numpy as np
from matplotlib import pyplot as plt

from utils import perceptron_function

class Perceptron:
    def __init__(self, taxa_aprendizado=0.5, maximo_iteracoes=1000, erro_desejado=0.01):
        self.taxa_aprendizado = taxa_aprendizado
        self.maximo_iteracoes = maximo_iteracoes
        self.erro_desejado = erro_desejado
        self.pesos = None
        self.hist_pesos = []
        self.hist_erros = []

    def iniciar_pesos(self, tam_escopo):
        return np.random.rand(tam_escopo)

    def treinar(self, entradas, saidas_desejadas):
        self.pesos = self.iniciar_pesos(entradas.shape[1])

        for iteracao in range(self.maximo_iteracoes):
            erro_total = 0

            for vetor_entrada, saida_desejada in zip(entradas, saidas_desejadas):
                saida = self.prever(vetor_entrada)
                erro = saida_desejada - saida

                self.pesos += self.taxa_aprendizado * erro * vetor_entrada

                erro_total += erro ** 2
                self.hist_pesos.append(self.pesos.copy())
                self.hist_erros.append(erro_total)

            if erro_total <= self.erro_desejado:
                break

    def prever(self, vetor_entrada):
        u = np.dot(vetor_entrada, self.pesos)
        return perceptron_function(u)

    def plotar_evolucao_pesos(self):
        plt.plot(self.hist_pesos)
        plt.title("Evolução dos Pesos")
        plt.xlabel("Iterações")
        plt.ylabel("Valor dos Pesos")
        plt.legend(['Peso 1', 'Peso 2', 'Viés'])

    def plotar_evolucao_erro(self):
        plt.plot(self.hist_erros)
        plt.title("Evolução do Erro")
        plt.xlabel("Iterações")
        plt.ylabel("Erro Total")
        plt.yscale('log')
