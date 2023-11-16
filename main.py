import matplotlib.pyplot as plt

from dataset import Dataset
from perceptron import Perceptron


def graficos(perceptron, dados):
    plt.figure(figsize=(12, 6))

    # Gráfico da evolução dos pesos
    plt.subplot(1, 2, 1)
    perceptron.plotar_evolucao_pesos()

    # Gráfico da evolução do erro
    plt.subplot(1, 2, 2)
    perceptron.plotar_evolucao_erro()

    plt.tight_layout()
    plt.show()


def main():
    try:
        dados = Dataset(tamanho=200, num_caracteristicas=5)
        dados.gerar_dados(media=2, desvio_padrao=3)

        perceptron = Perceptron(taxa_aprendizado=0.2, maximo_iteracoes=800, erro_desejado=0.001)
        perceptron.treinar(dados.entradas, dados.saidas_desejadas)

        graficos(perceptron, dados)

    except Exception as exception:
        print(f"Ocorreu um erro: {exception}")


if __name__ == "__main__":
    main()
