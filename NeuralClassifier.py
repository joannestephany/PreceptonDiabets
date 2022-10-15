import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Perceptron

mms = preprocessing.MinMaxScaler()


def preProcessar(base, isteste=False):
    aux = np.array(base[:, 0]) # pega a primeira coluna
    pregnancies = mms.fit_transform(aux.reshape(-1, 1)) # normaliza a primeira coluna

    aux = np.array(base[:, 1])
    glucose = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 2])
    bloodPressure = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 3])
    skinThickness = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 4])
    insulin = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 5])
    BMI = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 6])
    DPF = mms.fit_transform(aux.reshape(-1, 1))

    aux = np.array(base[:, 7])
    age = mms.fit_transform(aux.reshape(-1, 1))

    if isteste == False:
        outcome = base[:, 8]

    atributos_norm = np.column_stack((pregnancies, glucose, bloodPressure, skinThickness, insulin, BMI, DPF, age)) # junta as colunas normalizadas
    print("--------------------------------")
    print("Atributos de Entrada - Numéricos")
    print("--------------------------------")
    print(atributos_norm)

    if isteste == False:
        print("----------------------------------------")
        print("Classificação Supervisionada - Numéricos")
        print("----------------------------------------")
        diagnostico_norm = np.hstack((outcome)) 
        print(diagnostico_norm)
        return atributos_norm, diagnostico_norm
    return atributos_norm

# Carregando dados do arquivo CSV
# Coloquei o arquivozinho de testizinho que o prof mandou, esses, são para ensinar a rede tchucthuc
url = 'https://raw.githubusercontent.com/joannestephany/PreceptonDiabets/main/NeuralClassifier/diabetes.csv'

base_Treinamento = pd.read_csv(url, sep=',', encoding='latin1').values
print("---------------------------------")
print("Dados dos Pacientes - TREINAMENTO")
print("---------------------------------")
print(base_Treinamento)
print("---------------------------------")

# Extração dos Atributos a serem utilizadas pela rede... acho
print("Atributos de Entrada - Treinamento")
print("---------------------------------")
print(base_Treinamento[:, 0:8])

print("----------------------------")
print("Classificação Supervisionada")
print("----------------------------")
print(base_Treinamento[:,:8])

#PRE PROCESSAMENTO

atributos_norm, diagnostico_norm = preProcessar(base_Treinamento, False) # chama a função de pre processamento, que converte os dados para valores entre 0 e 1

#Treinamento do Neurônio Perceptron

modelo = Perceptron()
modelo.fit(atributos_norm, diagnostico_norm)
print('Acurácia: %.3f' % modelo.score(atributos_norm, diagnostico_norm))

# Validação do Aprendizado
# Coloquei o arquivozinho de testizinho só com os que eu quero descobrir o resultado, esses, são ver se ensinamos direito
url_testes = 'https://raw.githubusercontent.com/joannestephany/PreceptonDiabets/main/NeuralClassifier/diabetes_teste.csv'
base_Testes = pd.read_csv(url_testes, sep=';', encoding='latin1').values
print("----------------------------")
print("Dados dos Pacientes - TESTES")
print("----------------------------")
print(base_Testes)
print("---------------------------------")

# Extração dos Atributos a serem utilizadas pela rede
print("Atributos de Entrada - Testes")
print("---------------------------------")
print(base_Testes[:, 0:8])

atributos_norm = preProcessar(base_Testes, True)

base_Predicao = modelo.predict((atributos_norm))
print("\n----------------------------")
print("Classificações: ", base_Predicao) # 0 = Não Diabético, 1 = Diabético