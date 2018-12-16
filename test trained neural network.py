
import numpy as np
import tensorflow as tf 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.optimizers import Adam
import matplotlib.pyplot as plt 

##########

n = 500
np.random.seed(0) #pour que les random numbers restent fixés
#inputed_data
X1 = np.array([np.random.normal(13, 2, n),np.random.normal(12, 2, n)]).T #classe 0
X2 = np.array([np.random.normal(8, 2, n), np.random.normal(6, 2, n)]).T #classe 1

X = np.vstack((X1, X2)) #mettre les deux classe d'entré dans seule array
y = np.matrix(np.append(np.zeros(n), np.ones(n))).T # label c'est la sortie (target)qu'on veut predicter 

plt.scatter(X[:n,0], X[:n,1])
plt.scatter(X[n:,0], X[n:,1])

#############

#sequentiel : permet la creation de la neseau de neurone
model =Sequential()
# units : nombre de noeud en sortie dans notre cas =1
#input_shape: on 2 noeuds en entrée
#activation: la fonction d'activation en sortie est sigmoid =(1/(1+exp(-x)))
#Dense : permer la liaison entre les noeuds  
model.add(Dense(units = 1, input_shape=(2, ), activation='sigmoid'))
adam = Adam(lr =0.1)

##################

#loss: erreur binary_crossentropy (seule noeud en sortie)
#metrics: permet de choisir les parametres qu'on veut verifier/afficher dans notre cas en prend accuracy 
model.compile(adam, loss = 'binary_crossentropy', metrics=['accuracy'])
#model.fit :Permets l'entrainement de reseau 
# x= : inputed_data / y= c'est le target / verbose=1 : afficher les détailles d'entrainement
#epochs=500: nombre de fois toute la donné (inputed_data) va étre traiter par le réseau 
#batch_size = c'est la nombre d'exemplaire de données afin de mesurer l'erreur et faire la correction des parametres de réseau
#dans notre cas on fait 20 iteration pour atteindre un epoch
#shuffle=true: à la fin de chaque epoch on fait le mixage de données d'entrée 
h=model.fit(x=X, y=y, verbose=1, epochs=500, batch_size=50, shuffle='true')


################

plt.plot(h.history['acc'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy'])


plt.plot(h.history['loss'])
plt.title('loss')
plt.xlabel('epoch')
plt.legend(['loss'])

##################

def plot_descision_limite(X, y, model):# permet de donner une indication par coleurs les zones de chaque classe
    x_span = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1) # prendre 50 points linéarement équidistant sur l'axe x horizantale
    y_span = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1) # prendre 50 points linéarement équidistant sur l'axe y verticale
    #print(y_span)
    xx, yy=np.meshgrid(x_span, y_span) 
    # xx est un array de deux dimensions (50, 50) dont tous les lignes sont les memes (x_span)
    # yy est un array de deux dimensions (50, 50) dont tous les colones sont les memes (y_span)
    #print (xx)
    #print (yy)
    xx_, yy_=xx.ravel(), yy.ravel() # conversion en une seule dimension
    #print(yy_)
    #print(xx_)
    grid = np.c_[xx_, yy_] # concatination de deux array chaque 50 elements de xx_ correspond à une valeur de yy-
    #print(grid)
    
    #le but de tous ces fonctions est de préparer un matrice qui contient plusieurs combinaison possibles entre l'axe x et l'axe y 
    #afin de tester les predictions de réseau de neurones 
    
    pred=model.predict(grid) #prediction
    A=pred.reshape(xx.shape) # Redimensionnement en un array de deux dimension 
    print(A)
    plt.contourf(xx, yy, A) # tracage des contours selon les valeurs de prediction 


#############

plot_descision_limite(X, y, model)
plt.scatter(X[:n,0], X[:n,1])
plt.scatter(X[n:,0], X[n:,1])

