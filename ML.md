
# Maching Learning - Classification d'images
Dans cet atelier, vous allez découvrir comment utiliser TensorFlow pour créer un modèle de classification d'images à partir de zéro et le déployer à l'aide de Flask.
Vous apprendrez comment prétraiter les données d'images, construire un modèle de réseau de neurones, entraîner le modèle et l'évaluer. Ensuite, vous apprendrez à construire une interface web avec Flask qui permettra aux utilisateurs de télécharger une image et d'obtenir la prédiction du modèle de classification d'images en temps réel. Et par la suite comment dockeriser l'application dans l'atelier suivant .

Structure du projet : 

```
.
├── Deploy_Modele |── static |── css                   
|                 |          ├── js
|                 |          ├── modele ── Modele.H5
|                 |          └── upload
|                 |
|                 ├── templates  ├── base.html
|                 |              └── index.html                
|                 └── app.py                 
|                  
|                 
└── Dev_Modele|── modele.py 

```

# Développement du modèle

## 1 . Importation des bibliothèques 

Pour notre application on aura besoin de trois bibliothèques : 
* Tenserflow : Bibliothèque machine learning crée par google , bibliothèque logicielle open source pour l'apprentissage automatique et l'intelligence artificielle permet aux utilisateurs de construire des modèles .
* Numpy : Bibliothèque de Manipulation de matrice .
* Matplotlib : Bibliothèque de manipulation de graphe .
```
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
```
## 2 . Création du Dataset 

Dans ce code, la variable fashion_mnist est définie pour contenir le jeu de données Fashion MNIST à l'aide de la bibliothèque TensorFlow. La fonction load_data() est ensuite appelée sur cette variable pour charger les données en mémoire.

Ainsi, les instructions (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() chargent les images et les étiquettes d'entraînement et de test dans les variables train_images, train_labels, test_images et test_labels. Ces variables peuvent être utilisées pour entraîner et évaluer un modèle de classification d'images.

```
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```
Ces deux lignes de code permettent d'afficher la forme (shape) de train_images et la première image du jeu de données d'entraînement.

```
print(train_images.shape)
print(train_images[0])
```
Les données doivent être prétraitées avant de former le réseau. Si vous inspectez la première image de l'ensemble d'apprentissage, vous verrez que les valeurs de pixel se situent dans la plage de 0 à 255 :
```
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```
Pour vérifier que les données sont au format correct et que vous êtes prêt à construire et former le réseau, affichons les 25 premières images de l' ensemble de formation et affichons le nom de la classe sous chaque image.
```
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```
Pour ce faire, divisez les valeurs par 255. Il est important que l' ensemble d'apprentissage et l' ensemble de test soient prétraités de la même manière :
```
train_images = train_images / 255.0
test_images = test_images / 255.0
```
## 3 . Création du modèle 

Ce code définit un modèle de réseau de neurones séquentiel à l'aide de la bibliothèque TensorFlow.
* La première couche (Flatten) transforme l'entrée en un vecteur 1D de longueur 28*28=784 pixels. Cette couche sert à "aplatir" l'image en une seule dimension pour    qu'elle puisse être traitée par les couches suivantes.
* La deuxième couche (Dense) est une couche entièrement connectée de 128 neurones avec une fonction d'activation ReLU (rectified linear unit). Cette couche sert à apprendre des caractéristiques à partir des pixels de l'image.
* La troisième couche (Dense) est une couche entièrement connectée de 10 neurones qui représente les 10 classes d'images possibles dans le jeu de données. 
```
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```
Avant que le modèle ne soit prêt pour l'entraînement, il a besoin de quelques réglages supplémentaires. Ceux-ci sont ajoutés lors de l'étape de compilation du modèle :

* Fonction de perte : mesure la précision du modèle pendant l'entraînement. Vous souhaitez minimiser cette fonction pour "orienter" le modèle dans la bonne direction.
* Optimiseur : c'est ainsi que le modèle est mis à jour en fonction des données qu'il voit et de sa fonction de perte.
* Métriques : utilisées pour surveiller les étapes de formation et de test. L'exemple suivant utilise precision , la fraction des images qui sont correctement classées.
```
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
## 4 . Former le modèle

Pour commencer l'entraînement, appelez la méthode model.fit , ainsi appelée car elle "adapte" le modèle aux données d'entraînement 
L'argument epochs=10 indique le nombre d'itérations à effectuer sur l'ensemble des données d'entraînement.
```
model.fit(train_images, train_labels, epochs=10)
```
![image](https://user-images.githubusercontent.com/123757632/223147905-a6bfffa3-607b-4395-b708-61425bafc052.png)

## 5 . Tester le modèle 

Comparez les performances du modèle sur l'ensemble de données de test :
```
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
```
![image](https://user-images.githubusercontent.com/123757632/223148025-3e10cc5f-b83b-4283-9cec-e15f9328e4ec.png)

Avec le modèle formé, vous pouvez l'utiliser pour faire des prédictions sur certaines images :
```
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]
```
Représentez-le graphiquement pour examiner l'ensemble complet des 10 prédictions de classe : 

```
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
```
Avec le modèle formé, vous pouvez l'utiliser pour faire des prédictions sur certaines images.

Regardons la 0ème image, les prédictions et le tableau de prédiction. Les étiquettes de prédiction correctes sont bleues et les étiquettes de prédiction incorrectes sont rouges. Le nombre donne le pourcentage (sur 100) pour l'étiquette prédite.


```
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
```
![image](https://user-images.githubusercontent.com/123757632/223148247-385cf478-10c5-44f8-840c-cf7a22464a1d.png)

Traçons plusieurs images avec leurs prédictions. Notez que le modèle peut se tromper même lorsqu'il est très confiant.
```
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/123757632/223148390-d5859ea1-33ab-4235-a524-923d0dc88048.png)

## 6 . Sauvegarder le modèle

Cette ligne de code sauvegarde le modèle entraîné en format HDF5. Le format HDF5 est un format de fichier de données qui prend en charge la gestion des données volumineuses et complexes.
```
model.save('../Deploy_Model/static/modele/Modele.h5')
print('Modèle sauvegardé avec succès !')
```

# Déploiement du modèle 

Pour notre application on aura besoin de plusieurs bibliothèques : 
* Le module os (système d'exploitation) fournit des fonctions et des variables qui vous permettent d'interagir avec le système d'exploitation dans lequel votre code Python s'exécute.
* Numpy bibliothèque de Manipulation de matrice
* Keras est une bibliothèque (Deep learning) de réseaux de neurones de haut niveau pour Python 
* Flask est un framework web pour Python qui vous permet de deployer rapidement des applications
* Werkzeug fournit des outils pour gérer les requêtes HTTP et les réponses
* cv2 la bibliothèque OpenCV vous permet d'utiliser ses fonctions pour le traitement d'images et de vidéos.
```
import os
import numpy as np
from keras.models import load_model
import keras.utils as image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2
```
Avant de commencer notre application on doit configurer bibliothèque cv2 , pour cela cliquez sue : File > Preferences > settings 
![image](https://user-images.githubusercontent.com/123757632/223664080-56e18291-c6ef-4110-9c1c-cd7a2c7cee00.png)
Puis Cliquez sur l'

Definir que notre application est une application flask

```
app = Flask(__name__)
```
Cette ligne de code définit le chemin d'accès au fichier qui contient le modèle Keras entraîné que vous souhaitez charger
```
MODEL_PATH = './static/modele/Modele.h5'
```
Cette ligne de code charge le modèle Keras à partir du chemin de fichier spécifié et le stocke dans la variable model.
```
model = load_model(MODEL_PATH)
```
Cette fonction prend en entrée le chemin d'accès à une image et un modèle Keras, et elle retourne les prédictions du modèle pour cette image. 

La fonction commence par charger l'image à partir du chemin spécifié en utilisant la fonction image.load_img de Keras, puis elle la convertit en un tableau NumPy en utilisant la fonction np.asarray.

Elle convertit ensuite l'image en niveaux de gris en utilisant la fonction cv2.cvtColor de la bibliothèque OpenCV. 
Ensuite, la fonction normalise l'image en divisant tous les pixels par 255, puis elle ajoute une dimension à l'image en utilisant la fonction np.expand_dims pour qu'elle corresponde au format d'entrée attendu par le modèle Keras.

Enfin, la fonction effectue les prédictions du modèle en utilisant la méthode model.predict, puis elle retourne les prédictions.
```
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(28, 28)) 
    img_array = np.asarray(img)
    x = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    result = int(img_array[0][0][0])
    print(result)
    if result > 128:
      img = cv2.bitwise_not(x)
    else:
      img = x
    img = img/255#normalisée l'image
    img = (np.expand_dims(img,0)) 

    preds =  model.predict(img)#classer l'image
    print(preds)
    return preds
```
Cette fonction est une route Flask pour l'URL de base (/) de l'application. Elle utilise la méthode HTTP GET pour renvoyer la page HTML index.html en utilisant la fonction render_template fournie par Flask.
```
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

```
Cette fonction est une route Flask pour l'URL /predict, qui gère à la fois les demandes HTTP GET et POST. Lorsque l'application reçoit une demande POST, elle traite une image qui a été soumise via un formulaire HTML, enregistre l'image sur le disque dur, effectue des prédictions en utilisant un modèle Keras et renvoie la classe prédite.
```
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__) #basepath:chemin de base du fichier actuel
        file_path = os.path.join(
            basepath, './static/upload', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        predicted_label = np.argmax(preds)
        result = class_names[predicted_label]
        return result
    return None

```
Pour exécuter le serveur web Flask.
```
if __name__ == '__main__':
    app.run(debug=True)
```
