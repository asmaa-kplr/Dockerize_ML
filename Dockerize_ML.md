# Dockeriser un modèle machine learning
Dans cet atelier, nous allons vous montrer comment utiliser Docker pour containeriser une application Flask, ce qui vous permettra de la déployer facilement et rapidement sur différentes plateformes, sans vous soucier des dépendances ou des configurations du système.

Pour dockeriser une application Flask, vous aurez besoin des éléments suivants :

* Docker : Vous devez installer Docker sur votre ordinateur pour créer et exécuter des conteneurs. Vous pouvez télécharger la dernière version de Docker à partir du site officiel https://www.docker.com/products/docker-desktop . 

* Requirements.txt : Vous devez créer un fichier "requirements.txt" pour définir les dépendances de votre application Flask. Ce fichier contiendra les bibliothèques Python nécessaires pour exécuter votre application Flask.

* Dockerfile : Vous devez créer un fichier Dockerfile pour définir votre image de conteneur. Le Dockerfile contient les instructions pour créer l'image de conteneur, y compris les dépendances, les bibliothèques et les configurations.


## 1 . Configuration du fichier Requirements.txt

Lorsque vous explorez un projet Python existant, vous pourriez constater la présence d'un fichier "requirements.txt". Ce fichier contient une liste de paquets et de dépendances nécessaires pour exécuter le projet, ainsi que leurs versions correspondantes.

En utilisant ce fichier, vous pouvez facilement installer les paquets nécessaires dans un autre projet. Pour ce faire, vous pouvez simplement copier le fichier "requirements.txt" dans le projet souhaité et exécuter la commande suivante :
```
pip install -r requirements.txt
```
L’avantage de cette méthode est que vous n’avez pas à exécuter plusieurs fois la commande pip install pour chaque paquet.

Vous pouvez créer un fichier requirements.txt en utilisant la commande suivante dans votre invite de commande ou votre terminal:
```
pip freeze > requirements.txt
```

Example "requirements.txt" : 
```
Flask==2.2.3
keras==2.11.0
matplotlib==3.7.0
numpy==1.24.2
opencv-python==4.7.0.72
tensorflow==2.11.
Werkzeug==2.2.3
```
Maintenant, après avoir créé notre fichier "requirements.txt", nous passons à l'étape suivante.

## 2 . Configuration du fichier Docker 

Les Dockerfiles sont des fichiers qui permettent de construire une image Docker adaptée à nos besoins, étape par étape. 

La première chose à faire dans un Dockerfile est de définir de quelle image vous héritez. Pour notre example on aura :
```
FROM python:3.8
``` 
FROM permet de définir notre image de base, vous pouvez l'utiliser seulement une fois dans un Dockerfile.

Chenger le répertoire courant de notre image : 
```
WORKDIR /app
```

Copier le "requirements.txt" de notre emplacement local vers le répertoire de notre image :
```
COPY ./requirements.txt  /app/requirements.txt
```

Puis on l'execute en utilisant le mot clé RUN : 
```
RUN pip install -r requirements.txt
```
On copie l'intégralité du projet d'application dans le répertoire "app" : 
```
COPY . .
```
Nous exposons le port 5000 car l'application s'exécutera sur le port 5000 : 
```
EXPOSE 5000
```
Définissez la variable d'environnement FLASK_APP. Sinon, l'interpréteur peut signaler qu'il est incapable de trouver la variable :
```
ENV FLASK_APP=app.py
```
Et pour finir saisissez la commande "flask run --host 0.0.0.0" pour lancer le serveur. Cela permet de s'assurer que le serveur accepte les requêtes de tous les hôtes :
```
CMD ["flask", "run", "--host=0.0.0.0"]
```

Dockerfile complet pour dockeriser une application Flask : 

```
FROM python:3.8
WORKDIR /app
COPY ./requirements.txt  /app/requirements.txt
RUN pip install -r requirements.txt
COPY  . .
EXPOSE 5000
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host=0.0.0.0"]
```

## 3 . Construire l’image Docker

Pour construire l'image Docker on executera la commande suivante : 

```
docker image build -t flask-docker .
```

![image](https://user-images.githubusercontent.com/123757632/222764555-5454a3c0-62f9-46d7-9e78-5b3340dd5537.png)

Une fois que l'image est construite avec succès, celle-ci doit figurer dans la liste des images Docker avec la commande :
```
docker images
```

![image](https://user-images.githubusercontent.com/123757632/222764926-5c81a0c0-7c91-4c0a-b1ec-dd5cf62203fe.png)

## 4 . Exécution de l'image Docker

Après avoir réussi à construire l’image, l’étape suivante consiste à exécuter une instance de l’image. Voici comment effectuer cette opération : 
```
docker run -p 5000:5000 flask-docker
```
Cette commande exécute le conteneur et son application intégrée, chacun sur le port 5000 en utilisant une approche de liaison de port. Le premier 5000 est le port que nous allouons au conteneur sur notre machine. Le second 5000 est le port sur lequel l’application sera exécutée sur le conteneur.

![image](https://user-images.githubusercontent.com/123757632/222969587-162dc8f6-0e9b-4781-bcf9-b48c89fa20af.png)

## 5 . Tagger et pousser une image Docker vers Docker Hub 

Taguer l'image avec la commande "docker tag". Cette commande permet de donner un nom à l'image, ainsi que le nom du compte Docker Hub et le nom du référentiel Docker Hub vers lequel vous voulez pousser l'image.

```
docker tag mon_image:latest mon_nom_utilisateur/docker-repo:tag
```

Pousser l'image taguée vers Docker Hub avec la commande "docker push".

```
docker push mon_nom_utilisateur/docker-repo:tag
```
