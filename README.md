# Nuwa Project

## Introduction
This project is an open source project that aiming to show the capability of Google Large Language Model. It includes two scenario: NPC powered by Google LLM and Knowledge Search powered by Google LLM.

## GCP Services
This project uses PaLM API for generation, embedding and Vector Search (aka. Matching Engine) for embedding retrieval, and Bigtable for raw texts storage.
The frontend and backend are hosted on Cloud Run.

- Cloud Run
- PaLM and embedding
- Vector Search (Matching Engine)
- Bigtable

## Steps

1. step into story folder, and use Colab to open the notebook. Run the notebook to create Vector Search and Bigtable, and transform the story texts into embedding and store them into Vector Search for later retrieval.

2. Build docker image based on Dockerfile in backend folder

3. Push the docker image to Artifact Registry
```
gcloud auth configure-docker
docker push $IMAGE_URI
```

4. Create Cloud Run and deploy the docker image to Cloud Run
```
gcloud run deploy SERVICE_NAME --image IMAGE_URL
```

5. Get backend server url from Cloud Run, configure it in app.py in frontend folder. And create a Cloud Run job using same steps as above.

## Data
Data are configured in content folder. You should configure:
- world.txt: world settings
- new_hero_list_en.json: hero bio
- hero_name.txt: hero list
- hero.txt: hero list that will be requested from frontend
- story/story.txt: story background
- pre_conversation_chat/Jackie_Welles.json: talking style
- pre_conversation/Jackid_Welles.json: talking styles in another format

Currently the data is open source CyberPunk 2047, You can change it to your own data.