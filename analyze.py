from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

sentiment_pipeline = pipeline("sentiment-analysis")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

EMAIL_CLASSES = [
    "Work", "Sports", "Food"
]


def get_sentiment(text):
    response = sentiment_pipeline(text)
    return response


'''Check to see if there are new email from user.
If not then load the default EMAIL_CLASSES. If there are new email then update it with the new email classes'''
def update_email(NEW_CLASSES):
    global EMAIL_CLASSES
    if NEW_CLASSES not in EMAIL_CLASSES:
        EMAIL_CLASSES.append(NEW_CLASSES)
    # update_class_embeddings = compute_embeddings(EMAIL_CLASSES) #TypeError: Object of type zip is not JSON serializable-> convert zip file to json file

    update_class_embeddings = compute_embed(EMAIL_CLASSES)
    # print("Updated Embeddings:", update_class_embeddings)
    print("New class to add:", NEW_CLASSES)
    print("Updated EMAIL_CLASSES:", EMAIL_CLASSES)
    if update_class_embeddings is None:
        update_class_embeddings = {}
    return update_class_embeddings

# Convert embedded zip to list of dictionary to solve TypeError: Object of type zip is not JSON serializable
def compute_embed(embeddings=EMAIL_CLASSES):
    embeddings = model.encode(embeddings)
    embedded_dict = {}
    for k, v in enumerate(EMAIL_CLASSES):
        embedded_dict[v] = embeddings[k].tolist()
    return embedded_dict


def compute_embeddings(embeddings=EMAIL_CLASSES):
    embeddings = model.encode(embeddings)
    return zip(EMAIL_CLASSES, embeddings)


def classify_email(text):
    # Encode the input text
    text_embedding = model.encode([text])[0]

    # Get embeddings for all classes
    class_embeddings = compute_embeddings()

    # Calculate distances and return results
    results = []
    for class_name, class_embedding in class_embeddings:
        # Compute cosine similarity between text and class embedding
        similarity = np.dot(text_embedding, class_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(class_embedding))
        results.append({
            "class": class_name,
            "similarity": float(similarity)  # Convert tensor to float for JSON serialization
        })

    # Sort by similarity score descending
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results
