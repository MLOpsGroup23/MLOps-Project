from google.cloud import firestore
import pandas as pd

class CustomFirestoreClient(firestore.Client):
    def __init__(self, project=None, credentials=None, database=None, collection=None):
        super().__init__(project=project, credentials=credentials, database=database)

        self.collection_ref = self.collection(collection)

    def get_pd(self):
        # fetch documents from Firestore
        docs = self.collection_ref.stream()
        # Fetch documents from Firestore

        # Convert documents to a list of dictionaries
        data = [doc.to_dict() for doc in docs]

        # Create a Pandas DataFrame
        df = pd.DataFrame(data)

        return df

    def add_prediction(self, data):

        # Add a document with an auto-generated ID
        new_doc_ref = self.collection_ref.document()
        new_doc_ref.set(data)

        print("New document has been created with ID:", new_doc_ref.id)
        
        return True