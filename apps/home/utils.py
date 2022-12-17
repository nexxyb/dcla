import json


def get_metadata_from_dataframe(df):
    # Get the metadata as a string
    metadata = df.info()
    
    # Convert the metadata string to JSON format
    metadata_json = json.loads(metadata)
    
    # Return the metadata in JSON format
    return metadata_json