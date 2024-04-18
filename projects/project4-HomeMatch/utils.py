import random
import json


# Generating more than one listing at a time hits the token limit. So this function will generate one at a time.
def generate_listing(listing_id, model):
    """Generates a single real estate listing"""

    example = {
        "listing_id": 1,
        "property_type": "house",
        "address": "2999 Maple Ave.",
        "neighborhood": "Kitslano",
        "price": 800000,
        "bedrooms": 2,
        "bathrooms": 2,
        "description": "Welcome to the best kept secret of downtown...",
        "neighborhood_description": "don't miss out on Knickle & Grant's last remaining build in Samara Heights...",
        "sqft": 1000 
    }
    
    # Help gernerate more diverse listings by randomly setting these parameters.
    property_type = random.choice(['townhouse', 'condo', 'house', 'mansion'])
    address_number = random.randint(0,10000)
    
    # Help set up the example so it has correct values
    example['listing_id'] = listing_id
    example['property_type'] = property_type
    
    
    example_string = json.dumps(example)
    prompt = f"""
    Can you generate a real estate listing in Vancouver Canada in JSON format. For example:
    
    {example_string}

    Make sure the listing_id is equal to {listing_id}.

    Use {address_number} as the address number with a randomly generated street name.
    
    Make sure the property_type is {property_type} and generate new values for the rest of the fields that are sensible for a {property_type}.

    Make sure the description field has at least 75 words.
    """

    response = model(prompt)

    listing = json.loads(response)

    return listing

def see_collection(collection):
    """
    Check function to list all collections, get details of a specific collection,
    and perform operations on the collection.
    """

    # List all collections
    print(f"Collection: {collection.name}")
    print(f"Number of Items: {collection.count()}")
    # Get all items in the collection
    all_items = collection.get()
    print("\nItems in Collection:")
    for item in all_items:
        print(item)

def load_listing_into_db(listing, collection):
    """
    Process real estate listings

    Args:
        listing (pandas.Row): A row for a pandas dataframe

    Usage:
        df.apply(load_listing_into_db, axis=1)
    """
    fields = ["property_type","address", "neighborhood", "price", "bedrooms", "bathrooms", "sqft"]
    
    # The description and the neightborhood description are combined into one document
    # This is the main field that will be queried by the vector db and rewritten by the llm.
    document = (
        listing.description + "\n" +
        listing.neighborhood_description
    )

    # Add metadata for the document
    metadata = {
        field: listing[field]
        for field in fields
    }

    # Add the record to the database collection
    collection.add(
        documents=document,
        metadatas=metadata,
        ids=str(listing.listing_id),
    )
    return True

def get_listings_from_query(user_query, collection, n_results=2):
    """
    Retrieve listings from a vector database based on a query and generate context-aware responses.

    Args:
        query (str): The query string to search for relevant listings.
        vector_db (langchain.vectorstores.chroma.Chroma): An instance of a vector database containing listing vectors.

    Returns:
        dict: A dictionary containing context-aware responses for the retrieved listings.

    Example:
        >>> query = "spacious apartment in downtown"
        >>> result = get_listings_with_query(query, vector_db)
    """
    similar = collection.query(query_texts=user_query, n_results=n_results)

    results = [
        {
            'listing_id': listing[0],
            'description': listing[1],
            **listing[2]
        } for listing in zip(similar['ids'][0], similar['documents'][0], similar['metadatas'][0])
    ]
    return results

def get_personalized_descriptions(description, questions, answers, model):
    """
    Generate a personalized property description based on user answers to questions.

    Args:
        description (str): Original real estate description to be personalized.
        questions (list): List of questions asked to the user.
        answers (list): List of answers provided by the user.
        model (llm): Language model used for text generation.

    Returns:
        str: Personalized property description tailored to the user.

    """
    prompt = f"""
        You are AI that will rewrite the description for a property based on a users answers to some questions.
        
        You've asked the following questions
        
        {questions}
        
        and received the following answers
        
        {answers}
        
        Construct a user persona for the user and personalize the following real estate description so that it appeals to the user:
        
        {description}
        
        Make sure to use the users name in the description so they feel special.
        
        Make sure to mention particulars about their answers to the questions and how it relates to the listing.
        
        Make sure you only return the personalized description as this will be used in a data object.
    """
    
    response = model(prompt)
    return response
    

def get_personalized_listings(questions, answers, model, collection):
    """
    Generate personalized property listings based on user answers to questions.

    Args:
        questions (list): List of questions asked to the user.
        answers (list): List of answers provided by the user.
        model (llm): Language model used for text generation.

    Returns:
        list: List of personalized property listings tailored to the user.

    """
    # We now get a listing for each of the answers given by the user, only the most relavant is returned.
    listings = [get_listings_from_query(answer, collection, n_results=1)[0] for answer in answers[1:]]

    # Now we use a llm to customize the description so it appeals to the user.
    for listing in listings:
        listing['description'] = get_personalized_descriptions(
            description=listing['description'],
            questions=questions,
            answers=answers,
            model=model)
    return listings