import requests
import json
import pickle
import pandas as pd

""" 
Script used to fetch the meta-data of the GIFs from the GIFGIF media lab (including img url). 
The data retrieved is based on the user-specified configuration of the script.

In order to access the API, a private key is read from file. This file should be configured manually and 
be kept secret; an example is included in 'api_key_sample.json'.

See 'http://gifgif.media.mit.edu/labs/api' for full details on API usage.
"""


def main():
    # ----------------------------------------
    # Configuration
    # ----------------------------------------

    # Read the private API key from file.
    api_key_filepath = 'api_key.json'
    with open(api_key_filepath, 'rb') as f:
        api_key = json.load(f).get('api_key')
        if api_key is None:
            raise ValueError("Can't load API key. Check if '{}' exists and is configured.".format(api_key_filepath))

    # Parameters used to query the API.
    api_url = 'https://www.qnt.io/api/results'
    payload = {'pID': 'gifgif', 'limit': 5000, 'sort': 1, 'key': api_key, 'mID': None}

    # ----------------------------------------
    # Fetch mID
    # ----------------------------------------

    # Each category on GIFGIF has an associated ID with it. Grab these first so we can query the API properly.
    # e.g. amusement -> 54a309ae1c61be23aba0da53
    api_query_metrics_url = 'https://www.qnt.io/api/displaymetrics?pID=gifgif&mode=all&key={api_key}'.\
        format(api_key=api_key)

    response = requests.get(api_query_metrics_url)
    if response.status_code != requests.codes.ok:
        raise requests.ConnectionError()

    # Create a dictionary which maps each category to their respective mID's.
    response_content = json.loads(response.content)
    metric_to_mid_dict = dict([(x['metric'], x['mID']) for x in response_content])

    # ----------------------------------------
    # Fetch top-N results from each category
    # ----------------------------------------
    fetched_data = {}
    for (category, mid) in metric_to_mid_dict.items():
        payload['mID'] = mid

        response = requests.get(api_url, params=payload)
        if response.status_code != requests.codes.ok:
            raise requests.ConnectionError()

        response_content = json.loads(response.content)
        fetched_data[category] = response_content['results']

    # ----------------------------------------
    # Save to file for later use
    # ----------------------------------------
    df = pd.DataFrame(fetched_data)
    filename = './gif_metadata.p'
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
    print('Saved fetched data to: [{}]'.format(filename))
    print(df)


if __name__ == '__main__':
    main()
