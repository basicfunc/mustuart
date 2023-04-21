import requests

def identify(song):

    # Set the API endpoint and your API key
    endpoint = "https://api.audd.io/"
    api_key = "23078212abd500d3c1ed64966e59be9b"

    # Send an HTTP POST request to the API endpoint with the audio file and your API key
    response = requests.post(endpoint + "?method=recognize&api_token=" + api_key,
                             files={"file": open(song, "rb")})

    
    json = response.json()
    print(json)
    artist = json['result']['artist']
    # print(artist)
    # Define API endpoint and parameters
    endpoint = "https://musicbrainz.org/ws/2/"
    params = {
        "query": artist,  # Replace with your desired artist name
        "fmt": "json",
        "limit": 5  # Number of similar artists to retrieve
    }

    # Send API request and get response
    response = requests.get(endpoint + "artist", params=params)

    # Parse JSON response and extract similar artists
    similar_artists = []
    if response.status_code == 200:
        data = response.json()
        for artist in data['artists']:
            if 'score' in artist and 'name' in artist:
                similar_artists.append(artist['name'])
        print(similar_artists)
    else:
        print("Error: Could not retrieve similar artists.")

    endpoint = "https://musicbrainz.org/ws/2/"
    
    for artist_name in similar_artists:
        params = {
            "query": "artist:" + artist_name,
            "fmt": "json",
            "limit": 1,
            "inc": "releases"  # Include release information in response
        }
            
        # Send API request and get response
        response = requests.get(endpoint + "artist", params=params)

        # Parse JSON response and extract latest song
        latest_song = ""
        if response.status_code == 200:
            data = response.json()
            if 'releases' in data['artists'][0]:
                releases = data['artists'][0]['releases']
                latest_release = max(releases, key=lambda r: r['date'])
                if 'title' in latest_release:
                    latest_song = latest_release['title']
            print("Latest song by " + artist_name + ": " + latest_song)
        else:
            print("Error: Could not retrieve latest song.")

    # Print the API response
    # print(response.json()['result']['artist'])
    # print(response.json()['result']['title'])


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        raise Exception("Wrong arguments passed.")
    else:
        identify(sys.argv[1])