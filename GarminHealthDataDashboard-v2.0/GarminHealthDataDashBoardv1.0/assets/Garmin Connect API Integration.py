def load_garmin_data(self, metric):
    """Load real data from Garmin Connect API"""
    # Example implementation
    import requests
    
    # Your Garmin API credentials and endpoints
    headers = {'Authorization': f'Bearer {your_token}'}
    endpoint = f"https://healthapiapi.garmin.com/wellness-api/rest/{metric}"
    
    response = requests.get(endpoint, headers=headers)
    data = response.json()
    
    # Convert to pandas Series with timestamps
    return pd.Series(data['values'], index=pd.to_datetime(data['timestamps']))
