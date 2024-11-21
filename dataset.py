import pandas as pd

# Define your data
data1 = {
    'location': ['imadol'],
    'size': [3],
    'bedrooms': [7],
    'bathroom': [2],
    'floor': [3.5],
    'year_of_construction': [2072],
    'ft_road': [9],
    'price': 22000000
}

data2 = {
    'location': ['lagankhel'],
    'size': [6],
    'bedrooms': [10],
    'bathroom': [3],
    'floor': [3],
    'year_of_construction': [2075],
    'ft_road': [12],
    'price': 40000000
}

data3 = {
    'location': ['hattiban'],
    'size': [3],
    'bedrooms': [5],
    'bathroom': [2],
    'floor': [2],
    'year_of_construction': [2072],
    'ft_road': [10],
    'price': 24000000
}
data4 = {
    'location': ['imadol'],
    'size': [5],
    'bedrooms': [11],
    'bathroom': [5],
    'floor': [3],
    'year_of_construction': [2080],
    'ft_road': [13],
    'price': 57000000
}
data5 = {
    'location': ['imadol'],
    'size': [3],
    'bedrooms': [4],
    'bathroom': [2],
    'floor': [1],
    'year_of_construction': [2076],
    'ft_road': [8],
    'price': 16000000
}
data6 = {
    'location': ['hattiban'],
    'size': [4],
    'bedrooms': [8],
    'bathroom': [4],
    'floor': [3],
    'year_of_construction': [2080],
    'ft_road': [12],
    'price': 35500000
}
data7 = {
    'location': ['imadol'],
    'size': [3.1],
    'bedrooms': [7],
    'bathroom': [4],
    'floor': [2.5],
    'year_of_construction': [2081],
    'ft_road': [13],
    'price': 30000000
}
data8 = {
    'location': ['imadol'],
    'size': [5],
    'bedrooms': [9],
    'bathroom': [5],
    'floor': [2.5],
    'year_of_construction': [2081],
    'ft_road': [13],
    'price': 57500000
}

# Create DataFrames for each set of data
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
df4 = pd.DataFrame(data4)
df5 = pd.DataFrame(data5)
df6 = pd.DataFrame(data6)
df7 = pd.DataFrame(data7)
df8 = pd.DataFrame(data8)


# Concatenate DataFrames
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

# Save DataFrame to a CSV file
df.to_csv('house_prices.csv', index=False)
