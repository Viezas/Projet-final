import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense

def categorical(df, column):
    df = df.dropna()
    liste_ = list(df[column].value_counts().index)
    df[column] = df[column].apply(lambda x: liste_.index(x))
    return df

def train_model(data):
    # Read data
    df = pd.read_csv('data.csv')

    # Cleaning data
    df = clean_df(df)

    y = df.time_spent
    X = df.drop(['time_spent'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Sequential()
    model.add(Dense(30, input_dim=len(X.columns), activation='sigmoid'))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(5, activation='softmax'))
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
    history = model.fit(X_train,y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test))
    
    # Sauvegarder le modèle au format SavedModel
    save_model(model, 'model.h5')
    pass    

def clean_df(df):
    print(df)
    df = categorical(df, 'gender')
    df = categorical(df, 'platform')
    df = categorical(df, 'interests')
    df = categorical(df, 'location')
    df = categorical(df, 'demographics')
    df = categorical(df, 'profession')
    df = categorical(df, 'indebt')
    df = categorical(df, 'isHomeOwner')
    df = categorical(df, 'Owns_Car')
    return df