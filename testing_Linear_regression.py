import numpy as np
import pandas as pd
import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Importing dataset
def main():
    df = pd.read_csv( "salary_data.csv" )

    X = df['YearsExperience'].values.reshape(-1,1)

    Y = df['Salary'].values
    # Splitting dataset into train and test set

    X_train, X_test, Y_train, Y_test = train_test_split( 
      X, Y, test_size = 1/3, random_state = 0 )
    
    # Model training
    
    model = LinearRegression.LinearRegression( iterations = 1000, learning_rate = 0.01 )

    model.fit( X_train, Y_train )
    
    # Prediction on test set

    Y_pred = model.predict( X_test )
    
    print( "Predicted values ", np.round( Y_pred[:3], 2 ) ) 
    
    print( "Real values      ", Y_test[:3] )
    
    print( "Trained W        ", round( model.m[0], 2 ) )
    
    print( "Trained b        ", round( model.c, 2 ) )
    
    print( "Model Score     ", model.score(X_test, Y_test) )
    # Visualization on test set 
    
    plt.scatter( X_test, Y_test, color = 'blue' )
    
    plt.plot( X_test, Y_pred, color = 'orange' )
    
    plt.title( 'Salary vs Experience' )
    
    plt.xlabel( 'Years of Experience' )
    
    plt.ylabel( 'Salary' )
    
    plt.show()

if __name__ == "__main__" : 
    
    main()
