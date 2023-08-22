import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, ConvLSTM2D, Dropout, Flatten, RepeatVector, Reshape, \
    TimeDistributed, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping


def getAvgRuntimeWithSegment(dataframe, time_step):
    dataframe = dataframe.copy()

    # Convert the 'date' and 'start_time' columns to datetime objects
    dataframe['start_time'] = pd.to_datetime(dataframe['start_time'])
    # Set the start and end times for the time window
    start_time = pd.to_datetime('06:00:00').time()
    end_time = pd.to_datetime('19:00:00').time()

    # Create a list to store the data for the new dataframe
    data = []

    # Iterate over the unique dates in the 'date' column
    for date in dataframe['date'].unique():
        # Filter the rows for the current date
        df_date = dataframe[dataframe['date'] == date]
        df_date.loc[:, 'start_time'] = df_date['start_time'].dt.time  # Use .loc to modify the original DataFrame

        # Create a time range for the current date with the specified time step
        time_range = pd.date_range(date, date + pd.Timedelta(days=1), freq=f'{time_step}T')

        # Iterate over the time range
        for start, end in zip(time_range[:-1], time_range[1:]):
            # Check if the start time is within the specified time window
            if start.time() >= start_time and start.time() < end_time:
                # Filter the rows for the current time window
                mask = (df_date['start_time'] >= start.time()) & (df_date['start_time'] < end.time())
                df_time_window = df_date[mask]
                # Calculate the average run time for each segment in the current time window
                avg_run_time = df_time_window.groupby('segment')['run_time_in_seconds'].mean().reset_index()

                # Append the data to the data list
                for row in avg_run_time.itertuples():
                    data.append((date, start.time(), row.segment, row.run_time_in_seconds))

    # Create a new dataframe from the data list
    df_avg_run_time = pd.DataFrame(data, columns=['date', 'start_time', 'segment', 'avg_run_time'])
    return df_avg_run_time


# To handle outliers in the avg_run_time column of your DataFrame using the absolute deviation around the median method, you can follow these steps:
#
#     Calculate the median of the avg_run_time column.
#     Calculate the absolute deviations of each value from the median.
#     Calculate the median absolute deviation (MAD) using the absolute deviations.
#     Set a threshold (for example, a multiple of MAD) to identify outliers.
#     Replace the outlier values with a suitable value (e.g., the median).

def handleOutliers(df):
    # Calculate the median of avg_run_time
    median_avg_run_time = df['avg_run_time'].median()

    # Calculate the absolute deviations
    df['abs_deviation'] = np.abs(df['avg_run_time'] - median_avg_run_time)

    # Calculate the median absolute deviation (MAD)
    mad = df['abs_deviation'].median()

    # Set a threshold (for example, 3 times MAD) to identify outliers
    threshold = 3 * mad

    # Replace outliers with the median value
    df.loc[df['abs_deviation'] > threshold, 'avg_run_time'] = median_avg_run_time

    # Drop the temporary column
    df = df.drop(columns=['abs_deviation'])

    return df


def fillTimeSteps(df_avg_run_time, startTime, endTime, time_step, num_segments):
    start_Time = pd.to_datetime(startTime).time()

    end_Time = pd.to_datetime(endTime).time()
    # Create a new DataFrame to store the missing data
    data = []

    # Iterate over the unique dates in the 'date' column
    for date in df_avg_run_time['date'].unique():
        # Filter the rows for the current date
        df_date = df_avg_run_time[df_avg_run_time['date'] == date]

        # Create a time range for the current date with the specified time step
        time_range = pd.date_range(date, date + pd.Timedelta(days=1), freq=f'{time_step}T').time

        # Iterate over the time range
        for t in time_range:
            # Check if the start time is within the specified time window
            if start_Time <= t < end_Time:
                # Check if the current start time is present in the dataframe
                if not (df_date['start_time'] == t).any():
                    # Add rows for the missing start time and segments
                    for segment in range(1, num_segments + 1):
                        data.append((date, t, segment, 0))

    # Create a new DataFrame from the data list
    df_missing = pd.DataFrame(data, columns=['date', 'start_time', 'segment', 'avg_run_time'])

    # Concatenate the new DataFrame with the original DataFrame
    df_avg_run_time = pd.concat([df_avg_run_time, df_missing], ignore_index=True)
    # Sort the rows of the dataframe by the 'date', 'start_time', and 'segment' columns
    df_avg_run_time = df_avg_run_time.sort_values(by=['date', 'start_time', 'segment'])
    return df_avg_run_time


def normalizeSTD(df, num_segments, start_time, end_time, time_step, historical):
    # Filter out zero values when calculating mean for each segment, start time, and weekday
    mean_data = historical.groupby(['segment', 'start_time', df['date'].dt.weekday])['avg_run_time'].mean()
    # Filter out zero values when calculating standard deviation for each segment
    std_data = historical.groupby('segment')['avg_run_time'].std()

    # Create a MultiIndex with all combinations of date, start_time, and segment
    idx = pd.MultiIndex.from_product([df['date'].unique(), df['start_time'].unique(), range(1, num_segments + 1)],
                                     names=['date', 'start_time', 'segment'])

    # Reindex the dataframe using the new MultiIndex
    df = df.set_index(['date', 'start_time', 'segment']).reindex(idx, fill_value=0).reset_index()

    df = fillTimeSteps(df, start_time, end_time, time_step, num_segments)

    # Normalize values and create a new DataFrame
    normalized_data = []
    for index, row in df.iterrows():
        segment, start_time, weekday = row['segment'], row['start_time'], row['date'].weekday()
        mean_value = mean_data.get((segment, start_time, weekday), 0)  # Default to 0 if not found
        std_value = std_data.get(segment, 1)  # Default to 1 if not found to avoid division by zero

        # If the value is non-zero, calculate normalized value; otherwise, keep it as zero
        normalized_value = (row['avg_run_time'] - mean_value) / std_value if row['avg_run_time'] != 0 else 0
        normalized_data.append((row['date'], row['start_time'], segment, normalized_value))

    normalized_columns = ['date', 'start_time', 'segment', 'normalized_avg_run_time']
    df_normalized = pd.DataFrame(normalized_data, columns=normalized_columns)
    return df_normalized


def prepareModelInput(df_avg_run_time, w, s, pred_steps):
    num_samples = len(df_avg_run_time) - (pred_steps - 1)
    # Initialize an empty array to hold the input data
    input_data_x = np.empty((num_samples, w, s, 1))
    input_data_y = np.empty((num_samples, pred_steps, s, 1))

    for i in range(len(df_avg_run_time) - ((w + pred_steps) * s)):
        # Use array slicing to extract the required data
        sample_data_x = df_avg_run_time.iloc[i:i + w * s, 3].values
        sample_data_y = df_avg_run_time.iloc[i + (w * s):i + (w * s) + pred_steps * s, 3].values
        reshaped_data_x = sample_data_x.reshape(w, s, 1)
        reshaped_data_y = sample_data_y.reshape(pred_steps, s, 1)
        input_data_x[i] = reshaped_data_x
        input_data_y[i] = reshaped_data_y
    return input_data_x, input_data_y


def splitData(input_data_x, input_data_y, test_size):
    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(input_data_x, input_data_y, test_size=test_size, shuffle=False)
    return x_train, x_test, y_train, y_test


def build_model(input_timesteps, output_timesteps, num_links):
    model = Sequential()
    model.add(BatchNormalization(name='batch_norm_0', input_shape=(input_timesteps, num_links, 1, 1)))
    model.add(ConvLSTM2D(name='conv_lstm_1', filters=64, kernel_size=(10, 1), padding='same', return_sequences=True))
    model.add(Dropout(0.21, name='dropout_1'))
    model.add(BatchNormalization(name='batch_norm_1'))
    model.add(ConvLSTM2D(name='conv_lstm_2', filters=64, kernel_size=(5, 1), padding='same', return_sequences=False))
    model.add(Dropout(0.20, name='dropout_2'))
    model.add(BatchNormalization(name='batch_norm_2'))
    model.add(Flatten())
    model.add(RepeatVector(output_timesteps))
    model.add(Reshape((output_timesteps, num_links, 1, 64)))
    model.add(ConvLSTM2D(name='conv_lstm_3', filters=64, kernel_size=(10, 1), padding='same', return_sequences=True))
    model.add(Dropout(0.20, name='dropout_3'))
    model.add(BatchNormalization(name='batch_norm_3'))
    model.add(ConvLSTM2D(name='conv_lstm_4', filters=64, kernel_size=(5, 1), padding='same', return_sequences=True))
    model.add(TimeDistributed(Dense(units=1, name='dense_1', activation='relu')))

    optimizer = RMSprop()
    model.compile(loss="mse", optimizer=optimizer)

    return model


def trainModel(model, input_data_x, input_data_y, batch_size, epochs, validation_split):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
    history = model.fit(input_data_x, input_data_y, batch_size=batch_size, epochs=epochs,
                        validation_split=validation_split, callbacks=[early_stopping], verbose=1)
    return history


def evaluateModel(model, x_test, y_test):
    # Make predictions using the test data
    y_pred = model.predict(x_test)

    # Calculate the root mean squared error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse
