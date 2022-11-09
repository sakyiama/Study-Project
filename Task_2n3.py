import sys
import os
import pandas as pd
import scipy.stats as st
import re
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def Scaling_MinMax(dfTrain, dfTest):
    """
    Function to scale the pair of dataset using MinMax scaler function - It transforms all the features in dataset
    into the range from 0 to 1
    :param dfTrain: dataset of training file
    :param dfTest: dataset of testing file
    :return: Null
    """
    mScaler = MinMaxScaler()

    # Transforming train dataset and saving it to file
    aTrain = mScaler.fit_transform(dfTrain)
    mS_train_df = pd.DataFrame(aTrain)

    # Transforming test dataset and saving it to file
    Ans = input('Do you want to use same fit method on both train and test datasets? Type y or n')
    if Ans == 'y':
        aTest = mScaler.transform(dfTest)
    else:
        aTest = mScaler.fit_transform(dfTest)
    mS_test_df = pd.DataFrame(aTest)
    return [mS_train_df, mS_test_df]


def Scaling_Std(dfTrain, dfTest):
    """
    Function to scale the pair of dataset using Standard scaler function - It transforms all the features in dataset
    such that their distribution will have a mean value 0 and standard deviation of 1.
    :param dfTrain: dataset of training file
    :param dfTest: dataset of testing file
    :return: Null
    """
    sScaler = StandardScaler()

    # Transforming train dataset and saving it to file
    aTrain = sScaler.fit_transform(dfTrain)
    sS_train_df = pd.DataFrame(aTrain)

    # Transforming test dataset and saving it to file
    Ans = input('Do you want to use same fit method on both train and test datasets? Type y or n : ')
    if Ans == 'y':
        aTest = sScaler.transform(dfTest)
    else:
        aTest = sScaler.fit_transform(dfTest)
    sS_test_df = pd.DataFrame(aTest)
    return [sS_train_df, sS_test_df]


def ks_test(result_train_df, df_test_ks):
    sig_count = 0
    df_compare = pd.DataFrame(columns=result_train_df.columns)
    with open('KStest_output.txt', 'a') as f:
        for colName in result_train_df.columns.values:
            if not re.match(r"attack", colName):
                print('\nKS test for Measurement', colName, 'is:', file=f)
                result = st.kstest(result_train_df[colName], df_test_ks[colName])
                print(result, file=f)
                if result.statistic < 0.2 and result.pvalue > 0.05:
                    print('---------Distribution for', colName, 'passes the K-S test', file=f)
                    df_compare.at[0, colName] = 1
                    sig_count = sig_count + 1
        f.close()
    print(df_compare.to_string())
    return df_compare, sig_count


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Please provide dataset name and method')
        sys.exit()

    print('Task 2: Statistical Analysis of Physical Readings')

    dataSetName = sys.argv[1]
    methodName = sys.argv[2]
    print('Working on ' + dataSetName + ' and scaling method ' + methodName)

    path = '.\\HAI_dataset\\' + dataSetName + '\\'
    filenames = os.listdir(path)
    print('Available training and testing datasets:')
    print(filenames)
    fTrain = input('Please enter name of training dataset file:')
    fTest = input('Please enter name of testing dataset file:')
    print('You entered: ' + fTrain + ' and ' + fTest)

    df_train = pd.read_csv(path + fTrain, index_col=0, parse_dates=True)
    df_train.columns = df_train.columns.str.lower()

    df_test = pd.read_csv(path + fTest, index_col=0, parse_dates=True)
    df_test.columns = df_test.columns.str.lower()

    # Task2a: Feature scaling
    # Preparing data for scaling by removing columns related to attack values
    df_train_scaler = df_train.drop(columns=list(df_train.filter(regex='attack')))
    df_test_scaler = df_test.drop(columns=list(df_test.filter(regex='attack')))
    # Calling scaling functions
    if methodName == 'MinMax':
        result_train_df, result_test_df = Scaling_MinMax(df_train_scaler, df_test_scaler)
        print('Scaled data is stored in Train_MinMaxScaled and Test_MinMaxScaled')
    else:
        result_train_df, result_test_df = Scaling_Std(df_train_scaler, df_test_scaler)
        print('Scaled data is stored in Train_StandardScaled and Test_StandardScaled')

    result_train_df.columns = df_train_scaler.columns.str.upper()
    result_test_df.columns = df_test_scaler.columns.str.upper()

    result_train_df.index = df_train_scaler.index
    result_test_df.index = df_test_scaler.index

    result_train_df.to_csv('Train_' + methodName + 'Scaled.csv')  # Final dataframes after scaling
    result_test_df.to_csv('Test_' + methodName + 'Scaled.csv')

    # Task2b: K-S test
    # Dropping rows which had attack = 1
    df_test_ks = result_test_df.drop(df_test[list(df_test.attack == 1)].index)
    # Calling the K-S test function
    df_compare, sig_count = ks_test(result_train_df, df_test_ks)
    print('Output for KS test is stored in KStest_output.txt')

    # Task2c: System states

    # Creating actuator list
    if dataSetName == 'hai_20_07':
        listActuatorColNames = ['P2_AUTO', 'P2_ON']
    elif dataSetName == 'hai_21_03':
        listActuatorColNames = ['P1_PP01AR', 'P1_PP01BR', 'P1_PP02R', 'P1_STSP', 'P2_AUTOGO', 'P2_ONOFF']
    else:
        listActuatorColNames = ['P1_PP01AR', 'P1_PP01BR', 'P1_PP02R', 'P1_SOL01D', 'P1_SOL03D', 'P1_STSP',
                                'P2_ATSW_LAMP', 'P2_AUTOGO', 'P2_MASW', 'P2_ONOFF']

    # Getting the count of various states and printing it
    df_actuators_train = result_train_df.groupby(listActuatorColNames).size().reset_index().rename(columns={0: 'count'})
    print('The state(s) for training dataset is/are:')
    print(df_actuators_train.to_string())

    df_actuators_test = result_test_df.groupby(listActuatorColNames).size().reset_index().rename(columns={0: 'count'})
    print('The state(s) for testing dataset is/are:')
    print(df_actuators_test.to_string())

    df_both = pd.merge(df_actuators_train, df_actuators_test, how='outer', on=listActuatorColNames, indicator='exists')
    df_common = df_both.loc[df_both['exists'] == 'both']  # Filtering common states
    print(df_common.to_string())

    # count_train = df_actuators_train['count'].sum()
    # count_test = df_actuators_test['count'].sum()

    # Finding percentage of common system states
    c_trainc = (df_common['count_x'].sum() / df_actuators_train['count'].sum()) * 100
    c_testc = (df_common['count_y'].sum() / df_actuators_test['count'].sum()) * 100

    print('%.2f percentage of training system states are common' % c_trainc)
    print('%.2f percentage of testing system states are common' % c_testc)

    # Task 2d: K-S statistics for sensors

    df_common = df_common.drop(columns=['count_x', 'count_y', 'exists'])

    # Creating actuator list
    if dataSetName == 'hai_20_07':
        listSensorColNames = ['P1_FT01', 'P1_FT02', 'P1_FT03', 'P1_LIT01', 'P2_VYT02', 'P2_VXT02', 'P2_VYT03',
                              'P2_VXT03', 'P3_LT01']
        arr_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif dataSetName == 'hai_21_03':
        listSensorColNames = ['P1_FT01', 'P1_FT02', 'P1_FT03', 'P1_LIT01', 'P2_RTR', 'P3_LIT01', 'P3_PIT01']
        arr_stats = [0, 0, 0, 0, 0, 0, 0]
    else:
        listSensorColNames = ['P1_FT01', 'P1_FT02', 'P1_FT03', 'P1_LIT01', 'P2_RTR', 'P2_VIBTR01', 'P2_VIBTR02',
                              'P2_VIBTR03', 'P2_VIBTR04', 'P3_LIT01', 'P3_PIT01']
        arr_stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Performing K-S test
    for index, row in df_common.iterrows():
        result_train_copy = result_train_df
        result_test_copy = result_test_df
        i = 0
        for column in df_common:
            result_train_copy = result_train_copy[(result_train_copy[column] == row[column])]
            result_test_copy = result_test_copy[(result_test_copy[column] == row[column])]
        for item in listSensorColNames:
            result = st.kstest(result_train_copy[item], result_test_copy[item])
            # print(result)
            arr_stats[i] = arr_stats[i] + result.statistic
            i = i + 1

    # taking average
    arr_stats[:] = [x / 2 for x in arr_stats]

    # Identifying and Counting which sensors passed the test and which did not
    idx = 0
    com_count = 0
    av_count = 0
    for value in arr_stats:
        # if df_compare[listSensorColNames[idx]].values[0] == 1:
        #     av_count = av_count + 1
        if value < 0.2:
            print(listSensorColNames[idx] + ' sensor passed the K-S test')
            av_count = av_count + 1
            if df_compare[listSensorColNames[idx]].values[0] == 1:
                com_count = com_count + 1
        idx = idx + 1

    print('Out of ', len(listSensorColNames), ' sensors taken under consideration')
    print(com_count, 'passed the K-S without considering the system states whereas')
    print(av_count, 'passed the K-S with considering the common system states')

    # Task 3 : Graph plotting

    xvalue = [fTrain, fTest]
    yvalue = [c_trainc, c_testc]
    plt.bar(xvalue, yvalue, color='maroon', width=0.4)
    plt.xlabel('Datasets')
    plt.ylabel('Percentage')
    plt.title('Common system states')
    plt.show()

    x2value = ['Without', 'With']
    y2value = [com_count, av_count]
    plt.bar(x2value, y2value, color='olive', width=0.4)
    plt.xlabel('Condition')
    plt.ylabel('No. of sensors passing the tests')
    plt.title('Sensors passing K-S test')
    plt.show()
