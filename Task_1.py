import pandas as pd
import matplotlib.pyplot as plt
import re
import glob
import os

from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger


def plot_measurements(dataSetName, dataSet, fName):
    """
    Input:
    dataSetName - HAI version eg. hai_22_04
    dataSet - dataframe of the CSV read
    fName - file name within the HAI version

    Takes the dataframe and plots all the features, highlighting the period when attack is happening (attack column = 1)
    if any. The plot in store in the pdf file with the name resembling 'dataSetName + FName'
    """
    # Determine when the attack value changes
    if dataSetName == 'hai_22_04_':
        period = dataSet[dataSet.Attack.diff() != 0].index.values
    else:
        period = dataSet[dataSet.attack.diff() != 0].index.values

    with PdfPages(dataSetName + fName + '.pdf') as pdf:
        firstPage = plt.figure(figsize=(11.69, 8.27))
        txt = 'Plot for ' + fName
        firstPage.text(0.5, 0.5, txt, transform=firstPage.transFigure, size=24, ha="center")
        pdf.savefig()
        plt.close()
        for colName in dataSet.columns.values:
            if not re.match(r"attack", colName, re.IGNORECASE):
                figure = plt.figure(colName)
                dataSet[colName].plot()
                plt.title(colName, size='small', color='Green')
                plt.legend(loc="best")
                plt.ylabel("Values", fontsize=7)
                plt.xlabel("Time", fontsize=7)
                # Plot red line
                for item in period[1::]:
                    plt.axvline(item, ymin=0, ymax=1, color='red', lw=0.05)
                pdf.savefig(figure)
                plt.close(figure)


def read_plot(dataSetName, path):
    """
    Input:
    dataSetName - HAI version eg. hai_22_04
    path - path of the dataset version to be read

    Converts each CSV file in the folder into panda dataframe and calls the plot_measurements function
    """
    filenames = os.listdir(path)
    print(filenames)

    for file in filenames:
        dataSet = pd.read_csv(path + file, index_col=0, parse_dates=True)
        # print(dataSet.head())
        fName = os.path.splitext(file)[0]
        plot_measurements(dataSetName, dataSet, fName)


def collate_pdfs(dataSetName):
    """
    Collates all the pdf files starting with specific name into one
    """
    merger = PdfMerger()
    for pdf in glob.glob(dataSetName + '*.pdf'):
        merger.append(pdf)
    merger.write(dataSetName + '.pdf')
    merger.close()


if __name__ == '__main__':
    print('Task 1: Visualization of Physical Readings')

    # For dataset version hai-22.04 - Released April 29, 2022
    print('Working on dataset version hai-22.04')
    dataSetName = 'hai_22_04_'
    path = '.\\HAI_dataset\\hai_22_04\\'
    read_plot(dataSetName, path)    # Data is read and plotted
    collate_pdfs(dataSetName)       # All plots are collated
    print('Plot for ' + dataSetName + ' is completed and stored in file ' + dataSetName + '.pdf')

    # For dataset version hai-21.03 - Released Feb 17, 2021
    print('Working on dataset version hai-21.03')
    dataSetName = 'hai_21_03_'
    path = '.\\HAI_dataset\\hai_21_03\\'
    read_plot(dataSetName, path)    # Data is read and plotted
    collate_pdfs(dataSetName)       # All plots are collated
    print('Plot for ' + dataSetName + ' is completed and stored in file ' + dataSetName + '.pdf')

    # For dataset version hai-20.07 - Released July 22, 2020
    print('Working on dataset version hai-20.07')
    dataSetName = 'hai_20_07_'
    path = '.\\HAI_dataset\\hai_20_07\\'
    read_plot(dataSetName, path)    # Data is read and plotted
    collate_pdfs(dataSetName)       # All plots are collated
    print('Plot for ' + dataSetName + ' is completed and stored in file ' + dataSetName + '.pdf')
