import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
from random import random
from scipy.stats import pearsonr

def convertCSV(filestem):
	h5Filename = filestem + '.h5'
	if not os.path.isfile(h5Filename):
		# Read the CSV
		store = pd.HDFStore(h5Filename)
		csvFilename = filestem + '.csv.gz'
		columnNames = ['date','price','volume']
		importFilter = ['date','price']
		columnTypes = {'date': np.uint32, 'price': np.float32, 'volume': np.float32}
		print('Reading csv file...')
		df = pd.DataFrame(pd.read_csv(csvFilename, names = columnNames, dtype = columnTypes, usecols=importFilter))
		print('Saving as HDF...')
		store['df'] = df
	else:
		print('Reading h5 file...')
		df = pd.read_hdf(h5Filename, 'df')
	return df

def extractSegment(dataFrame, startIndex, segmentLength, fade = True):
	startTime = dataFrame['date'].iloc[startIndex]
	endTime = startTime + segmentLength
	endIndex = np.searchsorted(df['date'], endTime)
	data = dataFrame[startIndex:endIndex].copy(deep=True)
	if fade:
		# Demote oldest data points by 50%
		falloff = np.log(np.linspace(0.5,1,num=len(data)))
		data['price'] = data['price'].values + falloff
	return data

def correlateTimeseries(A, B):

	# Convert the time series to relative time
	aDate = A['date'] - A['date'].iat[0]
	bDate = B['date'] - B['date'].iat[0]

	# Prepare indices for matched data points
	datesMatched = np.searchsorted(aDate, bDate)
	l = len(aDate) - 1
	datesMatched[datesMatched > l] = l
	c = dict()
	keyword = 'price'
	# Select data according to matched indices
	a = np.array(A[keyword].values)
	aReduced = a[datesMatched]
	bReduced = np.array(B[keyword].values)
	# Correct to the baseline
	aReduced = aReduced - np.mean(aReduced)
	bReduced = bReduced - np.mean(bReduced)
	# Perform the z-transformation
	zA = aReduced / np.sqrt(np.sum(np.square(aReduced)) / l)
	zB = bReduced / np.sqrt(np.sum(np.square(bReduced)) / l)
	# Calculate the correlation
	r = pearsonr(zA,zB)
	return r[1]

# Load the wavelet database
filestem = 'bitfinexUSD'
df = convertCSV(filestem)

# See how much memory we're using
# memUsage = sum(block.values.nbytes for block in df.blocks.values())

# Convert to human-readable date format
# df['date'] = pd.to_datetime(df['date'], unit='s') # Increases 'date' memory usage by 50%

# Convert to logarithmic values
df['price'] = np.log(df['price'])

# Generate a data template
comparisonLength = 43200 # in seconds, 12 hours
templateStartIndex = int(len(df) * random()*0.8)
template = extractSegment(df, templateStartIndex, comparisonLength)

# Sweep the existing data series for a match with the template
maxLen = len(df) - comparisonLength - 1
maxLen = 800 # in data points
similarity = np.zeros([maxLen,])
xScale = xrange(maxLen)
previousPercentage = 0
print('Starting correlation...')
for t in xScale:
	segment = extractSegment(df, 15000+t, comparisonLength)
	percentage = int(np.floor(100 * float(t) / maxLen))
	if percentage > previousPercentage:
		print('{}%'.format(percentage))
		previousPercentage = percentage
	similarity[t] = correlateTimeseries(segment, template)

strongestIndex = np.argmax(similarity)
segment = extractSegment(df, 15000+strongestIndex, comparisonLength, fade = False)
template = extractSegment(df, templateStartIndex, comparisonLength, fade = False)
plt.subplot(211)
plt.plot(segment['date'], segment['price'])
plt.subplot(212)
plt.plot(template['date'], template['price'])
plt.show()